"""robocore.telemetry.flight_recorder -- black-box flight recording, real-to-sim
replay / auto-SysID, and terminal observability.

Deliverable 3 of the robocore Phase-1 runtime.

Architecture
------------
``RingBuffer1kHz``
    Preallocated numpy ring buffers holding the last N seconds of 1 kHz
    proprioceptive state (q, dq, tau, wrist F/T), policy latent vector and
    camera frame timestamps.  The RT-side writer only does an index increment
    plus fixed-slot writes -- no allocation, no I/O on the hot path.

``FlightRecorder.trigger(...)``
    Called by the physical executor's *episode supervisor thread* (never the
    1 kHz loop itself) when a FORCE_SPIKE / COLLISION / INFERENCE_DEADLINE /
    TASK_FAILURE is detected.  The active ring buffer is swapped out under a
    short lock (O(1) -- the writer instantly moves to a fresh buffer), then
    the sealed window is serialized to immutable, content-addressed
    artifacts: a Parquet (or npz fallback) telemetry dump plus a serialized
    sim-state blob (MuJoCo qpos/qvel/act/ctrl/time + model hash).  The
    snapshot is registered in the ledger with its sha256 so any later reader
    can prove artifact integrity.

``FlightRecorder.reconstruct_failure_in_sim(snapshot_hash)``
    Loads the dump, verifies content hashes, restores the exact MuJoCo
    state, replays the recorded torque trace open-loop, localizes where the
    sim trajectory diverges from reality, and runs a deterministic bounded
    coordinate-descent SysID search over friction/damping scales to minimize
    replay divergence.  The result is a SimToRealGapReport persisted to the
    ledger (which the planner-context builder surfaces as drift warnings).

``EventEmitter`` + the three analytical SQLite views give full-lab
observability from a terminal with zero SaaS dependency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from robocore.contracts import (
    BlackBoxSnapshotRecord,
    DifficultyStrataBin,
    EventType,
    FlightTrigger,
    RolloutDomain,
    RolloutTelemetryBundle,
    SimToRealGapReport,
    SysIDStatus,
    canonical_json,
    unix_now,
    unix_to_iso,
)
from robocore.storage.ledger_repo import ExperimentLedgerRepository

__all__ = [
    "FlightRecorderConfig",
    "RingBuffer1kHz",
    "SealedWindow",
    "SimStateBlob",
    "ISimStateProvider",
    "MujocoSimStateProvider",
    "StaticSimStateProvider",
    "IReplayBackend",
    "MujocoReplayBackend",
    "SysIDParamSpec",
    "SysIDOptimizer",
    "SysIDResult",
    "FailureReconstructionReport",
    "EventEmitter",
    "FlightRecorder",
    "ANALYTICAL_VIEWS_SQL",
]

_LOG = logging.getLogger("robocore.flight_recorder")

# ---------------------------------------------------------------------------
# Bit flags for the per-sample flags channel
# ---------------------------------------------------------------------------

FLAG_INFERENCE_DEADLINE = 1 << 0
FLAG_COLLISION = 1 << 1
FLAG_FORCE_SPIKE = 1 << 2
FLAG_ESTOP = 1 << 3


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FlightRecorderConfig:
    window_seconds: float = 5.0
    sample_rate_hz: float = 1000.0
    n_joints: int = 7          # Franka FR3 default (UR5e: 6)
    latent_dim: int = 32
    n_cameras: int = 2
    validate_shapes: bool = False  # RT hot path runs with False; tests True
    max_seal_queue: int = 4
    parquet_compression: str = "snappy"


class RingBuffer1kHz:
    """Single-writer, fixed-capacity ring buffer for 1 kHz telemetry.

    Locking discipline: exactly one thread calls :meth:`append` (the bridge
    subscriber).  :meth:`swap_out` is called by the trigger path and takes
    the buffer out of service in O(1); the sealed buffer is then copied and
    serialized without ever blocking the writer again.
    """

    __slots__ = (
        "_cap",
        "_nq",
        "_latent",
        "_ncam",
        "_w",  # monotonically increasing write counter
        "_t_ns",
        "_q",
        "_dq",
        "_tau",
        "_f_ext",
        "_latent_arr",
        "_cam_ts",
        "_flags",
    )

    def __init__(self, config: FlightRecorderConfig) -> None:
        self._cap = int(config.window_seconds * config.sample_rate_hz)
        self._nq = config.n_joints
        self._latent = config.latent_dim
        self._ncam = config.n_cameras
        self._w = 0
        self._t_ns = np.zeros(self._cap, dtype=np.int64)
        self._q = np.zeros((self._cap, self._nq), dtype=np.float64)
        self._dq = np.zeros((self._cap, self._nq), dtype=np.float64)
        self._tau = np.zeros((self._cap, self._nq), dtype=np.float64)
        self._f_ext = np.zeros((self._cap, 6), dtype=np.float64)
        self._latent_arr = np.zeros((self._cap, self._latent), dtype=np.float32)
        self._cam_ts = np.zeros((self._cap, self._ncam), dtype=np.float64)
        self._flags = np.zeros(self._cap, dtype=np.uint8)

    # -- hot path ----------------------------------------------------------

    def append(
        self,
        t_ns: int,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        f_ext: np.ndarray,
        policy_latent: np.ndarray,
        cam_timestamps: np.ndarray,
        flags: int = 0,
    ) -> None:
        idx = self._w % self._cap
        self._t_ns[idx] = t_ns
        self._q[idx] = q
        self._dq[idx] = dq
        self._tau[idx] = tau
        self._f_ext[idx] = f_ext
        self._latent_arr[idx] = policy_latent
        self._cam_ts[idx] = cam_timestamps
        self._flags[idx] = flags
        self._w += 1

    # -- lifecycle -----------------------------------------------------------

    @property
    def capacity(self) -> int:
        return self._cap

    @property
    def total_appends(self) -> int:
        return self._w

    def seal(self) -> "SealedWindow":
        """Copy the buffered window into a contiguous, time-ordered window.

        Call only after the buffer has been taken out of the writer path
        (``FlightRecorder`` swaps buffers first); a defensive copy is still
        made so the sealed window is immutable evidence.
        """
        n = min(self._w, self._cap)
        if n == 0:
            raise ValueError("cannot seal an empty ring buffer")
        if self._w <= self._cap:
            sl = slice(0, n)
            window = SealedWindow(
                t_ns=self._t_ns[sl].copy(),
                q=self._q[sl].copy(),
                dq=self._dq[sl].copy(),
                tau=self._tau[sl].copy(),
                f_ext=self._f_ext[sl].copy(),
                latent=self._latent_arr[sl].copy(),
                cam_ts=self._cam_ts[sl].copy(),
                flags=self._flags[sl].copy(),
                n=n,
            )
            return window
        # wraparound: oldest data lives at (w % cap) .. cap-1, then 0 .. (w%cap)-1
        start = self._w % self._cap
        head = slice(start, self._cap)
        tail = slice(0, start)
        window = SealedWindow(
            t_ns=np.concatenate([self._t_ns[head], self._t_ns[tail]]),
            q=np.concatenate([self._q[head], self._q[tail]], axis=0),
            dq=np.concatenate([self._dq[head], self._dq[tail]], axis=0),
            tau=np.concatenate([self._tau[head], self._tau[tail]], axis=0),
            f_ext=np.concatenate([self._f_ext[head], self._f_ext[tail]], axis=0),
            latent=np.concatenate(
                [self._latent_arr[head], self._latent_arr[tail]], axis=0
            ),
            cam_ts=np.concatenate([self._cam_ts[head], self._cam_ts[tail]], axis=0),
            flags=np.concatenate([self._flags[head], self._flags[tail]]),
            n=n,
        )
        return window


@dataclass
class SealedWindow:
    """Immutable, contiguous copy of one flight-recorder window."""

    t_ns: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    f_ext: np.ndarray
    latent: np.ndarray
    cam_ts: np.ndarray
    flags: np.ndarray
    n: int

    @property
    def n_joints(self) -> int:
        return int(self.q.shape[1])

    @property
    def duration_s(self) -> float:
        return float((self.t_ns[-1] - self.t_ns[0]) / 1e9)

    def column_arrays(self) -> Dict[str, np.ndarray]:
        cols: Dict[str, np.ndarray] = {"t_ns": self.t_ns, "flags": self.flags}
        for j in range(self.n_joints):
            cols[f"q_{j}"] = self.q[:, j]
            cols[f"dq_{j}"] = self.dq[:, j]
            cols[f"tau_{j}"] = self.tau[:, j]
        for k in range(6):
            cols[f"f_ext_{k}"] = self.f_ext[:, k]
        for k in range(self.latent.shape[1]):
            cols[f"latent_{k}"] = self.latent[:, k]
        for c in range(self.cam_ts.shape[1]):
            cols[f"cam{c}_ts"] = self.cam_ts[:, c]
        return cols

    def write(self, path: Path, *, compression: str = "snappy") -> str:
        """Write the window as Parquet (preferred) or npz fallback.

        Returns the format actually written (``"parquet"`` | ``"npz"``).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = self.column_arrays()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({k: pa.array(np.ascontiguousarray(v)) for k, v in cols.items()})
            pq.write_table(table, str(path), compression=compression)
            return "parquet"
        except ImportError:
            npz_path = path.with_suffix(".npz")
            np.savez_compressed(str(npz_path), **cols)
            return "npz"

    @classmethod
    def read(cls, path: Path) -> "SealedWindow":
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(str(path))
            cols = {name: table.column(name).to_numpy() for name in table.column_names}
        else:
            data = np.load(str(path))
            cols = {str(k): data[k] for k in data.files}
        n = int(len(cols["t_ns"]))
        nq = max(
            int(name.rsplit("_", 1)[1]) for name in cols if name.startswith("q_")
        ) + 1
        n_lat = sum(1 for name in cols if name.startswith("latent_"))
        n_cam = sum(1 for name in cols if name.startswith("cam") and name.endswith("_ts"))
        q = np.zeros((n, nq))
        dq = np.zeros((n, nq))
        tau = np.zeros((n, nq))
        for j in range(nq):
            q[:, j] = cols[f"q_{j}"]
            dq[:, j] = cols[f"dq_{j}"]
            tau[:, j] = cols[f"tau_{j}"]
        f_ext = np.zeros((n, 6))
        for k in range(6):
            f_ext[:, k] = cols[f"f_ext_{k}"]
        latent = (
            np.stack([cols[f"latent_{k}"] for k in range(n_lat)], axis=1)
            if n_lat
            else np.zeros((n, 0))
        )
        cam_ts = (
            np.stack([cols[f"cam{c}_ts"] for c in range(n_cam)], axis=1)
            if n_cam
            else np.zeros((n, 0))
        )
        return cls(
            t_ns=cols["t_ns"].astype(np.int64),
            q=q,
            dq=dq,
            tau=tau,
            f_ext=f_ext,
            latent=latent.astype(np.float32),
            cam_ts=cam_ts,
            flags=cols["flags"].astype(np.uint8),
            n=n,
        )


# ---------------------------------------------------------------------------
# Sim-state blobs
# ---------------------------------------------------------------------------


@dataclass
class SimStateBlob:
    """Serialized simulator state + model identity, self-contained in .npz."""

    model_hash: str
    arrays: Dict[str, np.ndarray]
    meta: Dict[str, Any]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.arrays)
        payload["_meta_json"] = np.frombuffer(
            canonical_json({"model_hash": self.model_hash, **self.meta}).encode(
                "utf-8"
            ),
            dtype=np.uint8,
        )
        np.savez_compressed(str(path), **payload)

    @classmethod
    def read(cls, path: Path) -> "SimStateBlob":
        data = np.load(str(path))
        meta_bytes = data["_meta_json"].tobytes()
        meta = json.loads(meta_bytes.decode("utf-8"))
        model_hash = meta.pop("model_hash")
        arrays = {k: data[k] for k in data.files if k != "_meta_json"}
        return cls(model_hash=model_hash, arrays=arrays, meta=meta)


class ISimStateProvider(Protocol):
    """Live capture of the simulator state aligned with the ring buffer."""

    def capture(self) -> SimStateBlob: ...


class MujocoSimStateProvider:
    """Captures full MuJoCo physics state + model XML hash.

    Requires the ``mujoco`` extra (``pip install robocore[mujoco]``).  The
    model hash is sha256 over the XML text so a dump can only be replayed
    against the exact model that produced it.
    """

    def __init__(self, model_xml_path: str | Path, data: Any, model: Any) -> None:
        import mujoco  # lazy: not needed for pure-recording deployments

        self._mujoco = mujoco
        self._xml_path = Path(model_xml_path)
        self._data = data
        self._model = model
        xml_bytes = self._xml_path.read_bytes()
        self._model_hash = (
            "mdl_" + hashlib.sha256(xml_bytes).hexdigest()
        )

    def capture(self) -> SimStateBlob:
        return SimStateBlob(
            model_hash=self._model_hash,
            arrays={
                "qpos": np.array(self._data.qpos, copy=True),
                "qvel": np.array(self._data.qvel, copy=True),
                "act": (
                    np.array(self._data.act, copy=True)
                    if self._data.act is not None and self._data.act.size
                    else np.zeros(0)
                ),
                "ctrl": np.array(self._data.ctrl, copy=True),
            },
            meta={
                "engine": "mujoco",
                "model_xml_path": str(self._xml_path),
                "nq": int(self._model.nq),
                "nv": int(self._model.nv),
                "nu": int(self._model.nu),
                "sim_time": float(self._data.time),
            },
        )


class StaticSimStateProvider:
    """Returns a fixed blob (synthetic / hardware-free deployments)."""

    def __init__(self, blob: SimStateBlob) -> None:
        self._blob = blob

    def capture(self) -> SimStateBlob:
        return SimStateBlob(
            model_hash=self._blob.model_hash,
            arrays={k: v.copy() for k, v in self._blob.arrays.items()},
            meta=dict(self._blob.meta),
        )


# ---------------------------------------------------------------------------
# Replay backends
# ---------------------------------------------------------------------------


class IReplayBackend(Protocol):
    """Deterministic real-to-sim replay engine used by auto-SysID."""

    def load_state(self, blob: SimStateBlob) -> None: ...

    def apply_param_offsets(
        self, friction: Dict[str, float], damping: Dict[str, float]
    ) -> None: ...

    def step_batch(self, ctrl: np.ndarray) -> np.ndarray:
        """Replay the recorded control trace; returns (T, 2*nq) [q, dq]."""
        ...


class MujocoReplayBackend:
    """Replays a recorded torque/ctrl trace against a MuJoCo model.

    Parameter offsets are *multiplicative scales* on model defaults, so the
    calibrated values stay interpretable relative to the shipped model:
      friction: geom_slide_friction_scale, geom_torsion_friction_scale,
                geom_rolling_friction_scale
      damping:  dof_damping_scale, dof_armature_scale,
                actuator_gain_scale
    """

    _FRICTION_KEYS = (
        "geom_slide_friction_scale",
        "geom_torsion_friction_scale",
        "geom_rolling_friction_scale",
    )
    _DAMPING_KEYS = (
        "dof_damping_scale",
        "dof_armature_scale",
        "actuator_gain_scale",
    )

    def __init__(self, blob: SimStateBlob) -> None:
        import mujoco

        self._mujoco = mujoco
        xml_path = Path(str(blob.meta["model_xml_path"]))
        xml_bytes = xml_path.read_bytes()
        actual = "mdl_" + hashlib.sha256(xml_bytes).hexdigest()
        if actual != blob.model_hash:
            raise ValueError(
                f"model mismatch for replay: blob expects {blob.model_hash},"
                f" file hashes to {actual}"
            )
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        self._defaults = {
            # geom_friction is (ngeom, 3): [sliding, torsional, rolling]
            "geom_friction": np.array(self._model.geom_friction, copy=True),
            "dof_damping": np.array(self._model.dof_damping, copy=True),
            "dof_armature": np.array(self._model.dof_armature, copy=True),
            "act_gain": np.array(self._model.actuator_gainprm[:, 0], copy=True),
        }
        self.load_state(blob)

    def load_state(self, blob: SimStateBlob) -> None:
        mujoco = self._mujoco
        self._data.qpos[:] = blob.arrays["qpos"]
        self._data.qvel[:] = blob.arrays["qvel"]
        if "act" in blob.arrays and blob.arrays["act"].size:
            self._data.act[:] = blob.arrays["act"]
        self._data.ctrl[:] = blob.arrays["ctrl"]
        self._data.time = float(blob.meta.get("sim_time", 0.0))
        mujoco.mj_forward(self._model, self._data)

    def apply_param_offsets(
        self, friction: Dict[str, float], damping: Dict[str, float]
    ) -> None:
        m = self._model
        # geom_friction columns: 0=sliding, 1=torsional, 2=rolling
        m.geom_friction[:, 0] = self._defaults["geom_friction"][:, 0] * friction.get(
            "geom_slide_friction_scale", 1.0
        )
        m.geom_friction[:, 1] = self._defaults["geom_friction"][:, 1] * friction.get(
            "geom_torsion_friction_scale", 1.0
        )
        m.geom_friction[:, 2] = self._defaults["geom_friction"][:, 2] * friction.get(
            "geom_rolling_friction_scale", 1.0
        )
        m.dof_damping[:] = self._defaults["dof_damping"] * damping.get(
            "dof_damping_scale", 1.0
        )
        m.dof_armature[:] = self._defaults["dof_armature"] * damping.get(
            "dof_armature_scale", 1.0
        )
        m.actuator_gainprm[:, 0] = self._defaults["act_gain"] * damping.get(
            "actuator_gain_scale", 1.0
        )

    def step_batch(self, ctrl: np.ndarray) -> np.ndarray:
        mujoco = self._mujoco
        T = ctrl.shape[0]
        nq = int(self._model.nq)
        out = np.zeros((T, 2 * nq))
        for t in range(T):
            nu = min(ctrl.shape[1], int(self._model.nu))
            self._data.ctrl[:nu] = ctrl[t, :nu]
            mujoco.mj_step(self._model, self._data)
            out[t, :nq] = self._data.qpos
            out[t, nq:] = self._data.qvel
        return out


# ---------------------------------------------------------------------------
# Deterministic bounded SysID optimizer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SysIDParamSpec:
    name: str
    group: str  # "friction" | "damping"
    lo: float
    hi: float
    init: float = 1.0


DEFAULT_SYSID_SPECS: Tuple[SysIDParamSpec, ...] = (
    SysIDParamSpec("geom_slide_friction_scale", "friction", 0.25, 4.0),
    SysIDParamSpec("geom_torsion_friction_scale", "friction", 0.25, 4.0),
    SysIDParamSpec("dof_damping_scale", "damping", 0.25, 4.0),
    SysIDParamSpec("dof_armature_scale", "damping", 0.25, 4.0),
)


@dataclass
class SysIDResult:
    friction: Dict[str, float] = field(default_factory=dict)
    damping: Dict[str, float] = field(default_factory=dict)
    cost_before: float = float("inf")
    cost_after: float = float("inf")
    n_evals: int = 0
    converged: bool = False
    eval_trace: List[Tuple[Dict[str, float], float]] = field(default_factory=list)


class SysIDOptimizer:
    """Bounded coordinate descent over multiplicative physics scales.

    Deterministic (no RNG): each parameter is probed at +/- step for a
    geometrically shrinking step schedule until the evaluation budget is
    exhausted or no parameter improves.  CMA-ES in the loop is a Phase 2
    hardening item; Phase 1 deliberately ships this simple, auditable
    optimizer because its cost traces are replayable offline.
    """

    def __init__(
        self,
        specs: Sequence[SysIDParamSpec] = DEFAULT_SYSID_SPECS,
        *,
        budget: int = 160,
        initial_step: float = 0.25,
        shrink: float = 0.5,
        min_step: float = 0.01,
    ) -> None:
        self._specs = tuple(specs)
        self._budget = int(budget)
        self._initial_step = float(initial_step)
        self._shrink = float(shrink)
        self._min_step = float(min_step)

    def minimize(
        self,
        cost: Callable[[Dict[str, float], Dict[str, float]], float],
    ) -> SysIDResult:
        params: Dict[str, float] = {s.name: s.init for s in self._specs}

        def split(p: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
            friction = {
                s.name: p[s.name] for s in self._specs if s.group == "friction"
            }
            damping = {s.name: p[s.name] for s in self._specs if s.group == "damping"}
            return friction, damping

        evals = 0
        fr, da = split(params)
        best = cost(fr, da)
        evals += 1
        result = SysIDResult(cost_before=best)
        step = self._initial_step
        while evals < self._budget and step >= self._min_step:
            improved = False
            for spec in self._specs:
                if evals >= self._budget:
                    break
                for direction in (+1.0, -1.0):
                    if evals >= self._budget:
                        break
                    cand = dict(params)
                    cand[spec.name] = float(
                        np.clip(
                            params[spec.name] + direction * step,
                            spec.lo,
                            spec.hi,
                        )
                    )
                    fr, da = split(cand)
                    c = cost(fr, da)
                    evals += 1
                    result.eval_trace.append((dict(cand), c))
                    if c < best - 1e-12:
                        params, best = cand, c
                        improved = True
            if not improved:
                step *= self._shrink
        fr, da = split(params)
        result.friction = fr
        result.damping = da
        result.cost_after = best
        result.n_evals = evals
        result.converged = best < result.cost_before
        return result


# ---------------------------------------------------------------------------
# Structured JSON event emitter
# ---------------------------------------------------------------------------


class EventEmitter:
    """Zero-SaaS observability: JSON Lines to stderr and a rotating file.

    Every line is self-describing (schema version, UTC timestamps, severity)
    so ``jq`` over the sink is a fully-fledged query console.
    """

    _SCHEMA = "robocore.event.v1"
    _MAX_BYTES = 10 * 1024 * 1024
    _KEEP = 3

    def __init__(
        self,
        sink_path: Optional[Path] = None,
        *,
        also_stderr: bool = True,
        min_level: str = "DEBUG",
    ) -> None:
        self._sink = Path(sink_path) if sink_path is not None else None
        self._also_stderr = also_stderr
        self._min = min_level
        self._lock = threading.Lock()
        if self._sink is not None:
            self._sink.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        event_type: str,
        payload: Dict[str, Any],
        *,
        severity: str = "INFO",
        cycle_id: Optional[str] = None,
        rollout_id: Optional[str] = None,
    ) -> None:
        levels = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
        if levels.get(severity, 1) < levels.get(self._min, 0):
            return
        record = {
            "schema": self._SCHEMA,
            "ts_unix": round(unix_now(), 6),
            "ts_iso": unix_to_iso(unix_now()),
            "severity": severity,
            "event_type": event_type,
            "cycle_id": cycle_id,
            "rollout_id": rollout_id,
            "payload": payload,
        }
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
        with self._lock:
            if self._also_stderr:
                print(line, flush=True)
            if self._sink is not None:
                self._rotate_if_needed()
                with open(self._sink, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def _rotate_if_needed(self) -> None:
        assert self._sink is not None
        if not self._sink.exists():
            return
        if self._sink.stat().st_size < self._MAX_BYTES:
            return
        for i in range(self._KEEP - 1, 0, -1):
            src = self._sink.with_suffix(f".{i}")
            dst = self._sink.with_suffix(f".{i + 1}")
            if src.exists():
                os.replace(src, dst)
        os.replace(self._sink, self._sink.with_suffix(".1"))


# ---------------------------------------------------------------------------
# Analytical views (installed into the same SQLite database)
# ---------------------------------------------------------------------------

ANALYTICAL_VIEWS_SQL = """
CREATE VIEW IF NOT EXISTS v_hourly_throughput AS
SELECT strftime('%Y-%m-%d %H:00:00',
                CAST(CAST(t_start_unix AS INTEGER) AS TEXT), 'unixepoch') AS hour,
       task_name,
       domain,
       COUNT(*) AS n_rollouts,
       SUM(success) AS n_success,
       ROUND(100.0 * AVG(success), 2) AS success_pct,
       ROUND(AVG(time_to_completion), 3) AS mean_ttc_s,
       ROUND(AVG(cumulative_jerk), 1) AS mean_cum_jerk,
       ROUND(AVG(torque_penalty), 2) AS mean_torque_penalty,
       COUNT(DISTINCT batch_id) AS n_batches
  FROM rollout_telemetry
 GROUP BY hour, task_name, domain
 ORDER BY hour DESC, task_name, domain;

CREATE VIEW IF NOT EXISTS v_failure_mode_pareto AS
SELECT rt.task_name,
       COALESCE(rt.vlm_failure_summary, '<unclassified>') AS failure_mode,
       COUNT(*) AS n_failures,
       ROUND(100.0 * COUNT(*) * 1.0
             / MAX(1, SUM(COUNT(*)) OVER (PARTITION BY rt.task_name)), 2)
             AS pct_of_task_failures,
       ROUND(AVG(rt.cumulative_jerk), 1) AS mean_cum_jerk_on_failure,
       ROUND(MAX(rt.force_spike_max), 2) AS max_force_spike,
       SUM(CASE WHEN rt.collision_count > 0 THEN 1 ELSE 0 END) AS n_with_collision
  FROM rollout_telemetry rt
 WHERE rt.success = 0
 GROUP BY rt.task_name, failure_mode
 ORDER BY rt.task_name, n_failures DESC;

CREATE VIEW IF NOT EXISTS v_sim_vs_real_drift AS
SELECT s.task_name,
       s.harness_hash,
       strftime('%Y-%m-%dT%H:%M:%SZ',
                CAST(CAST(s.calibrated_at_unix AS INTEGER) AS TEXT),
                'unixepoch') AS calibrated_at_iso,
       ROUND(s.sim_metric, 4) AS sim_success_24h,
       ROUND(s.real_metric, 4) AS real_success_24h,
       ROUND(s.gap, 4) AS success_gap,
       s.status AS sysid_status,
       s.friction_offset_json,
       s.damping_offset_json,
       ROUND(s.residual_before, 6) AS replay_residual_before,
       ROUND(s.residual_after, 6) AS replay_residual_after,
       (SELECT COUNT(*) FROM blackbox_dumps b
         WHERE b.created_at_unix >= s.calibrated_at_unix - 86400.0
           AND b.rollout_id IN (SELECT rollout_id FROM rollout_telemetry r2
                                 WHERE r2.task_name = s.task_name))
         AS blackbox_dumps_24h
  FROM sim_to_real_calibration s
 ORDER BY s.task_name, s.calibrated_at_unix DESC;
"""


# ---------------------------------------------------------------------------
# Failure reconstruction report
# ---------------------------------------------------------------------------


@dataclass
class FailureReconstructionReport:
    snapshot_hash: str
    rollout_id: str
    trigger: FlightTrigger
    n_samples: int
    window_seconds: float
    residual_before: float
    residual_after: float
    divergence_onset_s: Optional[float]
    friction_offsets: Dict[str, float]
    damping_offsets: Dict[str, float]
    sysid_status: SysIDStatus
    n_cost_evals: int
    report_path: str

    def to_json(self) -> str:
        return canonical_json(
            {
                "snapshot_hash": self.snapshot_hash,
                "rollout_id": self.rollout_id,
                "trigger": self.trigger.value,
                "n_samples": self.n_samples,
                "window_seconds": self.window_seconds,
                "residual_before": self.residual_before,
                "residual_after": self.residual_after,
                "divergence_onset_s": self.divergence_onset_s,
                "friction_offsets": self.friction_offsets,
                "damping_offsets": self.damping_offsets,
                "sysid_status": self.sysid_status.value,
                "n_cost_evals": self.n_cost_evals,
            }
        )


# ---------------------------------------------------------------------------
# FlightRecorder
# ---------------------------------------------------------------------------


class FlightRecorder:
    """Continuously buffers 1 kHz telemetry; seals immutable black boxes on
    trigger; reconstructs failures in sim and calibrates physics offsets.

    Wiring contract with the physical executor:

        executor.begin_rollout(rollout_id)        -> recorder.start_rollout(...)
        bridge 1 kHz subscriber                    -> recorder.append_sample(...)
        executor episode supervisor detects fault -> recorder.trigger(...)
        executor batch commit                      -> telemetry rows reference
                                                      snapshot_hash
    """

    def __init__(
        self,
        config: FlightRecorderConfig,
        ledger: ExperimentLedgerRepository,
        artifact_root: str | Path,
        sim_state_provider: ISimStateProvider,
        *,
        backend_factory: Optional[Callable[[SimStateBlob], IReplayBackend]] = None,
        emitter: Optional[EventEmitter] = None,
        sysid_specs: Sequence[SysIDParamSpec] = DEFAULT_SYSID_SPECS,
        sysid_budget: int = 120,
        sysid_min_improvement: float = 0.10,
        sysid_accept_residual: float = 0.02,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._artifact_root = Path(artifact_root)
        self._provider = sim_state_provider
        self._backend_factory = backend_factory
        self._emitter = emitter or EventEmitter(also_stderr=False)
        self._specs = tuple(sysid_specs)
        self._sysid_budget = int(sysid_budget)
        self._sysid_min_improvement = float(sysid_min_improvement)
        self._sysid_accept_residual = float(sysid_accept_residual)
        self._lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._active: Optional[RingBuffer1kHz] = None
        # Entries queued when a concurrent flush is already serializing.
        # Bounded: overflow drops the OLDEST sealed buffer (evidence priority
        # is the most recent failure) and increments a telemetry counter.
        self._sealed_pending: List[Tuple[str, FlightTrigger, RingBuffer1kHz, dict]] = []
        self._dropped_seals = 0
        self._current_rollout: Optional[str] = None
        self._current_harness: Optional[str] = None
        self._current_task: Optional[str] = None
        self._t0_wall_unix = 0.0
        self.install_analytical_views()

    # -- lifecycle -----------------------------------------------------------

    def install_analytical_views(self) -> None:
        self._ledger.apply_analytical_views(ANALYTICAL_VIEWS_SQL)

    def start_rollout(
        self, rollout_id: str, harness_hash: str, task_name: str
    ) -> None:
        self._drain_pending_seals()
        with self._lock:
            self._active = RingBuffer1kHz(self._config)
            self._current_rollout = rollout_id
            self._current_harness = harness_hash
            self._current_task = task_name
            self._t0_wall_unix = unix_now()

    def finish_rollout(self, rollout_id: str) -> None:
        self._drain_pending_seals()
        with self._lock:
            if self._current_rollout == rollout_id:
                self._active = None
                self._current_rollout = None

    # -- hot path ------------------------------------------------------------

    def append_sample(
        self,
        t_ns: int,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        f_ext: np.ndarray,
        policy_latent: np.ndarray,
        cam_timestamps: np.ndarray,
        flags: int = 0,
    ) -> None:
        buf = self._active
        if buf is None:
            return  # no rollout armed: drop (executor drives arming)
        if self._config.validate_shapes:
            nq = self._config.n_joints
            assert q.shape == (nq,) and dq.shape == (nq,) and tau.shape == (nq,)
            assert f_ext.shape == (6,)
            assert policy_latent.shape[0] == self._config.latent_dim
            assert cam_timestamps.shape[0] == self._config.n_cameras
        buf.append(t_ns, q, dq, tau, f_ext, policy_latent, cam_timestamps, flags)

    # -- trigger path ----------------------------------------------------------

    def trigger(
        self,
        trigger: FlightTrigger,
        rollout_id: str,
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[BlackBoxSnapshotRecord]:
        """Seal the last ``window_seconds`` and persist a black-box dump.

        MUST be called from a non-RT thread (the episode supervisor).  The
        1 kHz writer never blocks: the active buffer is swapped out in O(1)
        and serialization happens on the sealed buffer.  If another thread
        is currently flushing a dump, this one is queued and drained on the
        next ``start_rollout`` / ``finish_rollout`` (bounded, drop-oldest).
        """
        with self._lock:
            if self._active is None or self._current_rollout != rollout_id:
                _LOG.warning(
                    "trigger %s for rollout %s with no matching active buffer",
                    trigger.value,
                    rollout_id,
                )
                return None
            sealed_buf = self._active
            harness = self._current_harness or ""
            task = self._current_task or ""
            self._active = RingBuffer1kHz(self._config)
            if not self._flush_lock.acquire(blocking=False):
                if len(self._sealed_pending) >= self._config.max_seal_queue:
                    self._sealed_pending.pop(0)
                    self._dropped_seals += 1
                self._sealed_pending.append(
                    (rollout_id, trigger, sealed_buf, dict(meta or {}))
                )
                self._ledger.append_event(
                    EventType.FLIGHT_DUMP,
                    {
                        "snapshot_hash": None,
                        "rollout_id": rollout_id,
                        "trigger": trigger.value,
                        "queued": True,
                        "harness_hash": harness,
                    },
                )
                return None
        # --- outside both locks: heavy work on the sealed buffer ---------------
        try:
            window = sealed_buf.seal()
            blob = self._provider.capture()
            record = self._persist_snapshot(window, blob, trigger, rollout_id)
        finally:
            self._flush_lock.release()
        self._emitter.emit(
            "FLIGHT_DUMP",
            {
                "snapshot_hash": record.snapshot_hash,
                "rollout_id": rollout_id,
                "trigger": trigger.value,
                "num_samples": record.num_samples,
                "window_s": record.t_window_seconds,
            },
            severity="WARN",
            rollout_id=rollout_id,
        )
        self._ledger.append_event(
            EventType.FLIGHT_DUMP,
            {
                "snapshot_hash": record.snapshot_hash,
                "rollout_id": rollout_id,
                "trigger": trigger.value,
                "num_samples": record.num_samples,
                "harness_hash": harness,
                "task_name": task,
                "meta": meta or {},
            },
        )
        return record

    def _drain_pending_seals(self) -> None:
        """Serialize seals queued while another flush was in progress.

        Called with no locks held; takes the flush lock so at most one
        drain runs at a time.
        """
        if not self._sealed_pending:
            return
        with self._flush_lock:
            while self._sealed_pending:
                entry = self._sealed_pending.pop(0)
                rollout, trig, buf, meta = entry
                try:
                    window = buf.seal()
                    blob = self._provider.capture()
                    record = self._persist_snapshot(window, blob, trig, rollout)
                    self._ledger.append_event(
                        EventType.FLIGHT_DUMP,
                        {
                            "snapshot_hash": record.snapshot_hash,
                            "rollout_id": rollout,
                            "trigger": trig.value,
                            "num_samples": record.num_samples,
                            "late_drain": True,
                            "meta": meta,
                        },
                    )
                except Exception:  # noqa: BLE001 - telemetry must never kill RT
                    self._dropped_seals += 1
                    _LOG.exception("late seal drain failed")

    def _persist_snapshot(
        self,
        window: SealedWindow,
        blob: SimStateBlob,
        trigger: FlightTrigger,
        rollout_id: str,
    ) -> BlackBoxSnapshotRecord:
        tmp_dir = self._artifact_root / "blackbox" / ".tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_telem = tmp_dir / "telemetry.tmp"
        fmt = window.write(tmp_telem, compression=self._config.parquet_compression)
        if fmt == "npz":
            tmp_telem = tmp_telem.with_suffix(".npz")
        tmp_state = tmp_dir / "state.tmp.npz"
        blob.write(tmp_state)

        h = hashlib.sha256()
        for p in (tmp_telem, tmp_state):
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        snapshot_hash = "bbx_" + h.hexdigest()

        telem_final = (
            self._artifact_root / "blackbox" / snapshot_hash[:2]
            / f"{snapshot_hash}.parquet"
        )
        if fmt == "npz":
            telem_final = telem_final.with_suffix(".npz")
        state_final = (
            self._artifact_root / "blackbox" / snapshot_hash[:2]
            / f"{snapshot_hash}.state.npz"
        )
        telem_final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_telem, telem_final)
        os.replace(tmp_state, state_final)

        record = BlackBoxSnapshotRecord(
            snapshot_hash=snapshot_hash,
            rollout_id=rollout_id,
            trigger=trigger,
            parquet_path=str(telem_final),
            sim_state_path=str(state_final),
            format=fmt,
            t_window_seconds=round(window.duration_s, 6),
            sample_rate_hz=self._config.sample_rate_hz,
            num_samples=window.n,
            mujoco_model_hash=blob.model_hash,
            created_at_unix=unix_now(),
        )
        self._ledger.record_blackbox_dump(record)
        return record

    # -- real-to-sim replay & auto-SysID ----------------------------------------

    def reconstruct_failure_in_sim(self, snapshot_hash: str) -> FailureReconstructionReport:
        """Deterministically replay a physical failure inside the simulator.

        1. Load the black-box dump, verify artifact content hashes.
        2. Restore the exact sim state captured at trigger time.
        3. Open-loop replay the recorded torque trace; measure where the sim
           trajectory diverges from the recorded physical trajectory.
        4. Run bounded deterministic coordinate descent over friction /
           damping scales to minimize replay divergence.
        5. Persist a SimToRealGapReport (the planner consumes these as
           "which physics parameters are currently uncalibrated" warnings).
        """
        record = self._ledger.get_blackbox_snapshot(snapshot_hash)
        if self._backend_factory is None:
            raise RuntimeError(
                "no replay backend configured: pass backend_factory="
                "MujocoReplayBackend (requires the mujoco extra) or a test"
                " backend"
            )
        # integrity: content-addressed store must hash to its own name
        h = hashlib.sha256()
        for p in (Path(record.parquet_path), Path(record.sim_state_path)):
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        if "bbx_" + h.hexdigest() != record.snapshot_hash:
            raise RuntimeError(
                f"blackbox artifact integrity failure for {snapshot_hash}"
            )

        window = SealedWindow.read(Path(record.parquet_path))
        blob = SimStateBlob.read(Path(record.sim_state_path))
        backend = self._backend_factory(blob)
        nq = window.n_joints

        def replay_cost(friction: Dict[str, float], damping: Dict[str, float]) -> float:
            backend.load_state(blob)
            backend.apply_param_offsets(friction, damping)
            obs = backend.step_batch(window.tau)
            pos_err = np.linalg.norm(obs[:, :nq] - window.q, axis=1)
            vel_err = np.linalg.norm(obs[:, nq:] - window.dq, axis=1)
            return float(np.mean(pos_err) + 0.1 * np.mean(vel_err))

        residual_before = replay_cost({}, {})
        optimizer = SysIDOptimizer(
            self._specs,
            budget=self._sysid_budget,
        )
        result = optimizer.minimize(replay_cost)
        residual_after = result.cost_after

        # divergence onset: first sample where tracking error exceeds 3x the
        # median of the leading decile (ignores noise, catches divergence)
        backend.load_state(blob)
        obs = backend.step_batch(window.tau)
        pos_err = np.linalg.norm(obs[:, :nq] - window.q, axis=1)
        onset_s: Optional[float] = None
        lead = max(1, pos_err.shape[0] // 10)
        thresh = 3.0 * float(np.median(pos_err[:lead])) + 1e-9
        above = np.nonzero(pos_err > thresh)[0]
        if above.size:
            idx = int(above[0])
            onset_s = float((window.t_ns[idx] - window.t_ns[0]) / 1e9)

        improvement = (
            1.0 - residual_after / residual_before if residual_before > 0 else 0.0
        )
        if improvement >= self._sysid_min_improvement:
            # replay divergence materially reduced -> calibration is real
            status = SysIDStatus.APPLIED
        elif residual_after <= self._sysid_accept_residual:
            # already replays reality well and barely moved -> nothing to do
            status = SysIDStatus.INSUFFICIENT_CHANGE
        else:
            # cannot reduce divergence within budget -> drift warning
            status = SysIDStatus.FAILED_TO_CONVERGE

        report = FailureReconstructionReport(
            snapshot_hash=record.snapshot_hash,
            rollout_id=record.rollout_id,
            trigger=record.trigger,
            n_samples=record.num_samples,
            window_seconds=record.t_window_seconds,
            residual_before=round(residual_before, 8),
            residual_after=round(residual_after, 8),
            divergence_onset_s=onset_s,
            friction_offsets={k: round(v, 4) for k, v in result.friction.items()},
            damping_offsets={k: round(v, 4) for k, v in result.damping.items()},
            sysid_status=status,
            n_cost_evals=result.n_evals,
            report_path="",
        )
        report_dir = self._artifact_root / "sysid" / record.snapshot_hash[:2]
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"{record.snapshot_hash}.json"
        report_path.write_text(report.to_json(), encoding="utf-8")
        report.report_path = str(report_path)

        # Persist the calibration + drift record.  sim/real metrics use the
        # 24h success-rate convention (queryable history for the planner);
        # replay residuals ride along as first-class columns.
        rollout = self._lookup_rollout(record.rollout_id)
        task_name = rollout.task_name if rollout is not None else "unknown-task"
        harness_hash = rollout.harness_hash if rollout is not None else None
        sim_rate, real_rate = self._recent_success_rates(task_name)
        gap_report = SimToRealGapReport(
            task_name=task_name,
            harness_hash=harness_hash,
            sim_metric=sim_rate,
            real_metric=real_rate,
            gap=round(sim_rate - real_rate, 4),
            friction_offsets=report.friction_offsets,
            damping_offsets=report.damping_offsets,
            status=status,
            snapshot_hash=record.snapshot_hash,
            residual_before=report.residual_before,
            residual_after=report.residual_after,
        )
        self._ledger.record_sim_to_real_gap(gap_report)
        self._ledger.append_event(
            EventType.FAILURE_RECONSTRUCTED,
            {
                "snapshot_hash": record.snapshot_hash,
                "rollout_id": record.rollout_id,
                "residual_before": report.residual_before,
                "residual_after": report.residual_after,
                "improvement": round(improvement, 4),
                "sysid_status": status.value,
                "divergence_onset_s": onset_s,
                "friction_offsets": report.friction_offsets,
                "damping_offsets": report.damping_offsets,
            },
        )
        self._emitter.emit(
            "FAILURE_RECONSTRUCTED",
            {
                "snapshot_hash": record.snapshot_hash,
                "residual_before": report.residual_before,
                "residual_after": report.residual_after,
                "sysid_status": status.value,
            },
            severity="INFO",
        )
        return report

    def _lookup_rollout(
        self, rollout_id: str
    ) -> Optional[RolloutTelemetryBundle]:
        conn = self._ledger._require_open()  # noqa: SLF001 - same package, read-only
        row = conn.execute(
            "SELECT * FROM rollout_telemetry WHERE rollout_id = ?", (rollout_id,)
        ).fetchone()
        if row is None:
            return None
        return RolloutTelemetryBundle(
            rollout_id=row["rollout_id"],
            batch_id=row["batch_id"],
            harness_hash=row["harness_hash"],
            task_name=row["task_name"],
            domain=RolloutDomain(row["domain"]),
            stratum=DifficultyStrataBin(row["stratum"]),
            success=bool(row["success"]),
            time_to_completion=row["time_to_completion"],
            cumulative_jerk=float(row["cumulative_jerk"]),
            torque_penalty=float(row["torque_penalty"]),
            inference_deadline_misses=int(row["inference_deadline_misses"]),
            force_spike_max=float(row["force_spike_max"]),
            collision_count=int(row["collision_count"]),
            reset_outcome=None,
            flight_snapshot_hash=row["flight_snapshot_hash"],
            t_start_unix=float(row["t_start_unix"]),
            t_end_unix=float(row["t_end_unix"]),
            video_uri=row["video_uri"],
            vlm_failure_summary=row["vlm_failure_summary"],
        )

    def _recent_success_rates(
        self, task_name: str, window_s: float = 86400.0
    ) -> Tuple[float, float]:
        since = unix_now() - window_s
        conn = self._ledger._require_open()  # noqa: SLF001 - read-only
        out = []
        for domain in (RolloutDomain.SIM, RolloutDomain.PHYSICAL):
            row = conn.execute(
                "SELECT AVG(success) AS rate FROM rollout_telemetry"
                " WHERE task_name = ? AND domain = ? AND t_start_unix >= ?",
                (task_name, domain.value, since),
            ).fetchone()
            rate = float(row["rate"]) if row is not None and row["rate"] is not None else 0.0
            out.append(rate)
        return out[0], out[1]

    # -- terminal dashboard -------------------------------------------------------

    def render_dashboard(self, *, rows: int = 20) -> str:
        """One-screen, zero-SaaS lab status from the analytical views."""
        lines: List[str] = []
        lines.append("=" * 78)
        lines.append(f" ROBOCORE LAB DASHBOARD  ({unix_to_iso(unix_now())})")
        lines.append("=" * 78)

        lines.append("-- v_hourly_throughput " + "-" * 53)
        lines.append(
            f"{'hour':<20}{'task':<21}{'dom':<9}{'n':>4}{'succ%':>7}"
            f"{'ttc':>7}{'jerk':>9}{'bat':>4}"
        )
        for r in self._ledger.query_view("v_hourly_throughput", rows):
            lines.append(
                f"{str(r['hour']):<20}{str(r['task_name']):<21}"
                f"{str(r['domain']):<9}{int(r['n_rollouts']):>4}"
                f"{float(r['success_pct']):>6.1f}%"
                f"{(r['mean_ttc_s'] or 0.0):>7.2f}"
                f"{(r['mean_cum_jerk'] or 0.0):>9.1f}"
                f"{int(r['n_batches']):>4}"
            )

        lines.append("-- v_failure_mode_pareto " + "-" * 51)
        lines.append(
            f"{'task':<21}{'failure_mode':<28}{'n':>4}{'%task':>7}"
            f"{'maxF':>8}{'col':>5}"
        )
        for r in self._ledger.query_view("v_failure_mode_pareto", rows):
            lines.append(
                f"{str(r['task_name']):<21}{str(r['failure_mode'])[:27]:<28}"
                f"{int(r['n_failures']):>4}{float(r['pct_of_task_failures']):>6.1f}%"
                f"{float(r['max_force_spike'] or 0.0):>8.2f}"
                f"{int(r['n_with_collision'] or 0):>5}"
            )

        lines.append("-- v_sim_vs_real_drift " + "-" * 53)
        lines.append(
            f"{'task':<21}{'sim':>7}{'real':>7}{'gap':>8}"
            f"{'res_b':>9}{'res_a':>9}  status"
        )
        for r in self._ledger.query_view("v_sim_vs_real_drift", rows):
            lines.append(
                f"{str(r['task_name']):<21}{float(r['sim_success_24h'] or 0):>7.3f}"
                f"{float(r['real_success_24h'] or 0):>7.3f}"
                f"{float(r['success_gap'] or 0):>8.3f}"
                f"{float(r['replay_residual_before'] or 0):>9.5f}"
                f"{float(r['replay_residual_after'] or 0):>9.5f}"
                f"  {r['sysid_status']}"
            )
        lines.append("=" * 78)
        return "\n".join(lines)

    @property
    def dropped_seals(self) -> int:
        return self._dropped_seals
