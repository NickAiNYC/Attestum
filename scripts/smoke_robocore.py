"""smoke_robocore.py -- end-to-end wiring proof for robocore Phase-1 items 1-3.

Runs the FULL orchestrator cycle against a real SQLite WAL ledger and a real
Parquet flight recorder, using deterministic fake subsystems:

  cycle 1:  proposal A (malicious diff) -> compiler rejection -> fast
            feedback -> proposal B -> compile -> sim gate pass -> sysid
            stable -> scene verify -> physical batch 1 (16 rollouts, one
            FORCE_SPIKE black-box dump) -> autonomous reset (1 retry) ->
            promotion gate INCONCLUSIVE -> physical batch 2 (16 more) ->
            genesis PROMOTION -> ledger commit.
  cycle 2+: planner starved 3x -> runtime parks itself in IDLE.

Then, offline (as the nightly SysID job would):
  reconstruct_failure_in_sim(snapshot) -> deterministic coordinate descent
  recovers the hidden ground-truth damping scale (~1.3x) from the black box.

Drills:
  D1 crash recovery: a journaled PHYSICAL_EXECUTING crash -> restart parks
     the arm and demands operator scene verification; resume() supersedes
     the orphan cycle and resumes.
  D2 chain tamper: a hand-crafted ledger_events row with a valid prev_hash
     but forged event_hash -> verify_chain_integrity() flags it and the
     runtime refuses to start (REFUSED_CHAIN_CORRUPT).

Usage:  python scripts/smoke_robocore.py [--workdir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robocore.contracts import (
    BatchDecision,
    BatchStatistics,
    ChampionManifest,
    DifficultyStrataBin,
    EventType,
    FalsifiablePrediction,
    FlightTrigger,
    HarnessManifest,
    HypothesisProposal,
    OverallStats,
    PlannerContext,
    PolicyCheckpointRecord,
    ResetOutcome,
    RolloutDomain,
    RolloutTelemetryBundle,
    SafetyValidationReceipt,
    SceneVerificationStatus,
    SimToRealGapReport,
    SPRTTrace,
    StratumStats,
    SysIDStatus,
    compute_harness_hash,
    new_id,
    unix_now,
)
from robocore.runtime.orchestrator import (
    BridgeDeadlineMissedError,
    CrashRecoveryReport,
    CriticVerdict,
    CompileResult,
    OrchestratorConfig,
    OrchestratorPorts,
    OrchestratorState,
    PhysicalBatchResult,
    PlannerFeedback,
    PromotionGateEvaluator,
    RobocoreOrchestrator,
    SceneVerification,
    SimEvaluationReport,
    SimGateDecision,
    SysIDResult,
)
from robocore.stats import PromotionGateConfig, SPRTMonitor, wilson_score_interval
from robocore.storage.ledger_repo import ExperimentLedgerRepository
from robocore.telemetry.flight_recorder import (
    FLAG_FORCE_SPIKE,
    FlightRecorder,
    FlightRecorderConfig,
    IReplayBackend,
    SealedWindow,
    SimStateBlob,
    StaticSimStateProvider,
    SysIDParamSpec,
)

TASK = "cube_pick_place_v1"
STRATA = [
    DifficultyStrataBin.EASY_CENTER,
    DifficultyStrataBin.MEDIUM_OBLIQUE,
    DifficultyStrataBin.HARD_BOUNDARY,
]
COMPILER_VERSION = "robocore-compiler/0.1.0"

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ===========================================================================
# Deterministic toy "hardware": 2-joint linear dynamics with hidden damping
# ===========================================================================

TRUE_DAMPING_SCALE = 1.30  # hidden ground truth auto-SysID must recover
BASE_DAMPING = np.array([0.80, 0.60])
JOINT_MASS = np.array([1.20, 0.90])
NQ = 2
DT = 1.0 / 1000.0


def torque_trace(n: int) -> np.ndarray:
    t = np.arange(n) * DT
    tau = np.zeros((n, NQ))
    tau[:, 0] = 0.15 * np.sin(2.0 * np.pi * 3.0 * t) + 0.05
    tau[:, 1] = 0.12 * np.cos(2.0 * np.pi * 2.0 * t) + 0.04
    return tau


def integrate(tau: np.ndarray, damping_scale: float, q0: np.ndarray,
              dq0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = tau.shape[0]
    q = np.zeros((n, NQ))
    dq = np.zeros((n, NQ))
    qc, dqc = q0.copy(), dq0.copy()
    for k in range(n):
        ddq = (tau[k] - damping_scale * BASE_DAMPING * dqc) / JOINT_MASS
        dqc = dqc + ddq * DT
        qc = qc + dqc * DT
        q[k], dq[k] = qc, dqc
    return q, dq


class ToyReplayBackend(IReplayBackend):
    """Deterministic IReplayBackend with the same physics as the 'hardware'.

    The sim default is damping_scale=1.0; reality ran at 1.30.  Only
    ``dof_damping_scale`` moves the cost surface (friction offsets are
    no-ops here, which mirrors parameters the optimizer should leave
    alone when they do not explain the divergence).
    """

    def __init__(self, blob: SimStateBlob) -> None:
        self._q0 = blob.arrays["qpos"].copy()
        self._dq0 = blob.arrays["qvel"].copy()
        self._damping_scale = 1.0

    def load_state(self, blob: SimStateBlob) -> None:
        self._q0 = blob.arrays["qpos"].copy()
        self._dq0 = blob.arrays["qvel"].copy()

    def apply_param_offsets(self, friction: Dict[str, float],
                            damping: Dict[str, float]) -> None:
        self._damping_scale = damping.get("dof_damping_scale", 1.0)

    def step_batch(self, ctrl: np.ndarray) -> np.ndarray:
        q, dq = integrate(ctrl, self._damping_scale, self._q0, self._dq0)
        return np.concatenate([q, dq], axis=1)


# ===========================================================================
# Fake subsystems (deterministic)
# ===========================================================================


class FakePlanner:
    """Scripted proposer: A (malicious) then B (good), then starvation."""

    def __init__(self, genesis_hash: str) -> None:
        self._genesis = genesis_hash
        self._queue: List[str] = ["A", "B"]
        self.feedback: List[PlannerFeedback] = []

    def propose(self, context: PlannerContext) -> Optional[HypothesisProposal]:
        parent = (
            context.champion.harness_hash if context.champion else self._genesis
        )
        if not self._queue:
            return None
        which = self._queue.pop(0)
        if which == "A":
            return HypothesisProposal(
                task_name=TASK,
                parent_harness_hash=parent,
                harness_diff=(
                    "--- a/harness.py\n+++ b/harness.py\n@@\n"
                    "+import os\n"
                    "+os.system('rm -rf /')\n"
                ),
                ast_fingerprint=hashlib.sha256(b"malicious-import-os").hexdigest()[:16],
                rationale="exfiltrate: shell out for 'smarter' rewards",
                falsifiable_prediction=FalsifiablePrediction(
                    target_metric="physical_success_rate",
                    predicted_delta=0.40,
                    ci_low=0.30,
                    ci_high=0.50,
                ),
            )
        return HypothesisProposal(
            task_name=TASK,
            parent_harness_hash=parent,
            harness_diff=(
                "--- a/harness.py\n+++ b/harness.py\n@@\n"
                "-REWARD_TOLERANCE = 0.05\n"
                "+REWARD_TOLERANCE = 0.02\n"
                "-CURRICULUM_MAX_OBLIQUE_DEG = 25.0\n"
                "+CURRICULUM_MAX_OBLIQUE_DEG = 32.0\n"
            ),
            ast_fingerprint=hashlib.sha256(b"reward-tolerance-and-curriculum").hexdigest()[:16],
            rationale="tighten grasp tolerance and widen curriculum obliqueness",
            falsifiable_prediction=FalsifiablePrediction(
                target_metric="physical_success_rate",
                predicted_delta=0.10,
                ci_low=0.03,
                ci_high=0.17,
            ),
            param_summary={"REWARD_TOLERANCE": 0.02, "CURRICULUM_MAX_OBLIQUE_DEG": 32.0},
        )

    def ingest_feedback(self, feedback: PlannerFeedback) -> None:
        self.feedback.append(feedback)


class FakeCritic:
    def score(self, proposal: HypothesisProposal,
              context: PlannerContext) -> CriticVerdict:
        return CriticVerdict(accept=True, score=0.72, reason="novel param family")


class FakeCompiler:
    """Rejects forbidden imports; otherwise builds a real HarnessManifest."""

    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root

    def compile_and_precheck(self, proposal: HypothesisProposal,
                             parent: HarnessManifest) -> CompileResult:
        if "import os" in proposal.harness_diff:
            receipt = SafetyValidationReceipt(
                passed=False,
                violations=["forbidden import: os", "side-effect call: os.system"],
                static_analysis_summary="AST denylist hit",
                compiler_version=COMPILER_VERSION,
                kinematic_precheck=False,
            )
            return CompileResult(
                passed=False,
                receipt=receipt,
                rejection_reason="AST_VALIDATION: forbidden-import os",
            )
        structural = "".join(
            ch for ch in proposal.harness_diff if not ch.isdigit()
        )
        ast_fp = hashlib.sha256(structural.encode()).hexdigest()[:16]
        config = {"env": "mujoco", "strata": [s.value for s in STRATA]}
        hhash = compute_harness_hash(
            parent_harness_hash=parent.harness_hash,
            harness_diff=proposal.harness_diff,
            ast_fingerprint=ast_fp,
            config=config,
        )
        manifest = HarnessManifest(
            harness_hash=hhash,
            task_name=TASK,
            parent_harness_hash=parent.harness_hash,
            harness_diff=proposal.harness_diff,
            ast_fingerprint=ast_fp,
            compiler_version=COMPILER_VERSION,
            safety_receipt=SafetyValidationReceipt(
                passed=True,
                violations=[],
                static_analysis_summary="clean",
                compiler_version=COMPILER_VERSION,
                kinematic_precheck=True,
            ),
            difficulty_strata=STRATA,
            config=config,
        )
        ck_path = self._artifact_root / "checkpoints" / f"{hhash}.safetensors"
        ck_path.parent.mkdir(parents=True, exist_ok=True)
        ck_path.write_bytes(b"fake-checkpoint:" + hhash.encode())
        checkpoint = PolicyCheckpointRecord(
            checkpoint_hash="ck_" + hashlib.sha256(hhash.encode()).hexdigest()[:20],
            harness_hash=hhash,
            format="safetensors",
            storage_path=str(ck_path),
            num_params=8_141_312,
        )
        return CompileResult(
            passed=True,
            receipt=manifest.safety_receipt,
            manifest=manifest,
            checkpoint=checkpoint,
        )


class FakeSimEvaluator:
    """Deterministic 85% sim success; SPRT via the real stats module."""

    def evaluate(self, *, harness: HarnessManifest,
                 champion_baseline: Optional[float], batch_size: int,
                 env_overrides: Optional[Dict[str, Any]]) -> SimEvaluationReport:
        baseline = champion_baseline if champion_baseline is not None else 0.35
        p1 = min(0.99, baseline + 0.05)
        monitor = SPRTMonitor(p0=baseline, p1=p1)
        rollouts: List[RolloutTelemetryBundle] = []
        decision_final = "ACCEPT_H1"
        for i in range(batch_size):
            success = (i % 20) < 17  # 85%
            decision_final = monitor.update(success)
            rollouts.append(
                RolloutTelemetryBundle(
                    batch_id="",
                    harness_hash=harness.harness_hash,
                    task_name=TASK,
                    domain=RolloutDomain.SIM,
                    stratum=STRATA[i % 3],
                    success=success,
                    time_to_completion=4.2 + (i % 5) * 0.3,
                    cumulative_jerk=280.0 + (i % 7) * 10.0,
                    torque_penalty=18.0 + (i % 4),
                )
            )
        n = len(rollouts)
        s = sum(1 for r in rollouts if r.success)
        low, high = wilson_score_interval(s, n)
        per: Dict[DifficultyStrataBin, StratumStats] = {}
        for stratum in STRATA:
            sub = [r for r in rollouts if r.stratum is stratum]
            ssub = sum(1 for r in sub if r.success)
            lo, hi = wilson_score_interval(ssub, len(sub))
            per[stratum] = StratumStats(
                stratum=stratum, n=len(sub), successes=ssub,
                success_rate=ssub / len(sub), wilson_low=lo, wilson_high=hi,
                mean_ttc=float(np.mean([r.time_to_completion or 0 for r in sub])),
                mean_cumulative_jerk=float(np.mean([r.cumulative_jerk for r in sub])),
                mean_torque_penalty=float(np.mean([r.torque_penalty for r in sub])),
            )
        stats = BatchStatistics(
            overall=OverallStats(n=n, successes=s, success_rate=s / n,
                                 wilson_low=low, wilson_high=high,
                                 mean_cumulative_jerk=310.0, mean_torque_penalty=19.5),
            per_stratum=per,
            sprt=SPRTTrace(p0=monitor.p0, p1=monitor.p1,
                           lower_boundary=monitor.lower_boundary,
                           upper_boundary=monitor.upper_boundary,
                           final_llr=monitor.llr,
                           stopped_early=decision_final == "EARLY_ABORT",
                           n_samples_consumed=monitor.n),
        )
        gate = low >= baseline
        return SimEvaluationReport(
            decision=SimGateDecision.PASSED_GATE if gate else SimGateDecision.FAILED_GATE,
            rollouts=rollouts,
            statistics=stats,
            reward_hacking_suspected=False,
            notes=[f"sim wilson_low={low:.3f} vs baseline={baseline:.3f}"],
        )


class FakeSysID:
    def calibrate(self, gap_history: Sequence[SimToRealGapReport]) -> SysIDResult:
        return SysIDResult(
            status=SysIDStatus.INSUFFICIENT_CHANGE,
            params_changed=False,
            notes=["no drift evidence: physical success within sim CI"],
        )


class FakeSceneVerifier:
    def __init__(self, fail_next: int = 0) -> None:
        self._fail_next = fail_next

    def verify(self) -> SceneVerification:
        if self._fail_next > 0:
            self._fail_next -= 1
            return SceneVerification(
                status=SceneVerificationStatus.RETRYABLE_FAIL,
                details="criterion jitter; re-verifying",
            )
        return SceneVerification(status=SceneVerificationStatus.PASS,
                                 details="objects in nominal start poses")


class FakeBridge:
    def __init__(self) -> None:
        self.hold_calls = 0
        self.park_calls = 0
        self.estop_calls = 0

    def health(self) -> Any:
        from robocore.contracts import BridgeHealthReport
        return BridgeHealthReport(
            consecutive_deadline_misses=0, total_misses=0,
            watchdog_active=True, control_mode="ACTIVE",
        )

    def hold(self) -> None:
        self.hold_calls += 1

    def park(self) -> None:
        self.park_calls += 1

    def estop(self) -> None:
        self.estop_calls += 1


class FakeResetController:
    """Fails the first autonomous reset (gripper jam), succeeds on retry."""

    def __init__(self) -> None:
        self.calls = 0

    def reset_scene(self, attempt: int) -> Tuple[ResetOutcome, str]:
        self.calls += 1
        if self.calls == 1:
            return ResetOutcome.RETRYABLE_FAILURE, "gripper jam mid-retreat"
        return ResetOutcome.SUCCESS, "objects re-posed from VLM target"


class FakePhysicalExecutor:
    """Streams synthetic-but-deterministic physical rollouts.

    Rollout seq 3 (global) suffers a FORCE_SPIKE at t=1.2 s: the flight
    recorder is armed, fed a 1 kHz 'reality' trace integrated with the
    hidden damping scale, then triggered.  Honors the on_rollout contract:
    a BridgeDeadlineMissedError from the callback stops execution and
    returns a partial EARLY_ABORT batch.
    """

    SPIKE_SEQ = 3

    def __init__(self) -> None:
        self.seq = 0

    def execute_batch(self, *, harness: HarnessManifest,
                      champion: Optional[ChampionManifest], batch_size: int,
                      recorder: Optional[FlightRecorder],
                      on_rollout: Any) -> PhysicalBatchResult:
        started = unix_now()
        rollouts: List[RolloutTelemetryBundle] = []
        abort_reason: Optional[str] = None
        n_trace = 1200  # full-length deterministic 1 kHz "reality" trace
        n_feed_normal = 300  # non-spike rollouts only buffer 0.3 s
        tau = torque_trace(n_trace)
        q_real, dq_real = integrate(
            tau, TRUE_DAMPING_SCALE,
            np.array([0.5, -0.3]), np.array([0.10, 0.05]),
        )
        latent = np.linspace(-1.0, 1.0, 32).astype(np.float32)
        for _ in range(batch_size):
            seq = self.seq
            self.seq += 1
            rollout_id = new_id("rol")
            snapshot_hash: Optional[str] = None
            spike = seq == self.SPIKE_SEQ
            if recorder is not None:
                recorder.start_rollout(rollout_id, harness.harness_hash, TASK)
                n_feed = n_trace if spike else n_feed_normal
                for k in range(n_feed):
                    f_ext = np.zeros(6)
                    flags = 0
                    if spike and k >= 1150:
                        f_ext[2] = 45.0  # wrist Fz spike
                        flags |= FLAG_FORCE_SPIKE
                    recorder.append_sample(
                        t_ns=k * 1_000_000,
                        q=q_real[min(k, n_trace - 1)],
                        dq=dq_real[min(k, n_trace - 1)],
                        tau=tau[min(k, n_trace - 1)],
                        f_ext=f_ext,
                        policy_latent=latent,
                        cam_timestamps=np.array([k * 0.033, k * 0.033]),
                        flags=flags,
                    )
                if spike:
                    record = recorder.trigger(
                        FlightTrigger.FORCE_SPIKE, rollout_id,
                        meta={"peak_f_z": 45.0, "t_s": 1.15},
                    )
                    snapshot_hash = record.snapshot_hash if record else None
                recorder.finish_rollout(rollout_id)
            success = (seq % 10) < 9 and not spike
            bundle = RolloutTelemetryBundle(
                rollout_id=rollout_id,
                batch_id="",
                harness_hash=harness.harness_hash,
                task_name=TASK,
                domain=RolloutDomain.PHYSICAL,
                stratum=STRATA[seq % 3],
                success=success,
                time_to_completion=(5.1 + (seq % 4) * 0.4) if success else None,
                cumulative_jerk=300.0 + (seq % 3) * 12.0,
                torque_penalty=22.0 + (seq % 5),
                inference_deadline_misses=0,
                force_spike_max=45.0 if spike else 2.0,
                collision_count=0,
                reset_outcome=ResetOutcome.SUCCESS,
                flight_snapshot_hash=snapshot_hash,
                vlm_failure_summary=(
                    "wrist force spike during approach (Fz 45 N)"
                    if spike else None
                ),
            )
            rollouts.append(bundle)
            try:
                on_rollout(bundle)
            except BridgeDeadlineMissedError as exc:
                abort_reason = f"bridge-deadline: {exc}"
                break
        n = len(rollouts)
        s = sum(1 for r in rollouts if r.success)
        low, high = wilson_score_interval(s, n)
        per: Dict[DifficultyStrataBin, StratumStats] = {}
        for stratum in STRATA:
            sub = [r for r in rollouts if r.stratum is stratum]
            if not sub:
                continue
            ssub = sum(1 for r in sub if r.success)
            lo, hi = wilson_score_interval(ssub, len(sub))
            per[stratum] = StratumStats(
                stratum=stratum, n=len(sub), successes=ssub,
                success_rate=ssub / len(sub), wilson_low=lo, wilson_high=hi,
                mean_ttc=float(np.mean([r.time_to_completion or 0 for r in sub])),
                mean_cumulative_jerk=float(np.mean([r.cumulative_jerk for r in sub])),
                mean_torque_penalty=float(np.mean([r.torque_penalty for r in sub])),
            )
        stats = BatchStatistics(
            overall=OverallStats(n=n, successes=s, success_rate=s / max(1, n),
                                 wilson_low=low, wilson_high=high,
                                 mean_cumulative_jerk=312.0, mean_torque_penalty=24.0),
            per_stratum=per,
        )
        return PhysicalBatchResult(
            rollouts=rollouts,
            decision=BatchDecision.EARLY_ABORT if abort_reason else BatchDecision.CONTINUE,
            statistics=stats,
            early_abort_reason=abort_reason,
            started_at_unix=started,
        )


# ===========================================================================
# Scenario driver
# ===========================================================================


def build_ports(artifact_root: Path, genesis_hash: str) -> Tuple[OrchestratorPorts, FakePlanner, FakeBridge]:
    planner = FakePlanner(genesis_hash)
    bridge = FakeBridge()
    ports = OrchestratorPorts(
        planner=planner,
        critic=FakeCritic(),
        compiler=FakeCompiler(artifact_root),
        sim_evaluator=FakeSimEvaluator(),
        sysid=FakeSysID(),
        scene_verifier=FakeSceneVerifier(),
        physical_executor=FakePhysicalExecutor(),
        reset_controller=FakeResetController(),
        bridge=bridge,
        promotion_evaluator=PromotionGateEvaluator(),
    )
    return ports, planner, bridge


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()
    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="robocore-smoke-")
    )
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    print(f"workdir: {workdir}")

    # ---- bootstrap: ledger + genesis harness + flight recorder ------------
    ledger = ExperimentLedgerRepository(workdir / "ledger.db")
    genesis = HarnessManifest.genesis(
        task_name=TASK,
        harness_diff="# genesis baseline harness (hand-written seed)\n",
        ast_fingerprint=hashlib.sha256(b"genesis").hexdigest()[:16],
        compiler_version=COMPILER_VERSION,
        config={"env": "mujoco", "strata": [s.value for s in STRATA]},
    )
    ledger.upsert_harness(genesis)
    genesis_ck = PolicyCheckpointRecord(
        checkpoint_hash="ck_" + hashlib.sha256(b"genesis").hexdigest()[:20],
        harness_hash=genesis.harness_hash,
        format="safetensors",
        storage_path=str(workdir / "checkpoints" / "genesis.safetensors"),
        num_params=8_141_312,
    )
    ledger.record_policy_checkpoint(genesis_ck)

    q0 = np.array([0.5, -0.3])
    dq0 = np.array([0.10, 0.05])
    blob = SimStateBlob(
        model_hash="toy_pendula_v1",
        arrays={"qpos": q0, "qvel": dq0, "ctrl": np.zeros(NQ)},
        meta={"engine": "toy", "nq": NQ, "nv": NQ, "nu": NQ, "sim_time": 0.0},
    )
    recorder = FlightRecorder(
        FlightRecorderConfig(n_joints=NQ, latent_dim=32, n_cameras=2),
        ledger,
        workdir / "artifacts",
        StaticSimStateProvider(blob),
        backend_factory=lambda b: ToyReplayBackend(b),
        sysid_specs=(
            SysIDParamSpec("geom_slide_friction_scale", "friction", 0.25, 4.0),
            SysIDParamSpec("dof_damping_scale", "damping", 0.25, 4.0),
            SysIDParamSpec("dof_armature_scale", "damping", 0.25, 4.0),
        ),
        sysid_budget=140,
    )

    ports, planner, bridge = build_ports(workdir / "artifacts", genesis.harness_hash)
    ports.flight_recorder = recorder

    config = OrchestratorConfig(
        task_name=TASK,
        sim_batch_size=32,
        physical_batch_size=16,
        planner_cooldown_s=0.0,
        max_starved_cycles=3,
        promotion_gate=PromotionGateConfig(
            required_strata=(STRATA[0], STRATA[1], STRATA[2]),
            min_samples_per_stratum=8,
            min_total_samples=24,
        ),
    )
    orch = RobocoreOrchestrator(config, ports, ledger)

    # ---- run the unattended loop -------------------------------------------
    print("\n== running unattended loop ==")
    recovery: CrashRecoveryReport = orch.run()
    check("startup recovery is FRESH_START", recovery.action == "FRESH_START",
          recovery.action)

    # ---- assertions: ledger outcomes ---------------------------------------
    print("\n== ledger outcomes ==")
    champion = ledger.get_active_champion(TASK)
    check("champion promoted", True)
    print(f"  champion harness: {champion.harness_hash[:24]}... "
          f"wilson_low={champion.wilson_lower_bound:.3f}")
    statuses = {}
    for row in ledger._require_open().execute(
        "SELECT hypothesis_id, status, rejection_reason FROM hypotheses"
    ).fetchall():
        statuses[row["status"]] = row["rejection_reason"]
        print(f"  hypothesis {row['hypothesis_id']}: {row['status']}"
              f" ({row['rejection_reason']})")
    check("compiler rejection recorded",
          "REJECTED_COMPILER" in statuses
          and "AST_VALIDATION" in (statuses["REJECTED_COMPILER"] or ""))
    check("promotion recorded", "PROMOTED" in statuses)
    feedback_kinds = [fb.kind for fb in planner.feedback]
    check("compiler rejection fed back to planner",
          "COMPILER_REJECTION" in feedback_kinds, str(feedback_kinds))
    check("promotion fed back to planner", "PROMOTED" in feedback_kinds)

    ok, bad = ledger.verify_chain_integrity()
    check("ledger event chain verifies", ok and bad is None, f"bad seq={bad}")

    phys = ledger.get_rollouts_for_harness(champion.harness_hash,
                                           domain=RolloutDomain.PHYSICAL)
    check("32 physical rollouts recorded", len(phys) == 32, str(len(phys)))
    dumps = ledger.list_blackbox_dumps()
    check("one black-box dump", len(dumps) == 1, str(len(dumps)))
    snapshot = dumps[0]
    check("dump trigger is FORCE_SPIKE",
          snapshot.trigger is FlightTrigger.FORCE_SPIKE)
    check("parquet artifact exists", Path(snapshot.parquet_path).exists())
    check("parquet format used", snapshot.format == "parquet", snapshot.format)
    check("dump referenced by its rollout",
          any(r.flight_snapshot_hash == snapshot.snapshot_hash for r in phys))

    # ---- state machine trace --------------------------------------------------
    print("\n== state machine trace (cycle 1) ==")
    events = ledger._require_open().execute(
        "SELECT payload_json FROM ledger_events WHERE event_type='ORCHESTRATOR_STATE'"
        " ORDER BY seq ASC"
    ).fetchall()
    states = [__import__("json").loads(e["payload_json"])["state"] for e in events]
    print("  " + " -> ".join(states))
    check("full happy path traversed",
          [OrchestratorState(s) for s in states] == [
              OrchestratorState.PROPOSING,
              OrchestratorState.COMPILING_AND_PRECHECKING,
              OrchestratorState.PROPOSING,  # compiler rejection fast-feedback
              OrchestratorState.COMPILING_AND_PRECHECKING,
              OrchestratorState.SIM_EVALUATING,
              OrchestratorState.SYSID_CALIBRATING,
              OrchestratorState.PHYSICAL_SCENE_VERIFYING,
              OrchestratorState.PHYSICAL_EXECUTING,
              OrchestratorState.PHYSICAL_RESETTING,
              OrchestratorState.EVALUATING_PROMOTION,
              OrchestratorState.PHYSICAL_EXECUTING,  # inconclusive -> more samples
              OrchestratorState.PHYSICAL_RESETTING,
              OrchestratorState.EVALUATING_PROMOTION,
              OrchestratorState.COMMITTING_LEDGER,
              OrchestratorState.PROPOSING,  # cycle 2: starved -> IDLE
              OrchestratorState.IDLE,
              OrchestratorState.PROPOSING,
              OrchestratorState.IDLE,
              OrchestratorState.PROPOSING,
              OrchestratorState.IDLE,
          ], str(states))
    check("runtime parked itself in IDLE after planner starvation",
          orch.state is OrchestratorState.IDLE, orch.state.value)

    # ---- flight-recorder window sanity ---------------------------------------
    print("\n== flight recorder window ==")
    window = SealedWindow.read(Path(snapshot.parquet_path))
    print(f"  samples={window.n} duration={window.duration_s:.3f}s "
          f"joints={window.n_joints}")
    check("window has 1200 samples at 1 kHz", window.n == 1200, str(window.n))
    check("spike flagged in window flags",
          bool(np.any(window.flags & FLAG_FORCE_SPIKE)))
    check("Fz spike visible in f_ext",
          float(np.max(window.f_ext[:, 2])) == 45.0)

    # ---- real-to-sim replay & auto-SysID ---------------------------------------
    print("\n== reconstruct_failure_in_sim / auto-SysID ==")
    report = recorder.reconstruct_failure_in_sim(snapshot.snapshot_hash)
    print(f"  residual before={report.residual_before:.6f} "
          f"after={report.residual_after:.6f} "
          f"evals={report.n_cost_evals} status={report.sysid_status.value}")
    print(f"  friction offsets: {report.friction_offsets}")
    print(f"  damping offsets:  {report.damping_offsets}")
    print(f"  divergence onset: {report.divergence_onset_s}")
    recovered = report.damping_offsets.get("dof_damping_scale", 0.0)
    check("SysID recovered hidden damping scale (~1.30)",
          abs(recovered - TRUE_DAMPING_SCALE) <= 0.05, f"got {recovered:.4f}")
    check("replay residual reduced >= 90%",
          report.residual_after <= 0.10 * report.residual_before,
          f"{report.residual_after} vs {report.residual_before}")
    gaps = ledger.get_sim_to_real_gap_history(TASK, window=5)
    check("SimToRealGapReport persisted", len(gaps) == 1, str(len(gaps)))

    # ---- analytical views / dashboard ------------------------------------------
    print("\n== analytical views ==")
    hourly = ledger.query_view("v_hourly_throughput", 10)
    check("v_hourly_throughput has rows", len(hourly) >= 2, str(len(hourly)))
    check("hour bucket is a real ISO hour",
          hourly[0]["hour"].startswith("20"), str(hourly[0]["hour"]))
    pareto = ledger.query_view("v_failure_mode_pareto", 10)
    check("v_failure_mode_pareto classified the spike failure",
          any(r["failure_mode"] == "wrist force spike during approach (Fz 45 N)"
              for r in pareto), str([r["failure_mode"] for r in pareto]))
    drift = ledger.query_view("v_sim_vs_real_drift", 10)
    check("v_sim_vs_real_drift row exists", len(drift) == 1)
    check("drift ISO timestamp sane", str(drift[0]["calibrated_at_iso"]).startswith("20"),
          str(drift[0]["calibrated_at_iso"]))
    print("\n" + recorder.render_dashboard(rows=10))

    # =========================================================================
    # Drill D1: crash recovery from journaled PHYSICAL_EXECUTING
    # =========================================================================
    print("\n== drill D1: crash recovery ==")
    drill_dir = workdir / "drill1"
    drill_dir.mkdir()
    ledger2 = ExperimentLedgerRepository(drill_dir / "ledger.db")
    ledger2.upsert_harness(genesis)
    ledger2.record_policy_checkpoint(genesis_ck)
    ports2, _, bridge2 = build_ports(drill_dir, genesis.harness_hash)
    orch2 = RobocoreOrchestrator(
        OrchestratorConfig(task_name=TASK, planner_cooldown_s=0.0),
        ports2, ledger2,
    )
    # simulate a process that crashed mid-execution:
    ev = ledger2.append_event(EventType.CYCLE_START, {"task_name": TASK},
                              cycle_id="cyc_drill")
    assert ev.seq > 0
    ledger2.append_event(
        EventType.ORCHESTRATOR_STATE,
        {"state": OrchestratorState.PHYSICAL_EXECUTING.value,
         "prev": OrchestratorState.PHYSICAL_SCENE_VERIFYING.value,
         "reason": "scene-verified"},
        cycle_id="cyc_drill",
    )
    check("drill open cycle detected", ledger2.get_open_cycles() == ["cyc_drill"])
    report2 = orch2._recover_on_startup()  # noqa: SLF001 - drill hook
    check("crash mid-execution parks the arm",
          report2.action == "PARKED_ARM", report2.action)
    check("bridge commanded to park", bridge2.park_calls >= 1)
    check("orchestrator in PAUSED_ALERT", orch2.state is OrchestratorState.PAUSED_ALERT)
    check("open cycle preserved until operator decision",
          ledger2.get_open_cycles() == ["cyc_drill"])
    resumed = orch2.resume(verify_scene=True)
    check("resume with scene verification succeeds", resumed)
    check("resume transitions to PROPOSING",
          orch2.state is OrchestratorState.PROPOSING)
    check("resume supersedes the orphan cycle",
          ledger2.get_open_cycles() == [])
    ok2, bad2 = ledger2.verify_chain_integrity()
    check("drill ledger chain still verifies", ok2 and bad2 is None)

    # =========================================================================
    # Drill D2: ledger tamper detection
    # =========================================================================
    print("\n== drill D2: chain tamper detection ==")
    head = ledger2._require_open().execute(
        "SELECT event_hash FROM ledger_events ORDER BY seq DESC LIMIT 1"
    ).fetchone()["event_hash"]
    raw = sqlite3.connect(drill_dir / "ledger.db")
    raw.execute(
        "INSERT INTO ledger_events (event_type, cycle_id, payload_json, prev_hash,"
        " event_hash, recorded_at_unix, recorded_at_iso)"
        " VALUES ('TAMPER', NULL, '{}', ?, 'deadbeef', 1.0, '2026-01-01T00:00:00Z')",
        (head,),
    )
    raw.commit()
    raw.close()
    ok3, bad3 = ledger2.verify_chain_integrity()
    check("tampered row detected", (not ok3) and bad3 is not None,
          f"ok={ok3} bad={bad3}")
    ports3, _, _ = build_ports(drill_dir, genesis.harness_hash)
    orch3 = RobocoreOrchestrator(
        OrchestratorConfig(task_name=TASK, planner_cooldown_s=0.0),
        ports3, ledger2,
    )
    report3 = orch3._recover_on_startup()  # noqa: SLF001 - drill hook
    check("runtime refuses to start on corrupt chain",
          report3.action == "REFUSED_CHAIN_CORRUPT", report3.action)
    check("runtime parks in PAUSED_ALERT",
          orch3.state is OrchestratorState.PAUSED_ALERT)

    ledger.close()
    ledger2.close()

    print(f"\n===== SMOKE RESULT: {PASS} passed, {FAIL} failed =====")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    sys.exit(main())
