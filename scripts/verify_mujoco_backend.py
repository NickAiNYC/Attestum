"""verify_mujoco_backend.py -- prove the production MujocoReplayBackend path:
load a real MuJoCo model, generate 'reality' with 1.3x joint damping, replay
the recorded torque trace through MujocoReplayBackend at 1.0x (divergent)
and at the calibrated 1.3x (convergent).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from robocore.telemetry.flight_recorder import (  # noqa: E402
    MujocoReplayBackend,
    SimStateBlob,
)

XML = """<mujoco>
  <option timestep="0.001"/>
  <worldbody>
    <body name="b1" pos="0 0 0.5">
      <joint name="j1" type="hinge" axis="0 1 0" damping="0.8"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="1.2"/>
    </body>
    <body name="b2" pos="0 0 -0.3">
      <joint name="j2" type="hinge" axis="0 1 0" damping="0.6"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.9"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" ctrlrange="-2 2"/>
    <motor joint="j2" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

TRUE_SCALE = 1.3
T = 800


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="robocore-mj-"))
    xml_path = tmp / "pendula.xml"
    xml_path.write_text(XML)

    t = np.arange(T) * 0.001
    tau = np.zeros((T, 2))
    tau[:, 0] = 0.4 * np.sin(2 * np.pi * 2.0 * t)
    tau[:, 1] = 0.3 * np.cos(2 * np.pi * 1.5 * t)

    # ---- "reality": same model with damping scaled 1.3x ------------------
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.dof_damping[:] = model.dof_damping * TRUE_SCALE
    data.qpos[:] = [0.5, -0.3]
    data.qvel[:] = [0.1, 0.05]
    mujoco.mj_forward(model, data)
    q_real = np.zeros((T, 2))
    dq_real = np.zeros((T, 2))
    for k in range(T):
        data.ctrl[:] = tau[k]
        mujoco.mj_step(model, data)
        q_real[k] = data.qpos
        dq_real[k] = data.qvel

    blob = SimStateBlob(
        model_hash="mdl_" + __import__("hashlib").sha256(xml_path.read_bytes()).hexdigest(),
        arrays={
            "qpos": np.array([0.5, -0.3]),
            "qvel": np.array([0.1, 0.05]),
            "ctrl": np.zeros(2),
        },
        meta={"engine": "mujoco", "model_xml_path": str(xml_path),
              "nq": 2, "nv": 2, "nu": 2, "sim_time": 0.0},
    )

    # ---- replay at default params (1.0x): must diverge --------------------
    backend = MujocoReplayBackend(blob)
    obs = backend.step_batch(tau)
    err_default = float(np.mean(np.linalg.norm(obs[:, :2] - q_real, axis=1)))

    # ---- replay at calibrated 1.3x: must converge --------------------------
    backend.load_state(blob)
    backend.apply_param_offsets({}, {"dof_damping_scale": TRUE_SCALE})
    obs_cal = backend.step_batch(tau)
    err_cal = float(np.mean(np.linalg.norm(obs_cal[:, :2] - q_real, axis=1)))

    print(f"replay error @ 1.0x damping: {err_default:.6f}")
    print(f"replay error @ 1.3x damping: {err_cal:.6f}")
    assert err_cal < 0.25 * err_default, (
        f"calibration did not converge: {err_cal} vs {err_default}"
    )
    assert err_default > 1e-4, "reality trace looks identical to sim; test is vacuous"
    print("MUJOCO BACKEND VERIFY: OK "
          f"(calibrated error is {100 * (1 - err_cal / err_default):.1f}% lower)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
