"""robocore.safety.envelope -- hard physical envelope generation & clamping.

Deliverable 5 support module (additive extension).  Batch 1 froze
``contracts.py`` without physical-limit constants, so the hardcoded limits
live here as the single authority (see the batch report's DEVIATIONS
section).  The values are conservative defaults for a 7-DoF FR3-class
arm; operators may *tighten* them via config, never loosen them in code.

``envelope_generator`` is the only way the runtime materializes a physical
envelope (velocity / torque / position bounds per joint) from planner
parameter overrides.  Its contract is absolute: a finite-input call either
raises ``ValueError`` (unknown override key, non-finite value, inverted
position envelope) or returns an envelope in which **every** per-joint
bound is within ``PHYSICAL_LIMITS``.  No scaling, offset or override can
ever produce an out-of-limits envelope -- the clamp is structural, applied
last, per joint.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import FrozenSet, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, field_validator

__all__ = [
    "PHYSICAL_LIMITS",
    "PhysicalLimits",
    "PhysicalEnvelope",
    "ENVELOPE_OVERRIDE_KEYS",
    "envelope_generator",
]

_N_JOINTS = 7


@dataclass(frozen=True)
class PhysicalLimits:
    """Hardcoded physical limits (FR3-class conservative defaults).

    These are ceilings, not targets: everything the runtime ever commands
    is clamped *inside* them joint-by-joint.
    """

    n_joints: int = _N_JOINTS
    q_pos_lo: Tuple[float, ...] = (-2.8,) * _N_JOINTS  # rad
    q_pos_hi: Tuple[float, ...] = (2.8,) * _N_JOINTS  # rad
    q_vel_max: Tuple[float, ...] = (2.5,) * _N_JOINTS  # rad/s
    tau_max: Tuple[float, ...] = (30.0,) * _N_JOINTS  # N . m


#: The single hardcoded authority.
PHYSICAL_LIMITS = PhysicalLimits()


class PhysicalEnvelope(BaseModel):
    """A materialized per-joint physical envelope (always within limits)."""

    model_config = ConfigDict(frozen=True)

    q_pos_lo: Tuple[float, ...]
    q_pos_hi: Tuple[float, ...]
    q_vel_max: Tuple[float, ...]
    tau_max: Tuple[float, ...]
    source_note: str = "envelope_generator"

    @field_validator("q_pos_lo", "q_pos_hi", "q_vel_max", "tau_max")
    @classmethod
    def _right_arity(cls, v: Tuple[float, ...]) -> Tuple[float, ...]:
        if len(v) != PHYSICAL_LIMITS.n_joints:
            raise ValueError(
                f"envelope must have {PHYSICAL_LIMITS.n_joints} joints, got"
                f" {len(v)}"
            )
        return v


#: Global (all-joint) override keys; per-joint keys are regex-validated.
_GLOBAL_KEYS: FrozenSet[str] = frozenset(
    {"vel_scale", "tau_scale", "q_pos_margin"}
)
_JOINT_KEY_RE = re.compile(
    r"^(q_vel_max|tau_max|q_pos_lo|q_pos_hi)\.joint_([1-7])$"
)

#: The complete override-key vocabulary, materialized (anything else is a
#: ValueError).  Exposed for the fuzzing suite's strategy definitions.
ENVELOPE_OVERRIDE_KEYS: FrozenSet[str] = frozenset(
    list(_GLOBAL_KEYS)
    + [
        f"{bound}.joint_{j}"
        for bound in ("q_vel_max", "tau_max", "q_pos_lo", "q_pos_hi")
        for j in range(1, _N_JOINTS + 1)
    ]
)


def _joint_key(key: str) -> Tuple[str, int]:
    m = _JOINT_KEY_RE.match(key)
    if m is None:
        raise ValueError(
            f"unknown envelope override key: {key!r} (allowed: global"
            f" {sorted(_GLOBAL_KEYS)} or"
            " '<q_vel_max|tau_max|q_pos_lo|q_pos_hi>.joint_<1-7>')"
        )
    return m.group(1), int(m.group(2)) - 1


def envelope_generator(
    param_overrides: Mapping[str, float],
) -> PhysicalEnvelope:
    """Materialize a physical envelope from planner parameter overrides.

    Semantics (applied in order, then structurally clamped):

    * ``vel_scale`` / ``tau_scale`` -- multiply every per-joint velocity /
      torque ceiling (|value| is used: negative scales tighten through 0).
    * ``q_pos_margin`` -- pull both position bounds toward the joint center
      by that many rad (negative margins widen, clamped at the hard
      limits).
    * ``<bound>.joint_<1-7>`` -- set one specific joint bound directly.

    Fail-closed: non-finite values, unknown keys and inverted position
    envelopes raise ``ValueError``; every emitted bound is within
    ``PHYSICAL_LIMITS`` by construction.
    """
    for key, value in param_overrides.items():
        if key in _GLOBAL_KEYS or _JOINT_KEY_RE.match(key):
            if not math.isfinite(float(value)):
                raise ValueError(
                    f"envelope override {key}={value!r} is not finite"
                )
        else:
            raise ValueError(
                f"unknown envelope override key: {key!r} (allowed: global"
                f" {sorted(_GLOBAL_KEYS)} or"
                " '<q_vel_max|tau_max|q_pos_lo|q_pos_hi>.joint_<1-7>')"
            )

    lim = PHYSICAL_LIMITS
    vel = list(lim.q_vel_max)
    tau = list(lim.tau_max)
    lo = list(lim.q_pos_lo)
    hi = list(lim.q_pos_hi)

    if "vel_scale" in param_overrides:
        s = abs(float(param_overrides["vel_scale"]))
        vel = [min(v * s, lim.q_vel_max[i]) for i, v in enumerate(vel)]
    if "tau_scale" in param_overrides:
        s = abs(float(param_overrides["tau_scale"]))
        tau = [min(t * s, lim.tau_max[i]) for i, t in enumerate(tau)]
    if "q_pos_margin" in param_overrides:
        m = float(param_overrides["q_pos_margin"])
        lo = [max(v + m, lim.q_pos_lo[i]) for i, v in enumerate(lo)]
        hi = [min(v - m, lim.q_pos_hi[i]) for i, v in enumerate(hi)]

    for key, value in param_overrides.items():
        if _JOINT_KEY_RE.match(key):
            bound, j = _joint_key(key)
            v = float(value)
            if bound == "q_vel_max":
                vel[j] = min(abs(v), lim.q_vel_max[j])
            elif bound == "tau_max":
                tau[j] = min(abs(v), lim.tau_max[j])
            elif bound == "q_pos_lo":
                lo[j] = max(v, lim.q_pos_lo[j])
            else:
                hi[j] = min(v, lim.q_pos_hi[j])

    for j in range(lim.n_joints):
        if not lo[j] < hi[j]:
            raise ValueError(
                f"inverted position envelope at joint {j + 1}:"
                f" lo={lo[j]} >= hi={hi[j]}"
            )
        if not (math.isfinite(vel[j]) and math.isfinite(tau[j])):
            raise ValueError(f"non-finite bound at joint {j + 1}")

    return PhysicalEnvelope(
        q_pos_lo=tuple(lo),
        q_pos_hi=tuple(hi),
        q_vel_max=tuple(vel),
        tau_max=tuple(tau),
    )
