"""robocore.contracts -- canonical Pydantic data contracts for the robocore runtime.

These models are the *single source of truth* for the durable, cross-plane data
shapes exchanged between the Cognitive Plane (planner/critic), the
Orchestration & Training Plane (sim evaluation, SysID), and the 1 kHz Physical
Execution Plane (bridge, flight recorder).

Design rules enforced here:
  * Every persistent record is content-addressed or UUID-keyed.
  * Every record carries unix timestamps (float, UTC) for SQL indexing and
    ISO-8601 strings only where human readability matters.
  * Models are frozen where mutation would corrupt lineage.
  * ``harness_hash`` is computed by ONE canonical function
    (:func:`compute_harness_hash`) shared by the compiler and the ledger so
    the two can never disagree.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid as _uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "DifficultyStrataBin",
    "RolloutDomain",
    "HypothesisStatus",
    "BatchDecision",
    "FlightTrigger",
    "ResetOutcome",
    "EventType",
    "SysIDStatus",
    "SceneVerificationStatus",
    "PromotionDecisionKind",
    "utc_now",
    "unix_now",
    "iso_now",
    "unix_to_iso",
    "new_id",
    "canonical_json",
    "compute_harness_hash",
    "FalsifiablePrediction",
    "HypothesisProposal",
    "HypothesisRecord",
    "SafetyValidationReceipt",
    "HarnessManifest",
    "PolicyCheckpointRecord",
    "StratumStats",
    "OverallStats",
    "BatchStatistics",
    "SPRTTrace",
    "EvaluationBatchRecord",
    "RolloutTelemetryBundle",
    "ChampionManifest",
    "SimToRealGapReport",
    "BlackBoxSnapshotRecord",
    "LedgerEvent",
    "RejectedDiffSummary",
    "PlannerContext",
    "OperatorOverride",
    "BridgeHealthReport",
]


# ---------------------------------------------------------------------------
# Time / id helpers
# ---------------------------------------------------------------------------


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def unix_now() -> float:
    return time.time()


def iso_now() -> str:
    return utc_now().isoformat(timespec="microseconds")


def unix_to_iso(unix_ts: float) -> str:
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat(
        timespec="microseconds"
    )


def new_id(prefix: str) -> str:
    return f"{prefix}_{_uuid.uuid4().hex[:16]}"


def canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding: the ledger hash chain depends on it."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def compute_harness_hash(
    *,
    parent_harness_hash: Optional[str],
    harness_diff: str,
    ast_fingerprint: str,
    config: Dict[str, Any],
) -> str:
    """Content-addressed identity of a compiled harness.

    Parent hash + full diff text (constants included) + structural AST
    fingerprint + canonical config.  Two harnesses are byte-identical iff
    they share a hash; changing only a constant yields the same
    ``ast_fingerprint`` but a *different* ``harness_hash`` (this is what
    separates structural anti-looping from exact dedup).
    """
    payload = canonical_json(
        {
            "v": 1,
            "parent": parent_harness_hash or "",
            "diff": harness_diff,
            "ast": ast_fingerprint,
            "config": config,
        }
    )
    return "hsh_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DifficultyStrataBin(str, Enum):
    """Quantized task-difficulty strata used for stratified evaluation."""

    EASY_CENTER = "EASY_CENTER"
    MEDIUM_OBLIQUE = "MEDIUM_OBLIQUE"
    HARD_BOUNDARY = "HARD_BOUNDARY"
    OCCLUDED = "OCCLUDED"


class RolloutDomain(str, Enum):
    SIM = "SIM"
    PHYSICAL = "PHYSICAL"


class HypothesisStatus(str, Enum):
    PROPOSED = "PROPOSED"
    COMPILED = "COMPILED"
    EVALUATING = "EVALUATING"
    PROMOTED = "PROMOTED"
    REJECTED_COMPILER = "REJECTED_COMPILER"
    REJECTED_SIM = "REJECTED_SIM"
    REJECTED_PHYSICAL = "REJECTED_PHYSICAL"
    EARLY_ABORTED = "EARLY_ABORTED"
    SUPERSEDED = "SUPERSEDED"


class BatchDecision(str, Enum):
    CONTINUE = "CONTINUE"
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    EARLY_ABORT = "EARLY_ABORT"
    INCONCLUSIVE = "INCONCLUSIVE"


class FlightTrigger(str, Enum):
    """Events that must instantly seal the 1 kHz ring buffer to a black box."""

    FORCE_SPIKE = "FORCE_SPIKE"
    COLLISION = "COLLISION"
    INFERENCE_DEADLINE = "INFERENCE_DEADLINE"
    TASK_FAILURE = "TASK_FAILURE"
    MANUAL = "MANUAL"


class ResetOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    UNRECOVERABLE = "UNRECOVERABLE"


class SysIDStatus(str, Enum):
    APPLIED = "APPLIED"
    INSUFFICIENT_CHANGE = "INSUFFICIENT_CHANGE"
    FAILED_TO_CONVERGE = "FAILED_TO_CONVERGE"


class SceneVerificationStatus(str, Enum):
    PASS = "PASS"
    RETRYABLE_FAIL = "RETRYABLE_FAIL"
    FAIL = "FAIL"


class PromotionDecisionKind(str, Enum):
    PROMOTE = "PROMOTE"
    RETAIN_CHAMPION = "RETAIN_CHAMPION"
    INCONCLUSIVE = "INCONCLUSIVE"


class EventType(str, Enum):
    """Typed vocabulary for the append-only ``ledger_events`` chain."""

    CYCLE_START = "CYCLE_START"
    ORCHESTRATOR_STATE = "ORCHESTRATOR_STATE"
    CYCLE_COMMIT = "CYCLE_COMMIT"
    CYCLE_ABORT = "CYCLE_ABORT"
    CYCLE_SUPERSEDED = "CYCLE_SUPERSEDED"
    HYPOTHESIS_RECORDED = "HYPOTHESIS_RECORDED"
    HARNESS_COMPILED = "HARNESS_COMPILED"
    COMPILER_REJECTION = "COMPILER_REJECTION"
    BATCH_COMMITTED = "BATCH_COMMITTED"
    ROLLOUT_ANOMALY = "ROLLOUT_ANOMALY"
    CHAMPION_PROMOTED = "CHAMPION_PROMOTED"
    FLIGHT_DUMP = "FLIGHT_DUMP"
    SYSID_APPLIED = "SYSID_APPLIED"
    FAILURE_RECONSTRUCTED = "FAILURE_RECONSTRUCTED"
    OPERATOR_OVERRIDE = "OPERATOR_OVERRIDE"
    PAUSE_REQUESTED = "PAUSE_REQUESTED"
    RESUME_REQUESTED = "RESUME_REQUESTED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RECOVERY_PARKED = "RECOVERY_PARKED"
    PLANNER_STARVED = "PLANNER_STARVED"
    BRIDGE_DEADLINE_MISS = "BRIDGE_DEADLINE_MISS"


# ---------------------------------------------------------------------------
# Hypotheses / harnesses
# ---------------------------------------------------------------------------


class FalsifiablePrediction(BaseModel):
    """A prediction that can be scored against measured reality."""

    model_config = ConfigDict(frozen=True)

    target_metric: str  # e.g. "physical_success_rate"
    stratum: DifficultyStrataBin = DifficultyStrataBin.EASY_CENTER
    predicted_delta: float  # e.g. +0.08 vs current champion
    ci_low: float
    ci_high: float


class HypothesisProposal(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str = Field(default_factory=lambda: new_id("hyp"))
    task_name: str
    parent_harness_hash: str
    harness_diff: str  # unified diff of the harness source
    ast_fingerprint: str  # structural fingerprint of the mutated AST
    rationale: str
    falsifiable_prediction: FalsifiablePrediction
    param_summary: Dict[str, float] = Field(default_factory=dict)
    proposed_at_unix: float = Field(default_factory=unix_now)


class SafetyValidationReceipt(BaseModel):
    """Output of the AST/kinematic precheck sandbox."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    violations: List[str] = Field(default_factory=list)
    static_analysis_summary: str = ""
    compiler_version: str
    kinematic_precheck: bool = False


class HarnessManifest(BaseModel):
    """An immutable, content-addressed compiled harness."""

    model_config = ConfigDict(frozen=True)

    harness_hash: str
    task_name: str
    parent_harness_hash: Optional[str] = None
    harness_diff: str
    ast_fingerprint: str
    compiler_version: str
    safety_receipt: SafetyValidationReceipt
    difficulty_strata: List[DifficultyStrataBin] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at_unix: float = Field(default_factory=unix_now)

    @classmethod
    def genesis(
        cls,
        task_name: str,
        harness_diff: str,
        ast_fingerprint: str,
        compiler_version: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> "HarnessManifest":
        """Bootstrap harness (no parent). Must be recorded before cycle #1."""
        receipt = SafetyValidationReceipt(
            passed=True,
            violations=[],
            static_analysis_summary="genesis bootstrap",
            compiler_version=compiler_version,
            kinematic_precheck=True,
        )
        cfg = config or {}
        harness_hash = compute_harness_hash(
            parent_harness_hash=None,
            harness_diff=harness_diff,
            ast_fingerprint=ast_fingerprint,
            config=cfg,
        )
        return cls(
            harness_hash=harness_hash,
            task_name=task_name,
            parent_harness_hash=None,
            harness_diff=harness_diff,
            ast_fingerprint=ast_fingerprint,
            compiler_version=compiler_version,
            safety_receipt=receipt,
            difficulty_strata=[s for s in DifficultyStrataBin],
            config=cfg,
        )


class HypothesisRecord(BaseModel):
    """Read model of a hypothesis row (proposal + lifecycle state)."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    task_name: str
    parent_harness_hash: str
    harness_hash: Optional[str] = None
    harness_diff: str
    ast_fingerprint: str
    param_summary: Dict[str, float] = Field(default_factory=dict)
    rationale: str
    prediction: FalsifiablePrediction
    critic_score: float
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    rejection_reason: Optional[str] = None
    proposed_at_unix: float = 0.0
    decided_at_unix: Optional[float] = None


class PolicyCheckpointRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_hash: str
    harness_hash: str
    format: str = "safetensors"
    storage_path: str
    num_params: int
    created_at_unix: float = Field(default_factory=unix_now)


# ---------------------------------------------------------------------------
# Evaluation / telemetry
# ---------------------------------------------------------------------------


class OverallStats(BaseModel):
    """Batch-level aggregate stats (not tied to one difficulty stratum)."""

    model_config = ConfigDict(frozen=True)

    n: int
    successes: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    mean_ttc: Optional[float] = None
    mean_cumulative_jerk: Optional[float] = None
    mean_torque_penalty: Optional[float] = None


class StratumStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    stratum: DifficultyStrataBin
    n: int
    successes: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    mean_ttc: Optional[float] = None
    mean_cumulative_jerk: Optional[float] = None
    mean_torque_penalty: Optional[float] = None


class SPRTTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    p0: float
    p1: float
    lower_boundary: float
    upper_boundary: float
    final_llr: float
    stopped_early: bool
    n_samples_consumed: int


class BatchStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    overall: OverallStats
    per_stratum: Dict[DifficultyStrataBin, StratumStats] = Field(
        default_factory=dict
    )
    sprt: Optional[SPRTTrace] = None
    notes: List[str] = Field(default_factory=list)


class EvaluationBatchRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    batch_id: str = Field(default_factory=lambda: new_id("bat"))
    cycle_id: str
    hypothesis_id: str
    harness_hash: str
    champion_hash: Optional[str] = None
    domain: RolloutDomain
    task_name: str
    n_rollouts: int
    decision: BatchDecision
    statistics: BatchStatistics
    trigger: str = "CYCLE"
    started_at_unix: float = Field(default_factory=unix_now)
    committed_at_unix: Optional[float] = None


class RolloutTelemetryBundle(BaseModel):
    """One episode's digest as persisted in ``rollout_telemetry``."""

    model_config = ConfigDict(frozen=True)

    rollout_id: str = Field(default_factory=lambda: new_id("rol"))
    batch_id: str
    harness_hash: str
    task_name: str
    domain: RolloutDomain
    stratum: DifficultyStrataBin
    success: bool
    time_to_completion: Optional[float] = None
    cumulative_jerk: float = 0.0
    torque_penalty: float = 0.0
    inference_deadline_misses: int = 0
    force_spike_max: float = 0.0
    collision_count: int = 0
    reset_outcome: Optional[ResetOutcome] = None
    flight_snapshot_hash: Optional[str] = None
    t_start_unix: float = Field(default_factory=unix_now)
    t_end_unix: float = Field(default_factory=unix_now)
    video_uri: Optional[str] = None
    vlm_failure_summary: Optional[str] = None


class ChampionManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_name: str
    harness_hash: str
    policy_checkpoint_hash: str
    batch_id: str
    wilson_lower_bound: float
    stratum_stats: List[StratumStats] = Field(default_factory=list)
    promoted_at_unix: float = Field(default_factory=unix_now)


class SimToRealGapReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_name: str
    harness_hash: Optional[str] = None
    sim_metric: float  # recent sim success rate (or replay residual, see caller)
    real_metric: float  # recent physical success rate
    gap: float
    friction_offsets: Dict[str, float] = Field(default_factory=dict)
    damping_offsets: Dict[str, float] = Field(default_factory=dict)
    status: SysIDStatus = SysIDStatus.APPLIED
    snapshot_hash: Optional[str] = None
    residual_before: Optional[float] = None
    residual_after: Optional[float] = None
    calibrated_at_unix: float = Field(default_factory=unix_now)


class BlackBoxSnapshotRecord(BaseModel):
    """Persistent pointer at an immutable flight-recorder dump."""

    model_config = ConfigDict(frozen=True)

    snapshot_hash: str
    rollout_id: str
    trigger: FlightTrigger
    parquet_path: str
    sim_state_path: str
    format: str = "parquet"  # or "npz" fallback
    t_window_seconds: float
    sample_rate_hz: float
    num_samples: int
    mujoco_model_hash: str
    created_at_unix: float = Field(default_factory=unix_now)


class LedgerEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    event_type: str
    cycle_id: Optional[str] = None
    payload_json: str
    prev_hash: str
    event_hash: str
    recorded_at_unix: float
    recorded_at_iso: str

    def payload(self) -> Dict[str, Any]:
        return json.loads(self.payload_json)


class RejectedDiffSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    harness_diff: str
    ast_fingerprint: str
    param_summary: Dict[str, float] = Field(default_factory=dict)
    rejection_status: str
    rejection_reason: Optional[str] = None
    critic_score: float
    decided_at_unix: Optional[float] = None


class PlannerContext(BaseModel):
    """Ledger-derived context fed to the Cognitive Planner each cycle.

    Deliverable 4 (``cognitive/context_builder.py``) compiles the rich
    version; the orchestrator's default builder fills this same shape.
    """

    model_config = ConfigDict(frozen=True)

    task_name: str
    champion: Optional[ChampionManifest] = None
    champion_stratum_stats: List[StratumStats] = Field(default_factory=list)
    recent_failure_modes: List[Dict[str, Any]] = Field(default_factory=list)
    sim_to_real_warnings: List[SimToRealGapReport] = Field(default_factory=list)
    rejected_diff_patterns: List[RejectedDiffSummary] = Field(default_factory=list)
    recent_batches: List[EvaluationBatchRecord] = Field(default_factory=list)


class OperatorOverride(BaseModel):
    """Signed human intervention record; never mutates experimental lineage
    silently -- it is an explicit, hash-chained event."""

    model_config = ConfigDict(frozen=True)

    operator: str
    action: str  # e.g. "PAUSE", "RESUME", "FORCE_PROMOTE", "ESTOP_RESET"
    justification: str
    signature: str  # ed25519 hex signature over canonical_json(payload) in prod
    created_at_unix: float = Field(default_factory=unix_now)


class BridgeHealthReport(BaseModel):
    """Snapshot of the 1 kHz shared-memory bridge watchdog."""

    model_config = ConfigDict(frozen=True)

    consecutive_deadline_misses: int = 0
    total_misses: int = 0
    watchdog_active: bool = True
    control_mode: str = "ACTIVE"  # ACTIVE | HOLD | PARKED | ESTOP

    @property
    def ok(self) -> bool:
        return self.watchdog_active and self.consecutive_deadline_misses == 0
