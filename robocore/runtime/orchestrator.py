"""robocore.runtime.orchestrator -- the unattended closed-loop state machine.

Deliverable 1 of the robocore Phase-1 runtime.

Determinism model
-----------------
The orchestrator is a single-threaded macro-cycle state machine (the 1 kHz
real-time plane runs independently behind the bridge supervisor and fails
closed on its own watchdog).  Every state transition:

1. is validated against an explicit :data:`ALLOWED_TRANSITIONS` table
   (anything not in the table raises :class:`InvalidTransitionError`), and
2. is journaled to the append-only ``ledger_events`` chain *before* the
   in-memory state flips (write-ahead intent), so a crash between journal
   and effect can only ever over-estimate the danger zone -- which is
   exactly the fail-closed direction we want.

Partial-failure recovery paths implemented here:

* compiler rejection -> instant feedback loop to the Planner; sim and
  hardware are never touched (transition COMPILING -> PROPOSING);
* SPRT early-abort mid sim/physical batch -> EARLY_ABORTED -> commit the
  partial telemetry (physical rollouts already spent are still evidence)
  -> back to PROPOSING;
* autonomous reset retry exhaustion (``retry_budget == 2`` -> 3 attempts)
  -> deterministic hold + graceful park -> PAUSED_ALERT;
* 1 kHz bridge deadline misses: single miss is journaled telemetry;
  ``max_bridge_deadline_misses`` consecutive misses -> hold + park +
  PAUSED_ALERT (the RT watchdog itself has already failed closed by then);
* crash recovery on restart: verify the event chain, read the last
  uncommitted ORCHESTRATOR_STATE from SQLite WAL, and either resume clean
  (sim-side crash) or park the arm and demand operator scene verification
  (hardware-side crash).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from robocore.contracts import (
    BatchDecision,
    BatchStatistics,
    BridgeHealthReport,
    ChampionManifest,
    EvaluationBatchRecord,
    EventType,
    HarnessManifest,
    HypothesisProposal,
    HypothesisStatus,
    OperatorOverride,
    PlannerContext,
    PolicyCheckpointRecord,
    PromotionDecisionKind,
    ResetOutcome,
    RolloutDomain,
    RolloutTelemetryBundle,
    SafetyValidationReceipt,
    SceneVerificationStatus,
    SimToRealGapReport,
    StratumStats,
    SysIDStatus,
    new_id,
    unix_now,
)
from robocore.stats import (
    PromotionEvaluation,
    PromotionGate,
    PromotionGateConfig,
)
from robocore.storage.ledger_repo import (
    DuplicateHypothesisError,
    ExperimentLedgerRepository,
    LedgerError,
    NoActiveChampionError,
)
from robocore.telemetry.flight_recorder import EventEmitter, FlightRecorder

__all__ = [
    "OrchestratorState",
    "ALLOWED_TRANSITIONS",
    "OrchestratorConfig",
    "OrchestratorPorts",
    "RobocoreOrchestrator",
    "InvalidTransitionError",
    "EmergencyStopSignal",
    "CrashRecoveryReport",
]

_LOG = logging.getLogger("robocore.orchestrator")


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class OrchestratorState(str, Enum):
    IDLE = "IDLE"
    PROPOSING = "PROPOSING"
    COMPILING_AND_PRECHECKING = "COMPILING_AND_PRECHECKING"
    SIM_EVALUATING = "SIM_EVALUATING"
    SYSID_CALIBRATING = "SYSID_CALIBRATING"
    PHYSICAL_SCENE_VERIFYING = "PHYSICAL_SCENE_VERIFYING"
    PHYSICAL_EXECUTING = "PHYSICAL_EXECUTING"
    PHYSICAL_RESETTING = "PHYSICAL_RESETTING"
    EVALUATING_PROMOTION = "EVALUATING_PROMOTION"
    COMMITTING_LEDGER = "COMMITTING_LEDGER"
    EARLY_ABORTED = "EARLY_ABORTED"
    PAUSED_ALERT = "PAUSED_ALERT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


#: The one and only transition authority.  EMERGENCY_STOP is intentionally
#: NOT in this table: it is only reachable through the forced e-stop path.
ALLOWED_TRANSITIONS: Dict[OrchestratorState, frozenset] = {
    OrchestratorState.IDLE: frozenset({OrchestratorState.PROPOSING}),
    OrchestratorState.PROPOSING: frozenset(
        {
            OrchestratorState.COMPILING_AND_PRECHECKING,
            OrchestratorState.IDLE,  # planner starvation -> cooldown
        }
    ),
    OrchestratorState.COMPILING_AND_PRECHECKING: frozenset(
        {
            OrchestratorState.SIM_EVALUATING,
            OrchestratorState.PROPOSING,  # compiler rejection fast-feedback
        }
    ),
    OrchestratorState.SIM_EVALUATING: frozenset(
        {
            OrchestratorState.SYSID_CALIBRATING,
            OrchestratorState.EARLY_ABORTED,  # SPRT abort or sim-gate fail
        }
    ),
    OrchestratorState.SYSID_CALIBRATING: frozenset(
        {
            OrchestratorState.SIM_EVALUATING,  # recalibrated -> re-gate in sim
            OrchestratorState.PHYSICAL_SCENE_VERIFYING,
            OrchestratorState.EARLY_ABORTED,  # calibration divergent
        }
    ),
    OrchestratorState.PHYSICAL_SCENE_VERIFYING: frozenset(
        {OrchestratorState.PHYSICAL_EXECUTING}
    ),
    OrchestratorState.PHYSICAL_EXECUTING: frozenset(
        {
            OrchestratorState.PHYSICAL_RESETTING,
            OrchestratorState.EARLY_ABORTED,  # SPRT abort mid physical batch
        }
    ),
    OrchestratorState.PHYSICAL_RESETTING: frozenset(
        {OrchestratorState.EVALUATING_PROMOTION}
    ),
    OrchestratorState.EVALUATING_PROMOTION: frozenset(
        {
            OrchestratorState.PHYSICAL_EXECUTING,  # inconclusive -> more samples
            OrchestratorState.COMMITTING_LEDGER,
        }
    ),
    OrchestratorState.COMMITTING_LEDGER: frozenset(
        {OrchestratorState.PROPOSING}
    ),
    OrchestratorState.EARLY_ABORTED: frozenset(
        {OrchestratorState.COMMITTING_LEDGER}
    ),
    OrchestratorState.PAUSED_ALERT: frozenset(
        {
            OrchestratorState.PROPOSING,  # operator resume (+scene verify)
            OrchestratorState.IDLE,  # operator chose to stop
        }
    ),
    OrchestratorState.EMERGENCY_STOP: frozenset(
        {OrchestratorState.IDLE}  # only via operator estop-reset
    ),
}

#: States during which the arm may be moving or the cell is "hot".  A crash
#: journal showing one of these states forces park-and-alert on restart.
HARDWARE_TOUCHING_STATES: frozenset = frozenset(
    {
        OrchestratorState.PHYSICAL_SCENE_VERIFYING,
        OrchestratorState.PHYSICAL_EXECUTING,
        OrchestratorState.PHYSICAL_RESETTING,
    }
)


class InvalidTransitionError(LedgerError):
    """A transition not present in ALLOWED_TRANSITIONS was attempted."""


class EmergencyStopSignal(Exception):
    """Raised at guard points when an e-stop has been requested."""


class PauseRequestedError(Exception):
    """Raised at guard points when an operator pause has been requested."""


class BridgeDeadlineMissedError(Exception):
    """Consecutive 1 kHz bridge deadline misses exceeded the budget."""

    def __init__(self, health: BridgeHealthReport) -> None:
        super().__init__(
            f"bridge deadline misses: consecutive="
            f"{health.consecutive_deadline_misses} total={health.total_misses}"
        )
        self.health = health


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrchestratorConfig:
    task_name: str
    # -- planner / critic ------------------------------------------------
    max_proposal_attempts_per_cycle: int = 5
    max_starved_cycles: int = 3
    planner_cooldown_s: float = 30.0
    # -- sim gate ----------------------------------------------------------
    sim_batch_size: int = 32
    genesis_sim_baseline: float = 0.35
    sim_min_improvement: float = 0.05
    # -- sysid ---------------------------------------------------------------
    max_sysid_recalibration_loops: int = 1
    sysid_history_window: int = 5
    # -- physical plane --------------------------------------------------------
    scene_verify_retries: int = 1
    physical_batch_size: int = 16
    max_physical_rollouts_per_cycle: int = 48
    reset_retry_budget: int = 2  # 1 initial + 2 retries = 3 attempts
    max_bridge_deadline_misses: int = 3  # consecutive, at rollout granularity
    # -- promotion ---------------------------------------------------------
    promotion_gate: PromotionGateConfig = field(
        default_factory=PromotionGateConfig
    )
    # -- loop mechanics -------------------------------------------------------
    pause_poll_interval_s: float = 0.5


# ---------------------------------------------------------------------------
# Port protocols (subsystem interfaces; deliverable 5 supplies fakes)
# ---------------------------------------------------------------------------


class IPlanner(Protocol):
    def propose(self, context: PlannerContext) -> Optional[HypothesisProposal]: ...

    def ingest_feedback(self, feedback: "PlannerFeedback") -> None: ...


@dataclass(frozen=True)
class CriticVerdict:
    accept: bool
    score: float
    reason: str


class ICriticGuard(Protocol):
    def score(
        self, proposal: HypothesisProposal, context: PlannerContext
    ) -> CriticVerdict: ...


@dataclass(frozen=True)
class CompileResult:
    passed: bool
    receipt: SafetyValidationReceipt
    manifest: Optional[HarnessManifest] = None
    checkpoint: Optional[PolicyCheckpointRecord] = None
    rejection_reason: Optional[str] = None


class IHarnessCompiler(Protocol):
    def compile_and_precheck(
        self, proposal: HypothesisProposal, parent: HarnessManifest
    ) -> CompileResult: ...


class SimGateDecision(str, Enum):
    PASSED_GATE = "PASSED_GATE"
    FAILED_GATE = "FAILED_GATE"
    EARLY_ABORTED = "EARLY_ABORTED"


@dataclass
class SimEvaluationReport:
    decision: SimGateDecision
    rollouts: List[RolloutTelemetryBundle]
    statistics: BatchStatistics
    reward_hacking_suspected: bool = False
    notes: List[str] = field(default_factory=list)


class ISimEvaluator(Protocol):
    def evaluate(
        self,
        *,
        harness: HarnessManifest,
        champion_baseline: Optional[float],
        batch_size: int,
        env_overrides: Optional[Dict[str, Any]],
    ) -> SimEvaluationReport: ...


@dataclass
class SysIDResult:
    status: SysIDStatus
    params_changed: bool
    env_overrides: Dict[str, Any] = field(default_factory=dict)
    friction_offsets: Dict[str, float] = field(default_factory=dict)
    damping_offsets: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


class ISysIDCalibrator(Protocol):
    def calibrate(
        self, gap_history: Sequence[SimToRealGapReport]
    ) -> SysIDResult: ...


@dataclass(frozen=True)
class SceneVerification:
    status: SceneVerificationStatus
    details: str = ""
    vlm_summary: str = ""


class ISceneVerifier(Protocol):
    def verify(self) -> SceneVerification: ...


@dataclass
class PhysicalBatchResult:
    rollouts: List[RolloutTelemetryBundle]
    decision: BatchDecision  # CONTINUE | EARLY_ABORT
    statistics: BatchStatistics
    early_abort_reason: Optional[str] = None
    started_at_unix: float = 0.0


class IPhysicalExecutor(Protocol):
    def execute_batch(
        self,
        *,
        harness: HarnessManifest,
        champion: Optional[ChampionManifest],
        batch_size: int,
        recorder: Optional[FlightRecorder],
        on_rollout: Callable[[RolloutTelemetryBundle], None],
    ) -> PhysicalBatchResult:
        """Execute up to ``batch_size`` physical rollouts.

        Contract: ``on_rollout`` is called exactly once per completed
        rollout and MAY raise ``BridgeDeadlineMissedError``.  If it does,
        the executor MUST stop commanding motion immediately, keep the
        already-completed rollouts, and return a partial
        ``PhysicalBatchResult`` with ``decision=EARLY_ABORT`` so telemetry
        is never lost.  The orchestrator performs the park-and-alert.
        """
        ...


class IResetController(Protocol):
    def reset_scene(self, attempt: int) -> Tuple[ResetOutcome, str]: ...


class IBridgeSupervisor(Protocol):
    def health(self) -> BridgeHealthReport: ...

    def hold(self) -> None: ...

    def park(self) -> None: ...

    def estop(self) -> None: ...


class IStatisticalEvaluator(Protocol):
    def evaluate(
        self,
        rollouts: Sequence[RolloutTelemetryBundle],
        champion: Optional[ChampionManifest],
    ) -> PromotionEvaluation: ...


class IContextBuilder(Protocol):
    def build(self, task_name: str) -> PlannerContext: ...


@dataclass(frozen=True)
class PlannerFeedback:
    kind: str  # COMPILER_REJECTION | DUPLICATE | SIM_GATE_FAILED | EARLY_ABORT | PROMOTED | RETAINED | SYSID_DIVERGENT | CRITIC_REJECTED | STARVED
    hypothesis_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorPorts:
    """Every subsystem the orchestrator wires together, injected explicitly."""

    planner: IPlanner
    critic: ICriticGuard
    compiler: IHarnessCompiler
    sim_evaluator: ISimEvaluator
    sysid: ISysIDCalibrator
    scene_verifier: ISceneVerifier
    physical_executor: IPhysicalExecutor
    reset_controller: IResetController
    bridge: IBridgeSupervisor
    promotion_evaluator: IStatisticalEvaluator
    flight_recorder: Optional[FlightRecorder] = None
    context_builder: Optional[IContextBuilder] = None


# ---------------------------------------------------------------------------
# Default adapters
# ---------------------------------------------------------------------------


class PromotionGateEvaluator:
    """IStatisticalEvaluator backed by the stratified PromotionGate."""

    def __init__(self, config: Optional[PromotionGateConfig] = None) -> None:
        self._gate = PromotionGate(config)

    def evaluate(
        self,
        rollouts: Sequence[RolloutTelemetryBundle],
        champion: Optional[ChampionManifest],
    ) -> PromotionEvaluation:
        return self._gate.evaluate(rollouts, champion)


class LedgerContextBuilder:
    """Default IContextBuilder: everything the planner needs from the ledger.

    Deliverable 4 (``cognitive/context_builder.py``) replaces this with the
    full compiler (VLM keyframe diagnoses, negative constraints with
    parameter ranges, drift warnings); the interface is identical, so the
    orchestrator does not change.
    """

    def __init__(self, ledger: ExperimentLedgerRepository) -> None:
        self._ledger = ledger

    def build(self, task_name: str) -> PlannerContext:
        champion: Optional[ChampionManifest] = None
        try:
            champion = self._ledger.get_active_champion(task_name)
        except NoActiveChampionError:
            champion = None
        champion_stats: List[StratumStats] = []
        if champion is not None:
            champion_stats = self._ledger.get_stratum_stats(
                champion.harness_hash, RolloutDomain.PHYSICAL
            )
        parent_hash = champion.harness_hash if champion is not None else ""
        rejected = (
            self._ledger.query_similar_rejected_diffs(parent_hash, "", top_k=5)
            if parent_hash
            else []
        )
        return PlannerContext(
            task_name=task_name,
            champion=champion,
            champion_stratum_stats=champion_stats,
            recent_failure_modes=self._ledger.get_failure_mode_pareto(
                task_name, limit=8
            ),
            sim_to_real_warnings=self._ledger.get_sim_to_real_gap_history(
                task_name, window=5
            ),
            rejected_diff_patterns=rejected,
            recent_batches=self._ledger.get_recent_batches(task_name, limit=5),
        )


# ---------------------------------------------------------------------------
# Cycle context
# ---------------------------------------------------------------------------


@dataclass
class CycleContext:
    cycle_id: str
    task_name: str
    started_at_unix: float
    champion: Optional[ChampionManifest] = None
    proposal: Optional[HypothesisProposal] = None
    hypothesis_id: Optional[str] = None
    harness: Optional[HarnessManifest] = None
    checkpoint: Optional[PolicyCheckpointRecord] = None
    sim_batch_ids: List[str] = field(default_factory=list)
    sysid_loops: int = 0
    sysid_env_overrides: Optional[Dict[str, Any]] = None
    physical_rollouts: List[RolloutTelemetryBundle] = field(default_factory=list)
    physical_batch_ids: List[str] = field(default_factory=list)
    promotion: Optional[PromotionEvaluation] = None
    new_champion: Optional[ChampionManifest] = None
    final_status: Optional[HypothesisStatus] = None
    final_reason: str = ""


@dataclass(frozen=True)
class CrashRecoveryReport:
    action: str  # FRESH_START | RESUMED_CLEAN | PARKED_ARM | REFUSED_CHAIN_CORRUPT
    details: str
    open_cycles: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class RobocoreOrchestrator:
    """Unattended closed-loop runtime wiring every robocore subsystem.

    Threading model: ``run()`` owns the cycle loop on one thread.
    ``emergency_stop()`` / ``pause()`` / ``resume()`` are safe from any
    thread (CLI, signal handler); they flip flags that guard points inside
    the loop honor deterministically.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        ports: OrchestratorPorts,
        ledger: ExperimentLedgerRepository,
        *,
        emitter: Optional[EventEmitter] = None,
    ) -> None:
        self._config = config
        self._ports = ports
        self._ledger = ledger
        self._emitter = emitter or EventEmitter(also_stderr=False)
        if ports.context_builder is None:
            self._context_builder: IContextBuilder = LedgerContextBuilder(ledger)
        else:
            self._context_builder = ports.context_builder
        self._state = OrchestratorState.IDLE
        self._state_lock = threading.RLock()
        self._running = False
        self._estop_requested = False
        self._pause_requested = False
        self._resume_requested = False
        self._pause_reason = ""
        self._starved_cycles = 0
        self._ctx: Optional[CycleContext] = None

    # -- introspection ---------------------------------------------------------

    @property
    def state(self) -> OrchestratorState:
        return self._state

    @property
    def pause_reason(self) -> str:
        return self._pause_reason

    @property
    def current_cycle_id(self) -> Optional[str]:
        return self._ctx.cycle_id if self._ctx is not None else None

    # =====================================================================
    # Transition machinery
    # =====================================================================

    def _transition(
        self,
        next_state: OrchestratorState,
        reason: str,
        *,
        cycle_id: Optional[str] = None,
        force: bool = False,
    ) -> None:
        with self._state_lock:
            prev = self._state
            if not force and next_state not in ALLOWED_TRANSITIONS[prev]:
                raise InvalidTransitionError(
                    f"illegal transition {prev.value} -> {next_state.value}"
                    f" (reason={reason!r})"
                )
            # write-ahead intent: journal BEFORE the in-memory flip
            self._ledger.append_event(
                EventType.ORCHESTRATOR_STATE,
                {
                    "state": next_state.value,
                    "prev": prev.value,
                    "reason": reason,
                },
                cycle_id=cycle_id,
            )
            self._state = next_state
        _LOG.info("transition %s -> %s (%s)", prev.value, next_state.value, reason)
        self._emitter.emit(
            "ORCHESTRATOR_STATE",
            {"state": next_state.value, "prev": prev.value, "reason": reason},
            cycle_id=cycle_id,
        )

    def _force_estop_transition(self, reason: str) -> None:
        self._transition(
            OrchestratorState.EMERGENCY_STOP,
            reason,
            cycle_id=self.current_cycle_id,
            force=True,
        )

    def _guard(self) -> None:
        if self._estop_requested:
            raise EmergencyStopSignal()
        if (
            self._pause_requested
            and self._state not in (OrchestratorState.PAUSED_ALERT,)
            and self._state is not OrchestratorState.EMERGENCY_STOP
        ):
            raise PauseRequestedError(self._pause_reason)

    # =====================================================================
    # Crash recovery (called once at startup)
    # =====================================================================

    def _recover_on_startup(self) -> CrashRecoveryReport:
        # 1. Refuse to operate on a tampered ledger.
        ok, bad_seq = self._ledger.verify_chain_integrity()
        if not ok:
            self._safe_park("LEDGER_CHAIN_CORRUPT")
            self._pause_reason = f"LEDGER_CHAIN_CORRUPT (first bad seq={bad_seq})"
            self._transition(
                OrchestratorState.PAUSED_ALERT,
                self._pause_reason,
                force=True,
            )
            return CrashRecoveryReport(
                action="REFUSED_CHAIN_CORRUPT",
                details=self._pause_reason,
                open_cycles=tuple(self._ledger.get_open_cycles()),
            )

        last_state_ev = self._ledger.get_latest_orchestrator_state()
        open_cycles = self._ledger.get_open_cycles()

        if last_state_ev is None or not open_cycles:
            # clean start (or every cycle was committed before shutdown)
            self._supersede_open_hypotheses()
            self._state = OrchestratorState.IDLE
            return CrashRecoveryReport("FRESH_START", "no open cycles", ())

        last_state = OrchestratorState(last_state_ev.payload()["state"])
        if last_state in HARDWARE_TOUCHING_STATES:
            # 2. Arm state unknown: fail closed. Park and demand operator
            #    scene verification before anything else moves.
            self._safe_park("CRASH_RECOVERY_UNVERIFIED_ARM")
            self._pause_reason = (
                f"CRASH_RECOVERY_UNVERIFIED_ARM: last journaled state="
                f"{last_state.value} in open cycle {open_cycles[0]}"
            )
            self._transition(
                OrchestratorState.PAUSED_ALERT, self._pause_reason, force=True
            )
            return CrashRecoveryReport(
                action="PARKED_ARM", details=self._pause_reason,
                open_cycles=tuple(open_cycles),
            )

        # 3. Crashed on the sim side: supersede the orphaned cycle cleanly
        #    and start fresh -- no hardware was involved.
        for cycle_id in open_cycles:
            self._ledger.append_event(
                EventType.CYCLE_SUPERSEDED,
                {"reason": "crash-recovery: sim-side crash"},
                cycle_id=cycle_id,
            )
        self._supersede_open_hypotheses()
        self._state = OrchestratorState.IDLE
        return CrashRecoveryReport(
            action="RESUMED_CLEAN",
            details=f"superseded {len(open_cycles)} sim-side cycle(s)",
            open_cycles=tuple(open_cycles),
        )

    def _supersede_open_hypotheses(self) -> None:
        for hyp in self._ledger.get_open_hypotheses(self._config.task_name):
            self._ledger.update_hypothesis_status(
                hyp.hypothesis_id,
                HypothesisStatus.SUPERSEDED,
                rejection_reason="cycle superseded by crash recovery",
            )

    def _safe_park(self, reason: str) -> None:
        """Best-effort deterministic hold + graceful park of the arm."""
        try:
            self._ports.bridge.hold()
            self._ports.bridge.park()
            self._ledger.append_event(
                EventType.RECOVERY_PARKED, {"reason": reason}
            )
        except Exception:  # noqa: BLE001 - parking is best-effort by design
            _LOG.exception("safe_park failed (bridge may be unreachable)")

    # =====================================================================
    # Main loop
    # =====================================================================

    def run(self) -> CrashRecoveryReport:
        """Run the unattended loop until pause / estop / starvation."""
        recovery = self._recover_on_startup()
        if recovery.action == "REFUSED_CHAIN_CORRUPT":
            return recovery
        self._running = True
        while self._running:
            if self._estop_requested or (
                self._state is OrchestratorState.EMERGENCY_STOP
            ):
                break
            if self._state is OrchestratorState.PAUSED_ALERT:
                self._wait_for_resume()
                continue
            if self._state is OrchestratorState.IDLE:
                if self._starved_cycles >= self._config.max_starved_cycles:
                    self._ledger.append_event(
                        EventType.PLANNER_STARVED,
                        {"consecutive_starved_cycles": self._starved_cycles},
                    )
                    _LOG.warning(
                        "planner starved %d cycles; parking runtime in IDLE",
                        self._starved_cycles,
                    )
                    break
                time.sleep(self._config.planner_cooldown_s)
                self._transition(OrchestratorState.PROPOSING, "cycle-start")
                continue
            try:
                self._dispatch()
            except EmergencyStopSignal:
                self._handle_emergency_stop()
                break
            except BridgeDeadlineMissedError as exc:
                self._safe_park("BRIDGE_DEADLINE_MISS_EXCEEDED")
                self._pause_reason = f"BRIDGE_DEADLINE_MISS_EXCEEDED: {exc}"
                self._transition(
                    OrchestratorState.PAUSED_ALERT,
                    self._pause_reason,
                    cycle_id=self.current_cycle_id,
                    force=True,
                )
            except PauseRequestedError as exc:
                self._safe_park("OPERATOR_PAUSE")
                self._pause_reason = f"OPERATOR_PAUSE: {exc}"
                self._transition(
                    OrchestratorState.PAUSED_ALERT,
                    self._pause_reason,
                    cycle_id=self.current_cycle_id,
                    force=True,
                )
        return recovery

    def _dispatch(self) -> None:
        handler = {
            OrchestratorState.PROPOSING: self._handle_proposing,
            OrchestratorState.COMPILING_AND_PRECHECKING: self._handle_compiling,
            OrchestratorState.SIM_EVALUATING: self._handle_sim_evaluating,
            OrchestratorState.SYSID_CALIBRATING: self._handle_sysid,
            OrchestratorState.PHYSICAL_SCENE_VERIFYING: self._handle_scene_verify,
            OrchestratorState.PHYSICAL_EXECUTING: self._handle_physical_executing,
            OrchestratorState.PHYSICAL_RESETTING: self._handle_resetting,
            OrchestratorState.EVALUATING_PROMOTION: self._handle_evaluating_promotion,
            OrchestratorState.COMMITTING_LEDGER: self._handle_committing,
            OrchestratorState.EARLY_ABORTED: self._handle_early_aborted,
        }.get(self._state)
        if handler is None:
            raise InvalidTransitionError(
                f"no handler for state {self._state.value}"
            )
        assert self._ctx is not None or self._state in (
            OrchestratorState.IDLE,
            OrchestratorState.PROPOSING,
        )
        handler()

    # =====================================================================
    # State handlers
    # =====================================================================

    def _begin_cycle(self) -> CycleContext:
        cycle_id = new_id("cyc")
        champion: Optional[ChampionManifest]
        try:
            champion = self._ledger.get_active_champion(self._config.task_name)
        except NoActiveChampionError:
            champion = None
        self._ledger.append_event(
            EventType.CYCLE_START,
            {
                "task_name": self._config.task_name,
                "champion": champion.harness_hash if champion else None,
            },
            cycle_id=cycle_id,
        )
        ctx = CycleContext(
            cycle_id=cycle_id,
            task_name=self._config.task_name,
            started_at_unix=unix_now(),
            champion=champion,
        )
        self._ctx = ctx
        return ctx

    def _handle_proposing(self) -> None:
        ctx = self._ctx or self._begin_cycle()
        self._guard()
        for attempt in range(self._config.max_proposal_attempts_per_cycle):
            self._guard()
            context = self._context_builder.build(ctx.task_name)
            proposal = self._ports.planner.propose(context)
            if proposal is None:
                self._starved_cycles += 1
                self._ledger.append_event(
                    EventType.PLANNER_STARVED,
                    {
                        "cycle_id": ctx.cycle_id,
                        "attempt": attempt,
                        "starved_cycles": self._starved_cycles,
                    },
                    cycle_id=ctx.cycle_id,
                )
                self._ports.planner.ingest_feedback(
                    PlannerFeedback(kind="STARVED", details={"attempt": attempt})
                )
                self._transition(
                    OrchestratorState.IDLE,
                    "planner-starved",
                    cycle_id=ctx.cycle_id,
                )
                self._ledger.append_event(
                    EventType.CYCLE_ABORT,
                    {"reason": "PLANNER_STARVED"},
                    cycle_id=ctx.cycle_id,
                )
                self._ctx = None
                return
            verdict = self._ports.critic.score(proposal, context)
            if not verdict.accept:
                self._ports.planner.ingest_feedback(
                    PlannerFeedback(
                        kind="CRITIC_REJECTED",
                        hypothesis_id=proposal.hypothesis_id,
                        details={"reason": verdict.reason, "score": verdict.score},
                    )
                )
                continue
            try:
                self._ledger.record_hypothesis(proposal, verdict.score)
            except DuplicateHypothesisError:
                self._ports.planner.ingest_feedback(
                    PlannerFeedback(
                        kind="DUPLICATE",
                        hypothesis_id=proposal.hypothesis_id,
                        details={"parent": proposal.parent_harness_hash},
                    )
                )
                continue
            ctx.proposal = proposal
            ctx.hypothesis_id = proposal.hypothesis_id
            self._transition(
                OrchestratorState.COMPILING_AND_PRECHECKING,
                f"accepted-{proposal.hypothesis_id}",
                cycle_id=ctx.cycle_id,
            )
            return
        # every attempt rejected by the critic: this cycle is a no-op
        self._starved_cycles += 1
        self._ports.planner.ingest_feedback(
            PlannerFeedback(
                kind="CRITIC_REJECTED",
                details={
                    "exhausted_proposal_attempts":
                        self._config.max_proposal_attempts_per_cycle
                },
            )
        )
        self._transition(
            OrchestratorState.IDLE, "critic-exhausted", cycle_id=ctx.cycle_id
        )
        self._ledger.append_event(
            EventType.CYCLE_ABORT,
            {"reason": "CRITIC_EXHAUSTED"},
            cycle_id=ctx.cycle_id,
        )
        self._ctx = None

    def _handle_compiling(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        proposal = ctx.proposal
        assert proposal is not None
        try:
            parent = self._ledger.get_harness(proposal.parent_harness_hash)
        except LedgerError as exc:
            self._ports.planner.ingest_feedback(
                PlannerFeedback(
                    kind="CRITIC_REJECTED",
                    hypothesis_id=proposal.hypothesis_id,
                    details={"reason": f"unknown parent harness: {exc}"},
                )
            )
            self._ledger.update_hypothesis_status(
                ctx.hypothesis_id,
                HypothesisStatus.REJECTED_COMPILER,
                rejection_reason=f"unknown parent harness: {exc}",
            )
            self._transition(
                OrchestratorState.PROPOSING,
                "compiler-rejection",
                cycle_id=ctx.cycle_id,
            )
            return
        result = self._ports.compiler.compile_and_precheck(proposal, parent)
        if not result.passed:
            # Fast feedback loop: planner learns the exact violation; sim
            # and hardware are never touched on this path.
            reason = result.rejection_reason or ";".join(result.receipt.violations)
            self._ledger.update_hypothesis_status(
                ctx.hypothesis_id,
                HypothesisStatus.REJECTED_COMPILER,
                rejection_reason=reason,
            )
            self._ledger.append_event(
                EventType.COMPILER_REJECTION,
                {
                    "hypothesis_id": ctx.hypothesis_id,
                    "violations": result.receipt.violations,
                    "reason": reason,
                },
                cycle_id=ctx.cycle_id,
            )
            self._ports.planner.ingest_feedback(
                PlannerFeedback(
                    kind="COMPILER_REJECTION",
                    hypothesis_id=ctx.hypothesis_id,
                    details={
                        "violations": result.receipt.violations,
                        "reason": reason,
                    },
                )
            )
            self._transition(
                OrchestratorState.PROPOSING,
                "compiler-rejection",
                cycle_id=ctx.cycle_id,
            )
            return
        assert result.manifest is not None and result.checkpoint is not None
        self._ledger.upsert_harness(result.manifest)
        self._ledger.record_policy_checkpoint(result.checkpoint)
        self._ledger.update_hypothesis_status(
            ctx.hypothesis_id,
            HypothesisStatus.COMPILED,
            harness_hash=result.manifest.harness_hash,
        )
        self._ledger.append_event(
            EventType.HARNESS_COMPILED,
            {
                "hypothesis_id": ctx.hypothesis_id,
                "harness_hash": result.manifest.harness_hash,
                "parent": result.manifest.parent_harness_hash,
                "ast_fingerprint": result.manifest.ast_fingerprint,
            },
            cycle_id=ctx.cycle_id,
        )
        ctx.harness = result.manifest
        ctx.checkpoint = result.checkpoint
        self._transition(
            OrchestratorState.SIM_EVALUATING,
            f"compiled-{result.manifest.harness_hash[:16]}",
            cycle_id=ctx.cycle_id,
        )

    def _handle_sim_evaluating(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        assert ctx.harness is not None and ctx.hypothesis_id is not None
        baseline = (
            ctx.champion.wilson_lower_bound
            if ctx.champion is not None
            else self._config.genesis_sim_baseline
        )
        report = self._ports.sim_evaluator.evaluate(
            harness=ctx.harness,
            champion_baseline=baseline,
            batch_size=self._config.sim_batch_size,
            env_overrides=ctx.sysid_env_overrides,
        )
        decision_map = {
            SimGateDecision.PASSED_GATE: BatchDecision.CONTINUE,
            SimGateDecision.FAILED_GATE: BatchDecision.REJECT,
            SimGateDecision.EARLY_ABORTED: BatchDecision.EARLY_ABORT,
        }
        batch = EvaluationBatchRecordFactory.sim_batch(
            ctx, report, decision_map[report.decision]
        )
        self._ledger.record_evaluation_batch(batch, report.rollouts)
        ctx.sim_batch_ids.append(batch.batch_id)
        if report.decision is SimGateDecision.PASSED_GATE:
            self._transition(
                OrchestratorState.SYSID_CALIBRATING,
                "sim-gate-passed",
                cycle_id=ctx.cycle_id,
            )
            return
        # FAILED_GATE or SPRT EARLY_ABORTED: kill before touching hardware
        reason = (
            "sim-gate-failed"
            if report.decision is SimGateDecision.FAILED_GATE
            else "sprt-early-abort-sim"
        )
        if report.reward_hacking_suspected:
            reason += "+reward-hacking-suspected"
        ctx.final_status = HypothesisStatus.REJECTED_SIM
        ctx.final_reason = reason
        self._ports.planner.ingest_feedback(
            PlannerFeedback(
                kind="SIM_GATE_FAILED"
                if report.decision is SimGateDecision.FAILED_GATE
                else "EARLY_ABORT",
                hypothesis_id=ctx.hypothesis_id,
                details={
                    "reason": reason,
                    "notes": report.notes,
                    "reward_hacking_suspected": report.reward_hacking_suspected,
                },
            )
        )
        self._transition(
            OrchestratorState.EARLY_ABORTED, reason, cycle_id=ctx.cycle_id
        )

    def _handle_sysid(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        gap_history = self._ledger.get_sim_to_real_gap_history(
            ctx.task_name, window=self._config.sysid_history_window
        )
        result = self._ports.sysid.calibrate(gap_history)
        if result.status is SysIDStatus.FAILED_TO_CONVERGE:
            # Sim is too divergent to trust the gate we just passed:
            # physical hours are NOT spent on an uncalibrated sim.
            ctx.final_status = HypothesisStatus.REJECTED_SIM
            ctx.final_reason = "sysid-divergent"
            self._ports.planner.ingest_feedback(
                PlannerFeedback(
                    kind="SYSID_DIVERGENT",
                    hypothesis_id=ctx.hypothesis_id,
                    details={"notes": result.notes},
                )
            )
            self._transition(
                OrchestratorState.EARLY_ABORTED,
                "sysid-divergent",
                cycle_id=ctx.cycle_id,
            )
            return
        if (
            result.params_changed
            and ctx.sysid_loops < self._config.max_sysid_recalibration_loops
        ):
            ctx.sysid_env_overrides = result.env_overrides
            ctx.sysid_loops += 1
            self._transition(
                OrchestratorState.SIM_EVALUATING,
                "sysid-recalibrated-regate",
                cycle_id=ctx.cycle_id,
            )
            return
        self._transition(
            OrchestratorState.PHYSICAL_SCENE_VERIFYING,
            "sysid-stable",
            cycle_id=ctx.cycle_id,
        )

    def _handle_scene_verify(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        attempts = self._config.scene_verify_retries + 1
        for attempt in range(attempts):
            self._guard()
            verification = self._ports.scene_verifier.verify()
            if verification.status is SceneVerificationStatus.PASS:
                health = self._ports.bridge.health()
                if (
                    health.consecutive_deadline_misses
                    >= self._config.max_bridge_deadline_misses
                ):
                    raise BridgeDeadlineMissedError(health)
                self._transition(
                    OrchestratorState.PHYSICAL_EXECUTING,
                    "scene-verified",
                    cycle_id=ctx.cycle_id,
                )
                return
            if verification.status is SceneVerificationStatus.RETRYABLE_FAIL:
                continue
            break
        # unverified scene: fail closed
        self._safe_park("SCENE_VERIFICATION_FAILED")
        self._pause_reason = (
            f"SCENE_VERIFICATION_FAILED: {verification.details}"
        )
        self._transition(
            OrchestratorState.PAUSED_ALERT,
            self._pause_reason,
            cycle_id=ctx.cycle_id,
            force=True,
        )

    def _handle_physical_executing(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        assert ctx.harness is not None and ctx.hypothesis_id is not None

        def on_rollout(bundle: RolloutTelemetryBundle) -> None:
            ctx.physical_rollouts.append(bundle)
            health = self._ports.bridge.health()
            if (
                health.consecutive_deadline_misses
                >= self._config.max_bridge_deadline_misses
            ):
                raise BridgeDeadlineMissedError(health)
            if health.consecutive_deadline_misses > 0:
                self._ledger.append_event(
                    EventType.BRIDGE_DEADLINE_MISS,
                    {
                        "rollout_id": bundle.rollout_id,
                        "consecutive": health.consecutive_deadline_misses,
                        "total": health.total_misses,
                    },
                    cycle_id=ctx.cycle_id,
                )

        result = self._ports.physical_executor.execute_batch(
            harness=ctx.harness,
            champion=ctx.champion,
            batch_size=self._config.physical_batch_size,
            recorder=self._ports.flight_recorder,
            on_rollout=on_rollout,
        )
        batch = EvaluationBatchRecordFactory.physical_batch(ctx, result)
        self._ledger.record_evaluation_batch(batch, result.rollouts)
        ctx.physical_batch_ids.append(batch.batch_id)
        # Telemetry is safe on disk now; if the bridge is still unhealthy we
        # park before touching hardware again (covers the case where the
        # executor absorbed the callback exception and returned partial).
        health = self._ports.bridge.health()
        if (
            health.consecutive_deadline_misses
            >= self._config.max_bridge_deadline_misses
        ):
            raise BridgeDeadlineMissedError(health)
        if result.decision is BatchDecision.EARLY_ABORT:
            ctx.final_status = HypothesisStatus.EARLY_ABORTED
            ctx.final_reason = result.early_abort_reason or "sprt-early-abort-physical"
            self._ports.planner.ingest_feedback(
                PlannerFeedback(
                    kind="EARLY_ABORT",
                    hypothesis_id=ctx.hypothesis_id,
                    details={
                        "reason": ctx.final_reason,
                        "n_rollouts_so_far": len(ctx.physical_rollouts),
                    },
                )
            )
            self._transition(
                OrchestratorState.EARLY_ABORTED,
                ctx.final_reason,
                cycle_id=ctx.cycle_id,
            )
            return
        self._transition(
            OrchestratorState.PHYSICAL_RESETTING,
            "physical-batch-complete",
            cycle_id=ctx.cycle_id,
        )

    def _handle_resetting(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        attempts = 1 + self._config.reset_retry_budget
        outcome, details = ResetOutcome.RETRYABLE_FAILURE, ""
        for attempt in range(attempts):
            self._guard()
            outcome, details = self._ports.reset_controller.reset_scene(attempt)
            if outcome is ResetOutcome.SUCCESS:
                self._transition(
                    OrchestratorState.EVALUATING_PROMOTION,
                    f"reset-ok-attempt-{attempt}",
                    cycle_id=ctx.cycle_id,
                )
                return
            if outcome is ResetOutcome.UNRECOVERABLE:
                break
            _LOG.warning(
                "autonomous reset attempt %d failed: %s", attempt, details
            )
        # retry budget exhausted (or unrecoverable object state)
        self._safe_park("RESET_RETRY_EXHAUSTED")
        self._pause_reason = (
            f"RESET_{'UNRECOVERABLE' if outcome is ResetOutcome.UNRECOVERABLE else 'RETRY_EXHAUSTED'}"
            f": {details}"
        )
        self._transition(
            OrchestratorState.PAUSED_ALERT,
            self._pause_reason,
            cycle_id=ctx.cycle_id,
            force=True,
        )

    def _handle_evaluating_promotion(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        evaluation = self._ports.promotion_evaluator.evaluate(
            ctx.physical_rollouts, ctx.champion
        )
        ctx.promotion = evaluation
        if (
            evaluation.decision is PromotionDecisionKind.INCONCLUSIVE
            and len(ctx.physical_rollouts)
            < self._config.max_physical_rollouts_per_cycle
        ):
            self._transition(
                OrchestratorState.PHYSICAL_EXECUTING,
                "inconclusive-more-samples",
                cycle_id=ctx.cycle_id,
            )
            return
        if evaluation.decision is PromotionDecisionKind.PROMOTE:
            assert ctx.harness is not None and ctx.checkpoint is not None
            stats = [
                s for s in evaluation.per_stratum.values()
            ]
            ctx.new_champion = ChampionManifest(
                task_name=ctx.task_name,
                harness_hash=ctx.harness.harness_hash,
                policy_checkpoint_hash=ctx.checkpoint.checkpoint_hash,
                batch_id=ctx.physical_batch_ids[-1],
                wilson_lower_bound=evaluation.overall_wilson_low,
                stratum_stats=stats,
            )
            ctx.final_status = HypothesisStatus.PROMOTED
            ctx.final_reason = "uniform-wilson-low-dominance"
        else:
            ctx.final_status = HypothesisStatus.REJECTED_PHYSICAL
            ctx.final_reason = "; ".join(evaluation.reasons[:4])
        self._transition(
            OrchestratorState.COMMITTING_LEDGER,
            f"promotion-{evaluation.decision.value.lower()}",
            cycle_id=ctx.cycle_id,
        )

    def _handle_early_aborted(self) -> None:
        ctx = self._require_ctx()
        self._ledger.append_event(
            EventType.CYCLE_ABORT,
            {
                "reason": ctx.final_reason,
                "sim_batches": len(ctx.sim_batch_ids),
                "physical_batches": len(ctx.physical_batch_ids),
                "physical_rollouts": len(ctx.physical_rollouts),
            },
            cycle_id=ctx.cycle_id,
        )
        self._transition(
            OrchestratorState.COMMITTING_LEDGER,
            "commit-aborted-cycle",
            cycle_id=ctx.cycle_id,
        )

    def _handle_committing(self) -> None:
        ctx = self._require_ctx()
        self._guard()
        if ctx.hypothesis_id is not None and ctx.final_status is not None:
            self._ledger.update_hypothesis_status(
                ctx.hypothesis_id,
                ctx.final_status,
                rejection_reason=None
                if ctx.final_status is HypothesisStatus.PROMOTED
                else ctx.final_reason,
            )
        if ctx.new_champion is not None:
            self._ledger.record_champion(ctx.new_champion)
            self._ports.planner.ingest_feedback(
                PlannerFeedback(
                    kind="PROMOTED",
                    hypothesis_id=ctx.hypothesis_id,
                    details={
                        "harness_hash": ctx.new_champion.harness_hash,
                        "wilson_lower_bound": ctx.new_champion.wilson_lower_bound,
                    },
                )
            )
        else:
            self._ports.planner.ingest_feedback(
                PlannerFeedback(
                    kind="RETAINED_CHAMPION"
                    if ctx.final_status is HypothesisStatus.REJECTED_PHYSICAL
                    else "EARLY_ABORT",
                    hypothesis_id=ctx.hypothesis_id,
                    details={"reason": ctx.final_reason},
                )
            )
        self._starved_cycles = 0
        self._ledger.append_event(
            EventType.CYCLE_COMMIT,
            {
                "hypothesis_id": ctx.hypothesis_id,
                "final_status": ctx.final_status.value
                if ctx.final_status is not None
                else None,
                "promoted": ctx.new_champion is not None,
                "n_sim_batches": len(ctx.sim_batch_ids),
                "n_physical_batches": len(ctx.physical_batch_ids),
                "n_physical_rollouts": len(ctx.physical_rollouts),
            },
            cycle_id=ctx.cycle_id,
        )
        self._ctx = None
        self._transition(
            OrchestratorState.PROPOSING, "cycle-committed", cycle_id=ctx.cycle_id
        )

    # =====================================================================
    # Operator / interrupt surface
    # =====================================================================

    def emergency_stop(self) -> None:
        """Hard stop from any thread.  The RT watchdog has already failed
        closed on its own; this parks the macro loop and refuses restart
        until ``estop_reset``."""
        self._estop_requested = True
        self._running = False
        try:
            self._ports.bridge.estop()
        except Exception:  # noqa: BLE001
            _LOG.exception("bridge estop command failed (watchdog is authority)")
        self._ledger.append_event(
            EventType.EMERGENCY_STOP, {"requested_at_unix": unix_now()}
        )
        self._force_estop_transition("operator-estop")

    def estop_reset(self, override: OperatorOverride) -> None:
        """Clear EMERGENCY_STOP after the operator has made the cell safe."""
        if self._state is not OrchestratorState.EMERGENCY_STOP:
            raise InvalidTransitionError(
                f"estop_reset invalid from {self._state.value}"
            )
        self._ledger.record_operator_override(override)
        self._estop_requested = False
        self._pause_requested = False
        self._resume_requested = False
        self._transition(OrchestratorState.IDLE, "estop-reset")

    def pause(self, reason: str) -> None:
        self._pause_requested = True
        self._pause_reason = f"OPERATOR_PAUSE: {reason}"
        self._ledger.append_event(
            EventType.PAUSE_REQUESTED, {"reason": reason}
        )

    def resume(self, *, verify_scene: bool = True) -> bool:
        """Leave PAUSED_ALERT.  Scene verification is mandatory by default
        (fail-closed); passing verify_scene=False is an explicit operator
        risk acceptance that is journaled as such."""
        if self._state is not OrchestratorState.PAUSED_ALERT:
            return False
        if verify_scene:
            verification = self._ports.scene_verifier.verify()
            if verification.status is not SceneVerificationStatus.PASS:
                self._ledger.append_event(
                    EventType.RESUME_REQUESTED,
                    {
                        "granted": False,
                        "reason": f"scene verification failed: {verification.details}",
                    },
                )
                return False
        # open cycles from before the pause are superseded, not resumed:
        # their sim-side state may be stale relative to any operator action.
        for cycle_id in self._ledger.get_open_cycles():
            self._ledger.append_event(
                EventType.CYCLE_SUPERSEDED,
                {"reason": "superseded on operator resume"},
                cycle_id=cycle_id,
            )
        self._supersede_open_hypotheses()
        self._pause_requested = False
        self._pause_reason = ""
        self._ctx = None
        self._ledger.append_event(
            EventType.RESUME_REQUESTED,
            {"granted": True, "scene_verified": verify_scene},
        )
        self._transition(OrchestratorState.PROPOSING, "operator-resume")
        self._resume_requested = True
        return True

    def operator_force_promote(
        self, harness_hash: str, override: OperatorOverride
    ) -> ChampionManifest:
        """Operator-forced promotion (robocorectl force-promote).

        Lineage is preserved: the champion row points at the winning batch,
        the hypothesis keeps its honest evaluation status, and the signed
        override event explains why the gate was overridden.
        """
        self._ledger.record_operator_override(override)
        batch = self._ledger.get_latest_batch_for_harness(
            harness_hash, RolloutDomain.PHYSICAL
        )
        if batch is None:
            raise LedgerError(
                f"cannot force-promote {harness_hash}: no physical batch exists"
            )
        checkpoint = self._ledger.get_latest_checkpoint_for_harness(harness_hash)
        if checkpoint is None:
            raise LedgerError(
                f"cannot force-promote {harness_hash}: no policy checkpoint"
            )
        manifest = ChampionManifest(
            task_name=self._config.task_name,
            harness_hash=harness_hash,
            policy_checkpoint_hash=checkpoint.checkpoint_hash,
            batch_id=batch.batch_id,
            wilson_lower_bound=batch.statistics.overall.wilson_low,
            stratum_stats=list(batch.statistics.per_stratum.values()),
        )
        self._ledger.record_champion(manifest)
        return manifest

    def stop(self) -> None:
        """Graceful shutdown request (finishes the current guard segment)."""
        self._running = False
        self._pause_requested = True
        self._pause_reason = "SHUTDOWN"

    # =====================================================================
    # Internals
    # =====================================================================

    def _require_ctx(self) -> CycleContext:
        if self._ctx is None:
            raise InvalidTransitionError(
                f"state {self._state.value} requires an active cycle context"
            )
        return self._ctx

    def _wait_for_resume(self) -> None:
        while self._running and not self._resume_requested:
            if self._estop_requested:
                self._force_estop_transition("estop-while-paused")
                return
            time.sleep(self._config.pause_poll_interval_s)
        self._resume_requested = False

    def _handle_emergency_stop(self) -> None:
        try:
            self._force_estop_transition("estop-signal")
        except LedgerError:
            _LOG.exception("failed to journal estop transition")


class EvaluationBatchRecordFactory:
    """Builds EvaluationBatchRecord rows from evaluator reports."""

    @staticmethod
    def sim_batch(
        ctx: CycleContext, report: SimEvaluationReport, decision: BatchDecision
    ) -> EvaluationBatchRecord:
        assert ctx.harness is not None and ctx.hypothesis_id is not None
        return EvaluationBatchRecord(
            cycle_id=ctx.cycle_id,
            hypothesis_id=ctx.hypothesis_id,
            harness_hash=ctx.harness.harness_hash,
            champion_hash=ctx.champion.harness_hash if ctx.champion else None,
            domain=RolloutDomain.SIM,
            task_name=ctx.task_name,
            n_rollouts=len(report.rollouts),
            decision=decision,
            statistics=report.statistics,
            trigger="SIM_GATE",
            started_at_unix=ctx.started_at_unix,
            committed_at_unix=unix_now(),
        )

    @staticmethod
    def physical_batch(
        ctx: CycleContext, result: PhysicalBatchResult
    ) -> EvaluationBatchRecord:
        assert ctx.harness is not None and ctx.hypothesis_id is not None
        return EvaluationBatchRecord(
            cycle_id=ctx.cycle_id,
            hypothesis_id=ctx.hypothesis_id,
            harness_hash=ctx.harness.harness_hash,
            champion_hash=ctx.champion.harness_hash if ctx.champion else None,
            domain=RolloutDomain.PHYSICAL,
            task_name=ctx.task_name,
            n_rollouts=len(result.rollouts),
            decision=result.decision,
            statistics=result.statistics,
            trigger="PHYSICAL_EXECUTION",
            started_at_unix=result.started_at_unix or unix_now(),
            committed_at_unix=unix_now(),
        )
