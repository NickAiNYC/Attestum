"""test_critic_anti_loop.py -- closed-loop anti-loop enforcement tests.

Three spec-mandated behaviors:

1. **3 same-family rejections -> forced exploration**, asserted on the
   cognitive ``CriticVerdict`` (anti_loop_flagged + ForceExploration with
   the next catalog family).
2. **5 consecutive rejections -> PAUSED_ALERT**, asserted via the REAL
   orchestrator FSM: the guard's alarm callback invokes
   ``RobocoreOrchestrator.pause``, the run loop's guard point raises
   ``PauseRequestedError``, and the state machine parks itself in
   ``PAUSED_ALERT`` with journaled transitions and safe-park side effects.
3. **Scope creep (sys._getframe) rejected pre-AST**, asserted on
   ``SafetyValidationReceipt.ast_errors`` containing the specific reason
   string emitted by the pre-AST scope gate.

The orchestrator runs against fake ports (no sim, no hardware) but a REAL
SQLite ledger, so every transition is journaled and verifiable.  The fakes
are functional, not tripwires: if a defect ever let a rejected proposal
through, the cycle would proceed safely (sim-gate fail, no motion) and the
state assertions would still catch the bug.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from robocore.cognitive.anti_loop import AntiLoopCriticGuard, AntiLoopPolicy
from robocore.cognitive.schemas import (
    CriticDecision,
    ForceExploration,
    MutationType,
)
from robocore.contracts import (
    BatchDecision,
    BatchStatistics,
    DifficultyStrataBin,
    FalsifiablePrediction,
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
    StratumStats,
    SysIDStatus,
    compute_harness_hash,
)
from robocore.runtime.orchestrator import (
    CompileResult,
    CriticVerdict,
    OrchestratorConfig,
    OrchestratorPorts,
    OrchestratorState,
    PhysicalBatchResult,
    PromotionGateEvaluator,
    RobocoreOrchestrator,
    SceneVerification,
    SimEvaluationReport,
    SimGateDecision,
    SysIDResult,
)
from robocore.safety import validate_patch, validate_source
from robocore.storage.ledger_repo import ExperimentLedgerRepository
from robocore.testing.hypothesis_fuzzer import BASELINE_HARNESS_FILES

TASK = "peg_pick_place"
COMPILER_VERSION = "harnessc-0.3.1"


# ---------------------------------------------------------------------------
# Fake ports (deterministic; on the anti-loop path the critic rejects every
# proposal, so none of these are ever reached -- but they are functional)
# ---------------------------------------------------------------------------


@dataclass
class RejectAllCritic:
    """Base critic: rejects everything with a low information gain."""

    calls: int = field(default=0)

    def score(
        self, proposal: HypothesisProposal, context: PlannerContext
    ) -> CriticVerdict:
        self.calls += 1
        return CriticVerdict(
            accept=False,
            score=0.05,
            reason="insufficient expected info gain vs champion baseline",
        )


def _reward_proposal(attempt: int, parent: str) -> HypothesisProposal:
    """A unique REWARD_SCALING-family proposal (unique diff per attempt)."""
    diff = (
        "--- a/reward.py\n"
        "+++ b/reward.py\n"
        f"@@ -7,3 +7,3 @@\n"
        f"-REWARD_ATTEMPT_{attempt}_K = 0.30\n"
        f"+REWARD_ATTEMPT_{attempt}_K = 0.32\n"
    )
    return HypothesisProposal(
        task_name=TASK,
        parent_harness_hash=parent,
        harness_diff=diff,
        ast_fingerprint=f"afp_reward_attempt_{attempt}",
        rationale=f"attempt {attempt}: nudge reward slope",
        falsifiable_prediction=FalsifiablePrediction(
            target_metric="physical_success_rate",
            predicted_delta=0.06,
            ci_low=0.01,
            ci_high=0.11,
        ),
        param_summary={"attempt": float(attempt)},
    )


def _curriculum_proposal(attempt: int, parent: str) -> HypothesisProposal:
    diff = (
        "--- a/curriculum.yaml\n"
        "+++ b/curriculum.yaml\n"
        f"@@ -3,3 +3,3 @@\n"
        f"- oblique_start_deg: 12.0\n"
        f"+ oblique_start_deg: {12.0 + attempt}.0\n"
    )
    return HypothesisProposal(
        task_name=TASK,
        parent_harness_hash=parent,
        harness_diff=diff,
        ast_fingerprint=f"afp_curriculum_attempt_{attempt}",
        rationale=f"attempt {attempt}: widen curriculum",
        falsifiable_prediction=FalsifiablePrediction(
            target_metric="physical_success_rate",
            predicted_delta=0.05,
            ci_low=0.01,
            ci_high=0.09,
        ),
    )


def _empty_context() -> PlannerContext:
    return PlannerContext(task_name=TASK)


@dataclass
class LoopingPlanner:
    """Proposes reward-family diffs forever (one per attempt, all unique)."""

    genesis_hash: str
    attempts: int = field(default=0)

    def propose(self, context: PlannerContext) -> Optional[HypothesisProposal]:
        parent = (
            context.champion.harness_hash
            if context.champion is not None
            else self.genesis_hash
        )
        self.attempts += 1
        return _reward_proposal(self.attempts, parent)

    def ingest_feedback(self, feedback: object) -> None:
        return None


@dataclass
class FakeCompiler:
    """Builds a real manifest + checkpoint for any proposal (happy path)."""

    artifact_root: Path

    def compile_and_precheck(
        self, proposal: HypothesisProposal, parent: HarnessManifest
    ) -> CompileResult:
        hhash = compute_harness_hash(
            parent_harness_hash=parent.harness_hash,
            harness_diff=proposal.harness_diff,
            ast_fingerprint=proposal.ast_fingerprint,
            config={"env": "mujoco"},
        )
        manifest = HarnessManifest(
            harness_hash=hhash,
            task_name=proposal.task_name,
            parent_harness_hash=parent.harness_hash,
            harness_diff=proposal.harness_diff,
            ast_fingerprint=proposal.ast_fingerprint,
            compiler_version=COMPILER_VERSION,
            safety_receipt=SafetyValidationReceipt(
                passed=True,
                violations=[],
                static_analysis_summary="fake compiler clean",
                compiler_version=COMPILER_VERSION,
                kinematic_precheck=True,
            ),
            config={"env": "mujoco"},
        )
        ck_dir = self.artifact_root / "checkpoints"
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck_path = ck_dir / f"{hhash}.safetensors"
        ck_path.write_bytes(b"fake-checkpoint:" + hhash.encode())
        checkpoint = PolicyCheckpointRecord(
            checkpoint_hash="ck_" + hhash[4:24],
            harness_hash=hhash,
            format="safetensors",
            storage_path=str(ck_path),
            num_params=1024,
        )
        return CompileResult(
            passed=True,
            receipt=manifest.safety_receipt,
            manifest=manifest,
            checkpoint=checkpoint,
        )


def _one_rollout(harness_hash: str, success: bool) -> RolloutTelemetryBundle:
    return RolloutTelemetryBundle(
        rollout_id=f"rol_{harness_hash[4:12]}_{int(success)}",
        batch_id="",
        harness_hash=harness_hash,
        task_name=TASK,
        domain=RolloutDomain.SIM,
        stratum=DifficultyStrataBin.EASY_CENTER,
        success=success,
        time_to_completion=5.0 if success else None,
        cumulative_jerk=300.0,
        torque_penalty=20.0,
    )


def _mini_stats(rollouts) -> BatchStatistics:  # noqa: ANN001
    n = len(rollouts)
    s = sum(1 for r in rollouts if r.success)
    rate = s / n if n else 0.0
    return BatchStatistics(
        overall=OverallStats(
            n=n,
            successes=s,
            success_rate=rate,
            wilson_low=max(0.0, rate - 0.1),
            wilson_high=min(1.0, rate + 0.1),
        ),
        per_stratum={},
    )


@dataclass
class FakeSimEvaluator:
    """Always fails the sim gate (safe: nothing proceeds past sim here)."""

    def evaluate(
        self,
        *,
        harness: HarnessManifest,
        champion_baseline: Optional[float],
        batch_size: int,
        env_overrides: Optional[Dict[str, object]],
    ) -> SimEvaluationReport:
        rollouts = [_one_rollout(harness.harness_hash, False)]
        return SimEvaluationReport(
            decision=SimGateDecision.FAILED_GATE,
            rollouts=rollouts,
            statistics=_mini_stats(rollouts),
            reward_hacking_suspected=False,
            notes=["fake evaluator: gate fail"],
        )


@dataclass
class FakeSysID:
    def calibrate(self, gap_history) -> SysIDResult:  # noqa: ANN001
        return SysIDResult(
            status=SysIDStatus.INSUFFICIENT_CHANGE,
            params_changed=False,
        )


@dataclass
class FakeSceneVerifier:
    verified: int = field(default=0)

    def verify(self) -> SceneVerification:
        self.verified += 1
        return SceneVerification(
            status=SceneVerificationStatus.PASS,
            details="fake scene verified",
            vlm_summary="bench clear",
        )


@dataclass
class FakePhysicalExecutor:
    """Never commands motion; returns a single failed (safe) rollout."""

    def execute_batch(
        self,
        *,
        harness: HarnessManifest,
        champion,  # noqa: ANN001
        batch_size: int,
        recorder,  # noqa: ANN001
        on_rollout,  # noqa: ANN001
    ) -> PhysicalBatchResult:
        rollouts = [_one_rollout(harness.harness_hash, False)]
        on_rollout(rollouts[0])
        return PhysicalBatchResult(
            rollouts=rollouts,
            decision=BatchDecision.EARLY_ABORT,
            statistics=_mini_stats(rollouts),
            early_abort_reason="fake executor: no hardware in tests",
        )


@dataclass
class FakeResetController:
    def reset_scene(self, attempt: int) -> Tuple[ResetOutcome, str]:
        return ResetOutcome.SUCCESS, "fake reset"


@dataclass
class FakeBridge:
    holds: int = field(default=0)
    parks: int = field(default=0)
    estops: int = field(default=0)

    def health(self):  # pragma: no cover - unreachable on this path
        from robocore.contracts import BridgeHealthReport

        return BridgeHealthReport()

    def hold(self) -> None:
        self.holds += 1

    def park(self) -> None:
        self.parks += 1

    def estop(self) -> None:
        self.estops += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_three_same_family_rejections_force_exploration() -> Dict[str, int]:
    """3 same-family rejections -> ForceExploration on the CriticVerdict."""
    guard = AntiLoopCriticGuard(RejectAllCritic())
    parent = "hsh_antiloop_parent_000"
    for i in range(1, 4):
        verdict = guard.score(_reward_proposal(i, parent), _empty_context())
        assert verdict.accept is False
        rich = guard.verdicts[-1]
        assert isinstance(rich.force_exploration, ForceExploration) is (i >= 3)
        if i < 3:
            assert rich.anti_loop_flagged is False
    third = guard.verdicts[2]
    assert third.anti_loop_flagged is True
    assert third.decision is CriticDecision.REJECT
    assert third.force_exploration is not None
    assert (
        third.force_exploration.catalog_family is MutationType.CURRICULUM_SHIFT
    ), (
        "next-in-rotation after REWARD_SCALING must be CURRICULUM_SHIFT,"
        f" got {third.force_exploration.catalog_family}"
    )
    assert third.force_exploration.rejected_streak == 3
    assert all(
        f is MutationType.REWARD_SCALING
        for f in third.force_exploration.rejected_families
    )
    assert "FORCE_EXPLORATION" in third.reason
    assert guard.trailing_family_streak == 3
    # interleave a different family: the run resets; no flag until the new
    # family itself accumulates three consecutive rejections
    guard.score(_curriculum_proposal(9, parent), _empty_context())
    assert guard.trailing_family_streak == 1
    assert guard.verdicts[-1].anti_loop_flagged is False
    return {"verdicts": len(guard.verdicts), "streak": 3}


def test_five_consecutive_rejections_pause_alert(
    tmp_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """5 consecutive rejections -> orchestrator FSM parks in PAUSED_ALERT."""
    import tempfile

    work = (
        Path(tmp_dir)
        if tmp_dir is not None
        else Path(tempfile.mkdtemp(prefix="robocore_antiloop_"))
    )
    ledger = ExperimentLedgerRepository(work / "ledger.db")
    genesis = HarnessManifest.genesis(
        task_name=TASK,
        harness_diff="--- a/harness.yaml\n+++ b/harness.yaml\n@@\n+base\n",
        ast_fingerprint="afp_antiloop_genesis",
        compiler_version=COMPILER_VERSION,
        config={"env": "mujoco"},
    )
    ledger.upsert_harness(genesis)

    planner = LoopingPlanner(genesis_hash=genesis.harness_hash)
    bridge = FakeBridge()
    scene_verifier = FakeSceneVerifier()
    holder: Dict[str, RobocoreOrchestrator] = {}

    def _alarm(reason: str) -> None:
        orch = holder.get("orchestrator")
        assert orch is not None, "alarm fired before orchestrator wiring"
        orch.pause(reason)

    guard = AntiLoopCriticGuard(
        RejectAllCritic(),
        policy=AntiLoopPolicy(),
        on_exhausted=_alarm,
    )
    ports = OrchestratorPorts(
        planner=planner,
        critic=guard,
        compiler=FakeCompiler(artifact_root=work),
        sim_evaluator=FakeSimEvaluator(),
        sysid=FakeSysID(),
        scene_verifier=scene_verifier,
        physical_executor=FakePhysicalExecutor(),
        reset_controller=FakeResetController(),
        bridge=bridge,
        promotion_evaluator=PromotionGateEvaluator(),
        flight_recorder=None,
        context_builder=None,
    )
    config = OrchestratorConfig(
        task_name=TASK,
        planner_cooldown_s=0.0,
        pause_poll_interval_s=0.01,
        max_proposal_attempts_per_cycle=5,
        max_starved_cycles=3,
    )
    orchestrator = RobocoreOrchestrator(config, ports, ledger)
    holder["orchestrator"] = orchestrator

    thread = threading.Thread(target=orchestrator.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if orchestrator.state is OrchestratorState.PAUSED_ALERT:
                break
            time.sleep(0.005)
        assert orchestrator.state is OrchestratorState.PAUSED_ALERT, (
            "expected PAUSED_ALERT after 5 consecutive rejections;"
            f" orchestrator is in {orchestrator.state.value}"
        )
        assert (
            "ANTI_LOOP_CONSECUTIVE_REJECTIONS(5)" in orchestrator.pause_reason
        ), orchestrator.pause_reason
        assert planner.attempts >= 5
        assert guard.consecutive_rejections >= 5
        # fail-closed side effects: deterministic hold + park happened
        assert bridge.holds >= 1 and bridge.parks >= 1
        # the journal proves the FSM transition and the pause request
        last_state = ledger.get_last_event_of_type("ORCHESTRATOR_STATE")
        assert last_state is not None
        assert last_state.payload()["state"] == "PAUSED_ALERT"
        pause_event = ledger.get_last_event_of_type("PAUSE_REQUESTED")
        assert pause_event is not None
        assert "ANTI_LOOP_CONSECUTIVE_REJECTIONS(5)" in str(
            pause_event.payload()
        )
    finally:
        orchestrator.stop()
        thread.join(timeout=10.0)
        assert not thread.is_alive(), "orchestrator thread did not stop"

    # operator resume with scene verification -> back to PROPOSING
    assert scene_verifier.verified == 0
    assert orchestrator.resume(verify_scene=True) is True
    assert scene_verifier.verified == 1
    assert orchestrator.state is OrchestratorState.PROPOSING
    ok, bad_seq = ledger.verify_chain_integrity()
    assert ok, f"ledger chain corrupt at seq {bad_seq}"
    n_events = int(
        ledger._conn.execute(  # noqa: SLF001 - read-only count in test
            "SELECT COUNT(*) AS n FROM ledger_events"
        ).fetchone()["n"]
    )
    ledger.close()
    return {"planner_attempts": planner.attempts, "ledger_events": n_events}


def test_scope_creep_rejected_pre_ast() -> Dict[str, int]:
    """sys._getframe scope creep: rejected by the PRE-AST scope gate with
    the specific reason string present in SafetyValidationReceipt.ast_errors."""
    source = (
        "def leaky():\n"
        "    frame = sys._getframe(1)\n"
        "    return frame.f_locals\n"
    )
    receipt = validate_source(source)
    assert receipt.ast_passed is False
    matches = [
        e
        for e in receipt.ast_errors
        if "pre-ast scope violation" in e and "sys._getframe" in e
    ]
    assert matches, (
        "specific sys._getframe reason missing from ast_errors:"
        f" {receipt.ast_errors}"
    )
    # proof the rejection happened at the scope gate, not the AST walk
    assert "pre-ast scope gate" in receipt.static_analysis_summary
    # a harness diff carrying the same scope creep is equally doomed
    diff_with_creep = (
        "--- a/reward.py\n"
        "+++ b/reward.py\n"
        "@@ -1,2 +1,4 @@\n"
        " import math\n"
        " \n"
        "+def leaky():\n"
        "+    return sys._getframe(1).f_globals\n"
    )
    result, diff_receipt = validate_patch(
        BASELINE_HARNESS_FILES, diff_with_creep
    )
    assert diff_receipt.ast_passed is False
    assert any("sys._getframe" in e for e in diff_receipt.ast_errors)
    return {"scope_errors": len(matches)}
