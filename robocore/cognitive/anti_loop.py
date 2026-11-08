"""robocore.cognitive.anti_loop -- closed-loop anti-loop critic guard.

Deliverable 5 support module.  Batch 1 shipped the ``ICriticGuard`` port
but no anti-loop implementation; this decorator wraps ANY base critic and
adds the closed-loop disciplines:

* **family streak detection** -- when the last ``same_family_trigger``
  (default 3) consecutive rejections all share one mutation family, the
  guard flags the rejection and attaches a
  :class:`~robocore.cognitive.schemas.ForceExploration` directive naming
  the next family in the fixed catalog rotation;
* **consecutive-rejection alarm** -- when consecutive rejections (any
  family) reach ``consecutive_rejection_alarm`` (default 5), the guard
  fires the ``on_exhausted`` callback.  The runtime wires that callback to
  ``RobocoreOrchestrator.pause``: the FSM parks itself in ``PAUSED_ALERT``
  and demands an operator.  Fail-closed: a planner stuck in a loop never
  burns another robot-hour silently.

The guard keeps a rich-verdict trail (the cognitive
:class:`CriticVerdict`) alongside the orchestrator-port verdicts it
returns, so tests and telemetry can assert on ``anti_loop_flagged`` and
the forced-exploration directive directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from robocore.cognitive.schemas import (
    CriticDecision,
    CriticVerdict,
    ForceExploration,
    MutationType,
    classify_mutation,
    next_in_rotation,
)
from robocore.contracts import HypothesisProposal, PlannerContext
from robocore.runtime.orchestrator import CriticVerdict as PortVerdict
from robocore.runtime.orchestrator import ICriticGuard

__all__ = ["AntiLoopPolicy", "AntiLoopCriticGuard"]


@dataclass(frozen=True)
class AntiLoopPolicy:
    """Thresholds of the anti-loop guard (frozen; tune via injection)."""

    same_family_trigger: int = 3
    consecutive_rejection_alarm: int = 5


@dataclass
class AntiLoopCriticGuard(ICriticGuard):
    """Decorator over any ``ICriticGuard`` adding anti-loop telemetry.

    Parameters
    ----------
    base:
        The wrapped critic; accept/reject verdicts always come from it --
        this guard only *observes* rejections and attaches directives.
    policy:
        Streak thresholds.
    on_exhausted:
        Fired when the consecutive-rejection counter hits a multiple of
        ``policy.consecutive_rejection_alarm`` (each fresh burst of five
        rejections demands operator attention again).  The runtime passes
        ``orchestrator.pause`` here; tests pass a recorder.
    """

    base: ICriticGuard
    policy: AntiLoopPolicy = field(default_factory=AntiLoopPolicy)
    on_exhausted: Optional[Callable[[str], None]] = None

    def __post_init__(self) -> None:
        self._family_history: List[MutationType] = []
        self._rich_verdicts: List[CriticVerdict] = []
        self._consecutive_rejections = 0

    # -- introspection -------------------------------------------------------

    @property
    def consecutive_rejections(self) -> int:
        return self._consecutive_rejections

    @property
    def verdicts(self) -> List[CriticVerdict]:
        """Rich verdict trail (one entry per scored proposal)."""
        return list(self._rich_verdicts)

    @property
    def trailing_family_streak(self) -> int:
        """Length of the current same-family rejection run."""
        if not self._family_history:
            return 0
        last = self._family_history[-1]
        streak = 0
        for family in reversed(self._family_history):
            if family is not last:
                break
            streak += 1
        return streak

    # -- ICriticGuard --------------------------------------------------------

    def score(
        self, proposal: HypothesisProposal, context: PlannerContext
    ) -> PortVerdict:
        port_verdict = self.base.score(proposal, context)
        family = classify_mutation(proposal.harness_diff)

        if port_verdict.accept:
            # An accepted proposal breaks every loop: reset the counters.
            self._consecutive_rejections = 0
            self._family_history = []
            rich = CriticVerdict(
                decision=CriticDecision.ACCEPT,
                reason=port_verdict.reason,
                anti_loop_flagged=False,
                expected_info_gain=_clamp01(port_verdict.score),
                mutation_family=family,
            )
            self._rich_verdicts.append(rich)
            return port_verdict

        self._consecutive_rejections += 1
        self._family_history.append(family)
        streak = self.trailing_family_streak
        flagged = streak >= self.policy.same_family_trigger
        force: Optional[ForceExploration] = None
        reason = port_verdict.reason
        if flagged:
            force = ForceExploration(
                catalog_family=next_in_rotation(family),
                rejected_streak=streak,
                rejected_families=[family] * streak,
                reason=(
                    f"{streak} consecutive rejections in family"
                    f" {family.value}; the fixed catalog rotation directs"
                    f" the next proposal to"
                    f" {next_in_rotation(family).value}"
                ),
            )
            reason = (
                f"{port_verdict.reason} | FORCE_EXPLORATION: family"
                f" {family.value} exhausted after {streak} rejections ->"
                f" explore {next_in_rotation(family).value} next"
            )
        rich = CriticVerdict(
            decision=CriticDecision.REJECT,
            reason=reason,
            anti_loop_flagged=flagged,
            expected_info_gain=_clamp01(port_verdict.score),
            mutation_family=family,
            force_exploration=force,
        )
        self._rich_verdicts.append(rich)

        alarm = self.policy.consecutive_rejection_alarm
        if (
            self.on_exhausted is not None
            and self._consecutive_rejections >= alarm
            and self._consecutive_rejections % alarm == 0
        ):
            self.on_exhausted(
                f"ANTI_LOOP_CONSECUTIVE_REJECTIONS("
                f"{self._consecutive_rejections}): planner rejected"
                f" {alarm} proposals in a row; runtime paused for operator"
                " review (fail-closed)"
            )
        return PortVerdict(
            accept=False, score=port_verdict.score, reason=reason
        )


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))
