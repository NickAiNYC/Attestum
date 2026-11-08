"""robocore.stats -- statistical primitives shared by the evaluator, the
promotion gate, and the orchestrator.

Everything here is deterministic and dependency-light:

* :func:`wilson_score_interval` -- low-sample binomial CI (n < 40 physical
  rollouts makes the normal approximation unusable).
* :func:`bootstrap_mean_diff_ci` -- seeded bootstrap for TTC-style metrics.
* :class:`SPRTMonitor` -- Wald sequential probability ratio test that kills
  dead-end challengers mid-batch (physical robot hours are the scarcest
  resource; a 0/4 opening streak must abort the batch).
* :class:`PromotionGate` -- stratified, conservative promotion rule.  Success
  dominance is *necessary but not sufficient*: a challenger that regresses
  cumulative jerk or torque penalty by more than 15% over the champion is
  blocked even at 100% success.  Safety regressions are not tradeable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from robocore.contracts import (
    BatchDecision,
    ChampionManifest,
    DifficultyStrataBin,
    PromotionDecisionKind,
    RolloutTelemetryBundle,
    StratumStats,
)

__all__ = [
    "wilson_score_interval",
    "bootstrap_mean_diff_ci",
    "SPRTMonitor",
    "PromotionGateConfig",
    "PromotionEvaluation",
    "PromotionGate",
]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


def wilson_score_interval(
    successes: int, n: int, z: float = 1.96
) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (low, high) in [0, 1].  With n=0 returns (0.0, 0.0) -- callers
    must treat that as "no evidence", never as "measured zero".
    """
    if n <= 0:
        return 0.0, 0.0
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Seeded bootstrap CI for mean(a) - mean(b).

    Empty inputs are treated as NaN-free empty samples: the diff is defined
    as 0 with a degenerate interval, so callers can branch on length.
    """
    if len(a) == 0 and len(b) == 0:
        return 0.0, 0.0
    if len(a) == 0 or len(b) == 0:
        lone = np.asarray(a if len(a) else b, dtype=np.float64)
        rng = np.random.default_rng(seed)
        n = len(lone)
        samples = np.array(
            [
                float(np.mean(lone[rng.integers(0, n, n)]))
                if len(a) == 0
                else -float(np.mean(lone[rng.integers(0, n, n)]))
                for _ in range(n_boot)
            ]
        )
        lo, hi = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        return float(lo), float(hi)
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(arr_a), (n_boot, len(arr_a)))
    idx_b = rng.integers(0, len(arr_b), (n_boot, len(arr_b)))
    diffs = arr_a[idx_a].mean(axis=1) - arr_b[idx_b].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Sequential Probability Ratio Test
# ---------------------------------------------------------------------------


class SPRTMonitor:
    """Wald SPRT for H0: p <= p0 (champion level) vs H1: p >= p1.

    Decisions:
      * LLR >= log((1-beta)/alpha)  -> challenger is promotion-grade;
        caller continues to full sample (the promotion gate still decides).
      * LLR <= log(beta/(1-alpha))  -> EARLY_ABORT: the challenger is
        futility-bound; stop spending physical rollouts immediately.
      * otherwise CONTINUE.
    """

    def __init__(self, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.10) -> None:
        if not (0.0 < p0 < p1 < 1.0):
            raise ValueError(f"SPRT requires 0 < p0 < p1 < 1, got p0={p0}, p1={p1}")
        if not (0.0 < alpha < 0.5 and 0.0 < beta < 0.5):
            raise ValueError("alpha and beta must be in (0, 0.5)")
        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta
        self.upper_boundary = math.log((1.0 - beta) / alpha)
        self.lower_boundary = math.log(beta / (1.0 - alpha))
        self._k = 0  # successes
        self._n = 0  # trials
        self.llr = 0.0

    def update(self, success: bool) -> str:
        """Consume one rollout outcome; returns SPRTDecision as str."""
        self._n += 1
        if success:
            self._k += 1
        # log-likelihood ratio increment for binomial
        self.llr += math.log(self.p1 / self.p0) if success else math.log(
            (1.0 - self.p1) / (1.0 - self.p0)
        )
        if self.llr <= self.lower_boundary:
            return "EARLY_ABORT"
        if self.llr >= self.upper_boundary:
            return "ACCEPT_H1"
        return "CONTINUE"

    @property
    def successes(self) -> int:
        return self._k

    @property
    def n(self) -> int:
        return self._n

    def point_estimate(self) -> float:
        return self._k / self._n if self._n else 0.0


# ---------------------------------------------------------------------------
# Stratified promotion gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionGateConfig:
    required_strata: Tuple[DifficultyStrataBin, ...] = (
        DifficultyStrataBin.EASY_CENTER,
        DifficultyStrataBin.MEDIUM_OBLIQUE,
        DifficultyStrataBin.HARD_BOUNDARY,
    )
    min_samples_per_stratum: int = 8
    min_total_samples: int = 24
    # A challenger may not regress smoothness metrics vs the champion by more
    # than 15%, regardless of success rate.  Non-tradeable safety property.
    max_jerk_regression_frac: float = 0.15
    max_torque_penalty_regression_frac: float = 0.15
    # Genesis (no champion yet): absolute bars.
    genesis_min_wilson_low: float = 0.50
    genesis_max_abs_cumulative_jerk: float = 5000.0
    genesis_max_abs_torque_penalty: float = 500.0
    z: float = 1.96


@dataclass(frozen=True)
class PromotionEvaluation:
    decision: PromotionDecisionKind
    per_stratum: Dict[str, StratumStats]
    reasons: List[str] = field(default_factory=list)
    overall_wilson_low: float = 0.0
    jerk_blocked: bool = False
    dominated_stratum: Optional[str] = None


class PromotionGate:
    """Conservative stratified promotion decision.

    Rule order (first failing rule wins):
      1. sample sufficiency per required stratum -> else INCONCLUSIVE
      2. jerk / torque-penalty regression gate    -> RETAIN_CHAMPION (blocked)
      3. per-stratum Wilson-low dominance         -> PROMOTE / REJECT / INCONCLUSIVE
    """

    def __init__(self, config: Optional[PromotionGateConfig] = None) -> None:
        self.config = config or PromotionGateConfig()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _stratum_stats(
        rollouts: Sequence[RolloutTelemetryBundle], stratum: DifficultyStrataBin
    ) -> Optional[StratumStats]:
        sub = [r for r in rollouts if r.stratum == stratum]
        if not sub:
            return None
        n = len(sub)
        successes = sum(1 for r in sub if r.success)
        rate = successes / n
        low, high = wilson_score_interval(successes, n)
        ttcs = [r.time_to_completion for r in sub if r.time_to_completion is not None]
        jerks = [r.cumulative_jerk for r in sub]
        torques = [r.torque_penalty for r in sub]
        return StratumStats(
            stratum=stratum,
            n=n,
            successes=successes,
            success_rate=rate,
            wilson_low=low,
            wilson_high=high,
            mean_ttc=float(np.mean(ttcs)) if ttcs else None,
            mean_cumulative_jerk=float(np.mean(jerks)) if jerks else None,
            mean_torque_penalty=float(np.mean(torques)) if torques else None,
        )

    @staticmethod
    def _champion_stratum(
        champion: ChampionManifest, stratum: DifficultyStrataBin
    ) -> Optional[StratumStats]:
        for s in champion.stratum_stats:
            if s.stratum == stratum:
                return s
        return None

    # -- gate --------------------------------------------------------------

    def evaluate(
        self,
        rollouts: Sequence[RolloutTelemetryBundle],
        champion: Optional[ChampionManifest],
    ) -> PromotionEvaluation:
        cfg = self.config
        reasons: List[str] = []
        per_stratum: Dict[str, StratumStats] = {}

        # (1) sample sufficiency ------------------------------------------------
        for stratum in cfg.required_strata:
            stats = self._stratum_stats(rollouts, stratum)
            if stats is None or stats.n < cfg.min_samples_per_stratum:
                reasons.append(
                    f"insufficient samples in {stratum.value}: "
                    f"{0 if stats is None else stats.n}/{cfg.min_samples_per_stratum}"
                )
            if stats is not None:
                per_stratum[stratum.value] = stats
        if len(rollouts) < cfg.min_total_samples:
            reasons.append(
                f"insufficient total samples: {len(rollouts)}/{cfg.min_total_samples}"
            )
        if reasons:
            return PromotionEvaluation(
                decision=PromotionDecisionKind.INCONCLUSIVE,
                per_stratum=per_stratum,
                reasons=["SAMPLES_INSUFFICIENT: " + r for r in reasons],
            )

        total_n = len(rollouts)
        total_succ = sum(1 for r in rollouts if r.success)
        low_all, _ = wilson_score_interval(total_succ, total_n)

        # (2) smoothness / actuator-safety regression gate -----------------------
        challenger_jerk = float(np.mean([r.cumulative_jerk for r in rollouts]))
        challenger_torque = float(np.mean([r.torque_penalty for r in rollouts]))
        jerk_blocked = False
        if champion is None:
            if challenger_jerk > cfg.genesis_max_abs_cumulative_jerk:
                jerk_blocked = True
                reasons.append(
                    f"GENESIS_JERK_CAP_EXCEEDED: {challenger_jerk:.1f} > "
                    f"{cfg.genesis_max_abs_cumulative_jerk:.1f}"
                )
            if challenger_torque > cfg.genesis_max_abs_torque_penalty:
                jerk_blocked = True
                reasons.append(
                    f"GENESIS_TORQUE_CAP_EXCEEDED: {challenger_torque:.1f} > "
                    f"{cfg.genesis_max_abs_torque_penalty:.1f}"
                )
        else:
            champ_jerk = self._champion_mean(
                champion, "mean_cumulative_jerk", rollouts, "cumulative_jerk"
            )
            champ_torque = self._champion_mean(
                champion, "mean_torque_penalty", rollouts, "torque_penalty"
            )
            if champ_jerk > 0 and challenger_jerk > champ_jerk * (
                1.0 + cfg.max_jerk_regression_frac
            ):
                jerk_blocked = True
                reasons.append(
                    f"JERK_REGRESSION: challenger {challenger_jerk:.1f} vs champion "
                    f"{champ_jerk:.1f} exceeds +{cfg.max_jerk_regression_frac:.0%}"
                )
            if champ_torque > 0 and challenger_torque > champ_torque * (
                1.0 + cfg.max_torque_penalty_regression_frac
            ):
                jerk_blocked = True
                reasons.append(
                    f"TORQUE_PENALTY_REGRESSION: challenger {challenger_torque:.1f} vs "
                    f"champion {champ_torque:.1f} exceeds "
                    f"+{cfg.max_torque_penalty_regression_frac:.0%}"
                )
        if jerk_blocked:
            return PromotionEvaluation(
                decision=PromotionDecisionKind.RETAIN_CHAMPION,
                per_stratum=per_stratum,
                overall_wilson_low=low_all,
                reasons=reasons,
                jerk_blocked=True,
            )

        # (3) stratified Wilson dominance ----------------------------------------
        dominated: Optional[str] = None
        all_dominate = True
        for stratum in cfg.required_strata:
            stats = per_stratum[stratum.value]
            if champion is None:
                if stats.wilson_low < cfg.genesis_min_wilson_low:
                    all_dominate = False
                    reasons.append(
                        f"GENESIS_BAR_MISSED {stratum.value}: wilson_low "
                        f"{stats.wilson_low:.3f} < {cfg.genesis_min_wilson_low:.3f}"
                    )
                continue
            champ_stats = self._champion_stratum(champion, stratum)
            if champ_stats is None:
                # Champion never measured this stratum: require aggregate dominance.
                if low_all <= champion.wilson_lower_bound:
                    all_dominate = False
                    reasons.append(
                        f"UNMEASURED_STRATUM {stratum.value}: aggregate "
                        f"wilson_low {low_all:.3f} <= champion "
                        f"{champion.wilson_lower_bound:.3f}"
                    )
                continue
            if stats.wilson_low <= champ_stats.wilson_low:
                all_dominate = False
                reasons.append(
                    f"NO_DOMINANCE {stratum.value}: challenger wilson_low "
                    f"{stats.wilson_low:.3f} <= champion {champ_stats.wilson_low:.3f}"
                )
            if stats.wilson_high < champ_stats.wilson_low:
                dominated = dominated or stratum.value

        if dominated is not None:
            # Statistically dominated in a required stratum -> kill outright.
            return PromotionEvaluation(
                decision=PromotionDecisionKind.RETAIN_CHAMPION,
                per_stratum=per_stratum,
                overall_wilson_low=low_all,
                reasons=reasons + [f"DOMINATED_IN_STRATUM: {dominated}"],
                dominated_stratum=dominated,
            )
        if all_dominate:
            return PromotionEvaluation(
                decision=PromotionDecisionKind.PROMOTE,
                per_stratum=per_stratum,
                overall_wilson_low=low_all,
                reasons=["UNIFORM_WILSON_LOW_DOMINANCE"] + reasons,
            )
        return PromotionEvaluation(
            decision=PromotionDecisionKind.INCONCLUSIVE,
            per_stratum=per_stratum,
            overall_wilson_low=low_all,
            reasons=reasons,
        )

    @staticmethod
    def _champion_mean(
        champion: ChampionManifest,
        stats_field: str,
        rollouts: Sequence[RolloutTelemetryBundle],
        fallback_field: str,
    ) -> float:
        """Champion's mean for a metric; falls back to challenger-side prior
        (its own first-batch telemetry) when the champion manifest lacks it."""
        values = [
            getattr(s, stats_field)
            for s in champion.stratum_stats
            if getattr(s, stats_field) is not None
        ]
        if values:
            return float(np.mean(values))
        fb = [getattr(r, fallback_field) for r in rollouts]
        return float(np.mean(fb)) if fb else 0.0


def batch_decision_from_promotion(decision: PromotionDecisionKind) -> BatchDecision:
    return {
        PromotionDecisionKind.PROMOTE: BatchDecision.PROMOTE,
        PromotionDecisionKind.RETAIN_CHAMPION: BatchDecision.REJECT,
        PromotionDecisionKind.INCONCLUSIVE: BatchDecision.INCONCLUSIVE,
    }[decision]
