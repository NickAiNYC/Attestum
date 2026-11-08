"""attestum.templates.economic_case -- the economics of one assessment.

This model COMPUTES; it does not describe.  Every headline number the
customer sees (annualized value, payback period, confidence interval)
is derived by this module from the raw inputs, so two analysts filling
in the same inputs get the same case, byte for byte.

The confidence interval is not a free-text claim: it is the Wilson
score interval (from the frozen runtime's statistics engine,
robocore.stats) on the observed improvement, and the analyst's
projected improvement must lie inside it or the model refuses to
instantiate.

Value formula (all fields are challengeable inputs):

    annual_interventions_current = current_intervention_rate_per_hour
                                   * operating_hours_per_year
    annual_interventions_avoided = annual_interventions_current
                                   * projected_improvement_pct / 100
    gross_annual_savings_usd     = annual_interventions_avoided
                                   * cost_per_intervention_usd
    ongoing_engineering_cost_usd = engineering_hours_per_month * 12
                                   * loaded_hourly_rate_usd
    annualized_value_usd         = gross_annual_savings_usd
                                   - ongoing_engineering_cost_usd
    payback_period_weeks         = assessment_fee_usd
                                   / (annualized_value_usd / 52)
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

from robocore.stats import wilson_score_interval


class InterventionCostBreakdown(BaseModel):
    """Itemized cost of ONE intervention: labor + downtime + scrap."""

    model_config = ConfigDict(extra="forbid")

    labor_usd: float
    downtime_usd: float
    scrap_usd: float

    @computed_field
    @property
    def total_usd(self) -> float:
        return round(self.labor_usd + self.downtime_usd + self.scrap_usd, 2)


class EconomicCase(BaseModel):
    """The computed economic case for one assessed cell."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    customer_code: str
    prepared_utc: datetime
    currency: str = "USD"

    # --- inputs -----------------------------------------------------------
    current_intervention_rate_per_hour: float
    operating_hours_per_year: float
    cost_breakdown: InterventionCostBreakdown
    engineering_hours_per_month: float
    loaded_hourly_rate_usd: float
    observed_improvement_successes: int
    observed_improvement_trials: int
    projected_improvement_pct: float
    assessment_fee_usd: float

    # --- review surfaces ----------------------------------------------------
    #: Each entry must be a full sentence the customer can challenge.
    assumptions: List[str]
    #: Blank at generation; populated line by line during customer review.
    challenge_log: List[str] = []

    # --- computed -----------------------------------------------------------

    @computed_field
    @property
    def cost_per_intervention_usd(self) -> float:
        return self.cost_breakdown.total_usd

    @computed_field
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """Wilson 95% bounds on the observed improvement, in percent."""
        low, high = wilson_score_interval(
            self.observed_improvement_successes,
            self.observed_improvement_trials,
        )
        return (round(low * 100.0, 2), round(high * 100.0, 2))

    @computed_field
    @property
    def annual_interventions_current(self) -> float:
        return round(
            self.current_intervention_rate_per_hour * self.operating_hours_per_year, 2
        )

    @computed_field
    @property
    def annual_interventions_avoided(self) -> float:
        return round(
            self.annual_interventions_current * self.projected_improvement_pct / 100.0,
            2,
        )

    @computed_field
    @property
    def gross_annual_savings_usd(self) -> float:
        return round(
            self.annual_interventions_avoided * self.cost_per_intervention_usd, 2
        )

    @computed_field
    @property
    def ongoing_engineering_cost_usd(self) -> float:
        return round(
            self.engineering_hours_per_month * 12.0 * self.loaded_hourly_rate_usd, 2
        )

    @computed_field
    @property
    def annualized_value_usd(self) -> float:
        return round(
            self.gross_annual_savings_usd - self.ongoing_engineering_cost_usd, 2
        )

    @computed_field
    @property
    def weekly_value_usd(self) -> float:
        return round(self.annualized_value_usd / 52.0, 2)

    @computed_field
    @property
    def payback_period_weeks(self) -> float:
        if self.weekly_value_usd <= 0.0:
            raise ValueError(
                "payback_period_weeks is undefined: the modeled weekly value "
                f"is {self.weekly_value_usd} {self.currency}, so the case as "
                "modeled never pays the assessment fee back. Fix the inputs "
                "or record a clear_negative outcome instead."
            )
        return round(self.assessment_fee_usd / self.weekly_value_usd, 4)

    @computed_field
    @property
    def value_at_ci_low_usd(self) -> float:
        low, _high = wilson_score_interval(
            self.observed_improvement_successes,
            self.observed_improvement_trials,
        )
        return round(
            self.annual_interventions_current
            * (low * 100.0)
            / 100.0
            * self.cost_per_intervention_usd
            - self.ongoing_engineering_cost_usd,
            2,
        )

    @computed_field
    @property
    def value_at_ci_high_usd(self) -> float:
        _low, high = wilson_score_interval(
            self.observed_improvement_successes,
            self.observed_improvement_trials,
        )
        return round(
            self.annual_interventions_current
            * (high * 100.0)
            / 100.0
            * self.cost_per_intervention_usd
            - self.ongoing_engineering_cost_usd,
            2,
        )

    @model_validator(mode="after")
    def _projection_must_lie_inside_ci(self) -> "EconomicCase":
        low, high = self.confidence_interval
        if not (low <= self.projected_improvement_pct <= high):
            raise ValueError(
                f"projected_improvement_pct={self.projected_improvement_pct} "
                f"lies outside the Wilson 95% CI [{low}, {high}] computed from "
                f"{self.observed_improvement_successes}/"
                f"{self.observed_improvement_trials} observed events; a "
                "projection outside its own evidence is a claim, not a case"
            )
        return self


__all__ = ["EconomicCase", "InterventionCostBreakdown"]
