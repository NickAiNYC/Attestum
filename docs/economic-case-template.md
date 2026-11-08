# Economic Case — {{case_id}}

> Fill-in template. The Economic Case COMPUTES; it does not describe:
> every headline number below is derived by
> `attestum/templates/economic_case.py` from the inputs, and the
> confidence interval comes from the runtime's Wilson interval
> (`robocore.stats.wilson_score_interval`) on the observed improvement.

**Customer:** {{customer_code}} · **Prepared (UTC):** {{prepared_utc}} · **Currency:** {{currency}}

## Inputs (every line is challengeable)

| Input | Value |
| --- | --- |
| Current intervention rate | {{current_intervention_rate_per_hour}} per operating hour |
| Operating hours per year | {{operating_hours_per_year}} h |
| Cost per intervention (itemized) | {{labor_usd}} labor + {{downtime_usd}} downtime + {{scrap_usd}} scrap = **{{cost_per_intervention_usd}}** |
| Ongoing engineering burden | {{engineering_hours_per_month}} h/month at {{loaded_hourly_rate_usd}}/h (loaded) |
| Observed improvement | {{observed_improvement_successes}} of {{observed_improvement_trials}} matched failure events eliminated |
| Projected improvement | {{projected_improvement_pct}}% (Wilson 95% CI [{{ci_low_pct}}%, {{ci_high_pct}}%]) |
| Assessment fee | {{assessment_fee_usd}} |

## Computed value

```
annual_interventions = rate x hours_per_year
                     = {{current_intervention_rate_per_hour}} x {{operating_hours_per_year}} = {{annual_interventions_current}}
avoided              = annual_interventions x projected_improvement
                     = {{annual_interventions_current}} x {{projected_improvement_pct}}% = {{annual_interventions_avoided}}
gross_savings        = avoided x cost_per_intervention
                     = {{annual_interventions_avoided}} x {{cost_per_intervention_usd}} = {{gross_annual_savings_usd}}
engineering_cost     = hours_per_month x 12 x loaded_rate
                     = {{engineering_hours_per_month}} x 12 x {{loaded_hourly_rate_usd}} = {{ongoing_engineering_cost_usd}}
annualized_value     = gross_savings - engineering_cost
                     = {{gross_annual_savings_usd}} - {{ongoing_engineering_cost_usd}} = {{annualized_value_usd}}
payback_weeks        = assessment_fee / (annualized_value / 52)
                     = {{assessment_fee_usd}} / {{weekly_value_usd}} = {{payback_period_weeks}}
```

## Sensitivity at the Wilson bounds

- Value if the true improvement is the CI lower bound ({{ci_low_pct}}%): **{{value_at_ci_low_usd}}** per year
- Value if the true improvement is the CI upper bound ({{ci_high_pct}}%): **{{value_at_ci_high_usd}}** per year

## Assumptions (each one is a sentence the customer can challenge)

{{assumptions_rows}}

## Challenge log

{{challenge_log_note}}

---

_Prepared by {{company_name}} — {{contact_email}}. The machine-readable
artifact for the ledger is the JSON rendering of this case; both are
byte-stable for identical inputs._
