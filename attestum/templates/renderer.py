"""attestum.templates.renderer -- pure rendering functions.

All functions here are deterministic and side-effect free: no
filesystem writes, no network, no LLM calls.  Calling any of them
twice with the same model produces byte-identical output; that
property is asserted by the acceptance script.

Three model renderers:

- ``render_evidence_passport(model) -> str``          (customer markdown)
- ``render_economic_case(model) -> (str, dict)``      (markdown + ledger JSON)
- ``render_case_study_agreement(model) -> str``       (customer markdown)

Plus one generic slot filler for the fill-in markdown templates under
docs/, which refuses to return output containing unresolved
``{{slot}}`` markers (the same discipline the runtime's prompt
registry enforces on prompts).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Mapping, Tuple

from attestum.brand import BRAND
from attestum.templates.case_study_agreement import CaseStudyAgreement
from attestum.templates.economic_case import EconomicCase
from attestum.templates.evidence_passport import (
    SECTION_FIELDS,
    SECTION_TITLES,
    EvidencePassport,
)


class UnresolvedSlotError(ValueError):
    """A markdown template still contains {{slot}} markers after rendering."""


# ---------------------------------------------------------------------------
# Generic markdown template rendering (docs/ fill-in templates)
# ---------------------------------------------------------------------------

_SLOT_MARKER = re.compile(r"\{\{[^}]*\}\}")


def render_markdown_template(template_text: str, slots: Mapping[str, str]) -> str:
    """Fill ``{{slot}}`` markers; raise UnresolvedSlotError on leftovers.

    Slot names are matched literally (``{{name}}`` with no spaces); the
    function refuses to return any output that still contains ``{{``.
    """
    rendered = template_text
    for key, value in slots.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    leftover = _SLOT_MARKER.search(rendered)
    if leftover is not None or "{{" in rendered:
        match = leftover.group(0) if leftover is not None else "{{"
        raise UnresolvedSlotError(f"unresolved slot marker: {match!r}")
    return rendered


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _label(key: str) -> str:
    return key.replace("_", " ").title()


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _money0(value: float) -> str:
    return f"${value:,.0f}"


def _render_kv(key: str, value: Any) -> List[str]:
    """Render one model field as markdown lines (lists of scalars inline)."""
    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            lines = [f"- **{_label(key)}:**", ""]
            for index, item in enumerate(value, start=1):
                rendered_items = "; ".join(
                    f"**{_label(k)}:** {v}" for k, v in item.items()
                )
                lines.append(f"  {index}. {rendered_items}")
            return lines
        if not value:
            return [f"- **{_label(key)}:** (none)"]
        return [f"- **{_label(key)}:** " + "; ".join(str(v) for v in value)]
    return [f"- **{_label(key)}:** {value}"]


# ---------------------------------------------------------------------------
# Evidence Passport
# ---------------------------------------------------------------------------


def render_evidence_passport(model: EvidencePassport) -> str:
    """Render the passport as the customer-facing markdown document."""
    lines: List[str] = []
    lines.append(f"# Evidence Passport — {model.passport_id}")
    lines.append("")
    lines.append(
        f"**Customer:** {model.cell_identification.customer_code} · "
        f"**Site:** {model.cell_identification.site_code} · "
        f"**Cell:** {model.cell_identification.cell_id}"
    )
    lines.append(
        f"**Prepared by:** {model.prepared_by} · "
        f"**Issued (UTC):** {model.issued_utc.isoformat()} · "
        f"**Schema:** {model.schema_version}"
    )
    lines.append(
        f"**Passport hash** (sha256 over canonical JSON of this document "
        f"minus the hash itself): `{model.passport_hash}`"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    dump = model.model_dump(mode="json")
    for index, (field_name, title) in enumerate(
        zip(SECTION_FIELDS, SECTION_TITLES), start=1
    ):
        lines.append(f"## {index}. {title}")
        lines.append("")
        if field_name == "improvement_result":
            lines.extend(_render_outcome(dump[field_name]))
        else:
            for key, value in dump[field_name].items():
                lines.extend(_render_kv(key, value))
        lines.append("")

    lines.append("---")
    lines.append(
        f"_Issued by {BRAND.name} — {BRAND.contact_email}. Verify this "
        "document against its passport hash before relying on it._"
    )
    return "\n".join(lines) + "\n"


def _render_outcome(outcome: Dict[str, Any]) -> List[str]:
    """Render section 11: exactly one of the three outcomes is set."""
    if outcome.get("verified_improvement") is not None:
        v = outcome["verified_improvement"]
        return [
            "**Outcome: VERIFIED IMPROVEMENT**",
            "",
            f"- **Statement:** {v['statement']}",
            f"- **Metric:** {v['metric']}",
            f"- **Baseline:** {v['baseline_value']} → **Post:** {v['post_value']}"
            f" (**{v['relative_improvement_pct']:.1f}%** reduction)",
            f"- **Wilson 95% CI:** [{v['ci_low_pct']}%, {v['ci_high_pct']}%]",
            f"- **Gate decision:** {v['gate_decision']}",
        ]
    if outcome.get("clear_negative") is not None:
        c = outcome["clear_negative"]
        return [
            "**Outcome: CLEAR NEGATIVE**",
            "",
            f"- **Statement:** {c['statement']}",
            f"- **Finding:** {c['finding']}",
            "- **Evidence refs:** " + "; ".join(c["evidence_refs"]),
        ]
    n = outcome["no_go"]
    return [
        "**Outcome: NO-GO**",
        "",
        f"- **Statement:** {n['statement']}",
        f"- **Reason code:** {n['reason_code']}",
    ]


# ---------------------------------------------------------------------------
# Economic Case
# ---------------------------------------------------------------------------


def render_economic_case(model: EconomicCase) -> Tuple[str, Dict[str, Any]]:
    """Render the case as (customer markdown, ledger JSON artifact).

    The JSON artifact is the dict that gets hashed into the ledger
    entry ``evidence_passport_delivered``; it carries its own
    ``content_sha256`` (sha256 over the canonical JSON of the model)
    so the future ledger integration can reference it without
    re-deriving anything.
    """
    ci_low, ci_high = model.confidence_interval
    lines: List[str] = []
    lines.append(f"# Economic Case — {model.case_id}")
    lines.append("")
    lines.append(f"**Customer:** {model.customer_code}")
    lines.append(f"**Prepared (UTC):** {model.prepared_utc.isoformat()}")
    lines.append(f"**Currency:** {model.currency}")
    lines.append("")
    lines.append("## Inputs (every line is challengeable)")
    lines.append("")
    lines.append("| Input | Value |")
    lines.append("| --- | --- |")
    lines.append(
        "| Current intervention rate | "
        f"{model.current_intervention_rate_per_hour} per operating hour |"
    )
    lines.append(
        "| Operating hours per year | "
        f"{model.operating_hours_per_year:,.0f} h |"
    )
    lines.append(
        "| Cost per intervention (itemized) | "
        f"{_money(model.cost_breakdown.labor_usd)} labor + "
        f"{_money(model.cost_breakdown.downtime_usd)} downtime + "
        f"{_money(model.cost_breakdown.scrap_usd)} scrap = "
        f"**{_money(model.cost_per_intervention_usd)}** |"
    )
    lines.append(
        "| Ongoing engineering burden | "
        f"{model.engineering_hours_per_month} h/month at "
        f"{_money(model.loaded_hourly_rate_usd)}/h (loaded) |"
    )
    lines.append(
        "| Observed improvement | "
        f"{model.observed_improvement_successes} of "
        f"{model.observed_improvement_trials} matched failure events "
        "eliminated |"
    )
    lines.append(
        "| Projected improvement | "
        f"{model.projected_improvement_pct}% "
        f"(Wilson 95% CI [{ci_low}%, {ci_high}%]) |"
    )
    lines.append(
        "| Assessment fee | " f"{_money0(model.assessment_fee_usd)} |"
    )
    lines.append("")
    lines.append("## Computed value (not a description — the arithmetic)")
    lines.append("")
    lines.append("```")
    lines.append(
        "annual_interventions = rate x hours_per_year"
    )
    lines.append(
        f"                   = {model.current_intervention_rate_per_hour} x "
        f"{model.operating_hours_per_year:,.0f} = "
        f"{model.annual_interventions_current:,.0f}"
    )
    lines.append(
        "avoided             = annual_interventions x projected_improvement"
    )
    lines.append(
        f"                   = {model.annual_interventions_current:,.0f} x "
        f"{model.projected_improvement_pct}% = "
        f"{model.annual_interventions_avoided:,.0f}"
    )
    lines.append(
        "gross_savings       = avoided x cost_per_intervention"
    )
    lines.append(
        f"                   = {model.annual_interventions_avoided:,.0f} x "
        f"{_money(model.cost_per_intervention_usd)} = "
        f"{_money(model.gross_annual_savings_usd)}"
    )
    lines.append(
        "engineering_cost    = hours_per_month x 12 x loaded_rate"
    )
    lines.append(
        f"                   = {model.engineering_hours_per_month} x 12 x "
        f"{_money(model.loaded_hourly_rate_usd)} = "
        f"{_money(model.ongoing_engineering_cost_usd)}"
    )
    lines.append(
        "annualized_value    = gross_savings - engineering_cost"
    )
    lines.append(
        f"                   = {model.gross_annual_savings_usd:,.2f} - "
        f"{model.ongoing_engineering_cost_usd:,.2f} = "
        f"**{_money(model.annualized_value_usd)}**"
    )
    lines.append(
        "payback_weeks       = assessment_fee / (annualized_value / 52)"
    )
    lines.append(
        f"                   = {model.assessment_fee_usd:,.0f} / "
        f"{model.weekly_value_usd:,.2f} = "
        f"**{model.payback_period_weeks} weeks**"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Sensitivity at the Wilson bounds")
    lines.append("")
    lines.append(
        f"- Value if the true improvement is the CI lower bound "
        f"({ci_low}%): **{_money(model.value_at_ci_low_usd)}** per year"
    )
    lines.append(
        f"- Value if the true improvement is the CI upper bound "
        f"({ci_high}%): **{_money(model.value_at_ci_high_usd)}** per year"
    )
    lines.append("")
    lines.append("## Assumptions (each one is a sentence you can challenge)")
    lines.append("")
    for index, assumption in enumerate(model.assumptions, start=1):
        lines.append(f"{index}. {assumption}")
    lines.append("")
    lines.append("## Challenge log")
    lines.append("")
    if model.challenge_log:
        for index, entry in enumerate(model.challenge_log, start=1):
            lines.append(f"{index}. {entry}")
    else:
        lines.append(
            "_Blank at generation; populated line by line during customer "
            "review. A challenge that survives review becomes an assumption "
            "revision; one that does not becomes a note here._"
        )
    lines.append("")
    lines.append("---")
    lines.append(
        f"_Prepared by {BRAND.name} — {BRAND.contact_email}. The machine-"
        "readable artifact for the ledger is the JSON rendering of this "
        "case; both are byte-stable for identical inputs._"
    )

    ledger_artifact = _economic_ledger_artifact(model)
    return "\n".join(lines) + "\n", ledger_artifact


def _economic_ledger_artifact(model: EconomicCase) -> Dict[str, Any]:
    """The JSON artifact destined for the ledger entry.

    Carries ``content_sha256``: sha256 over the canonical JSON of the
    model dump (sorted keys, compact separators), matching the passport
    hash recipe so every attestum artifact is verified the same way.
    """
    data = model.model_dump(mode="json")
    canonical = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema": "attestum.economic_case/1.0.0",
        "ledger_entry": "evidence_passport_delivered",
        "model": data,
        "content_sha256": digest,
    }


# ---------------------------------------------------------------------------
# Case-Study Agreement
# ---------------------------------------------------------------------------


def render_case_study_agreement(model: CaseStudyAgreement) -> str:
    """Render the agreement as a signable markdown document."""
    lines: List[str] = []
    lines.append(f"# Case-Study Agreement — {model.agreement_id}")
    lines.append("")
    lines.append(f"**Between:** {BRAND.name} and **{model.customer_code}**")
    lines.append(f"**Effective date:** {model.effective_date.isoformat()}")
    lines.append(
        f"**Assessment fee (list):** {_money0(model.assessment_fee_usd)} — "
        "waived under the free-with-case-study path in clause 2"
    )
    lines.append(
        f"**Elected anonymization:** {model.anonymization_election}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    for clause in model.clauses:
        election = model.clause_elections.get(clause.number, "recommended")
        lines.append(f"## Clause {clause.number} — {clause.title}")
        lines.append("")
        lines.append(f"**Elected variant: {election}**")
        lines.append("")
        lines.append(f"**Recommended.** {clause.recommended_text}")
        lines.append("")
        lines.append(f"**Fallback.** {clause.fallback_text}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Signatures")
    lines.append("")
    lines.append(f"**{model.customer_code} signatory:**")
    lines.append("")
    lines.append("Name: ____________________  Title: ____________________")
    lines.append("")
    lines.append("Date: ____________________")
    lines.append("")
    lines.append(f"**{BRAND.name} signatory:**")
    lines.append("")
    lines.append("Name: ____________________  Title: ____________________")
    lines.append("")
    lines.append("Date: ____________________")
    lines.append("")
    return "\n".join(lines) + "\n"


__all__ = [
    "UnresolvedSlotError",
    "render_case_study_agreement",
    "render_economic_case",
    "render_evidence_passport",
    "render_markdown_template",
]
