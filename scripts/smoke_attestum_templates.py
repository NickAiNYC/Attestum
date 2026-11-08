#!/usr/bin/env python3
"""smoke_attestum_templates.py -- acceptance script for Item B (the
three contractual templates plus the retrospective scope document).

Six checks, each raising on failure (fail-closed; no PASS line is
printed unless every check passed):

  1. All three Pydantic models instantiate from the anonymized
     templates/examples/ data.
  2. Evidence Passport: exactly 11 sections; passport_hash recomputes
     from the canonical JSON recipe; the example's hash is stable
     across runs; the section-11 ternary rejects zero- and two-outcome
     payloads; the renderer covers all three outcome kinds.
  3. Economic Case: the worked example (6 interventions/h, $180 per
     intervention itemized 85/65/30, 22 engineering h/month at $95/h,
     35% projected improvement on 9/27 observed) computes an
     annualized value that matches an independent calculation within
     $100, with the Wilson CI and payback derived, and the model
     fails closed on a projection outside its own CI and on a
     negative-value case.
  4. Case-Study Agreement: all 8 clauses render; clause 2 contains
     the exact retroactive-fee sentence; clause 5 references clause 2
     by number; elections select the operative text.
  5. Renderer purity: every render function called twice returns
     byte-identical output.
  6. The four docs/ templates render from the example data with zero
     unresolved {{slot}} markers, and the slot filler fails closed on
     a missing slot.

Run: python scripts/smoke_attestum_templates.py   ->   "PASS 6/6"
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attestum.brand import BRAND  # noqa: E402
from attestum.templates import (  # noqa: E402
    ANONYMIZATION_OPTIONS,
    CaseStudyAgreement,
    ClearNegative,
    EconomicCase,
    EvidencePassport,
    ImprovementResult,
    NoGo,
    UnresolvedSlotError,
    VerifiedImprovement,
    build_clauses,
    render_case_study_agreement,
    render_economic_case,
    render_evidence_passport,
    render_markdown_template,
)
from attestum.templates.evidence_passport import (  # noqa: E402
    SECTION_FIELDS,
    SECTION_TITLES,
)
from robocore.stats import wilson_score_interval  # noqa: E402

EXAMPLES_DIR = REPO_ROOT / "templates" / "examples"
DOCS_DIR = REPO_ROOT / "docs"

RETRO_FEE_SENTENCE = (
    "If Customer exercises withdrawal rights under clause 5, the fee for "
    "the assessment is retroactively invoiced at $20,000 USD."
)

CLAUSE_TITLES = (
    "Scope of work and deliverables",
    "Fee schedule",
    "Data handling and anonymization",
    "Publication rights, timing, and approval",
    "Customer withdrawal",
    "Intellectual property boundaries",
    "Confidentiality and mutual NDA",
    "Acceptance and signature block",
)

# Computed fields are derived by the model and must not be fed back in
# on re-validation (the model is extra="forbid" by design).
_COMPUTED_ECONOMIC_KEYS = (
    "cost_per_intervention_usd",
    "confidence_interval",
    "annual_interventions_current",
    "annual_interventions_avoided",
    "gross_annual_savings_usd",
    "ongoing_engineering_cost_usd",
    "annualized_value_usd",
    "weekly_value_usd",
    "payback_period_weeks",
    "value_at_ci_low_usd",
    "value_at_ci_high_usd",
)


def _economic_inputs(case: EconomicCase) -> Dict[str, Any]:
    """The example's input fields only (computed fields stripped)."""
    dump = case.model_dump(mode="json")
    for key in _COMPUTED_ECONOMIC_KEYS:
        dump.pop(key, None)
    dump["cost_breakdown"].pop("total_usd", None)
    return dump


def _load_example(name: str) -> Dict[str, Any]:
    return json.loads((EXAMPLES_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Slot derivation for the docs/ templates (values come from the examples)
# ---------------------------------------------------------------------------


def _flatten_scalars(node: Any, out: Dict[str, str], leaf: str = "") -> None:
    """Flatten nested dicts to leaf-name -> str(value); lists and Nones skip.

    A duplicate leaf name with a DIFFERENT value raises: that means a
    template slot has become ambiguous.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            _flatten_scalars(value, out, key)
    elif isinstance(node, list) or node is None:
        return
    else:
        if not leaf:
            return
        text = str(node)
        if leaf in out and out[leaf] != text:
            raise AssertionError(
                f"ambiguous slot {leaf!r}: {out[leaf]!r} vs {text!r}"
            )
        out[leaf] = text


def _passport_slots(passport: EvidencePassport) -> Dict[str, str]:
    dump = passport.model_dump(mode="json")
    slots: Dict[str, str] = {}
    _flatten_scalars(dump, slots)
    slots["provenance_rows"] = "\n".join(
        f"- **{s['name']}** ({s['kind']}, {s['format']}) — "
        f"sha256 `{s['sha256']}`, exported {s['exported_utc']}"
        for s in dump["data_provenance"]["sources"]
    )
    slots["taxonomy_rows"] = "\n".join(
        f"| {label['label']} | {label['baseline_count']} | {label['definition']} |"
        for label in dump["failure_taxonomy"]["labels"]
    )
    slots["stop_conditions_rows"] = "; ".join(
        dump["safety_oversight"]["stop_conditions"]
    )
    slots["artifacts_rows"] = "\n".join(
        f"| {a['name']} | {a['kind']} | `{a['content_hash']}` | {a['size_bytes']:,} |"
        for a in dump["reproducible_artifacts"]["artifacts"]
    )
    slots["outcome_rows"] = _outcome_rows(dump["improvement_result"])
    slots["company_name"] = BRAND.name
    slots["contact_email"] = BRAND.contact_email
    return slots


def _outcome_rows(outcome: Dict[str, Any]) -> str:
    if outcome.get("verified_improvement") is not None:
        v = outcome["verified_improvement"]
        return "\n".join(
            [
                "**Outcome: VERIFIED IMPROVEMENT**",
                "",
                f"- **Statement:** {v['statement']}",
                f"- **Metric:** {v['metric']}",
                f"- **Baseline:** {v['baseline_value']} → **Post:** "
                f"{v['post_value']} (**{v['relative_improvement_pct']}%** reduction)",
                f"- **Wilson 95% CI:** [{v['ci_low_pct']}%, {v['ci_high_pct']}%]",
                f"- **Gate decision:** {v['gate_decision']}",
            ]
        )
    if outcome.get("clear_negative") is not None:
        c = outcome["clear_negative"]
        return "\n".join(
            [
                "**Outcome: CLEAR NEGATIVE**",
                "",
                f"- **Statement:** {c['statement']}",
                f"- **Finding:** {c['finding']}",
                "- **Evidence refs:** " + "; ".join(c["evidence_refs"]),
            ]
        )
    n = outcome["no_go"]
    return "\n".join(
        [
            "**Outcome: NO-GO**",
            "",
            f"- **Statement:** {n['statement']}",
            f"- **Reason code:** {n['reason_code']}",
        ]
    )


def _economic_slots(case: EconomicCase) -> Dict[str, str]:
    dump = case.model_dump(mode="json")
    slots: Dict[str, str] = {}
    _flatten_scalars(dump, slots)
    ci_low, ci_high = case.confidence_interval
    slots["ci_low_pct"] = str(ci_low)
    slots["ci_high_pct"] = str(ci_high)
    slots["assumptions_rows"] = "\n".join(
        f"{i}. {a}" for i, a in enumerate(dump["assumptions"], start=1)
    )
    slots["challenge_log_note"] = (
        "_Blank at generation; populated line by line during customer review._"
        if not dump["challenge_log"]
        else "\n".join(f"{i}. {c}" for i, c in enumerate(dump["challenge_log"], 1))
    )
    slots["company_name"] = BRAND.name
    slots["contact_email"] = BRAND.contact_email
    return slots


def _agreement_slots(agreement: CaseStudyAgreement) -> Dict[str, str]:
    dump = agreement.model_dump(mode="json")
    slots: Dict[str, str] = {}
    _flatten_scalars(dump, slots)
    slots["company_name"] = BRAND.name
    slots["anonymization_description"] = ANONYMIZATION_OPTIONS[
        agreement.anonymization_election
    ]
    for clause in dump["clauses"]:
        slots[f"clause_{clause['number']}_recommended"] = clause["recommended_text"]
        slots[f"clause_{clause['number']}_fallback"] = clause["fallback_text"]
    return slots


def _scope_slots() -> Dict[str, str]:
    return {
        "company_name": BRAND.name,
        "contact_email": BRAND.contact_email,
        "customer_code": "CUSTOMER-A",
        "assessment_id": "EP-2026-0114-A1-CELL03",
        "agreement_id": "AGR-2026-0114",
        "data_package_due": "5",
        "draft_due": "15",
        "review_due": "20",
        "final_due": "21",
        "assessment_fee_usd": "$20,000",
        "retroactive_fee_usd": "$20,000",
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_models_instantiate(
    passport: EvidencePassport, case: EconomicCase, agreement: CaseStudyAgreement
) -> str:
    assert passport.passport_id == "EP-2026-0114-A1-CELL03"
    assert case.annual_interventions_current == 24960.0
    assert len(agreement.clauses) == 8
    assert agreement.clause_elections.get(4) == "fallback"
    # Anchor the worked-example economics to the spec's numbers.
    assert case.current_intervention_rate_per_hour == 6.0
    assert case.cost_per_intervention_usd == 180.0
    assert case.engineering_hours_per_month == 22.0
    assert case.loaded_hourly_rate_usd == 95.0
    assert case.projected_improvement_pct == 35.0
    assert case.assessment_fee_usd == 20000.0
    return "all three models instantiate from templates/examples/ data"


def check_passport_integrity(passport: EvidencePassport) -> str:
    assert len(SECTION_FIELDS) == 11, "passport must have exactly 11 sections"
    assert len(SECTION_TITLES) == 11
    for field_name in SECTION_FIELDS:
        assert hasattr(passport, field_name), f"missing section {field_name}"

    rendered = render_evidence_passport(passport)
    for index, title in enumerate(SECTION_TITLES, start=1):
        assert f"## {index}. {title}" in rendered, f"rendered passport missing section {index}"

    # Independent recompute of the hash from the documented recipe.
    payload = passport.model_dump(mode="json", exclude={"passport_hash"})
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert passport.passport_hash == expected, "passport_hash does not recompute"

    # Stability across runs: a second instantiation is byte-identical.
    again = EvidencePassport.model_validate(
        _load_example("evidence-passport-example.json")
    )
    assert again.passport_hash == passport.passport_hash
    assert EvidencePassport.canonical_json(again) == EvidencePassport.canonical_json(passport)

    # The ternary fails closed on zero and on two simultaneous outcomes.
    for bad in (
        {},
        {
            "verified_improvement": passport.improvement_result.verified_improvement,
            "no_go": NoGo(reason_code="X", statement="two outcomes set"),
        },
    ):
        try:
            ImprovementResult(**bad)
        except ValueError:
            continue
        raise AssertionError("ImprovementResult accepted an invalid ternary payload")

    # The renderer covers the other two outcome kinds.
    base = passport.model_dump(mode="json")
    clear_negative = EvidencePassport.model_validate(
        {
            **base,
            "improvement_result": {
                "clear_negative": {
                    "finding": "gripper mechanical wear beyond spec",
                    "evidence_refs": ["maintenance_log_2026-08.xlsx"],
                    "statement": "The assessment found the problem is mechanical.",
                }
            },
        }
    )
    assert "**Outcome: CLEAR NEGATIVE**" in render_evidence_passport(clear_negative)
    no_go = EvidencePassport.model_validate(
        {
            **base,
            "improvement_result": {
                "no_go": {
                    "reason_code": "INSUFFICIENT_BASELINE",
                    "statement": "baseline window too short to support a claim",
                }
            },
        }
    )
    assert "**Outcome: NO-GO**" in render_evidence_passport(no_go)
    return (
        "passport: 11 sections, hash recomputes and is stable, "
        "ternary fails closed, renderer covers all three outcomes"
    )


def check_economic_case_worked_example(case: EconomicCase) -> str:
    assert case.operating_hours_per_year == 4160.0
    # Independent calculation (separate arithmetic path from the model).
    expected_value = (
        6.0 * 4160.0 * (35.0 / 100.0) * 180.0 - (22.0 * 12.0 * 95.0)
    )
    assert abs(case.annualized_value_usd - expected_value) <= 100.0, (
        f"annualized value {case.annualized_value_usd} vs expected {expected_value}"
    )
    # Itemized cost must sum to the headline cost.
    assert (
        case.cost_breakdown.labor_usd
        + case.cost_breakdown.downtime_usd
        + case.cost_breakdown.scrap_usd
        == case.cost_per_intervention_usd
        == 180.0
    )
    # CI derived, not asserted: matches robocore's Wilson interval on 9/27.
    low, high = wilson_score_interval(9, 27)
    assert case.confidence_interval == (round(low * 100.0, 2), round(high * 100.0, 2))
    ci_low, ci_high = case.confidence_interval
    assert ci_low <= 35.0 <= ci_high, "projection outside its own CI"
    assert abs(ci_low - 18.0) <= 1.5 and abs(ci_high - 52.0) <= 1.5, (
        "worked-example CI drifted from the illustrative [18%, 52%] bounds"
    )
    # Payback derived from the fee and the weekly value.
    expected_payback = 20000.0 / (expected_value / 52.0)
    assert abs(case.payback_period_weeks - expected_payback) <= 0.01

    # Fail-closed: a projection outside the CI is a claim, not a case.
    inputs = _economic_inputs(case)
    rejected_projection = False
    try:
        EconomicCase.model_validate({**inputs, "projected_improvement_pct": 60.0})
    except ValueError:
        rejected_projection = True
    assert rejected_projection, "EconomicCase accepted a projection outside its CI"

    # Fail-closed: payback is undefined when the case never pays back.
    losing = EconomicCase.model_validate(
        {
            **inputs,
            "case_id": "EC-NEGATIVE",
            "observed_improvement_successes": 3,
            "observed_improvement_trials": 1000,
            "projected_improvement_pct": 0.3,
        }
    )
    payback_rejected = False
    try:
        _ = losing.payback_period_weeks
    except ValueError:
        payback_rejected = True
    assert payback_rejected, "payback_period_weeks defined for a value-destroying case"
    return (
        f"economic case: annualized ${case.annualized_value_usd:,.2f} matches "
        f"independent calc ${expected_value:,.2f}; CI [{ci_low}, {ci_high}]; "
        "fails closed on bad projections"
    )


def check_agreement(agreement: CaseStudyAgreement) -> str:
    rendered = render_case_study_agreement(agreement)
    parts = rendered.split("## Clause ")
    assert len(parts) == 9, f"expected 8 clauses, got {len(parts) - 1}"
    for number, title in enumerate(CLAUSE_TITLES, start=1):
        assert parts[number].startswith(f"{number} — {title}"), (
            f"clause {number} missing or mistitled"
        )
    assert RETRO_FEE_SENTENCE in parts[2], "clause 2 lost the retroactive-fee sentence"
    assert "clause 2" in parts[5], "clause 5 must reference clause 2 by number"
    # Elections select the operative text.
    assert "Elected variant: fallback" in parts[4]
    assert (
        agreement.elected_text(4) == agreement.clauses[3].fallback_text
    ), "election did not select the fallback text"
    # Clause building is pure in fee and election.
    first = build_clauses(20000.0, "replace_customer_name_with_code")
    second = build_clauses(20000.0, "replace_customer_name_with_code")
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]
    return (
        "agreement: 8 clauses render, clause 2 carries the exact retroactive-fee "
        "sentence, clause 5 references clause 2, elections select text"
    )


def check_renderer_purity(
    passport: EvidencePassport, case: EconomicCase, agreement: CaseStudyAgreement
) -> str:
    assert render_evidence_passport(passport) == render_evidence_passport(passport)
    md_a, json_a = render_economic_case(case)
    md_b, json_b = render_economic_case(case)
    assert md_a == md_b and md_a, "economic-case markdown not byte-identical"
    assert json.dumps(json_a, sort_keys=True) == json.dumps(json_b, sort_keys=True)
    assert (
        render_case_study_agreement(agreement)
        == render_case_study_agreement(agreement)
    )
    # The ledger artifact is hash-carrying and deterministic.
    assert json_a["ledger_entry"] == "evidence_passport_delivered"
    assert json_a["content_sha256"] == json_b["content_sha256"]
    return "renderers are pure: two calls, byte-identical outputs"


def check_docs_templates_render(
    passport: EvidencePassport, case: EconomicCase, agreement: CaseStudyAgreement
) -> str:
    template_slots = {
        "evidence-passport-template.md": _passport_slots(passport),
        "economic-case-template.md": _economic_slots(case),
        "case-study-agreement-template.md": _agreement_slots(agreement),
        "retrospective-assessment-scope.md": _scope_slots(),
    }
    for name, slots in template_slots.items():
        template = (DOCS_DIR / name).read_text(encoding="utf-8")
        assert "{{" in template, f"{name} has no slots to render"
        rendered = render_markdown_template(template, slots)
        assert "{{" not in rendered, f"{name} rendered with unresolved markers"
    # The slot filler fails closed on a missing slot.
    rejected_marker = False
    try:
        render_markdown_template("Hello {{who}}", {})
    except UnresolvedSlotError:
        rejected_marker = True
    assert rejected_marker, "slot filler accepted an unresolved marker"
    return "all four docs/ templates render with zero unresolved {{slot}} markers"


def main() -> int:
    passport = EvidencePassport.model_validate(
        _load_example("evidence-passport-example.json")
    )
    case = EconomicCase.model_validate(
        _load_example("economic-case-example.json")
    )
    agreement = CaseStudyAgreement.model_validate(
        _load_example("case-study-agreement-example.json")
    )

    checks = (
        ("models instantiate", lambda: check_models_instantiate(passport, case, agreement)),
        ("passport integrity", lambda: check_passport_integrity(passport)),
        ("economic worked example", lambda: check_economic_case_worked_example(case)),
        ("agreement clauses", lambda: check_agreement(agreement)),
        (
            "renderer purity",
            lambda: check_renderer_purity(passport, case, agreement),
        ),
        (
            "docs templates render",
            lambda: check_docs_templates_render(passport, case, agreement),
        ),
    )
    passed = 0
    for name, check in checks:
        detail = check()
        passed += 1
        print(f"  ok  {name}: {detail}")
    print(f"PASS {passed}/{len(checks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
