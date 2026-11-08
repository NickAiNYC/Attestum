# Templates — how to fill them

Four fill-in documents live in `docs/`; one anonymized, fully-filled
example per contractual template lives in `examples/`. The examples
are **synthetic**: every customer name is a code (`CUSTOMER-A`),
every serial is a coded string, and every hash is illustrative. Never
put real customer data in this directory.

| Template | Model (code) | Example data (anonymized) |
| --- | --- | --- |
| `docs/evidence-passport-template.md` | `attestum.templates.evidence_passport.EvidencePassport` | `examples/evidence-passport-example.json` |
| `docs/economic-case-template.md` | `attestum.templates.economic_case.EconomicCase` | `examples/economic-case-example.json` |
| `docs/case-study-agreement-template.md` | `attestum.templates.case_study_agreement.CaseStudyAgreement` | `examples/case-study-agreement-example.json` |
| `docs/retrospective-assessment-scope.md` | (scope document; slot-filled, no Pydantic model) | slot values in `scripts/smoke_attestum_templates.py` |

## Rules that apply to all four

1. **Slots are required.** Every slot marker must be consumed at
   render time. Rendering with `render_markdown_template` raises
   `UnresolvedSlotError` if any marker survives; the acceptance script
   asserts this for all four templates.
2. **The company name comes from the brand record.** Never type the
   company name into a template or a filled document; the renderers
   interpolate it from `attestum.brand.BRAND`. The acceptance script
   greps the repository to enforce this.
3. **Hashes are content addresses, not decorations.** An artifact row
   without its sha256 is an incomplete row.

## Evidence Passport (11 sections)

Fill every section; the model refuses to instantiate with any section
missing. Section 11 is a ternary — set exactly one of
`verified_improvement`, `clear_negative`, `no_go`. `passport_hash` is
never filled by hand: it is the sha256 of the canonical JSON of the
model minus the hash itself, recomputed on every instantiation.

## Economic Case (computed, not written)

Fill ONLY the inputs: intervention rate, operating hours, the
itemized cost breakdown, the engineering burden, the observed
improvement counts, the projection, and the fee. Every other number
(annualized value, payback, confidence interval) is derived by the
model — the projected improvement must lie inside the Wilson 95% CI on
the observed counts or the model refuses to instantiate. Assumptions
are full sentences the customer can challenge; the challenge log
stays blank at generation and is filled during review.

## Case-Study Agreement (8 clauses, 2 texts each)

Fill the variable data only: agreement id, customer code, effective
date, fee, the elected anonymization method, and any clause elections
(default: all recommended). The clause texts are built from the brand
record and the fee by `build_clauses` — if a customer needs different
language, that is a fallback election, not a hand-edit. Clause 2
carries the retroactive fee sentence; clause 5 references clause 2 by
number. Do not break that cross-reference.

## Rendering programmatically

```python
import json
from pathlib import Path

from attestum.templates import (
    CaseStudyAgreement,
    EconomicCase,
    EvidencePassport,
    render_case_study_agreement,
    render_economic_case,
    render_evidence_passport,
)

examples = Path("templates/examples")

passport = EvidencePassport.model_validate(
    json.loads((examples / "evidence-passport-example.json").read_text())
)
print(render_evidence_passport(passport))

case = EconomicCase.model_validate(
    json.loads((examples / "economic-case-example.json").read_text())
)
markdown, ledger_json = render_economic_case(case)  # (customer md, ledger artifact)

agreement = CaseStudyAgreement.model_validate(
    json.loads((examples / "case-study-agreement-example.json").read_text())
)
print(render_case_study_agreement(agreement))
```

All renderers are pure: no filesystem writes, no network, no LLM
calls; identical inputs produce byte-identical output.
