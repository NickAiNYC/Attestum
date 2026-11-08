# Attestum Roadmap

What gets built, in what order, and — more importantly — what does
not get built yet. The sequencing rule comes from the strategy summary
([ATTESTUM_STRATEGY.md](../ATTESTUM_STRATEGY.md)): nothing operational
ships before the first signed assessment is delivered and published.

## Now (sprint: company shell and contractual surface)

| Item | Scope | Status |
| --- | --- | --- |
| Item A — Brand reconciliation & repo topology | company-first README, strategy summary, this roadmap, architecture doc, `attestum/brand.py`, brand acceptance script | delivered |
| Item B — Contractual templates | Evidence Passport, Economic Case, Case-Study Agreement models + renderers, retrospective assessment scope doc, anonymized worked examples | delivered |
| Item C — Retrospective assessment pipeline | the working pipeline that produces Item B's artifacts from a real customer data package | next; depends on Item B being stable |
| Item D — Outreach tooling | qualification scorecard as code, discovery script, one-pager | next; depends on Item B being stable |

Runtime batch 2, **Items 6 and 7 are explicitly deferred** (see below):
robocorectl (the operational CLI) and the Phase 1→2 hardening checklist.
They are recorded here as DEFERRED — post-first-signed-assessment.

## Deferred until first signed assessment

| Deferred item | Why it waits (one line) |
| --- | --- |
| **robocorectl** (operational CLI; runtime batch 2, Item 6) | Operational infrastructure before the first signed customer is the single largest failure mode available to this project; assessments run fine from the acceptance scripts. |
| **Phase 1→2 hardening checklist** (runtime batch 2, Item 7) | Phase 2 hardening is only meaningful after Phase 1 — a signed, delivered, published assessment — proves the product shape; hardening an unused system optimizes the wrong thing. |
| **Additional cognitive/prompt/safety infrastructure** | More runtime intelligence does not move the first signed assessment forward; the retrospective path needs templates and discipline, not more autonomy. |
| **Custom hardware support** | One standard 6/7-DoF cell profile keeps the assessment repeatable; custom hardware multiplies engineering cost before any repeatable value exists. |
| **Multi-industry expansion** | The two-result engine requires the second assessment to cost less than the first; that only holds within one vertical long enough to reuse the templates — multi-industry expansion before that breaks the compounding. |
| **Any claim of autonomous improvement ownership** | The customer owns their improvement; the company sells verified evidence of it — overclaiming ownership is the fastest way to lose the trust the passport depends on. |

## Trigger to revisit

The deferred table is revisited exactly when an Evidence Passport has
been delivered against a signed Case-Study Agreement and its case
study is published. At that point the company has one independently
verified result, and the only question worth asking is what reduces
the cost of the second one.
