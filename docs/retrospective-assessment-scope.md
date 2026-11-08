# Retrospective Assessment — Scope of Work

> {{company_name}} · assessment {{assessment_id}} · agreement {{agreement_id}} · customer {{customer_code}}.
> This is the two-page scope document referenced by clause 1 of the
> Case-Study Agreement. It defines exactly what each side provides,
> when, and what the engagement does and does not promise.

## 1. What the customer provides (Day 5 data package)

| Item | Format | Notes |
| --- | --- | --- |
| Robot controller event log | CSV export, UTC ISO-8601 timestamps | full window, unedited; sha256 recorded on receipt |
| Shop-floor video | MP4 (H.264), at least one camera angle covering the cell | continuous or event-triggered; sha256 recorded |
| Intervention records | CSV: timestamp, duration, cause code, operator note | as kept by the site today; no re-formatting required |
| Maintenance log (optional) | XLSX or CSV | used only to exclude maintenance-stop hours from windows |

## 2. What the customer does NOT provide

- No live access to the cell — no VPN, no remote desktop, no site visit.
- No safety review — the assessment touches no safety-relevant system.
- No PLC code, no network credentials, no OT access of any kind.
- No engineering time beyond one named reviewer for the Day 20 review call.

## 3. Deliverables and timeline

| Day | Milestone |
| --- | --- |
| Day 0 | agreement signed; data-request checklist sent |
| Day {{data_package_due}} | data package received and hash-verified |
| Day {{draft_due}} | draft deliverables: Evidence Passport + Economic Case |
| Day {{review_due}} | review call with the customer's named reviewer (challenge log worked line by line) |
| Day {{final_due}} | final deliverables + case-study draft |

## 4. Fee structure

The assessment fee is {{assessment_fee_usd}} USD list, **free when the
case study is published** under the agreement's clause 4. If the
customer withdraws before publication under clause 5, the fee converts
retroactively to {{retroactive_fee_usd}} USD (agreement clauses 2 and
5). There is no subscription, per-seat, or success fee.

## 5. The three possible outcomes

| Outcome | What it means for the customer |
| --- | --- |
| **Verified improvement** | the passport records a measured, statistically bounded improvement in the intervention rate, with reproducible artifacts |
| **Clear negative** | the assessment found the problem is mechanical (or otherwise not addressable by the runtime) — a publishable result that tells you what not to spend software money on |
| **No-go** | the evidence was insufficient to make a defensible claim either way; the passport records why, and what data would change that |

## 6. What {{company_name}} does not promise

- **Not a performance improvement.** The engagement measures and
  verifies; it does not guarantee an outcome, and the passport's
  numbers are reported with their uncertainty.
- **Not autonomous operation.** Nothing is deployed to, or executes
  on, the customer's cell; all analysis runs offline on the exported
  data package.
- **Not machinery safety certification.** This engagement is not a
  CE / ISO 13849 / RIA R15.06 assessment and does not replace one.

---

_{{company_name}} — {{contact_email}}. Questions on this scope go to
that address before signing, not after._
