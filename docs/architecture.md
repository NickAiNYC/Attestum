# Attestum — Architecture

Three planes, one direction of travel: data flows up from the
customer's physical cell, through the open-source measurement runtime,
into the commercial evidence layer. Nothing in the upper planes ever
reaches back down into production.

```
+---------------------------------------------------------------+
|  EVIDENCE PLANE                                    attestum/  |
|                                                               |
|  Evidence Passport - Economic Case - Case-Study Agreement     |
|  (Pydantic models + pure renderers; hash-verified artifacts)  |
+-----------------------------^---------------------------------+
                              |  verified artifacts
                              |  (markdown for the customer,
                              |   JSON for the ledger)
+-----------------------------+---------------------------------+
|  MEASUREMENT PLANE                               robocore/    |
|                                                               |
|  hash-chained WAL ledger - Wilson/SPRT gates - 1 kHz flight   |
|  recorder - MuJoCo replay - AST-validated toolchain           |
+-----------------------------^---------------------------------+
                              |  retrospective data package
                              |  (Day 5: logs, video,
                              |   intervention records)
+-----------------------------+---------------------------------+
|  PHYSICAL PLANE                            the customer's cell |
|                                                               |
|  robot - controller - vision - intervention records - video   |
+---------------------------------------------------------------+
```

## The physical plane (the customer's)

The robot cell itself: a 6/7-DoF manipulator, its controller, its
vision system, and the records the site already keeps. This plane is
never accessed live. The customer exports a data package — controller
event logs, shop-floor video, intervention records, optionally the
maintenance log — and hands it over once, on Day 5 of the engagement.
Nothing Attestum ships ever executes on this plane; that constraint is
what makes the assessment safe to buy and cheap to deliver.

## The measurement plane (the open-source runtime)

The frozen runtime package, `robocore/`, is the trust substrate. It
gives the evidence plane four things:

| Runtime capability | What it lends the evidence |
| --- | --- |
| `storage/ledger_repo.py` — 10-table SQLite WAL ledger, hash-chained events | every analytical statement can be anchored to an append-only, tamper-evident record |
| `stats.py` — Wilson score intervals, bootstrap CIs, SPRT, stratified `PromotionGate` | the uncertainty numbers in an Evidence Passport or Economic Case are computed, never asserted — e.g. the Economic Case's confidence interval IS `robocore.stats.wilson_score_interval` on the observed improvement |
| `telemetry/flight_recorder.py` — 1 kHz ring buffer, content-addressed Parquet dumps, MuJoCo replay | interventions can be replayed and reconstructed offline from exported data, so "what actually happened" is checkable, not recalled |
| `safety/` + `testing/` — AST guard with a 58-file adversarial corpus and property fuzzing | the offline toolchain that processes customer data is itself validated against known evasion classes |

One dependency rule keeps the planes honest: the evidence layer
imports from the runtime's statistics and safety surface
(`robocore.stats`), never from its orchestration internals. The
runtime stays a general-purpose measurement engine; the commercial
layer stays a consumer of its numbers.

## The evidence plane (the commercial layer)

The `attestum/` package: the brand record (`attestum/brand.py`, the
single source of truth for the company name and contact), the three
contractual template models, and the pure renderers
(`attestum/templates/renderer.py`). Its artifacts are designed to be
verified the same way the runtime verifies its own records:

- **Evidence Passport** — eleven sections; `passport_hash` is the
  sha256 of the canonical JSON of the model minus the hash itself.
- **Economic Case** — every headline number is computed from
  challengeable inputs; the ledger JSON artifact carries its own
  `content_sha256` (destined for the ledger entry
  `evidence_passport_delivered` once that integration is unblocked —
  see the roadmap).
- **Case-Study Agreement** — eight clauses, each with a recommended
  and a fallback text, built from the brand record and the fee so the
  company name and the price each exist in exactly one place.

Renderers are pure functions: no filesystem writes, no network, no
LLM calls; identical inputs produce byte-identical outputs, which the
acceptance scripts assert.

## Interfaces between the planes

| Interface | Direction | Form |
| --- | --- | --- |
| Data package | physical → measurement | CSV/MP4/XLSX exports with sha256 hashes, delivered Day 5 |
| Statistics | measurement → evidence | `wilson_score_interval` and friends, called by `attestum/templates/economic_case.py` |
| Deliverables | evidence → physical | Evidence Passport, Economic Case, case-study draft, returned Day 15–21 |
| Case study | evidence → public | published under the agreement's clause 4 (≤ 60 days after delivery) |

## Where the acceptance scripts sit

The acceptance scripts (`scripts/`) are the contract between planes:
the four runtime scripts prove the measurement plane still behaves as
frozen; the two commercial scripts prove the evidence plane's
artifacts instantiate, compute, and render deterministically. All six
print a single greppable `PASS N/M` line.
