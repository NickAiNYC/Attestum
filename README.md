# Attestum

**Pre-launch: no customers or assessments delivered yet.**

The intelligence layer for continuously improving industrial automation.
Attestum is a company that sells verified evidence: the Evidence Passport, an independently checkable record that an industrial robot cell improved — by how much, at what cost, and with what uncertainty.

## What we do

We run retrospective assessments of robotic work cells using only data our customers already have: controller logs, shop-floor video, and intervention records. There is no live access to the cell, no safety review, and no change to production. Twenty-one days after signing, the customer holds an Evidence Passport, an Economic Case, and a draft case study.

## What we sell

The Evidence Passport is the product. It has eleven sections — cell and robot identification, data provenance, frozen evaluation conditions, statistical uncertainty, safety oversight, reproducible artifacts and hashes — and every number in it traces back to a hash the customer can verify. The outcome is a ternary, not a binary: verified improvement, clear negative, or no-go. A clear negative is a publishable and chargeable result; telling a customer what not to spend software money on is worth paying for.

## How we price

The assessment is free when the case study is published. If the customer withdraws before publication, the fee is retroactively invoiced at $20,000 USD. This keeps the free path honest: it is taken by customers willing to stand behind the published numbers — exactly the customers whose passports are worth reading.

## What we deliberately do not do

Until the first signed assessment is delivered and published, we build no operational infrastructure: no operational CLI, no dashboards, no multi-industry coverage, no custom hardware support, no ownership claim over a customer's improvement. The deferred list, with one-line reasons, lives in [docs/roadmap.md](docs/roadmap.md); the strategy summary lives in [ATTESTUM_STRATEGY.md](ATTESTUM_STRATEGY.md).

## Open-source runtime

robocore is the open-source measurement runtime that makes our evidence reproducible. It is not the product — it is the reason the product can be trusted: hash-chained experiment ledgers, a 1 kHz flight recorder, Wilson/SPRT statistical gates, and an AST validator that refuses unsafe generated code. Assessments run through it offline, on exported customer data only.

| Module | Responsibility |
| --- | --- |
| `robocore/contracts.py` | Pydantic contract layer + canonical harness hashing |
| `robocore/stats.py` | Wilson score intervals, bootstrap CIs, SPRT, stratified `PromotionGate` |
| `robocore/storage/ledger_repo.py` | 10-table SQLite WAL experiment ledger: hash-chained events, FK integrity, triggers, state FSM, analytical views |
| `robocore/telemetry/flight_recorder.py` | 1 kHz ring-buffer flight recorder, content-addressed Parquet black-box dumps, MuJoCo replay |
| `robocore/runtime/orchestrator.py` | 13-state orchestration FSM, WAL checkpointing, partial-failure recovery paths (crash recovery, chain-tamper refusal) |
| `robocore/cognitive/` | Token-budgeted `CompiledContext` builder, sha256 `PromptRegistry`, anti-loop `ForceExploration` |
| `robocore/safety/` | AST guard (forbidden imports, dunder abuse, metaclass injection, `eval`/`exec`/`compile`), diff engine, safety envelope generator |
| `robocore/testing/` | 58-file adversarial AST corpus, 71-patch baseline corpus, Hypothesis property fuzzer |

See [docs/architecture.md](docs/architecture.md) for the three-plane view of the business and how the runtime fits under the commercial layer.

## Repository layout

| Path | Layer | Contents |
| --- | --- | --- |
| `attestum/` | commercial | brand record (single source of truth), contractual template models, pure renderers |
| `docs/` | both | strategy summary, roadmap, architecture, the fill-in document templates |
| `templates/` | commercial | how to fill each template, plus anonymized worked examples |
| `robocore/` | open-source runtime | frozen measurement runtime (batches 1–2, unchanged) |
| `scripts/` | both | acceptance scripts (below) |

## Acceptance gates

Every layer ships with acceptance scripts that print a single greppable `PASS N/M` line:

| Script | Layer | Gate |
| --- | --- | --- |
| `python scripts/smoke_robocore.py` | runtime | 38/38 checks: unattended loop, ledger chain, 1 kHz recorder, auto-SysID, recovery drills D1/D2 |
| `python scripts/verify_mujoco_backend.py` | runtime | MuJoCo replay calibration (calibrated error 0.000000) |
| `python scripts/smoke_context_builder.py` | runtime | PASS 6/6 — cognitive context compiler and prompt registry |
| `python scripts/smoke_ast_fuzzing.py` | runtime | PASS 5/5 — property-fuzzed AST safety validation |
| `python scripts/smoke_attestum_brand.py` | commercial | PASS 7/7 — brand reconciliation and repo hygiene |
| `python scripts/smoke_attestum_templates.py` | commercial | PASS 6/6 — contractual templates and renderers |

## Install

Requires Python >= 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mujoco,dev]"
```

Run the acceptance scripts from the repository root.

## Status

- **Runtime batch 1** — contracts, statistics engine, WAL ledger, flight recorder, orchestrator: delivered, frozen, verified (38/38).
- **Runtime batch 2** — cognitive context compilation and property-fuzzed AST safety: delivered, frozen, verified (6/6 and 5/5). Batch-2 items 6–7 (operational CLI, Phase 1→2 hardening checklist) are deferred until the first signed assessment.
- **Commercial layer** — brand reconciliation (Item A) and the three contractual templates plus retrospective scope (Item B): delivered. The retrospective assessment pipeline (Item C) and outreach tooling (Item D) follow.

## License

Apache-2.0 — see [LICENSE](LICENSE).
