# robocore

**Unattended closed-loop embodied-AI runtime** for a single 6/7-DoF robot arm.

The loop is fully autonomous: propose harness mutations → compile behind an AST
validator → evaluate in MuJoCo → verify on the physical scene → execute → gate
with statistics → promote or reject → commit to the experiment ledger — no human
in the loop, with crash recovery and tamper detection on every path.

```
PROPOSING → COMPILING_AND_PRECHECKING → SIM_EVALUATING → SYSID_CALIBRATING
          → PHYSICAL_SCENE_VERIFYING → PHYSICAL_EXECUTING → PHYSICAL_RESETTING
          → EVALUATING_PROMOTION → COMMITTING_LEDGER → (promote | reject | alert)
```

## Architecture

| Module | Responsibility |
| --- | --- |
| `robocore/contracts.py` | Pydantic contract layer + canonical harness hashing |
| `robocore/stats.py` | Wilson score intervals, bootstrap CIs, SPRT, stratified `PromotionGate` |
| `robocore/storage/ledger_repo.py` | 10-table SQLite WAL experiment ledger: hash-chained events, FK integrity, triggers, state FSM, analytical views |
| `robocore/telemetry/flight_recorder.py` | 1 kHz ring-buffer flight recorder, content-addressed Parquet black-box dumps, MuJoCo replay |
| `robocore/runtime/orchestrator.py` | 13-state orchestration FSM, WAL checkpointing, four partial-failure paths (incl. crash recovery and chain-tamper refusal) |
| `robocore/cognitive/` | Token-budgeted `CompiledContext` builder, sha256 `PromptRegistry`, anti-loop `ForceExploration` signal |
| `robocore/safety/` | AST guard (forbidden imports, dunder abuse, metaclass injection, `eval`/`exec`/`compile`), diff engine, safety envelope generator |
| `robocore/testing/` | 58-file adversarial AST corpus, 71-patch baseline corpus, Hypothesis property fuzzer |

## Cognitive layer (Item 4)

- `ContextBuilder.build(...)` compiles a **strictly typed** `CompiledContext` —
  no `dict[str, Any]` escape hatches.
- **Hard token budget** with cl100k_base counting (vocabulary vendored under
  `robocore/cognitive/data/`); over-budget raises `ContextOverflow`.
- **Deterministic priority**: recent physical failures > stratified regressions >
  recent proposals (accepted and rejected) > older entries; ties broken by
  ledger sequence number; entries deduplicated by harness hash.
- **Failure clusters** group by failure label (≤ 5 keyframes per label, hashes
  only — no image bytes ever enter the prompt).
- **Anti-loop**: 3 consecutive same-family rejections inject
  `ForceExploration(catalog_family=next)`; 5 consecutive escalate the
  orchestrator to `PAUSED_ALERT`.
- **PromptRegistry**: prompts register by sha256 at load time; unregistered
  files are refused; every `{{slot}}` marker must be consumed at render time.

## Safety layer (Item 5)

- **Adversarial corpus** — 58 files covering dangerous imports, dunder abuse,
  metaclass injection, `eval`/`exec`/`compile`, `__import__`,
  getattr-concatenation chains, `sys.modules`/`builtins` mutation, pickle,
  thread creation: **all rejected, 0 false negatives**.
- **Baseline corpus** — 71 legitimate harness patches:
  **all accepted, 0 false positives**.
- **Property fuzzing** — 13,000 Hypothesis iterations over `apply_diff`,
  `ast_validator`, and `envelope_generator`: **0 invariant violations**.
- Every rejection emits a structured `SafetyValidationReceipt`; the anti-loop
  assertion spans three levels: `CriticVerdict` → orchestrator FSM →
  `SafetyValidationReceipt`.

## Verification

Four acceptance scripts, each printing a greppable one-line result:

| Script | Gate |
| --- | --- |
| `python scripts/smoke_robocore.py` | 38/38 checks: unattended loop, ledger chain, 1 kHz recorder, auto-SysID, recovery drills D1/D2 |
| `python scripts/verify_mujoco_backend.py` | MuJoCo replay calibration (calibrated error 0.000000) |
| `python scripts/smoke_context_builder.py` | PASS 6/6 — Item 4 acceptance |
| `python scripts/smoke_ast_fuzzing.py` | PASS 5/5 — Item 5 acceptance |

Engineering constraints held throughout: no `pass`/`NotImplementedError`/
`TODO` placeholders, and every function is executed at least once by the
smoke scripts.

## Install

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[mujoco,dev]"
```

Run the acceptance suite:

```bash
python scripts/smoke_robocore.py
python scripts/verify_mujoco_backend.py
python scripts/smoke_context_builder.py
python scripts/smoke_ast_fuzzing.py
```

## Status

- **Batch 1** — contracts, statistics engine, WAL ledger, flight recorder,
  orchestrator: delivered, frozen, verified (38/38).
- **Batch 2** — cognitive context compilation (Item 4) and property-fuzzed AST
  safety validation (Item 5): delivered, verified (6/6 and 5/5).
- **Batch 3** — Items 6–7: pending.

## License

Apache-2.0 — see [LICENSE](LICENSE).
