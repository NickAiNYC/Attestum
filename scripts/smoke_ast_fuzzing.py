#!/usr/bin/env python3
"""smoke_ast_fuzzing.py -- acceptance script for robocore Deliverable 5
(hypothesis fuzzing & AST verification suite).

Five checks, each raising on failure (fail-closed; no PASS line is printed
unless every check passed):

  1. Adversarial corpus: 100% rejected -- 0 false negatives across all
     ``# @reject_reason``-annotated evasion files.
  2. Baseline corpus: 100% accepted -- 0 false rejections across all
     legitimate harness patches.
  3. Property fuzzing: >= 10,000 hypothesis iterations complete with zero
     unhandled exceptions and zero violations (apply_diff, ast_validator,
     envelope_generator).
  4. Receipt contract: every rejection yields
     ``SafetyValidationReceipt(ast_passed=False, ast_errors=[...])`` with
     non-empty, reason-annotated errors.
  5. Critic anti-loop suite: 3 same-family rejections -> forced exploration
     (asserted on CriticVerdict); 5 consecutive rejections -> PAUSED_ALERT
     (asserted via the orchestrator FSM); sys._getframe scope creep ->
     rejected pre-AST with the specific reason in ast_errors.

Run: python scripts/smoke_ast_fuzzing.py   ->   "PASS 5/5"
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from robocore.testing.hypothesis_fuzzer import (
    load_baseline_corpus,
    load_evasion_corpus,
    materialize_baseline_corpus,
    materialize_evasion_corpus,
)
from robocore.testing.test_ast_validator import (
    test_adversarial_corpus_fully_rejected,
    test_baseline_corpus_fully_accepted,
    test_fuzz_properties,
    test_receipt_contract_on_every_rejection,
    test_scope_creep_rejected_pre_ast,
)
from robocore.testing.test_critic_anti_loop import (
    test_five_consecutive_rejections_pause_alert,
    test_scope_creep_rejected_pre_ast as test_scope_creep_via_critic,
    test_three_same_family_rejections_force_exploration,
)


class _Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.total = 0

    def check(self, name: str, fn) -> None:  # noqa: ANN001
        self.total += 1
        result = fn()  # raises on failure
        self.passed += 1
        print(f"  ok  {name} {result}")


def main() -> int:
    checks = _Checks()

    # ---- check 1: adversarial corpus, 0 FN --------------------------------
    n_ev = materialize_evasion_corpus()
    assert n_ev >= 50, f"evasion corpus has {n_ev} files (need >= 50)"
    checks.check(
        f"adversarial corpus fully rejected (0 FN over"
        f" {len(load_evasion_corpus())} files)",
        test_adversarial_corpus_fully_rejected,
    )

    # ---- check 2: baseline corpus, 0 FP ------------------------------------
    n_bl = materialize_baseline_corpus()
    assert n_bl >= 50, f"baseline corpus has {n_bl} patches (need >= 50)"
    checks.check(
        f"baseline corpus fully accepted (0 FP over"
        f" {len(load_baseline_corpus())} patches)",
        test_baseline_corpus_fully_accepted,
    )

    # ---- check 3: >= 10,000 fuzz iterations --------------------------------
    checks.check(
        "property fuzzing (apply_diff / ast_validator /"
        " envelope_generator)",
        test_fuzz_properties,
    )

    # ---- check 4: receipt contract on every rejection ----------------------
    checks.check(
        "SafetyValidationReceipt(ast_passed=False, ast_errors=[...]) on"
        " every rejection",
        test_receipt_contract_on_every_rejection,
    )

    # ---- check 5: critic anti-loop suite ------------------------------------
    def _anti_loop_suite() -> dict:
        forced = test_three_same_family_rejections_force_exploration()
        paused = test_five_consecutive_rejections_pause_alert()
        scope_critic = test_scope_creep_via_critic()
        scope_ast = test_scope_creep_rejected_pre_ast()
        return {
            "forced_exploration": forced,
            "paused_alert": paused,
            "scope_pre_ast": {
                **scope_critic,
                "reason_in_ast_errors": scope_ast["scope_violations"],
            },
        }

    checks.check(
        "critic anti-loop: 3 same-family -> ForceExploration;"
        " 5 consecutive -> PAUSED_ALERT via orchestrator FSM;"
        " sys._getframe scope creep rejected pre-AST",
        _anti_loop_suite,
    )

    print(f"PASS {checks.passed}/{checks.total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
