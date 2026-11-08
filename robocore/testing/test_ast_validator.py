"""test_ast_validator.py -- adversarial + baseline corpus tests, receipt
contract, pre-AST scope gate, and the property-based fuzz invocation.

Every test is a plain pytest function and is also called directly by
``scripts/smoke_ast_fuzzing.py`` (the two-stage acceptance gate).
"""

from __future__ import annotations

from typing import Dict

from robocore.contracts import SafetyValidationReceipt
from robocore.safety import (
    ASTSafetyValidationReceipt,
    DiffApplyResult,
    apply_diff,
    precheck_scope,
    validate_patch,
    validate_source,
)
from robocore.testing.hypothesis_fuzzer import (
    BASELINE_HARNESS_FILES,
    load_baseline_corpus,
    load_evasion_corpus,
    materialize_baseline_corpus,
    materialize_evasion_corpus,
    run_all_fuzz,
    TOTAL_FUZZ_ITERATIONS,
)


def test_adversarial_corpus_fully_rejected() -> Dict[str, int]:
    """0 false negatives: every evasion file is rejected."""
    materialize_evasion_corpus()
    cases = load_evasion_corpus()
    assert len(cases) >= 50, f"corpus must hold >= 50 files, has {len(cases)}"
    rejected = 0
    for case in cases:
        receipt = validate_source(case.source)
        if receipt.ast_passed:
            raise AssertionError(
                f"FALSE NEGATIVE: {case.name} was accepted by the validator"
            )
        rejected += 1
    assert rejected == len(cases)
    return {"adversarial_files": len(cases), "rejected": rejected}


def test_baseline_corpus_fully_accepted() -> Dict[str, int]:
    """0 false rejections: every legitimate patch applies and validates."""
    materialize_baseline_corpus()
    cases = load_baseline_corpus()
    assert len(cases) >= 50, f"corpus must hold >= 50 patches, has {len(cases)}"
    accepted = 0
    for case in cases:
        result, receipt = validate_patch(
            BASELINE_HARNESS_FILES, case.patch_text
        )
        if not result.ok:
            raise AssertionError(
                f"FALSE REJECTION (diff engine): {case.name}:"
                f" {result.rejects[:2]}"
            )
        if not receipt.passed:
            raise AssertionError(
                f"FALSE REJECTION (ast guard): {case.name}:"
                f" {receipt.violations[:2]}"
            )
        accepted += 1
    assert accepted == len(cases)
    return {"baseline_patches": len(cases), "accepted": accepted}


def test_receipt_contract_on_every_rejection() -> Dict[str, int]:
    """Every rejection is a SafetyValidationReceipt with ast_passed=False
    and a non-empty, annotated ast_errors list."""
    cases = load_evasion_corpus()
    checked = 0
    for case in cases:
        receipt = validate_source(case.source)
        assert isinstance(receipt, SafetyValidationReceipt)
        assert isinstance(receipt, ASTSafetyValidationReceipt)
        assert receipt.ast_passed is False
        assert isinstance(receipt.ast_errors, list)
        assert len(receipt.ast_errors) > 0, (
            f"{case.name}: rejection without reasons"
        )
        # the annotated @reject_reason must appear in the error list
        assert any(
            case.expected_reason in err for err in receipt.ast_errors
        ), (
            f"{case.name}: expected reason {case.expected_reason!r} not in"
            f" {receipt.ast_errors[:3]}"
        )
        checked += 1
    # malformed diffs also produce receipts-with-reasons (not exceptions)
    bad_receipts = 0
    for corrupted in (
        "",
        "not a diff at all\n",
        "--- a/reward.py\n+++ b/reward.py\n",
        "--- a/reward.py\n+++ b/reward.py\n@@ -1,x +1,y @@\n-z\n",
        "--- a/reward.py\n+++ b/reward.py\n@@ -1,3 +1,3 @@\n nope\n",
    ):
        result, receipt = validate_patch(BASELINE_HARNESS_FILES, corrupted)
        assert isinstance(result, DiffApplyResult)
        assert not result.ok or not receipt.passed
        if not receipt.passed:
            assert len(receipt.ast_errors) > 0
            bad_receipts += 1
    return {"receipts_checked": checked, "malformed_receipts": bad_receipts}


def test_scope_creep_rejected_pre_ast() -> Dict[str, int]:
    """sys._getframe scope creep is rejected BEFORE any AST is built; the
    specific reason string is present in the receipt's ast_errors."""
    source = (
        "import sys\n"
        "\n"
        "\n"
        "def leaky():\n"
        "    frame = sys._getframe(1)\n"
        "    return frame.f_globals\n"
    )
    scope_errors = precheck_scope(source)
    assert any(
        "frame introspection" in e and "sys._getframe" in e
        for e in scope_errors
    ), scope_errors
    receipt = validate_source(source)
    assert receipt.ast_passed is False
    assert any(
        "pre-ast scope violation" in e and "sys._getframe" in e
        for e in receipt.ast_errors
    ), receipt.ast_errors
    # proof the rejection happened at the scope gate, not the AST walk:
    # the scope layer is the only one that emits this prefix
    assert receipt.static_analysis_summary.startswith(
        "ast_guard"
    ) and "pre-ast scope gate" in receipt.static_analysis_summary
    return {"scope_violations": len(scope_errors)}


def test_fuzz_properties() -> Dict[str, int]:
    """All hypothesis properties: >= 10,000 iterations, zero unhandled
    exceptions, zero violations."""
    outcome = run_all_fuzz()
    assert not outcome.violations, outcome.violations[:5]
    assert outcome.iterations >= 10_000, (
        f"only {outcome.iterations} fuzz iterations executed"
    )
    assert outcome.iterations >= TOTAL_FUZZ_ITERATIONS - 1_500, (
        "iteration budget collapsed -- check property registration"
    )
    return {"iterations": outcome.iterations, "violations": len(outcome.violations)}
