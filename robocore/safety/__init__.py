"""robocore.safety -- harness-diff application, AST guarding, physical
envelopes.

Additive extension module for Batch 2, Item 5: the subject-under-test of
the hypothesis fuzzing & AST verification suite (Batch 1 shipped the
``SafetyValidationReceipt`` contract but no validator, diff engine or
physical limits -- see the batch report's DEVIATIONS section).
"""

from robocore.safety.ast_guard import (
    ALLOWED_IMPORTS,
    AST_GUARD_VERSION,
    ASTSafetyValidationReceipt,
    FORBIDDEN_IMPORTS,
    SAFE_BUILTINS,
    SAFE_EXCEPTIONS,
    ast_validator,
    precheck_scope,
    validate_patch,
    validate_source,
)
from robocore.safety.diff_engine import DiffApplyResult, apply_diff, unified_diff_text
from robocore.safety.envelope import (
    ENVELOPE_OVERRIDE_KEYS,
    PHYSICAL_LIMITS,
    PhysicalEnvelope,
    PhysicalLimits,
    envelope_generator,
)

__all__ = [
    "ALLOWED_IMPORTS",
    "AST_GUARD_VERSION",
    "ASTSafetyValidationReceipt",
    "FORBIDDEN_IMPORTS",
    "SAFE_BUILTINS",
    "SAFE_EXCEPTIONS",
    "ast_validator",
    "precheck_scope",
    "validate_patch",
    "validate_source",
    "DiffApplyResult",
    "apply_diff",
    "unified_diff_text",
    "ENVELOPE_OVERRIDE_KEYS",
    "PHYSICAL_LIMITS",
    "PhysicalEnvelope",
    "PhysicalLimits",
    "envelope_generator",
]
