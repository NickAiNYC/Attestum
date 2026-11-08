"""robocore.testing -- hypothesis fuzzing & AST verification suite.

Deliverable 5 of the robocore Phase-1 runtime.  The ``test_*`` modules are
plain pytest functions AND importable by ``scripts/smoke_ast_fuzzing.py``
(the acceptance gate).  The two corpora live in ``ast_evasions/`` and
``baseline_harnesses/`` and are materialized deterministically by
:mod:`robocore.testing.hypothesis_fuzzer`.
"""

from robocore.testing.hypothesis_fuzzer import (
    BASELINE_HARNESS_FILES,
    EVASION_TEMPLATES,
    FuzzOutcome,
    load_baseline_corpus,
    load_evasion_corpus,
    make_valid_diff,
    materialize_baseline_corpus,
    materialize_evasion_corpus,
    run_all_fuzz,
)

__all__ = [
    "BASELINE_HARNESS_FILES",
    "EVASION_TEMPLATES",
    "FuzzOutcome",
    "load_baseline_corpus",
    "load_evasion_corpus",
    "make_valid_diff",
    "materialize_baseline_corpus",
    "materialize_evasion_corpus",
    "run_all_fuzz",
]
