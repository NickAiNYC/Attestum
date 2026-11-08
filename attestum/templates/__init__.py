"""Contractual template models and their pure renderers.

Re-exports the three Pydantic models (Evidence Passport, Economic
Case, Case-Study Agreement) and the renderer functions.  Import from
here; do not reach into submodules.
"""

from __future__ import annotations

from attestum.templates.case_study_agreement import (
    ANONYMIZATION_OPTIONS,
    AnonymizationMethod,
    CaseStudyAgreement,
    Clause,
    ClauseElection,
    build_clauses,
)
from attestum.templates.economic_case import (
    EconomicCase,
    InterventionCostBreakdown,
)
from attestum.templates.evidence_passport import (
    SECTION_FIELDS,
    SECTION_TITLES,
    ClearNegative,
    EvidencePassport,
    ImprovementResult,
    NoGo,
    VerifiedImprovement,
)
from attestum.templates.renderer import (
    UnresolvedSlotError,
    render_case_study_agreement,
    render_economic_case,
    render_evidence_passport,
    render_markdown_template,
)

__all__ = [
    "ANONYMIZATION_OPTIONS",
    "AnonymizationMethod",
    "CaseStudyAgreement",
    "Clause",
    "ClauseElection",
    "ClearNegative",
    "EconomicCase",
    "EvidencePassport",
    "ImprovementResult",
    "InterventionCostBreakdown",
    "NoGo",
    "SECTION_FIELDS",
    "SECTION_TITLES",
    "UnresolvedSlotError",
    "VerifiedImprovement",
    "build_clauses",
    "render_case_study_agreement",
    "render_economic_case",
    "render_evidence_passport",
    "render_markdown_template",
]
