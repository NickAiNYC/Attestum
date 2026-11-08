"""robocore.cognitive -- Cognitive Plane: context compiler & prompt registry.

Deliverable 4 of the robocore Phase-1 runtime.  Strictly-typed schemas
(:mod:`robocore.cognitive.schemas`), the ledger-derived context compiler
(:mod:`robocore.cognitive.context_builder`), the tamper-evident prompt
registry (:mod:`robocore.cognitive.prompt_registry`) and the anti-loop
critic guard (:mod:`robocore.cognitive.anti_loop`).
"""

from robocore.cognitive.anti_loop import AntiLoopCriticGuard, AntiLoopPolicy
from robocore.cognitive.context_builder import (
    ContextBuilder,
    TokenCounter,
    anti_loop_directive,
    build_critic_slots,
    build_diagnoser_slots,
    build_planner_slots,
    render_context_markdown,
)
from robocore.cognitive.prompt_registry import (
    DEFAULT_TOKEN_BUDGETS,
    PROMPTS_DIR,
    PromptRegistry,
    PromptRole,
    PromptSchemaMeta,
    SCHEMA_META,
    UnknownPromptVersionError,
    UnregisteredPromptError,
    UnresolvedSlotError,
)
from robocore.cognitive.schemas import (
    MAX_DIAGNOSIS_WORDS,
    ChampionSummary,
    CompiledContext,
    ContextOverflow,
    CriticDecision,
    CriticVerdict,
    FailureCluster,
    FailureClusterEntry,
    FailureDiagnosis,
    ForceExploration,
    MutationType,
    MUTATION_CATALOG,
    ProposalDigest,
    Refusal,
    RefusalReason,
    StratumRegression,
    classify_mutation,
    derive_keyframe_hash,
    ensure_keyframe_refs_subset,
    next_in_rotation,
)

__all__ = [
    "AntiLoopCriticGuard",
    "AntiLoopPolicy",
    "ContextBuilder",
    "TokenCounter",
    "anti_loop_directive",
    "build_critic_slots",
    "build_diagnoser_slots",
    "build_planner_slots",
    "render_context_markdown",
    "DEFAULT_TOKEN_BUDGETS",
    "PROMPTS_DIR",
    "PromptRegistry",
    "PromptRole",
    "PromptSchemaMeta",
    "SCHEMA_META",
    "UnknownPromptVersionError",
    "UnregisteredPromptError",
    "UnresolvedSlotError",
    "MAX_DIAGNOSIS_WORDS",
    "ChampionSummary",
    "CompiledContext",
    "ContextOverflow",
    "CriticDecision",
    "CriticVerdict",
    "FailureCluster",
    "FailureClusterEntry",
    "FailureDiagnosis",
    "ForceExploration",
    "MutationType",
    "MUTATION_CATALOG",
    "ProposalDigest",
    "Refusal",
    "RefusalReason",
    "StratumRegression",
    "classify_mutation",
    "derive_keyframe_hash",
    "ensure_keyframe_refs_subset",
    "next_in_rotation",
]
