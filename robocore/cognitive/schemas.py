"""robocore.cognitive.schemas -- strictly-typed Cognitive Plane contracts.

Deliverable 4 of the robocore Phase-1 runtime.  Everything the context
compiler emits, and every structured object the three LLM roles (planner,
critic, diagnoser) are allowed to emit, is defined here as a frozen Pydantic
model.  There are deliberately **no ``Dict[str, Any]`` escape hatches** in the
:class:`CompiledContext` tree: if a field is worth shipping to a model it is
worth a schema.

Additive extension note
-----------------------
Batch 1 froze ``robocore/contracts.py`` without a mutation catalog or any
cognitive-plane output models.  Rather than editing the frozen file, the
catalog, the classifier, the keyframe-pointer convention and the three LLM
output models live here (see the DEVIATIONS section of the batch report).
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from robocore.contracts import (
    FlightTrigger,
    HypothesisStatus,
    StratumStats,
)

__all__ = [
    "MutationType",
    "MUTATION_CATALOG",
    "next_in_rotation",
    "classify_mutation",
    "derive_keyframe_hash",
    "ContextOverflow",
    "ChampionSummary",
    "StratumRegression",
    "FailureClusterEntry",
    "FailureCluster",
    "ProposalDigest",
    "ForceExploration",
    "CompiledContext",
    "RefusalReason",
    "Refusal",
    "CriticDecision",
    "CriticVerdict",
    "FailureDiagnosis",
    "ensure_keyframe_refs_subset",
    "MAX_DIAGNOSIS_WORDS",
]


# ---------------------------------------------------------------------------
# Mutation catalog -- the fixed exploration rotation
# ---------------------------------------------------------------------------


class MutationType(str, Enum):
    """The closed catalog of mutation families the planner may propose.

    One proposal carries exactly one family (the planner prompt enforces
    "one mutation per proposal").  The catalog order below is the *fixed
    rotation* used by :class:`ForceExploration`: when the anti-loop detector
    fires, the planner is pushed to the next family in this order, wrapping
    around.  The order is deliberate: it walks from reward shaping (cheapest
    to falsify) through curriculum and physics randomization to observation
    and action pipelines, ending at termination/limits (most safety-critical,
    visited last).
    """

    REWARD_SCALING = "REWARD_SCALING"
    CURRICULUM_SHIFT = "CURRICULUM_SHIFT"
    DOMAIN_RAND_PERTURBATION = "DOMAIN_RAND_PERTURBATION"
    OBSERVATION_NOISE = "OBSERVATION_NOISE"
    ACTION_SMOOTHING = "ACTION_SMOOTHING"
    TERMINATION_TIGHTENING = "TERMINATION_TIGHTENING"


#: The fixed rotation.  Never reorder: the rotation is part of the frozen
#: anti-loop contract between the critic guard and the planner.
MUTATION_CATALOG: tuple[MutationType, ...] = (
    MutationType.REWARD_SCALING,
    MutationType.CURRICULUM_SHIFT,
    MutationType.DOMAIN_RAND_PERTURBATION,
    MutationType.OBSERVATION_NOISE,
    MutationType.ACTION_SMOOTHING,
    MutationType.TERMINATION_TIGHTENING,
)

#: Diff target path (substring, matched against ``+++ b/<path>`` headers and
#: the body of the unified diff) -> family.  Evaluation order is fixed and
#: independent of line order inside the diff, so classification is a pure
#: function of the set of files the diff touches.
_DIFF_TARGET_FAMILY: tuple[tuple[str, MutationType], ...] = (
    ("reward.py", MutationType.REWARD_SCALING),
    ("curriculum.yaml", MutationType.CURRICULUM_SHIFT),
    ("domain_rand.yaml", MutationType.DOMAIN_RAND_PERTURBATION),
    ("obs_encoder.py", MutationType.OBSERVATION_NOISE),
    ("observation", MutationType.OBSERVATION_NOISE),
    ("action_filter.py", MutationType.ACTION_SMOOTHING),
    ("control_policy.py", MutationType.ACTION_SMOOTHING),
    ("harness.yaml", MutationType.TERMINATION_TIGHTENING),
)

#: Residual family for diffs that touch none of the canonical targets
#: (e.g. a brand-new file): treated as an environment/termination change,
#: the most conservative bucket.
_DEFAULT_FAMILY = MutationType.TERMINATION_TIGHTENING


def classify_mutation(harness_diff: str) -> MutationType:
    """Deterministically map a unified harness diff to one mutation family.

    Classification keys on *which harness files the diff touches*, in the
    fixed priority order of ``_DIFF_TARGET_FAMILY`` -- never on diff text
    order, timestamps or identifiers, so the same diff always classifies
    identically across processes and replays.
    """
    for target, family in _DIFF_TARGET_FAMILY:
        if target in harness_diff:
            return family
    return _DEFAULT_FAMILY


def next_in_rotation(family: MutationType) -> MutationType:
    """Next family in the fixed catalog rotation (wraps around)."""
    idx = MUTATION_CATALOG.index(family)
    return MUTATION_CATALOG[(idx + 1) % len(MUTATION_CATALOG)]


def derive_keyframe_hash(snapshot_hash: str, frame_index: int) -> str:
    """Content-addressed pointer at one video keyframe of a black-box dump.

    The flight recorder never ships image bytes into the cognitive context;
    the diagnoser and the planner exchange *pointers* only.  The pointer is
    derived from the immutable snapshot hash plus the frame index inside its
    sealed window, so any consumer can re-derive the exact frame without a
    lookup table.
    """
    digest = hashlib.sha256(
        f"{snapshot_hash}:keyframe:{int(frame_index)}".encode("utf-8")
    ).hexdigest()[:24]
    return f"kf_{digest}"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContextOverflow(Exception):
    """The rendered cognitive context exceeds the token budget.

    Raised (never warned) by :meth:`ContextBuilder.build` when even the
    irreducible core -- champion summary plus anti-loop signal -- does not
    fit, or when the final assembled context would exceed the budget.
    """

    def __init__(self, budget: int, rendered_tokens: int, detail: str) -> None:
        super().__init__(
            f"context overflow: rendered {rendered_tokens} tokens against a"
            f" budget of {budget} ({detail})"
        )
        self.budget = budget
        self.rendered_tokens = rendered_tokens
        self.detail = detail


# ---------------------------------------------------------------------------
# Compiled context models (ContextBuilder output shape)
# ---------------------------------------------------------------------------


class StratumRegression(BaseModel):
    """One difficulty stratum where the champion is degrading post-promotion."""

    model_config = ConfigDict(frozen=True)

    stratum: str  # DifficultyStrataBin value
    promoted_rate: float = Field(ge=0.0, le=1.0)
    recent_rate: float = Field(ge=0.0, le=1.0)
    delta: float = Field(ge=-1.0, le=1.0)  # recent - promoted (<= 0 here)
    n_promoted: int = Field(ge=0)
    n_recent: int = Field(ge=0)
    ledger_seq: int = Field(ge=1)


class ChampionSummary(BaseModel):
    """Performance card of the harness the planner must diff against."""

    model_config = ConfigDict(frozen=True)

    task_name: str
    harness_hash: str
    policy_checkpoint_hash: str
    batch_id: str
    wilson_lower_bound: float = Field(ge=0.0, le=1.0)
    stratum_stats: List[StratumStats] = Field(default_factory=list)
    regressions: List[StratumRegression] = Field(default_factory=list)
    promoted_at_unix: float = 0.0
    ledger_seq: int = Field(ge=1)


class FailureClusterEntry(BaseModel):
    """One black-box dump inside a failure cluster (hashes only, no bytes)."""

    model_config = ConfigDict(frozen=True)

    failure_label: str
    snapshot_hash: str
    rollout_id: str
    keyframe_hash: str
    vlm_summary: str  # one-paragraph VLM diagnosis of the failure
    trigger: FlightTrigger
    harness_hash: Optional[str] = None
    occurred_at_unix: float = 0.0
    ledger_seq: int = Field(ge=1)


class FailureCluster(BaseModel):
    """All recent physical failures sharing one VLM failure label."""

    model_config = ConfigDict(frozen=True)

    failure_label: str
    entries: List[FailureClusterEntry] = Field(default_factory=list)  # cap 5
    total_observed: int = Field(ge=0)  # before cap/dedup, for honest sizing
    ledger_seq: int = Field(ge=1)  # max seq across kept entries


class ProposalDigest(BaseModel):
    """Compact record of a recent hypothesis decision (accepted or rejected)."""

    model_config = ConfigDict(frozen=True)

    hypothesis_id: str
    status: HypothesisStatus
    mutation: MutationType
    critic_score: float
    rationale_excerpt: str
    ledger_seq: int = Field(ge=1)


class ForceExploration(BaseModel):
    """Anti-loop directive: the planner is pushed off an exhausted family."""

    model_config = ConfigDict(frozen=True)

    catalog_family: MutationType  # the family to explore next
    rejected_streak: int = Field(ge=1)
    rejected_families: List[MutationType] = Field(default_factory=list)
    reason: str


class CompiledContext(BaseModel):
    """The exact shape ``ContextBuilder.build`` returns.

    Field order matches the prioritization contract: champion performance,
    recent physical failures, stratum regressions (inside the champion
    card), recent proposals, then the anti-loop signal, token accounting and
    the ledger sequence numbers of every source entry actually included.
    """

    model_config = ConfigDict(frozen=True)

    champion_summary: Optional[ChampionSummary] = None
    failure_clusters: List[FailureCluster] = Field(default_factory=list)
    recent_proposals: List[ProposalDigest] = Field(default_factory=list)
    anti_loop_signal: Optional[ForceExploration] = None
    token_estimate: int = Field(ge=0)
    source_entry_seqs: List[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM output models (planner / critic / diagnoser)
# ---------------------------------------------------------------------------


class RefusalReason(str, Enum):
    """The only legal reasons a cognitive role may decline to answer."""

    INSUFFICIENT_SIGNAL = "INSUFFICIENT_SIGNAL"
    ANTI_LOOP_WITHOUT_OVERRIDE = "ANTI_LOOP_WITHOUT_OVERRIDE"
    TOKEN_BUDGET_EXHAUSTED = "TOKEN_BUDGET_EXHAUSTED"


class Refusal(BaseModel):
    """Typed refusal.  A role that cannot answer must say so structurally."""

    model_config = ConfigDict(frozen=True)

    refusal_reason: RefusalReason
    detail: str = ""


class CriticDecision(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"


class CriticVerdict(BaseModel):
    """Output contract of ``critic_v1``.

    The critic is the closed-loop gate: it must emit a binary decision, a
    human-readable reason, an explicit anti-loop flag and a calibrated
    expected-information-gain in [0, 1].
    """

    model_config = ConfigDict(frozen=True)

    decision: CriticDecision
    reason: str
    anti_loop_flagged: bool = False
    expected_info_gain: float = Field(ge=0.0, le=1.0)
    mutation_family: Optional[MutationType] = None
    force_exploration: Optional[ForceExploration] = None


#: Hard word ceiling for diagnoser summaries (spec: <= 400 words).
MAX_DIAGNOSIS_WORDS: int = 400


class FailureDiagnosis(BaseModel):
    """Output contract of ``diagnoser_v1``.

    ``summary`` is capped at 400 words; ``keyframe_refs`` must be a subset of
    the keyframe hashes supplied in the prompt (checked by
    :func:`ensure_keyframe_refs_subset` at the call site, where the legal
    input set is known).
    """

    model_config = ConfigDict(frozen=True)

    failure_label: str
    summary: str
    keyframe_refs: List[str] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    snapshot_hash: Optional[str] = None
    trigger: Optional[FlightTrigger] = None

    @field_validator("summary")
    @classmethod
    def _summary_word_cap(cls, v: str) -> str:
        n_words = len(v.split())
        if n_words > MAX_DIAGNOSIS_WORDS:
            raise ValueError(
                f"diagnosis summary is {n_words} words; the hard cap is"
                f" {MAX_DIAGNOSIS_WORDS}"
            )
        return v


def ensure_keyframe_refs_subset(
    diagnosis: FailureDiagnosis, input_keyframe_hashes: List[str]
) -> None:
    """Raise ``ValueError`` unless every keyframe ref came from the prompt.

    The diagnoser may only reference keyframes it was actually shown; this
    is the guard against hallucinated evidence pointers.
    """
    legal = set(input_keyframe_hashes)
    bad = [ref for ref in diagnosis.keyframe_refs if ref not in legal]
    if bad:
        raise ValueError(
            f"diagnosis references keyframes not present in the prompt:"
            f" {bad}"
        )
