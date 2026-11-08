"""robocore.cognitive.context_builder -- the Cognitive Plane context compiler.

Deliverable 4 of the robocore Phase-1 runtime.

``ContextBuilder.build`` compiles the ledger state around one champion
harness into a :class:`CompiledContext`: champion performance card (with
post-promotion stratum regressions), recent physical-failure clusters
grouped by VLM failure label, recent proposals (accepted + rejected), and
the anti-loop signal -- all inside a hard tiktoken (cl100k_base) budget.

Determinism contract
--------------------
* Ordering is by ledger event ``seq`` (the append-only chain's insertion
  counter), never by wall-clock fields the repository stamps itself.
* Priority order (spec): recent physical failures > stratum regressions >
  recent proposals (accepted + rejected) > older entries.  The champion
  card and the anti-loop signal are the irreducible core.
* Dedup is by ``harness_hash`` within each failure label and within the
  proposal list; identical content can never buy context twice.
* If even the irreducible core exceeds the budget, :class:`ContextOverflow`
  is RAISED (fail-closed), never warned.

Reader isolation: the builder opens its own read-only SQLite connection
against the same database file -- exactly the WAL concurrent-reader pattern
the Batch-1 repository documents for dashboards and CLIs.  Nothing in
Batch 1 is modified.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tiktoken

from robocore.cognitive.prompt_registry import (
    DEFAULT_TOKEN_BUDGETS,
    PromptRole,
)
from robocore.cognitive.schemas import (
    ChampionSummary,
    CompiledContext,
    ContextOverflow,
    FailureCluster,
    FailureClusterEntry,
    ForceExploration,
    MutationType,
    ProposalDigest,
    StratumRegression,
    classify_mutation,
    derive_keyframe_hash,
    next_in_rotation,
)
from robocore.contracts import FlightTrigger, HypothesisStatus, RolloutDomain
from robocore.storage.ledger_repo import (
    ExperimentLedgerRepository,
    NoActiveChampionError,
)

__all__ = [
    "TokenCounter",
    "ContextBuilder",
    "render_context_markdown",
    "anti_loop_directive",
    "build_planner_slots",
    "build_critic_slots",
    "build_diagnoser_slots",
    "MAX_CLUSTER_ENTRIES_PER_LABEL",
    "REGRESSION_DELTA_THRESHOLD",
    "REGRESSION_MIN_N_RECENT",
]

#: Hard cap of black-box entries per failure label (spec: 5).
MAX_CLUSTER_ENTRIES_PER_LABEL: int = 5

#: A stratum is a regression when recent_rate < promoted_rate - threshold.
REGRESSION_DELTA_THRESHOLD: float = 0.05

#: Strata with fewer recent rollouts than this are not judged (too noisy).
REGRESSION_MIN_N_RECENT: int = 3

#: Rollouts at/after (promoted_at - tolerance) count as "recent" for
#: regression accounting; the tolerance absorbs batch-commit ordering and
#: the evaluation tail (the promotion event is appended right after its
#: batch, whose rollouts may start a few seconds earlier).
RECENT_WINDOW_TOLERANCE_S: float = 4.0

_VENDORED_BPE_DIR = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


class TokenCounter:
    """cl100k_base token accounting, hermetic by default.

    The cl100k_base BPE ranks are vendored into ``robocore/cognitive/data``
    and pointed at via ``TIKTOKEN_CACHE_DIR`` before the first load, so the
    budget math is byte-identical on an offline cell as well.
    """

    _encoding: Optional["tiktoken.Encoding"] = None

    @classmethod
    def _load(cls) -> "tiktoken.Encoding":
        if cls._encoding is None:
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(_VENDORED_BPE_DIR))
            try:
                cls._encoding = tiktoken.get_encoding("cl100k_base")
            except Exception as exc:  # noqa: BLE001 - surfaced as a hard error
                raise RuntimeError(
                    "cl100k_base encoding unavailable: the vendored BPE"
                    f" cache at {_VENDORED_BPE_DIR} is missing and the"
                    " online fetch failed"
                ) from exc
        return cls._encoding

    def count(self, text: str) -> int:
        """Exact cl100k_base token count of ``text``."""
        return len(self._load().encode(text))


# ---------------------------------------------------------------------------
# Internal read models (frozen rows pulled off the read-only connection)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SeqIndex:
    """Ledger-event seq anchors: id -> append-only chain position."""

    hypothesis_seq: Dict[str, int] = field(default_factory=dict)
    batch_seq: Dict[str, int] = field(default_factory=dict)
    champion_promotion_seqs: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class _FailureRow:
    rollout_id: str
    batch_id: str
    harness_hash: Optional[str]
    label: str
    snapshot_hash: str
    trigger: FlightTrigger
    num_samples: int
    t_start_unix: float
    vlm_summary: str


@dataclass(frozen=True)
class _HypothesisRow:
    hypothesis_id: str
    status: HypothesisStatus
    harness_diff: str
    critic_score: float
    rationale: str


_REJECTED_STATUSES = (
    HypothesisStatus.REJECTED_COMPILER,
    HypothesisStatus.REJECTED_SIM,
    HypothesisStatus.REJECTED_PHYSICAL,
    HypothesisStatus.EARLY_ABORTED,
)


def _resolve_db_path(repo: ExperimentLedgerRepository) -> Path:
    """Resolve the repository's database file for the read-only reader.

    Batch 1 froze the repository API without a public ``db_path`` accessor,
    while its module docstring explicitly sanctions independent WAL reader
    connections.  The private attribute is part of the frozen Batch-1
    layout and therefore stable; if a future batch renames it, this
    resolver fails loudly instead of silently reading the wrong database.
    """
    path = getattr(repo, "_db_path", None)
    if not isinstance(path, Path):
        raise RuntimeError(
            "cannot resolve the ledger database path from the repository"
            " instance (Batch-1 layout changed?); refusing to guess"
        )
    return path


class _LedgerReader:
    """Read-only, snapshot-consistent view over the experiment ledger."""

    def __init__(self, db_path: Path) -> None:
        uri = f"file:{db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "_LedgerReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- seq anchors ---------------------------------------------------------

    def seq_index(self) -> _SeqIndex:
        rows = self._conn.execute(
            "SELECT seq, event_type, payload_json FROM ledger_events"
            " WHERE event_type IN ('HYPOTHESIS_RECORDED','BATCH_COMMITTED',"
            " 'CHAMPION_PROMOTED') ORDER BY seq ASC"
        ).fetchall()
        idx = _SeqIndex()
        for row in rows:
            payload = json.loads(row["payload_json"])
            seq = int(row["seq"])
            if row["event_type"] == "HYPOTHESIS_RECORDED":
                idx.hypothesis_seq[str(payload["hypothesis_id"])] = seq
            elif row["event_type"] == "BATCH_COMMITTED":
                idx.batch_seq[str(payload["batch_id"])] = seq
            else:
                idx.champion_promotion_seqs.append(seq)
        return idx

    # -- failures -------------------------------------------------------------

    def recent_physical_failures(self, task_name: str, limit: int) -> List[_FailureRow]:
        """Failed physical rollouts that carry a sealed black-box dump."""
        rows = self._conn.execute(
            "SELECT r.rollout_id, r.batch_id, r.harness_hash,"
            "       COALESCE(r.vlm_failure_summary, '<unclassified>') AS label,"
            "       r.flight_snapshot_hash, r.t_start_unix, d.trigger,"
            "       d.num_samples"
            "  FROM rollout_telemetry r"
            "  JOIN blackbox_dumps d ON d.snapshot_hash = r.flight_snapshot_hash"
            " WHERE r.task_name = ? AND r.domain = 'PHYSICAL'"
            "   AND r.success = 0"
            " ORDER BY r.t_start_unix DESC, r.rollout_id ASC"
            " LIMIT ?",
            (task_name, int(limit)),
        ).fetchall()
        out = []
        for r in rows:
            label, paragraph = _split_label(str(r["label"]))
            out.append(
                _FailureRow(
                    rollout_id=str(r["rollout_id"]),
                    batch_id=str(r["batch_id"]),
                    harness_hash=r["harness_hash"],
                    label=label,
                    snapshot_hash=str(r["flight_snapshot_hash"]),
                    trigger=FlightTrigger(str(r["trigger"])),
                    num_samples=int(r["num_samples"]),
                    t_start_unix=float(r["t_start_unix"]),
                    vlm_summary=paragraph,
                )
            )
        return out

    # -- hypotheses ------------------------------------------------------------

    def recent_hypotheses(self, task_name: str, limit: int) -> List[_HypothesisRow]:
        rows = self._conn.execute(
            "SELECT hypothesis_id, status, harness_diff, critic_score,"
            "       rationale FROM hypotheses WHERE task_name = ?"
            " ORDER BY proposed_at_unix DESC, hypothesis_id ASC LIMIT ?",
            (task_name, int(limit)),
        ).fetchall()
        return [
            _HypothesisRow(
                hypothesis_id=str(r["hypothesis_id"]),
                status=HypothesisStatus(str(r["status"])),
                harness_diff=str(r["harness_diff"]),
                critic_score=float(r["critic_score"]),
                rationale=str(r["rationale"]),
            )
            for r in rows
        ]

    def hypothesis_diffs(self, task_name: str, limit: int) -> Dict[str, str]:
        rows = self._conn.execute(
            "SELECT hypothesis_id, harness_diff FROM hypotheses"
            " WHERE task_name = ? ORDER BY proposed_at_unix DESC,"
            " hypothesis_id ASC LIMIT ?",
            (task_name, int(limit)),
        ).fetchall()
        return {str(r["hypothesis_id"]): str(r["harness_diff"]) for r in rows}


# ---------------------------------------------------------------------------
# Section renderers (the exact text the LLM sees; hashes only, no image bytes)
# ---------------------------------------------------------------------------


def _split_label(raw_summary: str) -> tuple[str, str]:
    """Split a ledger ``vlm_failure_summary`` into (label, paragraph).

    The diagnoser convention is a label-led summary:
    ``"GRASP_MISALIGNED_ON_APPROACH: <one paragraph>"``.  The label is the
    first colon-separated UPPER_SNAKE_CASE token; everything after the
    colon is the one-paragraph VLM summary.  A bare tag (no paragraph) or
    a non-conforming string is treated as its own label with the raw text
    as summary -- never an error, always deterministic.
    """
    if ":" in raw_summary:
        head, _, tail = raw_summary.partition(":")
        label = head.strip()
        if (
            label
            and label.isupper()
            and label.replace("_", "").isalnum()
            and len(tail.strip()) > 0
        ):
            return label, tail.strip()
    return raw_summary, raw_summary


def _fmt_rate(x: float) -> str:
    return f"{x:.3f}"


def _render_champion_block(summary: ChampionSummary) -> str:
    lines: List[str] = [
        "## CHAMPION PERFORMANCE (diff target hash is quoted in the prompt"
        " header; it is intentionally not repeated here)",
        f"- task: {summary.task_name} | policy checkpoint:"
        f" {summary.policy_checkpoint_hash} | promotion batch:"
        f" {summary.batch_id} | promoted_at_unix:"
        f" {summary.promoted_at_unix:.1f}",
        f"- Wilson lower bound (physical success rate):"
        f" {_fmt_rate(summary.wilson_lower_bound)}",
    ]
    for st in summary.stratum_stats:
        lines.append(
            f"- stratum {st.stratum.value}: n={st.n} successes={st.successes}"
            f" rate={_fmt_rate(st.success_rate)}"
            f" wilson=[{_fmt_rate(st.wilson_low)}, {_fmt_rate(st.wilson_high)}]"
        )
    for reg in summary.regressions:
        lines.append(
            f"- REGRESSION: stratum {reg.stratum} recent"
            f" {_fmt_rate(reg.recent_rate)} vs promoted"
            f" {_fmt_rate(reg.promoted_rate)} (delta {reg.delta:+.3f},"
            f" n_recent={reg.n_recent})"
        )
    return "\n".join(lines)


def _render_cluster_block(cluster: FailureCluster) -> str:
    head = (
        f"### failure label {cluster.failure_label}"
        f" (observed {cluster.total_observed} dumps, showing"
        f" {len(cluster.entries)})"
    )
    lines = [head]
    for e in cluster.entries:
        lines.append(
            f"- [ledger seq {e.ledger_seq}] rollout {e.rollout_id} |"
            f" snapshot {e.snapshot_hash} | keyframe {e.keyframe_hash} |"
            f" trigger {e.trigger.value} | t={e.occurred_at_unix:.1f}"
        )
        lines.append(f"  VLM summary: {e.vlm_summary}")
    return "\n".join(lines)


def _render_proposal_block(digest: ProposalDigest) -> str:
    return (
        f"- [ledger seq {digest.ledger_seq}] {digest.hypothesis_id}"
        f" {digest.status.value} family={digest.mutation.value}"
        f" critic_score={digest.critic_score:.2f} :: {digest.rationale_excerpt}"
    )


def _render_anti_loop_block(signal: Optional[ForceExploration]) -> str:
    if signal is None:
        return (
            "## ANTI-LOOP SIGNAL\nINACTIVE: no mutation family is exhausted;"
            " propose freely within the one-mutation rule."
        )
    fams = ", ".join(f.value for f in signal.rejected_families)
    return (
        "## ANTI-LOOP SIGNAL\nACTIVE: the last"
        f" {signal.rejected_streak} rejected proposals exhausted mutation"
        f" family {fams}. Directing exploration to"
        f" {signal.catalog_family.value} next (fixed catalog rotation)."
        f" Reason: {signal.reason}"
    )


def render_context_markdown(ctx: CompiledContext) -> str:
    """Render the compiled context as the markdown the planner consumes."""
    parts: List[str] = []
    if ctx.champion_summary is not None:
        parts.append(_render_champion_block(ctx.champion_summary))
    else:
        parts.append(
            "## CHAMPION PERFORMANCE\nNo active champion (genesis mode): the"
            " diff target is the genesis harness quoted in the prompt header."
        )
    if ctx.failure_clusters:
        parts.append(
            "## RECENT PHYSICAL FAILURES (black-box clusters; keyframes are"
            " content hashes, image bytes never enter the context)"
        )
        for cluster in ctx.failure_clusters:
            parts.append(_render_cluster_block(cluster))
    if ctx.recent_proposals:
        parts.append("## RECENT PROPOSALS (accepted + rejected, ledger-ordered)")
        parts.extend(_render_proposal_block(d) for d in ctx.recent_proposals)
    parts.append(_render_anti_loop_block(ctx.anti_loop_signal))
    return "\n\n".join(parts)


def anti_loop_directive(ctx: CompiledContext) -> str:
    """The planner-prompt anti-loop directive line for this context."""
    if ctx.anti_loop_signal is None:
        return (
            "INACTIVE -- no mutation family is exhausted. You may propose in"
            " any family, subject to the one-mutation rule."
        )
    sig = ctx.anti_loop_signal
    return (
        f"ACTIVE -- the last {sig.rejected_streak} rejected proposals share"
        f" one mutation family. You MUST propose in family"
        f" {sig.catalog_family.value} (the next entry of the fixed catalog"
        " rotation), or refuse with ANTI_LOOP_WITHOUT_OVERRIDE if you"
        " believe that family is also exhausted."
    )


def build_planner_slots(
    ctx: CompiledContext, *, champion_hash: str, token_budget: int
) -> Dict[str, str]:
    """All five ``planner_v1`` slot values.  ``champion_hash`` is passed
    through verbatim so the rendered prompt quotes the diff target exactly
    once (in the DIFF TARGET header)."""
    return {
        "task_name": (
            ctx.champion_summary.task_name if ctx.champion_summary else ""
        ),
        "champion_hash": champion_hash,
        "token_budget": str(int(token_budget)),
        "anti_loop_directive": anti_loop_directive(ctx),
        "context_markdown": render_context_markdown(ctx),
    }


def build_critic_slots(
    *,
    task_name: str,
    rationale: str,
    predicted_delta: float,
    proposal_diff: str,
    champion_wilson_low: float,
    rejected_similar_digest: str,
) -> Dict[str, str]:
    """Slot values for ``critic_v1``."""
    return {
        "task_name": task_name,
        "rationale": rationale,
        "predicted_delta": f"{predicted_delta:+.4f}",
        "proposal_diff": proposal_diff,
        "champion_wilson_low": f"{champion_wilson_low:.4f}",
        "rejected_similar_digest": rejected_similar_digest,
    }


def build_diagnoser_slots(
    *,
    task_name: str,
    failure_label: str,
    telemetry_digest: str,
    keyframe_hashes: List[str],
) -> Dict[str, str]:
    """Slot values for ``diagnoser_v1``."""
    return {
        "task_name": task_name,
        "failure_label": failure_label,
        "telemetry_digest": telemetry_digest,
        "keyframe_hashes": "\n".join(f"- {h}" for h in keyframe_hashes),
    }


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------


class ContextBuilder:
    """Compiles ledger state into a budgeted, strictly-typed planner context."""

    def __init__(self, *, token_counter: Optional[TokenCounter] = None) -> None:
        self._counter = token_counter or TokenCounter()

    # -- public API -----------------------------------------------------------

    def build(
        self,
        ledger_repo: ExperimentLedgerRepository,
        champion_hash: str,
        token_budget: Optional[int] = None,
        *,
        role: PromptRole = PromptRole.PLANNER,
    ) -> CompiledContext:
        """Build the compiled context for the harness being diffed against.

        ``champion_hash`` must be a harness recorded in the ledger (in
        genesis mode, pass the genesis harness hash; no champion row exists
        yet and ``champion_summary`` comes back as ``None``).
        ``token_budget`` defaults to the role's default budget
        (planner 8000 / critic 2000 / diagnoser 4000).
        """
        budget = (
            int(token_budget)
            if token_budget is not None
            else DEFAULT_TOKEN_BUDGETS[role]
        )
        if budget <= 0:
            raise ValueError(f"token budget must be positive, got {budget}")
        if not champion_hash:
            raise ValueError(
                "champion_hash is required (pass the genesis harness hash in"
                " genesis mode)"
            )

        target = ledger_repo.get_harness(champion_hash)
        task_name = target.task_name

        with _LedgerReader(_resolve_db_path(ledger_repo)) as reader:
            seq_index = reader.seq_index()
            champion_summary = self._champion_summary(
                ledger_repo, task_name, champion_hash, seq_index
            )
            anti_loop = self._anti_loop_signal(
                reader, task_name, seq_index
            )
            failure_clusters = self._failure_clusters(
                reader, task_name, seq_index
            )
            recent, older = self._proposal_digests(
                reader, task_name, seq_index
            )

        # ---- budgeted assembly -------------------------------------------
        # Irreducible core: champion card (incl. regressions) + anti-loop.
        core_text = "\n\n".join(
            [
                _render_champion_block(champion_summary)
                if champion_summary is not None
                else (
                    "## CHAMPION PERFORMANCE\nNo active champion (genesis"
                    " mode): the diff target is the genesis harness quoted in"
                    " the prompt header."
                ),
                _render_anti_loop_block(anti_loop),
            ]
        )
        core_tokens = self._counter.count(core_text)
        if core_tokens > budget:
            raise ContextOverflow(budget, core_tokens, "irreducible core")

        seqs: set[int] = set()
        if champion_summary is not None:
            seqs.add(champion_summary.ledger_seq)
        included_clusters: List[FailureCluster] = []
        included_proposals: List[ProposalDigest] = []
        running = core_tokens

        # P1: recent physical failures, newest label first, capped per label.
        for cluster in failure_clusters:
            kept: List[FailureClusterEntry] = []
            for entry in cluster.entries:
                candidate = _render_cluster_block(
                    FailureCluster(
                        failure_label=cluster.failure_label,
                        entries=[entry],
                        total_observed=cluster.total_observed,
                        ledger_seq=cluster.ledger_seq,
                    )
                )
                candidate_tokens = self._counter.count(
                    _render_cluster_header(cluster) + "\n" + candidate
                )
                if running + candidate_tokens <= budget:
                    kept.append(entry)
                    running += candidate_tokens
                    seqs.add(entry.ledger_seq)
            if kept:
                included_clusters.append(
                    FailureCluster(
                        failure_label=cluster.failure_label,
                        entries=kept,
                        total_observed=cluster.total_observed,
                        ledger_seq=max(e.ledger_seq for e in kept),
                    )
                )

        # P3 then P4: recent proposals (accepted + rejected), then older.
        for digest in recent + older:
            candidate = _render_proposal_block(digest)
            candidate_tokens = self._counter.count(candidate)
            if running + candidate_tokens <= budget:
                included_proposals.append(digest)
                running += candidate_tokens
                seqs.add(digest.ledger_seq)

        ctx = CompiledContext(
            champion_summary=champion_summary,
            failure_clusters=included_clusters,
            recent_proposals=included_proposals,
            anti_loop_signal=anti_loop,
            token_estimate=running,
            source_entry_seqs=sorted(seqs),
        )

        # Belt and braces: the fully rendered context must fit, exactly.
        final_tokens = self._counter.count(render_context_markdown(ctx))
        if final_tokens > budget:
            raise ContextOverflow(budget, final_tokens, "final assembly")
        return ctx.model_copy(update={"token_estimate": final_tokens})

    # -- section compilers ----------------------------------------------------

    def _champion_summary(
        self,
        repo: ExperimentLedgerRepository,
        task_name: str,
        champion_hash: str,
        seq_index: _SeqIndex,
    ) -> Optional[ChampionSummary]:
        try:
            champion = repo.get_active_champion(task_name)
        except NoActiveChampionError:
            return None
        recent_stats = repo.get_stratum_stats(
            champion.harness_hash,
            RolloutDomain.PHYSICAL,
            since_unix=champion.promoted_at_unix - RECENT_WINDOW_TOLERANCE_S,
        )
        recent_by_stratum = {s.stratum: s for s in recent_stats}
        regressions: List[StratumRegression] = []
        for promoted in champion.stratum_stats:
            recent = recent_by_stratum.get(promoted.stratum)
            if recent is None or recent.n < REGRESSION_MIN_N_RECENT:
                continue
            delta = recent.success_rate - promoted.success_rate
            if delta < -REGRESSION_DELTA_THRESHOLD:
                regressions.append(
                    StratumRegression(
                        stratum=promoted.stratum.value,
                        promoted_rate=promoted.success_rate,
                        recent_rate=recent.success_rate,
                        delta=delta,
                        n_promoted=promoted.n,
                        n_recent=recent.n,
                        ledger_seq=seq_index.champion_promotion_seqs[-1]
                        if seq_index.champion_promotion_seqs
                        else 1,
                    )
                )
        promo_seq = (
            seq_index.champion_promotion_seqs[-1]
            if seq_index.champion_promotion_seqs
            else 1
        )
        return ChampionSummary(
            task_name=task_name,
            harness_hash=champion.harness_hash,
            policy_checkpoint_hash=champion.policy_checkpoint_hash,
            batch_id=champion.batch_id,
            wilson_lower_bound=champion.wilson_lower_bound,
            stratum_stats=champion.stratum_stats,
            regressions=regressions,
            promoted_at_unix=champion.promoted_at_unix,
            ledger_seq=promo_seq,
        )

    def _anti_loop_signal(
        self, reader: _LedgerReader, task_name: str, seq_index: _SeqIndex
    ) -> Optional[ForceExploration]:
        """Trigger when the last 3 rejected hypotheses share one family."""
        diffs = reader.hypothesis_diffs(task_name, limit=64)
        rejected: List[Tuple[int, str, MutationType]] = []
        for hyp_id, diff in diffs.items():
            seq = seq_index.hypothesis_seq.get(hyp_id)
            if seq is None:
                continue
            row = self._hypothesis_status(reader, task_name, hyp_id)
            if row is None or row not in _REJECTED_STATUSES:
                continue
            rejected.append((seq, hyp_id, classify_mutation(diff)))
        rejected.sort(key=lambda item: -item[0])
        last3 = rejected[:3]
        if len(last3) < 3:
            return None
        families = [f for _, _, f in last3]
        if len(set(families)) != 1:
            return None
        exhausted = families[0]
        hyp_ids = [h for _, h, _ in last3]
        return ForceExploration(
            catalog_family=next_in_rotation(exhausted),
            rejected_streak=len(last3),
            rejected_families=families,
            reason=(
                f"ledger shows three consecutive rejections"
                f" ({', '.join(hyp_ids)}) all classified as"
                f" {exhausted.value}; the fixed rotation directs the next"
                f" proposal to {next_in_rotation(exhausted).value}"
            ),
        )

    @staticmethod
    def _hypothesis_status(
        reader: _LedgerReader, task_name: str, hypothesis_id: str
    ) -> Optional[HypothesisStatus]:
        row = reader._conn.execute(  # noqa: SLF001 - same-module helper
            "SELECT status FROM hypotheses WHERE hypothesis_id = ?",
            (hypothesis_id,),
        ).fetchone()
        if row is None:
            return None
        return HypothesisStatus(str(row["status"]))

    def _failure_clusters(
        self, reader: _LedgerReader, task_name: str, seq_index: _SeqIndex
    ) -> List[FailureCluster]:
        rows = reader.recent_physical_failures(task_name, limit=32)
        # Deterministic recency: batch ledger seq DESC, rollout id ASC.
        def _sort_key(row: _FailureRow) -> Tuple[int, str]:
            seq = seq_index.batch_seq.get(row.batch_id, 0)
            return (-seq, row.rollout_id)

        rows = sorted(rows, key=_sort_key)
        by_label: Dict[str, List[_FailureRow]] = {}
        for row in rows:
            by_label.setdefault(row.label, []).append(row)

        clusters: List[FailureCluster] = []
        for label in sorted(
            by_label,
            key=lambda lbl: -max(
                seq_index.batch_seq.get(r.batch_id, 0)
                for r in by_label[lbl]
            ),
        ):
            label_rows = by_label[label]
            seen_harnesses: set[str] = set()
            entries: List[FailureClusterEntry] = []
            for row in label_rows:
                if row.harness_hash is not None:
                    if row.harness_hash in seen_harnesses:
                        continue  # dedup by harness_hash within the label
                    seen_harnesses.add(row.harness_hash)
                if len(entries) >= MAX_CLUSTER_ENTRIES_PER_LABEL:
                    break
                seq = seq_index.batch_seq.get(row.batch_id, 0) or 1
                frame = max(0, int(row.num_samples * 0.9) - 1)
                entries.append(
                    FailureClusterEntry(
                        failure_label=label,
                        snapshot_hash=row.snapshot_hash,
                        rollout_id=row.rollout_id,
                        keyframe_hash=derive_keyframe_hash(
                            row.snapshot_hash, frame
                        ),
                        vlm_summary=row.vlm_summary,
                        trigger=row.trigger,
                        harness_hash=row.harness_hash,
                        occurred_at_unix=row.t_start_unix,
                        ledger_seq=seq,
                    )
                )
            if entries:
                clusters.append(
                    FailureCluster(
                        failure_label=label,
                        entries=entries,
                        total_observed=len(label_rows),
                        ledger_seq=max(e.ledger_seq for e in entries),
                    )
                )
        return clusters

    def _proposal_digests(
        self, reader: _LedgerReader, task_name: str, seq_index: _SeqIndex
    ) -> Tuple[List[ProposalDigest], List[ProposalDigest]]:
        """(recent, older) proposal digests, ledger-seq ordered, deduped by
        harness hash where one exists."""
        rows = reader.recent_hypotheses(task_name, limit=20)
        enriched: List[Tuple[int, _HypothesisRow]] = []
        for row in rows:
            seq = seq_index.hypothesis_seq.get(row.hypothesis_id)
            if seq is None:
                continue
            enriched.append((seq, row))
        enriched.sort(key=lambda item: -item[0])

        digests: List[ProposalDigest] = []
        seen_harness: set[str] = set()
        harness_hashes = self._harness_hashes(reader, task_name)
        for seq, row in enriched:
            hhash = harness_hashes.get(row.hypothesis_id)
            if hhash is not None:
                if hhash in seen_harness:
                    continue
                seen_harness.add(hhash)
            digests.append(
                ProposalDigest(
                    hypothesis_id=row.hypothesis_id,
                    status=row.status,
                    mutation=classify_mutation(row.harness_diff),
                    critic_score=row.critic_score,
                    rationale_excerpt=_excerpt(row.rationale, 160),
                    ledger_seq=seq,
                )
            )
        return digests[:12], digests[12:]

    @staticmethod
    def _harness_hashes(
        reader: _LedgerReader, task_name: str
    ) -> Dict[str, str]:
        rows = reader._conn.execute(  # noqa: SLF001 - same-module helper
            "SELECT hypothesis_id, harness_hash FROM hypotheses"
            " WHERE task_name = ? AND harness_hash IS NOT NULL",
            (task_name,),
        ).fetchall()
        return {str(r["hypothesis_id"]): str(r["harness_hash"]) for r in rows}


def _excerpt(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_cluster_header(cluster: FailureCluster) -> str:
    return (
        f"### failure label {cluster.failure_label}"
        f" (observed {cluster.total_observed} dumps, showing"
        f" {len(cluster.entries)})"
    )
