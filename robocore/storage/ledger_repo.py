"""robocore.storage.ledger_repo -- SQLite WAL experiment & lineage database.

Deliverable 2 of the robocore Phase-1 runtime.

Storage engine design notes
---------------------------
* **WAL mode** (``PRAGMA journal_mode=WAL``): the operator CLI / dashboards
  read concurrently with the 1 write-capable orchestrator connection and are
  never blocked.  ``synchronous=NORMAL`` trades one fsync per commit for
  throughput; in WAL mode a power loss may drop the *tail* commits but never
  tears one (WAL frames are checksummed), so the hash chain stays verifiable
  via :meth:`ExperimentLedgerRepository.verify_chain_integrity`.
* **Append-only ledger_events chain**: every state-machine intent, batch
  commit, promotion and operator override is an event whose ``event_hash``
  is sha256 over (prev_hash, event_type, cycle_id, canonical payload JSON,
  ISO timestamp).  A BEFORE INSERT trigger enforces ``prev_hash == head`` at
  the SQLite level, so even a hand-written ``INSERT`` from the ``sqlite3``
  CLI cannot fork the chain silently.  UPDATE/DELETE on the table is
  trigger-blocked.
* **Foreign keys are strict and RESTRICT**: nothing is ever deleted; the
  database is a monotone experiment record.  The only intentionally
  FK-less edge is ``blackbox_dumps.rollout_id`` -- flight-recorder dumps are
  written the instant a trigger fires (mid-rollout), i.e. *before* the
  rollout row exists.  The forward reference is validated in
  :meth:`record_evaluation_batch` (snapshot's rollout_id must match the
  rollout claiming it).
* **Champions**: a partial UNIQUE index allows exactly one active champion
  per task; promotion is a single transaction (deactivate old, insert new,
  append CHAMPION_PROMOTED event).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from robocore.contracts import (
    BatchDecision,
    BatchStatistics,
    BlackBoxSnapshotRecord,
    ChampionManifest,
    DifficultyStrataBin,
    EvaluationBatchRecord,
    EventType,
    FalsifiablePrediction,
    FlightTrigger,
    HarnessManifest,
    HypothesisProposal,
    HypothesisRecord,
    HypothesisStatus,
    LedgerEvent,
    OperatorOverride,
    PolicyCheckpointRecord,
    RejectedDiffSummary,
    ResetOutcome,
    RolloutDomain,
    RolloutTelemetryBundle,
    SafetyValidationReceipt,
    SimToRealGapReport,
    StratumStats,
    SysIDStatus,
    canonical_json,
    iso_now,
    unix_now,
)
from robocore.stats import wilson_score_interval

__all__ = [
    "LedgerError",
    "ChainIntegrityError",
    "NoActiveChampionError",
    "DuplicateHypothesisError",
    "ExperimentLedgerRepository",
]

_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL_STATEMENTS: Tuple[str, ...] = (
    # -- harnesses: content-addressed compiled policy-harness trees ----------
    """
    CREATE TABLE IF NOT EXISTS harnesses (
        harness_hash           TEXT PRIMARY KEY,
        parent_harness_hash    TEXT REFERENCES harnesses(harness_hash)
                                  ON DELETE RESTRICT,
        task_name              TEXT NOT NULL,
        harness_diff           TEXT NOT NULL,
        ast_fingerprint        TEXT NOT NULL,
        compiler_version       TEXT NOT NULL,
        safety_receipt_json    TEXT NOT NULL,
        difficulty_strata_json TEXT NOT NULL DEFAULT '[]',
        config_json            TEXT NOT NULL DEFAULT '{}',
        created_at_unix        REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_harnesses_task_created"
    " ON harnesses(task_name, created_at_unix DESC)",
    "CREATE INDEX IF NOT EXISTS idx_harnesses_parent"
    " ON harnesses(parent_harness_hash)",
    # Same diff text against the same parent is byte-identical content and
    # therefore the same harness_hash (PK dedup).  Structural families with
    # different constants keep distinct hashes -- anti-looping dedup on
    # (parent, ast_fingerprint) is done through query_similar_rejected_diffs.
    "CREATE INDEX IF NOT EXISTS idx_harnesses_parent_ast"
    " ON harnesses(parent_harness_hash, ast_fingerprint)",

    # -- policy checkpoints ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS policy_checkpoints (
        checkpoint_hash  TEXT PRIMARY KEY,
        harness_hash     TEXT NOT NULL REFERENCES harnesses(harness_hash)
                           ON DELETE RESTRICT,
        format           TEXT NOT NULL,
        storage_path     TEXT NOT NULL,
        num_params       INTEGER NOT NULL,
        created_at_unix  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_checkpoints_harness"
    " ON policy_checkpoints(harness_hash, created_at_unix DESC)",

    # -- hypotheses -------------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS hypotheses (
        hypothesis_id       TEXT PRIMARY KEY,
        task_name           TEXT NOT NULL,
        parent_harness_hash TEXT NOT NULL REFERENCES harnesses(harness_hash)
                              ON DELETE RESTRICT,
        harness_hash        TEXT REFERENCES harnesses(harness_hash)
                              ON DELETE RESTRICT,
        harness_diff        TEXT NOT NULL,
        ast_fingerprint     TEXT NOT NULL,
        param_summary_json  TEXT NOT NULL DEFAULT '{}',
        rationale           TEXT NOT NULL,
        prediction_json     TEXT NOT NULL,
        critic_score        REAL NOT NULL,
        status              TEXT NOT NULL DEFAULT 'PROPOSED'
                            CHECK (status IN ('PROPOSED','COMPILED','EVALUATING',
                                              'PROMOTED','REJECTED_COMPILER',
                                              'REJECTED_SIM','REJECTED_PHYSICAL',
                                              'EARLY_ABORTED','SUPERSEDED')),
        rejection_reason    TEXT,
        proposed_at_unix    REAL NOT NULL,
        decided_at_unix     REAL
    )
    """,
    # Exact dedup: re-proposing a byte-identical diff against the same parent
    # is blocked at the storage layer (the critic guard should catch it first;
    # this is the last line of defense).
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_hypotheses_parent_diff"
    " ON hypotheses(parent_harness_hash, harness_diff)",
    "CREATE INDEX IF NOT EXISTS idx_hypotheses_task_status"
    " ON hypotheses(task_name, status)",
    "CREATE INDEX IF NOT EXISTS idx_hypotheses_parent_ast"
    " ON hypotheses(parent_harness_hash, ast_fingerprint)",

    # -- evaluation batches -------------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS evaluation_batches (
        batch_id          TEXT PRIMARY KEY,
        cycle_id          TEXT NOT NULL,
        hypothesis_id     TEXT NOT NULL REFERENCES hypotheses(hypothesis_id)
                            ON DELETE RESTRICT,
        harness_hash      TEXT NOT NULL REFERENCES harnesses(harness_hash)
                            ON DELETE RESTRICT,
        champion_hash     TEXT REFERENCES harnesses(harness_hash)
                            ON DELETE RESTRICT,
        domain            TEXT NOT NULL CHECK (domain IN ('SIM','PHYSICAL')),
        task_name         TEXT NOT NULL,
        n_rollouts        INTEGER NOT NULL,
        decision          TEXT NOT NULL CHECK (decision IN
                            ('CONTINUE','PROMOTE','REJECT','EARLY_ABORT','INCONCLUSIVE')),
        statistics_json   TEXT NOT NULL DEFAULT '{}',
        trigger           TEXT NOT NULL DEFAULT 'CYCLE',
        started_at_unix   REAL NOT NULL,
        committed_at_unix REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_batches_harness_domain"
    " ON evaluation_batches(harness_hash, domain, committed_at_unix DESC)",
    "CREATE INDEX IF NOT EXISTS idx_batches_task_time"
    " ON evaluation_batches(task_name, committed_at_unix DESC)",

    # -- blackbox flight-recorder dumps ------------------------------------------
    # rollout_id is an intentional forward reference (dumps are written
    # mid-rollout, before the rollout row exists); validated at batch commit.
    """
    CREATE TABLE IF NOT EXISTS blackbox_dumps (
        snapshot_hash    TEXT PRIMARY KEY,
        rollout_id       TEXT NOT NULL,
        trigger          TEXT NOT NULL CHECK (trigger IN
                           ('FORCE_SPIKE','COLLISION','INFERENCE_DEADLINE',
                            'TASK_FAILURE','MANUAL')),
        parquet_path     TEXT NOT NULL,
        sim_state_path   TEXT NOT NULL,
        format           TEXT NOT NULL DEFAULT 'parquet',
        t_window_seconds REAL NOT NULL,
        sample_rate_hz   REAL NOT NULL,
        num_samples      INTEGER NOT NULL,
        mujoco_model_hash TEXT NOT NULL,
        created_at_unix  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_blackbox_rollout"
    " ON blackbox_dumps(rollout_id)",
    "CREATE INDEX IF NOT EXISTS idx_blackbox_trigger_time"
    " ON blackbox_dumps(trigger, created_at_unix DESC)",
    # Rollout -> snapshot direction IS a real FK (dumps are always recorded
    # before the batch that owns the rollout).
    """
    CREATE TABLE IF NOT EXISTS rollout_telemetry (
        rollout_id     TEXT PRIMARY KEY,
        batch_id       TEXT NOT NULL REFERENCES evaluation_batches(batch_id)
                         ON DELETE RESTRICT,
        harness_hash   TEXT NOT NULL REFERENCES harnesses(harness_hash)
                         ON DELETE RESTRICT,
        task_name      TEXT NOT NULL,
        domain         TEXT NOT NULL CHECK (domain IN ('SIM','PHYSICAL')),
        stratum        TEXT NOT NULL CHECK (stratum IN
                         ('EASY_CENTER','MEDIUM_OBLIQUE','HARD_BOUNDARY','OCCLUDED')),
        success        INTEGER NOT NULL CHECK (success IN (0,1)),
        time_to_completion REAL,
        cumulative_jerk REAL NOT NULL DEFAULT 0.0,
        torque_penalty REAL NOT NULL DEFAULT 0.0,
        inference_deadline_misses INTEGER NOT NULL DEFAULT 0,
        force_spike_max REAL NOT NULL DEFAULT 0.0,
        collision_count INTEGER NOT NULL DEFAULT 0,
        reset_outcome  TEXT,
        flight_snapshot_hash TEXT REFERENCES blackbox_dumps(snapshot_hash)
                         ON DELETE RESTRICT,
        t_start_unix   REAL NOT NULL,
        t_end_unix     REAL NOT NULL,
        video_uri      TEXT,
        vlm_failure_summary TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_rollouts_batch"
    " ON rollout_telemetry(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_rollouts_harness_stratum"
    " ON rollout_telemetry(harness_hash, stratum)",
    "CREATE INDEX IF NOT EXISTS idx_rollouts_task_time"
    " ON rollout_telemetry(task_name, t_start_unix DESC)",
    "CREATE INDEX IF NOT EXISTS idx_rollouts_failure_mode"
    " ON rollout_telemetry(task_name, vlm_failure_summary) WHERE success = 0",

    # -- sim-to-real calibration history -------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS sim_to_real_calibration (
        calibration_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name           TEXT NOT NULL,
        harness_hash        TEXT REFERENCES harnesses(harness_hash)
                              ON DELETE RESTRICT,
        snapshot_hash       TEXT REFERENCES blackbox_dumps(snapshot_hash)
                              ON DELETE RESTRICT,
        sim_metric          REAL NOT NULL,
        real_metric         REAL NOT NULL,
        gap                 REAL NOT NULL,
        friction_offset_json TEXT NOT NULL DEFAULT '{}',
        damping_offset_json  TEXT NOT NULL DEFAULT '{}',
        residual_before     REAL,
        residual_after      REAL,
        status              TEXT NOT NULL CHECK (status IN
                              ('APPLIED','INSUFFICIENT_CHANGE','FAILED_TO_CONVERGE')),
        calibrated_at_unix  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sysid_task_time"
    " ON sim_to_real_calibration(task_name, calibrated_at_unix DESC)",

    # -- champion registry (derived index table; O(1) active lookup) ---------------
    """
    CREATE TABLE IF NOT EXISTS champions (
        champion_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name              TEXT NOT NULL,
        harness_hash           TEXT NOT NULL REFERENCES harnesses(harness_hash)
                                 ON DELETE RESTRICT,
        policy_checkpoint_hash TEXT NOT NULL REFERENCES
                                 policy_checkpoints(checkpoint_hash)
                                 ON DELETE RESTRICT,
        batch_id               TEXT NOT NULL REFERENCES
                                 evaluation_batches(batch_id) ON DELETE RESTRICT,
        wilson_lower_bound     REAL NOT NULL,
        stratum_stats_json     TEXT NOT NULL DEFAULT '[]',
        promoted_at_unix       REAL NOT NULL,
        active                 INTEGER NOT NULL DEFAULT 1
                                 CHECK (active IN (0,1))
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_champions_one_active"
    " ON champions(task_name) WHERE active = 1",
    "CREATE INDEX IF NOT EXISTS idx_champions_task_time"
    " ON champions(task_name, promoted_at_unix DESC)",

    # -- append-only event chain ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS ledger_events (
        seq              INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type       TEXT NOT NULL,
        cycle_id         TEXT,
        payload_json     TEXT NOT NULL,
        prev_hash        TEXT NOT NULL DEFAULT '',
        event_hash       TEXT NOT NULL,
        recorded_at_unix REAL NOT NULL,
        recorded_at_iso  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_type_seq"
    " ON ledger_events(event_type, seq DESC)",
    "CREATE INDEX IF NOT EXISTS idx_events_cycle"
    " ON ledger_events(cycle_id, seq)",
)

_TRIGGER_STATEMENTS: Tuple[str, ...] = (
    # Chain continuity: prev_hash must equal the current head (or '' for the
    # genesis event). Blocks out-of-band INSERTs from forking the chain.
    """
    CREATE TRIGGER IF NOT EXISTS trg_ledger_events_chain_guard
    BEFORE INSERT ON ledger_events
    BEGIN
        SELECT CASE
            WHEN NEW.prev_hash IS NULL OR NEW.prev_hash <> COALESCE(
                (SELECT event_hash FROM ledger_events
                  WHERE seq = (SELECT MAX(seq) FROM ledger_events)), '')
            THEN RAISE(ABORT,
                'LEDGER_CHAIN_BROKEN: prev_hash does not match chain head')
        END;
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_ledger_events_no_update
    BEFORE UPDATE ON ledger_events
    BEGIN
        SELECT RAISE(ABORT, 'ledger_events is append-only');
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS trg_ledger_events_no_delete
    BEFORE DELETE ON ledger_events
    BEGIN
        SELECT RAISE(ABORT, 'ledger_events is append-only');
    END;
    """,
    # Black-box dumps are physical evidence: immutable after write.
    """
    CREATE TRIGGER IF NOT EXISTS trg_blackbox_immutable
    BEFORE UPDATE ON blackbox_dumps
    BEGIN
        SELECT RAISE(ABORT, 'blackbox_dumps is append-only');
    END;
    """,
    # Hypothesis lifecycle state machine, enforced in the database.
    """
    CREATE TRIGGER IF NOT EXISTS trg_hypotheses_status_guard
    BEFORE UPDATE OF status ON hypotheses
    BEGIN
        SELECT CASE
            WHEN OLD.status = 'PROPOSED' AND NEW.status NOT IN
                ('COMPILED','REJECTED_COMPILER','SUPERSEDED')
            THEN RAISE(ABORT, 'illegal hypothesis transition from PROPOSED')
            WHEN OLD.status = 'COMPILED' AND NEW.status NOT IN
                ('EVALUATING','REJECTED_SIM','SUPERSEDED')
            THEN RAISE(ABORT, 'illegal hypothesis transition from COMPILED')
            WHEN OLD.status = 'EVALUATING' AND NEW.status NOT IN
                ('PROMOTED','REJECTED_PHYSICAL','REJECTED_SIM',
                 'EARLY_ABORTED','SUPERSEDED')
            THEN RAISE(ABORT, 'illegal hypothesis transition from EVALUATING')
            WHEN OLD.status IN ('PROMOTED','REJECTED_COMPILER','REJECTED_SIM',
                                'REJECTED_PHYSICAL','EARLY_ABORTED','SUPERSEDED')
                 AND NEW.status <> OLD.status
            THEN RAISE(ABORT, 'terminal hypothesis status is immutable')
        END;
    END;
    """,
)

_ALLOWED_VIEW_NAME_PREFIXES = ("v_",)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LedgerError(Exception):
    """Base class for ledger failures."""


class ChainIntegrityError(LedgerError):
    """The append-only event hash chain is broken."""


class NoActiveChampionError(LedgerError):
    """No champion has been promoted for the task yet (genesis mode)."""


class DuplicateHypothesisError(LedgerError):
    """A byte-identical diff was already proposed against the same parent."""


def _event_hash(
    prev_hash: str,
    event_type: str,
    cycle_id: Optional[str],
    payload_json: str,
    recorded_at_iso: str,
) -> str:
    blob = "\x1f".join(
        [prev_hash, event_type, cycle_id or "", payload_json, recorded_at_iso]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ExperimentLedgerRepository:
    """Typed repository over the SQLite WAL experiment ledger.

    Thread-safety: one connection with ``check_same_thread=False`` guarded by
    a reentrant lock for writers.  WAL readers (dashboard, CLI) should open
    their own connection and never see partial transactions.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,  # explicit transaction control
            check_same_thread=False,
            timeout=timeout_s,
        )
        self._conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        self._ensure_schema()

    # -- connection lifecycle ------------------------------------------------

    def _apply_pragmas(self) -> None:
        cur = self._conn.execute("PRAGMA journal_mode=WAL")
        mode = cur.fetchone()[0]
        if mode != "wal":
            raise LedgerError(f"failed to enable WAL mode (got {mode!r})")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute(f"PRAGMA busy_timeout={int(30_000)}")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA cache_size=-32000")  # 32 MiB page cache

    def _ensure_schema(self) -> None:
        with self._lock:
            # FK checks off during DDL (self-referencing tables), back on after.
            self._conn.execute("PRAGMA foreign_keys=OFF")
            try:
                for stmt in _DDL_STATEMENTS:
                    self._conn.execute(stmt)
                for stmt in _TRIGGER_STATEMENTS:
                    self._conn.execute(stmt)
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_meta ("
                    " key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value)"
                    " VALUES ('schema_version', ?)",
                    (str(_SCHEMA_VERSION),),
                )
            finally:
                self._conn.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        """Write transaction.  Reentrant: joins an outer transaction."""
        with self._lock:
            if self._conn.in_transaction:
                yield self._conn
                return
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            if self._conn.in_transaction:  # defensive: never leave a tx open
                self._conn.execute("ROLLBACK")
            self._conn.close()

    def __enter__(self) -> "ExperimentLedgerRepository":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _require_open(self) -> sqlite3.Connection:
        try:
            self._conn.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            raise LedgerError("repository is closed") from exc
        return self._conn

    # =====================================================================
    # Event chain (write-ahead journal for the orchestrator state machine)
    # =====================================================================

    def append_event(
        self,
        event_type: str | EventType,
        payload: Dict[str, Any],
        cycle_id: Optional[str] = None,
    ) -> LedgerEvent:
        """Append one hash-chained event.  Returns the persisted event."""
        etype = event_type.value if isinstance(event_type, EventType) else event_type
        payload_json = canonical_json(payload)
        iso = iso_now()
        unix = unix_now()
        with self._tx() as conn:
            row = conn.execute(
                "SELECT seq, event_hash FROM ledger_events ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = row["event_hash"] if row is not None else ""
            seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS next FROM ledger_events"
            ).fetchone()
            seq = int(seq_row["next"])
            ehash = _event_hash(prev_hash, etype, cycle_id, payload_json, iso)
            conn.execute(
                "INSERT INTO ledger_events (seq, event_type, cycle_id, payload_json,"
                " prev_hash, event_hash, recorded_at_unix, recorded_at_iso)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (seq, etype, cycle_id, payload_json, prev_hash, ehash, unix, iso),
            )
        return LedgerEvent(
            seq=seq,
            event_type=etype,
            cycle_id=cycle_id,
            payload_json=payload_json,
            prev_hash=prev_hash,
            event_hash=ehash,
            recorded_at_unix=unix,
            recorded_at_iso=iso,
        )

    def verify_chain_integrity(self) -> Tuple[bool, Optional[int]]:
        """Recompute the full chain.  Returns ``(ok, first_bad_seq)``.

        ``ok=True`` implies (a) every event_hash matches its recomputed
        digest and (b) every prev_hash links to the previous event_hash.
        """
        conn = self._require_open()
        cursor = conn.execute(
            "SELECT seq, event_type, cycle_id, payload_json, prev_hash,"
            " event_hash, recorded_at_unix, recorded_at_iso"
            " FROM ledger_events ORDER BY seq ASC"
        )
        expected_prev = ""
        last_seq = 0
        for row in cursor:
            last_seq = int(row["seq"])
            recomputed = _event_hash(
                row["prev_hash"],
                row["event_type"],
                row["cycle_id"],
                row["payload_json"],
                row["recorded_at_iso"],
            )
            if row["prev_hash"] != expected_prev:
                return False, last_seq
            if recomputed != row["event_hash"]:
                return False, last_seq
            expected_prev = row["event_hash"]
        return True, None

    def get_event(self, seq: int) -> Optional[LedgerEvent]:
        row = self._require_open().execute(
            "SELECT * FROM ledger_events WHERE seq = ?", (seq,)
        ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def get_last_event_of_type(
        self, event_type: str | EventType, *, cycle_id: Optional[str] = None
    ) -> Optional[LedgerEvent]:
        etype = event_type.value if isinstance(event_type, EventType) else event_type
        if cycle_id is None:
            row = self._require_open().execute(
                "SELECT * FROM ledger_events WHERE event_type = ?"
                " ORDER BY seq DESC LIMIT 1",
                (etype,),
            ).fetchone()
        else:
            row = self._require_open().execute(
                "SELECT * FROM ledger_events WHERE event_type = ? AND cycle_id = ?"
                " ORDER BY seq DESC LIMIT 1",
                (etype, cycle_id),
            ).fetchone()
        return self._row_to_event(row) if row is not None else None

    def get_events_for_cycle(
        self, cycle_id: str, *, event_type: Optional[str] = None
    ) -> List[LedgerEvent]:
        if event_type is None:
            rows = self._require_open().execute(
                "SELECT * FROM ledger_events WHERE cycle_id = ? ORDER BY seq ASC",
                (cycle_id,),
            ).fetchall()
        else:
            rows = self._require_open().execute(
                "SELECT * FROM ledger_events WHERE cycle_id = ? AND event_type = ?"
                " ORDER BY seq ASC",
                (cycle_id, event_type),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_latest_orchestrator_state(self) -> Optional[LedgerEvent]:
        """Latest ORCHESTRATOR_STATE journal entry (crash-recovery input)."""
        return self.get_last_event_of_type(EventType.ORCHESTRATOR_STATE)

    def get_open_cycles(self) -> List[str]:
        """Cycle ids that started but never committed / aborted / superseded."""
        rows = self._require_open().execute(
            "SELECT DISTINCT cycle_id FROM ledger_events"
            " WHERE event_type = 'CYCLE_START' AND cycle_id IS NOT NULL"
            "   AND cycle_id NOT IN ("
            "     SELECT cycle_id FROM ledger_events"
            "      WHERE event_type IN"
            "        ('CYCLE_COMMIT','CYCLE_ABORT','CYCLE_SUPERSEDED')"
            "        AND cycle_id IS NOT NULL)"
            " ORDER BY cycle_id ASC"
        ).fetchall()
        return [str(r["cycle_id"]) for r in rows]

    # =====================================================================
    # Harnesses / checkpoints
    # =====================================================================

    def upsert_harness(self, manifest: HarnessManifest) -> None:
        with self._tx() as conn:
            existing = conn.execute(
                "SELECT harness_diff, ast_fingerprint, task_name, config_json"
                " FROM harnesses WHERE harness_hash = ?",
                (manifest.harness_hash,),
            ).fetchone()
            if existing is not None:
                same = (
                    existing["harness_diff"] == manifest.harness_diff
                    and existing["ast_fingerprint"] == manifest.ast_fingerprint
                    and existing["task_name"] == manifest.task_name
                    and existing["config_json"] == canonical_json(manifest.config)
                )
                if not same:
                    raise LedgerError(
                        f"harness_hash collision or tamper: {manifest.harness_hash}"
                    )
                return
            conn.execute(
                "INSERT INTO harnesses (harness_hash, parent_harness_hash, task_name,"
                " harness_diff, ast_fingerprint, compiler_version,"
                " safety_receipt_json, difficulty_strata_json, config_json,"
                " created_at_unix) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    manifest.harness_hash,
                    manifest.parent_harness_hash,
                    manifest.task_name,
                    manifest.harness_diff,
                    manifest.ast_fingerprint,
                    manifest.compiler_version,
                    manifest.safety_receipt.model_dump_json(),
                    canonical_json([s.value for s in manifest.difficulty_strata]),
                    canonical_json(manifest.config),
                    manifest.created_at_unix,
                ),
            )

    def get_harness(self, harness_hash: str) -> HarnessManifest:
        row = self._require_open().execute(
            "SELECT * FROM harnesses WHERE harness_hash = ?", (harness_hash,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown harness: {harness_hash}")
        strata = [
            DifficultyStrataBin(s)
            for s in json.loads(row["difficulty_strata_json"])
        ]
        return HarnessManifest(
            harness_hash=row["harness_hash"],
            task_name=row["task_name"],
            parent_harness_hash=row["parent_harness_hash"],
            harness_diff=row["harness_diff"],
            ast_fingerprint=row["ast_fingerprint"],
            compiler_version=row["compiler_version"],
            safety_receipt=SafetyValidationReceipt.model_validate_json(
                row["safety_receipt_json"]
            ),
            difficulty_strata=strata,
            config=json.loads(row["config_json"]),
            created_at_unix=row["created_at_unix"],
        )

    def harness_exists(self, harness_hash: str) -> bool:
        row = self._require_open().execute(
            "SELECT 1 FROM harnesses WHERE harness_hash = ?", (harness_hash,)
        ).fetchone()
        return row is not None

    def record_policy_checkpoint(self, checkpoint: PolicyCheckpointRecord) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO policy_checkpoints"
                " (checkpoint_hash, harness_hash, format, storage_path,"
                "  num_params, created_at_unix) VALUES (?,?,?,?,?,?)",
                (
                    checkpoint.checkpoint_hash,
                    checkpoint.harness_hash,
                    checkpoint.format,
                    checkpoint.storage_path,
                    checkpoint.num_params,
                    checkpoint.created_at_unix,
                ),
            )

    def get_policy_checkpoint(self, checkpoint_hash: str) -> PolicyCheckpointRecord:
        row = self._require_open().execute(
            "SELECT * FROM policy_checkpoints WHERE checkpoint_hash = ?",
            (checkpoint_hash,),
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown checkpoint: {checkpoint_hash}")
        return PolicyCheckpointRecord(
            checkpoint_hash=row["checkpoint_hash"],
            harness_hash=row["harness_hash"],
            format=row["format"],
            storage_path=row["storage_path"],
            num_params=int(row["num_params"]),
            created_at_unix=row["created_at_unix"],
        )

    # =====================================================================
    # Hypotheses
    # =====================================================================

    def record_hypothesis(
        self, proposal: HypothesisProposal, critic_score: float
    ) -> str:
        """Persist a new hypothesis.  Returns ``hypothesis_id``.

        Raises :class:`DuplicateHypothesisError` when the exact same diff was
        already proposed against the same parent harness.
        """
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO hypotheses (hypothesis_id, task_name,"
                    " parent_harness_hash, harness_hash, harness_diff,"
                    " ast_fingerprint, param_summary_json, rationale,"
                    " prediction_json, critic_score, status, proposed_at_unix)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?, 'PROPOSED', ?)",
                    (
                        proposal.hypothesis_id,
                        proposal.task_name,
                        proposal.parent_harness_hash,
                        None,
                        proposal.harness_diff,
                        proposal.ast_fingerprint,
                        canonical_json(proposal.param_summary),
                        proposal.rationale,
                        proposal.falsifiable_prediction.model_dump_json(),
                        float(critic_score),
                        proposal.proposed_at_unix,
                    ),
                )
                self.append_event(
                    EventType.HYPOTHESIS_RECORDED,
                    {
                        "hypothesis_id": proposal.hypothesis_id,
                        "task_name": proposal.task_name,
                        "parent_harness_hash": proposal.parent_harness_hash,
                        "ast_fingerprint": proposal.ast_fingerprint,
                        "critic_score": float(critic_score),
                        "predicted_delta":
                            proposal.falsifiable_prediction.predicted_delta,
                    },
                )
        except sqlite3.IntegrityError as exc:
            if "uq_hypotheses_parent_diff" in str(exc):
                raise DuplicateHypothesisError(
                    "identical diff already proposed against parent "
                    f"{proposal.parent_harness_hash}"
                ) from exc
            raise
        return proposal.hypothesis_id

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
        *,
        harness_hash: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        with self._tx() as conn:
            if harness_hash is not None:
                conn.execute(
                    "UPDATE hypotheses SET status = ?, harness_hash = ?,"
                    " rejection_reason = ?, decided_at_unix = ?"
                    " WHERE hypothesis_id = ?",
                    (
                        status.value,
                        harness_hash,
                        rejection_reason,
                        unix_now(),
                        hypothesis_id,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE hypotheses SET status = ?, rejection_reason = ?,"
                    " decided_at_unix = ? WHERE hypothesis_id = ?",
                    (status.value, rejection_reason, unix_now(), hypothesis_id),
                )

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisRecord:
        row = self._require_open().execute(
            "SELECT * FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown hypothesis: {hypothesis_id}")
        return HypothesisRecord(
            hypothesis_id=row["hypothesis_id"],
            task_name=row["task_name"],
            parent_harness_hash=row["parent_harness_hash"],
            harness_hash=row["harness_hash"],
            harness_diff=row["harness_diff"],
            ast_fingerprint=row["ast_fingerprint"],
            param_summary=json.loads(row["param_summary_json"]),
            rationale=row["rationale"],
            prediction=FalsifiablePrediction.model_validate_json(
                row["prediction_json"]
            ),
            critic_score=float(row["critic_score"]),
            status=HypothesisStatus(row["status"]),
            rejection_reason=row["rejection_reason"],
            proposed_at_unix=row["proposed_at_unix"],
            decided_at_unix=row["decided_at_unix"],
        )

    def get_open_hypotheses(self, task_name: str) -> List[HypothesisRecord]:
        """Hypotheses in non-terminal states.

        The orchestrator serializes one live hypothesis per task, so after a
        crash these belong to abandoned cycles and are superseded at recovery.
        """
        rows = self._require_open().execute(
            "SELECT hypothesis_id FROM hypotheses"
            " WHERE task_name = ? AND status IN ('PROPOSED','COMPILED','EVALUATING')"
            " ORDER BY proposed_at_unix ASC",
            (task_name,),
        ).fetchall()
        return [self.get_hypothesis(str(r["hypothesis_id"])) for r in rows]

    def get_latest_checkpoint_for_harness(
        self, harness_hash: str
    ) -> Optional[PolicyCheckpointRecord]:
        row = self._require_open().execute(
            "SELECT checkpoint_hash FROM policy_checkpoints"
            " WHERE harness_hash = ? ORDER BY created_at_unix DESC LIMIT 1",
            (harness_hash,),
        ).fetchone()
        if row is None:
            return None
        return self.get_policy_checkpoint(str(row["checkpoint_hash"]))

    def query_similar_rejected_diffs(
        self,
        parent_harness_hash: str,
        ast_fingerprint: str,
        top_k: int = 5,
    ) -> List[RejectedDiffSummary]:
        """Anti-looping query for the Closed-Loop Critic Guard.

        Returns rejected / early-aborted hypotheses against the same parent:
        exact structural fingerprint matches first (highest loop risk), then
        highest critic score (the near-misses the planner is most likely to
        re-derive), then most recent decisions.
        """
        rows = self._require_open().execute(
            "SELECT hypothesis_id, harness_diff, ast_fingerprint,"
            "       param_summary_json, status AS rejection_status,"
            "       rejection_reason, critic_score, decided_at_unix"
            "  FROM hypotheses"
            " WHERE parent_harness_hash = ?"
            "   AND status IN ('REJECTED_COMPILER','REJECTED_SIM',"
            "                  'REJECTED_PHYSICAL','EARLY_ABORTED')"
            " ORDER BY (ast_fingerprint = ?) DESC,"
            "          critic_score DESC,"
            "          decided_at_unix DESC"
            " LIMIT ?",
            (parent_harness_hash, ast_fingerprint, int(top_k)),
        ).fetchall()
        return [
            RejectedDiffSummary(
                hypothesis_id=r["hypothesis_id"],
                harness_diff=r["harness_diff"],
                ast_fingerprint=r["ast_fingerprint"],
                param_summary=json.loads(r["param_summary_json"]),
                rejection_status=r["rejection_status"],
                rejection_reason=r["rejection_reason"],
                critic_score=float(r["critic_score"]),
                decided_at_unix=r["decided_at_unix"],
            )
            for r in rows
        ]

    # =====================================================================
    # Evaluation batches & rollout telemetry
    # =====================================================================

    def record_evaluation_batch(
        self,
        batch: EvaluationBatchRecord,
        rollouts: Sequence[RolloutTelemetryBundle],
    ) -> str:
        """Atomically persist one batch and all its rollout telemetry.

        Also validates the blackbox forward references: any rollout carrying
        a ``flight_snapshot_hash`` must reference a dump whose own
        ``rollout_id`` matches.  Auto-transitions the hypothesis
        COMPILED -> EVALUATING on the first batch.
        """
        if not rollouts:
            raise LedgerError("a batch must contain at least one rollout")
        stamped: List[RolloutTelemetryBundle] = []
        for r in rollouts:
            # Executors produce bundles without knowing the batch id; the
            # commit stamps it (only an empty or matching id is accepted).
            if r.batch_id == "":
                r = r.model_copy(update={"batch_id": batch.batch_id})
            if r.batch_id != batch.batch_id:
                raise LedgerError(
                    f"rollout {r.rollout_id} belongs to batch {r.batch_id},"
                    f" expected {batch.batch_id}"
                )
            if r.harness_hash != batch.harness_hash:
                raise LedgerError(
                    f"rollout {r.rollout_id} harness {r.harness_hash} != batch"
                    f" harness {batch.harness_hash}"
                )
            stamped.append(r)
        rollouts = stamped
        committed_at = batch.committed_at_unix or unix_now()
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO evaluation_batches (batch_id, cycle_id,"
                    " hypothesis_id, harness_hash, champion_hash, domain,"
                    " task_name, n_rollouts, decision, statistics_json, trigger,"
                    " started_at_unix, committed_at_unix)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        batch.batch_id,
                        batch.cycle_id,
                        batch.hypothesis_id,
                        batch.harness_hash,
                        batch.champion_hash,
                        batch.domain.value,
                        batch.task_name,
                        len(rollouts),
                        batch.decision.value,
                        batch.statistics.model_dump_json(),
                        batch.trigger,
                        batch.started_at_unix,
                        committed_at,
                    ),
                )
                for r in rollouts:
                    if r.flight_snapshot_hash is not None:
                        dump = conn.execute(
                            "SELECT rollout_id FROM blackbox_dumps"
                            " WHERE snapshot_hash = ?",
                            (r.flight_snapshot_hash,),
                        ).fetchone()
                        if dump is None:
                            raise LedgerError(
                                f"rollout {r.rollout_id} references unknown"
                                f" blackbox dump {r.flight_snapshot_hash}"
                            )
                        if dump["rollout_id"] != r.rollout_id:
                            raise LedgerError(
                                f"blackbox dump {r.flight_snapshot_hash} was sealed"
                                f" for rollout {dump['rollout_id']}, not"
                                f" {r.rollout_id}"
                            )
                    conn.execute(
                        "INSERT INTO rollout_telemetry (rollout_id, batch_id,"
                        " harness_hash, task_name, domain, stratum, success,"
                        " time_to_completion, cumulative_jerk, torque_penalty,"
                        " inference_deadline_misses, force_spike_max,"
                        " collision_count, reset_outcome, flight_snapshot_hash,"
                        " t_start_unix, t_end_unix, video_uri,"
                        " vlm_failure_summary) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            r.rollout_id,
                            r.batch_id,
                            r.harness_hash,
                            r.task_name,
                            r.domain.value,
                            r.stratum.value,
                            int(r.success),
                            r.time_to_completion,
                            r.cumulative_jerk,
                            r.torque_penalty,
                            r.inference_deadline_misses,
                            r.force_spike_max,
                            r.collision_count,
                            r.reset_outcome.value if r.reset_outcome else None,
                            r.flight_snapshot_hash,
                            r.t_start_unix,
                            r.t_end_unix,
                            r.video_uri,
                            r.vlm_failure_summary,
                        ),
                    )
                conn.execute(
                    "UPDATE hypotheses SET status = 'EVALUATING'"
                    " WHERE hypothesis_id = ? AND status = 'COMPILED'",
                    (batch.hypothesis_id,),
                )
                self.append_event(
                    EventType.BATCH_COMMITTED,
                    {
                        "batch_id": batch.batch_id,
                        "cycle_id": batch.cycle_id,
                        "hypothesis_id": batch.hypothesis_id,
                        "harness_hash": batch.harness_hash,
                        "domain": batch.domain.value,
                        "n_rollouts": len(rollouts),
                        "decision": batch.decision.value,
                    },
                    cycle_id=batch.cycle_id,
                )
        except sqlite3.IntegrityError as exc:
            raise LedgerError(f"batch insert failed: {exc}") from exc
        return batch.batch_id

    def get_batch(self, batch_id: str) -> EvaluationBatchRecord:
        row = self._require_open().execute(
            "SELECT * FROM evaluation_batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown batch: {batch_id}")
        return EvaluationBatchRecord(
            batch_id=row["batch_id"],
            cycle_id=row["cycle_id"],
            hypothesis_id=row["hypothesis_id"],
            harness_hash=row["harness_hash"],
            champion_hash=row["champion_hash"],
            domain=RolloutDomain(row["domain"]),
            task_name=row["task_name"],
            n_rollouts=int(row["n_rollouts"]),
            decision=BatchDecision(row["decision"]),
            statistics=BatchStatistics.model_validate_json(
                row["statistics_json"]
            ),
            trigger=row["trigger"],
            started_at_unix=row["started_at_unix"],
            committed_at_unix=row["committed_at_unix"],
        )

    def get_rollouts_for_batch(self, batch_id: str) -> List[RolloutTelemetryBundle]:
        rows = self._require_open().execute(
            "SELECT * FROM rollout_telemetry WHERE batch_id = ?"
            " ORDER BY t_start_unix ASC",
            (batch_id,),
        ).fetchall()
        return [self._row_to_rollout(r) for r in rows]

    def get_rollouts_for_harness(
        self, harness_hash: str, *, domain: Optional[RolloutDomain] = None
    ) -> List[RolloutTelemetryBundle]:
        if domain is None:
            rows = self._require_open().execute(
                "SELECT * FROM rollout_telemetry WHERE harness_hash = ?"
                " ORDER BY t_start_unix ASC",
                (harness_hash,),
            ).fetchall()
        else:
            rows = self._require_open().execute(
                "SELECT * FROM rollout_telemetry WHERE harness_hash = ?"
                " AND domain = ? ORDER BY t_start_unix ASC",
                (harness_hash, domain.value),
            ).fetchall()
        return [self._row_to_rollout(r) for r in rows]

    def get_latest_batch_for_harness(
        self, harness_hash: str, domain: RolloutDomain
    ) -> Optional[EvaluationBatchRecord]:
        row = self._require_open().execute(
            "SELECT batch_id FROM evaluation_batches"
            " WHERE harness_hash = ? AND domain = ?"
            " ORDER BY committed_at_unix DESC LIMIT 1",
            (harness_hash, domain.value),
        ).fetchone()
        if row is None:
            return None
        return self.get_batch(str(row["batch_id"]))

    def get_recent_batches(
        self, task_name: str, limit: int = 10
    ) -> List[EvaluationBatchRecord]:
        rows = self._require_open().execute(
            "SELECT batch_id FROM evaluation_batches WHERE task_name = ?"
            " ORDER BY committed_at_unix DESC LIMIT ?",
            (task_name, int(limit)),
        ).fetchall()
        return [self.get_batch(str(r["batch_id"])) for r in rows]

    # =====================================================================
    # Champions
    # =====================================================================

    def record_champion(self, manifest: ChampionManifest) -> None:
        """Atomically deactivate the previous champion and enthrone a new one."""
        with self._tx() as conn:
            conn.execute(
                "UPDATE champions SET active = 0 WHERE task_name = ? AND active = 1",
                (manifest.task_name,),
            )
            conn.execute(
                "INSERT INTO champions (task_name, harness_hash,"
                " policy_checkpoint_hash, batch_id, wilson_lower_bound,"
                " stratum_stats_json, promoted_at_unix, active)"
                " VALUES (?,?,?,?,?,?,?,1)",
                (
                    manifest.task_name,
                    manifest.harness_hash,
                    manifest.policy_checkpoint_hash,
                    manifest.batch_id,
                    manifest.wilson_lower_bound,
                    canonical_json(
                        [s.model_dump(mode="json") for s in manifest.stratum_stats]
                    ),
                    manifest.promoted_at_unix,
                ),
            )
            self.append_event(
                EventType.CHAMPION_PROMOTED,
                {
                    "task_name": manifest.task_name,
                    "harness_hash": manifest.harness_hash,
                    "policy_checkpoint_hash": manifest.policy_checkpoint_hash,
                    "batch_id": manifest.batch_id,
                    "wilson_lower_bound": manifest.wilson_lower_bound,
                },
            )

    def get_active_champion(self, task_name: str) -> ChampionManifest:
        row = self._require_open().execute(
            "SELECT * FROM champions WHERE task_name = ? AND active = 1",
            (task_name,),
        ).fetchone()
        if row is None:
            raise NoActiveChampionError(
                f"no active champion for task {task_name!r} (genesis mode)"
            )
        stats = [
            StratumStats.model_validate(d)
            for d in json.loads(row["stratum_stats_json"])
        ]
        return ChampionManifest(
            task_name=row["task_name"],
            harness_hash=row["harness_hash"],
            policy_checkpoint_hash=row["policy_checkpoint_hash"],
            batch_id=row["batch_id"],
            wilson_lower_bound=float(row["wilson_lower_bound"]),
            stratum_stats=stats,
            promoted_at_unix=float(row["promoted_at_unix"]),
        )

    def get_champion_history(
        self, task_name: str, limit: int = 10
    ) -> List[ChampionManifest]:
        rows = self._require_open().execute(
            "SELECT * FROM champions WHERE task_name = ?"
            " ORDER BY promoted_at_unix DESC LIMIT ?",
            (task_name, int(limit)),
        ).fetchall()
        out: List[ChampionManifest] = []
        for row in rows:
            stats = [
                StratumStats.model_validate(d)
                for d in json.loads(row["stratum_stats_json"])
            ]
            out.append(
                ChampionManifest(
                    task_name=row["task_name"],
                    harness_hash=row["harness_hash"],
                    policy_checkpoint_hash=row["policy_checkpoint_hash"],
                    batch_id=row["batch_id"],
                    wilson_lower_bound=float(row["wilson_lower_bound"]),
                    stratum_stats=stats,
                    promoted_at_unix=float(row["promoted_at_unix"]),
                )
            )
        return out

    # =====================================================================
    # Sim-to-real calibration
    # =====================================================================

    def record_sim_to_real_gap(self, report: SimToRealGapReport) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO sim_to_real_calibration (task_name, harness_hash,"
                " snapshot_hash, sim_metric, real_metric, gap,"
                " friction_offset_json, damping_offset_json, residual_before,"
                " residual_after, status, calibrated_at_unix)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    report.task_name,
                    report.harness_hash,
                    report.snapshot_hash,
                    report.sim_metric,
                    report.real_metric,
                    report.gap,
                    canonical_json(report.friction_offsets),
                    canonical_json(report.damping_offsets),
                    report.residual_before,
                    report.residual_after,
                    report.status.value,
                    report.calibrated_at_unix,
                ),
            )

    def get_sim_to_real_gap_history(
        self, task_name: str, window: int = 20
    ) -> List[SimToRealGapReport]:
        """Latest ``window`` calibration reports, oldest -> newest."""
        rows = self._require_open().execute(
            "SELECT * FROM sim_to_real_calibration WHERE task_name = ?"
            " ORDER BY calibrated_at_unix DESC LIMIT ?",
            (task_name, int(window)),
        ).fetchall()
        out: List[SimToRealGapReport] = []
        for row in reversed(rows):
            out.append(
                SimToRealGapReport(
                    task_name=row["task_name"],
                    harness_hash=row["harness_hash"],
                    sim_metric=float(row["sim_metric"]),
                    real_metric=float(row["real_metric"]),
                    gap=float(row["gap"]),
                    friction_offsets=json.loads(row["friction_offset_json"]),
                    damping_offsets=json.loads(row["damping_offset_json"]),
                    status=SysIDStatus(row["status"]),
                    snapshot_hash=row["snapshot_hash"],
                    residual_before=row["residual_before"],
                    residual_after=row["residual_after"],
                    calibrated_at_unix=float(row["calibrated_at_unix"]),
                )
            )
        return out

    # =====================================================================
    # Black-box dumps
    # =====================================================================

    def record_blackbox_dump(self, record: BlackBoxSnapshotRecord) -> None:
        parquet_ok = Path(record.parquet_path).exists()
        state_ok = Path(record.sim_state_path).exists()
        if not (parquet_ok and state_ok):
            raise LedgerError(
                f"blackbox artifacts missing on disk: parquet={parquet_ok}"
                f" state={state_ok} ({record.snapshot_hash})"
            )
        with self._tx() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO blackbox_dumps (snapshot_hash, rollout_id,"
                " trigger, parquet_path, sim_state_path, format, t_window_seconds,"
                " sample_rate_hz, num_samples, mujoco_model_hash, created_at_unix)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.snapshot_hash,
                    record.rollout_id,
                    record.trigger.value,
                    record.parquet_path,
                    record.sim_state_path,
                    record.format,
                    record.t_window_seconds,
                    record.sample_rate_hz,
                    record.num_samples,
                    record.mujoco_model_hash,
                    record.created_at_unix,
                ),
            )

    def get_blackbox_snapshot(self, snapshot_hash: str) -> BlackBoxSnapshotRecord:
        row = self._require_open().execute(
            "SELECT * FROM blackbox_dumps WHERE snapshot_hash = ?", (snapshot_hash,)
        ).fetchone()
        if row is None:
            raise LedgerError(f"unknown blackbox dump: {snapshot_hash}")
        return BlackBoxSnapshotRecord(
            snapshot_hash=row["snapshot_hash"],
            rollout_id=row["rollout_id"],
            trigger=FlightTrigger(row["trigger"]),
            parquet_path=row["parquet_path"],
            sim_state_path=row["sim_state_path"],
            format=row["format"],
            t_window_seconds=float(row["t_window_seconds"]),
            sample_rate_hz=float(row["sample_rate_hz"]),
            num_samples=int(row["num_samples"]),
            mujoco_model_hash=row["mujoco_model_hash"],
            created_at_unix=float(row["created_at_unix"]),
        )

    def list_blackbox_dumps(
        self, *, limit: int = 50, trigger: Optional[FlightTrigger] = None
    ) -> List[BlackBoxSnapshotRecord]:
        if trigger is None:
            rows = self._require_open().execute(
                "SELECT snapshot_hash FROM blackbox_dumps"
                " ORDER BY created_at_unix DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        else:
            rows = self._require_open().execute(
                "SELECT snapshot_hash FROM blackbox_dumps WHERE trigger = ?"
                " ORDER BY created_at_unix DESC LIMIT ?",
                (trigger.value, int(limit)),
            ).fetchall()
        return [self.get_blackbox_snapshot(str(r["snapshot_hash"])) for r in rows]

    # =====================================================================
    # Operator overrides (deliverable 6 writes through this)
    # =====================================================================

    def record_operator_override(self, override: OperatorOverride) -> LedgerEvent:
        return self.append_event(
            EventType.OPERATOR_OVERRIDE,
            {
                "operator": override.operator,
                "action": override.action,
                "justification": override.justification,
                "signature": override.signature,
                "created_at_unix": override.created_at_unix,
            },
        )

    # =====================================================================
    # Analytical queries used by the default planner-context builder and the
    # terminal dashboard (deliverable 3 installs the SQL VIEW versions).
    # =====================================================================

    def get_failure_mode_pareto(
        self, task_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        rows = self._require_open().execute(
            "SELECT COALESCE(vlm_failure_summary, '<unclassified>') AS mode,"
            "       COUNT(*) AS n,"
            "       AVG(cumulative_jerk) AS mean_jerk,"
            "       MAX(force_spike_max) AS max_spike,"
            "       SUM(CASE WHEN collision_count > 0 THEN 1 ELSE 0 END)"
            "         AS n_collision"
            "  FROM rollout_telemetry"
            " WHERE task_name = ? AND success = 0"
            " GROUP BY mode ORDER BY n DESC LIMIT ?",
            (task_name, int(limit)),
        ).fetchall()
        return [
            {
                "mode": r["mode"],
                "n": int(r["n"]),
                "mean_jerk": r["mean_jerk"],
                "max_spike": r["max_spike"],
                "n_collision": int(r["n_collision"] or 0),
            }
            for r in rows
        ]

    def get_stratum_stats(
        self,
        harness_hash: str,
        domain: RolloutDomain,
        *,
        since_unix: Optional[float] = None,
    ) -> List[StratumStats]:
        if since_unix is None:
            rows = self._require_open().execute(
                "SELECT stratum, COUNT(*) AS n, SUM(success) AS s,"
                "       AVG(time_to_completion) AS ttc,"
                "       AVG(cumulative_jerk) AS jerk,"
                "       AVG(torque_penalty) AS torque"
                "  FROM rollout_telemetry"
                " WHERE harness_hash = ? AND domain = ?"
                " GROUP BY stratum",
                (harness_hash, domain.value),
            ).fetchall()
        else:
            rows = self._require_open().execute(
                "SELECT stratum, COUNT(*) AS n, SUM(success) AS s,"
                "       AVG(time_to_completion) AS ttc,"
                "       AVG(cumulative_jerk) AS jerk,"
                "       AVG(torque_penalty) AS torque"
                "  FROM rollout_telemetry"
                " WHERE harness_hash = ? AND domain = ? AND t_start_unix >= ?"
                " GROUP BY stratum",
                (harness_hash, domain.value, since_unix),
            ).fetchall()
        out: List[StratumStats] = []
        for r in rows:
            n, s = int(r["n"]), int(r["s"] or 0)
            low, high = wilson_score_interval(s, n)
            out.append(
                StratumStats(
                    stratum=DifficultyStrataBin(r["stratum"]),
                    n=n,
                    successes=s,
                    success_rate=s / n,
                    wilson_low=low,
                    wilson_high=high,
                    mean_ttc=r["ttc"],
                    mean_cumulative_jerk=r["jerk"],
                    mean_torque_penalty=r["torque"],
                )
            )
        return out

    # =====================================================================
    # View management + typed view reads (used by the terminal dashboard)
    # =====================================================================

    def apply_analytical_views(self, view_sql: str) -> None:
        """Apply a SQL script restricted to CREATE/DROP VIEW statements only.

        The telemetry layer owns the view definitions (deliverable 3) but
        must not be able to mutate tables; this guard parses every statement
        and rejects anything that is not view DDL.
        """
        for raw in view_sql.split(";"):
            stmt = raw.strip()
            if not stmt:
                continue
            head = stmt.split("\n", 1)[0].strip().upper()
            if not (
                head.startswith("CREATE VIEW")
                or head.startswith("CREATE OR REPLACE VIEW")
                or head.startswith("CREATE IF NOT EXISTS VIEW")
                or head.startswith("DROP VIEW")
                or head.startswith("DROP VIEW IF EXISTS")
                or stmt.upper().startswith("CREATE VIEW")
                or stmt.upper().startswith("DROP VIEW")
            ):
                raise LedgerError(
                    f"apply_analytical_views rejects non-view statement: {head[:60]}"
                )
        with self._lock:
            self._conn.executescript(view_sql)

    _ALLOWED_VIEWS = (
        "v_hourly_throughput",
        "v_failure_mode_pareto",
        "v_sim_vs_real_drift",
    )

    def query_view(self, view_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        if view_name not in self._ALLOWED_VIEWS:
            raise LedgerError(
                f"view {view_name!r} is not whitelisted: {self._ALLOWED_VIEWS}"
            )
        rows = self._require_open().execute(
            f"SELECT * FROM {view_name} LIMIT ?", (int(limit),)
        ).fetchall()
        return [dict(r) for r in rows]

    # =====================================================================
    # Row mappers
    # =====================================================================

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> LedgerEvent:
        return LedgerEvent(
            seq=int(row["seq"]),
            event_type=row["event_type"],
            cycle_id=row["cycle_id"],
            payload_json=row["payload_json"],
            prev_hash=row["prev_hash"],
            event_hash=row["event_hash"],
            recorded_at_unix=float(row["recorded_at_unix"]),
            recorded_at_iso=row["recorded_at_iso"],
        )

    @staticmethod
    def _row_to_rollout(row: sqlite3.Row) -> RolloutTelemetryBundle:
        return RolloutTelemetryBundle(
            rollout_id=row["rollout_id"],
            batch_id=row["batch_id"],
            harness_hash=row["harness_hash"],
            task_name=row["task_name"],
            domain=RolloutDomain(row["domain"]),
            stratum=DifficultyStrataBin(row["stratum"]),
            success=bool(row["success"]),
            time_to_completion=row["time_to_completion"],
            cumulative_jerk=float(row["cumulative_jerk"]),
            torque_penalty=float(row["torque_penalty"]),
            inference_deadline_misses=int(row["inference_deadline_misses"]),
            force_spike_max=float(row["force_spike_max"]),
            collision_count=int(row["collision_count"]),
            reset_outcome=(
                None
                if row["reset_outcome"] is None
                else ResetOutcome(row["reset_outcome"])
            ),
            flight_snapshot_hash=row["flight_snapshot_hash"],
            t_start_unix=float(row["t_start_unix"]),
            t_end_unix=float(row["t_end_unix"]),
            video_uri=row["video_uri"],
            vlm_failure_summary=row["vlm_failure_summary"],
        )
