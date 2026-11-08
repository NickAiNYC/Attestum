#!/usr/bin/env python3
"""smoke_context_builder.py -- acceptance script for robocore Deliverable 4
(Cognitive context compiler & prompt registry).

Six checks, each raising on failure (fail-closed; no PASS line is printed
unless every check passed):

  1. Seeds a deterministic fixture of exactly 20 hash-chained ledger
     entries, builds the context, and asserts token_estimate <= budget.
     Also proves the budget gate raises ContextOverflow (raise, not warn)
     on an impossible budget, and that two identically-seeded ledgers
     compile to byte-identical contexts.
  2. The champion hash appears EXACTLY once in the rendered planner prompt.
  3. failure_clusters yields exactly min(5, N) entries per label across
     3 labels (5 / 3 / 3), hashes only.
  4. The registry refuses unregistered prompts (raises, caught, counted).
  5. Round-trip: render -> sha256(file bytes) -> lookup_by_hash matches the
     registered hash for all 3 prompts; sample LLM outputs validate against
     each prompt's output contract model.
  6. Anti-loop injection: last 3 rejected proposals sharing one mutation
     family populate ForceExploration with the next catalog family.

Run: python scripts/smoke_context_builder.py   ->   "PASS 6/6"
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from robocore.cognitive import (
    ContextBuilder,
    ContextOverflow,
    CriticVerdict,
    FailureDiagnosis,
    MutationType,
    PromptRegistry,
    PromptRole,
    build_critic_slots,
    build_diagnoser_slots,
    build_planner_slots,
    derive_keyframe_hash,
    ensure_keyframe_refs_subset,
    render_context_markdown,
)
from robocore.cognitive.prompt_registry import PROMPTS_DIR, sha256_file
from robocore.contracts import (
    BatchDecision,
    BatchStatistics,
    BlackBoxSnapshotRecord,
    ChampionManifest,
    DifficultyStrataBin,
    EvaluationBatchRecord,
    FlightTrigger,
    FalsifiablePrediction,
    HarnessManifest,
    HypothesisProposal,
    HypothesisStatus,
    OverallStats,
    PolicyCheckpointRecord,
    RolloutDomain,
    RolloutTelemetryBundle,
    SafetyValidationReceipt,
    SPRTTrace,
    StratumStats,
    compute_harness_hash,
)
from robocore.stats import wilson_score_interval
from robocore.storage.ledger_repo import ExperimentLedgerRepository

TASK = "peg_pick_place"
COMPILER_VERSION = "harnessc-0.3.1"
T0 = 1_750_000_000.0

LABEL_A = "GRASP_MISALIGNED_ON_APPROACH"
LABEL_B = "PREMATURE_GRIP_RELEASE"
LABEL_C = "WRIST_SINGULARITY_ESCAPE"

_SUMMARY_A = (
    "Approach vector converged nominally but the grasp axis was misaligned"
    " by roughly 12 degrees at contact: the wrist camera shows the peg edge"
    " loading the finger pad off-center, and joint-5 torque deviates from"
    " the pre-contact estimate 0.2 s before release. The failure is"
    " repeatable across oblique approach strata, which points at the"
    " grasp-offset compensation term rather than perception noise."
)
_SUMMARY_B = (
    "Grip force reached the commanded setpoint, then decayed below the"
    " hold threshold mid-lift before any external contact: telemetry shows"
    " the commanded wrench constant while measured reaction torque drops,"
    " i.e. the fingers slipped without a collision trigger. Occurs late in"
    " the lift phase on heavy payloads, suggesting the squeeze schedule is"
    " open-loop where it should track slip."
)
_SUMMARY_C = (
    "The arm drove itself into a wrist singularity while reorienting: joint"
    " 6 velocity saturated against its limit and the Jacobian condition"
    " number spiked two orders of magnitude within 40 ms, followed by the"
    " force spike that sealed this black box. The recovery policy escaped"
    " by retracting along the approach axis instead of a null-space move."
)

GENESIS_DIFF = (
    "--- a/harness.yaml\n"
    "+++ b/harness.yaml\n"
    "@@ -12,4 +12,4 @@\n"
    " control_hz: 1000\n"
    "-termination_jerk_limit: 45.0\n"
    "+termination_jerk_limit: 42.0\n"
)
CHAMPION_DIFF = (
    "--- a/harness.yaml\n"
    "+++ b/harness.yaml\n"
    "@@ -18,4 +18,4 @@\n"
    " watchdog_timeout_ms: 5\n"
    "-max_path_deviation_m: 0.020\n"
    "+max_path_deviation_m: 0.016\n"
)


@dataclass(frozen=True)
class SeedFixture:
    db_path: Path
    champion_hash: str
    genesis_hash: str
    label_counts: Dict[str, int]


def _passed_receipt() -> SafetyValidationReceipt:
    return SafetyValidationReceipt(
        passed=True,
        violations=[],
        static_analysis_summary="fixture: clean",
        compiler_version=COMPILER_VERSION,
        kinematic_precheck=True,
    )


def _harness(
    parent: str, diff: str, ast_fp: str, config: Dict[str, str]
) -> HarnessManifest:
    return HarnessManifest(
        harness_hash=compute_harness_hash(
            parent_harness_hash=parent,
            harness_diff=diff,
            ast_fingerprint=ast_fp,
            config=config,
        ),
        task_name=TASK,
        parent_harness_hash=parent,
        harness_diff=diff,
        ast_fingerprint=ast_fp,
        compiler_version=COMPILER_VERSION,
        safety_receipt=_passed_receipt(),
        difficulty_strata=list(DifficultyStrataBin),
        config=config,
    )


def _write_dump_artifacts(root: Path, snapshot_hash: str) -> Dict[str, str]:
    dump_dir = root / "dumps"
    dump_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = dump_dir / f"{snapshot_hash}.parquet"
    state_path = dump_dir / f"{snapshot_hash}.npz"
    n = 64
    t = np.linspace(0.0, 5.0, n)
    table = pa.table(
        {
            "t_s": t,
            **{f"q_joint_{j}": np.sin(t * (1.0 + j)) for j in range(7)},
            **{f"tau_joint_{j}": np.clip(t * j, 0.0, 4.0) for j in range(7)},
        }
    )
    pq.write_table(table, parquet_path)
    np.savez(state_path, qpos=np.zeros(7), qvel=np.zeros(7))
    return {"parquet": str(parquet_path), "state": str(state_path)}


_SUMMARY_BY_LABEL: Dict[str, str] = {
    LABEL_A: _SUMMARY_A,
    LABEL_B: _SUMMARY_B,
    LABEL_C: _SUMMARY_C,
}


def summary_by_label_text(label: str) -> str:
    """The one-paragraph VLM summary attached to every rollout of a label
    (label-led diagnoser convention: ``LABEL: paragraph``)."""
    return _SUMMARY_BY_LABEL[label]


def _rollout(
    rollout_id: str,
    harness_hash: str,
    *,
    stratum: DifficultyStrataBin,
    success: bool,
    t: float,
    label: str | None,
    snapshot_hash: str | None,
) -> RolloutTelemetryBundle:
    summary = (
        f"{label}: {summary_by_label_text(label)}"
        if label is not None
        else None
    )
    return RolloutTelemetryBundle(
        rollout_id=rollout_id,
        batch_id="",
        harness_hash=harness_hash,
        task_name=TASK,
        domain=RolloutDomain.PHYSICAL,
        stratum=stratum,
        success=success,
        time_to_completion=6.1 if success else None,
        cumulative_jerk=312.5 if success else 401.2,
        torque_penalty=19.0 if success else 27.5,
        force_spike_max=0.0 if success else 41.5,
        collision_count=0 if success else 1,
        flight_snapshot_hash=snapshot_hash,
        t_start_unix=t,
        t_end_unix=t + 6.2,
        vlm_failure_summary=summary,
    )


def _stats(
    rollouts: List[RolloutTelemetryBundle],
) -> BatchStatistics:
    n = len(rollouts)
    s = sum(1 for r in rollouts if r.success)
    low, high = wilson_score_interval(s, n)
    per: Dict[DifficultyStrataBin, StratumStats] = {}
    for stratum in sorted({r.stratum for r in rollouts}, key=lambda s: s.value):
        sub = [r for r in rollouts if r.stratum is stratum]
        ss = sum(1 for r in sub if r.success)
        slo, shi = wilson_score_interval(ss, len(sub))
        per[stratum] = StratumStats(
            stratum=stratum,
            n=len(sub),
            successes=ss,
            success_rate=ss / len(sub),
            wilson_low=slo,
            wilson_high=shi,
            mean_ttc=6.1,
            mean_cumulative_jerk=350.0,
            mean_torque_penalty=21.0,
        )
    return BatchStatistics(
        overall=OverallStats(
            n=n,
            successes=s,
            success_rate=s / n,
            wilson_low=low,
            wilson_high=high,
            mean_cumulative_jerk=350.0,
            mean_torque_penalty=21.0,
        ),
        per_stratum=per,
        sprt=SPRTTrace(
            p0=0.55,
            p1=0.65,
            lower_boundary=-2.94,
            upper_boundary=2.94,
            final_llr=1.1,
            stopped_early=False,
            n_samples_consumed=n,
        ),
    )


def seed_fixture(db_path: Path, artifact_root: Path) -> SeedFixture:
    """Seed exactly 20 hash-chained ledger events + supporting records.

    Composition (event order = seeding order):
      ev1  HYPOTHESIS_RECORDED  hyp_champ (champion lineage, PROMOTED)
      ev2  BATCH_COMMITTED      champion sim batch (4 SIM rollouts)
      ev3  BATCH_COMMITTED      champion physical batch (5 rollouts, 1
                                 sealed dump labeled LABEL_C)
      ev4  CHAMPION_PROMOTED    champion enthroned (stratum stats are the
                                 promoted-era reference: EASY 0.85, MED 0.70)
      ev5  HYPOTHESIS_RECORDED  hyp_x1 (curriculum family, REJECTED_COMPILER)
      ev6  HYPOTHESIS_RECORDED  hyp_x2 (domain-rand family, REJECTED_SIM)
      ev7  HYPOTHESIS_RECORDED  hA (reward family)
      ev8  HYPOTHESIS_RECORDED  hB (reward family)
      ev9  HYPOTHESIS_RECORDED  hC (reward family)
      ev10-14 BATCH_COMMITTED   5 physical batches, one LABEL_A dump each,
                                 on 5 distinct harnesses (dedup sees 5)
      ev15-17 BATCH_COMMITTED   3 physical batches, one LABEL_B dump each
      ev18-19 BATCH_COMMITTED   2 physical batches, one LABEL_C dump each

    Last-3-rejected by ledger seq are hC, hB, hA -- all REWARD_SCALING, so
    the anti-loop signal must fire and direct to CURRICULUM_SHIFT.
    Challenger variants share their hypothesis across batches: this is the
    orchestrator's own inconclusive-loop pattern (EVALUATING_PROMOTION ->
    PHYSICAL_EXECUTING adds more batches to one hypothesis), with the
    extra harnesses playing SysID-recalibrated recompiles.
    """
    artifact_root.mkdir(parents=True, exist_ok=True)
    ledger = ExperimentLedgerRepository(db_path)
    cfg = {"env": "mujoco", "cell": "bench-07"}

    genesis = HarnessManifest.genesis(
        task_name=TASK,
        harness_diff=GENESIS_DIFF,
        ast_fingerprint="afp_genesis_00",
        compiler_version=COMPILER_VERSION,
        config=cfg,
    )
    ledger.upsert_harness(genesis)
    champion = _harness(genesis.harness_hash, CHAMPION_DIFF, "afp_champ_00", cfg)
    ledger.upsert_harness(champion)

    checkpoint = PolicyCheckpointRecord(
        checkpoint_hash="ck_" + "c1" * 20,
        harness_hash=champion.harness_hash,
        format="safetensors",
        storage_path=str(artifact_root / "checkpoints" / "ck_champ.safetensors"),
        num_params=8_141_312,
        created_at_unix=T0 + 5.0,
    )
    ledger.record_policy_checkpoint(checkpoint)

    def _proposal(
        hyp_id: str, parent: str, diff: str, fp: str, rationale: str, t: float
    ) -> HypothesisProposal:
        return HypothesisProposal(
            hypothesis_id=hyp_id,
            task_name=TASK,
            parent_harness_hash=parent,
            harness_diff=diff,
            ast_fingerprint=fp,
            rationale=rationale,
            falsifiable_prediction=FalsifiablePrediction(
                target_metric="physical_success_rate",
                stratum=DifficultyStrataBin.EASY_CENTER,
                predicted_delta=0.10,
                ci_low=0.03,
                ci_high=0.17,
            ),
            param_summary={"delta": 0.10},
            proposed_at_unix=t,
        )

    # -- champion lineage (ev1..ev4) --------------------------------------
    ledger.record_hypothesis(
        _proposal(
            "hyp_champ",
            genesis.harness_hash,
            CHAMPION_DIFF,
            "afp_champ_00",
            "tighten path deviation limit; champion-era reference stats",
            T0 + 10.0,
        ),
        0.81,
    )  # ev1
    ledger.update_hypothesis_status(
        "hyp_champ", HypothesisStatus.COMPILED, harness_hash=champion.harness_hash
    )
    sim_rollouts = [
        RolloutTelemetryBundle(
            rollout_id=f"rol_cs{i}",
            batch_id="",
            harness_hash=champion.harness_hash,
            task_name=TASK,
            domain=RolloutDomain.SIM,
            stratum=DifficultyStrataBin.EASY_CENTER,
            success=True,
            time_to_completion=5.4,
            cumulative_jerk=290.0,
            torque_penalty=17.0,
            t_start_unix=T0 + 20.0 + i,
            t_end_unix=T0 + 25.0 + i,
        )
        for i in range(4)
    ]
    ledger.record_evaluation_batch(
        EvaluationBatchRecord(
            batch_id="bat_champ_sim",
            cycle_id="cyc_champ",
            hypothesis_id="hyp_champ",
            harness_hash=champion.harness_hash,
            champion_hash=None,
            domain=RolloutDomain.SIM,
            task_name=TASK,
            n_rollouts=len(sim_rollouts),
            decision=BatchDecision.CONTINUE,
            statistics=_stats(sim_rollouts),
            trigger="SIM_GATE",
            started_at_unix=T0 + 20.0,
            committed_at_unix=T0 + 25.0,
        ),
        sim_rollouts,
    )  # ev2

    # champion physical batch: EASY 2/3 (one LABEL_C dump), MEDIUM 1/2
    # The dump is sealed BEFORE the batch commits (mid-rollout forward
    # reference -- the repo validates dump.rollout_id == claiming rollout).
    champion_paths = _write_dump_artifacts(artifact_root, "snap_dc1")
    ledger.record_blackbox_dump(
        BlackBoxSnapshotRecord(
            snapshot_hash="snap_dc1",
            rollout_id="rol_cp3",
            trigger=FlightTrigger.FORCE_SPIKE,
            parquet_path=str(artifact_root / "dumps" / "snap_dc1.parquet"),
            sim_state_path=str(artifact_root / "dumps" / "snap_dc1.npz"),
            format="parquet",
            t_window_seconds=5.0,
            sample_rate_hz=1000.0,
            num_samples=4500,
            mujoco_model_hash="mj_fr3_bench07",
            created_at_unix=T0 + 31.0,
        )
    )
    champion_rollouts = [
        _rollout(
            "rol_cp1", champion.harness_hash,
            stratum=DifficultyStrataBin.EASY_CENTER, success=True,
            t=T0 + 30.0, label=None, snapshot_hash=None,
        ),
        _rollout(
            "rol_cp2", champion.harness_hash,
            stratum=DifficultyStrataBin.EASY_CENTER, success=True,
            t=T0 + 30.5, label=None, snapshot_hash=None,
        ),
        _rollout(
            "rol_cp3", champion.harness_hash,
            stratum=DifficultyStrataBin.EASY_CENTER, success=False,
            t=T0 + 31.0, label=LABEL_C, snapshot_hash="snap_dc1",
        ),
        _rollout(
            "rol_cp4", champion.harness_hash,
            stratum=DifficultyStrataBin.MEDIUM_OBLIQUE, success=True,
            t=T0 + 31.5, label=None, snapshot_hash=None,
        ),
        _rollout(
            "rol_cp5", champion.harness_hash,
            stratum=DifficultyStrataBin.MEDIUM_OBLIQUE, success=False,
            t=T0 + 32.0, label=LABEL_B, snapshot_hash=None,
        ),
    ]
    ledger.record_evaluation_batch(
        EvaluationBatchRecord(
            batch_id="bat_champ_phys",
            cycle_id="cyc_champ",
            hypothesis_id="hyp_champ",
            harness_hash=champion.harness_hash,
            champion_hash=None,
            domain=RolloutDomain.PHYSICAL,
            task_name=TASK,
            n_rollouts=len(champion_rollouts),
            decision=BatchDecision.PROMOTE,
            statistics=_stats(champion_rollouts),
            trigger="PHYSICAL_EXECUTION",
            started_at_unix=T0 + 30.0,
            committed_at_unix=T0 + 33.0,
        ),
        champion_rollouts,
    )  # ev3
    ledger.record_champion(
        ChampionManifest(
            task_name=TASK,
            harness_hash=champion.harness_hash,
            policy_checkpoint_hash=checkpoint.checkpoint_hash,
            batch_id="bat_champ_phys",
            wilson_lower_bound=0.62,
            stratum_stats=[
                StratumStats(
                    stratum=DifficultyStrataBin.EASY_CENTER,
                    n=40, successes=34, success_rate=0.85,
                    wilson_low=0.74, wilson_high=0.92,
                ),
                StratumStats(
                    stratum=DifficultyStrataBin.MEDIUM_OBLIQUE,
                    n=30, successes=21, success_rate=0.70,
                    wilson_low=0.52, wilson_high=0.84,
                ),
            ],
            promoted_at_unix=T0 + 33.5,
        )
    )  # ev4
    ledger.update_hypothesis_status(
        "hyp_champ", HypothesisStatus.PROMOTED
    )

    # -- two non-reward-family proposals for context variety (ev5, ev6) ----
    ledger.record_hypothesis(
        _proposal(
            "hyp_x1", champion.harness_hash,
            "--- a/curriculum.yaml\n+++ b/curriculum.yaml\n"
            "@@ -3,3 +3,3 @@\n- oblique_start_deg: 12.0\n+ oblique_start_deg: 14.0\n",
            "afp_x1", "advance obliqueness curriculum one notch", T0 + 40.0,
        ),
        0.44,
    )  # ev5
    ledger.update_hypothesis_status(
        "hyp_x1", HypothesisStatus.REJECTED_COMPILER,
        rejection_reason="kinematic precheck: oblique envelope collides with"
        " cell wall at 14 degrees",
    )
    ledger.record_hypothesis(
        _proposal(
            "hyp_x2", champion.harness_hash,
            "--- a/domain_rand.yaml\n+++ b/domain_rand.yaml\n"
            "@@ -5,3 +5,3 @@\n- friction_range: [0.35, 0.55]\n"
            "+ friction_range: [0.30, 0.60]\n",
            "afp_x2", "widen friction randomization band", T0 + 45.0,
        ),
        0.51,
    )  # ev6
    hx2 = _harness(
        champion.harness_hash,
        "--- a/domain_rand.yaml\n+++ b/domain_rand.yaml\n"
        "@@ -5,3 +5,3 @@\n- friction_range: [0.35, 0.55]\n"
        "+ friction_range: [0.30, 0.60]\n",
        "afp_x2", cfg,
    )
    ledger.upsert_harness(hx2)
    ledger.update_hypothesis_status(
        "hyp_x2", HypothesisStatus.COMPILED, harness_hash=hx2.harness_hash
    )
    ledger.update_hypothesis_status(
        "hyp_x2", HypothesisStatus.REJECTED_SIM,
        rejection_reason="sim gate: wilson low 0.31 below champion bound",
    )
    ledger.record_hypothesis(
        _proposal(
            "hyp_x0", champion.harness_hash,
            "--- a/obs_encoder.py\n+++ b/obs_encoder.py\n"
            "@@ -9,3 +9,3 @@\n-OBS_DROPOUT_P = 0.02\n+OBS_DROPOUT_P = 0.03\n",
            "afp_x0", "raise observation dropout to harden the encoder", T0 + 47.0,
        ),
        0.47,
    )  # ev7
    ledger.update_hypothesis_status(
        "hyp_x0", HypothesisStatus.REJECTED_COMPILER,
        rejection_reason="AST guard: dropout constant outside sanctioned"
        " range [0.00, 0.02]",
    )

    # labels: A gets 5 dumps/harnesses, B gets 3, C gets 2 (+1 champion = 3)
    plan = [
        ("hA", LABEL_A, 5, 10, "GRASP"),
        ("hB", LABEL_B, 3, 15, "GRIP"),
        ("hC", LABEL_C, 2, 18, "WRIST"),
    ]
    reward_variants = [
        ("hA", "afp_a", 0.62, "raise grasp tolerance reward slope slightly"),
        ("hB", "afp_b", 0.58, "steepen grip force ramp reward"),
        ("hC", "afp_c", 0.55, "penalize late release timing more strongly"),
    ]
    harness_by_hyp: Dict[str, List[HarnessManifest]] = {}
    for hyp_id, ast_fp, _score, _rationale in reward_variants:
        hs: List[HarnessManifest] = []
        for k in range(5):
            # the hyp_id is baked into the constant name so every
            # hypothesis proposes a byte-distinct diff (storage-level exact
            # dedup is on (parent, diff))
            diff = (
                "--- a/reward.py\n+++ b/reward.py\n"
                f"@@ -7,3 +7,3 @@\n-REWARD_{hyp_id}_K = {0.30 + 0.01 * k:.2f}\n"
                f"+REWARD_{hyp_id}_K = {0.32 + 0.01 * k:.2f}\n"
            )
            hs.append(_harness(champion.harness_hash, diff, f"{ast_fp}_{k}", cfg))
        harness_by_hyp[hyp_id] = hs
        for h in hs:
            ledger.upsert_harness(h)
    ledger.record_hypothesis(
        _proposal(
            "hA", champion.harness_hash,
            harness_by_hyp["hA"][0].harness_diff, "afp_a_0",
            "raise grasp tolerance reward slope slightly", T0 + 50.0,
        ),
        0.62,
    )  # ev7
    ledger.update_hypothesis_status(
        "hA", HypothesisStatus.COMPILED, harness_hash=harness_by_hyp["hA"][0].harness_hash
    )
    ledger.record_hypothesis(
        _proposal(
            "hB", champion.harness_hash,
            harness_by_hyp["hB"][0].harness_diff, "afp_b_0",
            "steepen grip force ramp reward", T0 + 55.0,
        ),
        0.58,
    )  # ev8
    ledger.update_hypothesis_status(
        "hB", HypothesisStatus.COMPILED, harness_hash=harness_by_hyp["hB"][0].harness_hash
    )
    ledger.record_hypothesis(
        _proposal(
            "hC", champion.harness_hash,
            harness_by_hyp["hC"][0].harness_diff, "afp_c_0",
            "penalize late release timing more strongly", T0 + 60.0,
        ),
        0.55,
    )  # ev9
    ledger.update_hypothesis_status(
        "hC", HypothesisStatus.COMPILED, harness_hash=harness_by_hyp["hC"][0].harness_hash
    )

    # -- dumps first (forward references), then challenger batches ---------
    summary_by_label = {
        LABEL_A: _SUMMARY_A,
        LABEL_B: _SUMMARY_B,
        LABEL_C: _SUMMARY_C,
    }
    trigger_cycle = [
        FlightTrigger.FORCE_SPIKE,
        FlightTrigger.COLLISION,
        FlightTrigger.TASK_FAILURE,
    ]
    dump_records: List[BlackBoxSnapshotRecord] = []

    stratum_by_label = {
        LABEL_A: DifficultyStrataBin.EASY_CENTER,
        LABEL_B: DifficultyStrataBin.MEDIUM_OBLIQUE,
        LABEL_C: DifficultyStrataBin.HARD_BOUNDARY,
    }
    for hyp_id, label, n_dumps, seq_offset, _tag in plan:
        harnesses = harness_by_hyp[hyp_id]
        for k in range(n_dumps):
            snapshot_hash = f"snap_d{hyp_id.lower()}_{k + 1}"
            rollout_id = f"rol_{hyp_id.lower()}_{k + 1}"
            paths = _write_dump_artifacts(artifact_root, snapshot_hash)
            dump_records.append(
                BlackBoxSnapshotRecord(
                    snapshot_hash=snapshot_hash,
                    rollout_id=rollout_id,
                    trigger=trigger_cycle[k % len(trigger_cycle)],
                    parquet_path=paths["parquet"],
                    sim_state_path=paths["state"],
                    format="parquet",
                    t_window_seconds=5.0,
                    sample_rate_hz=1000.0,
                    num_samples=4000 + 100 * k,
                    mujoco_model_hash="mj_fr3_bench07",
                    created_at_unix=T0 + 100.0 + seq_offset + k,
                )
            )
    for record in dump_records:
        ledger.record_blackbox_dump(record)

    t = T0 + 100.0
    for hyp_id, label, n_dumps, seq_offset, _tag in plan:
        harnesses = harness_by_hyp[hyp_id]
        for k in range(n_dumps):
            snapshot_hash = f"snap_d{hyp_id.lower()}_{k + 1}"
            rollout_id = f"rol_{hyp_id.lower()}_{k + 1}"
            harness = harnesses[k]
            rollouts = [
                _rollout(
                    rollout_id, harness.harness_hash,
                    stratum=stratum_by_label[label], success=False,
                    t=t, label=label, snapshot_hash=snapshot_hash,
                )
            ]
            ledger.record_evaluation_batch(
                EvaluationBatchRecord(
                    batch_id=f"bat_{hyp_id.lower()}_{k + 1}",
                    cycle_id=f"cyc_{hyp_id.lower()}",
                    hypothesis_id=hyp_id,
                    harness_hash=harness.harness_hash,
                    champion_hash=champion.harness_hash,
                    domain=RolloutDomain.PHYSICAL,
                    task_name=TASK,
                    n_rollouts=1,
                    decision=BatchDecision.REJECT,
                    statistics=_stats(rollouts),
                    trigger="PHYSICAL_EXECUTION",
                    started_at_unix=t,
                    committed_at_unix=t + 6.0,
                ),
                rollouts,
            )
            t += 7.0
    for hyp_id in ("hA", "hB", "hC"):
        ledger.update_hypothesis_status(
            hyp_id,
            HypothesisStatus.REJECTED_PHYSICAL,
            rejection_reason=f"physical gate: {hyp_id} below champion bound",
        )

    ledger.close()
    return SeedFixture(
        db_path=db_path,
        champion_hash=champion.harness_hash,
        genesis_hash=genesis.harness_hash,
        label_counts={LABEL_A: 5, LABEL_B: 3, LABEL_C: 3},
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


class _Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if not condition:
            raise AssertionError(f"CHECK FAILED: {name} {detail}")
        self.passed += 1
        print(f"  ok  {name}")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="robocore_ctx_smoke_"))
    checks = _Checks()

    try:
        # ---- check 1: deterministic 20-entry seed, budget, overflow -------
        fixture = seed_fixture(tmp / "ledger_a.db", tmp / "artifacts_a")
        ledger = ExperimentLedgerRepository(fixture.db_path)
        n_events = int(
            ledger._conn.execute(  # noqa: SLF001 - read-only count in smoke
                "SELECT COUNT(*) AS n FROM ledger_events"
            ).fetchone()["n"]
        )
        assert n_events == 20, f"expected 20 ledger events, got {n_events}"
        builder = ContextBuilder()
        ctx = builder.build(ledger, fixture.champion_hash)
        assert ctx.token_estimate <= 8000, (
            f"token_estimate {ctx.token_estimate} exceeds planner default"
            " budget 8000"
        )
        overflow_raised = False
        try:
            builder.build(ledger, fixture.champion_hash, token_budget=30)
        except ContextOverflow:
            overflow_raised = True
        assert overflow_raised, (
            "budget=30 must raise ContextOverflow (raise, not warn)"
        )
        fixture_b = seed_fixture(tmp / "ledger_b.db", tmp / "artifacts_b")
        ledger_b = ExperimentLedgerRepository(fixture_b.db_path)
        ctx_b = builder.build(ledger_b, fixture_b.champion_hash)
        assert ctx == ctx_b, "identically seeded ledgers must compile"
        " to identical contexts"
        ledger.close()
        ledger_b.close()
        checks.check(
            "seeded exactly 20 deterministic ledger entries; build fits"
            f" budget ({ctx.token_estimate}/8000 tokens); ContextOverflow"
            " raised on impossible budget; two seeds compile identically",
            True,
        )

        # ---- check 2: champion hash appears exactly once ------------------
        ledger = ExperimentLedgerRepository(fixture.db_path)
        ctx = builder.build(ledger, fixture.champion_hash)
        registry = PromptRegistry.load_default()
        slots = build_planner_slots(
            ctx, champion_hash=fixture.champion_hash, token_budget=8000
        )
        rendered = registry.render("planner_v1", **slots)
        count = rendered.count(fixture.champion_hash)
        assert count == 1, f"champion hash appears {count} times, expected 1"
        checks.check(
            "champion hash appears exactly once in the rendered planner"
            " prompt",
            True,
        )

        # ---- check 3: failure clusters min(5, N) across 3 labels ----------
        clusters = {c.failure_label: c for c in ctx.failure_clusters}
        assert len(clusters) == 3, f"expected 3 labels, got {sorted(clusters)}"
        for label, expected_n in fixture.label_counts.items():
            got = clusters[label]
            assert len(got.entries) == min(5, expected_n), (
                f"{label}: {len(got.entries)} entries, expected"
                f" {min(5, expected_n)}"
            )
            for entry in got.entries:
                assert entry.keyframe_hash.startswith("kf_")
                assert entry.snapshot_hash.startswith("snap_")
                assert len(entry.vlm_summary) > 40
        reg_present = (
            ctx.champion_summary is not None
            and len(ctx.champion_summary.regressions) >= 1
        )
        assert reg_present, "EASY_CENTER regression (0.667 vs 0.85) expected"
        checks.check(
            "failure_clusters = min(5, N) per label across 3 labels"
            f" ({ {k: len(v.entries) for k, v in clusters.items()} });"
            " hashes only; EASY stratum regression detected",
            True,
        )

        # ---- check 4: registry refuses unregistered prompts ---------------
        rogue_dir = tmp / "rogue_prompts"
        rogue_dir.mkdir()
        for src in PROMPTS_DIR.glob("*.md"):
            shutil.copy(src, rogue_dir / src.name)
        (rogue_dir / "rogue_v1.md").write_text(
            "# ROGUE prompt\n\n{{task_name}} unregistered\n", encoding="utf-8"
        )
        refused = 0
        try:
            PromptRegistry.load_dir(rogue_dir)
        except Exception as exc:  # noqa: BLE001 - counted, not propagated
            assert type(exc).__name__ == "UnregisteredPromptError", str(exc)
            refused += 1
        try:
            registry.lookup_by_hash("dead" * 16)
        except Exception as exc:  # noqa: BLE001
            assert type(exc).__name__ == "UnregisteredPromptError", str(exc)
            refused += 1
        assert refused == 2, f"expected 2 refusals, counted {refused}"
        checks.check(
            "registry refuses unregistered prompts (2 refusals raised,"
            " caught, counted)",
            True,
        )

        # ---- check 5: render -> sha256 -> lookup round-trip ---------------
        sample_critic = build_critic_slots(
            task_name=TASK,
            rationale="tighten grasp tolerance reward slope",
            predicted_delta=0.10,
            proposal_diff="--- a/reward.py\n+++ b/reward.py\n@@ -7 +7 @@\n-x\n+y\n",
            champion_wilson_low=0.62,
            rejected_similar_digest="none in window",
        )
        cluster_entry = ctx.failure_clusters[0].entries[0]
        sample_diagnoser = build_diagnoser_slots(
            task_name=TASK,
            failure_label=cluster_entry.failure_label,
            telemetry_digest="5 s @ 1 kHz; force spike 41.5 N at t-0.3 s",
            keyframe_hashes=[cluster_entry.keyframe_hash],
        )
        slot_sets = {
            "planner_v1": slots,
            "critic_v1": sample_critic,
            "diagnoser_v1": sample_diagnoser,
        }
        for version, version_slots in slot_sets.items():
            rendered_v = registry.render(version, **version_slots)
            assert "{{" not in rendered_v, f"{version}: unresolved markers"
            file_sha = sha256_file(PROMPTS_DIR / f"{version}.md")
            meta = registry.lookup_by_hash(file_sha)
            assert meta.version == version, (
                f"{version}: lookup_by_hash resolved {meta.version}"
            )
            assert registry.registered_hash(version) == file_sha
        from robocore.contracts import HypothesisProposal

        proposal_out = registry.output_model("planner_v1").model_validate(
            {
                "task_name": TASK,
                "parent_harness_hash": fixture.champion_hash,
                "harness_diff": "--- a/harness.yaml\n+++ b/harness.yaml\n",
                "ast_fingerprint": "afp_rt",
                "rationale": "sample round-trip",
                "falsifiable_prediction": {
                    "target_metric": "physical_success_rate",
                    "stratum": "EASY_CENTER",
                    "predicted_delta": 0.08,
                    "ci_low": 0.02,
                    "ci_high": 0.14,
                },
                "param_summary": {"termination_jerk_limit": 38.0},
            }
        )
        critic_out = CriticVerdict.model_validate(
            {
                "decision": "reject",
                "reason": "constant-only re-tune of a rejected reward diff",
                "anti_loop_flagged": True,
                "expected_info_gain": 0.05,
                "mutation_family": "REWARD_SCALING",
            }
        )
        diagnoser_out = FailureDiagnosis.model_validate(
            {
                "failure_label": cluster_entry.failure_label,
                "summary": "Force spike on joint 5 preceded release; grasp"
                " axis misaligned by 12 degrees at contact.",
                "keyframe_refs": [cluster_entry.keyframe_hash],
                "confidence": 0.8,
                "snapshot_hash": cluster_entry.snapshot_hash,
                "trigger": cluster_entry.trigger.value,
            }
        )
        ensure_keyframe_refs_subset(
            diagnoser_out, [cluster_entry.keyframe_hash]
        )
        hallucination_blocked = False
        try:
            ensure_keyframe_refs_subset(
                diagnoser_out, ["kf_not_in_prompt_000000000000"]
            )
        except ValueError:
            hallucination_blocked = True
        assert hallucination_blocked
        assert proposal_out.parent_harness_hash == fixture.champion_hash
        assert critic_out.anti_loop_flagged is True
        # role default budgets + typed refusal round-trip (the planner's
        # documented refusal channel)
        from robocore.cognitive import (
            DEFAULT_TOKEN_BUDGETS,
            PromptRole,
            Refusal,
            RefusalReason,
        )

        assert registry.default_budget("planner_v1") == 8000
        assert registry.default_budget("critic_v1") == 2000
        assert registry.default_budget("diagnoser_v1") == 4000
        assert DEFAULT_TOKEN_BUDGETS[PromptRole.PLANNER] == 8000
        assert DEFAULT_TOKEN_BUDGETS[PromptRole.CRITIC] == 2000
        assert DEFAULT_TOKEN_BUDGETS[PromptRole.DIAGNOSER] == 4000
        refusal = Refusal.model_validate(
            {
                "refusal_reason": "ANTI_LOOP_WITHOUT_OVERRIDE",
                "detail": "all catalog families exhausted recently",
            }
        )
        assert refusal.refusal_reason is RefusalReason.ANTI_LOOP_WITHOUT_OVERRIDE
        checks.check(
            "render -> sha256 -> lookup_by_hash matches registered hash for"
            " all 3 prompts; sample outputs validate against"
            " HypothesisProposal / CriticVerdict / FailureDiagnosis"
            " (keyframe subset enforced, hallucinated refs blocked)",
            True,
        )

        # ---- check 6: anti-loop injection ----------------------------------
        signal = ctx.anti_loop_signal
        assert signal is not None, "anti-loop signal must fire (hA,hB,hC all"
        " REWARD_SCALING, all rejected)"
        assert signal.catalog_family is MutationType.CURRICULUM_SHIFT, (
            f"expected CURRICULUM_SHIFT next, got {signal.catalog_family}"
        )
        assert signal.rejected_streak == 3
        assert all(
            f is MutationType.REWARD_SCALING for f in signal.rejected_families
        )
        directive = render_context_markdown(ctx)
        assert "ANTI-LOOP SIGNAL" in directive
        assert "CURRICULUM_SHIFT" in directive
        checks.check(
            "anti-loop: last 3 rejected proposals share REWARD_SCALING ->"
            " ForceExploration(catalog_family=CURRICULUM_SHIFT)",
            True,
        )

        ledger.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"PASS {checks.passed}/{checks.total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
