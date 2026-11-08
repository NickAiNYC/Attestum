"""attestum.templates.evidence_passport -- the product.

The Evidence Passport is a verifiable record that a robotic work cell
changed and by how much, with its uncertainty attached.  It carries
exactly eleven sections (per the North Star doc):

  1. Cell and robot identification
  2. Software, firmware, and configuration versions
  3. Data provenance and timestamps
  4. Baseline measurement window
  5. Frozen evaluation conditions
  6. Failure taxonomy
  7. Evaluation methodology
  8. Statistical uncertainty
  9. Safety approvals and human oversight
 10. Reproducible artifacts and hashes
 11. Improvement and regression results

Section 11 is a TERNARY, not a binary: exactly one of
{verified_improvement, clear_negative, no_go} must be set, because
"the assessment found the problem is mechanical" is a legitimate and
publishable outcome.

``passport_hash`` is the sha256 of the canonical JSON of the model
minus the hash field itself; it is always derived (recomputed by a
model validator), never free-floating.  This is the artifact a
customer can verify.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, model_validator


# ---------------------------------------------------------------------------
# Sections 1-10
# ---------------------------------------------------------------------------


class CellIdentification(BaseModel):
    """Section 1: cell and robot identification (coded, never named)."""

    model_config = ConfigDict(extra="forbid")

    customer_code: str
    site_code: str
    cell_id: str
    task_name: str
    robot_make: str
    robot_model: str
    robot_serial_code: str
    dof_count: int


class SoftwareVersions(BaseModel):
    """Section 2: software, firmware, and configuration versions."""

    model_config = ConfigDict(extra="forbid")

    runtime_package: str
    runtime_version: str
    harness_hash: str
    controller_firmware: str
    vision_firmware: str
    configuration_sha256: str
    prompt_registry_sha256: str


class DataSource(BaseModel):
    """One input file the customer provided, with its content hash."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    format: str
    sha256: str
    exported_utc: datetime


class DataProvenance(BaseModel):
    """Section 3: data provenance and timestamps."""

    model_config = ConfigDict(extra="forbid")

    sources: List[DataSource]
    site_timezone: str
    clock_sync_method: str
    export_operator_code: str


class BaselineWindow(BaseModel):
    """Section 4: baseline measurement window."""

    model_config = ConfigDict(extra="forbid")

    start_utc: datetime
    end_utc: datetime
    operating_hours: float
    intervention_count: int
    intervention_rate_per_hour: float


class FrozenEvaluationConditions(BaseModel):
    """Section 5: frozen evaluation conditions."""

    model_config = ConfigDict(extra="forbid")

    task_definition_sha256: str
    comparison_design: str
    exclusion_rule: str
    measurement_pipeline: str
    recorder_rate_hz: int
    evaluation_freeze_tag: str


class FailureLabel(BaseModel):
    """One entry of the failure taxonomy with its baseline count."""

    model_config = ConfigDict(extra="forbid")

    label: str
    definition: str
    baseline_count: int


class FailureTaxonomy(BaseModel):
    """Section 6: failure taxonomy."""

    model_config = ConfigDict(extra="forbid")

    labels: List[FailureLabel]
    taxonomy_source: str


class EvaluationMethodology(BaseModel):
    """Section 7: evaluation methodology."""

    model_config = ConfigDict(extra="forbid")

    design: str
    primary_metric: str
    sample_description: str
    interval_reference: str
    gate_reference: str
    acceptance_rule: str


class StatisticalUncertainty(BaseModel):
    """Section 8: statistical uncertainty."""

    model_config = ConfigDict(extra="forbid")

    observed_successes: int
    observed_trials: int
    point_estimate_pct: float
    ci_low_pct: float
    ci_high_pct: float
    ci_method: str
    z_value: float
    notes: str


class SafetyOversight(BaseModel):
    """Section 9: safety approvals and human oversight."""

    model_config = ConfigDict(extra="forbid")

    safety_approval_ref: str
    approver_role: str
    human_oversight_description: str
    stop_conditions: List[str]
    toolchain_guard_version: str
    analysis_only_attestation: str


class ArtifactRef(BaseModel):
    """One reproducible artifact with its content hash."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: str
    content_hash: str
    hash_algorithm: str = "sha256"
    size_bytes: int


class ReproducibleArtifacts(BaseModel):
    """Section 10: reproducible artifacts and hashes."""

    model_config = ConfigDict(extra="forbid")

    artifacts: List[ArtifactRef]


# ---------------------------------------------------------------------------
# Section 11: the ternary outcome
# ---------------------------------------------------------------------------


class VerifiedImprovement(BaseModel):
    """Outcome A: the improvement claim is supported by the evidence."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline_value: float
    post_value: float
    relative_improvement_pct: float
    ci_low_pct: float
    ci_high_pct: float
    gate_decision: str
    statement: str


class ClearNegative(BaseModel):
    """Outcome B: the assessment found the problem is mechanical (etc.).

    A clear negative is a legitimate and publishable outcome: it tells
    the customer exactly what NOT to spend software money on.
    """

    model_config = ConfigDict(extra="forbid")

    finding: str
    evidence_refs: List[str]
    statement: str


class NoGo(BaseModel):
    """Outcome C: the assessment could not produce a defensible claim."""

    model_config = ConfigDict(extra="forbid")

    reason_code: str
    statement: str


class ImprovementResult(BaseModel):
    """Section 11: improvement and regression results (exactly one outcome)."""

    model_config = ConfigDict(extra="forbid")

    verified_improvement: Optional[VerifiedImprovement] = None
    clear_negative: Optional[ClearNegative] = None
    no_go: Optional[NoGo] = None

    @model_validator(mode="after")
    def _exactly_one_outcome(self) -> "ImprovementResult":
        set_count = sum(
            outcome is not None
            for outcome in (self.verified_improvement, self.clear_negative, self.no_go)
        )
        if set_count != 1:
            raise ValueError(
                "Section 11 is a ternary: exactly one of "
                "{verified_improvement, clear_negative, no_go} must be set, "
                f"got {set_count}"
            )
        return self


# ---------------------------------------------------------------------------
# The passport
# ---------------------------------------------------------------------------

SECTION_FIELDS: Tuple[str, ...] = (
    "cell_identification",
    "software_versions",
    "data_provenance",
    "baseline_window",
    "frozen_conditions",
    "failure_taxonomy",
    "evaluation_methodology",
    "statistical_uncertainty",
    "safety_oversight",
    "reproducible_artifacts",
    "improvement_result",
)

SECTION_TITLES: Tuple[str, ...] = (
    "Cell and robot identification",
    "Software, firmware, and configuration versions",
    "Data provenance and timestamps",
    "Baseline measurement window",
    "Frozen evaluation conditions",
    "Failure taxonomy",
    "Evaluation methodology",
    "Statistical uncertainty",
    "Safety approvals and human oversight",
    "Reproducible artifacts and hashes",
    "Improvement and regression results",
)


class EvidencePassport(BaseModel):
    """The product: eleven sections plus a derived content hash."""

    model_config = ConfigDict(extra="forbid")

    passport_id: str
    issued_utc: datetime
    prepared_by: str
    schema_version: str = "1.0.0"
    passport_hash: str = ""

    cell_identification: CellIdentification
    software_versions: SoftwareVersions
    data_provenance: DataProvenance
    baseline_window: BaselineWindow
    frozen_conditions: FrozenEvaluationConditions
    failure_taxonomy: FailureTaxonomy
    evaluation_methodology: EvaluationMethodology
    statistical_uncertainty: StatisticalUncertainty
    safety_oversight: SafetyOversight
    reproducible_artifacts: ReproducibleArtifacts
    improvement_result: ImprovementResult

    @staticmethod
    def canonical_json(model: "EvidencePassport") -> str:
        """Canonical JSON: model minus passport_hash, sorted keys, tight.

        This is the byte string whose sha256 IS the passport hash, so it
        is deliberately boring and stable: ``json.dumps`` with
        ``sort_keys=True`` and compact separators, UTF-8 encoded.
        """
        payload = model.model_dump(mode="json", exclude={"passport_hash"})
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    @model_validator(mode="after")
    def _recompute_passport_hash(self) -> "EvidencePassport":
        self.passport_hash = hashlib.sha256(
            self.canonical_json(self).encode("utf-8")
        ).hexdigest()
        return self


__all__ = [
    "ArtifactRef",
    "BaselineWindow",
    "CellIdentification",
    "ClearNegative",
    "DataProvenance",
    "DataSource",
    "EvaluationMethodology",
    "EvidencePassport",
    "FailureLabel",
    "FailureTaxonomy",
    "FrozenEvaluationConditions",
    "ImprovementResult",
    "NoGo",
    "ReproducibleArtifacts",
    "SECTION_FIELDS",
    "SECTION_TITLES",
    "SafetyOversight",
    "SoftwareVersions",
    "StatisticalUncertainty",
    "VerifiedImprovement",
]
