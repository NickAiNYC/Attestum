"""attestum.templates.case_study_agreement -- the free-with-case-study contract.

Eight clauses, each carrying a ``recommended_text`` (the standard
position) and a ``fallback_text`` (the pre-authorized concession), so
negotiation happens inside a bounded space and never invents new
terms on a call.

The retroactive fee sentence in clause 2 is not a suggestion; it is
the mechanism that keeps the free path from being abused: the
assessment is free only when the case study is published, and
withdrawal converts the fee retroactively.  Clause 5 (withdrawal)
references clause 2 by number, and clause 2 references clause 5.

All texts are built by :func:`build_clauses` from the brand record
(attestum.brand.BRAND) and the fee, so the company name and the price
each exist in exactly one place.
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from attestum.brand import BRAND

AnonymizationMethod = Literal[
    "strip_identifying_fields",
    "blur_shop_floor_video",
    "replace_customer_name_with_code",
]

ClauseElection = Literal["recommended", "fallback"]

ANONYMIZATION_OPTIONS: Dict[str, str] = {
    "strip_identifying_fields": (
        "strip identifying fields from exported logs and records before "
        "any material leaves the customer's network"
    ),
    "blur_shop_floor_video": (
        "blur shop-floor video in all published material before "
        "publication"
    ),
    "replace_customer_name_with_code": (
        "replace the customer name with a code (for example CUSTOMER-A) "
        "in every published artifact"
    ),
}


class Clause(BaseModel):
    """One numbered clause with its recommended and fallback texts."""

    model_config = ConfigDict(extra="forbid")

    number: int
    title: str
    recommended_text: str
    fallback_text: str


def build_clauses(
    assessment_fee_usd: float, anonymization_election: AnonymizationMethod
) -> List[Clause]:
    """Build the eight standard clauses from the brand record and fee.

    Pure function of its arguments: same fee and election produce
    byte-identical clauses.
    """
    company = BRAND.name
    fee = f"{assessment_fee_usd:,.0f}"
    anonymization = ANONYMIZATION_OPTIONS[anonymization_election]
    return [
        Clause(
            number=1,
            title="Scope of work and deliverables",
            recommended_text=(
                f"{company} performs one retrospective assessment of the "
                "customer's robotic work cell using only the data the "
                "customer provides under the Retrospective Assessment Scope "
                "document. Deliverables: (a) an Evidence Passport for the "
                "assessed cell, (b) an Economic Case, and (c) a draft case "
                "study. Timeline: Day 0 (signed) through Day 21 (final "
                "deliverables) as defined in the scope document. No live "
                "access to the cell is requested or granted at any point."
            ),
            fallback_text=(
                "If the customer's data package arrives after Day 10, each "
                "week of delay extends every subsequent milestone by one "
                "week; the deliverables themselves are unchanged."
            ),
        ),
        Clause(
            number=2,
            title="Fee schedule",
            recommended_text=(
                "The assessment is free when the case study is published: "
                "the fee is $0 USD upon publication under clause 4. If "
                "Customer exercises withdrawal rights under clause 5, the "
                "fee for the assessment is retroactively invoiced at "
                f"${fee} USD. No subscription, per-seat, or success fees "
                "apply under this agreement."
            ),
            fallback_text=(
                f"The fee is payable 50% at signing (${fee} USD / 2) and "
                f"50% at delivery (${fee} USD / 2); the free-with-case-study "
                "conversion in the recommended text no longer applies."
            ),
        ),
        Clause(
            number=3,
            title="Data handling and anonymization",
            recommended_text=(
                "The customer elects the anonymization method for this "
                f"engagement: {anonymization}. Raw data is processed on "
                f"{company}-controlled infrastructure, is never resold or "
                "used to train products for the customer's competitors, and "
                "is deleted 90 days after final deliverables unless the "
                "customer requests earlier deletion in writing."
            ),
            fallback_text=(
                "The customer may instead deliver a pre-anonymized data "
                "package with a signed attestation that identifying fields "
                "were removed before transfer; only what is received is "
                "processed."
            ),
        ),
        Clause(
            number=4,
            title="Publication rights, timing, and approval",
            recommended_text=(
                f"{company} may publish the case study no earlier than "
                "delivery of the final deliverables and no later than 60 "
                "days after delivery. The customer reviews the draft "
                "through a single named reviewer with 5 business days to "
                "respond; if no response is received within 5 business "
                "days, the draft is deemed approved."
            ),
            fallback_text=(
                "The reviewer turnaround extends to 10 business days; "
                "silence after 10 business days is still deemed approval."
            ),
        ),
        Clause(
            number=5,
            title="Customer withdrawal",
            recommended_text=(
                "The customer may withdraw consent to publication at any "
                "time before the publication date by written notice from "
                "the signatory. Upon withdrawal, the fee conversion in "
                f"clause 2 applies: the assessment fee of ${fee} USD "
                "becomes due within 30 days. After publication, the case "
                "study remains published, but the customer may request "
                "de-identification corrections at any time."
            ),
            fallback_text=(
                "Withdrawal notices must come from the signatory of this "
                "agreement; the retroactive invoice under clause 2 is then "
                "payable within 45 days instead of 30."
            ),
        ),
        Clause(
            number=6,
            title="Intellectual property boundaries",
            recommended_text=(
                "The customer retains all rights to the customer's "
                "trajectory data, video, and cell configuration. "
                f"{company} retains all rights to calibration priors, "
                "models, and the assessment methodology. The Evidence "
                "Passport is delivered to the customer under a perpetual, "
                f"royalty-free license, and {company} may republish "
                "aggregated, anonymized findings."
            ),
            fallback_text=(
                "The customer additionally receives a perpetual, "
                "royalty-free license to the calibration priors generated "
                "specifically from the customer's own cell data; the "
                "methodology itself remains proprietary."
            ),
        ),
        Clause(
            number=7,
            title="Confidentiality and mutual NDA",
            recommended_text=(
                "Both parties keep the other's non-public information "
                "confidential for 3 years from the effective date. "
                "Standard carve-outs apply: information already public, "
                "independently developed, or required to be disclosed by "
                "law or regulation."
            ),
            fallback_text=(
                "The confidentiality term extends to 5 years from the "
                "effective date at the customer's request."
            ),
        ),
        Clause(
            number=8,
            title="Acceptance and signature block",
            recommended_text=(
                "This agreement is accepted by the signatures below and is "
                "effective on the date of the last signature. Customer "
                "signatory: name, title, date. "
                f"{company} signatory: name, title, date."
            ),
            fallback_text=(
                "Electronic acceptance is valid: an exchange of signed "
                "PDFs, or a contractually recognized e-signature exchanged "
                "by email between the signatories, constitutes acceptance."
            ),
        ),
    ]


class CaseStudyAgreement(BaseModel):
    """The full agreement: variable data plus the eight standard clauses."""

    model_config = ConfigDict(extra="forbid")

    agreement_id: str
    customer_code: str
    effective_date: date
    assessment_fee_usd: float
    anonymization_election: AnonymizationMethod
    #: Clause number -> elected variant; clauses not listed are "recommended".
    clause_elections: Dict[int, ClauseElection] = {}

    #: The eight clauses.  Auto-built from the brand record and the fee
    #: when not provided, so serialized agreements round-trip unchanged.
    clauses: List[Clause] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _build_clauses_if_absent(cls, data):
        if isinstance(data, dict) and not data.get("clauses"):
            rebuilt = {
                **data,
                "clauses": [
                    c.model_dump()
                    for c in build_clauses(
                        float(data["assessment_fee_usd"]),
                        data["anonymization_election"],
                    )
                ],
            }
            return rebuilt
        return data

    @model_validator(mode="after")
    def _clauses_must_be_exactly_eight(self) -> "CaseStudyAgreement":
        if len(self.clauses) != 8:
            raise ValueError(
                f"a case-study agreement has exactly 8 clauses, got "
                f"{len(self.clauses)}"
            )
        if [c.number for c in self.clauses] != list(range(1, 9)):
            raise ValueError("clauses must be numbered 1..8 in order")
        return self

    def elected_text(self, clause_number: int) -> str:
        """The operative text of one clause under its elected variant."""
        clause = self.clauses[clause_number - 1]
        election = self.clause_elections.get(clause_number, "recommended")
        if election == "recommended":
            return clause.recommended_text
        return clause.fallback_text


__all__ = [
    "ANONYMIZATION_OPTIONS",
    "AnonymizationMethod",
    "CaseStudyAgreement",
    "Clause",
    "ClauseElection",
    "build_clauses",
]
