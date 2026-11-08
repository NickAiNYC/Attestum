"""attestum.brand -- the single source of truth for the company brand.

Every module, template, and renderer in this package imports BRAND from
here instead of hardcoding the company name.  The repository-wide
acceptance script (scripts/smoke_attestum_brand.py) enforces this: a
case-sensitive grep for the literal company name outside a small
allowlist (this module, the root README, the strategy summary, and
docs/) must return zero hits.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Brand:
    """Immutable brand record; the field defaults are the canonical values.

    Attributes:
        name: company name.
        tagline: the one-line positioning statement.
        domain: primary web domain.
        repo_url: canonical public repository URL.
        package_version: version of the commercial layer package.
        contact_email: public contact address.
    """

    name: str = "Attestum"
    tagline: str = (
        "The intelligence layer for continuously improving industrial automation."
    )
    domain: str = "attestum.com"
    repo_url: str = "https://github.com/NickAiNYC/Attestum"
    package_version: str = "0.1.0"
    contact_email: str = "contact@attestum.com"


BRAND = Brand()
