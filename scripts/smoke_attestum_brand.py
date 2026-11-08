#!/usr/bin/env python3
"""smoke_attestum_brand.py -- acceptance script for Item A (brand
reconciliation and repo topology).

Seven checks, each raising on failure (fail-closed; no PASS line is
printed unless every check passed):

  1. The brand record (attestum.brand.Brand) instantiates with
     non-empty fields and matches the module-level BRAND constant.
  2. The root README's first 100 words contain the company name and
     the tagline, and do NOT contain the runtime package name.
  3. The README's "## Open-source runtime" heading sits below line 20.
  4. The strategy summary contains the load-bearing sentence verbatim.
  5. The roadmap records all six deferred items.
  6. Zero hardcoded company-name literals outside the allowlist
     (this module, the root README, the strategy summary, and docs/).
  7. All four pre-existing runtime acceptance scripts still pass when
     invoked as subprocesses (exit codes checked).

Run: python scripts/smoke_attestum_brand.py   ->   "PASS 7/7"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attestum.brand import BRAND, Brand  # noqa: E402

COMPANY = BRAND.name
TAGLINE = BRAND.tagline
LOAD_BEARING_SENTENCE = (
    "The company is built by producing two independently verified results "
    "where the second costs less than the first. Everything else is secondary."
)
DEFERRED_ITEMS = (
    "robocorectl",
    "Phase 1\u21922 hardening",
    "cognitive/prompt/safety",
    "custom hardware",
    "multi-industry",
    "autonomous improvement ownership",
)
PREEXISTING_ACCEPTANCE = (
    "scripts/smoke_robocore.py",
    "scripts/smoke_context_builder.py",
    "scripts/smoke_ast_fuzzing.py",
    "scripts/verify_mujoco_backend.py",
)
ALLOWED_LITERAL_FILES = frozenset(
    {
        "README.md",
        "ATTESTUM_STRATEGY.md",
        "attestum/brand.py",
    }
)
ALLOWED_LITERAL_TOP_DIRS = frozenset({"docs"})
SEARCH_SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".hypothesis", ".pytest_cache", ".venv", "node_modules"}
)


def check_brand_record() -> str:
    record = Brand()
    for field_name in (
        "name",
        "tagline",
        "domain",
        "repo_url",
        "package_version",
        "contact_email",
    ):
        value = getattr(record, field_name)
        assert isinstance(value, str) and value.strip(), f"Brand.{field_name} is empty"
        assert value == getattr(BRAND, field_name), (
            f"Brand.{field_name} default diverges from the BRAND constant"
        )
    return "brand record instantiates with non-empty fields"


def check_readme_first_hundred_words() -> str:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    first_hundred = " ".join(text.split()[:100])
    assert COMPANY in first_hundred, "company name missing from the first 100 words"
    assert TAGLINE in first_hundred, "tagline missing from the first 100 words"
    assert "robocore" not in first_hundred, (
        "the runtime package name must not appear in the first 100 words"
    )
    return "README first 100 words: company + tagline present, runtime absent"


def check_readme_runtime_heading() -> str:
    lines = (REPO_ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    matches = [
        index
        for index, line in enumerate(lines, start=1)
        if line.strip() == "## Open-source runtime"
    ]
    assert matches, "## Open-source runtime heading missing from README"
    assert matches[0] > 20, (
        f"'## Open-source runtime' sits at line {matches[0]}; it must be below line 20"
    )
    return f"'## Open-source runtime' heading at line {matches[0]} (> 20)"


def check_strategy_sentence() -> str:
    text = (REPO_ROOT / "ATTESTUM_STRATEGY.md").read_text(encoding="utf-8")
    assert LOAD_BEARING_SENTENCE in text, (
        "the load-bearing sentence is not present verbatim in the strategy summary"
    )
    return "strategy summary carries the load-bearing sentence verbatim"


def check_roadmap_deferred() -> str:
    text = (REPO_ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8").lower()
    missing = [item for item in DEFERRED_ITEMS if item.lower() not in text]
    assert not missing, f"roadmap is missing deferred items: {missing}"
    assert "## deferred until first signed assessment" in text, (
        "roadmap is missing the '## Deferred until first signed assessment' heading"
    )
    return "roadmap records all six deferred items under the required heading"


def check_no_hardcoded_company_name() -> str:
    hits = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        if any(part in SEARCH_SKIP_DIRS for part in relative.split("/")):
            continue
        if relative in ALLOWED_LITERAL_FILES:
            continue
        if relative.split("/")[0] in ALLOWED_LITERAL_TOP_DIRS:
            continue
        try:
            blob = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, ValueError):
            continue
        if COMPANY in blob:
            hits.append(relative)
    assert not hits, f"hardcoded company-name literals outside the allowlist: {hits}"
    return "zero hardcoded company-name literals outside the allowlist"


def check_preexisting_acceptance() -> str:
    for script in PREEXISTING_ACCEPTANCE:
        run = subprocess.run(
            [sys.executable, script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert run.returncode == 0, (
            f"{script} exited {run.returncode}\n"
            f"--- stdout tail ---\n{run.stdout[-1500:]}\n"
            f"--- stderr tail ---\n{run.stderr[-1500:]}"
        )
    return "all four pre-existing acceptance scripts pass (subprocess, exit 0)"


def main() -> int:
    checks = (
        ("brand record", check_brand_record),
        ("README first 100 words", check_readme_first_hundred_words),
        ("README runtime heading", check_readme_runtime_heading),
        ("strategy sentence", check_strategy_sentence),
        ("roadmap deferred items", check_roadmap_deferred),
        ("no hardcoded company name", check_no_hardcoded_company_name),
        ("pre-existing acceptance", check_preexisting_acceptance),
    )
    passed = 0
    for name, check in checks:
        detail = check()
        passed += 1
        print(f"  ok  {name}: {detail}")
    print(f"PASS {passed}/{len(checks)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
