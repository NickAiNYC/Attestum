"""robocore.cognitive.prompt_registry -- tamper-evident prompt registry.

Deliverable 4 of the robocore Phase-1 runtime.

Every prompt shipped to an LLM role is an immutable, content-addressed
asset.  The registry binds each prompt *version* to:

  * the sha256 of the exact file bytes currently on disk,
  * the role it serves (planner / critic / diagnoser),
  * the Pydantic model its output must validate against,
  * its default token budget.

Loading is fail-closed: a prompt file whose sha256 is not registered in
``SCHEMA_META`` raises :class:`UnregisteredPromptError`.  This means a
prompt can never be silently edited -- any byte change unregisters it until
an explicit, reviewable code change re-registers the new hash.

Render is equally strict: every ``{{slot}}`` marker in the template must be
consumed by a keyword argument (no missing slots, no extra slots), and the
rendered output must contain no unresolved ``{{...}}`` markers afterwards.
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, ConfigDict

from robocore.cognitive.schemas import CriticVerdict, FailureDiagnosis
from robocore.contracts import HypothesisProposal

__all__ = [
    "PromptRole",
    "PromptSchemaMeta",
    "SCHEMA_META",
    "DEFAULT_TOKEN_BUDGETS",
    "UnregisteredPromptError",
    "UnresolvedSlotError",
    "UnknownPromptVersionError",
    "PromptRegistry",
    "PROMPTS_DIR",
    "sha256_file",
]

PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

_MARKER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class PromptRole(str, Enum):
    PLANNER = "PLANNER"
    CRITIC = "CRITIC"
    DIAGNOSER = "DIAGNOSER"


#: Default token budgets by role (spec: planner 8000, critic 2000,
#: diagnoser 4000; tiktoken cl100k_base accounting).
DEFAULT_TOKEN_BUDGETS: Dict[PromptRole, int] = {
    PromptRole.PLANNER: 8000,
    PromptRole.CRITIC: 2000,
    PromptRole.DIAGNOSER: 4000,
}


class PromptSchemaMeta(BaseModel):
    """Registration record for one prompt version."""

    model_config = ConfigDict(frozen=True)

    version: str
    role: PromptRole
    file_sha256: str  # sha256 over the exact prompt file bytes
    default_token_budget: int
    output_model: str  # dotted path of the output contract model
    slots: List[str]  # every {{slot}} marker the template declares


_SCHEMAS: Dict[str, PromptSchemaMeta] = {
    "planner_v1": PromptSchemaMeta(
        version="planner_v1",
        role=PromptRole.PLANNER,
        file_sha256=(
            "20c39ed809232296118f72818dc79b0768993ed2cdb736a59c53b584be648811"
        ),
        default_token_budget=DEFAULT_TOKEN_BUDGETS[PromptRole.PLANNER],
        output_model="robocore.contracts.HypothesisProposal",
        slots=[
            "task_name",
            "champion_hash",
            "token_budget",
            "anti_loop_directive",
            "context_markdown",
        ],
    ),
    "critic_v1": PromptSchemaMeta(
        version="critic_v1",
        role=PromptRole.CRITIC,
        file_sha256=(
            "2ab5490103d07227338976104dfa060b0e6141385a6dd9a0e4f64fc54a01b5f1"
        ),
        default_token_budget=DEFAULT_TOKEN_BUDGETS[PromptRole.CRITIC],
        output_model="robocore.cognitive.schemas.CriticVerdict",
        slots=[
            "task_name",
            "rationale",
            "predicted_delta",
            "proposal_diff",
            "champion_wilson_low",
            "rejected_similar_digest",
        ],
    ),
    "diagnoser_v1": PromptSchemaMeta(
        version="diagnoser_v1",
        role=PromptRole.DIAGNOSER,
        file_sha256=(
            "50be3133439af5df6efceff2d4c28a4c5e6553722dfbd96d355b6b77d52ee5a4"
        ),
        default_token_budget=DEFAULT_TOKEN_BUDGETS[PromptRole.DIAGNOSER],
        output_model="robocore.cognitive.schemas.FailureDiagnosis",
        slots=[
            "task_name",
            "failure_label",
            "telemetry_digest",
            "keyframe_hashes",
        ],
    ),
}

#: The schema_meta registration table.  Keyed by version; the hash index
#: below is the actual gate (a file is legal iff its sha256 is a value
#: registered here).
SCHEMA_META: Dict[str, PromptSchemaMeta] = _SCHEMAS

_BY_SHA256: Dict[str, str] = {
    meta.file_sha256: version for version, meta in _SCHEMAS.items()
}

#: Output contract model per prompt version (resolved eagerly so a broken
#: import fails at registry import time, never mid-experiment).
_OUTPUT_MODELS: Dict[str, type[BaseModel]] = {
    "planner_v1": HypothesisProposal,
    "critic_v1": CriticVerdict,
    "diagnoser_v1": FailureDiagnosis,
}


class UnregisteredPromptError(Exception):
    """A prompt file that cannot be bound to a schema_meta registration.

    Covers three fail-closed cases: unregistered sha256 (includes silently
    edited files -- the hash no longer matches), a file whose bytes are
    registered under a different version name than its filename, and two
    files claiming the same version.
    """

    def __init__(self, path: Path, sha256: str, reason: str = "") -> None:
        detail = reason or (
            "sha256 is not registered in SCHEMA_META (unregistered or"
            " silently edited); register the new hash via an explicit code"
            " change, or restore the original file bytes"
        )
        super().__init__(
            f"prompt file {Path(path).name} (sha256 {sha256[:16]}...):"
            f" {detail}"
        )
        self.path = Path(path)
        self.sha256 = sha256


class UnresolvedSlotError(Exception):
    """Slot mismatch during render: missing slot value, extra slot value, or
    an unresolved ``{{...}}`` marker surviving in the rendered output."""

    def __init__(self, version: str, detail: str) -> None:
        super().__init__(f"prompt {version}: {detail}")
        self.version = version
        self.detail = detail


class UnknownPromptVersionError(Exception):
    """A prompt version that exists in no registration entry."""

    def __init__(self, version: str) -> None:
        super().__init__(
            f"unknown prompt version {version!r};"
            f" registered: {sorted(_SCHEMAS)}"
        )
        self.version = version


def sha256_file(path: Path) -> str:
    """sha256 over the exact file bytes (the registry's identity function)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PromptRegistry:
    """Loads, verifies and renders the cognitive-plane prompts.

    Usage::

        registry = PromptRegistry.load_default()
        rendered = registry.render(
            "planner_v1",
            task_name=...,
            champion_hash=...,
            token_budget=...,
            anti_loop_directive=...,
            context_markdown=...,
        )
        model = registry.output_model("planner_v1")  # for LLM reply parsing
    """

    def __init__(self, prompts_dir: Path, entries: Dict[str, str]) -> None:
        """Low-level constructor.  Prefer :meth:`load_default` /
        :meth:`load_dir`.

        ``entries`` maps version -> verified file path (only paths whose
        bytes were hash-checked against ``SCHEMA_META``).
        """
        self._prompts_dir = Path(prompts_dir)
        self._entries: Dict[str, Path] = {
            v: Path(p) for v, p in entries.items()
        }

    # -- loading -----------------------------------------------------------

    @classmethod
    def load_dir(cls, prompts_dir: Path) -> "PromptRegistry":
        """Load and hash-verify every ``*.md`` prompt in a directory.

        Raises :class:`UnregisteredPromptError` for the first file whose
        sha256 is not registered (fail-closed load; nothing is cached from
        a partially-verified directory).
        """
        prompts_dir = Path(prompts_dir)
        if not prompts_dir.is_dir():
            raise FileNotFoundError(f"prompts directory not found: {prompts_dir}")
        entries: Dict[str, str] = {}
        for path in sorted(prompts_dir.glob("*.md")):
            sha = sha256_file(path)
            version = _BY_SHA256.get(sha)
            if version is None:
                raise UnregisteredPromptError(path, sha)
            if path.stem != version:
                raise UnregisteredPromptError(
                    path,
                    sha,
                    reason=(
                        f"file content is registered as version {version!r}"
                        f" but the file is named {path.stem!r}; every prompt"
                        " version must live in its own named file"
                    ),
                )
            if version in entries:
                raise UnregisteredPromptError(
                    path,
                    sha,
                    reason=(
                        f"version {version!r} is already loaded from"
                        f" {entries[version]}; duplicate prompt files are"
                        " refused"
                    ),
                )
            entries[version] = str(path)
        missing = sorted(set(_SCHEMAS) - set(entries))
        if missing:
            raise UnknownPromptVersionError(missing[0])
        return cls(prompts_dir, entries)

    @classmethod
    def load_default(cls) -> "PromptRegistry":
        """Load the shipped ``robocore/cognitive/prompts`` directory."""
        return cls.load_dir(PROMPTS_DIR)

    # -- introspection -----------------------------------------------------

    def versions(self) -> List[str]:
        return sorted(self._entries)

    def meta(self, version: str) -> PromptSchemaMeta:
        if version not in _SCHEMAS:
            raise UnknownPromptVersionError(version)
        if version not in self._entries:
            raise UnknownPromptVersionError(version)
        return _SCHEMAS[version]

    def registered_hash(self, version: str) -> str:
        """The sha256 registered for (and verified at load time for) a version."""
        return self.meta(version).file_sha256

    def lookup_by_hash(self, file_sha256: str) -> PromptSchemaMeta:
        """Inverse map: sha256 -> registration (round-trip identity check)."""
        version = _BY_SHA256.get(file_sha256)
        if version is None:
            raise UnregisteredPromptError(
                Path(f"<sha256:{file_sha256[:12]}>"), file_sha256
            )
        return _SCHEMAS[version]

    def output_model(self, version: str) -> type[BaseModel]:
        """Pydantic model the version's LLM output must validate against."""
        if version not in _OUTPUT_MODELS:
            raise UnknownPromptVersionError(version)
        return _OUTPUT_MODELS[version]

    def default_budget(self, version: str) -> int:
        return self.meta(version).default_token_budget

    # -- rendering ---------------------------------------------------------

    def template(self, version: str) -> str:
        path = self._entries.get(version)
        if path is None:
            raise UnknownPromptVersionError(version)
        return Path(path).read_text(encoding="utf-8")

    def render(self, version: str, **slots: str) -> str:
        """Render a verified template with exactly its declared slots.

        Fail-closed in both directions:
          * every ``{{slot}}`` marker in the template must receive a value
            (missing markers -> :class:`UnresolvedSlotError`);
          * every keyword must correspond to a marker (extras -> same);
          * the rendered text must contain zero surviving ``{{...}}``
            markers (a slot value containing a marker is rejected too).
        """
        text = self.template(version)
        markers = _MARKER_RE.findall(text)
        expected = set(markers)
        provided = set(slots)
        missing = sorted(expected - provided)
        extra = sorted(provided - expected)
        if missing:
            raise UnresolvedSlotError(
                version, f"missing slot values for markers {missing}"
            )
        if extra:
            raise UnresolvedSlotError(
                version, f"extra slot values with no marker {extra}"
            )

        def _sub(match: "re.Match[str]") -> str:
            return str(slots[match.group(1)])

        rendered = _MARKER_RE.sub(_sub, text)
        residue = _MARKER_RE.findall(rendered)
        if residue:
            raise UnresolvedSlotError(
                version,
                f"unresolved markers remain in output: {sorted(set(residue))}",
            )
        if "{{" in rendered:
            raise UnresolvedSlotError(
                version, "marker-like text remains in output after render"
            )
        return rendered
