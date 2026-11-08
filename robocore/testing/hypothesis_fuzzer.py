"""robocore.testing.hypothesis_fuzzer -- property-based fuzzing & corpora.

Deliverable 5 of the robocore Phase-1 runtime.

Three fuzz targets (all via ``hypothesis``, derandomized for
reproducibility):

* **apply_diff** -- valid generated diffs must round-trip byte-exactly;
  strictly-invalid structural corruptions (dropped/swapped headers,
  non-numeric/negative/smaller/larger hunk counts, context & deletion text
  mutations, junk prefixes, reordered hunks) must ALWAYS be rejected with
  non-empty ``rejects`` and never raise.  Malformed input can never
  produce a valid-looking patch.
* **ast_validator** -- sources assembled from benign + forbidden snippet
  pools must be accepted iff no forbidden snippet is present; arbitrary
  junk text must always yield a receipt, never an exception.
* **envelope_generator** -- for arbitrary override values (including NaN
  and infinities): either ``ValueError`` or an envelope in which every
  per-joint bound lies within ``PHYSICAL_LIMITS``; unknown keys always
  raise.

Corpora (materialized deterministically on disk by
:func:`materialize_evasion_corpus` / :func:`materialize_baseline_corpus`):

* ``ast_evasions/`` -- 55 adversarial Python files, each beginning with
  ``# @reject_reason: <reason>``; every one must be rejected with the
  annotated reason in the receipt (0 false negatives).
* ``baseline_harnesses/`` -- 56 legitimate harness patches touching
  harness.yaml / reward.py / curriculum.yaml / domain_rand.yaml; every one
  must apply cleanly and pass AST validation (0 false rejections).

The evasion files are NEVER executed -- they are text-scanned and
``ast.parse``d only, which is the entire point of the corpus.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from hypothesis import given, settings
from hypothesis import strategies as st

from robocore.safety.ast_guard import validate_source
from robocore.safety.diff_engine import apply_diff, unified_diff_text
from robocore.safety.envelope import (
    ENVELOPE_OVERRIDE_KEYS,
    PHYSICAL_LIMITS,
    envelope_generator,
)

__all__ = [
    "TESTING_DIR",
    "EVASION_DIR",
    "BASELINE_DIR",
    "BASELINE_HARNESS_FILES",
    "EvasionCase",
    "BaselineCase",
    "materialize_evasion_corpus",
    "materialize_baseline_corpus",
    "load_evasion_corpus",
    "load_baseline_corpus",
    "FuzzOutcome",
    "run_apply_diff_fuzz",
    "run_ast_fuzz",
    "run_envelope_fuzz",
    "run_all_fuzz",
    "make_valid_diff",
    "TOTAL_FUZZ_ITERATIONS",
]

TESTING_DIR = Path(__file__).resolve().parent
EVASION_DIR = TESTING_DIR / "ast_evasions"
BASELINE_DIR = TESTING_DIR / "baseline_harnesses"

#: Budget across all fuzz properties; the acceptance gate requires at
#: least 10,000 executed examples with zero unhandled exceptions.
TOTAL_FUZZ_ITERATIONS = 10_500


# ---------------------------------------------------------------------------
# Adversarial corpus (>= 50 files, each '# @reject_reason: <reason>' led)
# ---------------------------------------------------------------------------

_E = Tuple[str, str, str]  # (slug, expected_reason_substring, code)

EVASION_TEMPLATES: List[_E] = [
    # -- forbidden imports ------------------------------------------------
    ("import_os", "forbidden import", 'import os\nos.system("id")\n'),
    ("import_sys", "forbidden import", "import sys\nargv = sys.argv\n"),
    (
        "import_subprocess",
        "forbidden import",
        'import subprocess\nsubprocess.run(["id"])\n',
    ),
    (
        "import_socket",
        "forbidden import",
        "import socket\ns = socket.socket()\n",
    ),
    (
        "import_ctypes",
        "forbidden import",
        "import ctypes\nlib = ctypes.CDLL(None)\n",
    ),
    (
        "import_resource",
        "forbidden import",
        "import resource\nlim = resource.getrlimit(0)\n",
    ),
    (
        "import_signal",
        "forbidden import",
        "import signal\nsignal.signal(2, None)\n",
    ),
    (
        "import_atexit",
        "forbidden import",
        "import atexit\natexit.register(int)\n",
    ),
    ("import_gc", "forbidden import", "import gc\ngc.collect()\n"),
    (
        "import_weakref",
        "forbidden import",
        "import weakref\nr = weakref.ref(object())\n",
    ),
    (
        "import_shutil",
        "forbidden import",
        'import shutil\nshutil.rmtree("/tmp/robocore-fixture")\n',
    ),
    (
        "import_tempfile",
        "forbidden import",
        "import tempfile\nd = tempfile.mkdtemp()\n",
    ),
    (
        "import_aliased",
        "forbidden import",
        'import subprocess as sp\nsp.Popen("id")\n',
    ),
    (
        "from_os_import",
        "forbidden import",
        'from os import system\nsystem("id")\n',
    ),
    (
        "import_os_alias",
        "forbidden import",
        'import os as o\no.system("id")\n',
    ),
    (
        "import_inspect",
        "forbidden import",
        "import inspect\nframes = inspect.stack()\n",
    ),
    (
        "import_ctypes_buffer",
        "forbidden import",
        "import ctypes\nbuf = ctypes.create_string_buffer(8)\n",
    ),
    # -- dunder abuse -------------------------------------------------------
    (
        "dunder_subclasses",
        "dunder traversal",
        "subs = object.__subclasses__()\n",
    ),
    (
        "dunder_reduce",
        "dunder traversal",
        "r = object.__reduce__()\n",
    ),
    ("dunder_mro", "dunder traversal", "class C:\n    pass\nm = C.__mro__\n"),
    (
        "dunder_bases",
        "dunder traversal",
        "class A:\n    pass\nb = A.__bases__\n",
    ),
    (
        "dunder_globals",
        "dunder traversal",
        "def f():\n    pass\ng = f.__globals__\n",
    ),
    (
        "dunder_builtins",
        "dunder traversal",
        "b = __builtins__\n",
    ),
    (
        "dunder_loader",
        "dunder traversal",
        'm = __loader__.load_module("os")\n',
    ),
    (
        "dunder_builtins_dict",
        "dunder traversal",
        'import builtins\nbuiltins.__dict__["eval"] = print\n',
    ),
    # -- metaclass injection --------------------------------------------------
    (
        "metaclass_keyword",
        "metaclass injection",
        "class Meta(type):\n    pass\nclass C(metaclass=Meta):\n    pass\n",
    ),
    (
        "metaclass_type_call",
        "metaclass injection",
        "C = type('C', (object,), {'K': 1})\n",
    ),
    # -- dynamic code execution -----------------------------------------------
    ("eval_call", "dynamic code execution", 'x = eval("1+1")\n'),
    ("exec_call", "dynamic code execution", 'exec("y = 1")\n'),
    (
        "compile_call",
        "dynamic code execution",
        "z = compile('1', '<s>', 'eval')\n",
    ),
    (
        "exec_compile_chain",
        "dynamic code execution",
        "w = exec(compile('1', '', 'eval'))\n",
    ),
    (
        "eval_in_comprehension",
        "dynamic code execution",
        'q = [eval("1") for _ in range(1)]\n',
    ),
    (
        "eval_in_comment",
        "dynamic code execution",
        "x = 1  # call eval() here\n",
    ),
    # -- __import__ -----------------------------------------------------------
    (
        "dunder_import_call",
        "__import__ escape",
        'm = __import__("os")\n',
    ),
    (
        "dunder_import_lambda",
        "__import__ escape",
        'f = lambda: __import__("sys")\n',
    ),
    # -- string-concat getattr chains -------------------------------------------
    (
        "getattr_concat_open",
        "dynamic attribute access",
        "g = getattr(__builtins__, 'op' + 'en')\n",
    ),
    (
        "getattr_concat_system",
        "dynamic attribute access",
        "s = 'sy' + 'stem'\nh = getattr(__import__('os'), s)\n",
    ),
    (
        "setattr_override",
        "dynamic attribute access",
        "setattr(str, 'upper', str.lower)\n",
    ),
    (
        "getattr_builtins_name",
        "dynamic attribute access",
        'n = "ev" + "al"\nf = getattr(__builtins__, n)\nf("1")\n',
    ),
    # -- sys.modules mutation ------------------------------------------------
    (
        "sys_modules_assign",
        "module mutation",
        'import sys\nsys.modules["fake"] = None\n',
    ),
    (
        "sys_modules_delete",
        "module mutation",
        'import sys\ndel sys.modules["os"]\n',
    ),
    # -- builtins mutation ----------------------------------------------------
    (
        "builtins_open_override",
        "builtins mutation",
        "import builtins\nbuiltins.open = None\n",
    ),
    # -- pickle deserialization -------------------------------------------------
    (
        "pickle_loads",
        "deserialization (pickle)",
        'import pickle\nobj = pickle.loads(b"")\n',
    ),
    (
        "pickle_load_alias",
        "forbidden import",
        'import pickle as p\nobj = p.load(open("/tmp/x", "rb"))\n',
    ),
    # -- thread creation ---------------------------------------------------------
    (
        "threading_thread",
        "thread creation",
        "import threading\nt = threading.Thread(target=print)\n",
    ),
    (
        "thread_start_new",
        "forbidden import",
        "import _thread\n_thread.start_new_thread(print, (1,))\n",
    ),
    # -- unbounded loops ----------------------------------------------------------
    ("while_true_plain", "unbounded loop", "while True:\n    acc = 1\n"),
    (
        "while_true_nested_break",
        "unbounded loop",
        "while True:\n    for i in range(3):\n        if i == 2:\n            break\n",
    ),
    (
        "while_true_func",
        "unbounded loop",
        "def spin():\n    while True:\n        pass\n",
    ),
    # -- NaN / Inf -----------------------------------------------------------------
    ("float_nan", "NaN/Inf", 'v = float("nan")\n'),
    ("float_inf", "NaN/Inf", 'w = float("inf")\n'),
    # -- file I/O & misc ---------------------------------------------------------
    ("open_etc_passwd", "file I/O (open)", 'fh = open("/etc/passwd")\n'),
    ("breakpoint_trap", "breakpoint trap", "breakpoint()\n"),
    # -- obfuscated references -----------------------------------------------------
    (
        "exec_via_map",
        "forbidden name reference",
        'out = list(map(exec, ["pass"]))\n',
    ),
    ("exec_bare_name", "forbidden name reference", "f = exec\n"),
    (
        "unicode_eval",
        "forbidden name reference",
        "e = ｅval('1')\n",
    ),
    # -- frame introspection (scope creep, pre-AST) ------------------------------
    (
        "sys_getframe",
        "frame introspection",
        "import sys\ndef f():\n    return sys._getframe(1).f_globals\n",
    ),
    (
        "sys_getframe_bare",
        "frame introspection",
        "def g():\n    fr = sys._getframe(0)\n    return fr\n",
    ),
]


@dataclass(frozen=True)
class EvasionCase:
    name: str
    expected_reason: str
    source: str


@dataclass(frozen=True)
class BaselineCase:
    name: str
    patch_text: str
    tag: str


def materialize_evasion_corpus(target_dir: Path = EVASION_DIR) -> int:
    """Write the adversarial corpus (idempotent, deterministic).  Each file
    begins with ``# @reject_reason: <reason>`` followed by the hostile
    code.  Returns the number of files written."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale in target_dir.glob("evasion_*.py"):
        stale.unlink()
    count = 0
    for idx, (slug, reason, code) in enumerate(EVASION_TEMPLATES, start=1):
        path = target_dir / f"evasion_{idx:03d}_{slug}.py"
        body = f"# @reject_reason: {reason}\n" + code
        path.write_text(body, encoding="utf-8")
        count += 1
    return count


def load_evasion_corpus(target_dir: Path = EVASION_DIR) -> List[EvasionCase]:
    cases: List[EvasionCase] = []
    for path in sorted(Path(target_dir).glob("evasion_*.py")):
        text = path.read_text(encoding="utf-8")
        first, _, rest = text.partition("\n")
        if not first.startswith("# @reject_reason: "):
            raise ValueError(f"{path.name}: missing @reject_reason header")
        reason = first[len("# @reject_reason: "):].strip()
        cases.append(EvasionCase(path.name, reason, rest))
    if not cases:
        raise ValueError(f"evasion corpus empty at {target_dir}")
    return cases


# ---------------------------------------------------------------------------
# Baseline corpus (>= 50 legitimate harness patches, 0 false rejections)
# ---------------------------------------------------------------------------

BASELINE_HARNESS_FILES: Dict[str, str] = {
    "harness.yaml": (
        "# robocore champion harness: peg pick-and-place, bench-07\n"
        "control_hz: 1000\n"
        "watchdog_timeout_ms: 5\n"
        "camera_expiry_ms: 20\n"
        "termination_jerk_limit: 42.0\n"
        "max_path_deviation_m: 0.016\n"
        "max_force_newton: 45.0\n"
        "strata_weights:\n"
        "  EASY_CENTER: 0.25\n"
        "  MEDIUM_OBLIQUE: 0.40\n"
        "  HARD_BOUNDARY: 0.20\n"
        "  OCCLUDED: 0.15\n"
        "curriculum_gate: true\n"
    ),
    "reward.py": (
        "import math\n"
        "\n"
        "REWARD_K = 0.30\n"
        "GRASP_BONUS = 1.20\n"
        "RELEASE_PENALTY = 0.80\n"
        "SMOOTHING_ALPHA = 0.15\n"
        "JERK_WEIGHT = 0.05\n"
        "TOLERANCE = 0.05\n"
        "\n"
        "\n"
        "def clamp(value, lo, hi):\n"
        "    return min(max(value, lo), hi)\n"
        "\n"
        "\n"
        "def shaped_reward(err, jerk):\n"
        "    prox = clamp(1.0 - err / TOLERANCE, 0.0, 1.0)\n"
        "    smooth = clamp(SMOOTHING_ALPHA * jerk, 0.0, 1.0)\n"
        "    return REWARD_K * prox + GRASP_BONUS - RELEASE_PENALTY * smooth\n"
        "\n"
        "\n"
        "def jerk_cost(jerk):\n"
        "    return JERK_WEIGHT * math.sqrt(jerk + 1.0)\n"
    ),
    "curriculum.yaml": (
        "# difficulty curriculum for peg insertion\n"
        "stages:\n"
        "  - name: centered_easy\n"
        "    oblique_start_deg: 12.0\n"
        "    samples_per_stage: 24\n"
        "  - name: oblique_medium\n"
        "    oblique_start_deg: 22.0\n"
        "    samples_per_stage: 32\n"
        "  - name: boundary_hard\n"
        "    oblique_start_deg: 34.0\n"
        "    samples_per_stage: 40\n"
        "occlusion_enabled: false\n"
    ),
    "domain_rand.yaml": (
        "# domain randomization band (sim side)\n"
        "friction_range: [0.35, 0.55]\n"
        "damping_range: [0.02, 0.08]\n"
        "mass_range_kg: [0.35, 0.45]\n"
        "latency_ms_range: [4, 9]\n"
        "camera_noise_std: 0.01\n"
    ),
}


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"baseline mutation anchor not unique: {old!r}")
    return text.replace(old, new)


def _reward_k(v: str) -> Tuple[str, str, str]:
    return ("reward.py", f"REWARD_K = {v}", f"reward_k_{v}")


_BASELINE_MUTATIONS: List[Tuple[str, Dict[str, Tuple[str, str]]]] = []


def _build_baseline_mutations() -> List[
    Tuple[str, Dict[str, Tuple[str, str]]]
]:
    """(tag, {file: (old, new)}) entries -- 56 legitimate mutations."""
    if _BASELINE_MUTATIONS:
        return _BASELINE_MUTATIONS
    m = _BASELINE_MUTATIONS

    def add(tag: str, file: str, old: str, new: str) -> None:
        m.append((tag, {file: (old, new)}))

    def add2(
        tag: str, f1: str, o1: str, n1: str, f2: str, o2: str, n2: str
    ) -> None:
        m.append((tag, {f1: (o1, n1), f2: (o2, n2)}))

    # reward.py coefficient sweeps (single-family REWARD_SCALING mutations)
    for v in ("0.28", "0.32", "0.34", "0.36", "0.38", "0.40"):
        add(f"reward_k_{v}", "reward.py", "REWARD_K = 0.30", f"REWARD_K = {v}")
    for v in ("1.05", "1.10", "1.30", "1.35"):
        add(
            f"grasp_bonus_{v}",
            "reward.py",
            "GRASP_BONUS = 1.20",
            f"GRASP_BONUS = {v}",
        )
    for v in ("0.70", "0.75", "0.90", "0.95"):
        add(
            f"release_penalty_{v}",
            "reward.py",
            "RELEASE_PENALTY = 0.80",
            f"RELEASE_PENALTY = {v}",
        )
    for v in ("0.10", "0.12", "0.18", "0.20"):
        add(
            f"smoothing_alpha_{v}",
            "reward.py",
            "SMOOTHING_ALPHA = 0.15",
            f"SMOOTHING_ALPHA = {v}",
        )
    for v in ("0.03", "0.04", "0.06", "0.08"):
        add(
            f"jerk_weight_{v}",
            "reward.py",
            "JERK_WEIGHT = 0.05",
            f"JERK_WEIGHT = {v}",
        )
    for v in ("0.03", "0.04", "0.06", "0.08"):
        add(
            f"tolerance_{v}",
            "reward.py",
            "TOLERANCE = 0.05",
            f"TOLERANCE = {v}",
        )
    # reward.py structural additions (still REWARD_SCALING family)
    add(
        "add_progress_helper",
        "reward.py",
        "def jerk_cost(jerk):",
        "def progress(err):\n"
        "    return clamp(1.0 - abs(err), 0.0, 1.0)\n"
        "\n"
        "\n"
        "def jerk_cost(jerk):",
    )
    add(
        "tighten_clamp_upper",
        "reward.py",
        "    return min(max(value, lo), hi)",
        "    return min(max(value, lo), min(hi, 1.0))",
    )
    add(
        "jerk_cost_quadratic",
        "reward.py",
        "    return JERK_WEIGHT * math.sqrt(jerk + 1.0)",
        "    return JERK_WEIGHT * (jerk * jerk + 1.0)",
    )
    add(
        "shaped_reward_scale",
        "reward.py",
        "    return REWARD_K * prox + GRASP_BONUS - RELEASE_PENALTY * smooth",
        "    return 0.5 * (REWARD_K * prox) + GRASP_BONUS"
        " - RELEASE_PENALTY * smooth",
    )
    # harness.yaml (TERMINATION_TIGHTENING family)
    add("control_hz_750", "harness.yaml", "control_hz: 1000", "control_hz: 750")
    add("watchdog_6ms", "harness.yaml", "watchdog_timeout_ms: 5", "watchdog_timeout_ms: 6")
    add("camera_expiry_18", "harness.yaml", "camera_expiry_ms: 20", "camera_expiry_ms: 18")
    for v in ("38.0", "40.0", "44.0", "46.0"):
        add(
            f"jerk_limit_{v}",
            "harness.yaml",
            "termination_jerk_limit: 42.0",
            f"termination_jerk_limit: {v}",
        )
    for v in ("0.012", "0.014", "0.018", "0.020"):
        add(
            f"path_dev_{v}",
            "harness.yaml",
            "max_path_deviation_m: 0.016",
            f"max_path_deviation_m: {v}",
        )
    add(
        "stratum_weight_easy",
        "harness.yaml",
        "  EASY_CENTER: 0.25",
        "  EASY_CENTER: 0.30",
    )
    add(
        "stratum_weight_oblique",
        "harness.yaml",
        "  MEDIUM_OBLIQUE: 0.40",
        "  MEDIUM_OBLIQUE: 0.35",
    )
    add(
        "stratum_weight_hard",
        "harness.yaml",
        "  HARD_BOUNDARY: 0.20",
        "  HARD_BOUNDARY: 0.25",
    )
    add(
        "stratum_weight_occluded",
        "harness.yaml",
        "  OCCLUDED: 0.15",
        "  OCCLUDED: 0.20",
    )
    add(
        "curriculum_gate_off",
        "harness.yaml",
        "curriculum_gate: true",
        "curriculum_gate: false",
    )
    # curriculum.yaml (CURRICULUM_SHIFT family)
    for v in ("14.0", "16.0", "10.0", "18.0"):
        add(
            f"oblique_easy_{v}",
            "curriculum.yaml",
            "    oblique_start_deg: 12.0",
            f"    oblique_start_deg: {v}",
        )
    for v in ("20.0", "24.0", "26.0"):
        add(
            f"oblique_medium_{v}",
            "curriculum.yaml",
            "    oblique_start_deg: 22.0",
            f"    oblique_start_deg: {v}",
        )
    for v in ("32.0", "36.0"):
        add(
            f"oblique_hard_{v}",
            "curriculum.yaml",
            "    oblique_start_deg: 34.0",
            f"    oblique_start_deg: {v}",
        )
    for v in ("16", "20", "28", "36"):
        add(
            f"samples_easy_{v}",
            "curriculum.yaml",
            "    samples_per_stage: 24",
            f"    samples_per_stage: {v}",
        )
    add(
        "occlusion_on",
        "curriculum.yaml",
        "occlusion_enabled: false",
        "occlusion_enabled: true",
    )
    # domain_rand.yaml (DOMAIN_RAND_PERTURBATION family)
    add(
        "friction_narrow",
        "domain_rand.yaml",
        "friction_range: [0.35, 0.55]",
        "friction_range: [0.40, 0.50]",
    )
    add(
        "friction_wide",
        "domain_rand.yaml",
        "friction_range: [0.35, 0.55]",
        "friction_range: [0.30, 0.60]",
    )
    add(
        "damping_low",
        "domain_rand.yaml",
        "damping_range: [0.02, 0.08]",
        "damping_range: [0.01, 0.05]",
    )
    add(
        "damping_high",
        "domain_rand.yaml",
        "damping_range: [0.02, 0.08]",
        "damping_range: [0.04, 0.10]",
    )
    add(
        "mass_narrow",
        "domain_rand.yaml",
        "mass_range_kg: [0.35, 0.45]",
        "mass_range_kg: [0.38, 0.42]",
    )
    add(
        "latency_narrow",
        "domain_rand.yaml",
        "latency_ms_range: [4, 9]",
        "latency_ms_range: [5, 8]",
    )
    add(
        "camera_noise_half",
        "domain_rand.yaml",
        "camera_noise_std: 0.01",
        "camera_noise_std: 0.005",
    )
    # multi-file legitimate diffs (one section per file)
    add2(
        "reward_and_jerk_limit",
        "reward.py", "REWARD_K = 0.30", "REWARD_K = 0.33",
        "harness.yaml", "termination_jerk_limit: 42.0", "termination_jerk_limit: 40.0",
    )
    add2(
        "curriculum_and_domain",
        "curriculum.yaml", "    samples_per_stage: 32", "    samples_per_stage: 28",
        "domain_rand.yaml", "friction_range: [0.35, 0.55]", "friction_range: [0.37, 0.53]",
    )
    add2(
        "tolerance_and_watchdog",
        "reward.py", "TOLERANCE = 0.05", "TOLERANCE = 0.04",
        "harness.yaml", "watchdog_timeout_ms: 5", "watchdog_timeout_ms: 4",
    )
    add2(
        "pathdev_and_damping",
        "harness.yaml", "max_path_deviation_m: 0.016", "max_path_deviation_m: 0.014",
        "domain_rand.yaml", "damping_range: [0.02, 0.08]", "damping_range: [0.02, 0.07]",
    )
    return m


def _mutation_files(
    mutation: Dict[str, Tuple[str, str]]
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(expected new tree, old->new per file) for one mutation entry."""
    new_files: Dict[str, str] = {}
    for path, (old, new) in mutation.items():
        base = BASELINE_HARNESS_FILES[path]
        new_files[path] = _replace_once(base, old, new)
    return new_files, dict(mutation)


def materialize_baseline_corpus(target_dir: Path = BASELINE_DIR) -> int:
    """Write the baseline corpus (idempotent, deterministic).  Returns the
    number of ``.patch`` files written."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for stale in target_dir.glob("baseline_*.patch"):
        stale.unlink()
    count = 0
    for idx, (tag, mutation) in enumerate(_build_baseline_mutations(), start=1):
        new_files, _ = _mutation_files(mutation)
        sections = []
        for path, new_text in new_files.items():
            section = unified_diff_text(
                BASELINE_HARNESS_FILES[path], new_text, path=path
            )
            if section.strip() == "":
                raise ValueError(
                    f"baseline mutation {tag!r} produced an empty diff"
                    f" section for {path}"
                )
            sections.append(section)
        patch_text = "".join(sections)
        (target_dir / f"baseline_{idx:03d}_{tag}.patch").write_text(
            patch_text, encoding="utf-8"
        )
        count += 1
    return count


def load_baseline_corpus(target_dir: Path = BASELINE_DIR) -> List[BaselineCase]:
    cases: List[BaselineCase] = []
    for path in sorted(Path(target_dir).glob("baseline_*.patch")):
        cases.append(
            BaselineCase(path.name, path.read_text(encoding="utf-8"), path.stem)
        )
    if not cases:
        raise ValueError(f"baseline corpus empty at {target_dir}")
    return cases


# ---------------------------------------------------------------------------
# Deterministic valid-diff generator (shared by fuzz properties)
# ---------------------------------------------------------------------------


def make_valid_diff(seed: int) -> Tuple[Dict[str, str], str, Dict[str, str]]:
    """Deterministically derive a VALID diff from a seed.

    Returns ``(base_files, diff_text, expected_files)``.  Even seeds touch
    reward.py (single hunk); odd seeds touch harness.yaml twice in distant
    regions (two hunks, for the hunk-reordering corruption).
    """
    mutations = _build_baseline_mutations()
    if seed % 2 == 0:
        tag, mutation = mutations[seed % len(mutations)]
        single = {k: v for k, v in mutation.items()}
        # guarantee single-file, single-region diffs for the even path
        first_file = next(iter(single))
        single = {first_file: single[first_file]}
        new_files, _ = _mutation_files(single)
        diffs = [
            unified_diff_text(
                BASELINE_HARNESS_FILES[p], t, path=p
            )
            for p, t in new_files.items()
        ]
        return dict(BASELINE_HARNESS_FILES), "".join(diffs), new_files
    # odd: two independent harness.yaml limits (guaranteed 2 hunks)
    base = BASELINE_HARNESS_FILES["harness.yaml"]
    a_old, a_new = "control_hz: 1000", f"control_hz: {800 + (seed % 5) * 50}"
    b_old, b_new = (
        "max_path_deviation_m: 0.016",
        f"max_path_deviation_m: 0.01{seed % 6}",
    )
    new_yaml = _replace_once(_replace_once(base, a_old, a_new), b_old, b_new)
    diff_text = unified_diff_text(base, new_yaml, path="harness.yaml")
    # expected = ONLY the touched files (patched carries touched files only)
    expected = {"harness.yaml": new_yaml}
    return dict(BASELINE_HARNESS_FILES), diff_text, expected


_STRICT_CORRUPTIONS = (
    "drop_minus_header",
    "swap_headers",
    "nonnumeric_counts",
    "negative_count",
    "smaller_counts",
    "larger_counts",
    "context_text",
    "deletion_text",
    "junk_prefix",
    "reorder_hunks",
)


def _corrupt(diff_text: str, op: str, seed: int) -> str:
    """Apply one STRICTLY-INVALID structural corruption to a valid diff."""
    lines = diff_text.split("\n")
    hunk_at = next(
        (i for i, ln in enumerate(lines) if ln.startswith("@@ ")), None
    )
    if op == "drop_minus_header":
        idx = next(i for i, ln in enumerate(lines) if ln.startswith("--- "))
        del lines[idx]
    elif op == "swap_headers":
        idx = next(i for i, ln in enumerate(lines) if ln.startswith("--- "))
        lines[idx] = lines[idx].replace("--- ", "+++ ", 1)
    elif op == "nonnumeric_counts":
        if hunk_at is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        lines[hunk_at] = re.sub(r"@@ -\\d+", "@@ -x", lines[hunk_at], count=1)
    elif op == "negative_count":
        if hunk_at is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        lines[hunk_at] = lines[hunk_at].replace(",", ",-", 1)
    elif op == "smaller_counts":
        if hunk_at is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        m = _HUNK_HEADER_RE.match(lines[hunk_at])
        if m is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        old_c = max(0, int(m.group(2)) - 2)
        new_c = max(0, int(m.group(4)) - 2)
        lines[hunk_at] = (
            f"@@ -{m.group(1)},{old_c} +{m.group(3)},{new_c} @@"
        )
    elif op == "larger_counts":
        if hunk_at is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        m = _HUNK_HEADER_RE.match(lines[hunk_at])
        if m is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        lines[hunk_at] = (
            f"@@ -{m.group(1)},{int(m.group(2)) + 3}"
            f" +{m.group(3)},{int(m.group(4)) + 3} @@"
        )
    elif op == "context_text":
        idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if i > (hunk_at or 0) and ln.startswith(" ")
            ),
            None,
        )
        if idx is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        lines[idx] = lines[idx] + "CORRUPTED"
    elif op == "deletion_text":
        idx = next(
            (
                i
                for i, ln in enumerate(lines)
                if i > (hunk_at or 0) and ln.startswith("-")
            ),
            None,
        )
        if idx is None:
            return "\n".join([f"garbage prefix {seed}", *lines])
        lines[idx] = lines[idx] + "CORRUPTED"
    elif op == "junk_prefix":
        lines = [f"garbage prefix {seed}", *lines]
    elif op == "reorder_hunks":
        trailing: List[str] = []
        while lines and lines[-1] == "":
            trailing.append(lines.pop())
        at = [i for i, ln in enumerate(lines) if ln.startswith("@@ ")]
        if len(at) < 2:
            return "\n".join([f"garbage prefix {seed}", *lines, *trailing])
        bounds = at + [len(lines)]
        blocks = [lines[a : bounds[i + 1]] for i, a in enumerate(at)]
        prefix = lines[: at[0]]
        reordered = (
            prefix
            + blocks[1]
            + blocks[0]
            + [ln for blk in blocks[2:] for ln in blk]
        )
        return "\n".join(reordered + trailing)
    corrupted = "\n".join(lines)
    if corrupted == diff_text:
        # the op was a structural no-op on this header shape (e.g. a token
        # that does not occur); every corruption must still be invalid
        return "\n".join([f"garbage prefix {seed}", *lines])
    return corrupted


_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+),(\d+) \+(\d+),(\d+) @@"
)


# ---------------------------------------------------------------------------
# Fuzz properties
# ---------------------------------------------------------------------------


@dataclass
class FuzzOutcome:
    iterations: int = 0
    violations: List[str] = field(default_factory=list)


@settings(max_examples=4000, deadline=None, derandomize=True)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def _prop_diff_roundtrip(seed: int, outcome: FuzzOutcome) -> None:
    outcome.iterations += 1
    base, diff_text, expected = make_valid_diff(seed)
    result = apply_diff(base, diff_text)
    if not result.ok:
        outcome.violations.append(
            f"valid diff rejected (seed={seed}): {result.rejects[:2]}"
        )
        return
    if result.patched != expected:
        outcome.violations.append(
            f"round-trip mismatch (seed={seed}): patched != expected"
        )


@settings(max_examples=2500, deadline=None, derandomize=True)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    op=st.sampled_from(_STRICT_CORRUPTIONS),
)
def _prop_diff_corruption(
    seed: int, op: str, outcome: FuzzOutcome
) -> None:
    outcome.iterations += 1
    base, diff_text, _expected = make_valid_diff(seed)
    corrupted = _corrupt(diff_text, op, seed)
    try:
        result = apply_diff(base, corrupted)
    except Exception as exc:  # noqa: BLE001 - must never happen
        outcome.violations.append(
            f"apply_diff raised on corrupted input (op={op}, seed={seed}):"
            f" {exc!r}"
        )
        return
    if result.ok:
        outcome.violations.append(
            f"corrupted diff accepted (op={op}, seed={seed})"
        )
    elif not result.rejects:
        outcome.violations.append(
            f"corrupted diff rejected without reasons (op={op}, seed={seed})"
        )


@settings(max_examples=500, deadline=None, derandomize=True)
@given(text=st.text(max_size=240))
def _prop_diff_junk(text: str, outcome: FuzzOutcome) -> None:
    outcome.iterations += 1
    try:
        result = apply_diff(dict(BASELINE_HARNESS_FILES), text)
    except Exception as exc:  # noqa: BLE001
        outcome.violations.append(f"apply_diff raised on junk: {exc!r}")
        return
    if result.ok and text.strip() and not text.startswith("--- "):
        # junk that was accepted must still have applied *something*
        if not result.patched:
            outcome.violations.append("junk accepted with empty patch")


_BENIGN_SNIPPETS: Tuple[str, ...] = (
    "x = math.sqrt(2.0)",
    "values = [1.0, 2.0, 3.0]\ntotal = sum(values)",
    "v = 0.5\nclamped = min(max(v, 0.0), 1.0)",
    "acc = 0\nfor i in range(4):\n    acc += i",
    "def step(t):\n    return t * 1.5",
    "y = 1.2\nif y > 0.5:\n    y = 0.5",
    "K = 0.25",
    "w = abs(-3.0) + max(1, 2)",
    "items = sorted([3, 1, 2])",
    "flag = all([True, False]) or any([False])",
)

_FORBIDDEN_SNIPPETS: Tuple[str, ...] = tuple(
    code for _slug, _reason, code in EVASION_TEMPLATES
)


def _assemble_source(
    k: int, n_forbidden: int, n_benign: int
) -> Tuple[str, bool]:
    """Deterministic assembly: benign-only (expect pass) or with >= 1
    forbidden snippet (expect reject)."""
    forbidden_pool = _FORBIDDEN_SNIPPETS
    benign_pool = _BENIGN_SNIPPETS
    parts: List[str] = ["import math", ""]
    for i in range(n_benign):
        parts.append(benign_pool[(k + i) % len(benign_pool)])
        parts.append("")
    has_forbidden = n_forbidden > 0
    for i in range(n_forbidden):
        parts.append(forbidden_pool[(k * 7 + i * 13) % len(forbidden_pool)])
        parts.append("")
    return "\n".join(parts), has_forbidden


@settings(max_examples=3000, deadline=None, derandomize=True)
@given(
    k=st.integers(min_value=0, max_value=2**31 - 1),
    n_forbidden=st.integers(min_value=0, max_value=3),
    n_benign=st.integers(min_value=1, max_value=4),
)
def _prop_ast_mixed(k: int, n_forbidden: int, n_benign: int, outcome: FuzzOutcome) -> None:
    outcome.iterations += 1
    source, expect_reject = _assemble_source(k, n_forbidden, n_benign)
    try:
        receipt = validate_source(source)
    except Exception as exc:  # noqa: BLE001
        outcome.violations.append(f"validator raised: {exc!r}")
        return
    if expect_reject and receipt.ast_passed:
        outcome.violations.append(
            "false negative: forbidden snippet accepted"
        )
    if not expect_reject and not receipt.ast_passed:
        outcome.violations.append(
            f"false positive on benign source: {receipt.ast_errors[:2]}"
        )


@settings(max_examples=500, deadline=None, derandomize=True)
@given(text=st.text(max_size=200))
def _prop_ast_junk(text: str, outcome: FuzzOutcome) -> None:
    outcome.iterations += 1
    try:
        receipt = validate_source(text)
    except Exception as exc:  # noqa: BLE001
        outcome.violations.append(f"validator raised on junk: {exc!r}")
        return
    if not receipt.ast_passed and not receipt.ast_errors:
        outcome.violations.append("rejection without reasons on junk")


@settings(max_examples=2000, deadline=None, derandomize=True)
@given(
    overrides=st.dictionaries(
        st.sampled_from(sorted(ENVELOPE_OVERRIDE_KEYS)),
        st.floats(allow_nan=True, allow_infinity=True, width=64),
        max_size=5,
    )
)
def _prop_envelope_known_keys(overrides: Dict[str, float], outcome: FuzzOutcome) -> None:
    outcome.iterations += 1
    try:
        env = envelope_generator(overrides)
    except ValueError:
        return  # fail-closed rejection is a legal outcome
    except Exception as exc:  # noqa: BLE001
        outcome.violations.append(f"envelope raised {exc!r}")
        return
    for j in range(PHYSICAL_LIMITS.n_joints):
        if not (env.q_pos_lo[j] >= PHYSICAL_LIMITS.q_pos_lo[j]):
            outcome.violations.append(f"q_pos_lo below hard limit at joint {j+1}")
        if not (env.q_pos_hi[j] <= PHYSICAL_LIMITS.q_pos_hi[j]):
            outcome.violations.append(f"q_pos_hi above hard limit at joint {j+1}")
        if not (env.q_vel_max[j] <= PHYSICAL_LIMITS.q_vel_max[j]):
            outcome.violations.append(f"q_vel_max above hard limit at joint {j+1}")
        if not (env.tau_max[j] <= PHYSICAL_LIMITS.tau_max[j]):
            outcome.violations.append(f"tau_max above hard limit at joint {j+1}")
        if not (env.q_pos_lo[j] < env.q_pos_hi[j]):
            outcome.violations.append(f"inverted envelope emitted at joint {j+1}")


@settings(max_examples=500, deadline=None, derandomize=True)
@given(keys=st.lists(st.text(max_size=14), min_size=0, max_size=4))
def _prop_envelope_unknown_keys(keys: List[str], outcome: FuzzOutcome) -> None:
    outcome.iterations += 1
    known = set(ENVELOPE_OVERRIDE_KEYS)
    overrides = {k: 1.0 for k in keys if k not in known}
    if not overrides:
        return
    try:
        envelope_generator(overrides)
    except ValueError:
        return
    except Exception as exc:  # noqa: BLE001
        outcome.violations.append(f"envelope raised {exc!r} on unknown keys")
        return
    outcome.violations.append(
        f"unknown override keys accepted: {sorted(overrides)}"
    )


def run_apply_diff_fuzz(outcome: Optional[FuzzOutcome] = None) -> FuzzOutcome:
    outcome = outcome if outcome is not None else FuzzOutcome()
    _prop_diff_roundtrip(outcome)
    _prop_diff_corruption(outcome)
    _prop_diff_junk(outcome)
    return outcome


def run_ast_fuzz(outcome: Optional[FuzzOutcome] = None) -> FuzzOutcome:
    outcome = outcome if outcome is not None else FuzzOutcome()
    _prop_ast_mixed(outcome)
    _prop_ast_junk(outcome)
    return outcome


def run_envelope_fuzz(outcome: Optional[FuzzOutcome] = None) -> FuzzOutcome:
    outcome = outcome if outcome is not None else FuzzOutcome()
    _prop_envelope_known_keys(outcome)
    _prop_envelope_unknown_keys(outcome)
    return outcome


def run_all_fuzz() -> FuzzOutcome:
    """Run every property; the acceptance gate asserts
    ``outcome.iterations >= 10_000`` and empty violations."""
    outcome = FuzzOutcome()
    run_apply_diff_fuzz(outcome)
    run_ast_fuzz(outcome)
    run_envelope_fuzz(outcome)
    return outcome
