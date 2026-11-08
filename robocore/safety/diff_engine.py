"""robocore.safety.diff_engine -- strict unified-diff application engine.

Deliverable 5 support module (additive extension; Batch 1 shipped no diff
engine).  ``apply_diff`` is the single choke point through which every
proposed harness mutation enters the system.  Its contract is fail-closed:

* malformed input NEVER produces a valid-looking patch: every structural
  error (bad headers, bogus hunk counts, truncated hunks, context mismatch,
  out-of-order hunks, trailing garbage, file deletion) is reported in
  ``rejects`` and ``ok`` comes back ``False``;
* the function never raises on malformed input -- the result object is the
  only error channel;
* when ``ok`` is ``True`` the patched text is exactly the base with the
  hunks applied, context-verified line by line.

Supported format (the harness-diff contract): zero or more git-style file
sections, each ``--- a/<path>`` / ``+++ b/<path>`` (new files via
``--- /dev/null``), followed by ``@@`` hunks with GNU-style counts.
``\\ No newline at end of file`` markers are honored.  Deletions
(``+++ /dev/null``) are rejected: the planner mutates harness files, it
never removes them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["DiffApplyResult", "apply_diff", "unified_diff_text"]

_HUNK_RE = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$"
)
_GIT_PREFIX_RE = re.compile(r"^(diff --git |index |old mode |new mode |"
                            r"new file mode |deleted file mode |similarity )")


@dataclass(frozen=True)
class DiffApplyResult:
    """Outcome of applying a unified diff to a harness file tree."""

    ok: bool
    patched: Dict[str, str] = field(default_factory=dict)  # touched files
    rejects: Tuple[str, ...] = ()

    def __bool__(self) -> bool:  # result reads as its verdict
        return self.ok


def unified_diff_text(
    base_text: str, new_text: str, *, path: str, context: int = 3
) -> str:
    """Render a canonical unified diff (the generator used by the baseline
    corpus and by tests; ``difflib`` output is exactly what ``apply_diff``
    accepts)."""
    import difflib

    base_lines = base_text.splitlines()
    new_lines = new_text.splitlines()
    diff = difflib.unified_diff(
        base_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
        n=context,
    )
    return "\n".join(diff) + "\n"


def _split_keep_terminator_info(text: str) -> Tuple[List[str], bool]:
    """(lines without terminators, whether the text ends with a newline)."""
    if text == "":
        return [], False
    ends_nl = text.endswith("\n")
    lines = text.split("\n")
    if ends_nl:
        lines = lines[:-1]
    return lines, ends_nl


class _FilePatch:
    __slots__ = ("path", "is_new", "hunks")

    def __init__(self, path: str, is_new: bool) -> None:
        self.path = path
        self.is_new = is_new
        self.hunks: List[Tuple[int, int, List[Tuple[str, str]]]] = []
        # hunk: (old_cursor, old_count, [(op, text), ...])  op in " -+\\"


def _parse(diff_text: str) -> Tuple[List[_FilePatch], List[str]]:
    rejects: List[str] = []
    lines = diff_text.split("\n")
    # tolerate exactly one trailing empty line from a final "\n"
    if lines and lines[-1] == "":
        lines = lines[:-1]
    if not lines:
        return [], ["empty diff: no content"]

    patches: List[_FilePatch] = []
    i = 0
    n = len(lines)

    def _fail(msg: str) -> None:
        rejects.append(f"line {i + 1}: {msg}")

    current: Optional[_FilePatch] = None
    while i < n:
        line = lines[i]
        if line.startswith("--- "):
            old_path = line[4:].strip()
            if i + 1 >= n or not lines[i + 1].startswith("+++ "):
                _fail("'---' header without following '+++' header")
                return patches, rejects
            new_path = lines[i + 1][4:].strip()
            i += 2
            if new_path == "/dev/null":
                _fail(f"file deletion is not sanctioned: {old_path}")
                return patches, rejects
            is_new = old_path == "/dev/null"
            norm_old = _strip_prefix(old_path)
            norm_new = _strip_prefix(new_path)
            if not is_new and norm_old != norm_new:
                rejects.append(
                    f"path mismatch: '--- {norm_old}' vs '+++ {norm_new}'"
                )
                return patches, rejects
            current = _FilePatch(norm_new, is_new)
            patches.append(current)
            continue
        if line.startswith("+++ "):
            _fail("'+++' header without preceding '---' header")
            return patches, rejects
        if line.startswith("@@ "):
            if current is None:
                _fail("hunk before any file header")
                return patches, rejects
            m = _HUNK_RE.match(line)
            if m is None:
                _fail(f"malformed hunk header: {line[:60]!r}")
                return patches, rejects
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            if old_start < 0 or old_count < 0 or new_start < 0 or new_count < 0:
                _fail(f"negative hunk counts: {line[:60]!r}")
                return patches, rejects
            if old_start == 0 and old_count != 0:
                _fail("old start 0 requires old count 0 (empty-file form)")
                return patches, rejects
            if new_start == 0 and new_count != 0:
                _fail("new start 0 requires new count 0 (empty-file form)")
                return patches, rejects
            i += 1
            body: List[Tuple[str, str]] = []
            seen_old = 0
            seen_new = 0
            no_newline_old = False
            no_newline_new = False
            while (seen_old < old_count or seen_new < new_count) and i < n:
                b = lines[i]
                if b.startswith("\\"):
                    # "\ No newline at end of file"
                    if not body:
                        _fail("no-newline marker at hunk start")
                        return patches, rejects
                    prev_op = body[-1][0]
                    if prev_op == " ":
                        no_newline_old = True
                        no_newline_new = True
                    elif prev_op == "-":
                        no_newline_old = True
                    elif prev_op == "+":
                        no_newline_new = True
                    else:
                        _fail("duplicate no-newline marker")
                        return patches, rejects
                    i += 1
                    continue
                if b.startswith(" "):
                    if seen_old >= old_count or seen_new >= new_count:
                        _fail("context line exceeds hunk counts")
                        return patches, rejects
                    body.append((" ", b[1:]))
                    seen_old += 1
                    seen_new += 1
                elif b.startswith("-"):
                    if seen_old >= old_count:
                        _fail("deletion line exceeds old count")
                        return patches, rejects
                    body.append(("-", b[1:]))
                    seen_old += 1
                elif b.startswith("+"):
                    if seen_new >= new_count:
                        _fail("addition line exceeds new count")
                        return patches, rejects
                    body.append(("+", b[1:]))
                    seen_new += 1
                else:
                    _fail(
                        f"unexpected line inside hunk (expected ' ', '-', '+',"
                        f" or '\\'): {b[:60]!r}"
                    )
                    return patches, rejects
                i += 1
            if seen_old != old_count or seen_new != new_count:
                _fail(
                    f"truncated hunk: header promised -{old_count}/+{new_count},"
                    f" body delivered -{seen_old}/+{seen_new}"
                )
                return patches, rejects
            # order: pure-insert hunks point AFTER line old_start
            old_cursor = old_start if old_count == 0 else old_start - 1
            if no_newline_old or no_newline_new:
                # encode both side-flags into a trailing marker entry
                body.append(
                    ("\\", f"{int(no_newline_old)}{int(no_newline_new)}")
                )
            current.hunks.append((old_cursor, old_count, body))
            continue
        if line == "":
            _fail("blank line outside any hunk")
            return patches, rejects
        if _GIT_PREFIX_RE.match(line):
            # tolerated git plumbing (between file sections)
            i += 1
            continue
        _fail(f"unrecognized diff line: {line[:60]!r}")
        return patches, rejects

    if not patches:
        rejects.append("no file sections found (missing '---'/'+++' headers)")
        return patches, rejects
    for patch in patches:
        if not patch.hunks:
            rejects.append(f"file section {patch.path} has no hunks")
    return patches, rejects


def _strip_prefix(path: str) -> str:
    if path.startswith("a/"):
        return path[2:]
    if path.startswith("b/"):
        return path[2:]
    return path


def _apply_file(
    base_text: str, patch: _FilePatch
) -> Tuple[Optional[str], List[str]]:
    errors: List[str] = []
    if patch.is_new:
        if base_text != "":
            return None, [f"new-file patch for {patch.path} but base exists"]
        base_lines: List[str] = []
        base_ends_nl = True
    else:
        base_lines, base_ends_nl = _split_keep_terminator_info(base_text)
    out: List[str] = []
    old_pos = 0
    last_hunk_end = 0
    final_no_newline = not base_ends_nl

    for old_cursor, _old_count, body in patch.hunks:
        if old_cursor < last_hunk_end:
            errors.append(
                f"hunk out of order/overlap in {patch.path}: cursor"
                f" {old_cursor} before previous hunk end {last_hunk_end}"
            )
            return None, errors
        if old_cursor > len(base_lines):
            errors.append(
                f"hunk start {old_cursor} beyond end of {patch.path}"
                f" ({len(base_lines)} lines)"
            )
            return None, errors
        # copy untouched lines up to the hunk
        out.extend(base_lines[old_pos:old_cursor])
        old_pos = old_cursor
        hunk_no_nl_new = False
        hunk_no_nl_old = False
        for idx, (op, text) in enumerate(body):
            if op == "\\":
                if idx != len(body) - 1:
                    errors.append(
                        f"no-newline marker must end the hunk in {patch.path}"
                    )
                    return None, errors
                if len(text) == 2:
                    hunk_no_nl_old = text[0] == "1"
                    hunk_no_nl_new = text[1] == "1"
                if hunk_no_nl_old and base_ends_nl:
                    errors.append(
                        f"no-newline marker contradicts {patch.path}: base"
                        " ends with a newline"
                    )
                    return None, errors
                continue
            if op == " ":
                if old_pos >= len(base_lines) or base_lines[old_pos] != text:
                    errors.append(
                        f"context mismatch in {patch.path} at base line"
                        f" {old_pos + 1}: expected {text!r}, found"
                        f" {(base_lines[old_pos] if old_pos < len(base_lines) else '<EOF>')!r}"
                    )
                    return None, errors
                out.append(text)
                old_pos += 1
            elif op == "-":
                if old_pos >= len(base_lines) or base_lines[old_pos] != text:
                    errors.append(
                        f"deletion mismatch in {patch.path} at base line"
                        f" {old_pos + 1}: expected {text!r}, found"
                        f" {(base_lines[old_pos] if old_pos < len(base_lines) else '<EOF>')!r}"
                    )
                    return None, errors
                old_pos += 1
            elif op == "+":
                out.append(text)
        last_hunk_end = old_pos
        final_no_newline = hunk_no_nl_new
    out.extend(base_lines[old_pos:])
    if not out:
        return "", errors
    text = "\n".join(out) + ("" if final_no_newline else "\n")
    return text, errors


def apply_diff(files: Dict[str, str], diff_text: str) -> DiffApplyResult:
    """Apply a unified diff to a harness file tree.  Never raises on
    malformed input; every problem lands in ``rejects`` with ``ok=False``."""
    try:
        patches, rejects = _parse(diff_text)
    except Exception as exc:  # noqa: BLE001 - parser must never escape
        return DiffApplyResult(
            ok=False,
            rejects=(f"internal parse error: {exc!r}",),
        )
    if rejects:
        return DiffApplyResult(ok=False, rejects=tuple(rejects))

    patched: Dict[str, str] = {}
    for patch in patches:
        if patch.path not in files and not patch.is_new:
            return DiffApplyResult(
                ok=False,
                rejects=(f"patch touches unknown file: {patch.path}",),
            )
        new_text, errors = _apply_file(files.get(patch.path, ""), patch)
        if errors:
            return DiffApplyResult(ok=False, rejects=tuple(errors))
        assert new_text is not None
        patched[patch.path] = new_text
    return DiffApplyResult(ok=True, patched=patched, rejects=())
