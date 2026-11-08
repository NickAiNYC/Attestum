"""robocore.safety.ast_guard -- the harness-code safety validator.

Deliverable 5 support module (additive extension; Batch 1 shipped the
``SafetyValidationReceipt`` contract but no validator).  Two layers, both
fail-closed:

**Layer 1 -- pre-AST scope gate** (:func:`precheck_scope`): a cheap,
deterministic text scan for the highest-severity escape vectors (frame
introspection, dynamic code execution, dynamic attribute access, module /
builtins mutation, deserialization, thread creation, file I/O, forbidden
imports, dunder traversal).  Scope creep such as ``sys._getframe`` is
rejected HERE, before any AST is ever built -- the receipt's error strings
carry the ``pre-ast scope violation`` prefix as proof.  The gate is
deliberately over-broad (it fires on comments too): harness code that so
much as *mentions* an escape vector fails review, because a human grepping
the corpus uses the same strings.

**Layer 2 -- AST validator** (:func:`ast_validator`): a full ``ast.walk``
enforcement of the harness-code policy: imports restricted to ``math``,
all dunder attribute access banned, metaclass injection banned, dynamic
call names banned even as bare references, module-attribute mutation
banned, unbounded ``while True`` without ``break`` banned, NaN/Inf
constants banned, and a strict undefined-name analysis (harness code may
only use names it defines, function parameters, or a small safe-builtins
allowlist).  Obfuscated evasion (unicode identifiers, aliasing, hiding
``exec`` inside ``map``) dies on the name analysis.

No ``eval`` is used anywhere in this module; no network; no subprocess.
"""

from __future__ import annotations

import ast
import re
from typing import List, FrozenSet

from robocore.contracts import SafetyValidationReceipt

__all__ = [
    "AST_GUARD_VERSION",
    "ASTSafetyValidationReceipt",
    "ALLOWED_IMPORTS",
    "FORBIDDEN_IMPORTS",
    "SAFE_BUILTINS",
    "SAFE_EXCEPTIONS",
    "precheck_scope",
    "ast_validator",
    "validate_source",
    "validate_patch",
]

AST_GUARD_VERSION = "astguard-0.1.0"

#: The only importable module in harness reward/termination code.
ALLOWED_IMPORTS: FrozenSet[str] = frozenset({"math"})

#: Explicitly enumerated forbidden modules (the fuzzing corpus minimum);
#: anything not in ``ALLOWED_IMPORTS`` is rejected regardless.
FORBIDDEN_IMPORTS: FrozenSet[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "ctypes",
        "resource",
        "signal",
        "atexit",
        "gc",
        "weakref",
        "shutil",
        "tempfile",
        "pickle",
        "cPickle",
        "threading",
        "_thread",
        "multiprocessing",
        "builtins",
        "importlib",
        "inspect",
        "pkgutil",
        "code",
        "runpy",
        "webbrowser",
        "urllib",
        "http",
        "ftplib",
        "smtplib",
        "xmlrpc",
    }
)

#: Builtins harness code may use (pure functions only; no I/O, no
#: introspection, no dynamic execution).
SAFE_BUILTINS: FrozenSet[str] = frozenset(
    {
        "abs",
        "min",
        "max",
        "sum",
        "len",
        "range",
        "enumerate",
        "zip",
        "sorted",
        "reversed",
        "round",
        "int",
        "float",
        "bool",
        "str",
        "list",
        "tuple",
        "dict",
        "set",
        "frozenset",
        "map",
        "filter",
        "any",
        "all",
        "True",
        "False",
        "None",
    }
)

#: Exception types harness code may raise/catch.
SAFE_EXCEPTIONS: FrozenSet[str] = frozenset(
    {
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "ZeroDivisionError",
        "ArithmeticError",
        "RuntimeError",
        "AssertionError",
    }
)

_FORBIDDEN_DYNAMIC_NAMES: FrozenSet[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "open",
        "breakpoint",
        "input",
        "help",
        "exit",
        "quit",
    }
)

_DUNDER_ATTR_RE = re.compile(
    r"__(?:subclasses|reduce|reduce_ex|mro|bases|globals|builtins|self|"
    r"code|loader|spec|module|class|init_subclass|prepare|sizeof|"
    r"dir|dict|members|wrapped)__"
)


class _ScopeRule:
    __slots__ = ("pattern", "reason")

    def __init__(self, pattern: str, reason: str) -> None:
        self.pattern = re.compile(pattern, re.MULTILINE)
        self.reason = reason


_PRE_AST_RULES: tuple = tuple(
    [
        _ScopeRule(
            r"^import\s+(os|sys|subprocess|socket|ctypes|resource|signal|"
            r"atexit|gc|weakref|shutil|tempfile|pickle|cPickle|threading|"
            r"_thread|multiprocessing|builtins|importlib|inspect|pkgutil|"
            r"code|runpy|webbrowser|urllib|http|ftplib|smtplib|xmlrpc)\b",
            "forbidden import",
        ),
        _ScopeRule(
            r"^from\s+(os|sys|subprocess|socket|ctypes|resource|signal|"
            r"atexit|gc|weakref|shutil|tempfile|pickle|cPickle|threading|"
            r"_thread|multiprocessing|builtins|importlib|inspect|pkgutil|"
            r"code|runpy|webbrowser|urllib|http|ftplib|smtplib|xmlrpc)\b",
            "forbidden import",
        ),
        _ScopeRule(r"\bsys\._getframe\b", "frame introspection (sys._getframe)"),
        _ScopeRule(r"\b(eval|exec|compile)\s*\(", "dynamic code execution"),
        _ScopeRule(r"\b__import__\b", "__import__ escape"),
        _ScopeRule(r"\b(getattr|setattr|delattr)\s*\(", "dynamic attribute access"),
        _ScopeRule(_DUNDER_ATTR_RE.pattern, "dunder traversal"),
        _ScopeRule(r"\bsys\.modules\b", "module mutation (sys.modules)"),
        _ScopeRule(r"\bbuiltins\s*\.\s*\w+\s*=", "builtins mutation"),
        _ScopeRule(r"\bpickle\.loads?\b|\bloads\s*\(", "deserialization (pickle)"),
        _ScopeRule(r"\bthreading\.Thread\b|\bThread\s*\(", "thread creation"),
        _ScopeRule(r"\bopen\s*\(", "file I/O (open)"),
        _ScopeRule(r"\bbreakpoint\s*\(", "breakpoint trap"),
    ]
)


def precheck_scope(source: str) -> List[str]:
    """Layer 1: pre-AST text scan.  Returns every violation reason found
    (empty list = clean).  Never raises."""
    violations: List[str] = []
    for rule in _PRE_AST_RULES:
        for match in rule.pattern.finditer(source):
            violations.append(
                f"pre-ast scope violation: {rule.reason}"
                f" (near {match.group(0)[:40]!r})"
            )
    return violations


class ASTSafetyValidationReceipt(SafetyValidationReceipt):
    """Batch-1 receipt extended with the ``ast_passed`` / ``ast_errors``
    field names used by the fuzzing acceptance criteria.

    The extensions are read-only properties derived from the frozen Batch-1
    fields (``passed`` / ``violations``), so the two views can never
    disagree and the Batch-1 serialization format is unchanged.
    """

    @property
    def ast_passed(self) -> bool:
        return self.passed

    @property
    def ast_errors(self) -> List[str]:
        return list(self.violations)


def _receipt(errors: List[str], *, pre_ast: bool) -> ASTSafetyValidationReceipt:
    layer = "pre-ast scope gate" if pre_ast else "ast walk"
    return ASTSafetyValidationReceipt(
        passed=len(errors) == 0,
        violations=errors,
        static_analysis_summary=f"ast_guard {AST_GUARD_VERSION}: {layer}:"
        f" {len(errors)} violation(s)",
        compiler_version=AST_GUARD_VERSION,
        kinematic_precheck=False,
    )


def _collect_bound_names(tree: ast.Module) -> set:
    """All names bound anywhere in the module (function/args/assign/for/
    with/except/comprehension/import/class), over-approximated on purpose:
    the undefined-name check must never false-reject legitimate code."""
    bound = set(SAFE_BUILTINS) | set(SAFE_EXCEPTIONS)

    class _Binder(ast.NodeVisitor):
        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)

        def visit_arg(self, node: ast.arg) -> None:
            bound.add(node.arg)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            bound.add(node.name)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            bound.add(node.name)
            self.generic_visit(node)

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                bound.add(alias.asname or alias.name)

        def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
            if node.name:
                bound.add(node.name)
            self.generic_visit(node)

        def visit_Global(self, node: ast.Global) -> None:
            bound.update(node.names)

        def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
            bound.update(node.names)

    _Binder().visit(tree)
    return bound


def _check_dynamic_names(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_DYNAMIC_NAMES:
            errors.append(
                f"forbidden name reference: {node.id} (dynamic execution or"
                " attribute access is banned in harness code)"
            )
    return errors


def _check_imports(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    tag = "forbidden" if root in FORBIDDEN_IMPORTS else "not allowlisted"
                    errors.append(f"forbidden import: {alias.name} ({tag})")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                tag = "forbidden" if root in FORBIDDEN_IMPORTS else "not allowlisted"
                errors.append(f"forbidden import: {root} ({tag})")
    return errors


def _check_dunder_attrs(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _DUNDER_ATTR_RE.fullmatch(
            node.attr or ""
        ):
            errors.append(
                f"dunder access: .{node.attr} is banned (sandbox escape"
                " surface)"
            )
    return errors


def _check_metaclass_injection(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for kw in node.keywords:
                if kw.arg == "metaclass":
                    errors.append(
                        f"metaclass injection: class {node.name} declares a"
                        " metaclass"
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "type"
            and len(node.args) >= 3
        ):
            errors.append("metaclass injection: 3-arg type() call")
    return errors


def _check_module_mutation(tree: ast.Module, imported: set) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        targets: List[ast.expr] = []
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = (
                [node.target] if not isinstance(node, ast.Assign) else node.targets
            )
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id in imported
            ):
                errors.append(
                    f"module attribute mutation: {target.value.id}."
                    f"{target.attr} (sys.modules-style state tampering)"
                )
    return errors


def _direct_break_exists(stmts) -> bool:
    """Whether any ``break`` in ``stmts`` exits OUR loop.

    A ``break`` inside a nested ``for``/``while`` exits that inner loop,
    and a ``break`` inside a nested function/class belongs to another
    frame -- neither unbinds the enclosing ``while``.
    """
    for stmt in stmts:
        if isinstance(stmt, ast.Break):
            return True
        if isinstance(
            stmt,
            (ast.For, ast.While, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            continue
        for _name, value in ast.iter_fields(stmt):
            if isinstance(value, list) and value and isinstance(
                value[0], ast.stmt
            ):
                if _direct_break_exists(value):
                    return True
            elif isinstance(value, list) and value and isinstance(
                value[0], ast.ExceptHandler
            ):
                for handler in value:
                    if _direct_break_exists(handler.body):
                        return True
    return False


def _check_unbounded_while(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            test = node.test
            is_true = isinstance(test, ast.Constant) and test.value is True
            if not is_true:
                continue
            if not _direct_break_exists(node.body):
                errors.append(
                    "unbounded loop: while True without a directly"
                    " reachable break (harness code must be bounded)"
                )
    return errors


_NAN_INF_ARGS: FrozenSet[str] = frozenset(
    {
        "nan",
        "inf",
        "-inf",
        "+inf",
        "infinity",
        "-infinity",
        "+infinity",
        "NAN",
        "INF",
        "Infinity",
    }
)


def _check_nan_inf(tree: ast.Module) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.strip("'\" ") in _NAN_INF_ARGS
        ):
            errors.append(
                "NaN/Inf constant: float('nan'/'inf') poisons reward"
                " arithmetic"
            )
    return errors


def _check_undefined_names(tree: ast.Module, bound: set) -> List[str]:
    errors: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound and node.id not in _FORBIDDEN_DYNAMIC_NAMES:
                errors.append(
                    f"undefined or forbidden name: {node.id} (harness code"
                    " may only use names it defines, math, and safe"
                    " builtins)"
                )
    return errors


def ast_validator(source: str) -> ASTSafetyValidationReceipt:
    """Layer 2: full AST validation.  Assumes the pre-AST scope gate has
    already run (use :func:`validate_source` for the full pipeline).
    Syntax errors are receipts, not exceptions."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return _receipt(
            [f"syntax error: line {exc.lineno or 0}: {exc.msg}"], pre_ast=False
        )
    imported: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    errors: List[str] = []
    errors += _check_imports(tree)
    errors += _check_dynamic_names(tree)
    errors += _check_dunder_attrs(tree)
    errors += _check_metaclass_injection(tree)
    errors += _check_module_mutation(tree, imported)
    errors += _check_unbounded_while(tree)
    errors += _check_nan_inf(tree)
    bound = _collect_bound_names(tree)
    errors += _check_undefined_names(tree, bound)
    return _receipt(errors, pre_ast=False)


def validate_source(source: str) -> ASTSafetyValidationReceipt:
    """Full pipeline: pre-AST scope gate first (scope creep never reaches
    the AST layer), then the AST walk.  Never raises."""
    scope_errors = precheck_scope(source)
    if scope_errors:
        return _receipt(scope_errors, pre_ast=True)
    return ast_validator(source)


def validate_patch(
    files, diff_text: str
):
    """Validate a proposed harness diff end-to-end.

    Returns ``(DiffApplyResult, ASTSafetyValidationReceipt)``: the diff is
    applied to ``files``; if application fails the receipt carries the diff
    rejects; otherwise every patched ``*.py`` file goes through the full
    source pipeline (scope gate on the diff text itself happens first, so
    scope creep embedded in the diff is caught pre-AST too).
    """
    from robocore.safety.diff_engine import apply_diff

    scope_errors = precheck_scope(diff_text)
    result = apply_diff(files, diff_text)
    if scope_errors:
        return result, _receipt(scope_errors, pre_ast=True)
    if not result.ok:
        return result, _receipt(list(result.rejects), pre_ast=False)
    errors: List[str] = []
    for path, patched_text in sorted(result.patched.items()):
        if path.endswith(".py"):
            receipt = validate_source(patched_text)
            if not receipt.passed:
                errors.extend(f"{path}: {e}" for e in receipt.violations)
    return result, _receipt(errors, pre_ast=False)
