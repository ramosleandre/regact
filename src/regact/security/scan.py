"""Static (AST) scan of agent-authored Python against a policy.

A generic AST utility. A feature applies it to its agent-authored deliverable before
the module body runs, to reject code that imports a forbidden module or calls an
escape hatch -- a contract on the shape of the submission, not OS-level confinement
(that is provided by ``runtime.py``). Pure and dependency-free: parse, walk, and flag
forbidden imports and calls.
"""

from __future__ import annotations

import ast

from regact.security.policy import SecurityPolicy


def scan_source(code: str, policy: SecurityPolicy, *, where: str = "<code>") -> list[str]:
    """Return a list of human-readable violations (empty == clean)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"{where}: syntax error ({exc})"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _top(alias.name) in policy.forbidden_imports:
                    violations.append(f"{where}: forbidden import {alias.name!r}")
        elif isinstance(node, ast.ImportFrom):
            if node.module and _top(node.module) in policy.forbidden_imports:
                violations.append(f"{where}: forbidden import from {node.module!r}")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in policy.forbidden_calls:
                violations.append(f"{where}: forbidden call {name!r}")
    return violations


def scan_file(path: str, policy: SecurityPolicy) -> list[str]:
    """Scan a ``.py`` file; a missing file is not a violation (nothing to run)."""
    try:
        with open(path, encoding="utf-8") as handle:
            code = handle.read()
    except OSError:
        return []
    return scan_source(code, policy, where=path)


def _top(module: str) -> str:
    return module.split(".", 1)[0]


def _call_name(func: ast.expr) -> str:
    """Dotted name of a call target: ``eval`` or ``inspect.getsource``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        prefix = _call_name(func.value)
        return f"{prefix}.{func.attr}" if prefix else func.attr
    return ""
