"""
agent_infra/sandbox.py
Shared safety plumbing for LLM-generated strategy + detector code.

  * ASTValidator — walk the AST of generated code, reject forbidden
    imports and dangerous builtins, require a specific class + method.
    Parameterised so the trend-engine agent and the Phase C setup agent
    can share the implementation with different allowlists.

  * run_subprocess — bounded-timeout subprocess runner that wraps stderr
    + nonzero-exit + JSON-parse failure into a single structured
    SubprocessError.

Kept stdlib-only so subprocesses can import without extra deps.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# AST validator
# ---------------------------------------------------------------------------

class ASTValidationError(Exception):
    """Raised when generated source fails a static safety check."""


@dataclass(frozen=True)
class ASTValidator:
    """Configurable AST safety check for generated Python modules.

    Usage:
        validator = ASTValidator(
            allowed_imports={"pandas", "numpy", "math"},
            forbidden_builtins={"open", "exec", "eval", "__import__"},
            required_class="Strategy",
            required_methods=("evaluate",),
        )
        validator.validate(source_code)

    Raises ASTValidationError on the first violation found.
    """
    allowed_imports: frozenset[str]
    forbidden_builtins: frozenset[str]
    required_class: str
    required_methods: tuple[str, ...] = ()
    allow_relative_imports: bool = False

    def __init__(
        self,
        *,
        allowed_imports: Iterable[str],
        forbidden_builtins: Iterable[str],
        required_class: str,
        required_methods: Iterable[str] = (),
        allow_relative_imports: bool = False,
    ) -> None:
        # frozen dataclass — use __setattr__ workaround
        object.__setattr__(self, "allowed_imports", frozenset(allowed_imports))
        object.__setattr__(self, "forbidden_builtins", frozenset(forbidden_builtins))
        object.__setattr__(self, "required_class", required_class)
        object.__setattr__(self, "required_methods", tuple(required_methods))
        object.__setattr__(self, "allow_relative_imports", allow_relative_imports)

    def validate(self, source: str) -> None:
        """Raise ASTValidationError on any policy violation."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise ASTValidationError(f"SyntaxError: {e}") from e

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top not in self.allowed_imports:
                        raise ASTValidationError(
                            f"Forbidden import: '{alias.name}'. "
                            f"Allowed top-level modules: "
                            f"{sorted(self.allowed_imports)}."
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                top = mod.split(".")[0]
                if not self.allow_relative_imports and node.level and node.level > 0:
                    raise ASTValidationError("Relative imports are not allowed.")
                if top and top not in self.allowed_imports:
                    raise ASTValidationError(
                        f"Forbidden import from '{mod}'. "
                        f"Allowed top-level modules: "
                        f"{sorted(self.allowed_imports)}."
                    )
            elif isinstance(node, ast.Call):
                func = node.func
                name = None
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                if name in self.forbidden_builtins:
                    raise ASTValidationError(
                        f"Forbidden builtin call: '{name}()'."
                    )

        classes = [
            n for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == self.required_class
        ]
        if not classes:
            raise ASTValidationError(
                f"Module must define a class named '{self.required_class}'."
            )
        if len(classes) > 1:
            raise ASTValidationError(
                f"Module must define exactly one '{self.required_class}' class."
            )

        if self.required_methods:
            klass = classes[0]
            methods = {n.name for n in ast.walk(klass) if isinstance(n, ast.FunctionDef)}
            for required in self.required_methods:
                if required not in methods:
                    raise ASTValidationError(
                        f"Class '{self.required_class}' must define a "
                        f"'{required}' method."
                    )


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

class SubprocessError(RuntimeError):
    """Raised when a subprocess fails (timeout, nonzero exit, bad output)."""


def run_subprocess(
    argv: list[str],
    *,
    timeout: int,
    cwd: Optional[Path] = None,
    parse_json: bool = True,
) -> dict | str:
    """Run `argv` with a bounded timeout and return parsed stdout.

    Args:
        argv: command line (must be a list, not a shell string).
        timeout: hard timeout in seconds.
        cwd: working directory for the subprocess.
        parse_json: when True, parse stdout as JSON and return the dict;
                    when False, return the raw stdout string.

    Raises:
        SubprocessError: timeout, nonzero exit, or parse failure.
    """
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except subprocess.TimeoutExpired:
        raise SubprocessError(f"subprocess timed out after {timeout}s: {argv[0]}")
    except FileNotFoundError as e:
        raise SubprocessError(f"subprocess exec failed: {e}") from e

    if proc.returncode != 0:
        raise SubprocessError(
            f"subprocess exited {proc.returncode}: "
            f"{(proc.stderr or '').strip()[:400]}"
        )

    if not parse_json:
        return proc.stdout

    stdout = proc.stdout.strip()
    if not stdout:
        raise SubprocessError("subprocess produced empty stdout")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise SubprocessError(
            f"subprocess stdout not JSON: {e} — {stdout[:200]}"
        ) from e
