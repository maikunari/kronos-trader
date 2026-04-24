"""Tests for agent_infra.sandbox — the shared ASTValidator + subprocess
runner used by both strategy_agent and setup_agent. Safety-critical."""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from agent_infra.sandbox import (
    ASTValidationError,
    ASTValidator,
    SubprocessError,
    run_subprocess,
)


# ---------------------------------------------------------------------------
# ASTValidator — import allowlist
# ---------------------------------------------------------------------------

def _make_validator(**overrides) -> ASTValidator:
    defaults = dict(
        allowed_imports={"pandas", "numpy", "math"},
        forbidden_builtins={"open", "exec", "eval", "__import__"},
        required_class="Strategy",
        required_methods=("evaluate",),
    )
    defaults.update(overrides)
    return ASTValidator(**defaults)


def test_validator_accepts_clean_module():
    src = textwrap.dedent("""\
        import pandas as pd
        import numpy as np

        class Strategy:
            def __init__(self): pass
            def evaluate(self, candles): return {"action": "flat"}
    """)
    _make_validator().validate(src)   # no raise


def test_validator_rejects_forbidden_import():
    src = "import os\nclass Strategy:\n  def evaluate(self, c): pass\n"
    with pytest.raises(ASTValidationError, match="Forbidden import.*os"):
        _make_validator().validate(src)


def test_validator_rejects_forbidden_from_import():
    src = "from subprocess import call\nclass Strategy:\n  def evaluate(self, c): pass\n"
    with pytest.raises(ASTValidationError, match="Forbidden import from.*subprocess"):
        _make_validator().validate(src)


def test_validator_rejects_relative_imports_by_default():
    src = "from . import helper\nclass Strategy:\n  def evaluate(self, c): pass\n"
    with pytest.raises(ASTValidationError, match="Relative imports"):
        _make_validator().validate(src)


def test_validator_rejects_dangerous_builtin_call():
    src = textwrap.dedent("""\
        class Strategy:
            def evaluate(self, c):
                data = open("/etc/passwd").read()
                return {"action": "flat"}
    """)
    with pytest.raises(ASTValidationError, match="Forbidden builtin.*open"):
        _make_validator().validate(src)


def test_validator_rejects_missing_required_class():
    src = "class WrongName:\n  def evaluate(self, c): pass\n"
    with pytest.raises(ASTValidationError, match="must define a class named 'Strategy'"):
        _make_validator().validate(src)


def test_validator_rejects_duplicate_required_class():
    src = textwrap.dedent("""\
        class Strategy:
            def evaluate(self, c): pass
        class Strategy:   # noqa: F811
            def evaluate(self, c): pass
    """)
    with pytest.raises(ASTValidationError, match="exactly one"):
        _make_validator().validate(src)


def test_validator_rejects_missing_required_method():
    src = "class Strategy:\n  def __init__(self): pass\n"
    with pytest.raises(ASTValidationError, match="must define a 'evaluate' method"):
        _make_validator().validate(src)


def test_validator_rejects_syntax_error():
    src = "class Strategy\n  def evaluate(self, c): pass\n"   # missing colon
    with pytest.raises(ASTValidationError, match="SyntaxError"):
        _make_validator().validate(src)


def test_validator_accepts_multiple_allowed_imports():
    src = textwrap.dedent("""\
        import math
        import pandas
        from numpy import array
        class Strategy:
            def evaluate(self, c): pass
    """)
    _make_validator().validate(src)   # no raise


def test_validator_class_name_is_configurable():
    src = "class Setup:\n  def detect(self, ctx): pass\n"
    v = _make_validator(required_class="Setup", required_methods=("detect",))
    v.validate(src)   # no raise


def test_validator_method_list_is_configurable():
    src = textwrap.dedent("""\
        class Setup:
            def detect(self, ctx): pass
            def warmup(self, data): pass
    """)
    v = _make_validator(
        required_class="Setup", required_methods=("detect", "warmup"),
    )
    v.validate(src)


def test_validator_rejects_from_import_of_phase_c_forbidden_when_not_allowed():
    # Setup-agent-style config — disallow 'os'
    src = "from os import environ\nclass Setup:\n  def detect(self, c): pass\n"
    v = _make_validator(
        allowed_imports={"pandas", "setups", "support_resistance"},
        required_class="Setup", required_methods=("detect",),
    )
    with pytest.raises(ASTValidationError, match="Forbidden import from.*os"):
        v.validate(src)


# ---------------------------------------------------------------------------
# run_subprocess
# ---------------------------------------------------------------------------

def test_run_subprocess_parses_json_on_success():
    argv = [sys.executable, "-c", "import json, sys; sys.stdout.write(json.dumps({'ok': 1}))"]
    out = run_subprocess(argv, timeout=5, parse_json=True)
    assert out == {"ok": 1}


def test_run_subprocess_returns_raw_stdout_when_parse_json_false():
    argv = [sys.executable, "-c", "print('hello')"]
    out = run_subprocess(argv, timeout=5, parse_json=False)
    assert out.strip() == "hello"


def test_run_subprocess_raises_on_nonzero_exit():
    argv = [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(2)"]
    with pytest.raises(SubprocessError, match="exited 2.*boom"):
        run_subprocess(argv, timeout=5)


def test_run_subprocess_raises_on_timeout():
    argv = [sys.executable, "-c", "import time; time.sleep(10)"]
    with pytest.raises(SubprocessError, match="timed out after 1s"):
        run_subprocess(argv, timeout=1)


def test_run_subprocess_raises_on_bad_json():
    argv = [sys.executable, "-c", "print('not json')"]
    with pytest.raises(SubprocessError, match="not JSON"):
        run_subprocess(argv, timeout=5, parse_json=True)


def test_run_subprocess_raises_on_empty_stdout_when_json_expected():
    argv = [sys.executable, "-c", "pass"]
    with pytest.raises(SubprocessError, match="empty stdout"):
        run_subprocess(argv, timeout=5, parse_json=True)


def test_run_subprocess_raises_on_missing_executable():
    with pytest.raises(SubprocessError, match="exec failed"):
        run_subprocess(["/no/such/binary/anywhere"], timeout=5)
