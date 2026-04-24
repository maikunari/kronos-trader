"""
agent_infra — shared plumbing for Kronos's LLM agent loops.

Currently hosts:
  * sandbox.ASTValidator  — parameterised AST validator (configurable
    import allowlist + forbidden builtins)
  * sandbox.run_subprocess — bounded-timeout subprocess runner with
    structured error wrapping

Used by:
  * strategy_agent.py  (Phase 1 trend-engine agent)
  * setup_agent.py     (Phase C setup-detector agent)

Keep this module dependency-free beyond the stdlib so the sandbox can be
imported in subprocesses without pulling in agent code.
"""
from agent_infra.sandbox import (
    ASTValidationError,
    ASTValidator,
    SubprocessError,
    run_subprocess,
)

__all__ = [
    "ASTValidationError",
    "ASTValidator",
    "SubprocessError",
    "run_subprocess",
]
