"""
Sandbox helpers for gemini_cli fixture tools.

These fixture tool implementations must not touch the host OS filesystem or spawn
local subprocesses directly. Instead, they should route all filesystem and shell
operations through NexAU's `BaseSandbox` abstraction (e.g. LocalSandbox/E2B).
"""

from __future__ import annotations

from pathlib import Path

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.sandbox import BaseSandbox
from nexau.archs.sandbox.base_sandbox import SandboxError


def get_sandbox(agent_state: AgentState | None) -> BaseSandbox:
    """Get sandbox from `agent_state`, or fall back to `LocalSandbox`."""

    if agent_state is not None:
        sandbox = agent_state.get_sandbox()
        if sandbox is not None:
            return sandbox

    # Fallback for unit tests / direct invocation of fixture functions.
    raise SandboxError("Sandbox not found")


def resolve_path(path: str, sandbox: BaseSandbox) -> str:
    """
    Resolve relative paths against sandbox work_dir.

    Note: This is *string* path resolution only; no filesystem checks here.
    """

    if Path(path).is_absolute():
        return path
    return str(Path(str(sandbox.work_dir)) / path)
