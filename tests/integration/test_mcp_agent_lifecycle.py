"""Agent-level MCP lifecycle tests using a real official-SDK stdio server."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau import Agent, AgentConfig

_SERVER_SCRIPT = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


def _windows_process_exists(pid: int) -> bool:
    """Check process liveness without relying on unsupported os.kill(pid, 0)."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    synchronize = 0x00100000
    wait_object_0 = 0
    wait_timeout = 258
    error_access_denied = 5
    error_invalid_parameter = 87

    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        if error == error_invalid_parameter:
            return False
        if error == error_access_denied:
            return True
        raise ctypes.WinError(error)  # type: ignore[attr-defined]

    try:
        wait_result = kernel32.WaitForSingleObject(handle, 0)
        if wait_result == wait_timeout:
            return True
        if wait_result == wait_object_0:
            return False
        raise ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined]
    finally:
        kernel32.CloseHandle(handle)


def _process_exists(pid: int) -> bool:
    if sys.platform == "win32":
        return _windows_process_exists(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_exit(pid: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _process_exists(pid):
            return
        time.sleep(0.02)
    raise AssertionError(f"stdio MCP child process {pid} was not cleaned up")


def _agent_config(pid_file: Path) -> AgentConfig:
    """Build through the public typed config path, not runtime internals."""
    return AgentConfig.from_dict(
        {
            "name": "mcp-lifecycle-agent",
            "system_prompt": "Use the configured MCP tools.",
            "llm_config": {
                "model": "test-model",
                "base_url": "https://example.invalid/v1",
                "api_key": "test-key",
            },
            "mcp_servers": [
                {
                    "name": "local",
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(_SERVER_SCRIPT), "stdio"],
                    "env": {
                        "MCP_TEST_PROFILE": "agent-config",
                        "MCP_TEST_PID_FILE": str(pid_file),
                    },
                }
            ],
        },
        base_path=Path.cwd(),
    )


async def _call_echo_from_child_task(agent: Agent, message: object) -> str:
    """Exercise ContextVar propagation used by concurrent executor tasks."""
    tool = agent.tool_registry["mcp__local__echo"]
    output = await asyncio.create_task(tool.execute_async(text=str(message)))
    structured = output.get("structuredContent")
    assert isinstance(structured, dict)
    assert structured["profile"] == "agent-config"
    return str(structured["text"])


def test_sync_agent_runs_twice_across_event_loops_and_cleans_stdio(tmp_path: Path) -> None:
    pid_file = tmp_path / "sync-mcp.pid"
    config = _agent_config(pid_file)

    with patch("nexau.archs.main_sub.agent.openai") as openai_module:
        openai_module.OpenAI.return_value = Mock()
        openai_module.AsyncOpenAI.return_value = AsyncMock()
        agent = Agent(config=config, openai_client=Mock())

        bootstrap_pid = int(pid_file.read_text(encoding="utf-8"))
        _wait_for_process_exit(bootstrap_pid)

        observed_pids: list[int] = []

        async def fake_run_inner(**kwargs: Any) -> str:
            pid = int(pid_file.read_text(encoding="utf-8"))
            assert _process_exists(pid)
            observed_pids.append(pid)
            return await _call_echo_from_child_task(agent, kwargs["message"])

        with patch.object(agent, "_run_async_inner", side_effect=fake_run_inner):
            assert agent.run(message="first") == "first"
            _wait_for_process_exit(observed_pids[-1])
            assert agent.run(message="second") == "second"
            _wait_for_process_exit(observed_pids[-1])

        asyncio.run(agent.aclose())

    assert len(set(observed_pids)) == 2


@pytest.mark.anyio
async def test_async_agent_create_runs_twice_and_closes_each_scope(tmp_path: Path) -> None:
    pid_file = tmp_path / "async-mcp.pid"
    config = _agent_config(pid_file)

    with patch("nexau.archs.main_sub.agent.openai") as openai_module:
        openai_module.OpenAI.return_value = Mock()
        openai_module.AsyncOpenAI.return_value = AsyncMock()
        agent = await Agent.create(config=config, openai_client=Mock())

        bootstrap_pid = int(pid_file.read_text(encoding="utf-8"))
        _wait_for_process_exit(bootstrap_pid)
        observed_pids: list[int] = []

        async def fake_run_inner(**kwargs: Any) -> str:
            pid = int(pid_file.read_text(encoding="utf-8"))
            assert _process_exists(pid)
            observed_pids.append(pid)
            return await _call_echo_from_child_task(agent, kwargs["message"])

        with patch.object(agent, "_run_async_inner", side_effect=fake_run_inner):
            assert await agent.run_async(message="first-async") == "first-async"
            _wait_for_process_exit(observed_pids[-1])
            assert await agent.run_async(message="second-async") == "second-async"
            _wait_for_process_exit(observed_pids[-1])

        await agent.aclose()

    assert len(set(observed_pids)) == 2


@pytest.mark.anyio
async def test_cancelled_agent_run_closes_official_sdk_scope_and_stdio(tmp_path: Path) -> None:
    pid_file = tmp_path / "cancelled-mcp.pid"
    config = _agent_config(pid_file)

    with patch("nexau.archs.main_sub.agent.openai") as openai_module:
        openai_module.OpenAI.return_value = Mock()
        openai_module.AsyncOpenAI.return_value = AsyncMock()
        agent = await Agent.create(config=config, openai_client=Mock())
        _wait_for_process_exit(int(pid_file.read_text(encoding="utf-8")))

        run_started = asyncio.Event()
        active_pid = 0

        async def wait_until_cancelled(**_kwargs: Any) -> str:
            nonlocal active_pid
            active_pid = int(pid_file.read_text(encoding="utf-8"))
            assert _process_exists(active_pid)
            run_started.set()
            await asyncio.Future()
            raise AssertionError("unreachable")

        with patch.object(agent, "_run_async_inner", side_effect=wait_until_cancelled):
            run_task = asyncio.create_task(agent.run_async(message="cancel me"))
            await asyncio.wait_for(run_started.wait(), timeout=5)
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

        _wait_for_process_exit(active_pid)
        await agent.aclose()
