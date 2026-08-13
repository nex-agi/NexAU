"""Official-SDK transport black-box tests for RFC-0029."""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from nexau.archs.tool.builtin.mcp_client import MCPRuntimeFactory

_SERVER_SCRIPT = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_port(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"MCP fixture exited early ({process.returncode}):\n{stdout}\n{stderr}")
        with socket.socket() as sock:
            sock.settimeout(0.1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError("MCP fixture did not become ready")


@pytest.fixture
def remote_mcp_server(request: pytest.FixtureRequest) -> Iterator[tuple[str, subprocess.Popen[str]]]:
    transport = str(request.param)
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(_SERVER_SCRIPT), transport, "--port", str(port)],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_port(port, process)
        yield f"http://127.0.0.1:{port}", process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        assert process.poll() is not None


async def _discover_and_call(factory: MCPRuntimeFactory, server_name: str) -> tuple[list[str], dict[str, object]]:
    async with factory.open_scope() as scope:
        tools = await scope.discover_tools()
        result = await scope.call_tool(server_name, "echo", {"text": "hello"})
        assert result.structured_content is not None
        return [tool.name for tool in tools], dict(result.structured_content)


@pytest.mark.anyio
async def test_stdio_discover_call_repeat_and_cleanup() -> None:
    server = {
        "name": "stdio",
        "type": "stdio",
        "command": sys.executable,
        "args": [str(_SERVER_SCRIPT), "stdio"],
        "env": {"MCP_TEST_PROFILE": "explicit-stdio"},
    }
    factory = MCPRuntimeFactory([server])

    first = await _discover_and_call(factory, "stdio")
    second = await _discover_and_call(factory, "stdio")

    assert first[0] == ["mcp__stdio__echo", "mcp__stdio__add"]
    assert first[1] == {"text": "hello", "profile": "explicit-stdio"}
    assert second == first


@pytest.mark.anyio
@pytest.mark.parametrize("remote_mcp_server", ["streamable-http"], indirect=True)
async def test_streamable_http_discover_call_repeat_and_cleanup(
    remote_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, process = remote_mcp_server
    factory = MCPRuntimeFactory([{"name": "http", "type": "http", "url": f"{base_url}/mcp"}])

    first = await _discover_and_call(factory, "http")
    second = await _discover_and_call(factory, "http")

    assert first[0] == ["mcp__http__echo", "mcp__http__add"]
    assert first[1] == {"text": "hello", "profile": "missing"}
    assert second == first
    assert process.poll() is None


@pytest.mark.anyio
@pytest.mark.parametrize("remote_mcp_server", ["sse"], indirect=True)
async def test_legacy_sse_discover_call_and_cleanup(
    remote_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, process = remote_mcp_server
    factory = MCPRuntimeFactory([{"name": "sse", "type": "sse", "url": f"{base_url}/sse"}])

    tools, output = await _discover_and_call(factory, "sse")

    assert tools == ["mcp__sse__echo", "mcp__sse__add"]
    assert output == {"text": "hello", "profile": "missing"}
    assert process.poll() is None


@pytest.mark.anyio
async def test_multi_server_partial_failure_and_isolation() -> None:
    good = {
        "name": "good",
        "type": "stdio",
        "command": sys.executable,
        "args": [str(_SERVER_SCRIPT), "stdio"],
    }
    bad = {
        "name": "bad",
        "type": "stdio",
        "command": str(Path.cwd() / "does-not-exist-mcp-command"),
    }

    async with MCPRuntimeFactory([bad, good]).open_scope() as scope:
        tools = await scope.discover_tools()

        assert [tool.name for tool in tools] == ["mcp__good__echo", "mcp__good__add"]
        assert set(scope.failures) == {"bad"}
        result = await scope.call_tool("good", "add", {"left": 1, "right": 2})
        assert result.structured_content == {"result": 3}


@pytest.mark.anyio
async def test_concurrent_calls_and_same_named_server_isolation() -> None:
    def factory(profile: str) -> MCPRuntimeFactory:
        return MCPRuntimeFactory(
            [
                {
                    "name": "shared-name",
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(_SERVER_SCRIPT), "stdio"],
                    "env": {"MCP_TEST_PROFILE": profile},
                }
            ]
        )

    async def exercise(runtime: MCPRuntimeFactory) -> tuple[str, list[int]]:
        async with runtime.open_scope() as scope:
            await scope.discover_tools()
            echo_result, *add_results = await asyncio.gather(
                scope.call_tool("shared-name", "echo", {"text": "isolated"}),
                *(scope.call_tool("shared-name", "add", {"left": value, "right": 1}) for value in range(8)),
            )
            assert echo_result.structured_content is not None
            values = [int(result.structured_content["result"]) for result in add_results if result.structured_content]
            return str(echo_result.structured_content["profile"]), values

    first, second = await asyncio.gather(exercise(factory("agent-a")), exercise(factory("agent-b")))

    assert first == ("agent-a", list(range(1, 9)))
    assert second == ("agent-b", list(range(1, 9)))
