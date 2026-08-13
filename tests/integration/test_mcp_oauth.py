"""Black-box OAuth tests against the official MCP SDK client and server.

RFC-0029: every flow uses Agent-compatible MCP server dictionaries and local
loopback services; no public server or real credential is required.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx2
import pytest
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import AuthorizationCodeResult

from nexau.archs.tool.builtin.mcp_auth import MCPAuthHost, build_http_client
from nexau.archs.tool.builtin.mcp_client import MCPRuntimeFactory

_SERVER_SCRIPT = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_ready(base_url: str, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(f"MCP fixture exited early ({process.returncode}):\n{stdout}\n{stderr}")
        try:
            response = httpx2.get(f"{base_url}/metrics", timeout=0.2)
            if response.status_code == 200:
                return
        except httpx2.HTTPError:
            pass
        time.sleep(0.05)
    raise AssertionError("MCP fixture did not become ready")


@pytest.fixture
def oauth_mcp_server(request: pytest.FixtureRequest) -> Iterator[tuple[str, subprocess.Popen[str]]]:
    fixture_mode = str(request.param)
    transport = "sse" if fixture_mode.startswith("sse-") else "streamable-http"
    mode = fixture_mode.removeprefix("sse-")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            sys.executable,
            str(_SERVER_SCRIPT),
            transport,
            "--port",
            str(port),
            "--stateless",
            "--auth",
            mode,
        ],
        cwd=Path.cwd(),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_ready(base_url, process)
        yield base_url, process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        assert process.poll() is not None


async def _call_add(server: dict[str, object], host: MCPAuthHost | None = None) -> int:
    async with build_http_client(server, auth_host=host) as http_client:
        transport = streamable_http_client(str(server["url"]), http_client=http_client)
        async with Client(transport, mode="auto") as client:
            result = await client.call_tool("add", {"left": 20, "right": 22})
            assert result.structured_content == {"result": 42}
            return int(result.structured_content["result"])


@pytest.mark.anyio
@pytest.mark.parametrize("oauth_mcp_server", ["bearer"], indirect=True)
async def test_static_bearer_from_agent_config(
    monkeypatch: pytest.MonkeyPatch,
    oauth_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, _process = oauth_mcp_server
    monkeypatch.setenv("MCP_TEST_BEARER", "static-token")
    server: dict[str, object] = {
        "name": "bearer",
        "type": "http",
        "url": f"{base_url}/mcp",
        "auth": {"type": "bearer", "token": {"source": "env", "key": "MCP_TEST_BEARER"}},
    }

    assert await _call_add(server) == 42


@pytest.mark.anyio
@pytest.mark.parametrize("oauth_mcp_server", ["headers"], indirect=True)
async def test_arbitrary_http_headers_reach_server(
    oauth_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, _process = oauth_mcp_server
    server: dict[str, object] = {
        "name": "custom-headers",
        "type": "http",
        "url": f"{base_url}/mcp",
        "headers": {"X-Test-Tenant": "north"},
    }

    assert await _call_add(server) == 42


@pytest.mark.anyio
@pytest.mark.parametrize("oauth_mcp_server", ["bearer"], indirect=True)
async def test_legacy_authorization_header_remains_compatible(
    oauth_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, _process = oauth_mcp_server
    server: dict[str, object] = {
        "name": "legacy-authorization",
        "type": "http",
        "url": f"{base_url}/mcp",
        "headers": {"Authorization": "Bearer static-token"},
    }

    assert await _call_add(server) == 42


@pytest.mark.anyio
@pytest.mark.parametrize("oauth_mcp_server", ["client-credentials"], indirect=True)
async def test_client_credentials_never_uses_interactive_callbacks(
    monkeypatch: pytest.MonkeyPatch,
    oauth_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, _process = oauth_mcp_server
    monkeypatch.setenv("MCP_TEST_CLIENT_SECRET", "service-secret")

    async def forbidden_redirect(_url: str) -> None:
        raise AssertionError("client credentials must not redirect")

    async def forbidden_callback() -> AuthorizationCodeResult:
        raise AssertionError("client credentials must not wait for a callback")

    host = MCPAuthHost(
        redirect_uri="http://127.0.0.1:1/callback",
        redirect_handler=forbidden_redirect,
        callback_handler=forbidden_callback,
    )
    server: dict[str, object] = {
        "name": "service",
        "type": "http",
        "url": f"{base_url}/mcp",
        "auth": {
            "type": "client_credentials",
            "client_id": "service-client",
            "client_secret": {"source": "env", "key": "MCP_TEST_CLIENT_SECRET"},
            "scopes": ["tools:call"],
        },
    }

    assert await _call_add(server, host) == 42
    metrics = httpx2.get(f"{base_url}/metrics").json()
    assert metrics["authorize_requests"] == 0
    assert metrics["registration_requests"] == 0
    assert metrics["token_requests"] >= 1


@pytest.mark.anyio
@pytest.mark.parametrize("oauth_mcp_server", ["sse-client-credentials"], indirect=True)
async def test_legacy_sse_supports_official_client_credentials_provider(
    monkeypatch: pytest.MonkeyPatch,
    oauth_mcp_server: tuple[str, subprocess.Popen[str]],
) -> None:
    base_url, _process = oauth_mcp_server
    monkeypatch.setenv("MCP_TEST_CLIENT_SECRET", "service-secret")
    factory = MCPRuntimeFactory(
        [
            {
                "name": "secured-sse",
                "type": "sse",
                "url": f"{base_url}/sse",
                "auth": {
                    "type": "client_credentials",
                    "client_id": "service-client",
                    "client_secret": {"source": "env", "key": "MCP_TEST_CLIENT_SECRET"},
                    "scopes": ["tools:call"],
                },
            }
        ]
    )

    async with factory.open_scope() as scope:
        tools = await scope.discover_tools()
        result = await scope.call_tool("secured-sse", "add", {"left": 20, "right": 22})

    assert [tool.name for tool in tools] == ["mcp__secured-sse__echo", "mcp__secured-sse__add"]
    assert result.structured_content == {"result": 42}
    metrics = httpx2.get(f"{base_url}/metrics").json()
    assert metrics["authorize_requests"] == 0
    assert metrics["token_requests"] >= 1


class _CallbackReceiver:
    def __init__(self) -> None:
        self.server: asyncio.AbstractServer | None = None
        self.result: asyncio.Future[AuthorizationCodeResult] | None = None
        self.redirect_uri = ""

    async def __aenter__(self) -> _CallbackReceiver:
        loop = asyncio.get_running_loop()
        self.result = loop.create_future()
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        sockets = self.server.sockets
        assert sockets
        port = int(sockets[0].getsockname()[1])
        self.redirect_uri = f"http://127.0.0.1:{port}/callback"
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        assert self.server is not None
        self.server.close()
        await self.server.wait_closed()

    async def redirect(self, authorization_url: str) -> None:
        async with httpx2.AsyncClient(follow_redirects=True) as client:
            response = await client.get(authorization_url)
            response.raise_for_status()

    async def callback(self) -> AuthorizationCodeResult:
        assert self.result is not None
        return await asyncio.wait_for(self.result, timeout=5)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request_head = await reader.readuntil(b"\r\n\r\n")
        target = request_head.split(b"\r\n", 1)[0].decode().split(" ", 2)[1]
        parsed = urlsplit(target)
        params = dict(parse_qsl(parsed.query))
        assert parsed.path == "/callback"
        assert self.result is not None
        self.result.set_result(AuthorizationCodeResult(code=params["code"], state=params.get("state"), iss=params.get("iss")))
        body = b"ok"
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n" + body)
        await writer.drain()
        writer.close()
        await writer.wait_closed()


@pytest.mark.anyio
@pytest.mark.parametrize("oauth_mcp_server", ["authorization-code"], indirect=True)
async def test_authorization_code_pkce_storage_and_refresh(oauth_mcp_server: tuple[str, subprocess.Popen[str]]) -> None:
    base_url, _process = oauth_mcp_server
    server: dict[str, object] = {
        "name": "user-oauth",
        "source_id": "local:mcp_server:user-oauth",
        "type": "http",
        "url": f"{base_url}/mcp",
        "auth": {
            "type": "authorization_code",
            "client_name": "NexAU integration test",
            "scopes": ["tools:read", "tools:call"],
        },
    }

    async with _CallbackReceiver() as receiver:
        host = MCPAuthHost(
            identity="integration-user",
            redirect_uri=receiver.redirect_uri,
            redirect_handler=receiver.redirect,
            callback_handler=receiver.callback,
        )
        assert await _call_add(server, host) == 42
        assert await _call_add(server, host) == 42

    metrics = httpx2.get(f"{base_url}/metrics").json()
    assert metrics["registration_requests"] == 1
    assert metrics["authorize_requests"] == 1
    assert metrics["refresh_requests"] >= 1
    assert metrics["token_requests"] >= 2
