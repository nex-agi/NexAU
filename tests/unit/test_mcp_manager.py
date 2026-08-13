"""Compatibility facade tests for RFC-0029 official MCP runtime."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexau.archs.tool.builtin.mcp_client import MCPClient, MCPManager, MCPServerConfig, MCPTool, get_mcp_manager


def test_programmatic_server_config_keeps_existing_fields() -> None:
    config = MCPServerConfig(
        name="remote",
        type="http",
        url="http://127.0.0.1:8123/mcp",
        headers={"X-Tenant": "north"},
        timeout=7,
        disable_parallel=True,
        auth={"type": "bearer", "token": {"source": "env", "key": "MCP_TOKEN"}},
    )

    assert config.name == "remote"
    assert config.type == "http"
    assert config.headers == {"X-Tenant": "north"}
    assert config.timeout == 7
    assert config.disable_parallel is True
    assert config.auth == {"type": "bearer", "token": {"source": "env", "key": "MCP_TOKEN"}}


def test_programmatic_server_config_repr_hides_sensitive_transport_fields() -> None:
    config = MCPServerConfig(
        name="remote",
        type="http",
        url="https://example.test/mcp?api_key=query-secret",
        headers={"Authorization": "Bearer header-secret"},
        args=["--token", "argument-secret"],
        env={"MCP_TOKEN": "environment-secret"},
        auth={"type": "bearer", "token": {"source": "env", "key": "MCP_TOKEN"}},
    )

    rendered = repr(config)
    for secret in ("query-secret", "header-secret", "argument-secret", "environment-secret", "MCP_TOKEN"):
        assert secret not in rendered


def test_deprecated_client_keeps_config_without_persistent_sessions() -> None:
    with pytest.deprecated_call(match="MCPClient is deprecated"):
        client = MCPClient()
    config = MCPServerConfig(name="stdio", type="stdio", command="python")

    client.add_server(config)

    assert client.servers == {"stdio": config}
    assert client.sessions == {}
    assert client.get_all_tools() == []


@pytest.mark.anyio
async def test_deprecated_client_bootstrap_closes_scope_and_keeps_descriptors() -> None:
    with pytest.deprecated_call(match="MCPClient is deprecated"):
        client = MCPClient()
    client.add_server(MCPServerConfig(name="stdio", type="stdio", command="python"))
    tool = MagicMock(spec=MCPTool)
    tool.name = "mcp__stdio__echo"
    scope = AsyncMock()
    scope.__aenter__.return_value = scope
    scope.discover_tools.return_value = [tool]
    scope.failures = {}
    factory = MagicMock()
    factory.open_scope.return_value = scope

    with patch("nexau.archs.tool.builtin.mcp_client.MCPRuntimeFactory", return_value=factory):
        assert await client.connect_to_server("stdio") is True

    scope.__aenter__.assert_awaited_once()
    scope.__aexit__.assert_awaited_once()
    assert client.get_all_tools() == [tool]
    assert client.sessions == {}


def test_deprecated_manager_does_not_use_process_global_state() -> None:
    with pytest.deprecated_call():
        first = get_mcp_manager()
    with pytest.deprecated_call():
        second = get_mcp_manager()

    assert isinstance(first, MCPManager)
    assert isinstance(second, MCPManager)
    assert first is not second
    assert first.client is not second.client


@pytest.mark.anyio
async def test_manager_initializes_each_config_and_shuts_down() -> None:
    with pytest.deprecated_call(match="MCPManager is deprecated"):
        manager = MCPManager()
    manager.add_server(name="one", server_type="stdio", command="python")
    manager.add_server(name="two", server_type="stdio", command="python")
    first_tool = MagicMock(spec=MCPTool)
    second_tool = MagicMock(spec=MCPTool)

    async def discover(name: str) -> list[MCPTool]:
        return [first_tool] if name == "one" else [second_tool]

    with (
        patch.object(manager.client, "connect_to_server", new=AsyncMock(return_value=True)),
        patch.object(manager.client, "discover_tools", side_effect=discover),
    ):
        initialized = await manager.initialize_servers()

    assert initialized == {"one": [first_tool], "two": [second_tool]}

    with patch.object(manager.client, "disconnect_all", new=AsyncMock()) as disconnect:
        await manager.shutdown()
        disconnect.assert_awaited_once()
