# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for MCPManager and MCPClient classes.

Tests cover:
- MCPManager initialization
- Server configuration management
- Parallel server initialization
- Error handling during initialization
- Tool discovery
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexau.archs.tool.builtin.mcp_client import (
    MCPClient,
    MCPManager,
    MCPServerConfig,
    MCPTool,
)


class TestMCPServerConfig:
    """Test MCPServerConfig dataclass."""

    def test_stdio_config(self):
        """Test stdio server configuration."""
        config = MCPServerConfig(
            name="test_server",
            type="stdio",
            command="python",
            args=["-m", "mcp_server"],
            env={"KEY": "value"},
        )

        assert config.name == "test_server"
        assert config.type == "stdio"
        assert config.command == "python"
        assert config.args == ["-m", "mcp_server"]
        assert config.env == {"KEY": "value"}

    def test_http_config(self):
        """Test HTTP server configuration."""
        config = MCPServerConfig(
            name="http_server",
            type="http",
            url="http://localhost:8080",
            headers={"Authorization": "Bearer token"},
        )

        assert config.name == "http_server"
        assert config.type == "http"
        assert config.url == "http://localhost:8080"
        assert config.headers == {"Authorization": "Bearer token"}

    def test_default_values(self):
        """Test default configuration values."""
        config = MCPServerConfig(name="minimal", type="stdio")

        assert config.command is None
        assert config.args is None
        assert config.env is None
        assert config.url is None
        assert config.headers is None
        assert config.timeout == 30  # Default timeout is 30 seconds
        assert config.use_cache is False
        assert config.disable_parallel is False


class TestMCPClient:
    """Test MCPClient class."""

    def test_init(self):
        """Test MCPClient initialization."""
        client = MCPClient()

        assert client.servers == {}
        assert client.sessions == {}
        assert client.tools == {}

    def test_add_server(self):
        """Test adding server configuration."""
        client = MCPClient()
        config = MCPServerConfig(
            name="test_server",
            type="stdio",
            command="python",
        )

        client.add_server(config)

        assert "test_server" in client.servers
        assert client.servers["test_server"] == config

    def test_add_multiple_servers(self):
        """Test adding multiple server configurations."""
        client = MCPClient()

        for i in range(3):
            config = MCPServerConfig(
                name=f"server_{i}",
                type="stdio",
                command="python",
            )
            client.add_server(config)

        assert len(client.servers) == 3
        assert all(f"server_{i}" in client.servers for i in range(3))

    def test_get_tool_not_found(self):
        """Test getting non-existent tool."""
        client = MCPClient()

        result = client.get_tool("non_existent")

        assert result is None

    def test_get_all_tools_empty(self):
        """Test getting all tools when none exist."""
        client = MCPClient()

        result = client.get_all_tools()

        assert result == []


class TestMCPManager:
    """Test MCPManager class."""

    def test_init(self):
        """Test MCPManager initialization."""
        manager = MCPManager()

        assert manager.client is not None
        assert isinstance(manager.client, MCPClient)
        assert manager.auto_connect is True

    def test_add_server(self):
        """Test adding server via manager."""
        manager = MCPManager()

        manager.add_server(
            name="test_server",
            server_type="stdio",
            command="python",
            args=["-m", "server"],
        )

        assert "test_server" in manager.client.servers
        config = manager.client.servers["test_server"]
        assert config.type == "stdio"
        assert config.command == "python"
        assert config.args == ["-m", "server"]

    def test_add_server_with_all_options(self):
        """Test adding server with all configuration options."""
        manager = MCPManager()

        manager.add_server(
            name="full_server",
            server_type="http",
            url="http://localhost:8080",
            headers={"Auth": "token"},
            timeout=30.0,
            use_cache=True,
            disable_parallel=True,
        )

        config = manager.client.servers["full_server"]
        assert config.type == "http"
        assert config.url == "http://localhost:8080"
        assert config.headers == {"Auth": "token"}
        assert config.timeout == 30.0
        assert config.use_cache is True
        assert config.disable_parallel is True

    def test_get_available_tools_empty(self):
        """Test getting available tools when none exist."""
        manager = MCPManager()

        tools = manager.get_available_tools()

        assert list(tools) == []


class TestMCPManagerParallelInitialization:
    """Test MCPManager parallel server initialization."""

    @pytest.mark.anyio
    async def test_initialize_servers_empty(self):
        """Test initializing with no servers configured."""
        manager = MCPManager()

        result = await manager.initialize_servers()

        assert result == {}

    @pytest.mark.anyio
    async def test_initialize_servers_single_success(self):
        """Test initializing a single server successfully."""
        manager = MCPManager()
        manager.add_server(name="server1", server_type="stdio", command="python")

        mock_tool = MagicMock(spec=MCPTool)
        mock_tool.name = "tool1"

        with (
            patch.object(manager.client, "connect_to_server", new_callable=AsyncMock) as mock_connect,
            patch.object(manager.client, "discover_tools", new_callable=AsyncMock) as mock_discover,
        ):
            mock_connect.return_value = True
            mock_discover.return_value = [mock_tool]

            result = await manager.initialize_servers()

            assert "server1" in result
            assert len(result["server1"]) == 1
            mock_connect.assert_called_once_with("server1")
            mock_discover.assert_called_once_with("server1")

    @pytest.mark.anyio
    async def test_initialize_servers_single_connection_failure(self):
        """Test initializing when server connection fails."""
        manager = MCPManager()
        manager.add_server(name="server1", server_type="stdio", command="python")

        with patch.object(manager.client, "connect_to_server", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = False

            result = await manager.initialize_servers()

            assert "server1" in result
            assert result["server1"] == []

    @pytest.mark.anyio
    async def test_initialize_servers_parallel_execution(self):
        """Test that multiple servers are initialized in parallel."""
        manager = MCPManager()

        # Add multiple servers
        for i in range(3):
            manager.add_server(
                name=f"server_{i}",
                server_type="stdio",
                command="python",
            )

        # Track call order and timing
        call_times = []
        call_order = []

        async def mock_connect(server_name):
            call_order.append(("connect", server_name))
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.1)  # Simulate network delay
            return True

        async def mock_discover(server_name):
            call_order.append(("discover", server_name))
            mock_tool = MagicMock(spec=MCPTool)
            mock_tool.name = f"tool_{server_name}"
            return [mock_tool]

        with (
            patch.object(manager.client, "connect_to_server", side_effect=mock_connect),
            patch.object(manager.client, "discover_tools", side_effect=mock_discover),
        ):
            start_time = asyncio.get_event_loop().time()
            result = await manager.initialize_servers()
            end_time = asyncio.get_event_loop().time()

            # Verify all servers were initialized
            assert len(result) == 3
            for i in range(3):
                assert f"server_{i}" in result
                assert len(result[f"server_{i}"]) == 1

            # Verify parallel execution (total time should be ~0.1s, not ~0.3s)
            # Allow some tolerance for test execution overhead
            total_time = end_time - start_time
            assert total_time < 0.25, f"Expected parallel execution, but took {total_time}s"

    @pytest.mark.anyio
    async def test_initialize_servers_partial_failure(self):
        """Test that one server failure doesn't affect others."""
        manager = MCPManager()

        manager.add_server(name="good_server", server_type="stdio", command="python")
        manager.add_server(name="bad_server", server_type="stdio", command="python")
        manager.add_server(name="another_good", server_type="stdio", command="python")

        async def mock_connect(server_name):
            if server_name == "bad_server":
                raise Exception("Connection failed")
            return True

        async def mock_discover(server_name):
            mock_tool = MagicMock(spec=MCPTool)
            mock_tool.name = f"tool_{server_name}"
            return [mock_tool]

        with (
            patch.object(manager.client, "connect_to_server", side_effect=mock_connect),
            patch.object(manager.client, "discover_tools", side_effect=mock_discover),
        ):
            result = await manager.initialize_servers()

            # Good servers should succeed
            assert "good_server" in result
            assert len(result["good_server"]) == 1
            assert "another_good" in result
            assert len(result["another_good"]) == 1

            # Bad server should not be in results (exception was caught)
            assert "bad_server" not in result

    @pytest.mark.anyio
    async def test_initialize_servers_all_failures(self):
        """Test handling when all servers fail to initialize."""
        manager = MCPManager()

        for i in range(3):
            manager.add_server(
                name=f"server_{i}",
                server_type="stdio",
                command="python",
            )

        async def mock_connect(server_name):
            raise Exception(f"Failed to connect to {server_name}")

        with patch.object(manager.client, "connect_to_server", side_effect=mock_connect):
            result = await manager.initialize_servers()

            # All servers failed, result should be empty
            assert result == {}

    @pytest.mark.anyio
    async def test_initialize_servers_mixed_results(self):
        """Test with mix of success, connection failure, and exception."""
        manager = MCPManager()

        manager.add_server(name="success", server_type="stdio", command="python")
        manager.add_server(name="conn_fail", server_type="stdio", command="python")
        manager.add_server(name="exception", server_type="stdio", command="python")

        async def mock_connect(server_name):
            if server_name == "conn_fail":
                return False
            if server_name == "exception":
                raise Exception("Unexpected error")
            return True

        async def mock_discover(server_name):
            mock_tool = MagicMock(spec=MCPTool)
            mock_tool.name = f"tool_{server_name}"
            return [mock_tool]

        with (
            patch.object(manager.client, "connect_to_server", side_effect=mock_connect),
            patch.object(manager.client, "discover_tools", side_effect=mock_discover),
        ):
            result = await manager.initialize_servers()

            # Success server should have tools
            assert "success" in result
            assert len(result["success"]) == 1

            # Connection failure should return empty list
            assert "conn_fail" in result
            assert result["conn_fail"] == []

            # Exception server should not be in results
            assert "exception" not in result

    @pytest.mark.anyio
    async def test_initialize_servers_discover_failure(self):
        """Test handling when tool discovery fails after successful connection."""
        manager = MCPManager()
        manager.add_server(name="server1", server_type="stdio", command="python")

        with (
            patch.object(manager.client, "connect_to_server", new_callable=AsyncMock) as mock_connect,
            patch.object(manager.client, "discover_tools", new_callable=AsyncMock) as mock_discover,
        ):
            mock_connect.return_value = True
            mock_discover.side_effect = Exception("Discovery failed")

            result = await manager.initialize_servers()

            # Server with discovery failure should not be in results
            assert "server1" not in result


class TestMCPManagerShutdown:
    """Test MCPManager shutdown functionality."""

    @pytest.mark.anyio
    async def test_shutdown(self):
        """Test manager shutdown disconnects all servers."""
        manager = MCPManager()

        with patch.object(manager.client, "disconnect_all", new_callable=AsyncMock) as mock_disconnect:
            await manager.shutdown()

            mock_disconnect.assert_called_once()
