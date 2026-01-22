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

"""Unit tests for SSE transport server."""

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.http import HTTPConfig, SSETransportServer


class TestSSETransportServer:
    """Test cases for SSETransportServer."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def http_config(self):
        """Create HTTP config."""
        return HTTPConfig(
            host="localhost",
            port=8080,
            log_level="info",
            cors_origins=["*"],
        )

    @pytest.fixture
    def server(self, engine, agent_config, http_config):
        """Create SSE server instance."""
        return SSETransportServer(
            engine=engine,
            config=http_config,
            default_agent_config=agent_config,
        )

    def test_initialization(self, server):
        """Test server initialization."""
        assert server.host == "localhost"
        assert server.port == 8080
        assert server.app is not None

    def test_health_url(self, server):
        """Test health URL property."""
        assert server.health_url == "http://localhost:8080/health"

    def test_info_url(self, server):
        """Test info URL property."""
        assert server.info_url == "http://localhost:8080/"

    def test_is_running_initially_false(self, server):
        """Test is_running is initially False."""
        assert server.is_running is False

    def test_start_sets_running(self, server):
        """Test start sets is_running to True."""
        server.start()
        assert server.is_running is True

    def test_stop_clears_running(self, server):
        """Test stop sets is_running to False."""
        server.start()
        server.stop()
        assert server.is_running is False

    def test_app_has_routes(self, server):
        """Test FastAPI app has expected routes."""
        routes = [route.path for route in server.app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/stream" in routes
        assert "/query" in routes

    def test_default_config(self, agent_config):
        """Test server with default HTTP config."""
        engine = InMemoryDatabaseEngine()
        server = SSETransportServer(
            engine=engine,
            default_agent_config=agent_config,
        )
        assert server.host == "127.0.0.1"
        assert server.port == 8000


class TestHTTPConfig:
    """Test cases for HTTPConfig."""

    def test_default_config(self):
        """Test default HTTP config values."""
        config = HTTPConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 8000
        assert config.log_level == "info"
        assert config.cors_origins == ["*"]
        assert config.cors_credentials is True
        assert config.cors_methods == ["*"]
        assert config.cors_headers == ["*"]

    def test_custom_config(self):
        """Test custom HTTP config values."""
        config = HTTPConfig(
            host="localhost",
            port=9000,
            log_level="debug",
            cors_origins=["http://localhost:3000"],
        )
        assert config.host == "localhost"
        assert config.port == 9000
        assert config.log_level == "debug"
        assert config.cors_origins == ["http://localhost:3000"]


class TestSSETransportServerEndpoints:
    """Tests for SSE server FastAPI endpoints."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    @pytest.fixture
    def server(self, engine, agent_config):
        """Create SSE server instance."""
        return SSETransportServer(
            engine=engine,
            config=HTTPConfig(host="localhost", port=8080),
            default_agent_config=agent_config,
        )

    def test_root_endpoint_running(self, server):
        """Test root endpoint when server is running."""
        from fastapi.testclient import TestClient

        server.start()
        client = TestClient(server.app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert "endpoints" in data

    def test_root_endpoint_not_running(self, server):
        """Test root endpoint when server is not running."""
        from fastapi.testclient import TestClient

        # Don't call start() - server is not running
        client = TestClient(server.app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uninitialized"

    def test_health_endpoint_running(self, server):
        """Test health endpoint when server is running."""
        from fastapi.testclient import TestClient

        server.start()
        client = TestClient(server.app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_not_running(self, server):
        """Test health endpoint when server is not running."""
        from fastapi.testclient import TestClient

        client = TestClient(server.app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"


class TestSSEServerLifespan:
    """Tests for SSE server lifespan events."""

    @pytest.fixture
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(model="gpt-4o-mini"),
        )

    def test_lifespan_context(self, engine, agent_config):
        """Test lifespan context manager sets _is_running."""
        from fastapi.testclient import TestClient

        server = SSETransportServer(
            engine=engine,
            config=HTTPConfig(),
            default_agent_config=agent_config,
        )

        # Before entering context
        assert server.is_running is False

        # TestClient automatically handles lifespan
        with TestClient(server.app):
            # During lifespan, _is_running should be True
            assert server.is_running is True

        # After exiting context, _is_running should be False
        assert server.is_running is False
