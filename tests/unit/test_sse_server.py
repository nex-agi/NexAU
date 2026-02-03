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

"""Unit tests for SSE transport server.

These tests use mocks to isolate transport layer logic:
- Error handling in routes
- Event formatting
- Request validation
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nexau.archs.llm.llm_aggregators.events import TextMessageStartEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.http import HTTPConfig, SSETransportServer


class TestSSEServerErrorHandling:
    """Unit tests for SSE server error handling."""

    @pytest.fixture(scope="function")
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture(scope="function")
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
        )

    @pytest.fixture(scope="function")
    def server(self, engine, agent_config):
        """Create SSE server instance."""
        return SSETransportServer(
            engine=engine,
            config=HTTPConfig(host="localhost", port=8080),
            default_agent_config=agent_config,
        )

    def test_query_endpoint_error_handling(self, server):
        """Test /query returns 500 when handler raises."""
        with patch.object(server, "handle_request", new=AsyncMock(side_effect=Exception("Test error"))):
            client = TestClient(server.app)
            response = client.post(
                "/query",
                json={"messages": "Hello", "user_id": "test_user"},
            )
            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]

    def test_stream_endpoint_handles_error(self, server):
        """Test /stream returns error event when handler raises."""

        async def mock_error_handler(*args, **kwargs):
            yield TextMessageStartEvent(message_id="msg_1", run_id="run_1")
            raise Exception("Stream error")

        with patch.object(server, "handle_streaming_request", mock_error_handler):
            client = TestClient(server.app)
            with client.stream(
                "POST",
                "/stream",
                json={"messages": "Hello", "user_id": "test_user"},
            ) as response:
                assert response.status_code == 200
                events = [line for line in response.iter_lines() if line.startswith("data: ")]
                assert len(events) >= 1
                assert "TRANSPORT_ERROR" in events[-1] or "Stream error" in events[-1]


class TestSSEServerRequestValidation:
    """Unit tests for request validation."""

    @pytest.fixture(scope="function")
    def engine(self):
        """Create in-memory engine."""
        return InMemoryDatabaseEngine()

    @pytest.fixture(scope="function")
    def agent_config(self):
        """Create default agent config."""
        return AgentConfig(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            llm_config=LLMConfig(),
        )

    @pytest.fixture(scope="function")
    def server(self, engine, agent_config):
        """Create SSE server instance."""
        return SSETransportServer(
            engine=engine,
            config=HTTPConfig(host="localhost", port=8080),
            default_agent_config=agent_config,
        )

    def test_query_missing_messages_returns_422(self, server):
        """Test /query returns 422 for missing messages."""
        client = TestClient(server.app)
        response = client.post(
            "/query",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422

    def test_stream_missing_messages_returns_422(self, server):
        """Test /stream returns 422 for missing messages."""
        client = TestClient(server.app)
        response = client.post(
            "/stream",
            json={"user_id": "test_user"},
        )
        assert response.status_code == 422
