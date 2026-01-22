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

"""Integration tests for SSE transport server with mocked agent."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nexau.archs.llm.llm_aggregators.events import (
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.http import HTTPConfig, SSETransportServer


class TestSSEServerQueryIntegration:
    """Integration tests for /query endpoint."""

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

    def test_query_endpoint_success_with_mocked_handler(self, server):
        """Test /query endpoint returns mocked response."""
        with patch.object(server, "handle_request", new=AsyncMock(return_value="Mocked response")):
            client = TestClient(server.app)
            response = client.post(
                "/query",
                json={"messages": "Hello", "user_id": "test_user"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["response"] == "Mocked response"

    def test_query_endpoint_error_handling(self, server):
        """Test /query endpoint handles errors properly."""
        with patch.object(server, "handle_request", new=AsyncMock(side_effect=Exception("Test error"))):
            client = TestClient(server.app)
            response = client.post(
                "/query",
                json={"messages": "Hello", "user_id": "test_user"},
            )

            assert response.status_code == 500
            assert "Test error" in response.json()["detail"]

    def test_query_endpoint_with_session_id(self, server):
        """Test /query endpoint with session_id parameter."""
        with patch.object(server, "handle_request", new=AsyncMock(return_value="Session response")) as mock:
            client = TestClient(server.app)
            response = client.post(
                "/query",
                json={
                    "messages": "Hello",
                    "user_id": "test_user",
                    "session_id": "sess_123",
                },
            )

            assert response.status_code == 200
            # Verify session_id was passed
            mock.assert_called_once()
            call_kwargs = mock.call_args[1]
            assert call_kwargs["session_id"] == "sess_123"


class TestSSEServerStreamIntegration:
    """Integration tests for /stream endpoint."""

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

    def test_stream_endpoint_yields_events(self, server):
        """Test /stream endpoint yields SSE events."""

        async def mock_streaming_handler(*args, **kwargs):
            """Mock handler that yields events."""
            yield RunStartedEvent(
                thread_id="thread_1",
                run_id="run_1",
                agent_id="agent_1",
                root_run_id="run_1",
            )
            yield TextMessageStartEvent(
                message_id="msg_1",
                run_id="run_1",
            )
            yield TextMessageContentEvent(
                message_id="msg_1",
                delta="Hello, ",
            )
            yield TextMessageContentEvent(
                message_id="msg_1",
                delta="world!",
            )
            yield TextMessageEndEvent(
                message_id="msg_1",
            )
            yield RunFinishedEvent(
                thread_id="thread_1",
                run_id="run_1",
            )

        with patch.object(server, "handle_streaming_request", mock_streaming_handler):
            client = TestClient(server.app)

            with client.stream(
                "POST",
                "/stream",
                json={"messages": "Hello", "user_id": "test_user"},
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

                events = []
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(line)

                # Should have 6 events
                assert len(events) == 6
                assert "RUN_STARTED" in events[0]
                assert "TEXT_MESSAGE_START" in events[1]
                assert "TEXT_MESSAGE_CONTENT" in events[2]
                assert "Hello, " in events[2]
                assert "TEXT_MESSAGE_CONTENT" in events[3]
                assert "world!" in events[3]
                assert "TEXT_MESSAGE_END" in events[4]
                assert "RUN_FINISHED" in events[5]

    def test_stream_endpoint_handles_error(self, server):
        """Test /stream endpoint handles errors gracefully."""

        async def mock_error_handler(*args, **kwargs):
            """Mock handler that raises an error."""
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

                events = []
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        events.append(line)

                # Should have at least the start event and an error event
                assert len(events) >= 1
                # Last event should be error
                assert "TRANSPORT_ERROR" in events[-1] or "Stream error" in events[-1]


class TestSSEServerRequestValidation:
    """Tests for request validation."""

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
