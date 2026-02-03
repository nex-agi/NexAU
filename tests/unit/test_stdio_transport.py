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

"""Unit tests for Stdio transport.

These tests use mocks to isolate transport layer logic:
- Event serialization and formatting
- Error handling
- JSON-RPC routing
"""

import asyncio
import io
from unittest.mock import AsyncMock, patch

import pytest

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
from nexau.archs.transports.stdio import StdioConfig, StdioTransport


class TestStdioTransportSyncRequest:
    """Unit tests for stdio transport sync request handling."""

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
            llm_config=LLMConfig(),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create stdio transport instance."""
        return StdioTransport(
            engine=engine,
            config=StdioConfig(),
            default_agent_config=agent_config,
        )

    def test_handle_sync_request_success(self, transport):
        """Test _handle_sync_request_model output format."""

        async def run_test():
            output = io.StringIO()
            transport._real_stdout = output

            with patch.object(transport, "handle_request", new=AsyncMock(return_value="Mocked response")):
                await transport._handle_sync_request_model(
                    request=type(
                        "AgentRequest",
                        (),
                        {
                            "messages": "Hello",
                            "user_id": "test_user",
                            "session_id": None,
                            "context": None,
                        },
                    )(),
                    rpc_id="req_1",
                )

            output_str = output.getvalue()
            assert '"result":"Mocked response"' in output_str or '"result": "Mocked response"' in output_str
            assert '"id":"req_1"' in output_str or '"id": "req_1"' in output_str

        asyncio.run(run_test())

    def test_handle_sync_request_error(self, transport):
        """Test _handle_sync_request_model error handling."""

        async def run_test():
            output = io.StringIO()
            transport._real_stdout = output

            with patch.object(transport, "handle_request", new=AsyncMock(side_effect=Exception("Test error"))):
                await transport._handle_sync_request_model(
                    request=type(
                        "AgentRequest",
                        (),
                        {
                            "messages": "Hello",
                            "user_id": "test_user",
                            "session_id": None,
                            "context": None,
                        },
                    )(),
                    rpc_id="req_1",
                )

            output_str = output.getvalue()
            assert '"error"' in output_str
            assert "Test error" in output_str
            assert '"-32000"' in output_str or "-32000" in output_str

        asyncio.run(run_test())


class TestStdioTransportStreamRequest:
    """Unit tests for stdio transport streaming."""

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
            llm_config=LLMConfig(),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create stdio transport instance."""
        return StdioTransport(
            engine=engine,
            config=StdioConfig(),
            default_agent_config=agent_config,
        )

    def test_handle_streaming_request_success(self, transport):
        """Test _handle_streaming_request_model event serialization."""

        async def mock_streaming_handler(*args, **kwargs):
            """Mock handler that yields events."""
            yield RunStartedEvent(
                thread_id="thread_1",
                run_id="run_1",
                agent_id="agent_1",
                root_run_id="run_1",
            )
            yield TextMessageStartEvent(message_id="msg_1", run_id="run_1")
            yield TextMessageContentEvent(message_id="msg_1", delta="Hello!")
            yield TextMessageEndEvent(message_id="msg_1")
            yield RunFinishedEvent(thread_id="thread_1", run_id="run_1")

        async def run_test():
            output = io.StringIO()
            transport._real_stdout = output

            with patch.object(transport, "handle_streaming_request", mock_streaming_handler):
                await transport._handle_streaming_request_model(
                    request=type(
                        "AgentRequest",
                        (),
                        {
                            "messages": "Hello",
                            "user_id": "test_user",
                            "session_id": None,
                            "context": None,
                        },
                    )(),
                    rpc_id="req_1",
                )

            output_str = output.getvalue()
            lines = output_str.strip().split("\n")

            # Should have 5 event frames + 1 success response
            assert len(lines) >= 5

            # Check events are present
            assert any("RUN_STARTED" in line for line in lines)
            assert any("TEXT_MESSAGE_START" in line for line in lines)
            assert any("TEXT_MESSAGE_CONTENT" in line for line in lines)
            assert any("Hello!" in line for line in lines)
            assert any("TEXT_MESSAGE_END" in line for line in lines)
            assert any("RUN_FINISHED" in line for line in lines)

            # Last line should be success response
            assert '"result":null' in lines[-1] or '"result": null' in lines[-1]

        asyncio.run(run_test())

    def test_handle_streaming_request_error(self, transport):
        """Test _handle_streaming_request_model error handling."""

        async def mock_error_handler(*args, **kwargs):
            """Mock handler that raises an error."""
            yield TextMessageStartEvent(message_id="msg_1", run_id="run_1")
            raise Exception("Stream error")

        async def run_test():
            output = io.StringIO()
            transport._real_stdout = output

            with patch.object(transport, "handle_streaming_request", mock_error_handler):
                await transport._handle_streaming_request_model(
                    request=type(
                        "AgentRequest",
                        (),
                        {
                            "messages": "Hello",
                            "user_id": "test_user",
                            "session_id": None,
                            "context": None,
                        },
                    )(),
                    rpc_id="req_1",
                )

            output_str = output.getvalue()

            # Should contain error response
            assert '"error"' in output_str
            assert "Stream error" in output_str

        asyncio.run(run_test())


class TestStdioTransportRouting:
    """Unit tests for _handle_line routing logic."""

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
            llm_config=LLMConfig(),
        )

    @pytest.fixture
    def transport(self, engine, agent_config):
        """Create stdio transport instance."""
        return StdioTransport(
            engine=engine,
            config=StdioConfig(),
            default_agent_config=agent_config,
        )

    def test_handle_line_agent_query_method(self, transport):
        """Test _handle_line routes to sync handler for agent.query."""

        async def run_test():
            output = io.StringIO()
            transport._real_stdout = output

            with patch.object(transport, "handle_request", new=AsyncMock(return_value="Query response")):
                await transport._handle_line(
                    '{"jsonrpc": "2.0-stream", "method": "agent.query", '
                    '"params": {"messages": "Hello", "user_id": "user123"}, "id": "req_1"}'
                )

            output_str = output.getvalue()
            assert "Query response" in output_str
            assert '"id":"req_1"' in output_str or '"id": "req_1"' in output_str

        asyncio.run(run_test())

    def test_handle_line_agent_stream_method(self, transport):
        """Test _handle_line routes to streaming handler for agent.stream."""

        async def mock_streaming_handler(*args, **kwargs):
            yield TextMessageStartEvent(message_id="msg_1", run_id="run_1")
            yield TextMessageContentEvent(message_id="msg_1", delta="Streamed!")
            yield TextMessageEndEvent(message_id="msg_1")

        async def run_test():
            output = io.StringIO()
            transport._real_stdout = output

            with patch.object(transport, "handle_streaming_request", mock_streaming_handler):
                await transport._handle_line(
                    '{"jsonrpc": "2.0-stream", "method": "agent.stream", '
                    '"params": {"messages": "Hello", "user_id": "user123"}, "id": "req_2"}'
                )

            output_str = output.getvalue()
            assert "TEXT_MESSAGE_START" in output_str
            assert "Streamed!" in output_str
            assert "TEXT_MESSAGE_END" in output_str

        asyncio.run(run_test())
