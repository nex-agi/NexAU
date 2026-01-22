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

"""Unit tests for SSE client."""

import pytest

from nexau.archs.llm.llm_aggregators.events import (
    ImageMessageContentEvent,
    ImageMessageEndEvent,
    ImageMessageStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from nexau.archs.transports.http.sse_client import SSEClient, _parse_event_dict


class TestParseEventDict:
    """Test cases for _parse_event_dict function."""

    def test_parse_text_message_start(self):
        """Test parsing TEXT_MESSAGE_START event."""
        event_data = {"type": "TEXT_MESSAGE_START", "message_id": "msg_123", "run_id": "run_123"}
        event = _parse_event_dict(event_data)
        assert isinstance(event, TextMessageStartEvent)
        assert event.message_id == "msg_123"

    def test_parse_text_message_content(self):
        """Test parsing TEXT_MESSAGE_CONTENT event."""
        event_data = {"type": "TEXT_MESSAGE_CONTENT", "message_id": "msg_123", "delta": "Hello"}
        event = _parse_event_dict(event_data)
        assert isinstance(event, TextMessageContentEvent)
        assert event.delta == "Hello"

    def test_parse_text_message_end(self):
        """Test parsing TEXT_MESSAGE_END event."""
        event_data = {"type": "TEXT_MESSAGE_END", "message_id": "msg_123"}
        event = _parse_event_dict(event_data)
        assert isinstance(event, TextMessageEndEvent)

    def test_parse_tool_call_start(self):
        """Test parsing TOOL_CALL_START event."""
        event_data = {"type": "TOOL_CALL_START", "tool_call_id": "tc_123", "tool_call_name": "search"}
        event = _parse_event_dict(event_data)
        assert isinstance(event, ToolCallStartEvent)
        assert event.tool_call_name == "search"

    def test_parse_tool_call_args(self):
        """Test parsing TOOL_CALL_ARGS event."""
        event_data = {"type": "TOOL_CALL_ARGS", "tool_call_id": "tc_123", "delta": '{"query": "test"}'}
        event = _parse_event_dict(event_data)
        assert isinstance(event, ToolCallArgsEvent)

    def test_parse_tool_call_end(self):
        """Test parsing TOOL_CALL_END event."""
        event_data = {"type": "TOOL_CALL_END", "tool_call_id": "tc_123"}
        event = _parse_event_dict(event_data)
        assert isinstance(event, ToolCallEndEvent)

    def test_parse_tool_call_result(self):
        """Test parsing TOOL_CALL_RESULT event."""
        event_data = {"type": "TOOL_CALL_RESULT", "tool_call_id": "tc_123", "content": "success"}
        event = _parse_event_dict(event_data)
        assert isinstance(event, ToolCallResultEvent)
        assert event.content == "success"

    def test_parse_run_started(self):
        """Test parsing RUN_STARTED event."""
        event_data = {
            "type": "RUN_STARTED",
            "thread_id": "thread_123",
            "run_id": "run_123",
            "agent_id": "agent_123",
            "root_run_id": "root_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, RunStartedEvent)
        assert event.thread_id == "thread_123"
        assert event.agent_id == "agent_123"

    def test_parse_run_finished(self):
        """Test parsing RUN_FINISHED event."""
        event_data = {
            "type": "RUN_FINISHED",
            "thread_id": "thread_123",
            "run_id": "run_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, RunFinishedEvent)

    def test_parse_run_error(self):
        """Test parsing RUN_ERROR event."""
        event_data = {
            "type": "RUN_ERROR",
            "message": "Something went wrong",
            "timestamp": 1234567890,
            "run_id": "run_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, RunErrorEvent)
        assert event.message == "Something went wrong"

    def test_parse_unknown_event_raises(self):
        """Test parsing unknown event type raises ValueError."""
        event_data = {"type": "UNKNOWN_EVENT"}
        with pytest.raises(ValueError, match="Unknown event type"):
            _parse_event_dict(event_data)


class TestSSEClient:
    """Test cases for SSEClient."""

    def test_initialization_default(self):
        """Test client initialization with defaults."""
        client = SSEClient()
        assert client.base_url == "http://127.0.0.1:8000"
        assert client._client is None

    def test_initialization_custom_url(self):
        """Test client initialization with custom URL."""
        client = SSEClient(base_url="http://localhost:9000")
        assert client.base_url == "http://localhost:9000"

    def test_initialization_strips_trailing_slash(self):
        """Test client strips trailing slash from URL."""
        client = SSEClient(base_url="http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_parse_thinking_text_message_start(self):
        """Test parsing THINKING_TEXT_MESSAGE_START event."""
        event_data = {
            "type": "THINKING_TEXT_MESSAGE_START",
            "parent_message_id": "msg_123",
            "thinking_message_id": "think_123",
            "run_id": "run_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, ThinkingTextMessageStartEvent)
        assert event.thinking_message_id == "think_123"

    def test_parse_thinking_text_message_content(self):
        """Test parsing THINKING_TEXT_MESSAGE_CONTENT event."""
        event_data = {
            "type": "THINKING_TEXT_MESSAGE_CONTENT",
            "thinking_message_id": "think_123",
            "delta": "Thinking...",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, ThinkingTextMessageContentEvent)
        assert event.delta == "Thinking..."

    def test_parse_thinking_text_message_end(self):
        """Test parsing THINKING_TEXT_MESSAGE_END event."""
        event_data = {
            "type": "THINKING_TEXT_MESSAGE_END",
            "thinking_message_id": "think_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, ThinkingTextMessageEndEvent)

    def test_parse_image_message_start(self):
        """Test parsing IMAGE_MESSAGE_START event."""
        event_data = {
            "type": "IMAGE_MESSAGE_START",
            "message_id": "img_123",
            "mime_type": "image/png",
            "run_id": "run_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, ImageMessageStartEvent)
        assert event.mime_type == "image/png"

    def test_parse_image_message_content(self):
        """Test parsing IMAGE_MESSAGE_CONTENT event."""
        event_data = {
            "type": "IMAGE_MESSAGE_CONTENT",
            "message_id": "img_123",
            "delta": "base64data...",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, ImageMessageContentEvent)
        assert event.delta == "base64data..."

    def test_parse_image_message_end(self):
        """Test parsing IMAGE_MESSAGE_END event."""
        event_data = {
            "type": "IMAGE_MESSAGE_END",
            "message_id": "img_123",
        }
        event = _parse_event_dict(event_data)
        assert isinstance(event, ImageMessageEndEvent)


class TestSSEClientQuery:
    """Test cases for SSEClient.query() method."""

    def test_query_success(self):
        """Test successful query."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def run_test():
            # Mock the response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success", "response": "Hello, world!"}
            mock_response.raise_for_status = MagicMock()

            # Mock the client
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = SSEClient()
                result = await client.query(messages="Hello", user_id="test_user")

            assert result == "Hello, world!"
            mock_client.aclose.assert_called_once()

        asyncio.run(run_test())

    def test_query_with_custom_client(self):
        """Test query with custom HTTP client (not closed)."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        async def run_test():
            # Mock the response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success", "response": "Response"}
            mock_response.raise_for_status = MagicMock()

            # Create custom mock client
            custom_client = AsyncMock()
            custom_client.post = AsyncMock(return_value=mock_response)
            custom_client.aclose = AsyncMock()

            client = SSEClient(http_client=custom_client)
            result = await client.query(messages="Hello")

            assert result == "Response"
            # Custom client should NOT be closed
            custom_client.aclose.assert_not_called()

        asyncio.run(run_test())

    def test_query_error_response(self):
        """Test query with error response."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def run_test():
            # Mock error response
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "error", "error": "Something went wrong"}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = SSEClient()
                with pytest.raises(ValueError, match="Server returned error"):
                    await client.query(messages="Hello")

        asyncio.run(run_test())

    def test_query_empty_response(self):
        """Test query with None response."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def run_test():
            mock_response = MagicMock()
            mock_response.json.return_value = {"status": "success", "response": None}
            mock_response.raise_for_status = MagicMock()

            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = SSEClient()
                result = await client.query(messages="Hello")

            assert result == ""

        asyncio.run(run_test())


class TestSSEClientStreamEvents:
    """Test cases for SSEClient.stream_events() method."""

    def test_stream_events(self):
        """Test streaming events."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch

        async def run_test():
            # Create async iterator for lines
            async def mock_aiter_lines():
                yield 'data: {"type": "TEXT_MESSAGE_START", "message_id": "msg_1", "run_id": "run_1"}'
                yield 'data: {"type": "TEXT_MESSAGE_CONTENT", "message_id": "msg_1", "delta": "Hello"}'
                yield 'data: {"type": "TEXT_MESSAGE_END", "message_id": "msg_1"}'

            # Mock the response context manager
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = mock_aiter_lines

            @asynccontextmanager
            async def mock_stream(*args, **kwargs):
                yield mock_response

            mock_client = AsyncMock()
            mock_client.stream = mock_stream
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = SSEClient()
                events = []
                async for event in client.stream_events(messages="Hello"):
                    events.append(event)

            assert len(events) == 3
            assert isinstance(events[0], TextMessageStartEvent)
            assert isinstance(events[1], TextMessageContentEvent)
            assert events[1].delta == "Hello"
            assert isinstance(events[2], TextMessageEndEvent)
            mock_client.aclose.assert_called_once()

        from contextlib import asynccontextmanager

        asyncio.run(run_test())

    def test_stream_events_with_custom_client(self):
        """Test streaming events with custom HTTP client."""
        import asyncio
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock

        async def run_test():
            async def mock_aiter_lines():
                yield 'data: {"type": "RUN_FINISHED", "thread_id": "t1", "run_id": "r1"}'

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = mock_aiter_lines

            @asynccontextmanager
            async def mock_stream(*args, **kwargs):
                yield mock_response

            custom_client = AsyncMock()
            custom_client.stream = mock_stream
            custom_client.aclose = AsyncMock()

            client = SSEClient(http_client=custom_client)
            events = []
            async for event in client.stream_events(messages="Hello"):
                events.append(event)

            assert len(events) == 1
            # Custom client should NOT be closed
            custom_client.aclose.assert_not_called()

        asyncio.run(run_test())

    def test_stream_events_skips_non_data_lines(self):
        """Test that non-data lines are skipped."""
        import asyncio
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        async def run_test():
            async def mock_aiter_lines():
                yield ": comment line"
                yield "event: ping"
                yield 'data: {"type": "TEXT_MESSAGE_START", "message_id": "msg_1", "run_id": "run_1"}'
                yield ""

            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.aiter_lines = mock_aiter_lines

            @asynccontextmanager
            async def mock_stream(*args, **kwargs):
                yield mock_response

            mock_client = AsyncMock()
            mock_client.stream = mock_stream
            mock_client.aclose = AsyncMock()

            with patch("httpx.AsyncClient", return_value=mock_client):
                client = SSEClient()
                events = []
                async for event in client.stream_events(messages="Hello"):
                    events.append(event)

            # Only the data line should produce an event
            assert len(events) == 1

        asyncio.run(run_test())
