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

"""Unit tests for AgentEventsMiddleware."""

import json
from unittest.mock import Mock

import pytest
from ag_ui.core.events import RunFinishedEvent, RunStartedEvent
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from nexau.archs.llm.llm_aggregators.events import RunErrorEvent, ToolCallResultEvent
from nexau.archs.main_sub.execution.hooks import (
    AfterAgentHookInput,
    AfterToolHookInput,
    BeforeAgentHookInput,
    ModelCallParams,
)
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import (
    AgentEventsMiddleware,
    is_anthropic_event,
    is_openai_responses_event,
)
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason


class TestAgentEventsMiddleware:
    """Test cases for AgentEventsMiddleware."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock agent state."""
        state = Mock()
        state.agent_id = "test_agent_123"
        state.run_id = "test_run_123"
        state.root_run_id = "test_root_run_123"
        state.parent_agent_state = None
        return state

    @pytest.fixture
    def mock_parent_agent_state(self):
        """Create a mock parent agent state."""
        parent = Mock()
        parent.agent_id = "parent_agent_456"
        parent.run_id = "parent_run_456"
        return parent

    @pytest.fixture
    def events_captured(self):
        """List to capture emitted events."""
        return []

    @pytest.fixture
    def middleware(self, events_captured):
        """Create middleware with event capture."""

        def capture_event(event):
            events_captured.append(event)

        return AgentEventsMiddleware(session_id="test_session", on_event=capture_event)

    def test_initialization(self):
        """Test middleware initialization."""
        middleware = AgentEventsMiddleware(session_id="session_123")
        assert middleware.session_id == "session_123"
        assert middleware.openai_chat_completion_aggregators == {}
        assert middleware._openai_responses_aggregator is None

    def test_before_agent_emits_run_started_event(self, middleware, mock_agent_state, events_captured):
        """Test before_agent emits RunStartedEvent."""
        hook_input = BeforeAgentHookInput(
            agent_state=mock_agent_state,
            messages=[],
        )

        result = middleware.before_agent(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, RunStartedEvent)
        assert event.thread_id == "test_session"
        assert event.run_id == "test_run_123"
        # HookResult.no_changes() returns a result with no modifications
        assert not result.has_modifications()

    def test_before_agent_with_parent_state(self, middleware, mock_agent_state, mock_parent_agent_state, events_captured):
        """Test before_agent includes parent_run_id for sub-agents."""
        mock_agent_state.parent_agent_state = mock_parent_agent_state

        hook_input = BeforeAgentHookInput(
            agent_state=mock_agent_state,
            messages=[],
        )

        middleware.before_agent(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        # parent_run_id is no longer part of RunStartedEvent in the new design
        assert isinstance(event, RunStartedEvent)

    def test_after_agent_emits_run_finished_event(self, middleware, mock_agent_state, events_captured):
        """Test after_agent emits RunFinishedEvent."""
        hook_input = AfterAgentHookInput(
            agent_state=mock_agent_state,
            messages=[],
            agent_response="Task completed successfully",
        )

        result = middleware.after_agent(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, RunFinishedEvent)
        assert event.thread_id == "test_session"
        assert event.run_id == "test_run_123"
        assert event.result == "Task completed successfully"
        assert not result.has_modifications()

    def test_after_agent_emits_run_error_event(self, middleware, mock_agent_state, events_captured):
        """Test after_agent emits RunErrorEvent when stop_reason is ERROR_OCCURRED."""
        hook_input = AfterAgentHookInput(
            agent_state=mock_agent_state,
            messages=[],
            agent_response="Task failed",
            stop_reason=AgentStopReason.ERROR_OCCURRED,
        )

        result = middleware.after_agent(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, RunErrorEvent)
        assert event.run_id == "test_run_123"
        assert event.message == "Task failed"
        assert not result.has_modifications()

    def test_after_tool_emits_tool_call_result_event(self, middleware, mock_agent_state, events_captured):
        """Test after_tool emits ToolCallResultEvent."""
        hook_input = AfterToolHookInput(
            agent_state=mock_agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output={"result": "success", "data": [1, 2, 3]},
        )

        result = middleware.after_tool(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, ToolCallResultEvent)
        assert event.tool_call_id == "call_123"
        # Content should be JSON string
        content = json.loads(event.content)
        assert content == {"result": "success", "data": [1, 2, 3]}
        assert not result.has_modifications()

    def test_stream_chunk_requires_agent_state(self, middleware):
        """Test stream_chunk raises error when agent_state is None."""
        chunk = Mock(spec=ChatCompletionChunk)
        params = ModelCallParams(
            messages=[],
            max_tokens=1000,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="openai",
            tools=None,
            api_params={},
        )

        with pytest.raises(RuntimeError, match="agent_state is required"):
            middleware.stream_chunk(chunk, params)

    def test_stream_chunk_with_chat_completion_chunk(self, middleware, mock_agent_state):
        """Test stream_chunk processes ChatCompletionChunk."""
        chunk = ChatCompletionChunk(
            id="chatcmpl_123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="Hello"),
                    finish_reason=None,
                )
            ],
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
        )
        params = ModelCallParams(
            messages=[],
            max_tokens=1000,
            force_stop_reason=None,
            agent_state=mock_agent_state,
            tool_call_mode="openai",
            tools=None,
            api_params={},
        )

        result = middleware.stream_chunk(chunk, params)

        # Should return the chunk unchanged
        assert result == chunk
        # Should create an aggregator for this chunk id
        assert "chatcmpl_123" in middleware.openai_chat_completion_aggregators

    def test_stream_chunk_reuses_aggregator_for_same_id(self, middleware, mock_agent_state):
        """Test stream_chunk reuses aggregator for chunks with same id."""
        chunk1 = ChatCompletionChunk(
            id="chatcmpl_123",
            choices=[Choice(index=0, delta=ChoiceDelta(content="Hello"), finish_reason=None)],
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
        )
        chunk2 = ChatCompletionChunk(
            id="chatcmpl_123",
            choices=[Choice(index=0, delta=ChoiceDelta(content=" World"), finish_reason=None)],
            created=1234567890,
            model="gpt-4",
            object="chat.completion.chunk",
        )
        params = ModelCallParams(
            messages=[],
            max_tokens=1000,
            force_stop_reason=None,
            agent_state=mock_agent_state,
            tool_call_mode="openai",
            tools=None,
            api_params={},
        )

        middleware.stream_chunk(chunk1, params)
        aggregator1 = middleware.openai_chat_completion_aggregators["chatcmpl_123"]

        middleware.stream_chunk(chunk2, params)
        aggregator2 = middleware.openai_chat_completion_aggregators["chatcmpl_123"]

        # Should be the same aggregator instance
        assert aggregator1 is aggregator2

    def test_openai_responses_aggregator_lazy_init(self, middleware):
        """Test openai_responses_aggregator is lazily initialized."""
        assert middleware._openai_responses_aggregator is None

        aggregator = middleware.openai_responses_aggregator(run_id="test_run")

        assert middleware._openai_responses_aggregator is not None
        assert aggregator is middleware._openai_responses_aggregator

    def test_openai_responses_aggregator_reuses_instance(self, middleware):
        """Test openai_responses_aggregator returns same instance."""
        aggregator1 = middleware.openai_responses_aggregator(run_id="run1")
        aggregator2 = middleware.openai_responses_aggregator(run_id="run2")

        assert aggregator1 is aggregator2


class TestTypeGuards:
    """Test type guard functions."""

    def test_is_anthropic_event_true(self):
        """Test is_anthropic_event returns True for anthropic events."""
        # Create a mock that looks like an anthropic event
        mock_event = Mock()
        mock_event.__class__.__module__ = "anthropic.types.message_delta_event"

        assert is_anthropic_event(mock_event) is True

    def test_is_anthropic_event_false(self):
        """Test is_anthropic_event returns False for non-anthropic events."""
        mock_event = Mock()
        mock_event.__class__.__module__ = "openai.types.chat"

        assert is_anthropic_event(mock_event) is False

    def test_is_openai_responses_event_true_responses(self):
        """Test is_openai_responses_event returns True for responses events."""
        mock_event = Mock()
        mock_event.__class__.__module__ = "openai.types.responses.response_stream_event"

        assert is_openai_responses_event(mock_event) is True

    def test_is_openai_responses_event_true_streaming(self):
        """Test is_openai_responses_event returns True for streaming lib events."""
        mock_event = Mock()
        mock_event.__class__.__module__ = "openai.lib.streaming.response"

        assert is_openai_responses_event(mock_event) is True

    def test_is_openai_responses_event_false(self):
        """Test is_openai_responses_event returns False for chat events."""
        mock_event = Mock()
        mock_event.__class__.__module__ = "openai.types.chat"

        assert is_openai_responses_event(mock_event) is False


class TestAgentEventsMiddlewareResponsesAPI:
    """Test cases for AgentEventsMiddleware with OpenAI Responses API."""

    @pytest.fixture
    def mock_agent_state(self):
        """Create a mock agent state."""
        state = Mock()
        state.agent_id = "test_agent_123"
        state.parent_agent_state = None
        return state

    @pytest.fixture
    def events_captured(self):
        """List to capture emitted events."""
        return []

    @pytest.fixture
    def middleware(self, events_captured):
        """Create middleware with event capture."""

        def capture_event(event):
            events_captured.append(event)

        return AgentEventsMiddleware(session_id="test_session", on_event=capture_event)

    def test_stream_chunk_with_response_created_event(self, middleware, mock_agent_state):
        """Test stream_chunk processes response.created event."""
        # Create a mock response.created event
        mock_event = Mock()
        mock_event.__class__.__module__ = "openai.types.responses.response_stream_event"
        mock_event.type = "response.created"
        mock_event.response = Mock()
        mock_event.response.id = "resp_123"
        mock_event.response.model = "gpt-4"
        mock_event.response.created_at = 1234567890

        params = ModelCallParams(
            messages=[],
            max_tokens=1000,
            force_stop_reason=None,
            agent_state=mock_agent_state,
            tool_call_mode="openai",
            tools=None,
            api_params={},
        )

        # First call to initialize
        result = middleware.stream_chunk(mock_event, params)

        # Should return the event unchanged
        assert result == mock_event
        # Verify aggregator was initialized
        assert middleware._openai_responses_aggregator is not None
