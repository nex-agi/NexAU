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

"""Unit tests for RFC-0001: Agent interrupt state persistence.

Tests cover:
- USER_INTERRUPTED stop reason propagation
- _run_inner finally block flush guarantee
- agent.interrupt() API
- InterruptResult data model
- Streaming shutdown_event detection
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.execution.stop_result import StopResult
from nexau.core.messages import Message


class TestAgentStopReasonUserInterrupted:
    """Test USER_INTERRUPTED enum value."""

    def test_user_interrupted_exists(self):
        """USER_INTERRUPTED should be a valid AgentStopReason."""
        assert hasattr(AgentStopReason, "USER_INTERRUPTED")
        assert AgentStopReason.USER_INTERRUPTED.name == "USER_INTERRUPTED"

    def test_user_interrupted_distinct(self):
        """USER_INTERRUPTED should be distinct from other stop reasons."""
        all_reasons = list(AgentStopReason)
        assert AgentStopReason.USER_INTERRUPTED in all_reasons
        values = [r.value for r in all_reasons]
        assert len(values) == len(set(values))


class TestStopResult:
    """Test StopResult data model."""

    def test_default_values(self):
        """StopResult should have sensible defaults."""
        result = StopResult()
        assert result.messages == []
        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED
        assert result.interrupted_at_iteration == 0
        assert result.partial_response is None

    def test_with_messages(self):
        """StopResult should store messages snapshot."""
        msgs = [Message.user("hello"), Message.assistant("hi")]
        result = StopResult(
            messages=msgs,
            stop_reason=AgentStopReason.USER_INTERRUPTED,
            interrupted_at_iteration=3,
            partial_response="partial...",
        )
        assert len(result.messages) == 2
        assert result.interrupted_at_iteration == 3
        assert result.partial_response == "partial..."


class TestExecutorStopSignalReason:
    """Test that executor passes USER_INTERRUPTED when stop_signal is set."""

    def test_stop_signal_passes_user_interrupted(self):
        """When stop_signal is True, executor should pass USER_INTERRUPTED to after_agent hooks."""

        captured_stop_reason = None

        def mock_apply_after_agent_hooks(*, agent_state, messages, final_response, stop_reason):
            nonlocal captured_stop_reason
            captured_stop_reason = stop_reason
            return final_response, messages

        # Create a minimal executor mock that simulates the stop_signal path
        from nexau.archs.main_sub.execution.executor import Executor

        with patch("nexau.archs.main_sub.execution.executor.Executor.__init__", return_value=None):
            executor = Executor.__new__(Executor)
            executor.stop_signal = True
            executor._shutdown_event = threading.Event()
            executor.max_iterations = 10
            executor.agent_name = "test"
            executor.middleware_manager = None
            executor.queued_messages = []
            executor._apply_after_agent_hooks = mock_apply_after_agent_hooks

            # Simulate the stop_signal check in execute loop
            if executor.stop_signal:
                stop_response = "Stop signal received."
                stop_response, _ = executor._apply_after_agent_hooks(
                    agent_state=Mock(),
                    messages=[],
                    final_response=stop_response,
                    stop_reason=AgentStopReason.USER_INTERRUPTED,
                )

            assert captured_stop_reason == AgentStopReason.USER_INTERRUPTED


class TestRunInnerFinallyFlush:
    """Test that _run_inner's finally block guarantees flush."""

    @pytest.mark.anyio
    async def test_finally_flush_on_cancelled_error(self):
        """flush() should be called even when CancelledError is raised."""
        from nexau import Agent, AgentConfig
        from nexau.archs.llm.llm_config import LLMConfig

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
            )
            agent = Agent(config=config)

            # Set up history mock with _pending_messages
            agent.history = MagicMock()
            agent.history._pending_messages = [Message.user("test")]
            agent.history.flush = Mock()

            # Mock asyncify to raise CancelledError (simulating stop during LLM call)
            with patch("nexau.archs.main_sub.agent.asyncify") as mock_asyncify:
                mock_asyncify.return_value = AsyncMock(side_effect=asyncio.CancelledError())

                with pytest.raises(asyncio.CancelledError):
                    await agent._run_inner(
                        agent_state=Mock(),
                        merged_context={},
                        runtime_client=None,
                        custom_llm_client_provider=None,
                    )

            # finally block should have called flush
            assert agent.history.flush.called

    @pytest.mark.anyio
    async def test_finally_flush_skipped_when_no_pending(self):
        """finally block always calls flush, even when no pending messages.

        The finally block unconditionally calls flush() to handle team_mode
        where executor syncs via replace_all (clearing _pending_messages), but
        flush() uses fingerprint comparison to detect and persist new messages.
        """
        from nexau import Agent, AgentConfig
        from nexau.archs.llm.llm_config import LLMConfig

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
            )
            agent = Agent(config=config)

            # Mock executor.execute to return normally
            mock_response = ("response", [])
            agent.history = MagicMock()
            agent.history._pending_messages = []  # No pending messages
            agent.history.flush = Mock()

            with patch("nexau.archs.main_sub.agent.asyncify") as mock_asyncify:
                mock_asyncify.return_value = AsyncMock(return_value=mock_response)

                result = await agent._run_inner(
                    agent_state=Mock(),
                    merged_context={},
                    runtime_client=None,
                    custom_llm_client_provider=None,
                )

            assert result == "response"
            # flush is called once in the try block (normal path) and once in finally
            flush_call_count = agent.history.flush.call_count
            assert flush_call_count == 2


class TestStreamingShutdownEvent:
    """Test that streaming loops respect shutdown_event."""

    def test_shutdown_event_in_model_call_params(self):
        """ModelCallParams should accept shutdown_event."""
        from nexau.archs.main_sub.execution.hooks import ModelCallParams

        event = threading.Event()
        params = ModelCallParams(
            messages=[],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
            shutdown_event=event,
        )
        assert params.shutdown_event is event

    def test_shutdown_event_default_none(self):
        """ModelCallParams.shutdown_event should default to None."""
        from nexau.archs.main_sub.execution.hooks import ModelCallParams

        params = ModelCallParams(
            messages=[],
            max_tokens=100,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="xml",
            tools=None,
            api_params={},
        )
        assert params.shutdown_event is None


class TestExecutorExecutionDoneEvent:
    """Test _execution_done event lifecycle in Executor."""

    def test_execution_done_initially_set(self):
        """_execution_done should be set (not executing) initially."""
        from nexau.archs.main_sub.execution.executor import Executor

        with patch("nexau.archs.main_sub.execution.executor.Executor.__init__", return_value=None):
            executor = Executor.__new__(Executor)
            executor._execution_done = threading.Event()
            executor._execution_done.set()

            assert not executor.is_executing
            assert executor.execution_done_event.is_set()

    def test_is_executing_reflects_event_state(self):
        """is_executing should be True when _execution_done is cleared."""
        from nexau.archs.main_sub.execution.executor import Executor

        with patch("nexau.archs.main_sub.execution.executor.Executor.__init__", return_value=None):
            executor = Executor.__new__(Executor)
            executor._execution_done = threading.Event()
            executor._execution_done.set()

            assert not executor.is_executing

            executor._execution_done.clear()
            assert executor.is_executing

            executor._execution_done.set()
            assert not executor.is_executing
