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

"""Unit tests for Agent.stop() method.

RFC-0001: Agent 中断时状态持久化

Tests cover every branch in Agent.stop():
- Happy path: stop_signal, wait, flush, persist, return
- force=True vs force=False
- _run_complete timeout (step 3)
- History flush exception (step 4)
- History flush skipped when no pending (step 4)
- Session persist exception (step 5)
- _last_context present vs absent (step 5)
- Custom timeout passthrough (step 2)
- _wait_for_execution_complete: not executing / timeout
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.execution.stop_result import StopResult
from nexau.core.messages import Message


def _make_agent():
    """Create a minimally-mocked Agent with dependencies stubbed out.

    Returns an Agent whose executor, history, session_manager, and
    _run_complete are all safe mocks so stop() can run in isolation.
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

    # Replace history with a mock that behaves like HistoryList
    history_mock = MagicMock()
    history_mock.has_pending_messages = False
    history_mock.__iter__ = Mock(return_value=iter([]))
    agent._history = history_mock

    # Executor mocks
    agent.executor.stop_signal = False
    agent.executor._shutdown_event = threading.Event()

    # _run_complete already set in __init__; ensure it's set (idle)
    agent._run_complete.set()

    # Mock _persist_session_state
    agent._persist_session_state = AsyncMock()

    return agent


# ── Happy path (force=False, default) ─────────────────────────────


class TestStopGraceful:
    """stop(force=False) completes all 5 steps and returns StopResult."""

    @pytest.mark.anyio
    async def test_sets_stop_signal_and_shutdown_event(self):
        """Step 1: stop_signal=True and shutdown_event is set."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()

        await agent.stop()

        assert agent.executor.stop_signal is True
        assert agent.executor.shutdown_event.is_set()

    @pytest.mark.anyio
    async def test_returns_stop_result_with_messages(self):
        """Return value contains history snapshot and USER_INTERRUPTED."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()

        msgs = [Message.user("hello"), Message.assistant("hi")]
        agent._history.__iter__ = Mock(return_value=iter(msgs))

        result = await agent.stop()

        assert isinstance(result, StopResult)
        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED
        assert len(result.messages) == 2

    @pytest.mark.anyio
    async def test_flushes_pending_messages(self):
        """Step 4: flush() called when has_pending_messages is True."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        agent._history.has_pending_messages = True

        await agent.stop()

        agent._history.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_skips_flush_when_no_pending(self):
        """Step 4: flush() NOT called when has_pending_messages is False."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        agent._history.has_pending_messages = False

        await agent.stop()

        agent._history.flush.assert_not_called()

    @pytest.mark.anyio
    async def test_persists_session_state_with_last_context(self):
        """Step 5: _persist_session_state called with _last_context."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        agent._last_context = {"key": "value"}

        await agent.stop()

        agent._persist_session_state.assert_awaited_once_with({"key": "value"})

    @pytest.mark.anyio
    async def test_persists_session_state_with_empty_dict_when_no_last_context(self):
        """Step 5: uses {} when _last_context attribute doesn't exist."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        if hasattr(agent, "_last_context"):
            del agent._last_context

        await agent.stop()

        agent._persist_session_state.assert_awaited_once_with({})

    @pytest.mark.anyio
    async def test_custom_timeout_passed_to_wait(self):
        """Step 2: timeout kwarg is forwarded to _wait_for_execution_complete."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()

        await agent.stop(timeout=5.0)

        agent._wait_for_execution_complete.assert_awaited_once_with(timeout=5.0)


# ── force=True path ───────────────────────────────────────────────


class TestStopForce:
    """stop(force=True) does hard cleanup but still persists state."""

    @pytest.mark.anyio
    async def test_force_calls_executor_cleanup(self):
        """force=True should call executor.cleanup() directly."""
        agent = _make_agent()
        agent.executor.cleanup = Mock()

        await agent.stop(force=True)

        agent.executor.cleanup.assert_called_once()

    @pytest.mark.anyio
    async def test_force_does_not_wait_for_execution(self):
        """force=True should NOT call _wait_for_execution_complete."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        agent.executor.cleanup = Mock()

        await agent.stop(force=True)

        agent._wait_for_execution_complete.assert_not_awaited()

    @pytest.mark.anyio
    async def test_force_still_flushes_pending(self):
        """force=True should still flush pending messages."""
        agent = _make_agent()
        agent.executor.cleanup = Mock()
        agent._history.has_pending_messages = True

        await agent.stop(force=True)

        agent._history.flush.assert_called_once()

    @pytest.mark.anyio
    async def test_force_still_persists_session_state(self):
        """force=True should still persist session state."""
        agent = _make_agent()
        agent.executor.cleanup = Mock()

        await agent.stop(force=True)

        agent._persist_session_state.assert_awaited_once()


# ── Error / timeout branches ─────────────────────────────────────


class TestStopErrorBranches:
    """stop() gracefully handles failures in steps 3-5."""

    @pytest.mark.anyio
    async def test_run_complete_timeout_does_not_raise(self):
        """Step 3: TimeoutError on _run_complete.wait() is caught."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()

        # _run_complete never gets set → asyncio.wait_for will timeout
        agent._run_complete.clear()

        # Should not raise
        result = await agent.stop()

        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED

    @pytest.mark.anyio
    async def test_flush_exception_does_not_raise(self):
        """Step 4: Exception during flush() is caught and logged."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        agent._history.has_pending_messages = True
        agent._history.flush.side_effect = RuntimeError("DB write failed")

        result = await agent.stop()

        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED

    @pytest.mark.anyio
    async def test_persist_exception_does_not_raise(self):
        """Step 5: Exception during _persist_session_state is caught."""
        agent = _make_agent()
        agent._wait_for_execution_complete = AsyncMock()
        agent._persist_session_state = AsyncMock(
            side_effect=RuntimeError("Session DB down"),
        )

        result = await agent.stop()

        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED


# ── _wait_for_execution_complete ──────────────────────────────────


class TestWaitForExecutionComplete:
    """Unit tests for _wait_for_execution_complete helper."""

    @pytest.mark.anyio
    async def test_returns_immediately_when_not_executing(self):
        """Early return when executor.is_executing is False."""
        agent = _make_agent()
        # execution_done is set → is_executing is False
        agent.executor._execution_done.set()

        # Should return instantly (no hang)
        await agent._wait_for_execution_complete(timeout=0.1)

    @pytest.mark.anyio
    async def test_waits_for_execution_done_event(self):
        """Blocks until execution_done_event is set."""
        agent = _make_agent()
        agent.executor._execution_done.clear()  # simulate running

        # Set the event after a short delay
        async def _set_later():
            await asyncio.sleep(0.05)
            agent.executor._execution_done.set()

        asyncio.get_event_loop().create_task(_set_later())

        # Should complete without timeout
        await agent._wait_for_execution_complete(timeout=2.0)

    @pytest.mark.anyio
    async def test_timeout_triggers_cleanup(self):
        """When wait times out, executor.cleanup() is called."""
        agent = _make_agent()
        agent.executor._execution_done.clear()  # simulate running
        agent.executor.cleanup = Mock()

        await agent._wait_for_execution_complete(timeout=0.1)

        agent.executor.cleanup.assert_called_once()
