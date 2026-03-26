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

"""Coverage improvement tests for Executor class (batch 3).

Targets uncovered paths in:
- nexau/archs/main_sub/execution/executor.py

Covers:
- Lines 606-611: sync execute team_mode system-only messages with HistoryList sync + wait
- Lines 623-632: sync execute team_mode assistant-ending messages with HistoryList sync + wait
- Lines 1128-1148: async _execute_iteration_async team_mode skip paths
- Lines 1341-1353: async _handle_stop_condition_async team_mode branching
- Lines 1511-1589: async tool execution path (async tool) + sub-agent call
"""

import threading
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.executor import (
    Executor,
    _AsyncIterationState,
    _IterationOutcome,
)
from nexau.archs.main_sub.execution.hooks import HookResult, Middleware
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse, SubAgentCall, ToolCall
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.core.messages import Message, Role, TextBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_registry(tools: dict[str, Tool] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if tools:
        registry.add_source("test", list(tools.values()))
    return registry


def _make_async_tool(name: str) -> Tool:
    """Create a tool with an async implementation."""

    async def _async_impl(**kwargs: Any) -> dict[str, Any]:
        return {"result": name}

    return Tool(
        name=name,
        description=f"Async tool {name}",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        implementation=_async_impl,
    )


def _make_config() -> LLMConfig:
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )


def _make_executor(
    *,
    team_mode: bool = False,
    tools: dict[str, Tool] | None = None,
    stop_tools: set[str] | None = None,
    middlewares: list[Middleware] | None = None,
) -> Executor:
    return Executor(
        agent_name="test_agent",
        agent_id="test_id",
        tool_registry=_make_tool_registry(tools),
        sub_agents={},
        stop_tools=stop_tools or set(),
        openai_client=Mock(),
        llm_config=_make_config(),
        team_mode=team_mode,
        middlewares=middlewares,
    )


def _make_agent_state() -> AgentState:
    from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage

    return AgentState(
        agent_name="test_agent",
        agent_id="test_id",
        run_id="run_1",
        root_run_id="run_1",
        context=AgentContext({}),
        global_storage=GlobalStorage(),
        tool_registry=ToolRegistry(),
    )


def _make_framework_context() -> FrameworkContext:
    return FrameworkContext(
        agent_name="test_agent",
        agent_id="test_id",
        run_id="run_1",
        root_run_id="run_1",
        _tool_registry=ToolRegistry(),
        _shutdown_event=threading.Event(),
    )


def _system_msg(text: str = "You are a helper.") -> Message:
    return Message(role=Role.SYSTEM, content=[TextBlock(text=text)])


def _user_msg(text: str = "Hello") -> Message:
    return Message(role=Role.USER, content=[TextBlock(text=text)])


def _assistant_msg(text: str = "Sure") -> Message:
    return Message(role=Role.ASSISTANT, content=[TextBlock(text=text)])


# ---------------------------------------------------------------------------
# Lines 606-611: sync execute — team_mode, system-only messages,
# HistoryList sync + _wait_for_messages
# ---------------------------------------------------------------------------


class TestSyncExecuteTeamModeSystemOnly:
    """Cover lines 606-611: team_mode skips LLM call when only system messages exist."""

    def test_system_only_with_history_list_syncs_and_breaks(self):
        """HistoryList origin → replace_all called, wait returns False → break."""
        executor = _make_executor(team_mode=True)
        agent_state = _make_agent_state()
        history_list = HistoryList([_system_msg()])

        with patch.object(executor, "_wait_for_messages", return_value=False):
            response, messages = executor.execute(history_list, agent_state)

        assert isinstance(response, str)

    def test_system_only_with_plain_list_breaks(self):
        """Plain list origin → HistoryList sync branch skipped, wait returns False → break."""
        executor = _make_executor(team_mode=True)
        agent_state = _make_agent_state()
        history = [_system_msg()]

        with patch.object(executor, "_wait_for_messages", return_value=False):
            response, messages = executor.execute(history, agent_state)

        assert isinstance(response, str)

    def test_system_only_wait_returns_true_continues_iteration(self):
        """Wait returns True → iteration increments and loop continues.
        A user message is injected so the second iteration proceeds past the
        system-only check; call_llm returns None to break cleanly.
        """
        executor = _make_executor(team_mode=True)
        agent_state = _make_agent_state()
        history_list = HistoryList([_system_msg()])

        call_count = 0

        def _fake_wait() -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                executor.queued_messages = [_user_msg("hi")]
                return True
            return False

        with (
            patch.object(executor, "_wait_for_messages", side_effect=_fake_wait),
            patch.object(executor.llm_caller, "call_llm", return_value=None),
        ):
            response, messages = executor.execute(history_list, agent_state)

        assert call_count >= 1


# ---------------------------------------------------------------------------
# Lines 623-632: sync execute — team_mode, last non-system message is assistant
# ---------------------------------------------------------------------------


class TestSyncExecuteTeamModeAssistantEnding:
    """Cover lines 623-632: team_mode skips LLM when last non-system msg is assistant."""

    def test_assistant_ending_with_history_list_syncs_and_breaks(self):
        """HistoryList origin → replace_all called, wait returns False → break."""
        executor = _make_executor(team_mode=True)
        agent_state = _make_agent_state()
        history_list = HistoryList([_system_msg(), _assistant_msg("Hello")])

        with patch.object(executor, "_wait_for_messages", return_value=False):
            response, messages = executor.execute(history_list, agent_state)

        assert isinstance(response, str)

    def test_assistant_ending_with_plain_list_breaks(self):
        """Plain list origin → HistoryList sync skipped, wait returns False → break."""
        executor = _make_executor(team_mode=True)
        agent_state = _make_agent_state()
        history = [_system_msg(), _assistant_msg("Hi")]

        with patch.object(executor, "_wait_for_messages", return_value=False):
            response, messages = executor.execute(history, agent_state)

        assert isinstance(response, str)

    def test_assistant_ending_wait_returns_true_continues(self):
        """Wait returns True → iteration increments and loop continues.
        A user message is injected to advance past assistant-ending check.
        call_llm returns None to break cleanly.
        """
        executor = _make_executor(team_mode=True)
        agent_state = _make_agent_state()
        history_list = HistoryList([_system_msg(), _assistant_msg("Hi")])

        call_count = 0

        def _fake_wait() -> bool:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                executor.queued_messages = [_user_msg("continue")]
                return True
            return False

        with (
            patch.object(executor, "_wait_for_messages", side_effect=_fake_wait),
            patch.object(executor.llm_caller, "call_llm", return_value=None),
        ):
            response, messages = executor.execute(history_list, agent_state)

        assert call_count >= 1


# ---------------------------------------------------------------------------
# Lines 1128-1148: async _execute_iteration_async — team_mode skip paths
# ---------------------------------------------------------------------------


class TestAsyncIterationTeamModeSkips:
    """Cover lines 1128-1148 in _execute_iteration_async."""

    def _make_state(
        self,
        msgs: list[Message],
        origin: list[Message] | HistoryList,
        agent_state: AgentState,
    ) -> _AsyncIterationState:
        return _AsyncIterationState(
            messages=list(msgs),
            final_response="",
            force_stop_reason=AgentStopReason.SUCCESS,
            iteration=0,
            agent_state=agent_state,
            token_trace_session=None,
            framework_context=_make_framework_context(),
            runtime_client=None,
            custom_llm_client_provider=None,
            origin_history=origin,
        )

    @pytest.mark.anyio
    async def test_system_only_with_history_list_break(self):
        """team_mode + system-only → sync HistoryList → wait returns False → BREAK."""
        executor = _make_executor(team_mode=True)
        history_list = HistoryList([_system_msg()])
        state = self._make_state([_system_msg()], history_list, _make_agent_state())

        with patch.object(executor, "_wait_for_messages", return_value=False):
            result = await executor._execute_iteration_async(state)

        assert result == _IterationOutcome.BREAK

    @pytest.mark.anyio
    async def test_system_only_with_history_list_continue(self):
        """team_mode + system-only → wait returns True → CONTINUE, iteration incremented."""
        executor = _make_executor(team_mode=True)
        history_list = HistoryList([_system_msg()])
        state = self._make_state([_system_msg()], history_list, _make_agent_state())

        with patch.object(executor, "_wait_for_messages", return_value=True):
            result = await executor._execute_iteration_async(state)

        assert result == _IterationOutcome.CONTINUE
        assert state.iteration == 1

    @pytest.mark.anyio
    async def test_system_only_plain_list_break(self):
        """team_mode + system-only + plain list origin → HistoryList branch skipped → BREAK."""
        executor = _make_executor(team_mode=True)
        state = self._make_state([_system_msg()], [_system_msg()], _make_agent_state())

        with patch.object(executor, "_wait_for_messages", return_value=False):
            result = await executor._execute_iteration_async(state)

        assert result == _IterationOutcome.BREAK

    @pytest.mark.anyio
    async def test_assistant_ending_with_history_list_break(self):
        """team_mode + assistant ending + HistoryList → wait returns False → BREAK."""
        executor = _make_executor(team_mode=True)
        msgs = [_system_msg(), _assistant_msg()]
        history_list = HistoryList(msgs)
        state = self._make_state(list(msgs), history_list, _make_agent_state())

        with patch.object(executor, "_wait_for_messages", return_value=False):
            result = await executor._execute_iteration_async(state)

        assert result == _IterationOutcome.BREAK

    @pytest.mark.anyio
    async def test_assistant_ending_with_history_list_continue(self):
        """team_mode + assistant ending + HistoryList → wait returns True → CONTINUE."""
        executor = _make_executor(team_mode=True)
        msgs = [_system_msg(), _assistant_msg()]
        history_list = HistoryList(msgs)
        state = self._make_state(list(msgs), history_list, _make_agent_state())

        with patch.object(executor, "_wait_for_messages", return_value=True):
            result = await executor._execute_iteration_async(state)

        assert result == _IterationOutcome.CONTINUE
        assert state.iteration == 1

    @pytest.mark.anyio
    async def test_assistant_ending_plain_list_break(self):
        """team_mode + assistant ending + plain list origin → HistoryList branch skipped."""
        executor = _make_executor(team_mode=True)
        msgs = [_system_msg(), _assistant_msg()]
        state = self._make_state(list(msgs), msgs, _make_agent_state())

        with patch.object(executor, "_wait_for_messages", return_value=False):
            result = await executor._execute_iteration_async(state)

        assert result == _IterationOutcome.BREAK


# ---------------------------------------------------------------------------
# Lines 1341-1353: async _handle_stop_condition_async — team_mode branching
# ---------------------------------------------------------------------------


class TestHandleStopConditionAsync:
    """Cover lines 1341-1353 in _handle_stop_condition_async."""

    def _make_state(
        self,
        origin: list[Message] | HistoryList | None = None,
        iteration: int = 0,
    ) -> _AsyncIterationState:
        return _AsyncIterationState(
            messages=[_system_msg(), _user_msg()],
            final_response="",
            force_stop_reason=AgentStopReason.SUCCESS,
            iteration=iteration,
            agent_state=_make_agent_state(),
            token_trace_session=None,
            framework_context=_make_framework_context(),
            runtime_client=None,
            custom_llm_client_provider=None,
            origin_history=origin or [],
        )

    @pytest.mark.anyio
    async def test_team_mode_finish_team_breaks(self):
        """team_mode + finish_team stop tool → BREAK with STOP_TOOL_TRIGGERED."""
        executor = _make_executor(team_mode=True)
        state = self._make_state()
        executor._last_stop_tool_name = "finish_team"

        result = await executor._handle_stop_condition_async(
            state,
            stop_tool_result="team finished",
            processed_response="response text",
        )

        assert result == _IterationOutcome.BREAK
        assert state.force_stop_reason == AgentStopReason.STOP_TOOL_TRIGGERED
        assert state.final_response == "team finished"

    @pytest.mark.anyio
    async def test_team_mode_non_finish_team_waits_and_continues(self):
        """team_mode + non-finish_team stop tool → wait → CONTINUE."""
        executor = _make_executor(team_mode=True)
        history_list = HistoryList([_system_msg(), _user_msg()])
        state = self._make_state(origin=history_list, iteration=5)
        executor._last_stop_tool_name = "ask_user"

        with patch.object(executor, "_wait_for_messages", return_value=True):
            result = await executor._handle_stop_condition_async(
                state,
                stop_tool_result="please provide input",
                processed_response="response text",
            )

        assert result == _IterationOutcome.CONTINUE
        assert state.iteration == 6

    @pytest.mark.anyio
    async def test_team_mode_non_finish_team_wait_fails_breaks(self):
        """team_mode + non-finish_team stop tool → wait returns False → BREAK."""
        executor = _make_executor(team_mode=True)
        history_list = HistoryList([_system_msg(), _user_msg()])
        state = self._make_state(origin=history_list, iteration=3)
        executor._last_stop_tool_name = "ask_user"

        with patch.object(executor, "_wait_for_messages", return_value=False):
            result = await executor._handle_stop_condition_async(
                state,
                stop_tool_result="please provide input",
                processed_response="response text",
            )

        assert result == _IterationOutcome.BREAK
        assert state.force_stop_reason == AgentStopReason.NO_MORE_TOOL_CALLS
        assert state.final_response == "response text"

    @pytest.mark.anyio
    async def test_team_mode_no_stop_tool_result_waits(self):
        """team_mode + stop_tool_result=None → still waits for messages."""
        executor = _make_executor(team_mode=True)
        state = self._make_state()
        executor._last_stop_tool_name = "some_other"

        with patch.object(executor, "_wait_for_messages", return_value=True):
            result = await executor._handle_stop_condition_async(
                state,
                stop_tool_result=None,
                processed_response="response text",
            )

        assert result == _IterationOutcome.CONTINUE
        assert state.iteration == 1

    @pytest.mark.anyio
    async def test_team_mode_history_list_replaced_before_wait(self):
        """team_mode: HistoryList.replace_all is called before blocking wait."""
        executor = _make_executor(team_mode=True)
        history_list = HistoryList([_system_msg()])
        state = self._make_state(origin=history_list)
        executor._last_stop_tool_name = "other_tool"

        with (
            patch.object(executor, "_wait_for_messages", return_value=True),
            patch.object(history_list, "replace_all") as mock_replace,
        ):
            result = await executor._handle_stop_condition_async(
                state,
                stop_tool_result=None,
                processed_response="resp",
            )

        mock_replace.assert_called_once_with(state.messages)
        assert result == _IterationOutcome.CONTINUE


# ---------------------------------------------------------------------------
# Lines 1511-1589: async tool execution path (async tool) + sub-agent call
# ---------------------------------------------------------------------------


class TestExecuteParsedCallsAsyncTool:
    """Cover lines 1511-1573 and 1575-1589 in _execute_parsed_calls_async."""

    @pytest.mark.anyio
    async def test_async_tool_execution_success(self):
        """Async tool executes via execute_async path (lines 1511-1571)."""
        async_tool = _make_async_tool("my_async_tool")
        executor = _make_executor(tools={"my_async_tool": async_tool})
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="my_async_tool", parameters={"x": "hello"}, tool_call_id="tc_1")
        parsed = ParsedResponse(
            original_response="calling async tool",
            tool_calls=[tc],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        _, _, _, feedbacks = result
        assert len(feedbacks) >= 1
        assert feedbacks[0]["call_type"] == "tool"
        assert feedbacks[0].get("is_error") is not True

    @pytest.mark.anyio
    async def test_async_tool_execution_outer_error(self):
        """Error outside execute_async (e.g. get_sandbox) → caught at lines 1572-1573."""
        async_tool = _make_async_tool("outer_error_tool")
        executor = _make_executor(tools={"outer_error_tool": async_tool})
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="outer_error_tool", parameters={"x": "hello"}, tool_call_id="tc_fail")
        parsed = ParsedResponse(
            original_response="calling tool",
            tool_calls=[tc],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        # get_sandbox raising triggers the outer except block (lines 1572-1573)
        with (
            patch("nexau.archs.main_sub.agent_context.get_context", return_value=None),
            patch.object(agent_state, "get_sandbox", side_effect=RuntimeError("sandbox boom")),
        ):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        _, _, _, feedbacks = result
        assert len(feedbacks) >= 1
        assert feedbacks[0]["is_error"] is True
        assert "sandbox boom" in feedbacks[0]["content"]

    @pytest.mark.anyio
    async def test_async_tool_with_stop_tool_marker(self):
        """Async tool in stop_tools gets _is_stop_tool marker (line 1547-1548)."""

        async def _stop_impl(**kwargs: Any) -> dict[str, Any]:
            return {"result": "stopped"}

        stop_tool = Tool(
            name="my_stop_tool",
            description="A stop tool",
            input_schema={"type": "object", "properties": {}},
            implementation=_stop_impl,
        )
        executor = _make_executor(tools={"my_stop_tool": stop_tool}, stop_tools={"my_stop_tool"})
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="my_stop_tool", parameters={}, tool_call_id="tc_stop")
        parsed = ParsedResponse(
            original_response="calling stop tool",
            tool_calls=[tc],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        _, should_stop, stop_tool_result, feedbacks = result
        assert should_stop is True
        assert stop_tool_result is not None

    @pytest.mark.anyio
    async def test_async_tool_with_middleware_hooks(self):
        """Async tool path calls before_tool and after_tool middleware (lines 1523-1566)."""

        class TrackingMiddleware(Middleware):
            def __init__(self) -> None:
                self.before_tool_called = False
                self.after_tool_called = False

            def before_tool(self, hook_input: Any) -> HookResult:
                self.before_tool_called = True
                return HookResult(tool_input=dict(hook_input.tool_input))

            def after_tool(self, hook_input: Any) -> HookResult:
                self.after_tool_called = True
                return HookResult(tool_output=hook_input.tool_output)

        mw = TrackingMiddleware()
        async_tool = _make_async_tool("mw_async_tool")
        executor = _make_executor(tools={"mw_async_tool": async_tool}, middlewares=[mw])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="mw_async_tool", parameters={"x": "test"}, tool_call_id="tc_mw")
        parsed = ParsedResponse(
            original_response="calling mw async tool",
            tool_calls=[tc],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        _, _, _, feedbacks = result
        assert len(feedbacks) >= 1
        assert mw.before_tool_called is True
        assert mw.after_tool_called is True

    @pytest.mark.anyio
    async def test_async_tool_skip_sandbox_for_loadskill(self):
        """LoadSkill tool skips sandbox creation (line 1518)."""

        async def _skill_impl(**kwargs: Any) -> dict[str, Any]:
            return {"result": "loaded"}

        skill_tool = Tool(
            name="LoadSkill",
            description="Load a skill",
            input_schema={"type": "object", "properties": {"skill_name": {"type": "string"}}},
            implementation=_skill_impl,
        )
        executor = _make_executor(tools={"LoadSkill": skill_tool})
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="LoadSkill", parameters={"skill_name": "my_skill"}, tool_call_id="tc_skill")
        parsed = ParsedResponse(
            original_response="calling LoadSkill",
            tool_calls=[tc],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        _, _, _, feedbacks = result
        assert len(feedbacks) >= 1
        assert feedbacks[0].get("is_error") is not True

    @pytest.mark.anyio
    async def test_sub_agent_call_success(self):
        """Sub-agent call executes successfully (lines 1575-1587)."""
        executor = _make_executor()
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        sac = SubAgentCall(agent_name="helper_agent", message="Do something", sub_agent_call_id="sac_1")
        parsed = ParsedResponse(
            original_response="delegating to sub-agent",
            tool_calls=[],
            sub_agent_calls=[sac],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            with patch.object(
                executor.subagent_manager,
                "call_sub_agent_async",
                new_callable=AsyncMock,
                return_value="sub-agent result",
            ):
                result = await executor._execute_parsed_calls_async(
                    parsed,
                    agent_state,
                    framework_context=framework_context,
                )

        _, _, _, feedbacks = result
        assert len(feedbacks) >= 1
        assert feedbacks[0]["call_type"] == "sub_agent"
        assert feedbacks[0]["content"] == "sub-agent result"
        assert feedbacks[0].get("is_error") is not True

    @pytest.mark.anyio
    async def test_sub_agent_call_error(self):
        """Sub-agent call raises exception → caught at line 1588-1589."""
        executor = _make_executor()
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        sac = SubAgentCall(agent_name="broken_agent", message="This will fail", sub_agent_call_id="sac_fail")
        parsed = ParsedResponse(
            original_response="delegating to broken sub-agent",
            tool_calls=[],
            sub_agent_calls=[sac],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            with patch.object(
                executor.subagent_manager,
                "call_sub_agent_async",
                new_callable=AsyncMock,
                side_effect=RuntimeError("sub-agent boom"),
            ):
                result = await executor._execute_parsed_calls_async(
                    parsed,
                    agent_state,
                    framework_context=framework_context,
                )

        _, _, _, feedbacks = result
        assert len(feedbacks) >= 1
        assert feedbacks[0]["is_error"] is True
        assert "sub-agent boom" in feedbacks[0]["content"]

    @pytest.mark.anyio
    async def test_async_tool_after_tool_error_clears_stop_marker(self):
        """After-tool hook returning error status clears _is_stop_tool (lines 1568-1569)."""

        class ErrorAfterToolMiddleware(Middleware):
            def after_tool(self, hook_input: Any) -> HookResult:
                # 返回带 error status 的结果，触发 _is_stop_tool 清除逻辑
                return HookResult(tool_output={"status": "error", "_is_stop_tool": True, "result": "failed"})

        async def _stop_impl(**kwargs: Any) -> dict[str, Any]:
            return {"result": "ok"}

        mw = ErrorAfterToolMiddleware()
        stop_tool = Tool(
            name="erroring_stop",
            description="A stop tool that errors in after_tool",
            input_schema={"type": "object", "properties": {}},
            implementation=_stop_impl,
        )
        executor = _make_executor(
            tools={"erroring_stop": stop_tool},
            stop_tools={"erroring_stop"},
            middlewares=[mw],
        )
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="erroring_stop", parameters={}, tool_call_id="tc_err_stop")
        parsed = ParsedResponse(
            original_response="calling erroring stop tool",
            tool_calls=[tc],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        _, should_stop, stop_tool_result, feedbacks = result
        # _is_stop_tool cleared by line 1569 because status was "error"
        assert should_stop is False
