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

"""Tests for _emit_tool_error_result and the early-return paths that invoke it.

Covers:
- _emit_tool_error_result invokes after_tool middleware with correct AfterToolHookInput
- _emit_tool_error_result is a no-op when middleware_manager is None
- _emit_tool_error_result swallows and logs exceptions from middleware
- _execute_parsed_calls_async emits error result on tool-not-found
- _execute_parsed_calls_async emits error result on shutdown
- _execute_tool_call_safe emits error result on tool-not-found
"""

import threading
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import AfterToolHookInput, Middleware
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse, ToolCall
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry

# ---------------------------------------------------------------------------
# Helpers  (aligned with test_executor_coverage3.py conventions)
# ---------------------------------------------------------------------------


def _make_tool_registry(tools: dict[str, Tool] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    if tools:
        registry.add_source("test", list(tools.values()))
    return registry


def _make_config() -> LLMConfig:
    return LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )


def _make_executor(
    *,
    tools: dict[str, Tool] | None = None,
    middlewares: list[Middleware] | None = None,
) -> Executor:
    return Executor(
        agent_name="test_agent",
        agent_id="test_id",
        tool_registry=_make_tool_registry(tools),
        sub_agents={},
        stop_tools=set(),
        openai_client=Mock(),
        llm_config=_make_config(),
        team_mode=False,
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


def _make_tool_call(
    tool_name: str = "foo",
    tool_call_id: str = "tc_1",
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        parameters={"x": "1"},
        source="structured",
    )


# ---------------------------------------------------------------------------
# Test _emit_tool_error_result directly
# ---------------------------------------------------------------------------


class TestEmitToolErrorResult:
    """Unit tests for the _emit_tool_error_result helper method."""

    def test_invokes_after_tool_middleware(self):
        """_emit_tool_error_result calls middleware_manager.run_after_tool with correct input."""
        mock_middleware = Mock(spec=Middleware)
        executor = _make_executor(middlewares=[mock_middleware])
        agent_state = _make_agent_state()
        tc = _make_tool_call(tool_name="missing_tool", tool_call_id="tc_42")

        with patch.object(executor.middleware_manager, "run_after_tool") as mock_run:
            executor._emit_tool_error_result(tc, "Tool 'missing_tool' not found", agent_state)

        mock_run.assert_called_once()
        call_args = mock_run.call_args
        hook_input: AfterToolHookInput = call_args[0][0]
        initial_output = call_args[0][1]
        initial_llm_output = call_args[0][2]

        # Verify AfterToolHookInput fields
        assert hook_input.tool_name == "missing_tool"
        assert hook_input.tool_call_id == "tc_42"
        assert hook_input.sandbox is None
        assert hook_input.agent_state is agent_state
        assert hook_input.tool_input == {"x": "1"}
        assert hook_input.tool_output == {"status": "error", "error": "Tool 'missing_tool' not found"}
        assert hook_input.llm_tool_output == {"status": "error", "error": "Tool 'missing_tool' not found"}

        # Verify initial_output and initial_llm_output match the error dict
        assert initial_output == {"status": "error", "error": "Tool 'missing_tool' not found"}
        assert initial_llm_output == {"status": "error", "error": "Tool 'missing_tool' not found"}

    def test_noop_without_middleware(self):
        """_emit_tool_error_result does nothing when middleware_manager is None."""
        executor = _make_executor(middlewares=None)
        # Executor always creates a MiddlewareManager; force it to None for this test path.
        executor.middleware_manager = None
        agent_state = _make_agent_state()
        tc = _make_tool_call()

        # Should not raise — the method short-circuits when middleware_manager is None
        assert executor.middleware_manager is None
        executor._emit_tool_error_result(tc, "some error", agent_state)

    def test_swallows_middleware_exception(self):
        """_emit_tool_error_result catches and logs exceptions from run_after_tool."""
        mock_middleware = Mock(spec=Middleware)
        executor = _make_executor(middlewares=[mock_middleware])
        agent_state = _make_agent_state()
        tc = _make_tool_call(tool_name="bad_tool", tool_call_id="tc_err")

        with (
            patch.object(
                executor.middleware_manager,
                "run_after_tool",
                side_effect=RuntimeError("middleware exploded"),
            ),
            patch("nexau.archs.main_sub.execution.executor.logger") as mock_logger,
        ):
            # Should NOT raise
            executor._emit_tool_error_result(tc, "Tool 'bad_tool' not found", agent_state)

        # Verify the warning was logged with exact arguments
        mock_logger.warning.assert_called_once_with(
            "Failed to emit error tool result for '%s' (tool_call_id=%s)",
            "bad_tool",
            "tc_err",
            exc_info=True,
        )

    def test_tool_call_id_passed_through(self):
        """The tool_call_id from ToolCall is passed through to hook_input.

        ToolCall.__post_init__ auto-generates a UUID-based id when None is given,
        so _emit_tool_error_result should forward whatever ToolCall has.
        """
        mock_middleware = Mock(spec=Middleware)
        executor = _make_executor(middlewares=[mock_middleware])
        agent_state = _make_agent_state()
        tc = ToolCall(tool_name="x", tool_call_id=None, parameters={}, source="structured")
        # __post_init__ should have assigned a generated id
        assert tc.tool_call_id is not None
        assert tc.tool_call_id.startswith("tool_call_")

        with patch.object(executor.middleware_manager, "run_after_tool") as mock_run:
            executor._emit_tool_error_result(tc, "err", agent_state)

        hook_input: AfterToolHookInput = mock_run.call_args[0][0]
        assert hook_input.tool_call_id == tc.tool_call_id


# ---------------------------------------------------------------------------
# Test _execute_parsed_calls_async early-return paths
# ---------------------------------------------------------------------------


class TestExecuteParsedCallsAsyncErrorPaths:
    """Cover early-return error paths in _execute_parsed_calls_async."""

    @pytest.mark.anyio
    async def test_tool_not_found_emits_result_event(self):
        """When tool registry doesn't have the requested tool, emits error result and returns error."""
        executor = _make_executor(tools=None)  # empty registry
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = _make_tool_call(tool_name="nonexistent_tool", tool_call_id="tc_nf")
        parsed = ParsedResponse(
            original_response="calling nonexistent",
            tool_calls=[tc],
        )

        with (
            patch.object(executor, "_emit_tool_error_result") as mock_emit,
            patch("nexau.archs.main_sub.agent_context.get_context", return_value=None),
        ):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        # _emit_tool_error_result should have been called for the missing tool
        mock_emit.assert_called_once()
        emit_args = mock_emit.call_args[0]
        assert emit_args[0].tool_name == "nonexistent_tool"
        assert "not found" in emit_args[1]
        assert emit_args[2] is agent_state

        # The result should indicate an error
        _, _, _, feedbacks, _ = result
        assert len(feedbacks) >= 1
        assert feedbacks[0]["is_error"] is True

    @pytest.mark.anyio
    async def test_shutdown_emits_result_event(self):
        """When shutdown_event is set mid-execution, emits error result before returning."""
        executor = _make_executor(tools=None)
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = _make_tool_call(tool_name="some_tool", tool_call_id="tc_sd")
        parsed = ParsedResponse(
            original_response="calling tool during shutdown",
            tool_calls=[tc],
        )

        # Set shutdown AFTER the outer check but ensure inner _run_tool sees it.
        # The outer check at the top of _execute_parsed_calls_async returns early
        # with an empty feedbacks list, so we need to let it pass first and then
        # have the inner _run_tool see shutdown.
        call_count = 0

        def _delayed_shutdown() -> bool:
            nonlocal call_count
            call_count += 1
            # First call is the outer guard at the top of _execute_parsed_calls_async;
            # let it pass. Subsequent calls (inside _run_tool) should see shutdown.
            if call_count <= 1:
                return False
            return True

        with (
            patch.object(executor._shutdown_event, "is_set", side_effect=_delayed_shutdown),
            patch.object(executor, "_emit_tool_error_result") as mock_emit,
            patch("nexau.archs.main_sub.agent_context.get_context", return_value=None),
        ):
            result = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        # _emit_tool_error_result should have been called for shutdown
        mock_emit.assert_called_once()
        emit_args = mock_emit.call_args[0]
        assert emit_args[0].tool_name == "some_tool"
        assert "Shutdown" in emit_args[1]
        assert emit_args[2] is agent_state

        # Result should show an error
        _, _, _, feedbacks, _ = result
        assert len(feedbacks) >= 1
        assert feedbacks[0]["is_error"] is True


# ---------------------------------------------------------------------------
# Test _execute_tool_call_safe early-return path
# ---------------------------------------------------------------------------


class TestExecuteToolCallSafeErrorPaths:
    """Cover the early tool-not-found check in _execute_tool_call_safe."""

    def test_tool_not_found_emits_result_event(self):
        """The sync path calls _emit_tool_error_result when tool is not in registry."""
        executor = _make_executor(tools=None)  # empty registry
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = _make_tool_call(tool_name="ghost_tool", tool_call_id="tc_ghost")

        with patch.object(executor, "_emit_tool_error_result") as mock_emit:
            result = executor._execute_tool_call_safe(tc, agent_state, framework_context)

        # Verify _emit_tool_error_result was called
        mock_emit.assert_called_once()
        emit_args = mock_emit.call_args[0]
        assert emit_args[0] is tc
        assert "not found" in emit_args[1]
        assert emit_args[2] is agent_state

        # Verify the error return tuple
        tool_name, error_msg, is_error = result
        assert tool_name == "ghost_tool"
        assert "not found" in error_msg
        assert is_error is True

    def test_tool_found_does_not_call_emit(self):
        """When tool exists, _emit_tool_error_result is NOT called."""
        dummy_tool = Tool(
            name="real_tool",
            description="A real tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            implementation=lambda **kwargs: {"ok": True},
        )
        executor = _make_executor(tools={"real_tool": dummy_tool})
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = _make_tool_call(tool_name="real_tool", tool_call_id="tc_real")

        with (
            patch.object(executor, "_emit_tool_error_result") as mock_emit,
            patch("nexau.archs.main_sub.agent_context.get_context", return_value=None),
        ):
            result = executor._execute_tool_call_safe(tc, agent_state, framework_context)

        # _emit_tool_error_result should NOT have been called
        mock_emit.assert_not_called()

        # Result should be successful
        tool_name, output, is_error = result
        assert tool_name == "real_tool"
        assert is_error is False
