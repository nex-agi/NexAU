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

"""Additional coverage tests for hooks.py.

Targets uncovered paths in:
- Middleware supports_on_event, get_event_handler, supports_set_event_emitter, supports_set_llm_runtime
- MiddlewareManager: run_before_agent, run_after_agent, run_after_model, run_after_tool,
  run_before_tool, wrap_model_call, wrap_tool_call, stream_chunk, _normalize_result
- LoggingMiddleware: wrap_model_call, stream_chunk, before_model, before_tool, after_tool
- FunctionMiddleware: before_tool, after_tool
- create_tool_after_approve_hook
"""

from unittest.mock import Mock

import pytest

from nexau.archs.main_sub.execution.hooks import (
    AfterAgentHookInput,
    AfterModelHookInput,
    AfterToolHookInput,
    BeforeAgentHookInput,
    BeforeModelHookInput,
    BeforeToolHookInput,
    FunctionMiddleware,
    HookResult,
    LoggingMiddleware,
    Middleware,
    MiddlewareManager,
    ModelCallParams,
    ToolCallParams,
    create_tool_after_approve_hook,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse
from nexau.core.messages import Message, Role, TextBlock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_state():
    state = Mock()
    state.agent_name = "test_agent"
    state.agent_id = "test_id"
    return state


@pytest.fixture
def model_call_params():
    return ModelCallParams(
        messages=[Message(role=Role.USER, content=[TextBlock(text="Hi")])],
        max_tokens=100,
        force_stop_reason=None,
        agent_state=None,
        tool_call_mode="xml",
        tools=None,
        api_params={},
    )


# ---------------------------------------------------------------------------
# Middleware base class properties
# ---------------------------------------------------------------------------


class TestMiddlewareProperties:
    def test_supports_on_event_false_by_default(self):
        mw = Middleware()
        assert mw.supports_on_event is False

    def test_supports_on_event_true_with_attr(self):
        mw = Middleware()
        mw.on_event = lambda evt: None  # type: ignore[attr-defined]
        assert mw.supports_on_event is True

    def test_get_event_handler_none_by_default(self):
        mw = Middleware()
        assert mw.get_event_handler() is None

    def test_get_event_handler_returns_callable(self):
        mw = Middleware()
        handler = lambda evt: None  # noqa: E731 — test stub
        mw.on_event = handler  # type: ignore[attr-defined]
        assert mw.get_event_handler() is handler

    def test_supports_set_event_emitter_false(self):
        mw = Middleware()
        assert mw.supports_set_event_emitter is False

    def test_supports_set_llm_runtime_false(self):
        mw = Middleware()
        assert mw.supports_set_llm_runtime is False

    def test_supports_set_event_emitter_true_when_overridden(self):
        class MyMiddleware(Middleware):
            def set_event_emitter(self, emitter):
                pass

        mw = MyMiddleware()
        assert mw.supports_set_event_emitter is True

    def test_supports_set_llm_runtime_true_when_overridden(self):
        class MyMiddleware(Middleware):
            def set_llm_runtime(self, llm_config, openai_client, **kwargs):
                pass

        mw = MyMiddleware()
        assert mw.supports_set_llm_runtime is True


# ---------------------------------------------------------------------------
# MiddlewareManager — run_before_agent / run_after_agent
# ---------------------------------------------------------------------------


class TestMiddlewareManagerAgentHooks:
    def test_run_before_agent_modifies_messages(self, agent_state):
        new_msgs = [Message(role=Role.USER, content=[TextBlock(text="modified")])]

        class ModifyMiddleware(Middleware):
            def before_agent(self, hook_input):
                return HookResult(messages=new_msgs)

        mm = MiddlewareManager([ModifyMiddleware()])
        hook_input = BeforeAgentHookInput(
            agent_state=agent_state,
            messages=[Message(role=Role.USER, content=[TextBlock(text="original")])],
        )
        result = mm.run_before_agent(hook_input)
        assert result[0].get_text_content() == "modified"

    def test_run_before_agent_failure_logged(self, agent_state):
        class FailingMiddleware(Middleware):
            def before_agent(self, hook_input):
                raise RuntimeError("boom")

        mm = MiddlewareManager([FailingMiddleware()])
        msgs = [Message(role=Role.USER, content=[TextBlock(text="hi")])]
        hook_input = BeforeAgentHookInput(agent_state=agent_state, messages=msgs)
        result = mm.run_before_agent(hook_input)
        assert result == msgs  # Original messages preserved

    def test_run_after_agent_modifies_response(self, agent_state):
        class ModifyMiddleware(Middleware):
            def after_agent(self, hook_input):
                return HookResult(agent_response="new response")

        mm = MiddlewareManager([ModifyMiddleware()])
        msgs = [Message(role=Role.USER, content=[TextBlock(text="hi")])]
        hook_input = AfterAgentHookInput(
            agent_state=agent_state,
            messages=msgs,
            agent_response="old response",
        )
        response, result_msgs = mm.run_after_agent(hook_input)
        assert response == "new response"

    def test_run_after_agent_modifies_messages(self, agent_state):
        new_msgs = [Message(role=Role.USER, content=[TextBlock(text="modified")])]

        class ModifyMiddleware(Middleware):
            def after_agent(self, hook_input):
                return HookResult(messages=new_msgs)

        mm = MiddlewareManager([ModifyMiddleware()])
        hook_input = AfterAgentHookInput(
            agent_state=agent_state,
            messages=[Message(role=Role.USER, content=[TextBlock(text="hi")])],
            agent_response="resp",
        )
        _, result_msgs = mm.run_after_agent(hook_input)
        assert result_msgs[0].get_text_content() == "modified"

    def test_run_after_agent_failure_logged(self, agent_state):
        class FailingMiddleware(Middleware):
            def after_agent(self, hook_input):
                raise RuntimeError("boom")

        mm = MiddlewareManager([FailingMiddleware()])
        hook_input = AfterAgentHookInput(
            agent_state=agent_state,
            messages=[],
            agent_response="resp",
        )
        response, _ = mm.run_after_agent(hook_input)
        assert response == "resp"


# ---------------------------------------------------------------------------
# MiddlewareManager — wrap_model_call chain
# ---------------------------------------------------------------------------


class TestMiddlewareManagerWrapModelCall:
    def test_chain_calls_all_middlewares(self, model_call_params):
        call_order = []

        class MW1(Middleware):
            def wrap_model_call(self, params, call_next):
                call_order.append("mw1")
                return call_next(params)

        class MW2(Middleware):
            def wrap_model_call(self, params, call_next):
                call_order.append("mw2")
                return call_next(params)

        def base_call(params):
            call_order.append("base")
            return Mock(spec=ModelResponse)

        mm = MiddlewareManager([MW1(), MW2()])
        mm.wrap_model_call(model_call_params, base_call)
        assert call_order == ["mw1", "mw2", "base"]


# ---------------------------------------------------------------------------
# MiddlewareManager — wrap_tool_call chain
# ---------------------------------------------------------------------------


class TestMiddlewareManagerWrapToolCall:
    def test_chain_calls_all_middlewares(self, agent_state):
        call_order = []

        class MW1(Middleware):
            def wrap_tool_call(self, params, call_next):
                call_order.append("mw1")
                return call_next(params)

        def base_call(params):
            call_order.append("base")
            return {"result": "ok"}

        tool_params = ToolCallParams(
            agent_state=agent_state,
            sandbox=None,
            tool_name="test",
            parameters={},
            tool_call_id="tc1",
            execution_params={},
        )
        mm = MiddlewareManager([MW1()])
        mm.wrap_tool_call(tool_params, base_call)
        assert call_order == ["mw1", "base"]


# ---------------------------------------------------------------------------
# MiddlewareManager — stream_chunk
# ---------------------------------------------------------------------------


class TestMiddlewareManagerStreamChunk:
    def test_passes_through_unchanged(self, model_call_params):
        mm = MiddlewareManager([Middleware()])
        result = mm.stream_chunk("chunk", model_call_params)
        assert result == "chunk"

    def test_middleware_drops_chunk(self, model_call_params):
        class DropMiddleware(Middleware):
            def stream_chunk(self, chunk, params):
                return None

        mm = MiddlewareManager([DropMiddleware()])
        result = mm.stream_chunk("chunk", model_call_params)
        assert result is None


# ---------------------------------------------------------------------------
# MiddlewareManager._normalize_result — async coroutine detection
# ---------------------------------------------------------------------------


class TestNormalizeResult:
    def test_none_returns_no_changes(self):
        result = MiddlewareManager._normalize_result(None)
        assert not result.has_modifications()

    def test_coroutine_raises_type_error(self):
        async def fake_hook():
            return HookResult()

        coro = fake_hook()
        with pytest.raises(TypeError, match="coroutine"):
            MiddlewareManager._normalize_result(coro)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# FunctionMiddleware
# ---------------------------------------------------------------------------


class TestFunctionMiddlewareCoverage:
    def test_before_tool_hook(self, agent_state):
        called = {}

        def hook(hook_input):
            called["invoked"] = True
            return HookResult(tool_input={"modified": True})

        fm = FunctionMiddleware(before_tool_hook=hook)
        hook_input = BeforeToolHookInput(
            agent_state=agent_state,
            sandbox=None,
            tool_name="test",
            tool_call_id="tc1",
            tool_input={},
        )
        result = fm.before_tool(hook_input)
        assert called["invoked"]
        assert result.tool_input == {"modified": True}

    def test_after_tool_hook(self, agent_state):
        called = {}

        def hook(hook_input):
            called["invoked"] = True
            return HookResult(tool_output="modified output")

        fm = FunctionMiddleware(after_tool_hook=hook)
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            sandbox=None,
            tool_name="test",
            tool_call_id="tc1",
            tool_input={},
            tool_output="original",
        )
        result = fm.after_tool(hook_input)
        assert called["invoked"]
        assert result.tool_output == "modified output"

    def test_no_hooks_returns_no_changes(self):
        fm = FunctionMiddleware()
        hook_input = Mock()
        assert not fm.before_model(hook_input).has_modifications()
        assert not fm.after_model(hook_input).has_modifications()
        assert not fm.before_tool(hook_input).has_modifications()
        assert not fm.after_tool(hook_input).has_modifications()


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


class TestLoggingMiddlewareCoverage:
    def test_before_model_logs(self, agent_state):
        mw = LoggingMiddleware(model_logger="test_model")
        hook_input = BeforeModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            messages=[Message(role=Role.USER, content=[TextBlock(text="hi")])],
        )
        result = mw.before_model(hook_input)
        assert not result.has_modifications()

    def test_before_model_no_logger(self, agent_state):
        mw = LoggingMiddleware()
        hook_input = BeforeModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            messages=[],
        )
        result = mw.before_model(hook_input)
        assert not result.has_modifications()

    def test_before_tool_logs(self, agent_state):
        mw = LoggingMiddleware(tool_logger="test_tool")
        hook_input = BeforeToolHookInput(
            agent_state=agent_state,
            sandbox=None,
            tool_name="test",
            tool_call_id="tc1",
            tool_input={},
        )
        result = mw.before_tool(hook_input)
        assert not result.has_modifications()

    def test_after_tool_logs_short_output(self, agent_state):
        mw = LoggingMiddleware(tool_logger="test_tool")
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            sandbox=None,
            tool_name="test",
            tool_call_id="tc1",
            tool_input={},
            tool_output="short output",
        )
        result = mw.after_tool(hook_input)
        assert not result.has_modifications()

    def test_after_tool_logs_long_output(self, agent_state):
        mw = LoggingMiddleware(tool_logger="test_tool", tool_preview_chars=10)
        hook_input = AfterToolHookInput(
            agent_state=agent_state,
            sandbox=None,
            tool_name="test",
            tool_call_id="tc1",
            tool_input={},
            tool_output="x" * 100,
        )
        result = mw.after_tool(hook_input)
        assert not result.has_modifications()

    def test_wrap_model_call_logs(self, model_call_params):
        mw = LoggingMiddleware(model_logger="test_model", log_model_calls=True)
        resp = Mock(spec=ModelResponse)
        resp.render_text.return_value = "response text"
        resp.content = "response text"
        result = mw.wrap_model_call(model_call_params, lambda p: resp)
        assert result is resp

    def test_wrap_model_call_none_response(self, model_call_params):
        mw = LoggingMiddleware(model_logger="test_model", log_model_calls=True)
        result = mw.wrap_model_call(model_call_params, lambda p: None)
        assert result is None

    def test_stream_chunk_logs(self, model_call_params):
        mw = LoggingMiddleware(model_logger="test_model")
        result = mw.stream_chunk("chunk_data", model_call_params)
        assert result == "chunk_data"


# ---------------------------------------------------------------------------
# create_tool_after_approve_hook
# ---------------------------------------------------------------------------


class TestCreateToolAfterApproveHook:
    def test_logs_when_tool_matches(self, agent_state):
        hook = create_tool_after_approve_hook("my_tool")
        parsed = ParsedResponse(
            original_response="",
            tool_calls=[Mock(tool_name="my_tool")],
            sub_agent_calls=[],
            batch_agent_calls=[],
        )
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            messages=[],
            original_response="",
            parsed_response=parsed,
        )
        result = hook(hook_input)
        assert not result.has_modifications()

    def test_no_parsed_response(self, agent_state):
        hook = create_tool_after_approve_hook("my_tool")
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            messages=[],
            original_response="",
            parsed_response=None,
        )
        result = hook(hook_input)
        assert not result.has_modifications()


# ---------------------------------------------------------------------------
# HookResult helpers
# ---------------------------------------------------------------------------


class TestHookResultHelpers:
    def test_has_tool_input(self):
        hr = HookResult(tool_input={"key": "value"})
        assert hr.has_modifications()

    def test_has_agent_response(self):
        hr = HookResult(agent_response="resp")
        assert hr.has_agent_response()
        assert hr.has_modifications()

    def test_no_changes(self):
        hr = HookResult.no_changes()
        assert not hr.has_modifications()

    def test_with_modifications(self):
        hr = HookResult.with_modifications(force_continue=True)
        assert hr.has_modifications()
        assert hr.force_continue
