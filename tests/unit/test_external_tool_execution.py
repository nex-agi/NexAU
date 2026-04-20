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

"""RFC-0018 T2: Executor-layer external tool pause/resume tests.

覆盖:
- `_execute_parsed_calls_async` 将 external tool 从派发列表中剥离并放入 pending 列表
- 混合 external + local 场景：local 正常执行，external 放入 pending
- 无 external 场景：pending 列表为空（向后兼容）
- `_process_xml_calls_async` 透传 pending 列表给上层
"""

import threading
from typing import Any
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import AfterModelHookInput
from nexau.archs.main_sub.execution.parse_structures import ParsedResponse, ToolCall
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry


def _make_local_tool(name: str) -> Tool:
    async def _impl(**kwargs: Any) -> dict[str, Any]:
        return {"result": f"local:{name}"}

    return Tool(
        name=name,
        description=f"Local tool {name}",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        implementation=_impl,
    )


def _make_external_tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"External tool {name}",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        implementation=None,
        kind="external",
    )


def _make_tool_registry(tools: list[Tool]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.add_source("rfc0018", tools)
    return registry


def _make_executor(tools: list[Tool]) -> Executor:
    llm_config = LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )
    return Executor(
        agent_name="rfc0018_agent",
        agent_id="rfc0018_id",
        tool_registry=_make_tool_registry(tools),
        sub_agents={},
        stop_tools=set(),
        openai_client=Mock(),
        llm_config=llm_config,
    )


def _make_agent_state() -> AgentState:
    return AgentState(
        agent_name="rfc0018_agent",
        agent_id="rfc0018_id",
        run_id="run_1",
        root_run_id="run_1",
        context=AgentContext({}),
        global_storage=GlobalStorage(),
        tool_registry=ToolRegistry(),
    )


def _make_framework_context() -> FrameworkContext:
    return FrameworkContext(
        agent_name="rfc0018_agent",
        agent_id="rfc0018_id",
        run_id="run_1",
        root_run_id="run_1",
        _tool_registry=ToolRegistry(),
        _shutdown_event=threading.Event(),
    )


class TestExecuteParsedCallsAsyncExternalDispatch:
    """RFC-0018: `_execute_parsed_calls_async` 识别并剥离 external tool calls。"""

    @pytest.mark.anyio
    async def test_pure_external_returns_pending_no_execution(self):
        """全部 external：feedbacks 为空，pending 含对应 ToolUseBlock。"""
        ext = _make_external_tool("remote_search")
        executor = _make_executor([ext])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="remote_search", parameters={"q": "hello"}, tool_call_id="call_ext_1")
        parsed = ParsedResponse(original_response="calling remote tool", tool_calls=[tc])

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            (
                _processed,
                should_stop,
                _stop_result,
                feedbacks,
                pending_external_calls,
            ) = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        assert should_stop is False
        assert feedbacks == []
        assert len(pending_external_calls) == 1
        block = pending_external_calls[0]
        assert block.name == "remote_search"
        assert block.id == "call_ext_1"
        assert block.input == {"q": "hello"}

    @pytest.mark.anyio
    async def test_mixed_local_and_external(self):
        """混合：local 进入 feedbacks，external 进入 pending。"""
        local = _make_local_tool("local_add")
        ext = _make_external_tool("remote_fetch")
        executor = _make_executor([local, ext])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc_local = ToolCall(tool_name="local_add", parameters={"x": "1"}, tool_call_id="call_local")
        tc_ext = ToolCall(tool_name="remote_fetch", parameters={"url": "example.com"}, tool_call_id="call_ext")
        parsed = ParsedResponse(original_response="mixed", tool_calls=[tc_local, tc_ext])

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            (
                _processed,
                _should_stop,
                _stop_result,
                feedbacks,
                pending_external_calls,
            ) = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        assert len(feedbacks) == 1
        assert feedbacks[0]["call"].tool_name == "local_add"
        assert feedbacks[0].get("is_error") is not True

        assert len(pending_external_calls) == 1
        assert pending_external_calls[0].name == "remote_fetch"
        assert pending_external_calls[0].input == {"url": "example.com"}

    @pytest.mark.anyio
    async def test_pure_local_pending_is_empty(self):
        """无 external：pending 为空，保持向后兼容。"""
        local = _make_local_tool("local_only")
        executor = _make_executor([local])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="local_only", parameters={"x": "v"}, tool_call_id="call_local_only")
        parsed = ParsedResponse(original_response="local only", tool_calls=[tc])

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            (
                _processed,
                _should_stop,
                _stop_result,
                feedbacks,
                pending_external_calls,
            ) = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        assert len(feedbacks) == 1
        assert pending_external_calls == []

    @pytest.mark.anyio
    async def test_unregistered_tool_treated_as_local_error(self):
        """未注册的 tool 不被视为 external，走下游 tool-not-found 错误路径。"""
        ext = _make_external_tool("known_ext")
        executor = _make_executor([ext])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="unregistered", parameters={}, tool_call_id="call_unk")
        parsed = ParsedResponse(original_response="unknown", tool_calls=[tc])

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            (
                _processed,
                _should_stop,
                _stop_result,
                feedbacks,
                pending_external_calls,
            ) = await executor._execute_parsed_calls_async(
                parsed,
                agent_state,
                framework_context=framework_context,
            )

        assert pending_external_calls == []
        assert len(feedbacks) == 1
        assert feedbacks[0].get("is_error") is True


class TestProcessXmlCallsAsyncPendingPassthrough:
    """RFC-0018: `_process_xml_calls_async` 透传 pending 列表给上层调用方。"""

    @pytest.mark.anyio
    async def test_pending_propagates_through_process_xml_calls(self):
        """`_process_xml_calls_async` 应将 `_execute_parsed_calls_async` 返回的 pending
        原样放到第 6 个元素返回。
        """
        ext = _make_external_tool("remote_x")
        executor = _make_executor([ext])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        tc = ToolCall(tool_name="remote_x", parameters={"k": "v"}, tool_call_id="tid_1")
        parsed = ParsedResponse(original_response="resp", tool_calls=[tc])

        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            original_response="resp",
            parsed_response=parsed,
            messages=[],
        )

        with patch("nexau.archs.main_sub.agent_context.get_context", return_value=None):
            (
                _processed,
                _should_stop,
                _stop_tool_result,
                _updated_messages,
                feedbacks,
                pending_external_calls,
            ) = await executor._process_xml_calls_async(
                hook_input,
                framework_context=framework_context,
            )

        assert feedbacks == []
        assert len(pending_external_calls) == 1
        assert pending_external_calls[0].name == "remote_x"
        assert pending_external_calls[0].input == {"k": "v"}

    @pytest.mark.anyio
    async def test_no_calls_returns_empty_pending(self):
        """无 tool_calls 时 pending 应为空列表。"""
        executor = _make_executor([])
        agent_state = _make_agent_state()
        framework_context = _make_framework_context()

        parsed = ParsedResponse(original_response="just text", tool_calls=[])
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            original_response="just text",
            parsed_response=parsed,
            messages=[],
        )

        (
            _processed,
            _should_stop,
            _stop_tool_result,
            _updated_messages,
            feedbacks,
            pending_external_calls,
        ) = await executor._process_xml_calls_async(
            hook_input,
            framework_context=framework_context,
        )

        assert feedbacks == []
        assert pending_external_calls == []
