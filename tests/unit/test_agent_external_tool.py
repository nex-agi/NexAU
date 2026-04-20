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

"""RFC-0018 T3: Agent API return-value extension tests.

覆盖:
- 纯 external tool 暂停：run_async 返回 (response, dict) 元组
- 向后兼容：pending 列表为空时 run_async 仍返回纯 str
- 恢复路径：调用方传入 ToolResultBlock 后 executor 能看到 tool result in history
- Tracing 分支：tracing 包装下元组返回值仍被正确透传
"""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock


class _DummyTracer(BaseTracer):
    """Minimal tracer stub used by the tracing path test."""

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict | None = None,
        parent_span: Span | None = None,
        attributes: dict | None = None,
    ) -> Span:
        return Span(
            id=name,
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(self, span: Span, outputs=None, error=None, attributes=None) -> None:
        span.outputs = outputs or {}
        span.error = str(error) if error else None
        if attributes:
            span.attributes.update(attributes)


@pytest.fixture
def global_storage() -> GlobalStorage:
    return GlobalStorage()


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        name="rfc0018_t3",
        system_prompt="You are a test assistant.",
        llm_config=LLMConfig(model="gpt-4o-mini"),
    )


def _assistant_with_tool_use(call_id: str, name: str, arguments: dict[str, Any]) -> Message:
    return Message(role=Role.ASSISTANT, content=[ToolUseBlock(id=call_id, name=name, input=arguments)])


class TestRunAsyncExternalTool:
    """Agent.run_async must surface the executor's pending_external_calls."""

    def test_returns_tuple_when_paused_on_external_tool(self, agent_config, global_storage):
        """纯 external tool 暂停：run_async 返回 (response, dict) 元组。"""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            agent = Agent(config=agent_config, global_storage=global_storage)

            pending = [ToolUseBlock(id="call_ext_1", name="remote_exec", input={"cmd": "ls"})]
            assistant_msg = _assistant_with_tool_use("call_ext_1", "remote_exec", {"cmd": "ls"})

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="sys")]),
                        Message.user("run ls"),
                        assistant_msg,
                    ],
                    pending,
                )

                result = agent.run(message="run ls")

            assert isinstance(result, tuple), f"Expected tuple when paused, got {type(result).__name__}"
            response_text, meta = result
            assert response_text == ""
            assert meta["stop_reason"] == AgentStopReason.EXTERNAL_TOOL_CALL.name
            assert meta["stop_reason"] == "EXTERNAL_TOOL_CALL"

            pending_dicts = meta["pending_tool_calls"]
            assert isinstance(pending_dicts, list)
            assert len(pending_dicts) == 1
            entry = pending_dicts[0]
            assert entry == {"id": "call_ext_1", "name": "remote_exec", "input": {"cmd": "ls"}}

    def test_returns_bare_string_when_no_pending(self, agent_config, global_storage):
        """向后兼容：无 pending 时 run_async 返回纯 str（不是 tuple）。"""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            agent = Agent(config=agent_config, global_storage=global_storage)

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "ok",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="sys")]),
                        Message.user("hi"),
                        Message.assistant("ok"),
                    ],
                    [],
                )

                result = agent.run(message="hi")

            assert result == "ok"
            assert isinstance(result, str), f"Expected bare str for backward compat, got {type(result).__name__}"

    def test_resume_path_feeds_tool_result_to_executor(self, agent_config, global_storage):
        """恢复路径：调用方再次调用 run_async 传入 ToolResult，executor 能在 history 末尾看到 tool_result。"""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            agent = Agent(config=agent_config, global_storage=global_storage)

            pending = [ToolUseBlock(id="call_remote", name="shell", input={"cmd": "pwd"})]
            assistant_msg = _assistant_with_tool_use("call_remote", "shell", {"cmd": "pwd"})

            histories_observed: list[list[Message]] = []

            async def fake_execute_async(
                history,
                agent_state,
                runtime_client=None,
                custom_llm_client_provider=None,
            ):
                del agent_state, runtime_client, custom_llm_client_provider
                histories_observed.append(list(history))

                if len(histories_observed) == 1:
                    # 第一轮：LLM 触发 external tool 调用，executor 暂停。
                    updated = [*history, assistant_msg]
                    return "", updated, pending

                # 第二轮：history 末尾应已经包含 ToolResultBlock，正常结束。
                updated = [*history, Message.assistant("final answer")]
                return "final answer", updated, []

            with patch.object(agent.executor, "execute_async", side_effect=fake_execute_async):
                first = agent.run(message="run pwd")
                assert isinstance(first, tuple)
                assert first[1]["stop_reason"] == "EXTERNAL_TOOL_CALL"

                # 调用方执行 external tool 后，通过 ToolResultBlock 恢复 agent。
                resume_message = [
                    Message(
                        role=Role.TOOL,
                        content=[ToolResultBlock(tool_use_id="call_remote", content="/home/user", is_error=False)],
                    ),
                ]
                second = agent.run(message=resume_message)

            assert second == "final answer"
            assert len(histories_observed) == 2

            # 第二轮的 history 末尾应该是 TOOL role，且带回我们喂进去的 tool_use_id。
            last_history = histories_observed[1]
            tool_msg = last_history[-1]
            assert tool_msg.role == Role.TOOL
            tool_result_block = tool_msg.content[0]
            assert isinstance(tool_result_block, ToolResultBlock)
            assert tool_result_block.tool_use_id == "call_remote"
            assert tool_result_block.content == "/home/user"

    def test_tracing_path_preserves_tuple_return(self, agent_config, global_storage):
        """Tracing 分支：tuple 返回值在 TraceContext 下被正确透传。"""
        tracer = _DummyTracer()
        agent_config.tracers = [tracer]
        agent_config.resolved_tracer = tracer

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            agent = Agent(config=agent_config, global_storage=global_storage)
            agent.global_storage.set("tracer", tracer)

            pending = [ToolUseBlock(id="call_traced", name="external_op", input={})]
            assistant_msg = _assistant_with_tool_use("call_traced", "external_op", {})

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="sys")]),
                        Message.user("traced"),
                        assistant_msg,
                    ],
                    pending,
                )

                result = agent.run(message="traced")

            assert isinstance(result, tuple)
            response_text, meta = result
            assert response_text == ""
            assert meta["stop_reason"] == "EXTERNAL_TOOL_CALL"
            assert meta["pending_tool_calls"] == [{"id": "call_traced", "name": "external_op", "input": {}}]
