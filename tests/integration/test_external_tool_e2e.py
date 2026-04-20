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

"""RFC-0018 T6: End-to-end integration tests for external-tool pause/resume.

测试策略：使用真实 Agent + Executor + ToolRegistry + SessionManager，
仅在 LLM 边界（`LLMCaller.call_llm_async`）上 mock。这样可以覆盖从 Agent
到 Tool 派发的完整路径，而不依赖真实 LLM provider。

覆盖场景：
- 纯 external tool 流程（pause → resume → complete）
- local + external 混合流程（local 先执行、external 暂停）
- 多轮 external（同一 session 内连续多个 external turn）
- 错误路径：`is_error=True` 的 ToolResultBlock
- 边界场景：`tool_use_id` 不匹配时恢复仍成立（history 照常追加，LLM 自行判断）
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool
from nexau.core.messages import Message, Role, ToolResultBlock, ToolUseBlock

# ----------------------------- Fixtures / helpers ---------------------------


def _make_local_tool(name: str, result: str = "local-ok") -> Tool:
    """本地工具：返回一个确定性的字符串结果。"""

    def _impl(**kwargs: Any) -> dict[str, Any]:
        # kwargs 在这里只是为了配合任意 schema，不做业务用途。
        del kwargs
        return {"ok": True, "tool": name, "result": result}

    return Tool(
        name=name,
        description=f"Local tool {name}",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        implementation=_impl,
    )


def _make_external_tool(name: str) -> Tool:
    """External tool：无本地 implementation，kind=\"external\"。"""
    return Tool(
        name=name,
        description=f"External tool {name}",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        implementation=None,
        kind="external",
    )


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ModelToolCall:
    import json

    return ModelToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


@pytest.fixture
def global_storage() -> GlobalStorage:
    return GlobalStorage()


def _make_agent(
    tools: list[Tool],
    *,
    global_storage: GlobalStorage,
    session_manager: SessionManager | None = None,
    session_id: str | None = None,
) -> Agent:
    """创建 Agent —— 真实 Executor / ToolRegistry / SessionManager，仅 patch openai 客户端。"""
    config = AgentConfig(
        name="rfc0018_e2e_agent",
        system_prompt="You are a test assistant.",
        llm_config=LLMConfig(model="gpt-4o-mini"),
        tools=tools,
        tool_call_mode="openai",
    )
    with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
        mock_openai.OpenAI.return_value = Mock()
        return Agent(
            config=config,
            global_storage=global_storage,
            session_manager=session_manager,
            user_id="user_e2e" if session_manager else None,
            session_id=session_id,
        )


# ---------------------------- Pure external flow ---------------------------


class TestPureExternalToolFlow:
    """场景 1：纯 external tool —— 暂停 → 喂回结果 → 完成。"""

    def test_pause_then_resume_completes_normally(self, global_storage):
        external = _make_external_tool("remote_search")
        agent = _make_agent([external], global_storage=global_storage)

        first_response = ModelResponse(
            content="I'll call the remote search tool.",
            tool_calls=[_make_tool_call("call_ext_1", "remote_search", {"q": "hello"})],
        )
        second_response = ModelResponse(content="Search returned hello-world.", tool_calls=[])

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response, second_response],
        ) as mock_call_llm:
            # 第一轮：触发 external tool，Agent 暂停。
            first = agent.run(message="please search for hello")

            assert isinstance(first, tuple), f"Expected tuple when paused, got {type(first).__name__}"
            response_text, meta = first
            assert response_text == ""
            assert meta["stop_reason"] == AgentStopReason.EXTERNAL_TOOL_CALL.name
            assert meta["pending_tool_calls"] == [
                {"id": "call_ext_1", "name": "remote_search", "input": {"q": "hello"}},
            ]
            # Pause 时仅发生一次 LLM 调用；resume 前不会再调。
            assert mock_call_llm.call_count == 1

            # 调用方拿到 pending 后执行并喂回结果，Agent 恢复并完成。
            resume = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(
                                tool_use_id="call_ext_1",
                                content="hello-world",
                                is_error=False,
                            ),
                        ],
                    ),
                ],
            )

            assert resume == "Search returned hello-world."
            assert mock_call_llm.call_count == 2

        # History 里应留下完整的 assistant(tool_use) + tool_result + final assistant 顺序。
        history = agent.history
        tool_use_msgs = [m for m in history if m.role == Role.ASSISTANT and any(isinstance(b, ToolUseBlock) for b in m.content)]
        assert any(any(isinstance(b, ToolUseBlock) and b.id == "call_ext_1" for b in m.content) for m in tool_use_msgs), (
            "assistant 消息里应保留 tool_use(call_ext_1)"
        )

        tool_result_msgs = [m for m in history if m.role == Role.TOOL]
        assert any(any(isinstance(b, ToolResultBlock) and b.tool_use_id == "call_ext_1" for b in m.content) for m in tool_result_msgs), (
            "history 末尾附近应有 call_ext_1 的 ToolResult"
        )

    def test_resume_isolated_tool_use_id(self, global_storage):
        """喂回的 ToolResultBlock tool_use_id 与 pending 不匹配时，history 仍然记录结果。

        验证 Agent 不会主动丢弃或替换这种数据，避免调用方错误被悄悄吞掉；LLM 层可自行决定如何处理。
        """
        external = _make_external_tool("remote_action")
        agent = _make_agent([external], global_storage=global_storage)

        first_response = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_real", "remote_action", {"q": "v"})],
        )
        second_response = ModelResponse(content="final", tool_calls=[])

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response, second_response],
        ):
            paused = agent.run(message="go")
            assert isinstance(paused, tuple)
            _, meta = paused
            assert meta["pending_tool_calls"][0]["id"] == "call_real"

            # 喂进去一个 "错号" 的 tool_use_id
            final = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(tool_use_id="call_wrong_id", content="ignored", is_error=False),
                        ],
                    ),
                ],
            )

        assert final == "final"
        tool_result_ids = {b.tool_use_id for m in agent.history if m.role == Role.TOOL for b in m.content if isinstance(b, ToolResultBlock)}
        # history 保留了调用方提交的（错号）结果，没有被框架篡改或丢弃。
        assert "call_wrong_id" in tool_result_ids


# ----------------------------- Mixed flow ---------------------------------


class TestMixedLocalExternalFlow:
    """场景 2：local + external 混合 —— local 先本地执行，external 暂停。"""

    def test_local_executes_external_pauses(self, global_storage):
        local = _make_local_tool("local_adder", result="42")
        external = _make_external_tool("remote_fetch")
        agent = _make_agent([local, external], global_storage=global_storage)

        first_response = ModelResponse(
            content="Delegating one local and one remote call.",
            tool_calls=[
                _make_tool_call("call_local", "local_adder", {"x": "1"}),
                _make_tool_call("call_ext", "remote_fetch", {"q": "extra"}),
            ],
        )
        second_response = ModelResponse(content="done", tool_calls=[])

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response, second_response],
        ):
            paused = agent.run(message="mixed call")

            assert isinstance(paused, tuple)
            _, meta = paused
            assert meta["stop_reason"] == "EXTERNAL_TOOL_CALL"
            # pending 只包含 external；local 已经执行并写入 history。
            assert len(meta["pending_tool_calls"]) == 1
            assert meta["pending_tool_calls"][0]["name"] == "remote_fetch"
            assert meta["pending_tool_calls"][0]["id"] == "call_ext"

            # Resume：喂回 external 结果后正常完成。
            resume = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(
                                tool_use_id="call_ext",
                                content="remote-ok",
                                is_error=False,
                            ),
                        ],
                    ),
                ],
            )

        assert resume == "done"

        # 验证 history：local 的 tool_result 应在 pause 前就被写入。
        tool_result_ids = [b.tool_use_id for m in agent.history if m.role == Role.TOOL for b in m.content if isinstance(b, ToolResultBlock)]
        assert "call_local" in tool_result_ids, "local tool 结果应已写入 history"
        assert "call_ext" in tool_result_ids, "resume 后 external tool 结果应写入 history"


# -------------------------- Multi-turn external ---------------------------


class TestMultiTurnExternal:
    """场景 3：同一 session 内出现多个 external turn。"""

    def test_two_separate_external_pauses_in_same_session(self, global_storage):
        external = _make_external_tool("remote_op")

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_multi",
        )

        turn1_llm = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_t1", "remote_op", {"q": "first"})],
        )
        turn1_final = ModelResponse(content="first-done", tool_calls=[])
        turn2_llm = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_t2", "remote_op", {"q": "second"})],
        )
        turn2_final = ModelResponse(content="second-done", tool_calls=[])

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[turn1_llm, turn1_final, turn2_llm, turn2_final],
        ):
            # Turn 1: pause + resume
            paused1 = agent.run(message="do first")
            assert isinstance(paused1, tuple)
            assert paused1[1]["pending_tool_calls"][0]["id"] == "call_t1"

            out1 = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[ToolResultBlock(tool_use_id="call_t1", content="r1", is_error=False)],
                    ),
                ],
            )
            assert out1 == "first-done"

            # Turn 2: 新一轮对话仍能再次暂停再恢复
            paused2 = agent.run(message="do second")
            assert isinstance(paused2, tuple)
            assert paused2[1]["pending_tool_calls"][0]["id"] == "call_t2"

            out2 = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[ToolResultBlock(tool_use_id="call_t2", content="r2", is_error=False)],
                    ),
                ],
            )
            assert out2 == "second-done"


# -------------------------- Error paths -----------------------------------


class TestExternalToolErrorPaths:
    """场景 4：ToolResultBlock 带 is_error=True 时仍能正常恢复；LLM 在第二轮看到错误上下文。"""

    def test_is_error_true_tool_result_round_trips(self, global_storage):
        external = _make_external_tool("remote_fail")
        agent = _make_agent([external], global_storage=global_storage)

        first_response = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_fail", "remote_fail", {"q": "bad"})],
        )
        second_response = ModelResponse(content="apologies, error observed", tool_calls=[])

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response, second_response],
        ):
            paused = agent.run(message="run bad")
            assert isinstance(paused, tuple)

            final = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(
                                tool_use_id="call_fail",
                                content="network unreachable",
                                is_error=True,
                            ),
                        ],
                    ),
                ],
            )

        assert final == "apologies, error observed"

        # Tool result 应在 history 中保留 is_error=True 标记，供 LLM 使用。
        err_blocks = [
            b
            for m in agent.history
            if m.role == Role.TOOL
            for b in m.content
            if isinstance(b, ToolResultBlock) and b.tool_use_id == "call_fail"
        ]
        assert len(err_blocks) == 1
        assert err_blocks[0].is_error is True
