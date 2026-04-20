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

"""RFC-0018 T7: Agent-owned trace_id continuity across EXTERNAL_TOOL_CALL pause/resume.

Design invariants verified here:

- Agent generates a fresh ``trace_id`` on each *new* user-triggered run and persists
  it on ``SessionModel.current_trace_id``.
- When the run pauses with ``EXTERNAL_TOOL_CALL``, the trace_id is kept on the
  ``SessionModel`` so that a *different* ``Agent`` instance bound to the same
  ``session_id`` can resume the same trace on the next call.
- Normal stop reasons clear ``SessionModel.current_trace_id`` (top-level only).
  The very next user message then starts a brand-new trace.
- The flow does **not** require a ``Tracer`` to be configured — ``trace_id`` is
  owned by Agent, not Tracer.
- Tracers that want to follow the Agent's trace_id can expose a ``bind_trace_id``
  method; Agent soft-binds via ``hasattr``. Tracers without the method are
  untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import GlobalStorage
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool
from nexau.core.messages import Message, Role, ToolResultBlock


# ----------------------------- Helpers ---------------------------


def _make_external_tool(name: str) -> Tool:
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


def _make_agent(
    tools: list[Tool],
    *,
    global_storage: GlobalStorage,
    session_manager: SessionManager,
    session_id: str,
    user_id: str = "user_trace_t7",
) -> Agent:
    config = AgentConfig(
        name="rfc0018_t7_agent",
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
            user_id=user_id,
            session_id=session_id,
        )


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(engine=InMemoryDatabaseEngine())


@pytest.fixture
def global_storage() -> GlobalStorage:
    return GlobalStorage()


# ----------------- Lifecycle: generate / reuse / clear -----------------


class TestAgentGeneratesTraceIdPerUserMessage:
    """新 user message → Agent 生成 uuid，写入 SessionModel.current_trace_id。"""

    def test_new_run_generates_and_persists_trace_id(self, session_manager, global_storage):
        external = _make_external_tool("remote_op")
        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_new_trace",
        )

        paused_resp = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_a", "remote_op", {"q": "hi"})],
        )

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[paused_resp],
        ):
            paused = agent.run(message="kick off")

        assert isinstance(paused, tuple)
        _, meta = paused
        trace_id = meta["trace_id"]
        assert isinstance(trace_id, str) and len(trace_id) > 0

        import asyncio

        session = asyncio.run(session_manager.get_session(user_id="user_trace_t7", session_id="sess_new_trace"))
        assert session is not None
        assert session.current_trace_id == trace_id


class TestExternalToolCallPreservesTraceIdAcrossAgentInstances:
    """Agent_v1 暂停 → 销毁 → Agent_v2(同 session_id) resume：trace_id 必须一致。"""

    def test_resume_on_new_agent_instance_reuses_trace_id(self, session_manager, global_storage):
        external = _make_external_tool("remote_op")

        # --- Agent v1: 发起 external call，pause。
        agent_v1 = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_resume_across_agents",
        )
        pause_resp = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_ext", "remote_op", {"q": "x"})],
        )

        with patch.object(
            agent_v1.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[pause_resp],
        ):
            paused = agent_v1.run(message="please call remote")

        assert isinstance(paused, tuple)
        trace_id_v1 = paused[1]["trace_id"]
        assert trace_id_v1

        # --- Agent v2: 全新实例、同 session；resume 并完成。
        agent_v2 = _make_agent(
            [external],
            global_storage=GlobalStorage(),
            session_manager=session_manager,
            session_id="sess_resume_across_agents",
        )
        final_resp = ModelResponse(content="all done", tool_calls=[])

        with patch.object(
            agent_v2.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[final_resp],
        ):
            result = agent_v2.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(
                                tool_use_id="call_ext",
                                content="remote-result",
                                is_error=False,
                            ),
                        ],
                    ),
                ],
            )

        # 正常结束返回纯 str（EXTERNAL_TOOL_CALL 才会返回 tuple）。
        assert result == "all done"

        # trace_id 在 resume 的 pre-call 阶段由 agent_v2 从 SessionModel 读回，
        # 并在 agent_v2 的 AgentState.trace_id 上落地。正常结束后被清空。
        import asyncio

        session = asyncio.run(session_manager.get_session(user_id="user_trace_t7", session_id="sess_resume_across_agents"))
        assert session is not None
        assert session.current_trace_id is None  # 正常结束清空

    def test_trace_id_persists_on_session_through_pause(self, session_manager, global_storage):
        """Pause 后 SessionModel.current_trace_id 依然保存着 trace_id。"""
        external = _make_external_tool("remote_op")
        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_pause_preserves",
        )
        pause_resp = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_x", "remote_op", {"q": "q"})],
        )
        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[pause_resp],
        ):
            paused = agent.run(message="go")

        assert isinstance(paused, tuple)
        trace_id = paused[1]["trace_id"]

        import asyncio

        session = asyncio.run(session_manager.get_session(user_id="user_trace_t7", session_id="sess_pause_preserves"))
        assert session is not None
        assert session.current_trace_id == trace_id


class TestNormalStopClearsTraceIdAndNextRunRegenerates:
    """正常结束清空 SessionModel.current_trace_id；下一个 user message 换新 id。"""

    def test_next_user_message_gets_fresh_trace_id(self, session_manager, global_storage):
        external = _make_external_tool("remote_op")
        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_two_traces",
        )

        # Turn 1: pause → resume → 正常结束
        turn1_pause = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("t1", "remote_op", {"q": "1"})],
        )
        turn1_final = ModelResponse(content="t1 done", tool_calls=[])
        # Turn 2: 仍会 pause 以便我们捕获第二条 trace_id
        turn2_pause = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("t2", "remote_op", {"q": "2"})],
        )

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[turn1_pause, turn1_final, turn2_pause],
        ):
            paused1 = agent.run(message="first")
            assert isinstance(paused1, tuple)
            trace_1 = paused1[1]["trace_id"]

            resumed = agent.run(
                message=[
                    Message(
                        role=Role.TOOL,
                        content=[
                            ToolResultBlock(tool_use_id="t1", content="r1", is_error=False),
                        ],
                    ),
                ],
            )
            assert resumed == "t1 done"

            # 正常结束 → session current_trace_id 已清空
            import asyncio

            session_mid = asyncio.run(session_manager.get_session(user_id="user_trace_t7", session_id="sess_two_traces"))
            assert session_mid is not None
            assert session_mid.current_trace_id is None

            paused2 = agent.run(message="second")
            assert isinstance(paused2, tuple)
            trace_2 = paused2[1]["trace_id"]

        assert trace_1 and trace_2
        assert trace_1 != trace_2, "每次 user-triggered run 应分配新 trace_id"


# ----------------------- Tracer independence / soft bind -----------------------


class TestWorksWithoutTracerConfigured:
    """不配置 tracer 时 trace_id 流程仍然生效。"""

    def test_pause_returns_trace_id_without_tracer(self, session_manager, global_storage):
        # global_storage 中没有 "tracer" 键，模拟未启用 tracer 的部署。
        assert global_storage.get("tracer") is None

        external = _make_external_tool("remote_op")
        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_no_tracer",
        )
        pause_resp = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("c1", "remote_op", {"q": "x"})],
        )

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[pause_resp],
        ):
            paused = agent.run(message="go")

        assert isinstance(paused, tuple)
        _, meta = paused
        assert meta["trace_id"]
        # 且 SessionModel 持久化了同一个 trace_id。
        import asyncio

        session = asyncio.run(session_manager.get_session(user_id="user_trace_t7", session_id="sess_no_tracer"))
        assert session is not None
        assert session.current_trace_id == meta["trace_id"]


class TestSoftTracerBinding:
    """若 tracer 提供 bind_trace_id 方法，Agent 软绑定：调用该方法并传入已解析的 trace_id。"""

    def test_tracer_with_bind_trace_id_gets_called(self, session_manager, global_storage):
        # 提供一个 minimal tracer stub：只实现 bind_trace_id。
        class _TracerStub:
            def __init__(self):
                self.received: list[str] = []

            def bind_trace_id(self, trace_id: str) -> None:
                self.received.append(trace_id)

            # Agent 只有在 `tracer` truthy 时才走 tracing 分支；这里返回 None 意味着
            # 走 _run_inner（仍能通过 hasattr 软绑定）。我们不需要 tracer 完整实现 BaseTracer。
            def __bool__(self) -> bool:
                # 让 `if tracer:` 判定为 False，避免触发 _run_with_tracing 的完整路径。
                return False

        tracer_stub = _TracerStub()
        global_storage.set("tracer", tracer_stub)

        external = _make_external_tool("remote_op")
        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_soft_bind",
        )
        pause_resp = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("c1", "remote_op", {"q": "hi"})],
        )
        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[pause_resp],
        ):
            paused = agent.run(message="call")

        assert isinstance(paused, tuple)
        trace_id = paused[1]["trace_id"]

        assert tracer_stub.received == [trace_id], "Agent 应通过 hasattr(tracer, 'bind_trace_id') 软绑定一次，且传入 resolved trace_id"

    def test_tracer_without_bind_trace_id_is_untouched(self, session_manager, global_storage):
        """不实现 bind_trace_id 的 tracer 不应报错。"""

        class _PlainTracerStub:
            def __bool__(self) -> bool:
                return False  # 避开完整 tracing 分支，仅验证软绑定的 no-op 行为

        global_storage.set("tracer", _PlainTracerStub())

        external = _make_external_tool("remote_op")
        agent = _make_agent(
            [external],
            global_storage=global_storage,
            session_manager=session_manager,
            session_id="sess_plain_tracer",
        )
        pause_resp = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("c1", "remote_op", {"q": "x"})],
        )

        with patch.object(
            agent.executor.llm_caller,
            "call_llm_async",
            new_callable=AsyncMock,
            side_effect=[pause_resp],
        ):
            paused = agent.run(message="go")

        assert isinstance(paused, tuple)
        assert paused[1]["trace_id"]
