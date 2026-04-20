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

"""RFC-0018 T6: HTTP transport end-to-end integration tests.

测试策略：真实的 ``SSETransportServer`` + ``InMemoryDatabaseEngine``，通过
FastAPI ``TestClient`` 发起真实请求，仅在 ``LLMCaller.call_llm_async``
（类方法）层面 mock。覆盖：

- ``/query`` 端点在 external tool 暂停时返回 ``stop_reason`` +
  ``pending_tool_calls``
- 调用方再次 POST ``/query``（复用相同 ``user_id`` + ``session_id``，携带
  ``Role.TOOL`` 消息）可完成完整恢复流程
- ``/stream`` SSE 流里能观察到 ``EXTERNAL_TOOL_CALL`` 事件
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.tool.tool import Tool
from nexau.archs.transports.http import HTTPConfig, SSETransportServer

# ----------------------------- Helpers ------------------------------------


def _make_local_tool(name: str, result: str = "local-ok") -> Tool:
    def _impl(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"ok": True, "tool": name, "result": result}

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
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        implementation=None,
        kind="external",
    )


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> ModelToolCall:
    return ModelToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


@pytest.fixture
def engine() -> InMemoryDatabaseEngine:
    return InMemoryDatabaseEngine()


def _make_server(engine: InMemoryDatabaseEngine, tools: list[Tool]) -> SSETransportServer:
    """搭建一个带 external tool 的真实 SSE server（不实际监听端口）。"""
    agent_config = AgentConfig(
        name="rfc0018_http_e2e",
        system_prompt="You are a test assistant.",
        llm_config=LLMConfig(model="gpt-4o-mini"),
        tools=tools,
        tool_call_mode="openai",
    )
    return SSETransportServer(
        engine=engine,
        config=HTTPConfig(host="localhost", port=0),
        default_agent_config=agent_config,
    )


# ------------------------- /query pause / resume ---------------------------


class TestQueryEndpointExternalToolE2E:
    """通过 TestClient 跑完整的 /query 暂停 + 恢复流程。"""

    def test_query_pause_then_resume(self, engine):
        external = _make_external_tool("remote_search")
        server = _make_server(engine, [external])

        first_response = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_http_1", "remote_search", {"q": "ping"})],
        )
        second_response = ModelResponse(content="search result: pong", tool_calls=[])

        # LLMCaller 是每个请求 per-agent 重新构造的实例，patch 类方法覆盖所有实例。
        with patch(
            "nexau.archs.main_sub.execution.llm_caller.LLMCaller.call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response, second_response],
        ):
            client = TestClient(server.app)

            # 第一次 /query：触发 external tool 暂停。
            pause = client.post(
                "/query",
                json={
                    "messages": "please search",
                    "user_id": "u-http-1",
                    "session_id": "sess-http-1",
                },
            )
            assert pause.status_code == 200, pause.text
            data1 = pause.json()
            assert data1["status"] == "success"
            assert data1["response"] == ""
            assert data1["stop_reason"] == "EXTERNAL_TOOL_CALL"
            assert data1["pending_tool_calls"] == [
                {"id": "call_http_1", "name": "remote_search", "input": {"q": "ping"}},
            ]

            # 第二次 /query：携带 ToolResult 消息恢复（复用同 session）。
            resume_messages = [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_http_1",
                            "content": "pong",
                            "is_error": False,
                        },
                    ],
                },
            ]
            resume = client.post(
                "/query",
                json={
                    "messages": resume_messages,
                    "user_id": "u-http-1",
                    "session_id": "sess-http-1",
                },
            )
            assert resume.status_code == 200, resume.text
            data2 = resume.json()
            assert data2["status"] == "success"
            assert data2["response"] == "search result: pong"
            # 完成路径不带 stop_reason / pending_tool_calls（向后兼容）。
            assert data2.get("stop_reason") is None
            assert data2.get("pending_tool_calls") is None

    def test_query_pure_normal_completion_no_pause_fields(self, engine):
        """向后兼容：无 external tool 时 /query 响应不包含 stop_reason / pending_tool_calls。"""
        server = _make_server(engine, tools=[])

        response = ModelResponse(content="just text", tool_calls=[])

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.LLMCaller.call_llm_async",
            new_callable=AsyncMock,
            side_effect=[response],
        ):
            client = TestClient(server.app)
            r = client.post(
                "/query",
                json={"messages": "hello", "user_id": "u-normal", "session_id": "sess-normal"},
            )

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "success"
        assert data["response"] == "just text"
        assert data.get("stop_reason") is None
        assert data.get("pending_tool_calls") is None

    def test_query_mixed_local_external_returns_external_only_as_pending(self, engine):
        """混合调用：/query 的 pending_tool_calls 只包含 external，local 已执行。"""
        local = _make_local_tool("local_echo", result="local-done")
        external = _make_external_tool("remote_ping")
        server = _make_server(engine, [local, external])

        first_response = ModelResponse(
            content="",
            tool_calls=[
                _make_tool_call("call_local", "local_echo", {"x": "a"}),
                _make_tool_call("call_ext", "remote_ping", {"q": "b"}),
            ],
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.LLMCaller.call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response],
        ):
            client = TestClient(server.app)
            r = client.post(
                "/query",
                json={"messages": "mixed", "user_id": "u-mix", "session_id": "sess-mix"},
            )

        assert r.status_code == 200, r.text
        data = r.json()
        assert data["stop_reason"] == "EXTERNAL_TOOL_CALL"
        assert len(data["pending_tool_calls"]) == 1
        assert data["pending_tool_calls"][0]["name"] == "remote_ping"
        assert data["pending_tool_calls"][0]["id"] == "call_ext"


# --------------------------- /stream SSE event ----------------------------


class TestStreamEndpointExternalToolEvent:
    """/stream SSE 流中应发出 ``EXTERNAL_TOOL_CALL`` 事件。"""

    def test_stream_emits_external_tool_call_event(self, engine):
        external = _make_external_tool("remote_emit")
        server = _make_server(engine, [external])

        first_response = ModelResponse(
            content="",
            tool_calls=[_make_tool_call("call_stream_1", "remote_emit", {"q": "sse"})],
        )

        with patch(
            "nexau.archs.main_sub.execution.llm_caller.LLMCaller.call_llm_async",
            new_callable=AsyncMock,
            side_effect=[first_response],
        ):
            client = TestClient(server.app)
            with client.stream(
                "POST",
                "/stream",
                json={"messages": "stream me", "user_id": "u-sse", "session_id": "sess-sse"},
            ) as r:
                assert r.status_code == 200
                body_lines = [line for line in r.iter_lines() if line.startswith("data: ")]

        # 至少有一条 SSE data 事件
        assert len(body_lines) >= 1
        # 事件流中应能找到 EXTERNAL_TOOL_CALL
        found_external = False
        external_payload: dict[str, Any] | None = None
        for line in body_lines:
            evt = json.loads(line[len("data: ") :])
            if evt.get("type") == "EXTERNAL_TOOL_CALL":
                found_external = True
                external_payload = evt
                break
        assert found_external, f"Expected EXTERNAL_TOOL_CALL event in stream, got: {body_lines}"
        assert external_payload is not None
        # Payload 必须包含 pending tool call 的 id/name/input（RFC-0018 T5 合约）
        tool_calls = external_payload.get("tool_calls")
        assert isinstance(tool_calls, list) and len(tool_calls) == 1
        entry = tool_calls[0]
        assert entry["id"] == "call_stream_1"
        assert entry["name"] == "remote_emit"
        assert entry["input"] == {"q": "sse"}
