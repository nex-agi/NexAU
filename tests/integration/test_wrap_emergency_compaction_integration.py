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

"""Integration test for wrap fallback emergency compaction."""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import Mock

from nexau.archs.llm.llm_aggregators.events import CompactionFinishedEvent, CompactionStartedEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware
from nexau.archs.main_sub.execution.middleware.context_compaction import ContextCompactionMiddleware
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool


class _OverflowingChatModel:
    """Stateful mock model that overflows on large prompts before emergency summary appears."""

    def __init__(self, *, provider_limit_chars: int, tool_payload_chars: int) -> None:
        self.provider_limit_chars = provider_limit_chars
        self.tool_payload_chars = tool_payload_chars
        self.overflow_count = 0
        self.tool_call_count = 0

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content)

    def _build_response(
        self,
        *,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> Any:
        message = types.SimpleNamespace(content=content, tool_calls=tool_calls or [])
        usage = {
            "prompt_tokens": max(1, len(content) // 4),
            "completion_tokens": max(1, len(content) // 8),
            "total_tokens": max(2, len(content) // 3),
        }
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=usage,
        )

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        tools = kwargs.get("tools")
        joined = "\n".join(self._message_text(m) for m in messages if isinstance(m, dict))
        last_role = messages[-1].get("role") if messages and isinstance(messages[-1], dict) else ""

        # Emergency summarization calls from compaction strategy do not pass tools.
        if not tools:
            return self._build_response(
                content=(
                    "1. [EMERGENCY_SUMMARY] Keep core objective, hard constraints, and accepted decisions only. "
                    "2. Preserve exact file paths, API names, config keys, and errors needed for continuation. "
                    "3. Keep verified results and unresolved blockers, remove background narrative and examples. "
                    "4. Capture current state transitions and what was attempted successfully or failed. "
                    "5. Provide immediate next steps only, no repetition."
                )
            )

        if len(joined) > self.provider_limit_chars and "[EMERGENCY_SUMMARY]" not in joined:
            self.overflow_count += 1
            raise RuntimeError("maximum context length exceeded")

        if last_role == "tool":
            return self._build_response(content="done")

        self.tool_call_count += 1
        return self._build_response(
            tool_calls=[
                {
                    "id": f"call_{self.tool_call_count}",
                    "type": "function",
                    "function": {
                        "name": "big_blob_writer",
                        "arguments": json.dumps({"payload_size": self.tool_payload_chars}),
                    },
                }
            ]
        )


def test_wrap_emergency_compaction_triggers_with_large_tool_outputs(monkeypatch):
    """Large tool outputs + large user messages should trigger wrap emergency compaction."""
    model = _OverflowingChatModel(provider_limit_chars=6500, tool_payload_chars=2600)
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = model.create

    monkeypatch.setattr(Agent, "_initialize_openai_client", lambda _self: mock_client)

    events: list[Any] = []
    events_middleware = AgentEventsMiddleware(
        session_id="wrap_compaction_session",
        on_event=events.append,
    )
    compaction_middleware = ContextCompactionMiddleware(
        compaction_strategy="tool_result_compaction",
        auto_compact=True,
        emergency_compact_enabled=True,
        max_context_tokens=4000,
        threshold=0.75,
        keep_iterations=3,
    )

    tool = Tool(
        name="big_blob_writer",
        description="Write a large payload into tool result.",
        input_schema={
            "type": "object",
            "properties": {"payload_size": {"type": "integer"}},
            "required": ["payload_size"],
        },
        implementation=lambda payload_size: {"blob": "X" * int(payload_size)},
    )

    config = AgentConfig(
        name="wrap_compaction_agent",
        system_prompt="Use tool 'big_blob_writer' on every user request, then finish.",
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_chat_completion",
            stream=False,
            max_tokens=512,
        ),
        tools=[tool],
        middlewares=[compaction_middleware, events_middleware],
        max_iterations=8,
        retry_attempts=1,
        max_context_tokens=4000,
        overflow_max_tokens_stop_enabled=False,
        tool_call_mode="openai",
    )

    session_manager = SessionManager(engine=InMemoryDatabaseEngine())
    agent = Agent(
        config=config,
        session_manager=session_manager,
        user_id="integration_user",
        session_id="integration_session",
    )

    for i in range(4):
        response = agent.run(message=f"round={i} " + ("U" * 1200))
        assert isinstance(response, str)

    wrap_started = [e for e in events if isinstance(e, CompactionStartedEvent) and e.phase == "wrap_model_call" and e.mode == "emergency"]
    wrap_finished = [e for e in events if isinstance(e, CompactionFinishedEvent) and e.phase == "wrap_model_call" and e.mode == "emergency"]

    assert model.tool_call_count >= 4
    assert model.overflow_count >= 1
    assert len(wrap_started) >= 1
    assert len(wrap_finished) >= 1
    assert any(event.success for event in wrap_finished)
