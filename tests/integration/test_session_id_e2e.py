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

"""E2E tests: session_id injection through the full Agent → LLM stack.

Each test creates a real ``Agent`` with a specific ``api_type`` and
``session_id``, runs a single turn, and spies on the underlying SDK
method to verify that the correct provider fields carry the session_id.

- OpenAI Chat Completion → ``user`` field
- OpenAI Responses API  → ``safety_identifier``, ``prompt_cache_key``
- Anthropic Messages    → ``metadata.user_id`` field

Requires LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in the environment (or .env).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import dotenv
import pytest

from nexau import Agent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig

dotenv.load_dotenv()

_BASE_URL = os.getenv("LLM_BASE_URL", "")
_API_KEY = os.getenv("LLM_API_KEY", "")
_MODEL = os.getenv("LLM_MODEL", "")

_SESSION_ID = "e2e-agent-session-test-12345"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _BASE_URL or not _API_KEY or not _MODEL,
        reason="LLM_BASE_URL / LLM_API_KEY / LLM_MODEL not set",
    ),
]


def _anthropic_base_url() -> str:
    """Anthropic SDK adds /v1/messages itself; strip trailing /v1 if present."""
    base = _BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        return base[:-3]
    return base


def _make_agent(api_type: str, **llm_overrides: Any) -> Agent:
    """Create a real Agent with the given api_type and session_id."""
    base_url = _anthropic_base_url() if api_type == "anthropic_chat_completion" else _BASE_URL

    llm_config = LLMConfig(
        model=_MODEL,
        base_url=base_url,
        api_key=_API_KEY,
        temperature=0.0,
        max_tokens=30,
        api_type=api_type,
        **llm_overrides,
    )
    config = AgentConfig(
        name=f"session-id-e2e-{api_type}",
        system_prompt="You are a helpful assistant. Always reply in at most 5 words.",
        llm_config=llm_config,
    )
    return Agent(config=config, session_id=_SESSION_ID)


# ---------------------------------------------------------------------------
# OpenAI Chat Completion
# ---------------------------------------------------------------------------


class TestOpenAIChatCompletionSessionIdE2E:
    """E2E: Agent with openai_chat_completion → ``user`` field."""

    def test_session_id_in_user_field(self) -> None:
        agent = _make_agent("openai_chat_completion")

        # spy: 拦截 async client.chat.completions.create，捕获 kwargs
        # async/sync 技术债修复后 Agent.run() → execute_async → AsyncOpenAI
        captured_kwargs: dict[str, Any] = {}
        original_create = agent._async_openai_client.chat.completions.create

        async def spy_create(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return await original_create(**kwargs)

        with patch.object(agent._async_openai_client.chat.completions, "create", side_effect=spy_create):
            response = agent.run(message="Say hi")

        # 1. 验证 Agent 调用成功
        assert response, "Agent returned empty response"

        # 2. 验证 session_id 被注入到 user 字段
        assert "user" in captured_kwargs, f"'user' missing from kwargs: {list(captured_kwargs.keys())}"
        assert captured_kwargs["user"] == _SESSION_ID

        agent.sync_cleanup()


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------


class TestOpenAIResponsesSessionIdE2E:
    """E2E: Agent with openai_responses → ``safety_identifier``, ``prompt_cache_key``."""

    def test_session_id_in_responses_fields(self) -> None:
        agent = _make_agent("openai_responses", stream=True)

        captured_kwargs: dict[str, Any] = {}
        original_stream = agent._async_openai_client.responses.stream

        def spy_stream(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return original_stream(**kwargs)

        with patch.object(agent._async_openai_client.responses, "stream", side_effect=spy_stream):
            # 某些代理可能不支持全部字段，允许 API 层面失败
            try:
                response = agent.run(message="Say hi")
                if response:
                    assert len(response) > 0
            except Exception:
                pass

        # 核心断言：验证 kwargs 中的 session_id 字段（即使 API 返回错误）
        assert captured_kwargs, "spy was never called — Agent did not reach the SDK"

        # 'user' 不应被注入（已废弃，部分后端会拒绝）
        assert "user" not in captured_kwargs, f"'user' should NOT be injected for Responses API, got {captured_kwargs.get('user')!r}"
        assert captured_kwargs.get("safety_identifier") == _SESSION_ID, (
            f"'safety_identifier' mismatch: expected {_SESSION_ID!r}, got {captured_kwargs.get('safety_identifier')!r}"
        )

        # prompt_cache_key: 可能在 top-level 或 extra_body 中
        pck_top = captured_kwargs.get("prompt_cache_key")
        pck_extra = (captured_kwargs.get("extra_body") or {}).get("prompt_cache_key")
        actual_pck = pck_top or pck_extra
        # Agent.__init__ 会注入一个 UUID 作为 prompt_cache_key（优先级高于 session_id），
        # 所以这里只需验证 prompt_cache_key 存在且非空
        assert actual_pck, f"'prompt_cache_key' should be set, got top-level={pck_top!r}, extra_body={pck_extra!r}"

        agent.sync_cleanup()


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


class TestAnthropicSessionIdE2E:
    """E2E: Agent with anthropic_chat_completion → ``metadata.user_id``."""

    def test_session_id_in_metadata(self) -> None:
        agent = _make_agent("anthropic_chat_completion")

        captured_kwargs: dict[str, Any] = {}
        original_create = agent._async_openai_client.messages.create

        async def spy_create(**kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return await original_create(**kwargs)

        with patch.object(agent._async_openai_client.messages, "create", side_effect=spy_create):
            response = agent.run(message="Say hi")

        # 1. 验证 Agent 调用成功
        assert response, "Agent returned empty response"

        # 2. 验证 session_id 被注入到 metadata.user_id
        assert "metadata" in captured_kwargs, f"'metadata' missing from kwargs: {list(captured_kwargs.keys())}"
        metadata = captured_kwargs["metadata"]
        assert metadata.get("user_id") == _SESSION_ID, f"metadata.user_id mismatch: expected {_SESSION_ID!r}, got {metadata!r}"

        agent.sync_cleanup()
