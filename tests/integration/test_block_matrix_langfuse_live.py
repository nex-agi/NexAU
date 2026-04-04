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

"""Cross-api 4x4x4 live block matrix with Langfuse tracing.

RFC-0014: validate UMP history replay across provider boundaries.

This file defines a strict live matrix:
- 4 source API types
- 4 target API types
- 4 block families: text / reasoning / tool / image

Total: 64 live cases.

Each case:
1. Uses a real provider API for turn 1 (source API type).
2. Persists the resulting history in a real session.
3. Switches to a potentially different real provider API for turn 2 (target API type).
4. Transparently captures the actual target-side second-turn request payload.
5. Uploads a Langfuse trace with source/target/block metadata.

Important behavior rules are aligned with the unit matrix in
``tests/unit/test_two_turn_payload_matrix.py``:
- OpenAI Chat target preserves reasoning as ``reasoning_content``.
- OpenAI Responses target replays typed reasoning items.
- Anthropic target only emits signed ``thinking`` for Anthropic-origin reasoning.
  Other reasoning sources must downgrade to plain text.
- Gemini target emits reasoning as ``thought`` parts and preserves Gemini thought signatures.

Tool and image block cases keep target-side thinking enabled so that block replay is
validated under the more demanding mixed history + thinking path.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import dotenv
import httpx
import pytest
import requests

from nexau import Agent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer
from nexau.core.messages import ToolOutputImage

dotenv.load_dotenv()

BlockKind = Literal["text", "reasoning", "tool", "image"]
SourceName = Literal["completion", "responses", "claude", "gemini"]
TargetName = Literal["completion", "responses", "claude", "gemini"]

_TEXT_TASK_A = "Reply with exactly: BLOCK_TEXT_A"
_TEXT_TASK_B = "Repeat the prior assistant answer exactly, then append BLOCK_TEXT_B"

_REASONING_TASK_A = "Think step by step, but keep the visible answer short. Compute 17 * 19, then answer with exactly: REASONING_DONE 323"
_REASONING_TASK_B = "Use the previous hidden reasoning if available, then answer with exactly: REASONING_FOLLOWUP"

_TOOL_TASK_A = (
    "Think before acting. You MUST call the tool named echo_tool exactly once with "
    '{"text": "BLOCK_TOOL_TEXT"}. After the tool returns, reply with exactly TOOL_DONE and nothing else.'
)
_TOOL_TASK_B = "Use the previous tool result and any prior hidden reasoning, then reply with exactly TOOL_FOLLOWUP"

_IMAGE_TASK_A = (
    "Think before acting. You MUST call the tool named image_tool exactly once. "
    "After the tool returns, reply with exactly IMAGE_DONE and nothing else."
)
_IMAGE_TASK_B = (
    "Think before acting. You MUST call the tool named image_tool exactly once. "
    "Use the previous tool image, the newly returned tool image, and any prior hidden reasoning, "
    "then reply with exactly IMAGE_FOLLOWUP and nothing else."
)

_TEST_IMAGE_1 = Path("tests/20260327-131337.jpg")
_TEST_IMAGE_2 = Path("tests/20260327-131415.jpg")

_HOST = os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
_PK = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SK = os.getenv("LANGFUSE_SECRET_KEY", "")
_HAS_LANGFUSE = bool(_PK and _SK)


@dataclass(frozen=True, slots=True)
class ProviderEnv:
    api_type: str
    model: str
    base_url: str
    api_key: str


@dataclass(frozen=True, slots=True)
class MatrixCase:
    block: BlockKind
    source_name: SourceName
    target_name: TargetName
    source_env: ProviderEnv | None
    target_env: ProviderEnv | None
    first_prompt: str
    second_prompt: str
    target_thinking_enabled: bool

    @property
    def case_id(self) -> str:
        return f"{self.block}-{self.source_name}-to-{self.target_name}"


def _from_env(model_key: str, base_key: str, key_key: str, api_type: str) -> ProviderEnv | None:
    model = os.getenv(model_key, "")
    base_url = os.getenv(base_key, "")
    api_key = os.getenv(key_key, "")
    if not model or not base_url or not api_key or api_key == "test-key-not-used":
        return None
    return ProviderEnv(api_type=api_type, model=model, base_url=base_url, api_key=api_key)


_PROVIDER_ENVS: dict[TargetName, ProviderEnv | None] = {
    "completion": _from_env(
        "LIVE_OPENAI_CHAT_MODEL",
        "LIVE_OPENAI_CHAT_BASE_URL",
        "LIVE_OPENAI_CHAT_API_KEY",
        "openai_chat_completion",
    ),
    "responses": _from_env(
        "LIVE_OPENAI_RESPONSES_MODEL",
        "LIVE_OPENAI_RESPONSES_BASE_URL",
        "LIVE_OPENAI_RESPONSES_API_KEY",
        "openai_responses",
    ),
    "claude": _from_env(
        "LIVE_ANTHROPIC_MODEL",
        "LIVE_ANTHROPIC_BASE_URL",
        "LIVE_ANTHROPIC_API_KEY",
        "anthropic_chat_completion",
    ),
    "gemini": _from_env(
        "LIVE_GEMINI_MODEL",
        "LIVE_GEMINI_BASE_URL",
        "LIVE_GEMINI_API_KEY",
        "gemini_rest",
    ),
}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.external,
    pytest.mark.skipif(not _HAS_LANGFUSE, reason="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"),
]


def _jpeg_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _echo_tool() -> Tool:
    def impl(text: str) -> str:
        return f"TOOL_ECHO:{text}"

    return Tool(
        name="echo_tool",
        description="Echo a text payload.",
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        implementation=impl,
    )


def _image_tool(image_path: Path, *, caption: str) -> Tool:
    image_url = _jpeg_data_url(image_path)

    def impl() -> dict[str, Any]:
        return {
            "content": [
                {"type": "text", "text": caption},
                ToolOutputImage(image_url=image_url, detail="high"),
            ]
        }

    return Tool(
        name="image_tool",
        description="Return a caption plus one real image from a local test fixture.",
        input_schema={"type": "object", "properties": {}, "required": []},
        implementation=impl,
    )


class _RecordedPayloads:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def append(self, payload: dict[str, Any]) -> None:
        self.payloads.append(payload)


class _SyncOpenAIChatCompletionsRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._payloads.append(dict(kwargs))
        return self._inner.create(*args, **kwargs)


class _AsyncOpenAIChatCompletionsRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        self._payloads.append(dict(kwargs))
        return await self._inner.create(*args, **kwargs)


class _SyncOpenAIChatRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self.completions = _SyncOpenAIChatCompletionsRecorder(inner.completions, payloads)


class _AsyncOpenAIChatRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self.completions = _AsyncOpenAIChatCompletionsRecorder(inner.completions, payloads)


class _SyncOpenAIResponsesRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._payloads.append(dict(kwargs))
        return self._inner.create(*args, **kwargs)


class _AsyncOpenAIResponsesRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        self._payloads.append(dict(kwargs))
        return await self._inner.create(*args, **kwargs)


class _SyncOpenAIClientRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self.chat = _SyncOpenAIChatRecorder(inner.chat, payloads)
        self.responses = _SyncOpenAIResponsesRecorder(inner.responses, payloads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AsyncOpenAIClientRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self.chat = _AsyncOpenAIChatRecorder(inner.chat, payloads)
        self.responses = _AsyncOpenAIResponsesRecorder(inner.responses, payloads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _SyncAnthropicMessagesRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    def create(self, *args: Any, **kwargs: Any) -> Any:
        self._payloads.append(dict(kwargs))
        return self._inner.create(*args, **kwargs)


class _AsyncAnthropicMessagesRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        self._payloads.append(dict(kwargs))
        return await self._inner.create(*args, **kwargs)


class _SyncAnthropicClientRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self.messages = _SyncAnthropicMessagesRecorder(inner.messages, payloads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AsyncAnthropicClientRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self.messages = _AsyncAnthropicMessagesRecorder(inner.messages, payloads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _GeminiRequestsPostSpy:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._original = requests.post

    def __call__(self, url: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"url": url, **dict(kwargs)})
        return self._original(url, *args, **kwargs)


def _install_runtime_capture(agent: Agent, api_type: str, payloads: _RecordedPayloads) -> None:
    if api_type in {"openai_chat_completion", "openai_responses"}:
        if agent.openai_client is not None:
            agent.openai_client = _SyncOpenAIClientRecorder(agent.openai_client, payloads)
        if getattr(agent, "_async_openai_client", None) is not None:
            agent._async_openai_client = _AsyncOpenAIClientRecorder(agent._async_openai_client, payloads)
        agent.executor.llm_caller.openai_client = agent.openai_client
        agent.executor.llm_caller.async_openai_client = getattr(agent, "_async_openai_client", None)
        return

    if api_type == "anthropic_chat_completion":
        if agent.openai_client is not None:
            agent.openai_client = _SyncAnthropicClientRecorder(agent.openai_client, payloads)
        if getattr(agent, "_async_openai_client", None) is not None:
            agent._async_openai_client = _AsyncAnthropicClientRecorder(agent._async_openai_client, payloads)
        agent.executor.llm_caller.openai_client = agent.openai_client
        agent.executor.llm_caller.async_openai_client = getattr(agent, "_async_openai_client", None)


@pytest.fixture
def session_manager() -> SessionManager:
    return SessionManager(engine=InMemoryDatabaseEngine())


def _make_langfuse_tracer(*, session_id: str, case: MatrixCase) -> LangfuseTracer:
    return LangfuseTracer(
        public_key=_PK,
        secret_key=_SK,
        host=_HOST,
        session_id=session_id,
        tags=[
            "rfc-0014",
            "block-matrix",
            f"block:{case.block}",
            f"source:{case.source_name}",
            f"target:{case.target_name}",
        ],
        metadata={
            "test": "block_matrix_live",
            "block": case.block,
            "source_api_type": case.source_name,
            "target_api_type": case.target_name,
            "mode": "real_api_cross_provider_with_transparent_payload_capture",
        },
        debug=True,
    )


def _build_llm_kwargs(env: ProviderEnv, *, thinking_enabled: bool) -> dict[str, Any]:
    llm_kwargs: dict[str, Any] = {
        "model": env.model,
        "base_url": env.base_url,
        "api_key": env.api_key,
        "api_type": env.api_type,
        "temperature": 0.0,
        "max_tokens": 16384,
        "max_retries": 1,
        "stream": False,
    }
    if thinking_enabled:
        if env.api_type == "openai_chat_completion":
            llm_kwargs["reasoning_effort"] = "high"
        elif env.api_type == "openai_responses":
            llm_kwargs["reasoning"] = {"effort": "high"}
            llm_kwargs["include"] = ["reasoning.encrypted_content"]
        elif env.api_type == "anthropic_chat_completion":
            # Anthropic 要求 thinking 开启时 temperature 必须为 1
            llm_kwargs["temperature"] = 1.0
            llm_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8192}
        elif env.api_type == "gemini_rest":
            llm_kwargs["thinkingConfig"] = {"includeThoughts": True, "thinkingBudget": 8192}
    return llm_kwargs


def _make_agent(
    env: ProviderEnv,
    session_manager: SessionManager,
    session_id: str,
    *,
    case: MatrixCase,
    thinking_enabled: bool,
    tools: list[Tool] | None = None,
) -> Agent:
    tracer = _make_langfuse_tracer(session_id=session_id, case=case)
    return Agent(
        config=AgentConfig(
            name=f"block-matrix-{case.case_id}",
            system_prompt="You are a precise assistant. Follow the instructions exactly.",
            llm_config=LLMConfig(**_build_llm_kwargs(env, thinking_enabled=thinking_enabled)),
            tools=tools or [],
            tracers=[tracer],
        ),
        session_manager=session_manager,
        user_id="block-matrix-user",
        session_id=session_id,
    )


def _fetch_traces_for_session(session_id: str) -> list[dict[str, Any]]:
    for attempt in range(10):
        response = requests.get(
            f"{_HOST}/api/public/traces",
            params={"sessionId": session_id},
            auth=(_PK, _SK),
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json().get("data", [])
            if data:
                return data
        if attempt < 9:
            time.sleep(2)
    pytest.fail(f"No Langfuse traces found for session_id={session_id}")


def _assert_langfuse_uploaded(session_id: str, *, case: MatrixCase) -> None:
    traces = _fetch_traces_for_session(session_id)
    assert traces, f"No Langfuse traces uploaded for session={session_id}"
    matched = []
    for trace in traces:
        metadata = trace.get("metadata")
        tags = trace.get("tags")
        trace_session_id = trace.get("sessionId") or trace.get("session_id")
        if not isinstance(metadata, dict):
            continue
        if trace_session_id != session_id:
            continue
        if metadata.get("block") != case.block:
            continue
        if metadata.get("source_api_type") != case.source_name:
            continue
        if metadata.get("target_api_type") != case.target_name:
            continue
        if not isinstance(tags, list):
            continue
        if f"block:{case.block}" not in tags or f"source:{case.source_name}" not in tags or f"target:{case.target_name}" not in tags:
            continue
        matched.append(trace)
    assert matched, f"Langfuse trace metadata/tags missing for session={session_id} case={case.case_id}"


def _looks_like_unavailable_provider(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "model_not_found",
            "无可用渠道",
            "no available distributor",
            "no available channel",
            "model is not available",
            "unsupported model",
        )
    )


def _run_case_and_capture(case: MatrixCase, session_manager: SessionManager) -> tuple[str, dict[str, Any], str]:
    assert case.source_env is not None
    assert case.target_env is not None

    session_id = f"block-matrix-{case.case_id}-{uuid.uuid4().hex[:8]}"
    source_tools: list[Tool] = []
    target_tools: list[Tool] = []
    if case.block == "tool":
        source_tools = [_echo_tool()]
        target_tools = [_echo_tool()]
    elif case.block == "image":
        source_tools = [_image_tool(_TEST_IMAGE_1, caption="IMAGE_CAPTION_1")]
        target_tools = [_image_tool(_TEST_IMAGE_2, caption="IMAGE_CAPTION_2")]

    source_agent = _make_agent(
        case.source_env,
        session_manager,
        session_id,
        case=case,
        thinking_enabled=(case.block in {"reasoning", "tool", "image"}),
        tools=source_tools,
    )
    first = source_agent.run(message=case.first_prompt)
    assert first

    target_agent = _make_agent(
        case.target_env,
        session_manager,
        session_id,
        case=case,
        thinking_enabled=case.target_thinking_enabled,
        tools=target_tools,
    )

    if case.target_env.api_type == "gemini_rest":
        gemini_requests_spy = _GeminiRequestsPostSpy()
        gemini_httpx_calls: list[dict[str, Any]] = []
        original_async_post = httpx.AsyncClient.post

        async def _capture_gemini_httpx_post(
            client: httpx.AsyncClient,
            url: str,
            *args: Any,
            **kwargs: Any,
        ) -> httpx.Response:
            gemini_httpx_calls.append({"url": url, **dict(kwargs)})
            return await original_async_post(client, url, *args, **kwargs)

        with (
            patch("requests.post", side_effect=gemini_requests_spy),
            patch.object(
                httpx.AsyncClient,
                "post",
                new=_capture_gemini_httpx_post,
            ),
        ):
            second = target_agent.run(message=case.second_prompt)
        assert second
        _assert_langfuse_uploaded(session_id, case=case)
        if gemini_httpx_calls:
            return str(second), gemini_httpx_calls[-1], session_id
        if gemini_requests_spy.calls:
            return str(second), gemini_requests_spy.calls[-1], session_id
        raise AssertionError(f"No outbound Gemini payload captured for case={case.case_id}")

    payloads = _RecordedPayloads()
    _install_runtime_capture(target_agent, case.target_env.api_type, payloads)
    second = target_agent.run(message=case.second_prompt)
    assert second
    _assert_langfuse_uploaded(session_id, case=case)
    assert payloads.payloads, f"No outbound payload captured for case={case.case_id}"
    return str(second), payloads.payloads[-1], session_id


def _assistant_text_for_source(source_name: SourceName, block: BlockKind) -> str:
    if block == "text":
        return "BLOCK_TEXT_A"
    if block == "reasoning":
        return "REASONING_DONE 323"
    if block == "tool":
        return "TOOL_DONE"
    if block == "image":
        return "IMAGE_DONE"
    raise AssertionError(f"Unexpected block {block}")


def _source_reasoning_text(source_name: SourceName, block: BlockKind) -> str | None:
    if block not in {"reasoning", "tool", "image"}:
        return None
    if source_name == "completion":
        return None
    if source_name == "responses":
        return None
    if source_name == "claude":
        return None
    if source_name == "gemini":
        return None
    raise AssertionError(f"Unexpected source {source_name}")


def _assert_text_block(case: MatrixCase, payload: dict[str, Any]) -> None:
    assistant_text = _assistant_text_for_source(case.source_name, "text")
    if case.target_name == "completion":
        assistant_messages = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert any(assistant_text in str(m.get("content", "")) for m in assistant_messages)
        return
    if case.target_name == "responses":
        items = payload.get("input", [])
        assert any(
            item.get("type") == "message"
            and item.get("role") == "assistant"
            and any(block.get("text") == assistant_text for block in item.get("content", []))
            for item in items
        )
        return
    if case.target_name == "claude":
        assistant_messages = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert any(
            any(block.get("type") == "text" and block.get("text") == assistant_text for block in msg.get("content", []))
            for msg in assistant_messages
        )
        return
    body = payload.get("json", {})
    contents = body.get("contents", [])
    assert any(part.get("text") == assistant_text for content in contents for part in content.get("parts", []))


def _assert_reasoning_block(case: MatrixCase, payload: dict[str, Any]) -> None:
    assistant_text = _assistant_text_for_source(case.source_name, "reasoning")

    if case.target_name == "completion":
        assert payload.get("reasoning_effort") == "high"
        assistant_messages = [m for m in payload["messages"] if m.get("role") == "assistant"]
        assert any(assistant_text in str(m.get("content", "")) for m in assistant_messages)
        return

    if case.target_name == "responses":
        assert payload.get("reasoning", {}).get("effort") == "high"
        items = payload.get("input", [])
        reasoning_items = [item for item in items if item.get("type") == "reasoning"]
        if case.source_name == "responses":
            assert reasoning_items, "Responses-origin live history should preserve a typed reasoning replay item"
        elif not reasoning_items:
            pytest.skip(f"Upstream live source {case.source_name} did not expose a replayable reasoning artifact for target responses")
        assert any(
            item.get("type") == "message"
            and item.get("role") == "assistant"
            and any(block.get("text") == assistant_text for block in item.get("content", []))
            for item in items
        )
        return

    if case.target_name == "claude":
        assert payload.get("thinking", {}).get("type") == "enabled"
        assistant_messages = [m for m in payload.get("messages", []) if m.get("role") == "assistant"]
        assert assistant_messages
        if case.source_name == "claude":
            assert any(any(block.get("type") == "thinking" for block in msg.get("content", [])) for msg in assistant_messages)
        else:
            assert not any(any(block.get("type") == "thinking" for block in msg.get("content", [])) for msg in assistant_messages)
            assert any(any(block.get("type") == "text" for block in msg.get("content", [])) for msg in assistant_messages)
        return

    body = payload.get("json", {})
    assert body.get("generationConfig", {}).get("thinkingConfig", {}).get("includeThoughts") is True
    contents = body.get("contents", [])
    assert any(any(part.get("thought") is True for part in content.get("parts", [])) for content in contents)


def _assert_tool_block(case: MatrixCase, payload: dict[str, Any]) -> None:
    if case.target_name == "completion":
        assert payload.get("reasoning_effort") == "high"
        messages = payload["messages"]
        assert any(msg.get("role") == "assistant" and msg.get("tool_calls") for msg in messages)
        assert any(msg.get("role") == "tool" and "TOOL_ECHO:BLOCK_TOOL_TEXT" in str(msg.get("content", "")) for msg in messages)
        return

    if case.target_name == "responses":
        assert payload.get("reasoning", {}).get("effort") == "high"
        items = payload.get("input", [])
        assert any(item.get("type") == "function_call" for item in items)
        assert any(item.get("type") == "function_call_output" for item in items)
        return

    if case.target_name == "claude":
        assert payload.get("thinking", {}).get("type") == "enabled"
        messages = payload.get("messages", [])
        assert any(
            any(block.get("type") == "tool_use" for block in msg.get("content", [])) for msg in messages if msg.get("role") == "assistant"
        )
        assert any(
            any(block.get("type") == "tool_result" for block in msg.get("content", [])) for msg in messages if msg.get("role") == "user"
        )
        return

    body = payload.get("json", {})
    assert body.get("generationConfig", {}).get("thinkingConfig", {}).get("includeThoughts") is True
    contents = body.get("contents", [])
    assert any(any("functionCall" in part for part in content.get("parts", [])) for content in contents)
    assert any(any("functionResponse" in part for part in content.get("parts", [])) for content in contents)


def _assert_image_block(case: MatrixCase, payload: dict[str, Any]) -> None:
    if case.target_name == "completion":
        assert payload.get("reasoning_effort") == "high"
        messages = payload["messages"]
        assert any(msg.get("role") == "tool" and "IMAGE_CAPTION_2" in str(msg.get("content", "")) for msg in messages)
        assert any(
            msg.get("role") == "user"
            and isinstance(msg.get("content"), list)
            and any(part.get("type") == "image_url" for part in msg.get("content", []))
            for msg in messages
        )
        return

    if case.target_name == "responses":
        assert payload.get("reasoning", {}).get("effort") == "high"
        items = payload.get("input", [])
        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert outputs
        assert any(any(part.get("type") == "input_image" for part in item.get("output", [])) for item in outputs)
        assert any(
            any(part.get("type") == "input_text" and part.get("text") == "IMAGE_CAPTION_2" for part in item.get("output", []))
            for item in outputs
        )
        return

    if case.target_name == "claude":
        assert payload.get("thinking", {}).get("type") == "enabled"
        messages = payload.get("messages", [])
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        assert any(any(block.get("type") == "tool_result" for block in msg.get("content", [])) for msg in user_messages)
        assert any(any(block.get("type") == "image" for block in msg.get("content", [])) for msg in user_messages)
        return

    body = payload.get("json", {})
    assert body.get("generationConfig", {}).get("thinkingConfig", {}).get("includeThoughts") is True
    contents = body.get("contents", [])
    assert any(any("functionResponse" in part for part in content.get("parts", [])) for content in contents)


def _assert_payload(case: MatrixCase, payload: dict[str, Any]) -> None:
    if case.block == "text":
        _assert_text_block(case, payload)
        return
    if case.block == "reasoning":
        _assert_reasoning_block(case, payload)
        return
    if case.block == "tool":
        _assert_tool_block(case, payload)
        return
    if case.block == "image":
        _assert_image_block(case, payload)
        return
    raise AssertionError(f"Unexpected block kind: {case.block}")


_SOURCE_NAMES: list[SourceName] = ["completion", "responses", "claude", "gemini"]
_TARGET_NAMES: list[TargetName] = ["completion", "responses", "claude", "gemini"]
_BLOCKS: list[BlockKind] = ["text", "reasoning", "tool", "image"]
_PROMPTS: dict[BlockKind, tuple[str, str]] = {
    "text": (_TEXT_TASK_A, _TEXT_TASK_B),
    "reasoning": (_REASONING_TASK_A, _REASONING_TASK_B),
    "tool": (_TOOL_TASK_A, _TOOL_TASK_B),
    "image": (_IMAGE_TASK_A, _IMAGE_TASK_B),
}

_CASES: list[MatrixCase] = []
for block in _BLOCKS:
    for source_name in _SOURCE_NAMES:
        for target_name in _TARGET_NAMES:
            first_prompt, second_prompt = _PROMPTS[block]
            _CASES.append(
                MatrixCase(
                    block=block,
                    source_name=source_name,
                    target_name=target_name,
                    source_env=_PROVIDER_ENVS[source_name],
                    target_env=_PROVIDER_ENVS[target_name],
                    first_prompt=first_prompt,
                    second_prompt=second_prompt,
                    target_thinking_enabled=(block in {"reasoning", "tool", "image"}),
                )
            )


@pytest.mark.parametrize("case", _CASES, ids=[case.case_id for case in _CASES])
def test_live_block_matrix(case: MatrixCase, session_manager: SessionManager) -> None:
    if case.source_env is None:
        pytest.skip(f"LIVE env vars not set for source api type {case.source_name}")
    if case.target_env is None:
        pytest.skip(f"LIVE env vars not set for target api type {case.target_name}")

    try:
        second_text, payload, session_id = _run_case_and_capture(case, session_manager)
    except Exception as exc:
        if _looks_like_unavailable_provider(exc):
            pytest.skip(f"Provider unavailable for case={case.case_id}: {exc}")
        raise

    assert second_text.strip(), f"Empty second-turn response for case={case.case_id}"
    _assert_payload(case, payload)
    print(f"\n✅ case={case.case_id} session={session_id}")
