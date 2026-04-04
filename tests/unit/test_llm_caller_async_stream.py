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

"""Coverage tests for async streaming and _prepare_responses_api_input tool-call/reasoning paths.

Targets uncovered lines:
- 1724~1769: call_llm_with_openai_chat_completion_async stream (with/without tracing)
- 1851~1886: call_llm_with_anthropic_chat_completion_async stream (with/without tracing)
- 1947~1954: call_llm_with_openai_responses_async non-stream (with/without tracing)
- 1961~1979: call_llm_with_openai_responses_async stream with tracing
- 1982~1992: call_llm_with_openai_responses_async stream without tracing
- 2131~2168: _prepare_responses_api_input tool_calls reconstruction + reasoning items
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.hooks import MiddlewareManager, ModelCallParams
from nexau.archs.main_sub.execution.llm_caller import (
    call_llm_with_anthropic_chat_completion_async,
    call_llm_with_openai_chat_completion_async,
    call_llm_with_openai_responses_async,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.core.messages import Message
from nexau.core.serializers.openai_responses import (
    prepare_openai_responses_api_input as _prepare_responses_api_input,
)

# ---------------------------------------------------------------------------
# Helpers: async iterator / context-manager fakes for streaming
# ---------------------------------------------------------------------------


class _AsyncChunkIterator:
    """Async iterator that yields pre-defined chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._index = 0

    def __aiter__(self) -> _AsyncChunkIterator:
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk

    async def __aenter__(self) -> _AsyncChunkIterator:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _AsyncStreamContextManager:
    """Awaitable that returns an async context-manager yielding an async iterator."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __await__(self):  # type: ignore[override]
        return self._return_self().__await__()

    async def _return_self(self) -> _AsyncStreamContextManager:
        return self

    async def __aenter__(self) -> _AsyncChunkIterator:
        return _AsyncChunkIterator(self._chunks)

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake tracer helpers
# ---------------------------------------------------------------------------


class _FakeSpan:
    pass


class _FakeTraceContext:
    """Mimics TraceContext as a context manager that records calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._outputs: Any = None
        self._attributes: dict[str, Any] = {}

    def __enter__(self) -> _FakeTraceContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_outputs(self, outputs: Any) -> None:
        self._outputs = outputs

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        self._attributes.update(attrs)


class _FakeTracer(BaseTracer):
    """Minimal concrete BaseTracer for testing — no real tracing."""

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        return Span(id="fake-span", name=name, type=span_type)

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_chat_stream_chunks() -> list[dict[str, Any]]:
    """Minimal OpenAI chat-completion stream chunks."""
    return [
        {
            "model": "gpt-4o-mini",
            "choices": [{"delta": {"role": "assistant", "content": "Hello"}}],
        },
        {
            "model": "gpt-4o-mini",
            "choices": [{"delta": {"content": " world"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    ]


@pytest.fixture
def anthropic_stream_events() -> list[dict[str, Any]]:
    """Minimal Anthropic streaming events."""
    return [
        {
            "type": "message_start",
            "message": {"role": "assistant", "model": "claude-3"},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi there"},
        },
        {
            "type": "content_block_stop",
            "index": 0,
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 3},
        },
        {
            "type": "message_stop",
            "message": {},
        },
    ]


@pytest.fixture
def openai_responses_stream_events() -> list[dict[str, Any]]:
    """Minimal OpenAI Responses API stream events."""
    return [
        {
            "type": "response.created",
            "response": {"id": "resp-1", "model": "gpt-4o-mini"},
        },
        {
            "type": "response.output_item.added",
            "item": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg-1",
            "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg-1",
            "content_index": 0,
            "delta": "Hello!",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello!"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        },
    ]


# ===========================================================================
# Tests: call_llm_with_openai_chat_completion_async — stream paths (L1724-1769)
# ===========================================================================


class TestOpenAIChatCompletionAsyncStream:
    """Cover stream paths with and without tracing, shutdown, and _process_stream_chunk filtering."""

    @pytest.mark.anyio
    async def test_stream_without_tracing(self, openai_chat_stream_chunks: list[dict[str, Any]]) -> None:
        """Lines 1757-1767: stream path without tracer."""
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_AsyncChunkIterator(openai_chat_stream_chunks))

        result = await call_llm_with_openai_chat_completion_async(
            client,
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
        )

        assert isinstance(result, ModelResponse)
        assert "Hello world" in (result.content or "")

    @pytest.mark.anyio
    async def test_stream_with_tracing(self, openai_chat_stream_chunks: list[dict[str, Any]]) -> None:
        """Lines 1732-1755: stream path with tracer + get_current_span."""
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_AsyncChunkIterator(openai_chat_stream_chunks))
        tracer = _FakeTracer()

        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.get_current_span",
                return_value=_FakeSpan(),
            ),
            patch(
                "nexau.archs.main_sub.execution.llm_caller.TraceContext",
                side_effect=lambda *a, **kw: _FakeTraceContext(*a, **kw),
            ),
        ):
            result = await call_llm_with_openai_chat_completion_async(
                client,
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                tracer=tracer,
            )

        assert isinstance(result, ModelResponse)
        assert "Hello world" in (result.content or "")

    @pytest.mark.anyio
    async def test_stream_shutdown_event_breaks(self, openai_chat_stream_chunks: list[dict[str, Any]]) -> None:
        """Lines 1760-1762: shutdown event during streaming breaks the loop."""
        shutdown = threading.Event()
        shutdown.set()
        params = Mock(spec=ModelCallParams)
        params.shutdown_event = shutdown

        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_AsyncChunkIterator(openai_chat_stream_chunks))

        # Even with chunks available, shutdown stops iteration so aggregator raises
        with pytest.raises(RuntimeError, match="No stream chunks"):
            await call_llm_with_openai_chat_completion_async(
                client,
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                model_call_params=params,
            )

    @pytest.mark.anyio
    async def test_stream_process_chunk_filters_none(self) -> None:
        """Lines 1763-1765: _process_stream_chunk returning None skips the chunk."""
        raw_chunks = [
            {"model": "gpt-4o-mini", "choices": [{"delta": {"role": "assistant", "content": "Ok"}}]},
        ]
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_AsyncChunkIterator(raw_chunks))

        mm = Mock(spec=MiddlewareManager)
        # First call returns None (filtered), no second chunk => aggregator raises
        mm.stream_chunk.return_value = None
        params = Mock(spec=ModelCallParams)
        params.shutdown_event = None

        with pytest.raises(RuntimeError, match="No stream chunks"):
            await call_llm_with_openai_chat_completion_async(
                client,
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                middleware_manager=mm,
                model_call_params=params,
            )

    @pytest.mark.anyio
    async def test_stream_with_llm_config_stream_flag(self, openai_chat_stream_chunks: list[dict[str, Any]]) -> None:
        """Stream activated via llm_config.stream rather than kwargs['stream']."""
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_chat_completion",
            stream=True,
        )
        client = AsyncMock()
        client.chat.completions.create = AsyncMock(return_value=_AsyncChunkIterator(openai_chat_stream_chunks))

        result = await call_llm_with_openai_chat_completion_async(
            client,
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            llm_config=llm_config,
        )

        assert isinstance(result, ModelResponse)


# ===========================================================================
# Tests: call_llm_with_anthropic_chat_completion_async — stream paths (L1851-1886)
# ===========================================================================


def _make_anthropic_params(shutdown_event: threading.Event | None = None) -> ModelCallParams:
    """Build a real ModelCallParams for Anthropic async tests."""
    return ModelCallParams(
        messages=[Message.user("Hi")],
        max_tokens=100,
        force_stop_reason=None,
        agent_state=None,
        tool_call_mode="structured",
        tools=None,
        api_params={},
        shutdown_event=shutdown_event,
    )


class TestAnthropicChatCompletionAsyncStream:
    """Cover Anthropic async stream paths with and without tracing."""

    @pytest.mark.anyio
    async def test_stream_without_tracing(self, anthropic_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1874-1884: stream path without tracer."""
        client = Mock()
        client.messages.stream = Mock(return_value=_AsyncStreamContextManager(anthropic_stream_events))
        params = _make_anthropic_params()

        result = await call_llm_with_anthropic_chat_completion_async(
            client,
            {
                "model": "claude-3",
                "max_tokens": 100,
                "stream": True,
            },
            model_call_params=params,
        )

        assert isinstance(result, ModelResponse)
        assert "Hi there" in (result.content or "")

    @pytest.mark.anyio
    async def test_stream_with_tracing(self, anthropic_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1854-1873: stream path with tracer."""
        client = Mock()
        client.messages.stream = Mock(return_value=_AsyncStreamContextManager(anthropic_stream_events))
        tracer = _FakeTracer()
        params = _make_anthropic_params()

        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.get_current_span",
                return_value=_FakeSpan(),
            ),
            patch(
                "nexau.archs.main_sub.execution.llm_caller.TraceContext",
                side_effect=lambda *a, **kw: _FakeTraceContext(*a, **kw),
            ),
        ):
            result = await call_llm_with_anthropic_chat_completion_async(
                client,
                {
                    "model": "claude-3",
                    "max_tokens": 100,
                    "stream": True,
                },
                tracer=tracer,
                model_call_params=params,
            )

        assert isinstance(result, ModelResponse)
        assert "Hi there" in (result.content or "")

    @pytest.mark.anyio
    async def test_stream_shutdown_event(self, anthropic_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1877-1879: shutdown event breaks out of stream loop."""
        shutdown = threading.Event()
        shutdown.set()
        params = _make_anthropic_params(shutdown_event=shutdown)

        client = Mock()
        client.messages.stream = Mock(return_value=_AsyncStreamContextManager(anthropic_stream_events))

        with pytest.raises(RuntimeError, match="No stream chunks"):
            await call_llm_with_anthropic_chat_completion_async(
                client,
                {
                    "model": "claude-3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
                model_call_params=params,
            )

    @pytest.mark.anyio
    async def test_stream_process_chunk_returns_none(self, anthropic_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1880-1882: when _process_stream_chunk returns None, event is skipped."""
        client = Mock()
        client.messages.stream = Mock(return_value=_AsyncStreamContextManager(anthropic_stream_events))

        mm = Mock(spec=MiddlewareManager)
        mm.stream_chunk.return_value = None
        params = _make_anthropic_params()

        with pytest.raises(RuntimeError, match="No stream chunks"):
            await call_llm_with_anthropic_chat_completion_async(
                client,
                {
                    "model": "claude-3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 100,
                    "stream": True,
                },
                middleware_manager=mm,
                model_call_params=params,
            )


# ===========================================================================
# Tests: call_llm_with_openai_responses_async (L1947-1992)
# ===========================================================================


class TestOpenAIResponsesAsyncNonStream:
    """Cover non-stream paths for Responses API (L1947-1954)."""

    @pytest.mark.anyio
    async def test_non_stream_without_tracing(self) -> None:
        """Lines 1952-1954: non-stream path without tracer."""
        response_payload = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Non-stream reply"}],
                }
            ],
            output_text="Non-stream reply",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=response_payload)

        result = await call_llm_with_openai_responses_async(
            client,
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
        )

        assert isinstance(result, ModelResponse)
        assert "Non-stream reply" in (result.content or "")

    @pytest.mark.anyio
    async def test_non_stream_with_tracing(self) -> None:
        """Lines 1947-1951: non-stream path with tracer."""
        response_payload = SimpleNamespace(
            output=[
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Traced reply"}],
                }
            ],
            output_text="Traced reply",
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        client = AsyncMock()
        client.responses.create = AsyncMock(return_value=response_payload)
        tracer = _FakeTracer()

        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.get_current_span",
                return_value=_FakeSpan(),
            ),
            patch(
                "nexau.archs.main_sub.execution.llm_caller.TraceContext",
                side_effect=lambda *a, **kw: _FakeTraceContext(*a, **kw),
            ),
        ):
            result = await call_llm_with_openai_responses_async(
                client,
                {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
                tracer=tracer,
            )

        assert isinstance(result, ModelResponse)
        assert "Traced reply" in (result.content or "")


class TestOpenAIResponsesAsyncStream:
    """Cover stream paths for Responses API (L1961-1992)."""

    @pytest.mark.anyio
    async def test_stream_without_tracing(self, openai_responses_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1981-1992: stream path without tracer."""
        client = AsyncMock()
        client.responses.stream = Mock(return_value=_AsyncStreamContextManager(openai_responses_stream_events))

        result = await call_llm_with_openai_responses_async(
            client,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert isinstance(result, ModelResponse)

    @pytest.mark.anyio
    async def test_stream_with_tracing(self, openai_responses_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1960-1979: stream path with tracer."""
        client = AsyncMock()
        client.responses.stream = Mock(return_value=_AsyncStreamContextManager(openai_responses_stream_events))
        tracer = _FakeTracer()

        with (
            patch(
                "nexau.archs.main_sub.execution.llm_caller.get_current_span",
                return_value=_FakeSpan(),
            ),
            patch(
                "nexau.archs.main_sub.execution.llm_caller.TraceContext",
                side_effect=lambda *a, **kw: _FakeTraceContext(*a, **kw),
            ),
        ):
            result = await call_llm_with_openai_responses_async(
                client,
                {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                tracer=tracer,
            )

        assert isinstance(result, ModelResponse)

    @pytest.mark.anyio
    async def test_stream_shutdown_event(self, openai_responses_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1967-1969 / 1983-1985: shutdown event breaks out of stream loop."""
        shutdown = threading.Event()
        shutdown.set()
        params = Mock(spec=ModelCallParams)
        params.shutdown_event = shutdown

        client = AsyncMock()
        client.responses.stream = Mock(return_value=_AsyncStreamContextManager(openai_responses_stream_events))

        # Shutdown immediately, aggregator gets no completed response => empty output
        result = await call_llm_with_openai_responses_async(
            client,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
            model_call_params=params,
        )
        assert isinstance(result, ModelResponse)

    @pytest.mark.anyio
    async def test_stream_process_chunk_returns_none(self, openai_responses_stream_events: list[dict[str, Any]]) -> None:
        """Lines 1986-1988: when _process_stream_chunk returns None, event is skipped."""
        client = AsyncMock()
        client.responses.stream = Mock(return_value=_AsyncStreamContextManager(openai_responses_stream_events))

        mm = Mock(spec=MiddlewareManager)
        mm.stream_chunk.return_value = None
        params = Mock(spec=ModelCallParams)
        params.shutdown_event = None

        result = await call_llm_with_openai_responses_async(
            client,
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
            middleware_manager=mm,
            model_call_params=params,
        )
        assert isinstance(result, ModelResponse)

    @pytest.mark.anyio
    async def test_stream_via_llm_config(self, openai_responses_stream_events: list[dict[str, Any]]) -> None:
        """Stream activated via llm_config.stream flag."""
        llm_config = LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_responses",
            stream=True,
        )
        client = AsyncMock()
        client.responses.stream = Mock(return_value=_AsyncStreamContextManager(openai_responses_stream_events))

        result = await call_llm_with_openai_responses_async(
            client,
            {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hi"}]},
            llm_config=llm_config,
        )

        assert isinstance(result, ModelResponse)


# ===========================================================================
# Tests: _prepare_responses_api_input — tool_calls + reasoning (L2131-2168)
# ===========================================================================


class TestPrepareResponsesApiInputToolCalls:
    """Cover tool_calls reconstruction on assistant messages (L2131-2162)."""

    def test_tool_calls_with_string_arguments(self) -> None:
        """Tool calls with string arguments are converted to function_call items."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me help.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                    }
                ],
            }
        ]
        prepared, instructions = _prepare_responses_api_input(messages)

        # Should have the message item + the function_call item
        fn_calls = [item for item in prepared if item.get("type") == "function_call"]
        assert len(fn_calls) == 1
        assert fn_calls[0]["name"] == "get_weather"
        assert fn_calls[0]["call_id"] == "call_1"
        assert fn_calls[0]["arguments"] == '{"city": "Tokyo"}'

    def test_tool_calls_with_dict_arguments(self) -> None:
        """Dict arguments are JSON-serialized (L2152-2153)."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "NexAU"},
                        },
                    }
                ],
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        fn_calls = [item for item in prepared if item.get("type") == "function_call"]
        assert len(fn_calls) == 1
        parsed = json.loads(fn_calls[0]["arguments"])
        assert parsed == {"query": "NexAU"}

    def test_tool_calls_non_mapping_items_skipped(self) -> None:
        """Non-mapping items in tool_calls list are skipped (L2133-2134)."""
        messages = [
            {
                "role": "assistant",
                "content": "Hi",
                "tool_calls": [
                    "not-a-mapping",
                    42,
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "foo",
                            "arguments": "{}",
                        },
                    },
                ],
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        fn_calls = [item for item in prepared if item.get("type") == "function_call"]
        assert len(fn_calls) == 1
        assert fn_calls[0]["name"] == "foo"

    def test_multiple_tool_calls(self) -> None:
        """Multiple tool calls on the same assistant message."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "function": {"name": "tool_a", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "function": {"name": "tool_b", "arguments": '{"x":1}'},
                    },
                ],
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        fn_calls = [item for item in prepared if item.get("type") == "function_call"]
        assert len(fn_calls) == 2
        names = {fc["name"] for fc in fn_calls}
        assert names == {"tool_a", "tool_b"}

    def test_tool_calls_missing_function_raises(self) -> None:
        """Tool call without 'function' key raises assertion (L2142)."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_bad"},  # missing 'function'
                ],
            }
        ]
        with pytest.raises(ValueError, match="function"):
            _prepare_responses_api_input(messages)

    def test_tool_calls_function_not_dict_raises(self) -> None:
        """Tool call with non-dict 'function' raises ValueError."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_bad", "function": "not-a-dict"},
                ],
            }
        ]
        with pytest.raises(ValueError, match="must be a dict"):
            _prepare_responses_api_input(messages)

    def test_tool_calls_function_missing_name_raises(self) -> None:
        """Tool call function without 'name' raises ValueError."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_bad", "function": {"arguments": "{}"}},
                ],
            }
        ]
        with pytest.raises(ValueError, match="name"):
            _prepare_responses_api_input(messages)

    def test_tool_calls_function_missing_arguments_raises(self) -> None:
        """Tool call function without 'arguments' raises ValueError."""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_bad", "function": {"name": "foo"}},
                ],
            }
        ]
        with pytest.raises(ValueError, match="arguments"):
            _prepare_responses_api_input(messages)


class TestPrepareResponsesApiInputReasoning:
    """Cover reasoning items inclusion on assistant messages (L2164-2168)."""

    def test_reasoning_list_included(self) -> None:
        """Reasoning items (list) are extended into prepared output (L2166-2168)."""
        reasoning_items = [
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking..."}]},
        ]
        messages = [
            {
                "role": "assistant",
                "content": "Result",
                "reasoning": reasoning_items,
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        # The reasoning items should appear in the prepared output
        reasoning_in_prepared = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(reasoning_in_prepared) >= 1

    def test_reasoning_non_list_ignored(self) -> None:
        """Non-list reasoning values are ignored (isinstance check at L2167)."""
        messages = [
            {
                "role": "assistant",
                "content": "Result",
                "reasoning": "not-a-list",
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        reasoning_in_prepared = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(reasoning_in_prepared) == 0

    def test_reasoning_empty_list(self) -> None:
        """Empty reasoning list is falsy, so the branch is skipped."""
        messages = [
            {
                "role": "assistant",
                "content": "Result",
                "reasoning": [],
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        # Only the message item should exist, no reasoning items
        reasoning_in_prepared = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(reasoning_in_prepared) == 0

    def test_tool_calls_and_reasoning_together(self) -> None:
        """Both tool_calls and reasoning on the same assistant message."""
        messages = [
            {
                "role": "assistant",
                "content": "Let me think and act.",
                "tool_calls": [
                    {
                        "id": "call_x",
                        "function": {"name": "do_thing", "arguments": "{}"},
                    }
                ],
                "reasoning": [
                    {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "reasoning"}]},
                ],
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        fn_calls = [item for item in prepared if item.get("type") == "function_call"]
        reasoning = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(fn_calls) == 1
        assert len(reasoning) >= 1

    def test_no_reasoning_key_at_all(self) -> None:
        """No 'reasoning' key on message — the branch is simply not entered."""
        messages = [
            {
                "role": "assistant",
                "content": "Simple reply",
            }
        ]
        prepared, _ = _prepare_responses_api_input(messages)

        msg_items = [item for item in prepared if item.get("type") == "message"]
        assert len(msg_items) == 1
        reasoning_in_prepared = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(reasoning_in_prepared) == 0
