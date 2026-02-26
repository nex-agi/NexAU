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

"""Tests for the non-tracing Anthropic streaming branch.

Covers the else branch in llm_stream_call (no tracer active):
- Normal streaming without tracing
- shutdown_event interruption mid-stream
- Middleware filtering (_process_stream_chunk returning None)
- No model_call_params (shutdown_ev is None)
"""

from __future__ import annotations

import threading
from types import SimpleNamespace, TracebackType
from typing import Any
from unittest.mock import Mock

from nexau.archs.main_sub.execution import llm_caller
from nexau.archs.main_sub.execution.hooks import (
    MiddlewareManager,
    ModelCallParams,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse


class _IterableStream:
    """Context-manager wrapper for faking client.messages.stream()."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


# -- Helpers ------------------------------------------------------------------


def _anthropic_text_events() -> list[dict[str, Any]]:
    """Return a fresh set of Anthropic text streaming events."""
    return [
        {
            "type": "message_start",
            "message": {
                "role": "assistant",
                "model": "claude-test",
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]


def _make_anthropic_client(events: list[dict[str, Any]]) -> Any:
    """Build a fake Anthropic client whose messages.create(stream=True) yields *events*."""

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_payload: Any) -> _IterableStream:
            if _payload.get("stream"):
                return _IterableStream(events)
            raise NotImplementedError("non-stream create should not be called")

    return _FakeClient()


# -- Tests --------------------------------------------------------------------


def test_anthropic_stream_no_tracing_normal_flow() -> None:
    """The else branch produces a valid ModelResponse when no tracer is active."""
    client = _make_anthropic_client(_anthropic_text_events())
    kwargs: dict[str, Any] = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    # No tracer -> hits the else branch
    resp = llm_caller.call_llm_with_anthropic_chat_completion(client, kwargs, tracer=None)

    assert isinstance(resp, ModelResponse)
    assert resp.content == "Hello world"


def test_anthropic_stream_no_tracing_with_tool_use() -> None:
    """Tool-use blocks are correctly aggregated in the non-tracing streaming path."""
    partial_json_1 = '{"city": "Tokyo'
    partial_json_2 = '"}'
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {"role": "assistant", "model": "claude-test", "usage": {"input_tokens": 5, "output_tokens": 0}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial_json_1},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": partial_json_2},
        },
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    client = _make_anthropic_client(events)
    kwargs: dict[str, Any] = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "weather?"}],
        "stream": True,
    }

    resp = llm_caller.call_llm_with_anthropic_chat_completion(client, kwargs, tracer=None)

    assert isinstance(resp, ModelResponse)
    assert resp.has_tool_calls()
    assert resp.tool_calls[0].name == "get_weather"
    assert resp.tool_calls[0].arguments == {"city": "Tokyo"}


def test_anthropic_stream_shutdown_event_interrupts_mid_stream() -> None:
    """When shutdown_event is set, the stream loop breaks and returns a partial response."""
    shutdown_ev = threading.Event()

    # Build events where the shutdown fires after the first text delta
    class _InterruptingStream:
        def __init__(self) -> None:
            self._events = list(_anthropic_text_events())

        def __iter__(self):
            for event in self._events:
                yield event
                # Set shutdown after the first text delta
                if isinstance(event, dict) and event.get("type") == "content_block_delta":
                    shutdown_ev.set()

        def __enter__(self):
            return self

        def __exit__(self, *_: object) -> None:
            return None

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_payload: Any) -> _InterruptingStream:
            return _InterruptingStream()

    client = _FakeClient()

    model_call_params = Mock(spec=ModelCallParams)
    model_call_params.shutdown_event = shutdown_ev

    kwargs: dict[str, Any] = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    resp = llm_caller.call_llm_with_anthropic_chat_completion(
        client,
        kwargs,
        model_call_params=model_call_params,
        tracer=None,
    )

    assert isinstance(resp, ModelResponse)
    # Should have partial content ("Hello") but not the full "Hello world"
    assert resp.content is not None
    assert "Hello" in resp.content
    assert shutdown_ev.is_set()


def test_anthropic_stream_middleware_filters_chunks() -> None:
    """Chunks for which _process_stream_chunk returns None are skipped."""

    def _filtering_stream_chunk(chunk: Any, params: Any) -> Any:  # noqa: ARG001
        if not isinstance(chunk, dict):
            return chunk
        payload: dict[str, Any] = chunk
        # Filter out the second text delta (" world")
        if payload.get("type") == "content_block_delta":
            delta = payload.get("delta")
            if isinstance(delta, dict) and delta.get("text") == " world":
                return None
        return chunk

    middleware = Mock(spec=MiddlewareManager)
    middleware.stream_chunk = Mock(side_effect=_filtering_stream_chunk)

    model_call_params = Mock(spec=ModelCallParams)
    model_call_params.shutdown_event = None

    client = _make_anthropic_client(_anthropic_text_events())
    kwargs: dict[str, Any] = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    resp = llm_caller.call_llm_with_anthropic_chat_completion(
        client,
        kwargs,
        middleware_manager=middleware,
        model_call_params=model_call_params,
        tracer=None,
    )

    assert isinstance(resp, ModelResponse)
    # " world" was filtered out by middleware
    assert resp.content == "Hello"
    assert middleware.stream_chunk.call_count == 6  # 6 events in _anthropic_text_events()


def test_anthropic_stream_no_model_call_params_skips_shutdown_check() -> None:
    """When model_call_params is None, shutdown_ev is None and the loop runs fully."""
    client = _make_anthropic_client(_anthropic_text_events())
    kwargs: dict[str, Any] = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    # model_call_params=None -> _shutdown_ev = None -> no shutdown check
    resp = llm_caller.call_llm_with_anthropic_chat_completion(
        client,
        kwargs,
        model_call_params=None,
        tracer=None,
    )

    assert isinstance(resp, ModelResponse)
    assert resp.content == "Hello world"


def test_anthropic_stream_shutdown_event_not_set_runs_fully() -> None:
    """When shutdown_event exists but is never set, the full stream is consumed."""
    shutdown_ev = threading.Event()  # never set

    model_call_params = Mock(spec=ModelCallParams)
    model_call_params.shutdown_event = shutdown_ev

    client = _make_anthropic_client(_anthropic_text_events())
    kwargs: dict[str, Any] = {
        "model": "claude-test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }

    resp = llm_caller.call_llm_with_anthropic_chat_completion(
        client,
        kwargs,
        model_call_params=model_call_params,
        tracer=None,
    )

    assert isinstance(resp, ModelResponse)
    assert resp.content == "Hello world"
    assert not shutdown_ev.is_set()
