from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace, TracebackType
from typing import Any

import pytest

from nexau.archs.main_sub.execution import llm_caller
from nexau.archs.tracer.adapters.in_memory import InMemoryTracer
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import SpanType


class _IterableStream:
    def __init__(self, items: Iterable[Any]) -> None:
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __enter__(self):
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        return None


def _patch_time(monkeypatch: pytest.MonkeyPatch, values: list[float]) -> None:
    it = iter(values)
    monkeypatch.setattr(llm_caller.time, "time", lambda: next(it))


def test_openai_chat_stream_records_time_to_first_token_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_time(monkeypatch, [1000.0, 1000.1])

    chunks = [
        {
            "model": "gpt-test",
            "choices": [
                {
                    "delta": {"role": "assistant", "content": "hi"},
                }
            ],
        }
    ]

    class FakeOpenAIClient:
        def __init__(self) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, *, stream: bool, **payload: Any) -> _IterableStream:  # noqa: ARG002
            assert stream is True
            return _IterableStream(chunks)

    tracer = InMemoryTracer()
    client: Any = FakeOpenAIClient()
    kwargs = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True}

    with TraceContext(tracer, "parent", SpanType.AGENT):
        resp = llm_caller.call_llm_with_openai_chat_completion(client, kwargs, tracer=tracer)

    assert resp.content == "hi"

    llm_spans = [s for s in tracer.spans.values() if s.name == "OpenAI chat.completions.create (stream)"]
    assert len(llm_spans) == 1
    assert abs(float(llm_spans[0].attributes["time_to_first_token_ms"]) - 100.0) < 1e-6


def test_openai_responses_stream_records_time_to_first_token_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_time(monkeypatch, [2000.0, 2000.25])

    events = [
        {
            "type": "response.output_item.added",
            "item": {"type": "message", "role": "assistant", "id": "msg_1", "content": [{"type": "output_text", "text": ""}]},
        },
        {"type": "response.output_text.delta", "item_id": "msg_1", "content_index": 0, "delta": "hi"},
        {"type": "response.completed", "response": {"id": "resp_1", "model": "gpt-test", "usage": {"input_tokens": 1, "output_tokens": 1}}},
    ]

    class FakeResponsesClient:
        def __init__(self) -> None:
            self.responses = SimpleNamespace(stream=self._stream)

        def _stream(self, **payload: Any) -> _IterableStream:  # noqa: ARG002
            return _IterableStream(events)

    tracer = InMemoryTracer()
    client = FakeResponsesClient()
    kwargs = {"model": "gpt-test", "messages": [{"role": "user", "content": "hello"}], "stream": True}

    with TraceContext(tracer, "parent", SpanType.AGENT):
        resp = llm_caller.call_llm_with_openai_responses(client, kwargs, tracer=tracer)

    assert resp.content == "hi"

    llm_spans = [s for s in tracer.spans.values() if s.name == "OpenAI responses.stream"]
    assert len(llm_spans) == 1
    assert abs(float(llm_spans[0].attributes["time_to_first_token_ms"]) - 250.0) < 1e-6


def test_anthropic_stream_records_time_to_first_token_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_time(monkeypatch, [3000.0, 3000.05])

    events = [
        {
            "type": "message_start",
            "message": {"role": "assistant", "model": "claude-test", "usage": {"input_tokens": 1, "output_tokens": 1}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]

    class FakeAnthropicClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **payload: Any) -> _IterableStream:  # noqa: ARG002
            return _IterableStream(events)

    tracer = InMemoryTracer()
    client = FakeAnthropicClient()
    kwargs = {"model": "claude-test", "messages": [{"role": "user", "content": "hello"}], "stream": True}

    with TraceContext(tracer, "parent", SpanType.AGENT):
        resp = llm_caller.call_llm_with_anthropic_chat_completion(client, kwargs, tracer=tracer)

    assert resp.content == "hi"

    llm_spans = [s for s in tracer.spans.values() if s.name == "Anthropic messages.stream"]
    assert len(llm_spans) == 1
    assert abs(float(llm_spans[0].attributes["time_to_first_token_ms"]) - 50.0) < 1e-6
