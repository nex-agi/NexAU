from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nexau.archs.main_sub.execution.llm_caller import call_llm_with_openai_chat_completion
from nexau.archs.tracer.context import reset_current_span, set_current_span
from nexau.archs.tracer.core import BaseTracer, Span, SpanType


class RecordingTracer(BaseTracer):
    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.end_calls: list[dict[str, Any]] = []

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        self.start_calls.append({"name": name, "span_type": span_type, "inputs": inputs, "parent": parent_span})
        return Span(
            id="span",
            name=name,
            type=span_type,
            parent_id=getattr(parent_span, "id", None),
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(self, span: Span, outputs: Any = None, error: Exception | None = None, attributes: dict[str, Any] | None = None) -> None:
        self.end_calls.append({"span": span, "outputs": outputs, "error": error, "attributes": attributes})


class FakeStream:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        yield from self._chunks


class FakeCompletions:
    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = chunks
        self.create_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeStream:
        self.create_calls.append(kwargs)
        return FakeStream(self._chunks)


class FakeChat:
    def __init__(self, chunks: list[Any]) -> None:
        self.completions = FakeCompletions(chunks)


class FakeOpenAIClient:
    def __init__(self, chunks: list[Any]) -> None:
        self.chat = FakeChat(chunks)


def test_openai_stream_call_is_wrapped_in_trace_context_when_current_span_exists() -> None:
    tracer = RecordingTracer()
    current = Span(id="parent", name="parent", type=SpanType.AGENT)
    token = set_current_span(current)
    try:
        chunks = [
            {
                "model": "gpt-test",
                "choices": [{"delta": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ]
        client = FakeOpenAIClient(chunks)

        response = call_llm_with_openai_chat_completion(
            client,  # type: ignore[arg-type]
            {
                "model": "gpt-test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            tracer=tracer,
        )

        assert response.content == "hi"
        assert any(call["name"] == "OpenAI chat.completions.create (stream)" for call in tracer.start_calls)

        end_call = next(call for call in tracer.end_calls if call["span"].name == "OpenAI chat.completions.create (stream)")
        assert end_call["outputs"]["content"] == "hi"
    finally:
        reset_current_span(token)
