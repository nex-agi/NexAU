"""Unit tests for tracer context management, composite tracer, and Langfuse adapter."""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from nexau.archs.tracer.adapters import InMemoryTracer, LangfuseTracer
from nexau.archs.tracer.composite import CompositeTracer
from nexau.archs.tracer.context import (
    TraceContext,
    get_current_span,
    reset_current_span,
    set_current_span,
)
from nexau.archs.tracer.core import BaseTracer, Span, SpanType


class RecordingTracer(BaseTracer):
    """Simple tracer that records interactions for assertions."""

    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.end_calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        span = Span(
            id=str(uuid.uuid4()),
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )
        self.start_calls.append(
            {
                "name": name,
                "span_type": span_type,
                "parent_id": parent_span.id if parent_span else None,
            }
        )
        return span

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.end_calls.append(
            {
                "span": span,
                "outputs": outputs,
                "error": error,
                "attributes": attributes,
            }
        )

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


class FakeBackendTracer(BaseTracer):
    """Tracer used to validate CompositeTracer behavior with vendor objects."""

    def __init__(self) -> None:
        self.start_calls: list[dict[str, Any]] = []
        self.end_calls: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        vendor_handle: dict[str, Any] = {"name": name, "events": []}
        self.start_calls.append({"name": name, "parent_vendor": getattr(parent_span, "vendor_obj", None)})
        return Span(
            id=str(uuid.uuid4()),
            name=name,
            type=span_type,
            vendor_obj=vendor_handle,
        )

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.end_calls.append(
            {
                "vendor_obj": span.vendor_obj,
                "outputs": outputs,
                "error": error,
                "attributes": attributes,
            }
        )

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


class DummyLangfuseObject:
    """Minimal Langfuse object mock that stores interactions."""

    def __init__(self) -> None:
        self.start_span_calls: list[dict[str, Any]] = []
        self.start_observation_calls: list[dict[str, Any]] = []
        self.update_calls: list[dict[str, Any]] = []
        self.ended = False
        self.metadata: dict[str, Any] = {}

    def start_span(self, **kwargs: Any) -> DummyLangfuseObject:
        self.start_span_calls.append(kwargs)
        child = DummyLangfuseObject()
        child.metadata = kwargs.get("metadata", {})
        return child

    def start_observation(self, **kwargs: Any) -> DummyLangfuseObject:
        self.start_observation_calls.append(kwargs)
        child = DummyLangfuseObject()
        child.metadata = kwargs.get("metadata", {})
        return child

    def update(self, **kwargs: Any) -> None:
        self.update_calls.append(kwargs)

    def end(self) -> None:
        self.ended = True


class DummyLangfuseClient:
    """Fake Langfuse client used to instantiate LangfuseTracer without real deps."""

    instances: list[DummyLangfuseClient] = []

    def __init__(self, **kwargs: Any) -> None:
        self.__class__.instances.append(self)
        self.kwargs = kwargs
        self.start_span_calls: list[dict[str, Any]] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def start_span(self, **kwargs: Any) -> DummyLangfuseObject:
        self.start_span_calls.append(kwargs)
        root = DummyLangfuseObject()
        root.metadata = kwargs.get("metadata", {})
        return root

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture(autouse=True)
def patch_langfuse(monkeypatch):
    """Patch Langfuse SDK with dummy objects for all tests in this module."""

    DummyLangfuseClient.instances.clear()
    monkeypatch.setattr("nexau.archs.tracer.adapters.langfuse.Langfuse", DummyLangfuseClient)
    yield


def test_trace_context_manages_parent_and_outputs():
    tracer = RecordingTracer()

    parent_ctx = TraceContext(tracer, "parent", SpanType.AGENT)
    with parent_ctx as parent_span:
        child_ctx = TraceContext(tracer, "child", SpanType.TOOL)
        child_ctx.set_outputs({"value": 42})
        with child_ctx as child_span:
            assert child_span.parent_id == parent_span.id
            assert get_current_span() == child_span

        # After child exits, parent becomes current again
        assert get_current_span() == parent_span

    child_end_call = next(call for call in tracer.end_calls if call["span"].name == "child")
    assert child_end_call["outputs"] == {"value": 42}
    assert get_current_span() is None


def test_trace_context_propagates_errors():
    tracer = RecordingTracer()

    with pytest.raises(RuntimeError):
        with TraceContext(tracer, "error", SpanType.AGENT):
            raise RuntimeError("boom")

    assert isinstance(tracer.end_calls[0]["error"], RuntimeError)


def test_context_helpers_set_and_reset():
    tracer = RecordingTracer()
    span = tracer.start_span("helper", SpanType.AGENT)
    token = set_current_span(span)
    assert get_current_span() == span
    reset_current_span(token)
    assert get_current_span() is None


def test_composite_tracer_broadcasts_calls():
    backend_a = FakeBackendTracer()
    backend_b = FakeBackendTracer()
    composite = CompositeTracer([backend_a, backend_b])

    root_span = composite.start_span("root", SpanType.AGENT)
    child_span = composite.start_span("child", SpanType.TOOL, parent_span=root_span)

    assert isinstance(child_span.vendor_obj, dict)
    assert backend_a.start_calls[1]["parent_vendor"] == root_span.vendor_obj[0]

    composite.end_span(child_span, outputs={"status": "ok"})
    assert backend_b.end_calls[0]["outputs"] == {"status": "ok"}

    composite.flush()
    composite.shutdown()
    assert backend_a.flush_count == 1
    assert backend_b.shutdown_count == 1


def test_composite_tracer_handles_backend_errors():
    class ExplodingTracer(FakeBackendTracer):
        def start_span(self, *args: Any, **kwargs: Any) -> Span:
            raise ValueError("fail")

    backend = FakeBackendTracer()
    composite = CompositeTracer([ExplodingTracer(), backend])

    span = composite.start_span("root", SpanType.AGENT)
    assert span.vendor_obj[1]["name"] == "root"


def test_in_memory_tracer_dumps_nested_spans():
    tracer = InMemoryTracer()
    root = tracer.start_span("root", SpanType.AGENT, inputs={"foo": "bar"})
    child = tracer.start_span("child", SpanType.TOOL, parent_span=root, attributes={"nested": True})

    tracer.end_span(child, outputs="child-result")
    tracer.end_span(root, outputs={"status": "ok"}, attributes={"stage": "test"})

    traces = tracer.dump_traces()
    assert len(traces) == 1

    root_dump = traces[0]
    assert root_dump["name"] == "root"
    assert root_dump["outputs"]["status"] == "ok"
    assert root_dump["children"][0]["name"] == "child"
    assert root_dump["children"][0]["outputs"]["result"] == "child-result"
    assert root_dump["children"][0]["parent_id"] == root.id


def test_in_memory_tracer_works_with_composite():
    memory_tracer = InMemoryTracer()
    other_tracer = FakeBackendTracer()
    composite = CompositeTracer([memory_tracer, other_tracer])

    root = composite.start_span("root", SpanType.AGENT)
    child = composite.start_span("child", SpanType.LLM, parent_span=root)

    composite.end_span(child, outputs={"text": "hello"})
    composite.end_span(root, outputs={"status": "done"})

    dumped = memory_tracer.dump_traces()
    assert dumped[0]["outputs"]["status"] == "done"
    assert dumped[0]["children"][0]["outputs"]["text"] == "hello"


def test_langfuse_tracer_creates_trace_and_generation():
    tracer = LangfuseTracer(debug=True)
    root_inputs = {"payload": (1, "two")}
    root_span = tracer.start_span("root", SpanType.AGENT, inputs=root_inputs)

    client = DummyLangfuseClient.instances[-1]
    assert client.start_span_calls[0]["input"] == {"payload": [1, "two"]}
    assert isinstance(root_span.vendor_obj, DummyLangfuseObject)

    llm_span = tracer.start_span("llm", SpanType.LLM, parent_span=root_span, attributes={"foo": "bar"})
    assert root_span.vendor_obj.start_observation_calls[0]["as_type"] == "span"
    assert llm_span.vendor_obj.metadata["span_type"] == SpanType.LLM.value


def test_langfuse_tracer_end_span_updates_and_flushes():
    tracer = LangfuseTracer(debug=True)
    span = tracer.start_span("tool", SpanType.TOOL)
    outputs = {"model": "gpt", "usage": {"input_tokens": 1}, "result": "ok"}

    tracer.end_span(span, outputs=outputs, attributes={"color": "blue"})

    vendor_obj: DummyLangfuseObject = span.vendor_obj  # type: ignore[assignment]
    output_call = next(call for call in vendor_obj.update_calls if "output" in call)
    model_call = next(call for call in vendor_obj.update_calls if "model" in call)
    assert output_call["output"]["result"] == "ok"
    assert model_call["model"] == "gpt"
    assert vendor_obj.ended is True
    assert tracer.client.flush_count == 1  # type: ignore[union-attr]


def test_langfuse_tracer_disabled_skips_client():
    tracer = LangfuseTracer(enabled=False)
    span = tracer.start_span("noop", SpanType.AGENT)
    assert span.vendor_obj is None
    tracer.end_span(span, outputs={"ignored": True})


def test_langfuse_serialization_handles_custom_objects():
    class Custom:
        def __str__(self) -> str:
            return "custom-object"

    data = {"values": (1, 2), "obj": Custom()}
    serialized = LangfuseTracer._serialize_for_langfuse(data)
    assert serialized["values"] == [1, 2]
    assert serialized["obj"].replace('"', "") == "custom-object"
