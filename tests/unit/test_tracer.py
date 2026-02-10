"""Unit tests for tracer context management, composite tracer, and Langfuse adapter."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from typing import Any, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
        self.trace_id: str | None = None

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


class DummyLangfuseObjectWithTrace(DummyLangfuseObject):
    """Dummy Langfuse object that also supports trace-level updates."""

    def __init__(self) -> None:
        super().__init__()
        self.update_trace_calls: list[dict[str, Any]] = []

    def update_trace(self, **kwargs: Any) -> None:
        self.update_trace_calls.append(kwargs)


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
        root.trace_id = kwargs.get("trace_context", {}).get("trace_id", None)
        root.metadata = kwargs.get("metadata", {})
        return root

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture(autouse=True)
def patch_langfuse(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Patch Langfuse SDK with dummy objects for all tests in this module."""

    DummyLangfuseClient.instances.clear()
    monkeypatch.setattr("nexau.archs.tracer.adapters.langfuse.Langfuse", DummyLangfuseClient)
    # Provide dummy credentials so LangfuseTracer can initialize the (patched) client.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.test")
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


def test_trace_context_set_attributes_updates_end_span_attributes() -> None:
    tracer = RecordingTracer()

    ctx = TraceContext(tracer, "attrs", SpanType.TOOL, attributes={"a": 1})
    with ctx:
        ctx.set_attributes({"b": 2})

    end_call = tracer.end_calls[0]
    assert end_call["span"].name == "attrs"
    assert end_call["attributes"] == {"a": 1, "b": 2}


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
    assert isinstance(root_span.vendor_obj, dict)
    root_vendor_obj = cast(dict[int, Any], root_span.vendor_obj)
    assert backend_a.start_calls[1]["parent_vendor"] == root_vendor_obj[0]

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
    assert isinstance(span.vendor_obj, dict)
    vendor_obj = cast(dict[int, Any], span.vendor_obj)
    assert vendor_obj[1]["name"] == "root"


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
    llm_vendor_obj = cast(DummyLangfuseObject, llm_span.vendor_obj)
    assert llm_vendor_obj.metadata["span_type"] == SpanType.LLM.value


def test_langfuse_tracer_end_span_updates_and_flushes():
    tracer = LangfuseTracer(debug=True)
    span = tracer.start_span("tool", SpanType.TOOL)
    outputs = {"model": "gpt", "usage": {"input_tokens": 1}, "result": "ok"}

    tracer.end_span(span, outputs=outputs, attributes={"color": "blue"})

    vendor_obj = cast(DummyLangfuseObject, span.vendor_obj)
    output_call = next(call for call in vendor_obj.update_calls if "output" in call)
    model_call = next(call for call in vendor_obj.update_calls if "model" in call)
    assert output_call["output"]["result"] == "ok"
    assert model_call["model"] == "gpt"
    assert vendor_obj.ended is True
    assert tracer.client is not None
    client = cast(DummyLangfuseClient, tracer.client)
    assert client.flush_count == 1


def test_langfuse_tracer_end_span_updates_trace_fields_when_supported() -> None:
    tracer = LangfuseTracer(
        debug=True,
        session_id="sess-123",
        user_id="user-456",
        tags=["tag-a", "tag-b"],
        metadata={"meta": "value"},
    )
    span = tracer.start_span("tool", SpanType.TOOL)
    assert span.vendor_obj is not None

    # Swap the vendor object for one that supports trace-level updates.
    trace_vendor = DummyLangfuseObjectWithTrace()
    span.vendor_obj = trace_vendor

    tracer.end_span(span, outputs={"result": "ok"})

    # Root span (parent_id is None) updates trace name and output first,
    # then metadata, user_id, session_id, and tags.
    assert trace_vendor.update_trace_calls == [
        {"name": "tool", "output": {"result": "ok"}},  # Root span sets trace name/output
        {"metadata": {"meta": "value"}},
        {"user_id": "user-456"},
        {"session_id": "sess-123"},
        {"tags": ["tag-a", "tag-b"]},
    ]


def test_langfuse_tracer_disabled_skips_client():
    tracer = LangfuseTracer(enabled=False)
    span = tracer.start_span("noop", SpanType.AGENT)
    assert span.vendor_obj is None
    tracer.end_span(span, outputs={"ignored": True})


def test_langfuse_serialization_handles_custom_objects():
    class Custom:
        def __str__(self) -> str:
            return "custom-object"

    tracer = LangfuseTracer(debug=True)
    span = tracer.start_span("root", SpanType.AGENT, inputs={"values": (1, 2), "obj": Custom()})
    assert span.vendor_obj is not None

    client = DummyLangfuseClient.instances[-1]
    serialized = client.start_span_calls[0]["input"]
    assert serialized["values"] == [1, 2]
    assert str(serialized["obj"]).replace('"', "") == "custom-object"


def test_langfuse_ensure_client_missing_keys_warns_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    tracer = LangfuseTracer()

    # Simulate warmup: credentials not available yet.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    caplog.set_level(logging.WARNING)
    span_1 = tracer.start_span("first", SpanType.AGENT)
    span_2 = tracer.start_span("second", SpanType.AGENT)

    assert span_1.vendor_obj is None
    assert span_2.vendor_obj is None
    assert sum("public_key/secret_key missing" in rec.message for rec in caplog.records) == 1


def test_langfuse_ensure_client_rotates_and_flushes_old_client(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = LangfuseTracer()
    root_span = tracer.start_span("root", SpanType.AGENT)
    assert root_span.vendor_obj is not None

    old_client = cast(DummyLangfuseClient, tracer.client)
    assert old_client is not None

    # Rotate credentials to force client replacement and best-effort flush/shutdown of the old one.
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-rotated")
    _ = tracer.start_span("after-rotate", SpanType.AGENT)

    assert old_client.flush_count == 1
    assert old_client.shutdown_count == 1
    # In prod this is `Langfuse | None`, but tests monkeypatch it to `DummyLangfuseClient`.
    assert cast(Any, tracer.client) is not old_client


def test_langfuse_ensure_client_init_failure_logs_warning(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("init-fail")

    monkeypatch.setattr("nexau.archs.tracer.adapters.langfuse.Langfuse", ExplodingLangfuse)

    tracer = LangfuseTracer()
    caplog.set_level(logging.WARNING)
    span = tracer.start_span("root", SpanType.AGENT)

    assert span.vendor_obj is None
    assert tracer.client is None
    assert any("Langfuse tracer failed to initialize" in rec.message for rec in caplog.records)


def test_langfuse_activate_and_deactivate_span_success(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = LangfuseTracer()

    class DummyVendor:
        def __init__(self) -> None:
            self._otel_span = object()

    entered: list[bool] = []
    exited: list[bool] = []

    class DummyCtx:
        def __enter__(self) -> DummyCtx:
            entered.append(True)
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            exited.append(True)

    def fake_use_span(span: Any, *, end_on_exit: bool = False) -> DummyCtx:
        assert span is not None
        assert end_on_exit is False
        return DummyCtx()

    monkeypatch.setattr("nexau.archs.tracer.adapters.langfuse.otel_trace_api.use_span", fake_use_span)

    span = Span(id="s", name="n", type=SpanType.AGENT, vendor_obj=DummyVendor())
    token = tracer.activate_span(span)
    assert token is not None
    assert entered == [True]

    tracer.deactivate_span(token)
    assert exited == [True]


def test_langfuse_activate_span_handles_use_span_error(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = LangfuseTracer()

    class DummyVendor:
        def __init__(self) -> None:
            self._otel_span = object()

    def exploding_use_span(span: Any, *, end_on_exit: bool = False) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr("nexau.archs.tracer.adapters.langfuse.otel_trace_api.use_span", exploding_use_span)

    span = Span(id="s", name="n", type=SpanType.AGENT, vendor_obj=DummyVendor())
    assert tracer.activate_span(span) is None


def test_langfuse_deactivate_span_swallows_exit_errors() -> None:
    tracer = LangfuseTracer()

    class ExplodingToken:
        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            raise RuntimeError("exit-fail")

    tracer.deactivate_span(ExplodingToken())


def test_langfuse_tracer_flush_calls_client_and_logs_debug(caplog: pytest.LogCaptureFixture) -> None:
    tracer = LangfuseTracer(debug=True)
    _ = tracer.start_span("root", SpanType.AGENT)
    assert tracer.client is not None
    client = cast(DummyLangfuseClient, tracer.client)
    before = client.flush_count

    caplog.set_level(logging.DEBUG)
    tracer.flush()

    assert client.flush_count == before + 1
    assert any("Flushed Langfuse data" in rec.message for rec in caplog.records)


def test_langfuse_tracer_flush_handles_exception(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingClient:
        def flush(self) -> None:
            raise RuntimeError("flush-fail")

    tracer = LangfuseTracer(debug=True)
    tracer.client = cast(Any, ExplodingClient())

    caplog.set_level(logging.WARNING)
    tracer.flush()

    assert any("Failed to flush Langfuse data" in rec.message for rec in caplog.records)


def test_langfuse_tracer_shutdown_calls_client_and_logs_info(caplog: pytest.LogCaptureFixture) -> None:
    tracer = LangfuseTracer()
    tracer.client = cast(Any, DummyLangfuseClient())

    caplog.set_level(logging.INFO)
    tracer.shutdown()

    assert any("Langfuse tracer shutdown" in rec.message for rec in caplog.records)


def test_langfuse_tracer_shutdown_handles_exception(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingClient:
        def shutdown(self) -> None:
            raise RuntimeError("shutdown-fail")

    tracer = LangfuseTracer()
    tracer.client = cast(Any, ExplodingClient())

    caplog.set_level(logging.WARNING)
    tracer.shutdown()

    assert any("Failed to shutdown Langfuse client" in rec.message for rec in caplog.records)


def test_composite_tracer_flush_and_activate_deactivate_best_effort(caplog: pytest.LogCaptureFixture) -> None:
    class ExplodingFlushTracer(FakeBackendTracer):
        def flush(self) -> None:
            raise RuntimeError("flush-fail")

    class ActivatingTracer(FakeBackendTracer):
        def __init__(self) -> None:
            super().__init__()
            self.deactivated: list[Any] = []

        def activate_span(self, span: Span) -> Any | None:  # noqa: ANN401
            return {"token_for": span.vendor_obj}

        def deactivate_span(self, token: Any | None) -> None:  # noqa: ANN401
            self.deactivated.append(token)

    class ExplodingActivateTracer(FakeBackendTracer):
        def activate_span(self, span: Span) -> Any | None:  # noqa: ANN401
            raise RuntimeError("activate-fail")

    caplog.set_level(logging.WARNING)
    good = ActivatingTracer()
    composite = CompositeTracer([ExplodingFlushTracer(), ExplodingActivateTracer(), good])

    composite.flush()
    assert any("Failed to flush tracer 0" in rec.message for rec in caplog.records)

    # Not a vendor map => no-op for activate/deactivate
    assert composite.activate_span(Span(id="x", name="x", type=SpanType.AGENT, vendor_obj=None)) is None
    composite.deactivate_span("not-a-dict")

    # Vendor map: activate should ignore exploding tracer and return token for good tracer only.
    span = Span(id="s", name="n", type=SpanType.AGENT, vendor_obj={0: {"v": 0}, 1: {"v": 1}, 2: {"v": 2}})
    token = composite.activate_span(span)
    assert isinstance(token, dict)
    assert 2 in token
    assert 1 not in token

    token_map = cast(dict[int, Any], token)
    composite.deactivate_span(token_map)
    assert len(good.deactivated) == 1


def test_langfuse_tracer_id(caplog: pytest.LogCaptureFixture) -> None:
    trace_id = "c86a9db70f561c9ad82d134217485867"
    tracer = LangfuseTracer(debug=True, trace_id=trace_id)
    span = tracer.start_span("root", SpanType.AGENT)
    assert tracer.client is not None
    span_vendor_obj = cast(DummyLangfuseObject, span.vendor_obj)
    assert span_vendor_obj.trace_id == trace_id


def test_langfuse_tracer_end_span_updates_trace_name_for_root_span() -> None:
    """Verify that ending a root span calls update_trace(name=...) to set trace name.

    This is critical defensive programming: when using trace_context.trace_id,
    Langfuse SDK creates a trace with empty name. LangfuseTracer must explicitly
    call update_trace(name=span.name) to ensure the trace has a meaningful name.

    Without this fix, users who enable auto-instrumentation may see unnamed traces
    in Langfuse UI because the SDK creates trace records without inheriting span names.
    """
    tracer = LangfuseTracer(
        debug=True,
        session_id="sess-123",
        user_id="user-456",
    )
    root_span = tracer.start_span(
        "Agent: test-agent",
        SpanType.AGENT,
        inputs={"message": "hello"},
    )
    assert root_span.vendor_obj is not None
    assert root_span.parent_id is None  # Confirm this is a root span

    # Swap vendor object for one that supports trace-level updates
    trace_vendor = DummyLangfuseObjectWithTrace()
    root_span.vendor_obj = trace_vendor

    # End the root span with outputs
    tracer.end_span(root_span, outputs={"response": "world"})

    # CRITICAL: Verify update_trace was called with name for root span
    # This ensures trace name is set even when Langfuse SDK creates empty trace
    name_update_calls = [c for c in trace_vendor.update_trace_calls if "name" in c]
    assert len(name_update_calls) >= 1, (
        "update_trace(name=...) was not called for root span! "
        "This will cause unnamed traces when using trace_context.trace_id. "
        "Fix: Add update_trace(name=span.name) in end_span() for root spans."
    )
    assert name_update_calls[0]["name"] == "Agent: test-agent"

    # Also verify input/output are set on trace for root spans
    input_update_calls = [c for c in trace_vendor.update_trace_calls if "input" in c]
    output_update_calls = [c for c in trace_vendor.update_trace_calls if "output" in c]
    assert len(input_update_calls) >= 1, "update_trace(input=...) was not called for root span"
    assert len(output_update_calls) >= 1, "update_trace(output=...) was not called for root span"


def test_langfuse_tracer_end_span_does_not_update_trace_name_for_child_span() -> None:
    """Verify that ending a child span does NOT call update_trace(name=...).

    Only root spans should update trace-level fields. Child spans should not
    overwrite the trace name set by the root span.
    """
    tracer = LangfuseTracer(debug=True)
    root_span = tracer.start_span("Agent: parent", SpanType.AGENT)
    child_span = tracer.start_span("Tool: child", SpanType.TOOL, parent_span=root_span)
    assert child_span.parent_id == root_span.id  # Confirm this is a child span

    # Swap vendor object for one that supports trace-level updates
    trace_vendor = DummyLangfuseObjectWithTrace()
    child_span.vendor_obj = trace_vendor

    tracer.end_span(child_span, outputs={"result": "done"})

    # Child span should NOT update trace name
    name_update_calls = [c for c in trace_vendor.update_trace_calls if "name" in c]
    assert len(name_update_calls) == 0, "update_trace(name=...) was called for child span! Only root spans should update trace-level name."


def test_langfuse_tracer_root_span_without_inputs_skips_input_update() -> None:
    """Verify that root span without inputs does not call update_trace(input=...).

    Covers the `if span.inputs:` guard in end_span().
    """
    tracer = LangfuseTracer(debug=True)
    # Start root span WITHOUT inputs
    root_span = tracer.start_span("Agent: no-input", SpanType.AGENT)
    assert root_span.parent_id is None

    trace_vendor = DummyLangfuseObjectWithTrace()
    root_span.vendor_obj = trace_vendor

    # End without outputs either
    tracer.end_span(root_span)

    # Should have update_trace(name=...) but NOT input or output
    name_calls = [c for c in trace_vendor.update_trace_calls if "name" in c]
    input_calls = [c for c in trace_vendor.update_trace_calls if "input" in c]
    output_calls = [c for c in trace_vendor.update_trace_calls if "output" in c]

    assert len(name_calls) == 1, "Root span should always set trace name"
    assert name_calls[0]["name"] == "Agent: no-input"
    assert len(input_calls) == 0, "Root span without inputs should not call update_trace(input=...)"
    assert len(output_calls) == 0, "Root span without outputs should not call update_trace(output=...)"


def test_langfuse_tracer_root_span_with_inputs_but_no_outputs() -> None:
    """Verify root span with inputs but no outputs sets input but not output on trace."""
    tracer = LangfuseTracer(debug=True)
    root_span = tracer.start_span(
        "Agent: input-only",
        SpanType.AGENT,
        inputs={"query": "hello"},
    )
    assert root_span.parent_id is None

    trace_vendor = DummyLangfuseObjectWithTrace()
    root_span.vendor_obj = trace_vendor

    tracer.end_span(root_span)  # No outputs

    name_calls = [c for c in trace_vendor.update_trace_calls if "name" in c]
    assert len(name_calls) == 1
    # The first call should have name and input but NOT output
    first_call = name_calls[0]
    assert first_call["name"] == "Agent: input-only"
    assert "input" in first_call, "Root span with inputs should include input in trace update"
    assert first_call["input"] == {"query": "hello"}
    assert "output" not in first_call, "Root span without outputs should not include output"


def test_langfuse_ensure_client_passes_tracer_provider() -> None:
    """Verify that _ensure_client passes tracer_provider to Langfuse constructor.

    This is critical: without isolated TracerProvider, Langfuse v3+ would
    overwrite the global TracerProvider, causing all spans to be sent to Langfuse.
    """
    captured_kwargs: list[dict[str, Any]] = []

    class CapturingLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.append(kwargs)

    import nexau.archs.tracer.adapters.langfuse as langfuse_mod

    original = langfuse_mod.Langfuse
    langfuse_mod.Langfuse = CapturingLangfuse  # type: ignore[assignment,misc]
    try:
        tracer = LangfuseTracer(
            public_key="pk-test",
            secret_key="sk-test",
            host="http://test.local",
        )
        tracer._ensure_client()

        assert len(captured_kwargs) == 1, "Langfuse should be instantiated once"
        kwargs = captured_kwargs[0]
        assert "tracer_provider" in kwargs, (
            "tracer_provider must be passed to Langfuse() to prevent global TracerProvider pollution. "
            "See: https://langfuse.com/faq/all/existing-otel-setup"
        )
        from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider

        assert isinstance(kwargs["tracer_provider"], SdkTracerProvider), (
            f"tracer_provider should be an isolated SdkTracerProvider, got {type(kwargs['tracer_provider'])}"
        )
    finally:
        langfuse_mod.Langfuse = original  # type: ignore[assignment,misc]


def test_parallel_tool_spans_have_correct_parent_with_batched_snapshots():
    """Verify that parallel tool spans all have the Agent span as parent when using
    the 'snapshot-before-activate' pattern (batch copy_context() before submission).

    This is the core regression test for the fix-span-overlap bug: when multiple
    tool calls are submitted to ThreadPoolExecutor, each thread's TraceContext
    should create a Tool span whose parent_id is the Agent span, not another
    tool's span or a polluted intermediate state.

    Validates: Requirements 3.1, 3.2
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from contextvars import copy_context

    tracer = RecordingTracer()
    num_tools = 5
    tool_spans_from_threads: list[Span] = []
    errors: list[Exception] = []

    with TraceContext(tracer, "Agent: test-agent", SpanType.AGENT) as agent_span:
        # Step 1: Batch all context snapshots BEFORE any task submission
        # This is the fixed pattern from executor.py
        tool_snapshots = [(copy_context(), f"Tool: tool_{i}") for i in range(num_tools)]

        # Step 2: Submit all tasks using pre-created snapshots
        with ThreadPoolExecutor(max_workers=num_tools) as executor:

            def run_tool_in_thread(tool_name: str) -> Span:
                """Simulate a tool execution inside a TraceContext."""
                with TraceContext(tracer, tool_name, SpanType.TOOL) as tool_span:
                    # The tool span should see the Agent span as its parent
                    # because copy_context() was called while Agent span was current
                    return tool_span

            futures = {}
            for task_ctx, tool_name in tool_snapshots:
                future = executor.submit(task_ctx.run, run_tool_in_thread, tool_name)
                futures[future] = tool_name

            for future in as_completed(futures):
                try:
                    tool_span = future.result()
                    tool_spans_from_threads.append(tool_span)
                except Exception as e:
                    errors.append(e)

    # No errors should have occurred
    assert not errors, f"Errors during parallel execution: {errors}"

    # All tool spans should have been created
    assert len(tool_spans_from_threads) == num_tools

    # CRITICAL: Every tool span's parent_id must be the Agent span's id
    for tool_span in tool_spans_from_threads:
        assert tool_span.parent_id == agent_span.id, (
            f"Tool span '{tool_span.name}' has parent_id={tool_span.parent_id}, "
            f"expected Agent span id={agent_span.id}. "
            "This indicates OTel context pollution between parallel tool threads."
        )

    # Verify all tool spans are distinct (no duplicates)
    tool_span_ids = [s.id for s in tool_spans_from_threads]
    assert len(set(tool_span_ids)) == num_tools, "Tool span IDs should all be unique"

    # Verify the RecordingTracer captured the correct number of start/end calls
    # 1 Agent span + N Tool spans = N+1 total
    assert len(tracer.start_calls) == num_tools + 1
    assert len(tracer.end_calls) == num_tools + 1

    # Verify all tool start_calls have the correct parent_id
    tool_start_calls = [c for c in tracer.start_calls if c["span_type"] == SpanType.TOOL]
    assert len(tool_start_calls) == num_tools
    for call in tool_start_calls:
        assert call["parent_id"] == agent_span.id, (
            f"RecordingTracer captured tool start_call with parent_id={call['parent_id']}, expected {agent_span.id}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests (hypothesis)
# ---------------------------------------------------------------------------


@given(num_tool_calls=st.integers(min_value=1, max_value=20))
@settings(max_examples=100)
def test_property_context_snapshot_consistency(num_tool_calls: int) -> None:
    """Property 1: Context snapshot consistency.

    For any set of N tool calls (N >= 1) being submitted to ThreadPoolExecutor,
    all N context snapshots created by copy_context() should contain the same
    OTel current span (the parent/Agent span), regardless of the number of tools
    or their execution order.

    This validates the "snapshot-before-activate" pattern: when copy_context()
    calls are batched before any task submission, every snapshot captures the
    same, unpolluted OTel context state.

    **Validates: Requirements 1.1, 1.2, 1.3, 2.1, 2.2**
    """
    from contextvars import copy_context

    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace_api
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    # Create a known parent span with a deterministic trace/span ID
    parent_span_context = SpanContext(
        trace_id=0x1234567890ABCDEF1234567890ABCDEF,
        span_id=0xFEDCBA0987654321,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    parent_span = NonRecordingSpan(context=parent_span_context)

    # Attach the parent span to the current OTel context
    new_ctx = otel_trace_api.set_span_in_context(parent_span)
    token = otel_context.attach(new_ctx)

    try:
        # Batch N copy_context() calls — the "snapshot-before-activate" pattern
        snapshots = [copy_context() for _ in range(num_tool_calls)]

        # Verify every snapshot contains the same OTel current span
        for i, snapshot in enumerate(snapshots):
            captured_span = snapshot.run(otel_trace_api.get_current_span)
            assert captured_span is parent_span, (
                f"Snapshot {i} captured span {captured_span!r} instead of the "
                f"expected parent span {parent_span!r}. "
                f"This indicates context pollution between copy_context() calls."
            )

            # Also verify the span context attributes match
            captured_ctx = captured_span.get_span_context()
            assert captured_ctx.trace_id == parent_span_context.trace_id, (
                f"Snapshot {i} trace_id mismatch: {captured_ctx.trace_id:#x} != {parent_span_context.trace_id:#x}"
            )
            assert captured_ctx.span_id == parent_span_context.span_id, (
                f"Snapshot {i} span_id mismatch: {captured_ctx.span_id:#x} != {parent_span_context.span_id:#x}"
            )
    finally:
        # Always detach to restore the OTel context
        otel_context.detach(token)


@given(num_parallel_tools=st.integers(min_value=2, max_value=10))
@settings(max_examples=100)
def test_property_parallel_tool_span_parenting(num_parallel_tools: int) -> None:
    """Property 3: Parallel tool span parenting.

    For any set of N parallel tool executions (N >= 2), each tool's active span
    inside its TraceContext should be that tool's own span, not another tool's
    span or the server span. Specifically, for each tool thread,
    get_current_span() should return the span created by that tool's
    TraceContext.__enter__, and each Tool span's parent_id should be the Agent
    span's id.

    This validates the "snapshot-before-activate" pattern: when copy_context()
    calls are batched before any task submission, each thread's TraceContext
    creates a Tool span whose parent is the Agent span, and the active span
    inside each thread is that thread's own Tool span.

    **Validates: Requirements 3.1, 3.2, 4.1**
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from contextvars import copy_context

    tracer = RecordingTracer()
    results: list[dict[str, Any]] = []
    errors: list[Exception] = []

    with TraceContext(tracer, "Agent: property-test", SpanType.AGENT) as agent_span:
        # Step 1: Batch all context snapshots BEFORE any task submission
        tool_snapshots = [(copy_context(), f"Tool: prop_tool_{i}") for i in range(num_parallel_tools)]

        # Step 2: Submit all tasks using pre-created snapshots
        with ThreadPoolExecutor(max_workers=num_parallel_tools) as executor:

            def run_tool_in_thread(tool_name: str) -> dict[str, Any]:
                """Execute a tool inside TraceContext and capture active span info."""
                with TraceContext(tracer, tool_name, SpanType.TOOL) as tool_span:
                    # Capture the active span inside this thread's TraceContext
                    active_span = get_current_span()
                    return {
                        "tool_name": tool_name,
                        "tool_span_id": tool_span.id,
                        "tool_span_name": tool_span.name,
                        "tool_span_parent_id": tool_span.parent_id,
                        "active_span_id": active_span.id if active_span else None,
                        "active_span_name": active_span.name if active_span else None,
                    }

            futures = {}
            for task_ctx, tool_name in tool_snapshots:
                future = executor.submit(task_ctx.run, run_tool_in_thread, tool_name)
                futures[future] = tool_name

            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    errors.append(e)

    # No errors should have occurred during parallel execution
    assert not errors, f"Errors during parallel execution: {errors}"

    # All tools should have produced results
    assert len(results) == num_parallel_tools

    for result in results:
        tool_name = result["tool_name"]

        # CRITICAL Property 3a: Each tool span's parent_id must be the Agent span's id
        assert result["tool_span_parent_id"] == agent_span.id, (
            f"Tool '{tool_name}' has parent_id={result['tool_span_parent_id']}, "
            f"expected Agent span id={agent_span.id}. "
            "This indicates OTel context pollution — the tool's parent is wrong."
        )

        # CRITICAL Property 3b: The active span inside each thread's TraceContext
        # must be that thread's own Tool span, not another tool's span
        assert result["active_span_id"] == result["tool_span_id"], (
            f"Tool '{tool_name}' active span id={result['active_span_id']} "
            f"does not match its own tool span id={result['tool_span_id']}. "
            "This indicates context cross-contamination between parallel threads."
        )
        assert result["active_span_name"] == tool_name, (
            f"Tool '{tool_name}' active span name='{result['active_span_name']}' "
            f"does not match expected name='{tool_name}'. "
            "Another tool's span is active in this thread's context."
        )

    # All tool span IDs should be unique (no duplicates)
    tool_span_ids = [r["tool_span_id"] for r in results]
    assert len(set(tool_span_ids)) == num_parallel_tools, (
        f"Expected {num_parallel_tools} unique tool span IDs, got {len(set(tool_span_ids))}. Duplicate spans detected."
    )


@given(nesting_depth=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_property_context_restoration_after_exit(nesting_depth: int) -> None:
    """Property 4: Context restoration after TraceContext exit.

    For any TraceContext usage, after __exit__ completes, the OTel current span
    should be restored to the span that was active before __enter__ was called.

    This test creates nested TraceContext instances to a random depth (1-5) and
    verifies that after each __exit__, the current span is correctly restored to
    the span that was active before the corresponding __enter__.

    **Validates: Requirements 3.3**
    """
    tracer = RecordingTracer()

    # Define span types to cycle through for nested contexts
    span_types = [SpanType.AGENT, SpanType.TOOL, SpanType.LLM, SpanType.SUB_AGENT, SpanType.TOOL]

    # Before any context is entered, current span should be None
    assert get_current_span() is None

    # Build nested contexts: enter them one by one, tracking each span
    contexts: list[TraceContext] = []
    spans: list[Span] = []

    for depth in range(nesting_depth):
        span_type = span_types[depth % len(span_types)]
        name = f"span_depth_{depth}"
        ctx = TraceContext(tracer, name, span_type)
        span = ctx.__enter__()
        contexts.append(ctx)
        spans.append(span)

        # After entering, the current span should be the one we just created
        assert get_current_span() is span, (
            f"After entering depth {depth}, current span should be '{name}' (id={span.id}), but got {get_current_span()!r}"
        )

        # Verify parent relationship: each span's parent should be the previous span
        if depth > 0:
            assert span.parent_id == spans[depth - 1].id, (
                f"Span at depth {depth} should have parent_id={spans[depth - 1].id}, but got parent_id={span.parent_id}"
            )
        else:
            # The first span should have no parent (None was current before entering)
            assert span.parent_id is None, f"Root span at depth 0 should have parent_id=None, but got parent_id={span.parent_id}"

    # Now exit contexts in reverse order (LIFO) and verify restoration
    for depth in range(nesting_depth - 1, -1, -1):
        ctx = contexts[depth]
        ctx.__exit__(None, None, None)

        if depth > 0:
            # After exiting this context, the current span should be the parent span
            expected_span = spans[depth - 1]
            actual_span = get_current_span()
            assert actual_span is expected_span, (
                f"After exiting depth {depth}, current span should be restored to "
                f"'{expected_span.name}' (id={expected_span.id}), "
                f"but got {actual_span!r}"
            )
        else:
            # After exiting the outermost context, current span should be None
            assert get_current_span() is None, (
                f"After exiting the outermost context (depth 0), current span should be None, but got {get_current_span()!r}"
            )

    # Final verification: no span should be active
    assert get_current_span() is None, "After all contexts have exited, no span should be active"


@given(num_tracers=st.integers(min_value=1, max_value=5))
@settings(max_examples=100)
def test_property_lifo_deactivation_order(num_tracers: int) -> None:
    """Property 2: LIFO deactivation order.

    For any sequence of activate_span calls across K tracers in a CompositeTracer,
    calling deactivate_span should restore the OTel context to the state before
    the corresponding activate_span, following LIFO (last-in-first-out) order.

    This test creates K OTel-context-aware tracers in a CompositeTracer, activates
    a span, then deactivates it, and verifies the OTel current span is restored
    to the pre-activation state.

    **Validates: Requirements 2.3**
    """
    from opentelemetry import context as otel_context
    from opentelemetry import trace as otel_trace_api
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    class OTelContextTracer(BaseTracer):
        """Tracer that modifies OTel context on activate/deactivate, similar to JaegerTracer.

        Each instance creates its own OTel span and attaches it to the global
        OTel context on activate_span, then detaches on deactivate_span.
        """

        def __init__(self, tracer_id: int) -> None:
            self.tracer_id = tracer_id

        def start_span(
            self,
            name: str,
            span_type: SpanType,
            inputs: dict[str, Any] | None = None,
            parent_span: Span | None = None,
            attributes: dict[str, Any] | None = None,
        ) -> Span:
            # Create a unique OTel span for this tracer
            otel_span = NonRecordingSpan(
                context=SpanContext(
                    trace_id=0xAAAABBBBCCCCDDDD0000000000000000 + self.tracer_id,
                    span_id=0x1111222233330000 + self.tracer_id,
                    is_remote=False,
                    trace_flags=TraceFlags(TraceFlags.SAMPLED),
                )
            )
            return Span(
                id=str(uuid.uuid4()),
                name=name,
                type=span_type,
                vendor_obj=otel_span,
            )

        def end_span(
            self,
            span: Span,
            outputs: Any = None,
            error: Exception | None = None,
            attributes: dict[str, Any] | None = None,
        ) -> None:
            pass

        def activate_span(self, span: Span) -> Any | None:  # noqa: ANN401
            """Attach the OTel span to the global context, like JaegerTracer does."""
            otel_span = span.vendor_obj
            if otel_span is None:
                return None
            new_ctx = otel_trace_api.set_span_in_context(cast(otel_trace_api.Span, otel_span))
            token = otel_context.attach(new_ctx)
            return token

        def deactivate_span(self, token: Any | None) -> None:  # noqa: ANN401
            """Detach the OTel context, restoring the previous state."""
            if token is None:
                return
            otel_context.detach(token)

    # Set up a known "pre-activation" OTel context with a parent span
    parent_span_context = SpanContext(
        trace_id=0xDEADBEEFDEADBEEF0000000000000000,
        span_id=0xCAFEBABECAFE0000,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    parent_otel_span = NonRecordingSpan(context=parent_span_context)

    # Attach the parent span to establish the pre-activation state
    pre_ctx = otel_trace_api.set_span_in_context(parent_otel_span)
    pre_token = otel_context.attach(pre_ctx)

    try:
        # Capture the pre-activation OTel current span
        pre_activation_span = otel_trace_api.get_current_span()
        assert pre_activation_span is parent_otel_span, "Pre-activation span should be the parent span we just attached"

        # Create K OTel-context-aware tracers and compose them
        tracers: list[BaseTracer] = [OTelContextTracer(i) for i in range(num_tracers)]
        composite = CompositeTracer(tracers)

        # Create a span via the CompositeTracer
        span = composite.start_span("test-span", SpanType.TOOL)

        # Activate the span — this will call activate_span on each tracer,
        # each of which attaches its own OTel span to the context
        activate_token = composite.activate_span(span)

        # After activation, the OTel current span should NOT be the parent span
        # (it should be the last tracer's span, since each tracer overwrites)
        post_activation_span = otel_trace_api.get_current_span()
        if num_tracers > 0:
            assert post_activation_span is not parent_otel_span, (
                "After activation, the OTel current span should have been changed by the tracers' activate_span calls"
            )

        # Deactivate the span — this should restore the OTel context
        # to the pre-activation state (LIFO unwinding)
        composite.deactivate_span(activate_token)

        # CRITICAL PROPERTY: After deactivation, the OTel current span
        # should be exactly the same as before activation
        post_deactivation_span = otel_trace_api.get_current_span()
        assert post_deactivation_span is parent_otel_span, (
            f"After deactivate_span, the OTel current span should be restored "
            f"to the pre-activation parent span. "
            f"Expected: {parent_otel_span!r} (trace_id={parent_span_context.trace_id:#x}), "
            f"Got: {post_deactivation_span!r}. "
            f"This indicates LIFO deactivation order is broken with {num_tracers} tracers."
        )

        # Also verify the span context attributes match exactly
        restored_ctx = post_deactivation_span.get_span_context()
        assert restored_ctx.trace_id == parent_span_context.trace_id, (
            f"Restored trace_id {restored_ctx.trace_id:#x} != expected {parent_span_context.trace_id:#x}"
        )
        assert restored_ctx.span_id == parent_span_context.span_id, (
            f"Restored span_id {restored_ctx.span_id:#x} != expected {parent_span_context.span_id:#x}"
        )
    finally:
        # Always restore the OTel context to clean state
        otel_context.detach(pre_token)
