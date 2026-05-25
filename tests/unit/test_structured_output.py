"""RFC-0027 structured output helper tests."""

from __future__ import annotations

from typing import Any

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.archs.workflow.structured_output import (
    StructuredOutputError,
    _build_structured_agent_config,
    build_complete_task_tool,
    parse_json_block,
    parse_structured_response,
    validate_json_schema_output,
)
from nexau.archs.workflow.types import JsonObject


def _schema() -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["passed", "failed"]},
            "evidence": {"type": "string"},
        },
        "required": ["status", "evidence"],
        "additionalProperties": False,
    }


class _NonDeepcopyableTracer(BaseTracer):
    def __deepcopy__(self, memo: dict[int, object]) -> _NonDeepcopyableTracer:
        del memo
        raise AssertionError("tracer should be preserved by reference")

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        return Span(
            id="span",
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(
        self,
        span: Span,
        outputs: Any = None,
        error: Exception | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        del span, outputs, error, attributes


def test_structured_agent_config_preserves_runtime_tracer_by_reference() -> None:
    tracer = _NonDeepcopyableTracer()
    config = AgentConfig(
        name="structured_agent",
        system_prompt="Return structured output.",
        llm_config=LLMConfig(model="test-model", base_url="https://example.invalid/v1", api_key="test-key"),
        tracers=[tracer],
    )

    structured = _build_structured_agent_config(
        config=config,
        output_schema=_schema(),
        output_mode="complete_task",
        output_name=None,
        tracer=tracer,
    )

    assert config.tracers == [tracer]
    assert config.resolved_tracer is tracer
    assert structured.tracers == [tracer]
    assert structured.resolved_tracer is tracer
    assert any(tool.name == "complete_task" for tool in structured.tools)


def test_dynamic_complete_task_tool_validates_schema() -> None:
    tool = build_complete_task_tool(_schema(), output_name="case_result")

    assert tool.name == "complete_task"
    assert tool.execute(status="passed", evidence="ok") == {"status": "passed", "evidence": "ok"}

    with pytest.raises(ValueError, match="Invalid parameters"):
        tool.execute(status="blocked", evidence="bad")


def test_parse_json_block_requires_one_json_object() -> None:
    assert parse_json_block('```json\n{"status":"passed","evidence":"ok"}\n```') == {
        "status": "passed",
        "evidence": "ok",
    }

    with pytest.raises(StructuredOutputError):
        parse_json_block("no json here")


def test_structured_response_validation() -> None:
    output = parse_structured_response('{"status":"failed","evidence":"trace"}', _schema())

    assert output == {"status": "failed", "evidence": "trace"}

    with pytest.raises(StructuredOutputError, match="schema validation"):
        validate_json_schema_output({"status": "blocked", "evidence": "trace"}, _schema())
