"""RFC-0027 structured output helper tests."""

from __future__ import annotations

import pytest

from nexau.archs.workflow.structured_output import (
    StructuredOutputError,
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
