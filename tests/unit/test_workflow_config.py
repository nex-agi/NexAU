"""RFC-0027 workflow config parser tests."""

from __future__ import annotations

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.workflow import WorkflowConfig


def _base_workflow() -> dict[str, object]:
    return {
        "type": "workflow",
        "version": "1",
        "name": "unit_workflow",
        "nodes": {
            "start": {"type": "start", "output": {"value": "{{ inputs.value }}"}},
            "finish": {"type": "end", "input": {"value": "{{ nodes.start.output.value }}"}},
        },
        "edges": {"start": "finish"},
    }


def test_workflow_config_accepts_valid_definition() -> None:
    config = WorkflowConfig.model_validate(_base_workflow())

    assert config.name == "unit_workflow"
    assert config.start_node_id == "start"
    assert config.nodes["finish"].type == "end"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda data: data["nodes"].pop("start"), "exactly one start"),
        (lambda data: data["nodes"].__setitem__("other_start", {"type": "start"}), "exactly one start"),
        (lambda data: data["edges"].__setitem__("missing", "finish"), "unknown node"),
        (lambda data: data["edges"].__setitem__("finish", "missing"), "unknown node"),
        (lambda data: data["edges"].__setitem__("finish", "start"), "cycle"),
    ],
)
def test_workflow_config_rejects_invalid_graphs(mutate, message: str) -> None:
    data = _base_workflow()
    mutate(data)

    with pytest.raises(ValueError, match=message):
        WorkflowConfig.model_validate(data)


def test_workflow_config_rejects_while_without_max_iterations() -> None:
    data = _base_workflow()
    data["nodes"] = {
        "start": {"type": "start"},
        "loop": {"type": "while", "condition": "true", "body": "body"},
        "body": {"type": "transform"},
    }
    data["edges"] = {"start": "loop"}

    with pytest.raises(ValueError, match="max_iterations"):
        WorkflowConfig.model_validate(data)


def test_workflow_config_rejects_external_write_without_idempotency_or_uncertain_policy() -> None:
    data = _base_workflow()
    data["nodes"] = {
        "start": {"type": "start"},
        "send": {"type": "tool", "tool": "send_email", "side_effect": "external_write"},
    }
    data["edges"] = {"start": "send"}

    with pytest.raises(ValueError, match="requires 'idempotency_key'"):
        WorkflowConfig.model_validate(data)


def test_agent_config_loads_structured_output_fields() -> None:
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string"}},
        "required": ["status"],
    }

    config = AgentConfig(
        name="structured_agent",
        llm_config=LLMConfig(model="test", base_url="https://example.invalid", api_key="test"),
        output_schema=schema,
        output_mode="json_block",
        output_retries=2,
        output_name="agent_result",
    )

    assert config.output_schema == schema
    assert config.output_mode == "json_block"
    assert config.output_retries == 2
    assert config.output_name == "agent_result"
