"""RFC-0027 workflow config parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.workflow import WorkflowConfig
from nexau.archs.workflow.types import json_object


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


def test_workflow_config_loads_external_subgraph(tmp_path: Path) -> None:
    child_path = tmp_path / "child.workflow.yaml"
    child_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: child_review",
                "nodes:",
                "  start:",
                "    type: start",
                "  finish:",
                "    type: end",
                "    output:",
                "      ok: true",
                "edges:",
                "  start: finish",
            ]
        ),
        encoding="utf-8",
    )
    parent_path = tmp_path / "parent.workflow.yaml"
    parent_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: parent",
                "includes:",
                "  graphs:",
                "    child_review: ./child.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "  call_child:",
                "    type: subgraph",
                "    graph: child_review",
                "edges:",
                "  start: call_child",
            ]
        ),
        encoding="utf-8",
    )

    config = WorkflowConfig.from_yaml(parent_path)

    assert config.nodes["call_child"].type == "subgraph"
    assert config.included_graphs["child_review"].name == "child_review"
    snapshot = config.definition_snapshot()
    included_graphs = json_object(snapshot["_included_graphs"], label="snapshot._included_graphs")
    child_snapshot = json_object(included_graphs["child_review"], label="snapshot._included_graphs.child_review")
    assert child_snapshot["name"] == "child_review"


def test_workflow_config_rejects_unknown_subgraph_reference() -> None:
    data = _base_workflow()
    data["nodes"] = {
        "start": {"type": "start"},
        "call_child": {"type": "subgraph", "graph": "missing_child"},
    }
    data["edges"] = {"start": "call_child"}

    with pytest.raises(ValueError, match="unknown includes.graphs"):
        WorkflowConfig.model_validate(data)


def test_workflow_config_rejects_recursive_graph_includes(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.workflow.yaml"
    child_path = tmp_path / "child.workflow.yaml"
    parent_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: parent",
                "includes:",
                "  graphs:",
                "    child: ./child.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "  call_child:",
                "    type: subgraph",
                "    graph: child",
                "edges:",
                "  start: call_child",
            ]
        ),
        encoding="utf-8",
    )
    child_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: child",
                "includes:",
                "  graphs:",
                "    parent: ./parent.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recursion"):
        WorkflowConfig.from_yaml(parent_path)


def test_workflow_config_accepts_parallel_map_definition() -> None:
    data = _base_workflow()
    data["durable"] = {"mode": "node_boundary", "default_parallelism": 2, "max_parallelism": 4}
    data["nodes"] = {
        "start": {"type": "start", "output": {"items": "{{ inputs.items }}"}},
        "map_items": {
            "type": "parallel_map",
            "items": "{{ nodes.start.output.items }}",
            "item_name": "case",
            "index_name": "case_index",
            "item_key": "{{ case.id }}",
            "max_concurrency": 2,
            "body": "run_item",
            "result_order": "input",
            "failure_policy": "collect_errors",
            "collect": {"output": {"results": "{{ results }}", "errors": "{{ errors }}", "stats": "{{ stats }}"}},
            "state_in": {"seed": "{{ case.id }}"},
            "state_out": {"mapped": "{{ stats.completed }}"},
        },
        "run_item": {
            "type": "agent",
            "agent": "worker",
            "input": {"case": "{{ case }}", "index": "{{ case_index }}"},
        },
        "finish": {"type": "end", "output": {"results": "{{ nodes.map_items.output.results }}"}},
    }
    data["edges"] = {"start": "map_items", "map_items": "finish"}

    config = WorkflowConfig.model_validate(data)

    assert config.durable.default_parallelism == 2
    assert config.durable.max_parallelism == 4
    assert config.nodes["map_items"].type == "parallel_map"
    assert config.nodes["map_items"].body == "run_item"


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda node: node.pop("items"), "requires 'items'"),
        (lambda node: node.pop("body"), "requires 'body'"),
        (lambda node: node.__setitem__("body", "missing"), "unknown body node"),
        (lambda node: node.__setitem__("max_concurrency", 5), "max_concurrency exceeds"),
        (lambda node: node.__setitem__("item_name", "state"), "reserved context name"),
        (lambda node: node.__setitem__("index_name", "item"), "invalid or reserved"),
        (lambda node: node.__setitem__("result_order", "random"), "Input should be"),
        (lambda node: node.__setitem__("failure_policy", "ignore"), "Input should be"),
    ],
)
def test_workflow_config_rejects_invalid_parallel_map_nodes(mutate, message: str) -> None:
    data = _base_workflow()
    data["durable"] = {"mode": "node_boundary", "max_parallelism": 4}
    data["nodes"] = {
        "start": {"type": "start"},
        "map_items": {
            "type": "parallel_map",
            "items": "{{ inputs.items }}",
            "body": "run_item",
            "max_concurrency": 2,
        },
        "run_item": {"type": "agent", "agent": "worker"},
    }
    data["edges"] = {"start": "map_items"}
    nodes = json_object(data["nodes"], label="nodes")
    map_node = json_object(nodes["map_items"], label="map_items")
    mutate(map_node)

    with pytest.raises(ValueError, match=message):
        WorkflowConfig.model_validate(data)


def test_workflow_config_rejects_parallel_map_body_on_normal_edges() -> None:
    data = _base_workflow()
    data["nodes"] = {
        "start": {"type": "start"},
        "map_items": {"type": "parallel_map", "items": "{{ inputs.items }}", "body": "run_item"},
        "run_item": {"type": "agent", "agent": "worker"},
    }
    data["edges"] = {"start": "map_items", "map_items": "run_item"}

    with pytest.raises(ValueError, match="must not be referenced by normal edges"):
        WorkflowConfig.model_validate(data)


def test_workflow_config_rejects_parallel_map_external_write_body_without_item_key() -> None:
    data = _base_workflow()
    data["durable"] = {"mode": "node_boundary", "default_retry_policy": {"on_uncertain": "human_review"}}
    data["nodes"] = {
        "start": {"type": "start"},
        "map_items": {"type": "parallel_map", "items": "{{ inputs.items }}", "body": "send_item"},
        "send_item": {
            "type": "tool",
            "tool": "send_tool",
            "side_effect": "external_write",
            "idempotency_key": "{{ run.id }}:{{ node.scope_path }}",
        },
    }
    data["edges"] = {"start": "map_items"}

    with pytest.raises(ValueError, match="requires explicit item_key"):
        WorkflowConfig.model_validate(data)
