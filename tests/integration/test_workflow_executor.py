"""RFC-0027 durable workflow executor integration tests."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nexau import AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.session.models.workflow import WorkflowNodeStatus
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.adapters import InMemoryTracer
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import Span, SpanType
from nexau.archs.workflow import WorkflowConfig, WorkflowExecutor, WorkflowResumeError, WorkflowStore
from nexau.archs.workflow.store import event_payload
from nexau.archs.workflow.types import JsonObject


def _qa_workflow() -> WorkflowConfig:
    return WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "qa_regression",
            "durable": {"mode": "node_boundary", "default_retry_policy": {"max_attempts": 2, "on_uncertain": "human_review"}},
            "nodes": {
                "start": {"type": "start", "output": {"requirement": "{{ inputs.requirement }}"}},
                "generate_cases": {
                    "type": "agent",
                    "agent": "qa_planner",
                    "input": {"requirement": "{{ nodes.start.output.requirement }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "cases": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "string"},
                                        "title": {"type": "string"},
                                        "steps": {"type": "array", "items": {"type": "string"}},
                                        "expected": {"type": "string"},
                                    },
                                    "required": ["id", "title", "steps", "expected"],
                                },
                            }
                        },
                        "required": ["cases"],
                    },
                },
                "review_cases": {
                    "type": "human",
                    "prompt": "Review generated QA cases.",
                    "input": {"cases": "{{ nodes.generate_cases.output.cases }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}, "cases": {"type": "array"}},
                        "required": ["approved", "cases"],
                    },
                },
                "route_review": {
                    "type": "if_else",
                    "branches": [{"if": "nodes.review_cases.output.approved == true", "next": "run_cases"}],
                    "else": "generate_cases",
                },
                "run_cases": {
                    "type": "while",
                    "condition": "state.remaining_cases.length > 0",
                    "max_iterations": 5,
                    "scope_key": "case-{{ state.remaining_cases[0].id }}",
                    "body": "run_one_case",
                    "init": {"remaining_cases": "{{ nodes.review_cases.output.cases }}", "results": []},
                },
                "run_one_case": {
                    "type": "agent",
                    "agent": "qa_runner",
                    "side_effect": "external_write",
                    "idempotency_key": "{{ run.id }}:{{ node.scope_path }}",
                    "input": {"case": "{{ state.remaining_cases[0] }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["passed", "failed", "blocked"]},
                            "evidence": {"type": "string"},
                        },
                        "required": ["case_id", "status", "evidence"],
                    },
                    "update": {
                        "remaining_cases": "{{ state.remaining_cases[1:] }}",
                        "results": "{{ state.results + [nodes.run_one_case.output] }}",
                    },
                },
                "summarize": {
                    "type": "transform",
                    "output": {"results": "{{ state.results }}", "count": "{{ state.results.length }}"},
                },
            },
            "edges": {
                "start": "generate_cases",
                "generate_cases": "review_cases",
                "review_cases": "route_review",
                "run_cases": "summarize",
            },
        }
    )


def _write_subgraph_parent(tmp_path: Path) -> Path:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    _write_review_subgraph(graph_dir / "review_cases.workflow.yaml", version="v1")
    parent_path = tmp_path / "qa_subgraph.workflow.yaml"
    parent_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: qa_subgraph",
                "durable:",
                "  mode: node_boundary",
                "  default_retry_policy:",
                "    max_attempts: 2",
                "includes:",
                "  graphs:",
                "    review_cases_graph: ./graphs/review_cases.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      requirement: "{{ inputs.requirement }}"',
                "      cases:",
                "        - id: C001",
                "          title: checkout",
                "        - id: C002",
                "          title: retry",
                "  remember_policy:",
                "    type: set_state",
                "    update:",
                "      reviewer_policy: strict",
                "  review_cases:",
                "    type: subgraph",
                "    graph: review_cases_graph",
                "    input:",
                '      cases: "{{ nodes.start.output.cases }}"',
                '      reviewer_hint: "QA lead review"',
                "    state_in:",
                '      policy: "{{ state.reviewer_policy }}"',
                "    state_out:",
                '      reviewed_case_count: "{{ nodes.finish.output.cases.length }}"',
                '      review_policy_seen: "{{ state.policy }}"',
                "    output_schema:",
                "      type: object",
                "      properties:",
                "        approved:",
                "          type: boolean",
                "        cases:",
                "          type: array",
                "        policy:",
                "          type: string",
                "        version:",
                "          type: string",
                "      required: [approved, cases, policy, version]",
                "  summarize:",
                "    type: end",
                "    output:",
                '      approved: "{{ nodes.review_cases.output.approved }}"',
                '      case_count: "{{ state.reviewed_case_count }}"',
                '      policy: "{{ nodes.review_cases.output.policy }}"',
                '      version: "{{ nodes.review_cases.output.version }}"',
                "edges:",
                "  start: remember_policy",
                "  remember_policy: review_cases",
                "  review_cases: summarize",
            ]
        ),
        encoding="utf-8",
    )
    return parent_path


def _write_review_subgraph(path: Path, *, version: str) -> None:
    path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: review_cases_graph",
                "description: Review generated QA cases inside a reusable child graph.",
                "inputs:",
                "  cases:",
                "    type: array",
                "  reviewer_hint:",
                "    type: string",
                "output_schema:",
                "  type: object",
                "  properties:",
                "    approved:",
                "      type: boolean",
                "    cases:",
                "      type: array",
                "    policy:",
                "      type: string",
                "    version:",
                "      type: string",
                "  required: [approved, cases, policy, version]",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      cases: "{{ inputs.cases }}"',
                '      reviewer_hint: "{{ inputs.reviewer_hint }}"',
                '      policy: "{{ state.policy }}"',
                "  review:",
                "    type: human",
                '    prompt: "{{ nodes.start.output.reviewer_hint }}"',
                "    input:",
                '      cases: "{{ nodes.start.output.cases }}"',
                '      policy: "{{ nodes.start.output.policy }}"',
                "    output_schema:",
                "      type: object",
                "      properties:",
                "        approved:",
                "          type: boolean",
                "        cases:",
                "          type: array",
                "      required: [approved, cases]",
                "  finish:",
                "    type: end",
                "    output:",
                '      approved: "{{ nodes.review.output.approved }}"',
                '      cases: "{{ nodes.review.output.cases }}"',
                '      policy: "{{ nodes.start.output.policy }}"',
                f"      version: {version}",
                "edges:",
                "  start: review",
                "  review: finish",
            ]
        ),
        encoding="utf-8",
    )


def test_subgraph_human_resume_uses_scoped_state_and_definition_snapshot(tmp_path: Path) -> None:
    asyncio.run(_run_subgraph_human_resume_uses_scoped_state_and_definition_snapshot(tmp_path))


async def _run_subgraph_human_resume_uses_scoped_state_and_definition_snapshot(tmp_path: Path) -> None:
    parent_path = _write_subgraph_parent(tmp_path)
    workflow = WorkflowConfig.from_yaml(parent_path)
    store = WorkflowStore(InMemoryDatabaseEngine())
    executor = WorkflowExecutor(workflow=workflow, store=store)

    waiting = await executor.run_async(inputs={"requirement": "checkout retry"}, run_id="wf_subgraph_review")

    assert waiting.status.value == "waiting"
    assert waiting.checkpoint_id is not None
    checkpoint = await store.get_checkpoint(waiting.checkpoint_id)
    assert checkpoint is not None
    assert checkpoint.node_id == "review"
    assert checkpoint.scope_path == "review_cases/review"
    assert checkpoint.prompt == "QA lead review"
    assert checkpoint.input["policy"] == "strict"

    folded = await store.fold("wf_subgraph_review")
    assert "review" not in folded.node_outputs
    assert folded.node_context("review_cases")["start"]["output"]["policy"] == "strict"
    assert folded.scoped_state["review_cases"] == {"policy": "strict"}

    _write_review_subgraph(tmp_path / "graphs" / "review_cases.workflow.yaml", version="v2")
    recovered_workflow = WorkflowConfig.from_yaml(parent_path)
    recovered_executor = WorkflowExecutor(workflow=recovered_workflow, store=store)
    completed = await recovered_executor.resume_async(
        run_id="wf_subgraph_review",
        checkpoint_id=waiting.checkpoint_id,
        output={
            "approved": True,
            "cases": [
                {"id": "C001", "title": "checkout"},
                {"id": "C002", "title": "retry"},
            ],
        },
    )

    assert completed.status.value == "completed"
    assert completed.output == {"approved": True, "case_count": 2, "policy": "strict", "version": "v1"}
    assert completed.state == {"reviewer_policy": "strict", "reviewed_case_count": 2, "review_policy_seen": "strict"}

    events = await store.list_events("wf_subgraph_review")
    event_types = [event.event_type for event in events]
    assert "subgraph_started" in event_types
    assert "subgraph_waiting" in event_types
    assert "subgraph_completed" in event_types
    scoped_completed = {(event.node_id, event.scope_path) for event in events if event.event_type == "node_completed"}
    assert ("start", "review_cases/start") in scoped_completed
    assert ("finish", "review_cases/finish") in scoped_completed


def test_subgraph_inside_while_keeps_iteration_scopes(tmp_path: Path) -> None:
    asyncio.run(_run_subgraph_inside_while_keeps_iteration_scopes(tmp_path))


async def _run_subgraph_inside_while_keeps_iteration_scopes(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    child_path = graph_dir / "process_item.workflow.yaml"
    child_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: process_item",
                "inputs:",
                "  items:",
                "    type: array",
                "  results:",
                "    type: array",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      current: "{{ inputs.items[0] }}"',
                '      remaining: "{{ inputs.items[1:] }}"',
                '      results: "{{ inputs.results }}"',
                "  finish:",
                "    type: end",
                "    output:",
                '      remaining: "{{ nodes.start.output.remaining }}"',
                '      results: "{{ nodes.start.output.results + [nodes.start.output.current] }}"',
                "edges:",
                "  start: finish",
            ]
        ),
        encoding="utf-8",
    )
    parent_path = tmp_path / "while_subgraph.workflow.yaml"
    parent_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: while_subgraph",
                "includes:",
                "  graphs:",
                "    process_item: ./graphs/process_item.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      items: "{{ inputs.items }}"',
                "  process_all:",
                "    type: while",
                "    condition: state.remaining.length > 0",
                "    max_iterations: 5",
                '    scope_key: "{{ state.remaining[0] }}"',
                "    body: process_one",
                "    init:",
                '      remaining: "{{ nodes.start.output.items }}"',
                "      results: []",
                "  process_one:",
                "    type: subgraph",
                "    graph: process_item",
                "    input:",
                '      items: "{{ state.remaining }}"',
                '      results: "{{ state.results }}"',
                "    state_out:",
                '      remaining: "{{ nodes.finish.output.remaining }}"',
                '      results: "{{ nodes.finish.output.results }}"',
                "  finish:",
                "    type: end",
                "    output:",
                '      results: "{{ state.results }}"',
                "edges:",
                "  start: process_all",
                "  process_all: finish",
            ]
        ),
        encoding="utf-8",
    )

    store = WorkflowStore(InMemoryDatabaseEngine())
    executor = WorkflowExecutor(workflow=WorkflowConfig.from_yaml(parent_path), store=store)
    result = await executor.run_async(inputs={"items": ["A", "B"]}, run_id="wf_while_subgraph")

    assert result.status.value == "completed"
    assert result.output == {"results": ["A", "B"]}
    events = await store.list_events("wf_while_subgraph")
    completed_scopes = {event.scope_path for event in events if event.event_type == "node_completed"}
    assert "process_all[A]/process_one/start" in completed_scopes
    assert "process_all[B]/process_one/start" in completed_scopes


def test_subgraph_internal_external_write_reconcile(tmp_path: Path) -> None:
    asyncio.run(_run_subgraph_internal_external_write_reconcile(tmp_path))


async def _run_subgraph_internal_external_write_reconcile(tmp_path: Path) -> None:
    graph_dir = tmp_path / "graphs"
    graph_dir.mkdir()
    child_path = graph_dir / "send.workflow.yaml"
    child_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: send_graph",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      value: "{{ inputs.value }}"',
                "  send:",
                "    type: tool",
                "    tool: send_tool",
                "    side_effect: external_write",
                '    idempotency_key: "{{ run.id }}:{{ node.scope_path }}"',
                "    input:",
                '      value: "{{ nodes.start.output.value }}"',
                "  finish:",
                "    type: end",
                "    output:",
                '      sent: "{{ nodes.send.output.sent }}"',
                "edges:",
                "  start: send",
                "  send: finish",
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
                "name: subgraph_uncertain",
                "durable:",
                "  mode: node_boundary",
                "  default_retry_policy:",
                "    max_attempts: 1",
                "    on_uncertain: human_review",
                "includes:",
                "  graphs:",
                "    send_graph: ./graphs/send.workflow.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      value: "{{ inputs.value }}"',
                "  call_send:",
                "    type: subgraph",
                "    graph: send_graph",
                "    input:",
                '      value: "{{ nodes.start.output.value }}"',
                "  finish:",
                "    type: end",
                "    output:",
                '      sent: "{{ nodes.call_send.output.sent }}"',
                "edges:",
                "  start: call_send",
                "  call_send: finish",
            ]
        ),
        encoding="utf-8",
    )
    workflow = WorkflowConfig.from_yaml(parent_path)
    store = WorkflowStore(InMemoryDatabaseEngine())
    snapshot = workflow.definition_snapshot()
    await store.create_run(
        run_id="wf_subgraph_uncertain",
        workflow_name=workflow.name,
        inputs={"value": "ok"},
        definition_snapshot=snapshot,
    )
    await store.append_event(
        run_id="wf_subgraph_uncertain",
        event_type="workflow_run_started",
        payload=event_payload(inputs={"value": "ok"}, definition_snapshot=snapshot),
    )
    await store.append_event(
        run_id="wf_subgraph_uncertain",
        event_type="node_completed",
        node_id="start",
        payload=event_payload(output={"value": "ok"}),
    )
    await store.append_event(
        run_id="wf_subgraph_uncertain",
        event_type="node_started",
        node_id="call_send",
        payload=event_payload(input={"value": "ok"}),
    )
    await store.upsert_node_run(
        run_id="wf_subgraph_uncertain",
        node_id="call_send",
        scope_path="",
        status=WorkflowNodeStatus.RUNNING,
        attempt=1,
        node_input={"value": "ok"},
    )
    await store.append_event(
        run_id="wf_subgraph_uncertain",
        event_type="node_completed",
        node_id="start",
        scope_path="call_send/start",
        payload=event_payload(output={"value": "ok"}),
    )
    await store.append_event(
        run_id="wf_subgraph_uncertain",
        event_type="node_started",
        node_id="send",
        scope_path="call_send/send",
        attempt=1,
        payload=event_payload(input={"value": "ok"}),
    )
    await store.upsert_node_run(
        run_id="wf_subgraph_uncertain",
        node_id="send",
        scope_path="call_send/send",
        status=WorkflowNodeStatus.RUNNING,
        attempt=1,
        node_input={"value": "ok"},
        lease_expires_at=datetime.now() - timedelta(seconds=10),
    )

    executor = WorkflowExecutor(workflow=workflow, store=store)
    uncertain = await executor.run_async(inputs={"value": "ok"}, run_id="wf_subgraph_uncertain")

    assert uncertain.status.value == "uncertain"
    reconciled = await executor.reconcile_async(
        run_id="wf_subgraph_uncertain",
        node_id="send",
        scope_path="call_send/send",
        decision="completed",
        output={"sent": True},
    )

    assert reconciled.status.value == "completed"
    assert reconciled.output == {"sent": True}


def test_full_qa_workflow_human_resume_durable_recovery() -> None:
    asyncio.run(_run_full_qa_workflow_human_resume_durable_recovery())


async def _run_full_qa_workflow_human_resume_durable_recovery() -> None:
    store = WorkflowStore(InMemoryDatabaseEngine())
    calls: list[tuple[str, str]] = []

    async def fake_agent_runner(
        *,
        agent_name: str,
        agent_config: object | None,
        input_data: JsonObject,
        output_schema: JsonObject | None,
        run_id: str,
        node_id: str,
        scope_path: str,
    ) -> JsonObject:
        calls.append((agent_name, scope_path))
        if agent_name == "qa_planner":
            return {
                "cases": [
                    {"id": "C001", "title": "checkout", "steps": ["open"], "expected": "works"},
                    {"id": "C002", "title": "retry", "steps": ["retry"], "expected": "works"},
                ]
            }
        case = input_data["case"]
        if isinstance(case, dict):
            return {"case_id": str(case["id"]), "status": "passed", "evidence": f"ran {case['id']}"}
        raise AssertionError("case input was not rendered as an object")

    executor = WorkflowExecutor(workflow=_qa_workflow(), store=store, agent_runner=fake_agent_runner)
    waiting = await executor.run_async(inputs={"requirement": "checkout retry"}, run_id="wf_test_qa")

    assert waiting.status.value == "waiting"
    assert waiting.checkpoint_id is not None
    assert calls == [("qa_planner", "")]

    recovered_executor = WorkflowExecutor(workflow=_qa_workflow(), store=store, agent_runner=fake_agent_runner)
    completed = await recovered_executor.resume_async(
        run_id="wf_test_qa",
        checkpoint_id=waiting.checkpoint_id,
        output={
            "approved": True,
            "cases": [
                {"id": "C001", "title": "checkout", "steps": ["open"], "expected": "works"},
                {"id": "C002", "title": "retry", "steps": ["retry"], "expected": "works"},
            ],
        },
    )

    assert completed.status.value == "completed"
    assert completed.output == {
        "results": [
            {"case_id": "C001", "status": "passed", "evidence": "ran C001"},
            {"case_id": "C002", "status": "passed", "evidence": "ran C002"},
        ],
        "count": 2,
    }
    assert calls == [
        ("qa_planner", ""),
        ("qa_runner", "run_cases[case-C001]"),
        ("qa_runner", "run_cases[case-C002]"),
    ]

    events = await store.list_events("wf_test_qa")
    assert [event.event_type for event in events].count("node_completed") >= 6

    with pytest.raises(WorkflowResumeError):
        await recovered_executor.resume_async(
            run_id="wf_test_qa",
            checkpoint_id=waiting.checkpoint_id,
            output={"approved": True, "cases": []},
        )


def test_workflow_tracing_wraps_each_node_and_preserves_agent_spans() -> None:
    asyncio.run(_run_workflow_tracing_wraps_each_node_and_preserves_agent_spans())


async def _run_workflow_tracing_wraps_each_node_and_preserves_agent_spans() -> None:
    store = WorkflowStore(InMemoryDatabaseEngine())
    tracer = InMemoryTracer()

    async def traced_agent_runner(
        *,
        agent_name: str,
        agent_config: object | None,
        input_data: JsonObject,
        output_schema: JsonObject | None,
        run_id: str,
        node_id: str,
        scope_path: str,
    ) -> JsonObject:
        del agent_config, output_schema, run_id, node_id
        with TraceContext(tracer, f"Agent: {agent_name}", SpanType.AGENT, {"input": input_data}):
            if agent_name == "qa_planner":
                return {
                    "cases": [
                        {"id": "C001", "title": "checkout", "steps": ["open"], "expected": "works"},
                        {"id": "C002", "title": "retry", "steps": ["retry"], "expected": "works"},
                    ]
                }
            case = input_data["case"]
            if isinstance(case, dict):
                return {"case_id": str(case["id"]), "status": "passed", "evidence": f"ran {case['id']} in {scope_path}"}
            raise AssertionError("case input was not rendered as an object")

    executor = WorkflowExecutor(workflow=_qa_workflow(), store=store, agent_runner=traced_agent_runner, tracer=tracer)
    waiting = await executor.run_async(inputs={"requirement": "checkout retry"}, run_id="wf_trace")
    assert waiting.status.value == "waiting"
    assert waiting.checkpoint_id is not None

    completed = await executor.resume_async(
        run_id="wf_trace",
        checkpoint_id=waiting.checkpoint_id,
        output={
            "approved": True,
            "cases": [
                {"id": "C001", "title": "checkout", "steps": ["open"], "expected": "works"},
                {"id": "C002", "title": "retry", "steps": ["retry"], "expected": "works"},
            ],
        },
    )
    assert completed.status.value == "completed"

    spans = list(tracer.spans.values())
    workflow_spans = [span for span in spans if span.type == SpanType.WORKFLOW]
    node_spans = [span for span in spans if span.type == SpanType.WORKFLOW_NODE]
    agent_spans = [span for span in spans if span.type == SpanType.AGENT]

    assert len(workflow_spans) == 2
    node_ids = [str(span.attributes["workflow.node_id"]) for span in node_spans]
    assert node_ids == [
        "start",
        "generate_cases",
        "review_cases",
        "route_review",
        "run_cases",
        "run_one_case",
        "run_one_case",
        "summarize",
    ]

    node_span_by_id = _single_node_span(node_spans)
    assert node_span_by_id["start"].parent_id == workflow_spans[0].id
    assert node_span_by_id["generate_cases"].parent_id == workflow_spans[0].id
    assert node_span_by_id["review_cases"].parent_id == workflow_spans[0].id
    assert node_span_by_id["route_review"].parent_id == workflow_spans[1].id
    assert node_span_by_id["run_cases"].parent_id == workflow_spans[1].id
    assert node_span_by_id["summarize"].parent_id == workflow_spans[1].id

    run_one_case_spans = [span for span in node_spans if span.attributes["workflow.node_id"] == "run_one_case"]
    assert len(run_one_case_spans) == 2
    assert {span.parent_id for span in run_one_case_spans} == {node_span_by_id["run_cases"].id}
    assert {str(span.attributes["workflow.scope_path"]) for span in run_one_case_spans} == {
        "run_cases[case-C001]",
        "run_cases[case-C002]",
    }

    agent_parents = {span.name: span.parent_id for span in agent_spans}
    assert agent_parents["Agent: qa_planner"] == node_span_by_id["generate_cases"].id
    runner_parent_ids = {span.parent_id for span in agent_spans if span.name == "Agent: qa_runner"}
    assert runner_parent_ids == {span.id for span in run_one_case_spans}


def _single_node_span(node_spans: list[Span]) -> dict[str, Span]:
    result: dict[str, Span] = {}
    for span in node_spans:
        node_id = str(span.attributes["workflow.node_id"])
        if node_id == "run_one_case":
            continue
        result[node_id] = span
    return result


def test_tool_and_mcp_nodes_use_registered_tools() -> None:
    asyncio.run(_run_tool_and_mcp_nodes_use_registered_tools())


async def _run_tool_and_mcp_nodes_use_registered_tools() -> None:
    registry = ToolRegistry()
    registry.add_source(
        "test",
        [
            Tool(
                name="echo_tool",
                description="Echo a value.",
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
                implementation=lambda value: {"echo": value},
            )
        ],
    )
    mcp_tool = Tool(
        name="mcp_echo",
        description="Fake MCP echo.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        implementation=lambda value: {"mcp_echo": value},
    )
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "tool_mcp_workflow",
            "nodes": {
                "start": {"type": "start", "output": {"value": "{{ inputs.value }}"}},
                "tool_node": {"type": "tool", "tool": "echo_tool", "input": {"value": "{{ nodes.start.output.value }}"}},
                "mcp_node": {"type": "mcp", "server": "fake", "tool": "mcp_echo", "input": {"value": "{{ nodes.tool_node.output.echo }}"}},
            },
            "edges": {"start": "tool_node", "tool_node": "mcp_node"},
        }
    )

    store = WorkflowStore(InMemoryDatabaseEngine())
    executor = WorkflowExecutor(workflow=workflow, store=store, tool_registry=registry, mcp_tools={"fake.mcp_echo": mcp_tool})

    result = await executor.run_async(inputs={"value": "hello"}, run_id="wf_tool_mcp")

    assert result.status.value == "completed"
    assert result.output == {"mcp_echo": "hello"}


def test_tool_node_can_load_included_tool_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    asyncio.run(_run_tool_node_can_load_included_tool_yaml(tmp_path, monkeypatch))


async def _run_tool_node_can_load_included_tool_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    impl_path = tmp_path / "included_tool_impl.py"
    impl_path.write_text(
        "\n".join(
            [
                "def included_echo_tool(value: str) -> dict[str, str]:",
                "    return {'included': value}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    tool_path = tmp_path / "included_echo.tool.yaml"
    tool_path.write_text(
        "\n".join(
            [
                "type: tool",
                "name: included_echo",
                "description: Included echo tool.",
                "binding: included_tool_impl:included_echo_tool",
                "input_schema:",
                "  type: object",
                "  properties:",
                "    value:",
                "      type: string",
                "  required: [value]",
            ]
        ),
        encoding="utf-8",
    )
    workflow_path = tmp_path / "included_tool.workflow.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "type: workflow",
                'version: "1"',
                "name: included_tool_workflow",
                "includes:",
                "  tools:",
                "    included_echo: ./included_echo.tool.yaml",
                "nodes:",
                "  start:",
                "    type: start",
                "    output:",
                '      value: "{{ inputs.value }}"',
                "  call_tool:",
                "    type: tool",
                "    tool: included_echo",
                "    input:",
                '      value: "{{ nodes.start.output.value }}"',
                "edges:",
                "  start: call_tool",
            ]
        ),
        encoding="utf-8",
    )

    workflow = WorkflowConfig.from_yaml(workflow_path)
    executor = WorkflowExecutor(workflow=workflow, store=WorkflowStore(InMemoryDatabaseEngine()))
    result = await executor.run_async(inputs={"value": "from include"}, run_id="wf_included_tool")

    assert result.status.value == "completed"
    assert result.output == {"included": "from include"}


@pytest.mark.parametrize("tool_call_mode", ["structured", "xml"])
def test_agent_node_complete_task_supports_structured_and_xml_modes(tool_call_mode: str) -> None:
    asyncio.run(_run_agent_node_complete_task_supports_structured_and_xml_modes(tool_call_mode))


async def _run_agent_node_complete_task_supports_structured_and_xml_modes(tool_call_mode: str) -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": f"agent_complete_task_{tool_call_mode}",
            "nodes": {
                "start": {"type": "start", "output": {"requirement": "{{ inputs.requirement }}"}},
                "agent_node": {
                    "type": "agent",
                    "agent": "structured_worker",
                    "input": {"requirement": "{{ nodes.start.output.requirement }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["passed", "failed"]},
                            "evidence": {"type": "string"},
                        },
                        "required": ["status", "evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "edges": {"start": "agent_node"},
        }
    )
    agent_config = AgentConfig(
        name="structured_worker",
        system_prompt="Return a schema-valid task result.",
        llm_config=LLMConfig(
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
        ),
        tool_call_mode=tool_call_mode,
        max_iterations=2,
    )
    observed_modes: list[str] = []

    async def fake_call_llm_async(
        self: LLMCaller,
        messages: object,
        **kwargs: object,
    ) -> ModelResponse:
        del self, messages
        observed_modes.append(str(kwargs["tool_call_mode"]))
        if tool_call_mode == "structured":
            tools = kwargs["tools"]
            assert isinstance(tools, list)
            tool_names = [tool.get("name") for tool in tools if isinstance(tool, dict)]
            assert "complete_task" in tool_names
            return ModelResponse(
                content="",
                tool_calls=[
                    ModelToolCall(
                        call_id="call_complete",
                        name="complete_task",
                        arguments={"status": "passed", "evidence": "structured complete_task"},
                        raw_arguments='{"status":"passed","evidence":"structured complete_task"}',
                    )
                ],
            )
        return ModelResponse(
            content=(
                "<tool_use>"
                "<tool_name>complete_task</tool_name>"
                "<parameter>"
                "<status>passed</status>"
                "<evidence>xml complete_task</evidence>"
                "</parameter>"
                "</tool_use>"
            )
        )

    store = WorkflowStore(InMemoryDatabaseEngine())
    session_manager = SessionManager(engine=InMemoryDatabaseEngine())
    with (
        patch("nexau.archs.main_sub.agent.openai") as mock_openai,
        patch.object(LLMCaller, "call_llm_async", new=fake_call_llm_async),
    ):
        mock_openai.OpenAI.return_value = Mock()
        mock_openai.AsyncOpenAI.return_value = Mock()
        executor = WorkflowExecutor(workflow=workflow, store=store, agents={"structured_worker": agent_config})
        result = await executor.run_async(
            inputs={"requirement": "prove complete_task works"},
            run_id=f"wf_agent_{tool_call_mode}",
            session_manager=session_manager,
        )

    assert result.status.value == "completed"
    assert result.output == {"status": "passed", "evidence": f"{tool_call_mode} complete_task"}
    assert observed_modes == [tool_call_mode]


def test_agent_node_complete_task_preserves_schema_result_field() -> None:
    asyncio.run(_run_agent_node_complete_task_preserves_schema_result_field())


async def _run_agent_node_complete_task_preserves_schema_result_field() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "agent_complete_task_result_field",
            "nodes": {
                "start": {"type": "start", "output": {"requirement": "{{ inputs.requirement }}"}},
                "agent_node": {
                    "type": "agent",
                    "agent": "result_worker",
                    "input": {"requirement": "{{ nodes.start.output.requirement }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                        "required": ["result"],
                        "additionalProperties": False,
                    },
                },
            },
            "edges": {"start": "agent_node"},
        }
    )
    agent_config = AgentConfig(
        name="result_worker",
        system_prompt="Return a schema-valid task result.",
        llm_config=LLMConfig(model="test-model", base_url="https://example.invalid/v1", api_key="test-key"),
        tool_call_mode="structured",
        max_iterations=2,
    )

    async def fake_call_llm_async(
        self: LLMCaller,
        messages: object,
        **kwargs: object,
    ) -> ModelResponse:
        del self, messages, kwargs
        return ModelResponse(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="call_complete",
                    name="complete_task",
                    arguments={"result": "kept as object field"},
                    raw_arguments='{"result":"kept as object field"}',
                )
            ],
        )

    store = WorkflowStore(InMemoryDatabaseEngine())
    session_manager = SessionManager(engine=InMemoryDatabaseEngine())
    with (
        patch("nexau.archs.main_sub.agent.openai") as mock_openai,
        patch.object(LLMCaller, "call_llm_async", new=fake_call_llm_async),
    ):
        mock_openai.OpenAI.return_value = Mock()
        mock_openai.AsyncOpenAI.return_value = Mock()
        executor = WorkflowExecutor(workflow=workflow, store=store, agents={"result_worker": agent_config})
        result = await executor.run_async(
            inputs={"requirement": "return a result field"},
            run_id="wf_agent_result_field",
            session_manager=session_manager,
        )

    assert result.status.value == "completed"
    assert result.output == {"result": "kept as object field"}


def test_note_set_state_and_end_nodes() -> None:
    asyncio.run(_run_note_set_state_and_end_nodes())


async def _run_note_set_state_and_end_nodes() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "state_nodes",
            "nodes": {
                "start": {"type": "start", "output": {"value": "{{ inputs.value }}"}},
                "note": {"type": "note", "description": "No-op annotation."},
                "set_value": {"type": "set_state", "update": {"remembered": "{{ nodes.start.output.value }}"}},
                "finish": {"type": "end", "output": {"remembered": "{{ state.remembered }}"}},
            },
            "edges": {"start": "note", "note": "set_value", "set_value": "finish"},
        }
    )
    executor = WorkflowExecutor(workflow=workflow, store=WorkflowStore(InMemoryDatabaseEngine()))

    result = await executor.run_async(inputs={"value": "stored"}, run_id="wf_state_nodes")

    assert result.status.value == "completed"
    assert result.output == {"remembered": "stored"}
    assert result.state == {"remembered": "stored"}


def test_lease_expiry_retries_read_only_and_uncertains_external_write() -> None:
    asyncio.run(_run_lease_expiry_retries_read_only_and_uncertains_external_write())


async def _run_lease_expiry_retries_read_only_and_uncertains_external_write() -> None:
    read_workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "read_retry",
            "nodes": {
                "start": {"type": "start", "output": {"value": "{{ inputs.value }}"}},
                "read": {"type": "tool", "tool": "read_tool", "input": {"value": "{{ nodes.start.output.value }}"}},
            },
            "edges": {"start": "read"},
        }
    )
    read_tool = Tool(
        name="read_tool",
        description="Read.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        implementation=lambda value: {"value": value},
    )
    store = WorkflowStore(InMemoryDatabaseEngine())
    await store.create_run(run_id="wf_retry", workflow_name="read_retry", inputs={"value": "ok"}, definition_snapshot={})
    await store.append_event(run_id="wf_retry", event_type="workflow_run_started", payload=event_payload(inputs={"value": "ok"}))
    await store.append_event(run_id="wf_retry", event_type="node_completed", node_id="start", payload=event_payload(output={"value": "ok"}))
    await store.append_event(run_id="wf_retry", event_type="node_started", node_id="read", attempt=1)
    await store.upsert_node_run(
        run_id="wf_retry",
        node_id="read",
        scope_path="",
        status=WorkflowNodeStatus.RUNNING,
        attempt=1,
        lease_expires_at=datetime.now() - timedelta(seconds=10),
    )

    result = await WorkflowExecutor(workflow=read_workflow, store=store, tools={"read_tool": read_tool}).run_async(
        inputs={"value": "ok"},
        run_id="wf_retry",
    )

    assert result.status.value == "completed"
    assert result.output == {"value": "ok"}
    assert any(event.event_type == "node_retry_scheduled" for event in await store.list_events("wf_retry"))

    external_workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "external_uncertain",
            "durable": {"mode": "node_boundary", "default_retry_policy": {"max_attempts": 1, "on_uncertain": "human_review"}},
            "nodes": {
                "start": {"type": "start", "output": {"value": "{{ inputs.value }}"}},
                "send": {
                    "type": "tool",
                    "tool": "send_tool",
                    "side_effect": "external_write",
                    "idempotency_key": "{{ run.id }}:send",
                    "input": {"value": "{{ nodes.start.output.value }}"},
                },
            },
            "edges": {"start": "send"},
        }
    )
    uncertain_store = WorkflowStore(InMemoryDatabaseEngine())
    await uncertain_store.create_run(
        run_id="wf_uncertain",
        workflow_name="external_uncertain",
        inputs={"value": "ok"},
        definition_snapshot={},
    )
    await uncertain_store.append_event(
        run_id="wf_uncertain",
        event_type="workflow_run_started",
        payload=event_payload(inputs={"value": "ok"}),
    )
    await uncertain_store.append_event(
        run_id="wf_uncertain",
        event_type="node_completed",
        node_id="start",
        payload=event_payload(output={"value": "ok"}),
    )
    await uncertain_store.append_event(run_id="wf_uncertain", event_type="node_started", node_id="send", attempt=1)
    await uncertain_store.upsert_node_run(
        run_id="wf_uncertain",
        node_id="send",
        scope_path="",
        status=WorkflowNodeStatus.RUNNING,
        attempt=1,
        lease_expires_at=datetime.now() - timedelta(seconds=10),
    )

    uncertain = await WorkflowExecutor(workflow=external_workflow, store=uncertain_store, tools={"send_tool": read_tool}).run_async(
        inputs={"value": "ok"},
        run_id="wf_uncertain",
    )

    assert uncertain.status.value == "uncertain"
    reconciled = await WorkflowExecutor(workflow=external_workflow, store=uncertain_store, tools={"send_tool": read_tool}).reconcile_async(
        run_id="wf_uncertain",
        node_id="send",
        decision="completed",
        output={"value": "ok"},
    )
    assert reconciled.status.value == "completed"
