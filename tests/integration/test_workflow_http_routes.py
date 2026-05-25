"""RFC-0027 workflow HTTP route integration tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import cast
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexau import AgentConfig
from nexau.archs.llm.llm_aggregators.events import (
    Event,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.tool.tool import Tool
from nexau.archs.transports.http.workflow_routes import WorkflowRegistry, create_workflow_router
from nexau.archs.workflow import WorkflowConfig, WorkflowStore
from nexau.archs.workflow.types import JsonObject, json_object


def _review_workflow() -> WorkflowConfig:
    return WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "review_flow",
            "nodes": {
                "start": {"type": "start", "output": {"value": "{{ inputs.value }}"}},
                "review": {
                    "type": "human",
                    "prompt": "Approve value.",
                    "input": {"value": "{{ nodes.start.output.value }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}, "value": {"type": "string"}},
                        "required": ["approved", "value"],
                    },
                },
                "finish": {
                    "type": "transform",
                    "output": {
                        "approved": "{{ nodes.review.output.approved }}",
                        "value": "{{ nodes.review.output.value }}",
                    },
                },
            },
            "edges": {"start": "review", "review": "finish"},
        }
    )


def _stream_entries(body: str) -> list[tuple[str, JsonObject]]:
    entries: list[tuple[str, JsonObject]] = []
    for block in body.strip().split("\n\n"):
        event_name = ""
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))
        if event_name and data_lines:
            entries.append((event_name, json_object(json.loads("\n".join(data_lines)), label="sse.data")))
    return entries


def _agent_event_handler(caller: LLMCaller) -> Callable[[Event], None]:
    manager = caller.middleware_manager
    assert manager is not None
    for middleware in manager.middlewares:
        handler = middleware.get_event_handler()
        if handler is not None:
            return cast(Callable[[Event], None], handler)
    raise AssertionError("AgentEventsMiddleware handler was not installed")


def _emit_agent_stream_events(caller: LLMCaller, agent_run_id: str) -> None:
    handler = _agent_event_handler(caller)
    handler(TextMessageStartEvent(message_id="msg_live", role="assistant", run_id=agent_run_id, timestamp=1))
    handler(TextMessageContentEvent(message_id="msg_live", delta="checking release", timestamp=2))
    handler(
        ToolCallStartEvent(
            tool_call_id="call_complete",
            tool_call_name="complete_task",
            parent_message_id="msg_live",
            timestamp=3,
        )
    )
    handler(
        ToolCallArgsEvent(
            tool_call_id="call_complete",
            delta='{"status":"passed","evidence":"streamed tool call"}',
            timestamp=4,
        )
    )
    handler(ToolCallEndEvent(tool_call_id="call_complete", timestamp=5))
    handler(TextMessageEndEvent(message_id="msg_live", timestamp=6))


def test_workflow_http_start_query_events_resume_cancel() -> None:
    store = WorkflowStore(InMemoryDatabaseEngine())
    registry = WorkflowRegistry(store)
    registry.register(_review_workflow())
    app = FastAPI()
    app.include_router(create_workflow_router(registry))
    client = TestClient(app)

    start = client.post("/workflows/review_flow/runs", json={"run_id": "wf_http", "inputs": {"value": "ok"}})
    assert start.status_code == 200
    start_body: JsonObject = start.json()
    checkpoint_id = start_body["checkpoint_id"]
    assert start_body["status"] == "waiting"
    assert isinstance(checkpoint_id, str)

    query = client.get("/workflow-runs/wf_http")
    assert query.status_code == 200
    assert query.json()["status"] == "waiting"

    events = client.get("/workflow-runs/wf_http/events")
    assert events.status_code == 200
    assert "checkpoint_created" in events.text

    invalid_resume = client.post(
        "/workflow-runs/wf_http/resume",
        json={"checkpoint_id": checkpoint_id, "output": {"approved": "yes", "value": "ok"}},
    )
    assert invalid_resume.status_code == 400

    resume = client.post(
        "/workflow-runs/wf_http/resume",
        json={"checkpoint_id": checkpoint_id, "output": {"approved": True, "value": "ok"}},
    )
    assert resume.status_code == 200
    assert resume.json()["status"] == "completed"
    assert resume.json()["output"] == {"approved": True, "value": "ok"}

    cancel_completed = client.post("/workflow-runs/wf_http/cancel", json={})
    assert cancel_completed.status_code == 200
    assert cancel_completed.json()["status"] == "cancelled"


def test_workflow_http_parallel_map_events_include_item_metadata() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "parallel_http",
            "nodes": {
                "start": {"type": "start", "output": {"items": "{{ inputs.items }}"}},
                "map_items": {
                    "type": "parallel_map",
                    "items": "{{ nodes.start.output.items }}",
                    "item_key": "{{ item.id }}",
                    "max_concurrency": 2,
                    "body": "echo_item",
                    "failure_policy": "collect_errors",
                },
                "echo_item": {"type": "tool", "tool": "echo_tool", "input": {"value": "{{ item.value }}"}},
            },
            "edges": {"start": "map_items"},
        }
    )
    tool = Tool(
        name="echo_tool",
        description="Echo.",
        input_schema={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        implementation=lambda value: {"echo": value},
    )
    store = WorkflowStore(InMemoryDatabaseEngine())
    registry = WorkflowRegistry(store)
    registry.register(workflow, tools={"echo_tool": tool})
    app = FastAPI()
    app.include_router(create_workflow_router(registry))
    client = TestClient(app)

    start = client.post(
        "/workflows/parallel_http/runs",
        json={"run_id": "wf_parallel_http", "inputs": {"items": [{"id": "A", "value": "alpha"}, {"id": "B", "value": "beta"}]}},
    )
    assert start.status_code == 200
    assert start.json()["status"] == "completed"
    assert start.json()["waiting_checkpoint_ids"] == []

    query = client.get("/workflow-runs/wf_parallel_http")
    assert query.status_code == 200
    assert query.json()["waiting_checkpoint_ids"] == []

    events = client.get("/workflow-runs/wf_parallel_http/events")
    assert events.status_code == 200
    assert "parallel_item_completed" in events.text
    assert '"parallel_node_id": "map_items"' in events.text
    assert '"item_key": "A"' in events.text
    assert '"item_scope_path": "map_items[A]"' in events.text
    assert '"body_node_id": "echo_item"' in events.text


def test_workflow_http_live_stream_includes_workflow_and_agent_events() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "agent_live_http",
            "nodes": {
                "start": {"type": "start", "output": {"requirement": "{{ inputs.requirement }}"}},
                "agent_node": {
                    "type": "agent",
                    "agent": "release_checker",
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
        name="release_checker",
        system_prompt="Check whether a release is ready.",
        llm_config=LLMConfig(model="test-model", base_url="https://example.invalid/v1", api_key="test-key"),
        tool_call_mode="structured",
        max_iterations=2,
    )

    def fake_call_once_sync(
        self: LLMCaller,
        params: object,
    ) -> ModelResponse:
        del params
        _emit_agent_stream_events(self, "agent_live")
        return ModelResponse(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="call_complete",
                    name="complete_task",
                    arguments={"status": "passed", "evidence": "streamed tool call"},
                    raw_arguments='{"status":"passed","evidence":"streamed tool call"}',
                )
            ],
        )

    store = WorkflowStore(InMemoryDatabaseEngine())
    registry = WorkflowRegistry(store)
    registry.register(workflow, agents={"release_checker": agent_config})
    app = FastAPI()
    app.include_router(create_workflow_router(registry))
    client = TestClient(app)

    with (
        patch("nexau.archs.main_sub.agent.openai") as mock_openai,
        patch.object(LLMCaller, "_call_once_sync", new=fake_call_once_sync),
    ):
        mock_openai.OpenAI.return_value = Mock()
        mock_openai.AsyncOpenAI.return_value = Mock()
        response = client.post(
            "/workflows/agent_live_http/runs/stream",
            json={"run_id": "wf_live_http", "inputs": {"requirement": "ship release"}},
        )

    assert response.status_code == 200
    entries = _stream_entries(response.text)
    event_names = [name for name, _payload in entries]
    assert "workflow_event" in event_names
    assert "agent_event" in event_names
    assert event_names[-1] == "complete"

    workflow_event_types = [
        json_object(payload["workflow_event"], label="workflow_event")["event_type"]
        for name, payload in entries
        if name == "workflow_event"
    ]
    assert "workflow_run_started" in workflow_event_types
    assert "node_started" in workflow_event_types
    assert "node_completed" in workflow_event_types
    assert "workflow_run_completed" in workflow_event_types

    agent_event_types = [
        json_object(json_object(payload["agent"], label="agent")["event"], label="agent.event")["type"]
        for name, payload in entries
        if name == "agent_event"
    ]
    assert "RUN_STARTED" in agent_event_types
    assert "TEXT_MESSAGE_START" in agent_event_types
    assert "TEXT_MESSAGE_CONTENT" in agent_event_types
    assert "TOOL_CALL_START" in agent_event_types
    assert "TOOL_CALL_ARGS" in agent_event_types
    assert "TOOL_CALL_END" in agent_event_types
    assert "TOOL_CALL_RESULT" in agent_event_types
    assert "USAGE_UPDATE" in agent_event_types
    assert "RUN_FINISHED" in agent_event_types

    workflow_context = json_object(
        next(payload["workflow"] for name, payload in entries if name == "agent_event"),
        label="agent.workflow",
    )
    assert workflow_context["node_id"] == "agent_node"
    assert workflow_context["workflow_name"] == "agent_live_http"

    persisted_events = client.get("/workflow-runs/wf_live_http/events")
    assert persisted_events.status_code == 200
    assert "workflow_run_completed" in persisted_events.text
    assert "TEXT_MESSAGE_CONTENT" not in persisted_events.text


def test_workflow_http_live_stream_scopes_parallel_agent_events() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "parallel_agent_live_http",
            "nodes": {
                "start": {"type": "start", "output": {"items": "{{ inputs.items }}"}},
                "run_cases": {
                    "type": "parallel_map",
                    "items": "{{ nodes.start.output.items }}",
                    "item_name": "qa_case",
                    "item_key": "{{ qa_case.id }}",
                    "max_concurrency": 2,
                    "body": "run_one_case",
                    "failure_policy": "collect_errors",
                },
                "run_one_case": {
                    "type": "agent",
                    "agent": "runner",
                    "input": {"case": "{{ qa_case }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string"},
                            "status": {"type": "string", "enum": ["passed", "failed"]},
                        },
                        "required": ["case_id", "status"],
                        "additionalProperties": False,
                    },
                },
            },
            "edges": {"start": "run_cases"},
        }
    )
    agent_config = AgentConfig(
        name="runner",
        system_prompt="Run one QA case.",
        llm_config=LLMConfig(model="test-model", base_url="https://example.invalid/v1", api_key="test-key"),
        tool_call_mode="structured",
        max_iterations=2,
    )

    async def fake_call_llm_async(
        self: LLMCaller,
        messages: object,
        **kwargs: object,
    ) -> ModelResponse:
        del messages
        agent_state = kwargs.get("agent_state")
        agent_run_id = agent_state.run_id if isinstance(agent_state, AgentState) else "agent_parallel"
        _emit_agent_stream_events(self, agent_run_id)
        return ModelResponse(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="call_complete",
                    name="complete_task",
                    arguments={"case_id": "case", "status": "passed"},
                    raw_arguments='{"case_id":"case","status":"passed"}',
                )
            ],
        )

    store = WorkflowStore(InMemoryDatabaseEngine())
    registry = WorkflowRegistry(store)
    registry.register(workflow, agents={"runner": agent_config})
    app = FastAPI()
    app.include_router(create_workflow_router(registry))
    client = TestClient(app)

    with (
        patch("nexau.archs.main_sub.agent.openai") as mock_openai,
        patch.object(LLMCaller, "call_llm_async", new=fake_call_llm_async),
    ):
        mock_openai.OpenAI.return_value = Mock()
        mock_openai.AsyncOpenAI.return_value = Mock()
        response = client.post(
            "/workflows/parallel_agent_live_http/runs/stream",
            json={
                "run_id": "wf_parallel_agent_live",
                "inputs": {"items": [{"id": "A"}, {"id": "B"}]},
            },
        )

    assert response.status_code == 200
    entries = _stream_entries(response.text)
    agent_workflow_contexts = [
        json_object(payload["workflow"], label="agent.workflow") for name, payload in entries if name == "agent_event"
    ]
    item_keys = {context["item_key"] for context in agent_workflow_contexts}
    item_scope_paths = {context["item_scope_path"] for context in agent_workflow_contexts}
    assert item_keys == {"A", "B"}
    assert item_scope_paths == {"run_cases[A]", "run_cases[B]"}
    assert {context["parallel_node_id"] for context in agent_workflow_contexts} == {"run_cases"}


def test_workflow_http_resume_stream_includes_followup_agent_events() -> None:
    workflow = WorkflowConfig.model_validate(
        {
            "type": "workflow",
            "version": "1",
            "name": "resume_agent_live_http",
            "nodes": {
                "start": {"type": "start", "output": {"title": "{{ inputs.title }}"}},
                "review": {
                    "type": "human",
                    "prompt": "Approve title.",
                    "input": {"title": "{{ nodes.start.output.title }}"},
                    "output_schema": {
                        "type": "object",
                        "properties": {"approved": {"type": "boolean"}, "title": {"type": "string"}},
                        "required": ["approved", "title"],
                    },
                },
                "agent_node": {
                    "type": "agent",
                    "agent": "release_checker",
                    "input": {"title": "{{ nodes.review.output.title }}"},
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
            "edges": {"start": "review", "review": "agent_node"},
        }
    )
    agent_config = AgentConfig(
        name="release_checker",
        system_prompt="Check a reviewed title.",
        llm_config=LLMConfig(model="test-model", base_url="https://example.invalid/v1", api_key="test-key"),
        tool_call_mode="structured",
        max_iterations=2,
    )

    async def fake_call_llm_async(
        self: LLMCaller,
        messages: object,
        **kwargs: object,
    ) -> ModelResponse:
        del messages
        agent_state = kwargs.get("agent_state")
        agent_run_id = agent_state.run_id if isinstance(agent_state, AgentState) else "agent_resume"
        _emit_agent_stream_events(self, agent_run_id)
        return ModelResponse(
            content="",
            tool_calls=[
                ModelToolCall(
                    call_id="call_complete",
                    name="complete_task",
                    arguments={"status": "passed", "evidence": "resume stream"},
                    raw_arguments='{"status":"passed","evidence":"resume stream"}',
                )
            ],
        )

    store = WorkflowStore(InMemoryDatabaseEngine())
    registry = WorkflowRegistry(store)
    registry.register(workflow, agents={"release_checker": agent_config})
    app = FastAPI()
    app.include_router(create_workflow_router(registry))
    client = TestClient(app)

    start = client.post(
        "/workflows/resume_agent_live_http/runs",
        json={"run_id": "wf_resume_agent_live", "inputs": {"title": "Release checklist"}},
    )
    assert start.status_code == 200
    checkpoint_id = start.json()["checkpoint_id"]
    assert isinstance(checkpoint_id, str)

    with (
        patch("nexau.archs.main_sub.agent.openai") as mock_openai,
        patch.object(LLMCaller, "call_llm_async", new=fake_call_llm_async),
    ):
        mock_openai.OpenAI.return_value = Mock()
        mock_openai.AsyncOpenAI.return_value = Mock()
        response = client.post(
            "/workflow-runs/wf_resume_agent_live/resume/stream",
            json={"checkpoint_id": checkpoint_id, "output": {"approved": True, "title": "Release checklist"}},
        )

    assert response.status_code == 200
    entries = _stream_entries(response.text)
    workflow_event_types = [
        json_object(payload["workflow_event"], label="workflow_event")["event_type"]
        for name, payload in entries
        if name == "workflow_event"
    ]
    agent_event_types = [
        json_object(json_object(payload["agent"], label="agent")["event"], label="agent.event")["type"]
        for name, payload in entries
        if name == "agent_event"
    ]
    assert "checkpoint_resumed" in workflow_event_types
    assert "workflow_run_completed" in workflow_event_types
    assert "RUN_STARTED" in agent_event_types
    assert "TEXT_MESSAGE_CONTENT" in agent_event_types
    assert "TOOL_CALL_RESULT" in agent_event_types
    assert entries[-1][0] == "complete"
