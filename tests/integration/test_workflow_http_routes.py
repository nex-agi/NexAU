"""RFC-0027 workflow HTTP route integration tests."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexau.archs.session import InMemoryDatabaseEngine
from nexau.archs.transports.http.workflow_routes import WorkflowRegistry, create_workflow_router
from nexau.archs.workflow import WorkflowConfig, WorkflowStore
from nexau.archs.workflow.types import JsonObject


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
