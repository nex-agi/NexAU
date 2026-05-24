"""HTTP routes for durable workflow runs.

RFC-0027: Transport API for workflow run/resume/reconcile/events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.core import BaseTracer
from nexau.archs.workflow import WorkflowConfig, WorkflowExecutionError, WorkflowExecutor, WorkflowResumeError, WorkflowStore
from nexau.archs.workflow.executor import AgentNodeRunner
from nexau.archs.workflow.structured_output import StructuredOutputError
from nexau.archs.workflow.types import JsonObject, json_object


def _empty_json_object() -> JsonObject:
    return {}


class WorkflowRunRequest(BaseModel):
    """Request body for starting a workflow run."""

    model_config = ConfigDict(extra="forbid")

    inputs: JsonObject = Field(default_factory=_empty_json_object)
    run_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None


class WorkflowResumeRequest(BaseModel):
    """Request body for resuming a workflow checkpoint."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str
    output: JsonObject
    user_id: str | None = None
    session_id: str | None = None


class WorkflowCancelRequest(BaseModel):
    """Request body for cancelling a workflow run."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_id: str | None = None


class WorkflowReconcileRequest(BaseModel):
    """Request body for reconciling an uncertain node."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    scope_path: str = ""
    decision: str
    output: JsonObject | None = None


class WorkflowRouteResponse(BaseModel):
    """HTTP response for workflow operations."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: str
    output: JsonObject | None = None
    state: JsonObject = Field(default_factory=_empty_json_object)
    checkpoint_id: str | None = None


@dataclass
class RegisteredWorkflow:
    """Runtime dependencies for one registered workflow."""

    workflow: WorkflowConfig
    agents: dict[str, AgentConfig]
    tools: dict[str, Tool]
    mcp_tools: dict[str, Tool]
    tool_registry: ToolRegistry | None
    agent_runner: AgentNodeRunner | None
    tracer: BaseTracer | None


class WorkflowRegistryProtocol(Protocol):
    """Registry protocol consumed by HTTP routes."""

    def create_executor(self, workflow_name: str) -> WorkflowExecutor:
        """Return an executor for *workflow_name*."""

        ...


class WorkflowRegistry:
    """In-process registry for workflow HTTP routes."""

    def __init__(self, store: WorkflowStore):
        self._store = store
        self._workflows: dict[str, RegisteredWorkflow] = {}

    def register(
        self,
        workflow: WorkflowConfig,
        *,
        agents: dict[str, AgentConfig] | None = None,
        tools: dict[str, Tool] | None = None,
        mcp_tools: dict[str, Tool] | None = None,
        tool_registry: ToolRegistry | None = None,
        agent_runner: AgentNodeRunner | None = None,
        tracer: BaseTracer | None = None,
    ) -> None:
        """Register a workflow config and runtime dependencies."""

        self._workflows[workflow.name] = RegisteredWorkflow(
            workflow=workflow,
            agents=agents or {},
            tools=tools or {},
            mcp_tools=mcp_tools or {},
            tool_registry=tool_registry,
            agent_runner=agent_runner,
            tracer=tracer,
        )

    def create_executor(self, workflow_name: str) -> WorkflowExecutor:
        """Create an executor for a registered workflow."""

        registered = self._workflows.get(workflow_name)
        if registered is None:
            raise KeyError(workflow_name)
        return WorkflowExecutor(
            workflow=registered.workflow,
            store=self._store,
            agents=registered.agents,
            tools=registered.tools,
            mcp_tools=registered.mcp_tools,
            tool_registry=registered.tool_registry,
            agent_runner=registered.agent_runner,
            tracer=registered.tracer,
        )

    @property
    def store(self) -> WorkflowStore:
        """Return the backing workflow store."""

        return self._store


def create_workflow_router(registry: WorkflowRegistry) -> APIRouter:
    """Create workflow HTTP routes."""

    router = APIRouter(tags=["workflow"])

    @router.post("/workflows/{workflow_name}/runs")
    async def start_workflow(workflow_name: str, request: WorkflowRunRequest) -> WorkflowRouteResponse:
        try:
            executor = registry.create_executor(workflow_name)
            result = await executor.run_async(
                inputs=request.inputs,
                run_id=request.run_id,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            return _response(result.run_id, result.status.value, result.output, result.state, result.checkpoint_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_name}") from exc
        except WorkflowExecutionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/workflow-runs/{run_id}")
    async def get_workflow_run(run_id: str) -> WorkflowRouteResponse:
        run = await registry.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Workflow run not found: {run_id}")
        return _response(
            run.run_id,
            run.status,
            json_object(run.output, label="run.output") if run.output is not None else None,
            json_object(run.state, label="run.state"),
            run.waiting_checkpoint_id,
        )

    @router.get("/workflow-runs/{run_id}/events")
    async def workflow_events(run_id: str) -> StreamingResponse:
        events = await registry.store.list_events(run_id)

        async def stream_events():
            for event in events:
                payload = {
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "node_id": event.node_id,
                    "scope_path": event.scope_path,
                    "attempt": event.attempt,
                    "graph_id": event.payload.get("graph_id"),
                    "parent_node_id": event.payload.get("parent_node_id"),
                    "subgraph": event.payload.get("subgraph"),
                    "depth": event.payload.get("depth"),
                    "payload": event.payload,
                    "created_at": event.created_at.isoformat(),
                }
                yield f"event: {event.event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(stream_events(), media_type="text/event-stream")

    @router.post("/workflow-runs/{run_id}/resume")
    async def resume_workflow(run_id: str, request: WorkflowResumeRequest) -> WorkflowRouteResponse:
        try:
            run = await registry.store.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Workflow run not found: {run_id}")
            executor = registry.create_executor(run.workflow_name)
            result = await executor.resume_async(
                run_id=run_id,
                checkpoint_id=request.checkpoint_id,
                output=request.output,
                user_id=request.user_id,
                session_id=request.session_id,
            )
            return _response(result.run_id, result.status.value, result.output, result.state, result.checkpoint_id)
        except WorkflowResumeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except StructuredOutputError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/workflow-runs/{run_id}/cancel")
    async def cancel_workflow(run_id: str, request: WorkflowCancelRequest) -> WorkflowRouteResponse:
        run = await registry.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Workflow run not found: {run_id}")
        executor = registry.create_executor(run.workflow_name)
        result = await executor.cancel_async(run_id=run_id, checkpoint_id=request.checkpoint_id)
        return _response(result.run_id, result.status.value, result.output, result.state, result.checkpoint_id)

    @router.post("/workflow-runs/{run_id}/reconcile")
    async def reconcile_workflow(run_id: str, request: WorkflowReconcileRequest) -> WorkflowRouteResponse:
        try:
            run = await registry.store.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail=f"Workflow run not found: {run_id}")
            executor = registry.create_executor(run.workflow_name)
            result = await executor.reconcile_async(
                run_id=run_id,
                node_id=request.node_id,
                scope_path=request.scope_path,
                decision=request.decision,
                output=request.output,
            )
            return _response(result.run_id, result.status.value, result.output, result.state, result.checkpoint_id)
        except WorkflowExecutionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


def _response(
    run_id: str,
    status: str,
    output: JsonObject | None,
    state: JsonObject,
    checkpoint_id: str | None,
) -> WorkflowRouteResponse:
    return WorkflowRouteResponse(
        run_id=run_id,
        status=status,
        output=output,
        state=state,
        checkpoint_id=checkpoint_id,
    )
