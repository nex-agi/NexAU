"""Durable workflow executor.

RFC-0027: Durable WorkflowExecutor
RFC-0028: Workflow 子图支持

This executor implements node-boundary durability over an append-only event log.
It does not replay Python call stacks. Recovery folds persisted events, skips
completed node runs, and resumes from the next safe workflow boundary.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import SessionManager
from nexau.archs.session.models.workflow import WorkflowCheckpointStatus, WorkflowNodeStatus, WorkflowRunStatus
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import BaseTracer, SpanType
from nexau.archs.workflow.config import RetryPolicy, WorkflowConfig, WorkflowNode
from nexau.archs.workflow.expression import EvaluationContext, evaluate_condition, render_template
from nexau.archs.workflow.store import FoldedWorkflowState, WorkflowStore, event_payload, node_run_key
from nexau.archs.workflow.structured_output import run_agent_structured_async, validate_json_schema_output
from nexau.archs.workflow.types import JsonObject, JsonValue, json_object, json_value


class WorkflowExecutionError(RuntimeError):
    """Raised when workflow execution fails."""


class WorkflowResumeError(RuntimeError):
    """Raised when checkpoint resume is invalid."""


class _WorkflowBoundaryPausedError(RuntimeError):
    """Internal control flow for waiting or uncertain nested workflow work."""

    def __init__(self, status: WorkflowRunStatus):
        super().__init__(status.value)
        self.status = status


class AgentNodeRunner(Protocol):
    """Test seam for deterministic Agent node execution."""

    async def __call__(
        self,
        *,
        agent_name: str,
        agent_config: AgentConfig | None,
        input_data: JsonObject,
        output_schema: JsonObject | None,
        run_id: str,
        node_id: str,
        scope_path: str,
    ) -> JsonObject:
        """Run an agent node and return structured output."""

        ...


@dataclass(frozen=True)
class WorkflowRunResult:
    """Workflow run result returned by executor APIs."""

    run_id: str
    status: WorkflowRunStatus
    output: JsonObject | None
    state: JsonObject
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class _NodeExecutionResult:
    output: JsonObject
    raw_output: JsonObject | None = None
    state_patch: JsonObject | None = None


@dataclass(frozen=True)
class _ExecutionFrame:
    """Executable graph frame with isolated inputs, state, and node context."""

    graph: WorkflowConfig
    graph_id: str
    scope_prefix: str
    state_scope_path: str
    inputs: JsonObject
    parent_node_id: str | None = None
    subgraph: str | None = None
    depth: int = 0


class WorkflowExecutor:
    """Execute a workflow config with durable node-boundary persistence."""

    def __init__(
        self,
        *,
        workflow: WorkflowConfig,
        store: WorkflowStore,
        agents: dict[str, AgentConfig] | None = None,
        tools: dict[str, Tool] | None = None,
        tool_registry: ToolRegistry | None = None,
        mcp_tools: dict[str, Tool] | None = None,
        agent_runner: AgentNodeRunner | None = None,
        tracer: BaseTracer | None = None,
        worker_id: str | None = None,
    ):
        self.workflow = workflow
        self.store = store
        self.agents = agents or {}
        self.tools = tools or {}
        self.tool_registry = tool_registry
        self.mcp_tools = mcp_tools or {}
        self.agent_runner = agent_runner
        self.tracer = tracer
        self.worker_id = worker_id or f"wf_worker_{uuid.uuid4().hex[:8]}"

    async def run_async(
        self,
        *,
        inputs: JsonObject,
        run_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        session_manager: SessionManager | None = None,
    ) -> WorkflowRunResult:
        """Start or recover a workflow run."""

        await self.store.setup()
        effective_run_id = run_id or f"wf_run_{uuid.uuid4().hex}"
        existing = await self.store.get_run(effective_run_id)
        if existing is None:
            snapshot = self.workflow.definition_snapshot()
            await self.store.create_run(
                run_id=effective_run_id,
                workflow_name=self.workflow.name,
                inputs=inputs,
                definition_snapshot=snapshot,
                user_id=user_id,
                session_id=session_id,
            )
            await self.store.append_event(
                run_id=effective_run_id,
                event_type="workflow_run_started",
                payload=event_payload(inputs=inputs, definition_snapshot=snapshot),
            )
        elif existing.definition_snapshot:
            # RFC-0028: 恢复运行时优先使用启动时固化的图快照，避免 YAML 文件修改影响未完成 run
            self.workflow = WorkflowConfig.from_snapshot(json_object(existing.definition_snapshot, label="run.definition_snapshot"))

        await self._recover_expired_attempts(effective_run_id)
        return await self._drive_until_boundary(
            run_id=effective_run_id,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )

    def run(
        self,
        *,
        inputs: JsonObject,
        run_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        session_manager: SessionManager | None = None,
    ) -> WorkflowRunResult:
        """Synchronous wrapper for ``run_async``."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.run_async(
                    inputs=inputs,
                    run_id=run_id,
                    user_id=user_id,
                    session_id=session_id,
                    session_manager=session_manager,
                )
            )
        raise RuntimeError("WorkflowExecutor.run() cannot be called from a running event loop; use run_async()")

    async def resume_async(
        self,
        *,
        run_id: str,
        checkpoint_id: str,
        output: JsonObject,
        session_manager: SessionManager | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> WorkflowRunResult:
        """Resume a human checkpoint and continue from the successor node."""

        await self._load_snapshot_for_run(run_id)
        checkpoint = await self.store.get_checkpoint(checkpoint_id)
        if checkpoint is None or checkpoint.run_id != run_id:
            raise WorkflowResumeError(f"Checkpoint not found for run: {checkpoint_id}")
        if checkpoint.status != WorkflowCheckpointStatus.OPEN.value:
            raise WorkflowResumeError(f"Checkpoint is not open: {checkpoint_id}")
        if checkpoint.output_schema is not None:
            validate_json_schema_output(output, json_object(checkpoint.output_schema, label="checkpoint.output_schema"))

        await self.store.update_checkpoint(checkpoint, status=WorkflowCheckpointStatus.RESUMED, output=output)
        await self.store.append_event(
            run_id=run_id,
            event_type="checkpoint_resumed",
            node_id=checkpoint.node_id,
            scope_path=checkpoint.scope_path,
            payload=event_payload(checkpoint_id=checkpoint_id, output=output),
        )
        folded = await self.store.fold(run_id)
        frame = self._frame_for_scope(scope_path=checkpoint.scope_path, folded=folded)
        await self._complete_node(
            run_id=run_id,
            node_id=checkpoint.node_id,
            scope_path=checkpoint.scope_path,
            attempt=1,
            output=output,
            raw_output=output,
            state_patch=None,
            state_scope_path=frame.state_scope_path,
            frame=frame,
        )

        run = await self.store.get_run(run_id)
        if run is not None:
            await self.store.update_run(run, status=WorkflowRunStatus.RUNNING)

        return await self._drive_until_boundary(
            run_id=run_id,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )

    async def cancel_async(self, *, run_id: str, checkpoint_id: str | None = None) -> WorkflowRunResult:
        """Cancel a workflow run or a specific open checkpoint."""

        if checkpoint_id is not None:
            checkpoint = await self.store.get_checkpoint(checkpoint_id)
            if checkpoint is None or checkpoint.run_id != run_id:
                raise WorkflowResumeError(f"Checkpoint not found for run: {checkpoint_id}")
            if checkpoint.status != WorkflowCheckpointStatus.OPEN.value:
                raise WorkflowResumeError(f"Checkpoint is not open: {checkpoint_id}")
            await self.store.update_checkpoint(checkpoint, status=WorkflowCheckpointStatus.CANCELLED)

        await self.store.append_event(run_id=run_id, event_type="workflow_run_cancelled")
        folded = await self.store.fold(run_id)
        run = await self.store.get_run(run_id)
        if run is not None:
            await self.store.update_run(run, status=WorkflowRunStatus.CANCELLED, state=folded.state)
        return WorkflowRunResult(run_id=run_id, status=WorkflowRunStatus.CANCELLED, output=None, state=folded.state)

    async def reconcile_async(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str = "",
        decision: str,
        output: JsonObject | None = None,
    ) -> WorkflowRunResult:
        """Reconcile an uncertain node and continue when possible."""

        await self._load_snapshot_for_run(run_id)
        folded = await self.store.fold(run_id)
        key = node_run_key(run_id, node_id, scope_path)
        if key not in folded.uncertain_node_runs:
            raise WorkflowExecutionError(f"Node is not uncertain: {node_id} {scope_path}")

        match decision:
            case "completed":
                frame = self._frame_for_scope(scope_path=scope_path, folded=folded)
                await self._complete_node(
                    run_id=run_id,
                    node_id=node_id,
                    scope_path=scope_path,
                    attempt=1,
                    output=output or {},
                    raw_output=output or {},
                    state_patch=None,
                    state_scope_path=frame.state_scope_path,
                    frame=frame,
                )
            case "failed":
                await self.store.append_event(run_id=run_id, event_type="workflow_run_failed", node_id=node_id, scope_path=scope_path)
                run = await self.store.get_run(run_id)
                if run is not None:
                    await self.store.update_run(run, status=WorkflowRunStatus.FAILED)
                folded = await self.store.fold(run_id)
                return WorkflowRunResult(run_id=run_id, status=WorkflowRunStatus.FAILED, output=folded.output, state=folded.state)
            case "retry":
                await self.store.append_event(run_id=run_id, event_type="node_retry_scheduled", node_id=node_id, scope_path=scope_path)
                await self.store.upsert_node_run(
                    run_id=run_id,
                    node_id=node_id,
                    scope_path=scope_path,
                    status=WorkflowNodeStatus.SCHEDULED,
                    attempt=1,
                )
            case _:
                raise WorkflowExecutionError("decision must be completed, failed, or retry")

        run = await self.store.get_run(run_id)
        if run is not None:
            await self.store.update_run(run, status=WorkflowRunStatus.RUNNING)
        return await self._drive_until_boundary(run_id=run_id)

    async def _drive_until_boundary(
        self,
        *,
        run_id: str,
        session_manager: SessionManager | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> WorkflowRunResult:
        if self.tracer is None:
            return await self._drive_until_boundary_inner(
                run_id=run_id,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )

        if session_id is not None:
            self.tracer.set_session_id(session_id)
        attributes: dict[str, object] = {
            "workflow.name": self.workflow.name,
            "workflow.run_id": run_id,
        }
        if user_id is not None:
            attributes["workflow.user_id"] = user_id
        if session_id is not None:
            attributes["workflow.session_id"] = session_id
        trace_ctx = TraceContext(
            self.tracer,
            f"Workflow: {self.workflow.name}",
            SpanType.WORKFLOW,
            inputs={"run_id": run_id},
            attributes=attributes,
        )
        with trace_ctx:
            result = await self._drive_until_boundary_inner(
                run_id=run_id,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )
            trace_ctx.set_outputs(
                {
                    "status": result.status.value,
                    "output": result.output,
                    "checkpoint_id": result.checkpoint_id,
                }
            )
            return result

    async def _drive_until_boundary_inner(
        self,
        *,
        run_id: str,
        session_manager: SessionManager | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> WorkflowRunResult:
        while True:
            folded = await self.store.fold(run_id)
            if folded.status in {
                WorkflowRunStatus.WAITING,
                WorkflowRunStatus.COMPLETED,
                WorkflowRunStatus.FAILED,
                WorkflowRunStatus.CANCELLED,
                WorkflowRunStatus.UNCERTAIN,
            }:
                return await self._result_from_folded(folded)

            root_frame = self._root_frame(folded)
            next_node_id = self._next_node_id_for_frame(root_frame, folded)
            if next_node_id is None:
                return await self._complete_workflow(run_id=run_id, folded=folded)

            node = root_frame.graph.nodes[next_node_id]
            if node.type == "note":
                await self._complete_node(
                    run_id=run_id,
                    node_id=next_node_id,
                    scope_path=self._node_scope_path(root_frame, next_node_id),
                    attempt=1,
                    output={},
                    raw_output={},
                    state_patch=None,
                    state_scope_path=root_frame.state_scope_path,
                    frame=root_frame,
                )
                continue

            await self._execute_node_boundary(
                run_id=run_id,
                node_id=next_node_id,
                node=node,
                scope_path=self._node_scope_path(root_frame, next_node_id),
                folded=folded,
                frame=root_frame,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )

    def _root_frame(self, folded: FoldedWorkflowState) -> _ExecutionFrame:
        return _ExecutionFrame(
            graph=self.workflow,
            graph_id=self.workflow.name,
            scope_prefix="",
            state_scope_path="",
            inputs=folded.inputs,
        )

    def _next_node_id_for_frame(self, frame: _ExecutionFrame, folded: FoldedWorkflowState) -> str | None:
        completed_node_id = folded.last_completed_node_id_for_scope(frame.scope_prefix, set(frame.graph.nodes))
        if completed_node_id is None:
            return frame.graph.start_node_id

        completed_node = frame.graph.nodes[completed_node_id]
        if completed_node.type == "if_else":
            branch_scope_path = self._node_scope_path(frame, completed_node_id)
            branch_output = folded.output_for_node(folded.run_id, completed_node_id, branch_scope_path) or {}
            next_value = branch_output.get("next")
            return str(next_value) if isinstance(next_value, str) else None
        if completed_node.type == "end":
            return None

        edge = frame.graph.edges.get(completed_node_id)
        if edge is None:
            return None
        if isinstance(edge, str):
            return edge
        return edge[0] if edge else None

    @staticmethod
    def _node_scope_path(frame: _ExecutionFrame, node_id: str) -> str:
        if frame.scope_prefix == "":
            return ""
        return f"{frame.scope_prefix}/{node_id}"

    async def _load_snapshot_for_run(self, run_id: str) -> None:
        run = await self.store.get_run(run_id)
        if run is not None and run.definition_snapshot:
            self.workflow = WorkflowConfig.from_snapshot(json_object(run.definition_snapshot, label="run.definition_snapshot"))

    def _frame_for_scope(self, *, scope_path: str, folded: FoldedWorkflowState) -> _ExecutionFrame:
        frame = self._root_frame(folded)
        if scope_path == "":
            return frame

        graph = self.workflow
        prefix_parts: list[str] = []
        for segment in scope_path.split("/"):
            node_id = self._scope_segment_node_id(segment)
            node = graph.nodes.get(node_id)
            if node is not None and node.type == "subgraph" and node.graph is not None:
                child_graph = graph.included_graphs.get(node.graph)
                if child_graph is None:
                    raise WorkflowExecutionError(f"Subgraph {node.graph!r} was not resolved for node {node_id!r}")
                parent_scope_path = "/".join(prefix_parts)
                child_scope_prefix = self._subgraph_scope_prefix(scope_path=parent_scope_path, node_id=node_id)
                child_inputs = folded.node_inputs_by_key.get(node_run_key(folded.run_id, node_id, parent_scope_path), {})
                frame = _ExecutionFrame(
                    graph=child_graph,
                    graph_id=child_graph.name,
                    scope_prefix=child_scope_prefix,
                    state_scope_path=child_scope_prefix,
                    inputs=child_inputs,
                    parent_node_id=node_id,
                    subgraph=node.graph,
                    depth=frame.depth + 1,
                )
                graph = child_graph
            prefix_parts.append(segment)
        return frame

    @staticmethod
    def _scope_segment_node_id(segment: str) -> str:
        bracket_index = segment.find("[")
        return segment if bracket_index < 0 else segment[:bracket_index]

    def _event_metadata(self, *, frame: _ExecutionFrame) -> JsonObject:
        metadata: JsonObject = {
            "graph_id": frame.graph_id,
            "depth": frame.depth,
        }
        if frame.parent_node_id is not None:
            metadata["parent_node_id"] = frame.parent_node_id
        if frame.subgraph is not None:
            metadata["subgraph"] = frame.subgraph
        return metadata

    async def _execute_node_boundary(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        folded: FoldedWorkflowState,
        frame: _ExecutionFrame,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject | None:
        scoped_key = node_run_key(run_id, node_id, scope_path)
        if scoped_key in folded.completed_node_runs:
            return folded.output_for_node(run_id, node_id, scope_path) or {}

        existing = await self.store.get_node_run(run_id, node_id, scope_path)
        attempt = (existing.attempt + 1) if existing is not None else 1
        context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame)
        node_input = self._render_object(node.input if node.input is not None else {}, context, label=f"{node_id}.input")
        idempotency_key = self._render_optional_string(node.idempotency_key, context)

        if self.tracer is not None:
            trace_ctx = TraceContext(
                self.tracer,
                f"Workflow node: {node_id}",
                SpanType.WORKFLOW_NODE,
                inputs={"input": node_input},
                attributes=self._node_trace_attributes(
                    run_id=run_id,
                    node_id=node_id,
                    node=node,
                    scope_path=scope_path,
                    attempt=attempt,
                    idempotency_key=idempotency_key,
                    frame=frame,
                ),
            )
            with trace_ctx:
                output = await self._execute_node_boundary_inner(
                    run_id=run_id,
                    node_id=node_id,
                    node=node,
                    scope_path=scope_path,
                    attempt=attempt,
                    node_input=node_input,
                    idempotency_key=idempotency_key,
                    folded=folded,
                    frame=frame,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
                trace_ctx.set_outputs(await self._node_trace_outputs(run_id, node_id, scope_path, output))
                return output

        return await self._execute_node_boundary_inner(
            run_id=run_id,
            node_id=node_id,
            node=node,
            scope_path=scope_path,
            attempt=attempt,
            node_input=node_input,
            idempotency_key=idempotency_key,
            folded=folded,
            frame=frame,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )

    async def _execute_node_boundary_inner(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        attempt: int,
        node_input: JsonObject,
        idempotency_key: str | None,
        folded: FoldedWorkflowState,
        frame: _ExecutionFrame,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject | None:
        metadata = self._event_metadata(frame=frame)
        await self.store.append_event(
            run_id=run_id,
            event_type="node_scheduled",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=metadata,
        )
        await self.store.upsert_node_run(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=WorkflowNodeStatus.SCHEDULED,
            attempt=attempt,
            node_input=node_input,
            idempotency_key=idempotency_key,
        )

        lease_expires_at = datetime.now() + timedelta(seconds=frame.graph.durable.lease_timeout_seconds)
        await self.store.append_event(
            run_id=run_id,
            event_type="node_started",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=event_payload(input=node_input, idempotency_key=idempotency_key or "", **metadata),
        )
        await self.store.upsert_node_run(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=WorkflowNodeStatus.RUNNING,
            attempt=attempt,
            node_input=node_input,
            idempotency_key=idempotency_key,
            lease_expires_at=lease_expires_at,
        )
        await self.store.create_lease(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            worker_id=self.worker_id,
            ttl_seconds=frame.graph.durable.lease_timeout_seconds,
        )

        if node.type == "human":
            await self._create_human_checkpoint(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt,
                node=node,
                node_input=node_input,
                frame=frame,
            )
            return None

        try:
            result = await self._execute_node(
                run_id=run_id,
                node_id=node_id,
                node=node,
                scope_path=scope_path,
                folded=folded,
                node_input=node_input,
                frame=frame,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )
            if node.output_schema is not None:
                validate_json_schema_output(result.output, node.output_schema)
            await self._complete_node(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt,
                output=result.output,
                raw_output=result.raw_output or result.output,
                state_patch=result.state_patch,
                state_scope_path=frame.state_scope_path,
                frame=frame,
            )
            return result.output
        except _WorkflowBoundaryPausedError as exc:
            paused_status = WorkflowNodeStatus.UNCERTAIN if exc.status == WorkflowRunStatus.UNCERTAIN else WorkflowNodeStatus.WAITING
            await self.store.upsert_node_run(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                status=paused_status,
                attempt=attempt,
                node_input=node_input,
            )
            await self.store.close_leases(run_id=run_id, node_id=node_id, scope_path=scope_path, status=paused_status.value)
            return None
        except Exception as exc:
            await self._fail_or_retry_node(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt,
                node=node,
                frame=frame,
                error=str(exc),
            )
            return None

    def _node_trace_attributes(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        attempt: int,
        idempotency_key: str | None,
        frame: _ExecutionFrame,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "workflow.name": self.workflow.name,
            "workflow.run_id": run_id,
            "workflow.graph_id": frame.graph_id,
            "workflow.node_id": node_id,
            "workflow.node_type": node.type,
            "workflow.scope_path": scope_path,
            "workflow.depth": frame.depth,
            "workflow.attempt": attempt,
            "workflow.side_effect": node.side_effect,
        }
        if frame.parent_node_id is not None:
            attributes["workflow.parent_node_id"] = frame.parent_node_id
        if frame.subgraph is not None:
            attributes["workflow.subgraph"] = frame.subgraph
        if node.agent is not None:
            attributes["workflow.agent"] = node.agent
        if node.tool is not None:
            attributes["workflow.tool"] = node.tool
        if node.server is not None:
            attributes["workflow.server"] = node.server
        if idempotency_key is not None:
            attributes["workflow.idempotency_key"] = idempotency_key
        return attributes

    async def _node_trace_outputs(
        self,
        run_id: str,
        node_id: str,
        scope_path: str,
        output: JsonObject | None,
    ) -> dict[str, object]:
        node_run = await self.store.get_node_run(run_id, node_id, scope_path)
        trace_output: dict[str, object] = {"status": node_run.status if node_run is not None else "unknown"}
        if output is not None:
            trace_output["output"] = output
        elif node_run is not None and node_run.output is not None:
            trace_output["output"] = node_run.output
        if node_run is not None and node_run.status == WorkflowNodeStatus.WAITING.value:
            folded = await self.store.fold(run_id)
            if folded.waiting_checkpoint_id is not None:
                trace_output["checkpoint_id"] = folded.waiting_checkpoint_id
        return trace_output

    async def _execute_node(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        folded: FoldedWorkflowState,
        node_input: JsonObject,
        frame: _ExecutionFrame,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> _NodeExecutionResult:
        match node.type:
            case "start":
                context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame)
                output_value = render_template(node.output if node.output is not None else frame.inputs, context)
                return _NodeExecutionResult(output=self._ensure_object(output_value, label=f"{node_id}.output"))
            case "transform":
                output_value = render_template(
                    node.output if node.output is not None else node_input,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame),
                )
                return _NodeExecutionResult(output=self._ensure_object(output_value, label=f"{node_id}.output"))
            case "set_state":
                patch_value = render_template(
                    node.update if node.update is not None else node_input,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame),
                )
                patch = self._ensure_object(patch_value, label=f"{node_id}.state_patch")
                return _NodeExecutionResult(output=patch, state_patch=patch)
            case "if_else":
                return _NodeExecutionResult(
                    output={
                        "next": self._choose_branch(
                            node,
                            run_id=run_id,
                            node_id=node_id,
                            scope_path=scope_path,
                            folded=folded,
                            frame=frame,
                        )
                    }
                )
            case "while":
                return await self._execute_while_node(
                    run_id=run_id,
                    node_id=node_id,
                    node=node,
                    scope_path=scope_path,
                    folded=folded,
                    frame=frame,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
            case "subgraph":
                return await self._execute_subgraph_node(
                    run_id=run_id,
                    node_id=node_id,
                    node=node,
                    scope_path=scope_path,
                    folded=folded,
                    node_input=node_input,
                    frame=frame,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
            case "tool":
                tool = self._resolve_tool(str(node.tool), frame.graph)
                raw = await tool.execute_async(**node_input)
                output = json_object(json_value(raw), label=f"{node_id}.tool_output")
                state_patch = self._render_update(node, run_id, node_id, scope_path, folded, output, frame)
                return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)
            case "mcp":
                tool = self._resolve_mcp_tool(str(node.server), str(node.tool), frame.graph)
                raw = await tool.execute_async(**node_input)
                output = json_object(json_value(raw), label=f"{node_id}.mcp_output")
                state_patch = self._render_update(node, run_id, node_id, scope_path, folded, output, frame)
                return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)
            case "agent":
                output = await self._run_agent_node(
                    run_id=run_id,
                    node_id=node_id,
                    node=node,
                    scope_path=scope_path,
                    input_data=node_input,
                    frame=frame,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
                state_patch = self._render_update(node, run_id, node_id, scope_path, folded, output, frame)
                return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)
            case "end":
                output_value = render_template(
                    node.output if node.output is not None else node_input,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame),
                )
                return _NodeExecutionResult(output=self._ensure_object(output_value, label=f"{node_id}.output"))
            case "note" | "human":
                return _NodeExecutionResult(output={})
        raise WorkflowExecutionError(f"Unsupported node type: {node.type}")

    async def _execute_subgraph_node(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        folded: FoldedWorkflowState,
        node_input: JsonObject,
        frame: _ExecutionFrame,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> _NodeExecutionResult:
        if node.graph is None:
            raise WorkflowExecutionError(f"subgraph node {node_id!r} requires graph")
        graph = frame.graph.included_graphs.get(node.graph)
        if graph is None:
            raise WorkflowExecutionError(f"Subgraph {node.graph!r} was not resolved for node {node_id!r}")

        subgraph_scope = self._subgraph_scope_prefix(scope_path=scope_path, node_id=node_id)
        child_frame = _ExecutionFrame(
            graph=graph,
            graph_id=graph.name,
            scope_prefix=subgraph_scope,
            state_scope_path=subgraph_scope,
            inputs=node_input,
            parent_node_id=node_id,
            subgraph=node.graph,
            depth=frame.depth + 1,
        )

        current_folded = folded
        if node.state_in is not None and subgraph_scope not in current_folded.scoped_state:
            state_value = render_template(
                node.state_in,
                self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded, frame=frame),
            )
            initial_state_patch = self._ensure_object(state_value, label=f"{node_id}.state_in")
            if initial_state_patch:
                await self.store.append_event(
                    run_id=run_id,
                    event_type="state_patched",
                    node_id=node_id,
                    scope_path=scope_path,
                    payload=event_payload(
                        patch=initial_state_patch,
                        state_scope_path=child_frame.state_scope_path,
                        **self._event_metadata(frame=child_frame),
                    ),
                )
                current_folded = await self.store.fold(run_id)

        await self.store.append_event(
            run_id=run_id,
            event_type="subgraph_started",
            node_id=node_id,
            scope_path=scope_path,
            payload=event_payload(graph_ref=node.graph, scope_prefix=subgraph_scope, **self._event_metadata(frame=child_frame)),
        )

        try:
            if self.tracer is None:
                output = await self._execute_frame_until_boundary(
                    run_id=run_id,
                    frame=child_frame,
                    folded=current_folded,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
            else:
                trace_ctx = TraceContext(
                    self.tracer,
                    f"Workflow subgraph: {graph.name}",
                    SpanType.WORKFLOW_SUBGRAPH,
                    inputs={"input": node_input},
                    attributes={
                        "workflow.name": self.workflow.name,
                        "workflow.run_id": run_id,
                        "workflow.graph_id": graph.name,
                        "workflow.parent_node_id": node_id,
                        "workflow.subgraph": node.graph,
                        "workflow.scope_path": subgraph_scope,
                        "workflow.depth": child_frame.depth,
                    },
                )
                with trace_ctx:
                    output = await self._execute_frame_until_boundary(
                        run_id=run_id,
                        frame=child_frame,
                        folded=current_folded,
                        session_manager=session_manager,
                        user_id=user_id,
                        session_id=session_id,
                    )
                    trace_ctx.set_outputs({"output": output})
        except _WorkflowBoundaryPausedError as exc:
            await self.store.append_event(
                run_id=run_id,
                event_type="subgraph_waiting" if exc.status == WorkflowRunStatus.WAITING else "subgraph_uncertain",
                node_id=node_id,
                scope_path=scope_path,
                payload=event_payload(graph_ref=node.graph, scope_prefix=subgraph_scope, **self._event_metadata(frame=child_frame)),
            )
            raise
        except Exception as exc:
            await self.store.append_event(
                run_id=run_id,
                event_type="subgraph_failed",
                node_id=node_id,
                scope_path=scope_path,
                payload=event_payload(
                    graph_ref=node.graph,
                    scope_prefix=subgraph_scope,
                    error=str(exc),
                    **self._event_metadata(frame=child_frame),
                ),
            )
            raise

        final_folded = await self.store.fold(run_id)
        state_patch: JsonObject | None = None
        if node.state_out is not None:
            state_out_value = render_template(
                node.state_out,
                self._context(
                    run_id=run_id,
                    node_id=node_id,
                    scope_path=scope_path,
                    folded=final_folded,
                    frame=child_frame,
                    output_override=output,
                ),
            )
            state_patch = self._ensure_object(state_out_value, label=f"{node_id}.state_out")

        await self.store.append_event(
            run_id=run_id,
            event_type="subgraph_completed",
            node_id=node_id,
            scope_path=scope_path,
            payload=event_payload(
                graph_ref=node.graph,
                scope_prefix=subgraph_scope,
                output=output,
                **self._event_metadata(frame=child_frame),
            ),
        )
        return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)

    async def _execute_frame_until_boundary(
        self,
        *,
        run_id: str,
        frame: _ExecutionFrame,
        folded: FoldedWorkflowState,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject:
        current_folded = folded
        while True:
            if current_folded.status == WorkflowRunStatus.WAITING:
                raise _WorkflowBoundaryPausedError(WorkflowRunStatus.WAITING)
            if current_folded.status == WorkflowRunStatus.UNCERTAIN:
                raise _WorkflowBoundaryPausedError(WorkflowRunStatus.UNCERTAIN)
            if current_folded.status == WorkflowRunStatus.FAILED:
                raise WorkflowExecutionError(f"Workflow graph {frame.graph_id!r} failed")

            next_node_id = self._next_node_id_for_frame(frame, current_folded)
            if next_node_id is None:
                return self._frame_output(run_id=run_id, frame=frame, folded=current_folded)

            node = frame.graph.nodes[next_node_id]
            node_scope_path = self._node_scope_path(frame, next_node_id)
            if node.type == "note":
                await self._complete_node(
                    run_id=run_id,
                    node_id=next_node_id,
                    scope_path=node_scope_path,
                    attempt=1,
                    output={},
                    raw_output={},
                    state_patch=None,
                    state_scope_path=frame.state_scope_path,
                    frame=frame,
                )
            else:
                await self._execute_node_boundary(
                    run_id=run_id,
                    node_id=next_node_id,
                    node=node,
                    scope_path=node_scope_path,
                    folded=current_folded,
                    frame=frame,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
            current_folded = await self.store.fold(run_id)

    def _frame_output(self, *, run_id: str, frame: _ExecutionFrame, folded: FoldedWorkflowState) -> JsonObject:
        completed_node_id = folded.last_completed_node_id_for_scope(frame.scope_prefix, set(frame.graph.nodes))
        context_node_id = completed_node_id or frame.graph.start_node_id
        context_scope_path = self._node_scope_path(frame, context_node_id)
        if frame.graph.output is not None:
            output_value = render_template(
                frame.graph.output,
                self._context(
                    run_id=run_id,
                    node_id=context_node_id,
                    scope_path=context_scope_path,
                    folded=folded,
                    frame=frame,
                ),
            )
            output = self._ensure_object(output_value, label=f"{frame.graph_id}.output")
            if frame.graph.output_schema is not None:
                validate_json_schema_output(output, frame.graph.output_schema)
            return output

        end_node_ids = [node_id for node_id, node in frame.graph.nodes.items() if node.type == "end"]
        if len(end_node_ids) == 1:
            end_output = folded.output_for_node(run_id, end_node_ids[0], self._node_scope_path(frame, end_node_ids[0]))
            if end_output is not None:
                if frame.graph.output_schema is not None:
                    validate_json_schema_output(end_output, frame.graph.output_schema)
                return end_output

        if completed_node_id is None:
            return {}
        output = folded.output_for_node(run_id, completed_node_id, self._node_scope_path(frame, completed_node_id)) or {}
        if frame.graph.output_schema is not None:
            validate_json_schema_output(output, frame.graph.output_schema)
        return output

    @staticmethod
    def _subgraph_scope_prefix(*, scope_path: str, node_id: str) -> str:
        return f"{scope_path}/{node_id}" if scope_path else node_id

    async def _execute_while_node(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        folded: FoldedWorkflowState,
        frame: _ExecutionFrame,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> _NodeExecutionResult:
        if node.body is None or node.condition is None or node.max_iterations is None:
            raise WorkflowExecutionError(f"Invalid while node: {node_id}")

        current_folded = folded
        init_patch: JsonObject = {}
        if node.init is not None:
            init_value = render_template(
                node.init,
                self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded, frame=frame),
            )
            rendered_init = self._ensure_object(init_value, label=f"{node_id}.init")
            current_state = current_folded.state_for_scope(frame.state_scope_path)
            init_patch = {key: value for key, value in rendered_init.items() if key not in current_state}
            if init_patch:
                await self.store.append_event(
                    run_id=run_id,
                    event_type="state_patched",
                    node_id=node_id,
                    scope_path=scope_path,
                    payload=event_payload(patch=init_patch, state_scope_path=frame.state_scope_path, **self._event_metadata(frame=frame)),
                )
                current_folded = await self.store.fold(run_id)

        iterations = 0
        while evaluate_condition(
            node.condition,
            self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded, frame=frame),
        ):
            if iterations >= node.max_iterations:
                raise WorkflowExecutionError(f"while node {node_id!r} exceeded max_iterations={node.max_iterations}")
            body_node_id = node.body
            body_node = frame.graph.nodes[body_node_id]
            scope_key = str(iterations)
            if node.scope_key:
                rendered_scope = render_template(
                    node.scope_key,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded, frame=frame),
                )
                scope_key = str(rendered_scope)
            body_scope_path = f"{frame.scope_prefix}/{node_id}[{scope_key}]" if frame.scope_prefix else f"{node_id}[{scope_key}]"
            body_output = await self._execute_node_boundary(
                run_id=run_id,
                node_id=body_node_id,
                node=body_node,
                scope_path=body_scope_path,
                folded=current_folded,
                frame=frame,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )
            current_folded = await self.store.fold(run_id)
            if current_folded.status == WorkflowRunStatus.WAITING:
                raise _WorkflowBoundaryPausedError(WorkflowRunStatus.WAITING)
            if current_folded.status == WorkflowRunStatus.UNCERTAIN:
                raise _WorkflowBoundaryPausedError(WorkflowRunStatus.UNCERTAIN)
            if current_folded.status == WorkflowRunStatus.FAILED:
                raise WorkflowExecutionError(f"while body {body_node_id!r} failed")
            if body_output is None and node_run_key(run_id, body_node_id, body_scope_path) not in current_folded.completed_node_runs:
                continue
            iterations += 1

        output: JsonObject = {"iterations": iterations, "state": current_folded.state_for_scope(frame.state_scope_path)}
        return _NodeExecutionResult(output=output)

    async def _run_agent_node(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        input_data: JsonObject,
        frame: _ExecutionFrame,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject:
        agent_config = self._resolve_agent_config(str(node.agent), frame.graph)
        if self.agent_runner is not None:
            return await self.agent_runner(
                agent_name=str(node.agent),
                agent_config=agent_config,
                input_data=input_data,
                output_schema=node.output_schema,
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
            )
        if agent_config is None:
            raise WorkflowExecutionError(f"Agent node {node_id!r} could not resolve agent {node.agent!r}")

        traced_agent_config = self._agent_config_with_workflow_tracer(agent_config)
        message = (
            "Execute this workflow agent node using the JSON input below.\n"
            "Return the requested final structured output only.\n\n"
            f"```json\n{json.dumps(input_data, ensure_ascii=False, indent=2)}\n```"
        )
        if node.output_schema is None:
            from nexau.archs.main_sub.agent import Agent

            agent = await Agent.create(config=traced_agent_config, session_manager=session_manager, user_id=user_id, session_id=session_id)
            response = await agent.run_async(message=message)
            response_text = response[0] if isinstance(response, tuple) else response
            return {"result": response_text}
        return await run_agent_structured_async(
            config=agent_config,
            message=message,
            output_schema=node.output_schema,
            output_mode=node.output_mode or agent_config.output_mode,
            output_retries=node.output_retries if node.output_retries is not None else agent_config.output_retries,
            output_name=node.output_name or agent_config.output_name,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            tracer=self.tracer,
        )

    def _agent_config_with_workflow_tracer(self, agent_config: AgentConfig) -> AgentConfig:
        if self.tracer is None:
            return agent_config
        traced_config = agent_config.model_copy()
        traced_config.tracers = [self.tracer]
        traced_config.resolved_tracer = self.tracer
        return traced_config

    async def _create_human_checkpoint(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        attempt: int,
        node: WorkflowNode,
        node_input: JsonObject,
        frame: _ExecutionFrame,
    ) -> None:
        checkpoint_id = f"wf_ckpt_{uuid.uuid4().hex}"
        folded = await self.store.fold(run_id)
        rendered_prompt = self._render_optional_string(
            node.prompt or "",
            self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame),
        )
        prompt = rendered_prompt or ""
        await self.store.create_checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            prompt=prompt,
            checkpoint_input=node_input,
            output_schema=node.output_schema,
        )
        await self.store.append_event(
            run_id=run_id,
            event_type="checkpoint_created",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=event_payload(checkpoint_id=checkpoint_id, prompt=prompt, input=node_input, **self._event_metadata(frame=frame)),
        )
        await self.store.upsert_node_run(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=WorkflowNodeStatus.WAITING,
            attempt=attempt,
            node_input=node_input,
        )
        await self.store.close_leases(run_id=run_id, node_id=node_id, scope_path=scope_path, status="waiting")
        folded = await self.store.fold(run_id)
        run = await self.store.get_run(run_id)
        if run is not None:
            await self.store.update_run(run, status=WorkflowRunStatus.WAITING, state=folded.state, waiting_checkpoint_id=checkpoint_id)

    async def _complete_node(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        attempt: int,
        output: JsonObject,
        raw_output: JsonObject,
        state_patch: JsonObject | None,
        state_scope_path: str,
        frame: _ExecutionFrame,
    ) -> None:
        payload = event_payload(output=output, raw_output=raw_output, **self._event_metadata(frame=frame))
        if state_patch is not None:
            payload["state_patch"] = state_patch
            payload["state_scope_path"] = state_scope_path
        await self.store.append_event(
            run_id=run_id,
            event_type="node_completed",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=json_object(payload, label="node_completed payload"),
        )
        if state_patch is not None:
            await self.store.append_event(
                run_id=run_id,
                event_type="state_patched",
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt,
                payload=event_payload(patch=state_patch, state_scope_path=state_scope_path, **self._event_metadata(frame=frame)),
            )
        await self.store.upsert_node_run(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=WorkflowNodeStatus.COMPLETED,
            attempt=attempt,
            output=output,
            raw_output=raw_output,
        )
        await self.store.close_leases(run_id=run_id, node_id=node_id, scope_path=scope_path, status="completed")
        folded = await self.store.fold(run_id)
        await self.store.save_state_snapshot(run_id=run_id, state=folded.state)
        run = await self.store.get_run(run_id)
        if run is not None and run.status != WorkflowRunStatus.WAITING.value:
            await self.store.update_run(run, status=WorkflowRunStatus.RUNNING, state=folded.state)

    async def _fail_or_retry_node(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        attempt: int,
        node: WorkflowNode,
        frame: _ExecutionFrame,
        error: str,
    ) -> None:
        await self.store.append_event(
            run_id=run_id,
            event_type="node_failed",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=event_payload(error=error, **self._event_metadata(frame=frame)),
        )
        await self.store.close_leases(run_id=run_id, node_id=node_id, scope_path=scope_path, status="failed")
        retry_policy = self._retry_policy(node, frame)
        if attempt < retry_policy.max_attempts:
            await self.store.append_event(
                run_id=run_id,
                event_type="node_retry_scheduled",
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt + 1,
                payload=event_payload(error=error, **self._event_metadata(frame=frame)),
            )
            await self.store.upsert_node_run(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                status=WorkflowNodeStatus.SCHEDULED,
                attempt=attempt,
            )
            return
        await self.store.upsert_node_run(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=WorkflowNodeStatus.FAILED,
            attempt=attempt,
        )
        await self.store.append_event(run_id=run_id, event_type="workflow_run_failed", node_id=node_id, scope_path=scope_path)
        run = await self.store.get_run(run_id)
        folded = await self.store.fold(run_id)
        if run is not None:
            await self.store.update_run(run, status=WorkflowRunStatus.FAILED, state=folded.state)

    async def _complete_workflow(self, *, run_id: str, folded: FoldedWorkflowState) -> WorkflowRunResult:
        last_output = folded.node_outputs.get(folded.last_completed_node_id or "", {})
        output = last_output or {"state": folded.state}
        await self.store.append_event(run_id=run_id, event_type="workflow_run_completed", payload=event_payload(output=output))
        run = await self.store.get_run(run_id)
        if run is not None:
            await self.store.update_run(run, status=WorkflowRunStatus.COMPLETED, output=output, state=folded.state)
        return WorkflowRunResult(run_id=run_id, status=WorkflowRunStatus.COMPLETED, output=output, state=folded.state)

    async def _result_from_folded(self, folded: FoldedWorkflowState) -> WorkflowRunResult:
        run = await self.store.get_run(folded.run_id)
        output = folded.output
        checkpoint_id = folded.waiting_checkpoint_id
        if run is not None:
            output = json_object(run.output, label="run.output") if run.output is not None else output
            checkpoint_id = run.waiting_checkpoint_id or checkpoint_id
        return WorkflowRunResult(
            run_id=folded.run_id,
            status=folded.status,
            output=output,
            state=folded.state,
            checkpoint_id=checkpoint_id,
        )

    async def _recover_expired_attempts(self, run_id: str) -> None:
        await self._load_snapshot_for_run(run_id)
        folded = await self.store.fold(run_id)
        expired_nodes = await self.store.list_expired_running_nodes(run_id)
        for node_run in expired_nodes:
            frame = self._frame_for_scope(scope_path=node_run.scope_path, folded=folded)
            node = frame.graph.nodes[node_run.node_id]
            if node.side_effect == "external_write":
                await self.store.append_event(
                    run_id=run_id,
                    event_type="node_uncertain",
                    node_id=node_run.node_id,
                    scope_path=node_run.scope_path,
                    attempt=node_run.attempt,
                    payload=event_payload(reason="lease_expired", **self._event_metadata(frame=frame)),
                )
                await self.store.upsert_node_run(
                    run_id=run_id,
                    node_id=node_run.node_id,
                    scope_path=node_run.scope_path,
                    status=WorkflowNodeStatus.UNCERTAIN,
                    attempt=node_run.attempt,
                )
                await self.store.close_leases(run_id=run_id, node_id=node_run.node_id, scope_path=node_run.scope_path, status="expired")
                run = await self.store.get_run(run_id)
                folded = await self.store.fold(run_id)
                if run is not None:
                    await self.store.update_run(run, status=WorkflowRunStatus.UNCERTAIN, state=folded.state)
                continue

            await self.store.append_event(
                run_id=run_id,
                event_type="node_retry_scheduled",
                node_id=node_run.node_id,
                scope_path=node_run.scope_path,
                attempt=node_run.attempt + 1,
                payload=event_payload(reason="lease_expired", **self._event_metadata(frame=frame)),
            )
            await self.store.upsert_node_run(
                run_id=run_id,
                node_id=node_run.node_id,
                scope_path=node_run.scope_path,
                status=WorkflowNodeStatus.SCHEDULED,
                attempt=node_run.attempt,
            )
            await self.store.close_leases(run_id=run_id, node_id=node_run.node_id, scope_path=node_run.scope_path, status="expired")

    def _choose_branch(
        self,
        node: WorkflowNode,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        folded: FoldedWorkflowState,
        frame: _ExecutionFrame,
    ) -> str:
        context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame)
        for branch in node.branches:
            if evaluate_condition(branch.if_, context):
                return branch.next
        if node.else_ is None:
            raise WorkflowExecutionError("No if_else branch matched and no else target was configured")
        return node.else_

    def _render_update(
        self,
        node: WorkflowNode,
        run_id: str,
        node_id: str,
        scope_path: str,
        folded: FoldedWorkflowState,
        output: JsonObject,
        frame: _ExecutionFrame,
    ) -> JsonObject | None:
        if node.update is None:
            return None
        context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, frame=frame, output_override=output)
        rendered = render_template(node.update, context)
        return self._ensure_object(rendered, label=f"{node_id}.update")

    def _context(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        folded: FoldedWorkflowState,
        frame: _ExecutionFrame,
        state_override: JsonObject | None = None,
        output_override: JsonObject | None = None,
    ) -> EvaluationContext:
        nodes = folded.node_context(frame.scope_prefix)
        if output_override is not None:
            nodes[node_id] = {"status": WorkflowNodeStatus.COMPLETED.value, "output": output_override}
        state = state_override if state_override is not None else folded.state_for_scope(frame.state_scope_path)
        return {
            "inputs": frame.inputs,
            "vars": frame.graph.vars,
            "state": state,
            "nodes": nodes,
            "run": {"id": run_id},
            "node": {"id": node_id, "scope_path": scope_path, "graph_id": frame.graph_id, "depth": frame.depth},
        }

    def _render_object(self, value: JsonValue, context: EvaluationContext, *, label: str) -> JsonObject:
        return self._ensure_object(render_template(value, context), label=label)

    @staticmethod
    def _ensure_object(value: JsonValue, *, label: str) -> JsonObject:
        if isinstance(value, dict):
            return value
        return {"result": value}

    @staticmethod
    def _render_optional_string(template: str | None, context: EvaluationContext) -> str | None:
        if template is None:
            return None
        rendered = render_template(template, context)
        return str(rendered)

    @staticmethod
    def _retry_policy(node: WorkflowNode, frame: _ExecutionFrame) -> RetryPolicy:
        return node.retry_policy or frame.graph.durable.default_retry_policy

    def _resolve_agent_config(self, agent_name: str, graph: WorkflowConfig) -> AgentConfig | None:
        if agent_name in self.agents:
            return self.agents[agent_name]
        include_path = graph.includes.agents.get(agent_name)
        if include_path is not None:
            return AgentConfig.from_yaml(graph.base_path / include_path)
        return None

    def _resolve_tool(self, tool_name: str, graph: WorkflowConfig) -> Tool:
        if tool_name in self.tools:
            return self.tools[tool_name]
        if self.tool_registry is not None:
            tool = self.tool_registry.get_tool(tool_name)
            if tool is not None:
                return tool
        include_path = graph.includes.tools.get(tool_name)
        if include_path is not None:
            return Tool.from_yaml(str(graph.base_path / include_path))
        raise WorkflowExecutionError(f"Tool not found: {tool_name}")

    def _resolve_mcp_tool(self, server: str, tool_name: str, graph: WorkflowConfig) -> Tool:
        for key in (f"{server}.{tool_name}", f"{server}:{tool_name}", tool_name):
            if key in self.mcp_tools:
                return self.mcp_tools[key]
        return self._resolve_tool(tool_name, graph)
