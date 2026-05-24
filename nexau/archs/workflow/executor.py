"""Durable workflow executor.

RFC-0027: Durable WorkflowExecutor

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
            snapshot = json_object(json_value(self.workflow.model_dump(mode="json", by_alias=True, exclude={"source_path"})))
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
        await self._complete_node(
            run_id=run_id,
            node_id=checkpoint.node_id,
            scope_path=checkpoint.scope_path,
            attempt=1,
            output=output,
            raw_output=output,
            state_patch=None,
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

        folded = await self.store.fold(run_id)
        key = node_run_key(run_id, node_id, scope_path)
        if key not in folded.uncertain_node_runs:
            raise WorkflowExecutionError(f"Node is not uncertain: {node_id} {scope_path}")

        match decision:
            case "completed":
                await self._complete_node(
                    run_id=run_id,
                    node_id=node_id,
                    scope_path=scope_path,
                    attempt=1,
                    output=output or {},
                    raw_output=output or {},
                    state_patch=None,
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

            next_node_id = self._next_node_id(folded)
            if next_node_id is None:
                return await self._complete_workflow(run_id=run_id, folded=folded)

            node = self.workflow.nodes[next_node_id]
            if node.type == "note":
                await self._complete_node(
                    run_id=run_id,
                    node_id=next_node_id,
                    scope_path="",
                    attempt=1,
                    output={},
                    raw_output={},
                    state_patch=None,
                )
                continue

            await self._execute_node_boundary(
                run_id=run_id,
                node_id=next_node_id,
                node=node,
                scope_path="",
                folded=folded,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )

    def _next_node_id(self, folded: FoldedWorkflowState) -> str | None:
        if folded.last_completed_node_id is None:
            return self.workflow.start_node_id

        completed_node = self.workflow.nodes[folded.last_completed_node_id]
        if completed_node.type == "if_else":
            branch_output = folded.node_outputs.get(folded.last_completed_node_id, {})
            next_value = branch_output.get("next")
            return str(next_value) if isinstance(next_value, str) else None
        if completed_node.type == "end":
            return None

        edge = self.workflow.edges.get(folded.last_completed_node_id)
        if edge is None:
            return None
        if isinstance(edge, str):
            return edge
        return edge[0] if edge else None

    async def _execute_node_boundary(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        folded: FoldedWorkflowState,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject | None:
        scoped_key = node_run_key(run_id, node_id, scope_path)
        if scoped_key in folded.completed_node_runs:
            return folded.node_outputs.get(node_id, {})

        existing = await self.store.get_node_run(run_id, node_id, scope_path)
        attempt = (existing.attempt + 1) if existing is not None else 1
        context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded)
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
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject | None:
        await self.store.append_event(run_id=run_id, event_type="node_scheduled", node_id=node_id, scope_path=scope_path, attempt=attempt)
        await self.store.upsert_node_run(
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=WorkflowNodeStatus.SCHEDULED,
            attempt=attempt,
            node_input=node_input,
            idempotency_key=idempotency_key,
        )

        lease_expires_at = datetime.now() + timedelta(seconds=self.workflow.durable.lease_timeout_seconds)
        await self.store.append_event(
            run_id=run_id,
            event_type="node_started",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=event_payload(input=node_input, idempotency_key=idempotency_key or ""),
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
            ttl_seconds=self.workflow.durable.lease_timeout_seconds,
        )

        if node.type == "human":
            await self._create_human_checkpoint(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt,
                node=node,
                node_input=node_input,
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
            )
            return result.output
        except Exception as exc:
            await self._fail_or_retry_node(
                run_id=run_id,
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt,
                node=node,
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
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "workflow.name": self.workflow.name,
            "workflow.run_id": run_id,
            "workflow.node_id": node_id,
            "workflow.node_type": node.type,
            "workflow.scope_path": scope_path,
            "workflow.attempt": attempt,
            "workflow.side_effect": node.side_effect,
        }
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
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> _NodeExecutionResult:
        match node.type:
            case "start":
                context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded)
                output_value = render_template(node.output if node.output is not None else folded.inputs, context)
                return _NodeExecutionResult(output=self._ensure_object(output_value, label=f"{node_id}.output"))
            case "transform":
                output_value = render_template(
                    node.output if node.output is not None else node_input,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded),
                )
                return _NodeExecutionResult(output=self._ensure_object(output_value, label=f"{node_id}.output"))
            case "set_state":
                patch_value = render_template(
                    node.update if node.update is not None else node_input,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded),
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
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
            case "tool":
                tool = self._resolve_tool(str(node.tool))
                raw = await tool.execute_async(**node_input)
                output = json_object(json_value(raw), label=f"{node_id}.tool_output")
                state_patch = self._render_update(node, run_id, node_id, scope_path, folded, output)
                return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)
            case "mcp":
                tool = self._resolve_mcp_tool(str(node.server), str(node.tool))
                raw = await tool.execute_async(**node_input)
                output = json_object(json_value(raw), label=f"{node_id}.mcp_output")
                state_patch = self._render_update(node, run_id, node_id, scope_path, folded, output)
                return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)
            case "agent":
                output = await self._run_agent_node(
                    run_id=run_id,
                    node_id=node_id,
                    node=node,
                    scope_path=scope_path,
                    input_data=node_input,
                    session_manager=session_manager,
                    user_id=user_id,
                    session_id=session_id,
                )
                state_patch = self._render_update(node, run_id, node_id, scope_path, folded, output)
                return _NodeExecutionResult(output=output, raw_output=output, state_patch=state_patch)
            case "end":
                output_value = render_template(
                    node.output if node.output is not None else node_input,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded),
                )
                return _NodeExecutionResult(output=self._ensure_object(output_value, label=f"{node_id}.output"))
            case "note" | "human":
                return _NodeExecutionResult(output={})
        raise WorkflowExecutionError(f"Unsupported node type: {node.type}")

    async def _execute_while_node(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        folded: FoldedWorkflowState,
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
                self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded),
            )
            rendered_init = self._ensure_object(init_value, label=f"{node_id}.init")
            init_patch = {key: value for key, value in rendered_init.items() if key not in current_folded.state}
            if init_patch:
                await self.store.append_event(
                    run_id=run_id,
                    event_type="state_patched",
                    node_id=node_id,
                    payload=event_payload(patch=init_patch),
                )
                current_folded = await self.store.fold(run_id)

        iterations = 0
        while evaluate_condition(
            node.condition,
            self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded),
        ):
            if iterations >= node.max_iterations:
                raise WorkflowExecutionError(f"while node {node_id!r} exceeded max_iterations={node.max_iterations}")
            body_node_id = node.body
            body_node = self.workflow.nodes[body_node_id]
            scope_key = str(iterations)
            if node.scope_key:
                rendered_scope = render_template(
                    node.scope_key,
                    self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=current_folded),
                )
                scope_key = str(rendered_scope)
            body_scope_path = f"{node_id}[{scope_key}]"
            body_output = await self._execute_node_boundary(
                run_id=run_id,
                node_id=body_node_id,
                node=body_node,
                scope_path=body_scope_path,
                folded=current_folded,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
            )
            current_folded = await self.store.fold(run_id)
            if current_folded.status in {WorkflowRunStatus.WAITING, WorkflowRunStatus.FAILED, WorkflowRunStatus.UNCERTAIN}:
                raise WorkflowExecutionError(f"while body {body_node_id!r} did not complete")
            if body_output is None and node_run_key(run_id, body_node_id, body_scope_path) not in current_folded.completed_node_runs:
                continue
            iterations += 1

        output: JsonObject = {"iterations": iterations, "state": current_folded.state}
        return _NodeExecutionResult(output=output)

    async def _run_agent_node(
        self,
        *,
        run_id: str,
        node_id: str,
        node: WorkflowNode,
        scope_path: str,
        input_data: JsonObject,
        session_manager: SessionManager | None,
        user_id: str | None,
        session_id: str | None,
    ) -> JsonObject:
        agent_config = self._resolve_agent_config(str(node.agent))
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
    ) -> None:
        checkpoint_id = f"wf_ckpt_{uuid.uuid4().hex}"
        await self.store.create_checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            prompt=node.prompt or "",
            checkpoint_input=node_input,
            output_schema=node.output_schema,
        )
        await self.store.append_event(
            run_id=run_id,
            event_type="checkpoint_created",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=event_payload(checkpoint_id=checkpoint_id, prompt=node.prompt or "", input=node_input),
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
    ) -> None:
        payload = event_payload(output=output, raw_output=raw_output)
        if state_patch is not None:
            payload["state_patch"] = state_patch
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
                payload=event_payload(patch=state_patch),
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
        error: str,
    ) -> None:
        await self.store.append_event(
            run_id=run_id,
            event_type="node_failed",
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=event_payload(error=error),
        )
        await self.store.close_leases(run_id=run_id, node_id=node_id, scope_path=scope_path, status="failed")
        retry_policy = self._retry_policy(node)
        if attempt < retry_policy.max_attempts:
            await self.store.append_event(
                run_id=run_id,
                event_type="node_retry_scheduled",
                node_id=node_id,
                scope_path=scope_path,
                attempt=attempt + 1,
                payload=event_payload(error=error),
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
        expired_nodes = await self.store.list_expired_running_nodes(run_id)
        for node_run in expired_nodes:
            node = self.workflow.nodes[node_run.node_id]
            if node.side_effect == "external_write":
                await self.store.append_event(
                    run_id=run_id,
                    event_type="node_uncertain",
                    node_id=node_run.node_id,
                    scope_path=node_run.scope_path,
                    attempt=node_run.attempt,
                    payload=event_payload(reason="lease_expired"),
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
                payload=event_payload(reason="lease_expired"),
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
    ) -> str:
        context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded)
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
    ) -> JsonObject | None:
        if node.update is None:
            return None
        context = self._context(run_id=run_id, node_id=node_id, scope_path=scope_path, folded=folded, output_override=output)
        rendered = render_template(node.update, context)
        return self._ensure_object(rendered, label=f"{node_id}.update")

    def _context(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        folded: FoldedWorkflowState,
        state_override: JsonObject | None = None,
        output_override: JsonObject | None = None,
    ) -> EvaluationContext:
        nodes = folded.node_context()
        if output_override is not None:
            nodes[node_id] = {"status": WorkflowNodeStatus.COMPLETED.value, "output": output_override}
        state = state_override if state_override is not None else folded.state
        return {
            "inputs": folded.inputs,
            "vars": self.workflow.vars,
            "state": state,
            "nodes": nodes,
            "run": {"id": run_id},
            "node": {"id": node_id, "scope_path": scope_path},
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

    def _retry_policy(self, node: WorkflowNode) -> RetryPolicy:
        return node.retry_policy or self.workflow.durable.default_retry_policy

    def _resolve_agent_config(self, agent_name: str) -> AgentConfig | None:
        if agent_name in self.agents:
            return self.agents[agent_name]
        include_path = self.workflow.includes.agents.get(agent_name)
        if include_path is not None:
            return AgentConfig.from_yaml(self.workflow.base_path / include_path)
        return None

    def _resolve_tool(self, tool_name: str) -> Tool:
        if tool_name in self.tools:
            return self.tools[tool_name]
        if self.tool_registry is not None:
            tool = self.tool_registry.get_tool(tool_name)
            if tool is not None:
                return tool
        include_path = self.workflow.includes.tools.get(tool_name)
        if include_path is not None:
            return Tool.from_yaml(str(self.workflow.base_path / include_path))
        raise WorkflowExecutionError(f"Tool not found: {tool_name}")

    def _resolve_mcp_tool(self, server: str, tool_name: str) -> Tool:
        for key in (f"{server}.{tool_name}", f"{server}:{tool_name}", tool_name):
            if key in self.mcp_tools:
                return self.mcp_tools[key]
        return self._resolve_tool(tool_name)
