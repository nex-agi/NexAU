"""Durable workflow event store and fold logic.

RFC-0027: append-only event log is the canonical workflow state. Materialized
run, node, checkpoint, lease, and state rows are maintained for fast lookup,
but recovery folds events first and treats summaries as indexes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from nexau.archs.session.models.workflow import (
    WORKFLOW_MODELS,
    WorkflowCheckpointModel,
    WorkflowCheckpointStatus,
    WorkflowEventModel,
    WorkflowLeaseModel,
    WorkflowNodeRunModel,
    WorkflowNodeStatus,
    WorkflowRunModel,
    WorkflowRunStatus,
    WorkflowStateModel,
)
from nexau.archs.session.orm import AndFilter, ComparisonFilter, DatabaseEngine
from nexau.archs.workflow.types import JsonObject, JsonValue, json_object, merge_json_objects


def _empty_json_object() -> JsonObject:
    return {}


def _empty_node_outputs() -> dict[str, JsonObject]:
    return {}


def _empty_node_status() -> dict[str, str]:
    return {}


def _empty_string_set() -> set[str]:
    return set()


def node_run_key(run_id: str, node_id: str, scope_path: str = "") -> str:
    """Return the stable materialized key for one scoped node instance."""

    normalized_scope = scope_path or "$"
    return f"{run_id}:{node_id}:{normalized_scope}"


def event_payload(**items: JsonValue) -> JsonObject:
    """Build a DB JSON payload from JSON-compatible values."""

    return dict(items)


@dataclass
class FoldedWorkflowState:
    """State derived by folding the workflow event log."""

    run_id: str
    status: WorkflowRunStatus = WorkflowRunStatus.PENDING
    inputs: JsonObject = field(default_factory=_empty_json_object)
    state: JsonObject = field(default_factory=_empty_json_object)
    output: JsonObject | None = None
    node_outputs: dict[str, JsonObject] = field(default_factory=_empty_node_outputs)
    node_status: dict[str, str] = field(default_factory=_empty_node_status)
    completed_node_runs: set[str] = field(default_factory=_empty_string_set)
    failed_node_runs: set[str] = field(default_factory=_empty_string_set)
    uncertain_node_runs: set[str] = field(default_factory=_empty_string_set)
    waiting_checkpoint_id: str | None = None
    waiting_node_id: str | None = None
    waiting_scope_path: str = ""
    last_completed_node_id: str | None = None
    last_completed_scope_path: str = ""

    def node_context(self) -> JsonObject:
        """Return the ``nodes`` expression context."""

        result: JsonObject = {}
        for node_id, status in self.node_status.items():
            node_entry: JsonObject = {"status": status}
            output = self.node_outputs.get(node_id)
            if output is not None:
                node_entry["output"] = output
            result[node_id] = node_entry
        return result


class WorkflowStore:
    """Repository for workflow durable execution models."""

    def __init__(self, engine: DatabaseEngine):
        self._engine = engine
        self._initialized = False

    async def setup(self) -> None:
        """Initialize workflow storage models."""

        if self._initialized:
            return
        await self._engine.setup_models(WORKFLOW_MODELS)
        self._initialized = True

    async def create_run(
        self,
        *,
        run_id: str,
        workflow_name: str,
        inputs: JsonObject,
        definition_snapshot: JsonObject,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> WorkflowRunModel:
        """Create the materialized run summary."""

        await self.setup()
        run = WorkflowRunModel(
            run_id=run_id,
            workflow_name=workflow_name,
            status=WorkflowRunStatus.RUNNING.value,
            user_id=user_id,
            session_id=session_id,
            input=dict(inputs),
            definition_snapshot=dict(definition_snapshot),
        )
        return await self._engine.create(run)

    async def get_run(self, run_id: str) -> WorkflowRunModel | None:
        """Fetch a materialized run summary."""

        await self.setup()
        return await self._engine.find_first(WorkflowRunModel, filters=ComparisonFilter.eq("run_id", run_id))

    async def update_run(
        self,
        run: WorkflowRunModel,
        *,
        status: WorkflowRunStatus | None = None,
        output: JsonObject | None = None,
        state: JsonObject | None = None,
        waiting_checkpoint_id: str | None = None,
    ) -> WorkflowRunModel:
        """Update a materialized run summary."""

        if status is not None:
            run.status = status.value
        if output is not None:
            run.output = dict(output)
        if state is not None:
            run.state = dict(state)
        run.waiting_checkpoint_id = waiting_checkpoint_id
        run.updated_at = datetime.now()
        return await self._engine.update(run)

    async def append_event(
        self,
        *,
        run_id: str,
        event_type: str,
        payload: JsonObject | None = None,
        node_id: str | None = None,
        scope_path: str = "",
        attempt: int | None = None,
    ) -> WorkflowEventModel:
        """Append one canonical workflow event."""

        await self.setup()
        sequence = await self._engine.count(WorkflowEventModel, filters=ComparisonFilter.eq("run_id", run_id)) + 1
        event = WorkflowEventModel(
            event_id=f"wf_evt_{uuid.uuid4().hex}",
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            payload=dict(payload or {}),
        )
        return await self._engine.create(event)

    async def list_events(self, run_id: str) -> list[WorkflowEventModel]:
        """List events for a run in canonical order."""

        await self.setup()
        return await self._engine.find_many(
            WorkflowEventModel,
            filters=ComparisonFilter.eq("run_id", run_id),
            order_by="sequence",
        )

    async def fold(self, run_id: str) -> FoldedWorkflowState:
        """Fold the canonical event log into runtime state."""

        events = await self.list_events(run_id)
        folded = FoldedWorkflowState(run_id=run_id)
        checkpoints: dict[str, str] = {}
        for event in events:
            payload = json_object(event.payload, label="workflow event payload")
            match event.event_type:
                case "workflow_run_started":
                    folded.status = WorkflowRunStatus.RUNNING
                    raw_inputs = payload.get("inputs", {})
                    folded.inputs = json_object(raw_inputs, label="workflow_run_started.inputs")
                case "node_scheduled":
                    if event.node_id is not None:
                        folded.node_status[event.node_id] = WorkflowNodeStatus.SCHEDULED.value
                case "node_started":
                    if event.node_id is not None:
                        folded.node_status[event.node_id] = WorkflowNodeStatus.RUNNING.value
                case "node_completed":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.completed_node_runs.add(key)
                        folded.failed_node_runs.discard(key)
                        folded.uncertain_node_runs.discard(key)
                        folded.node_status[event.node_id] = WorkflowNodeStatus.COMPLETED.value
                        if folded.status in {WorkflowRunStatus.FAILED, WorkflowRunStatus.UNCERTAIN}:
                            folded.status = WorkflowRunStatus.RUNNING
                        raw_output = payload.get("output", {})
                        output = json_object(raw_output, label="node_completed.output")
                        folded.node_outputs[event.node_id] = output
                        raw_patch = payload.get("state_patch")
                        if raw_patch is not None:
                            folded.state = merge_json_objects(folded.state, json_object(raw_patch, label="node_completed.state_patch"))
                        folded.last_completed_node_id = event.node_id
                        folded.last_completed_scope_path = event.scope_path
                case "node_failed":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.failed_node_runs.add(key)
                        folded.node_status[event.node_id] = WorkflowNodeStatus.FAILED.value
                        folded.status = WorkflowRunStatus.FAILED
                case "node_retry_scheduled":
                    if event.node_id is not None:
                        folded.node_status[event.node_id] = WorkflowNodeStatus.SCHEDULED.value
                        folded.status = WorkflowRunStatus.RUNNING
                case "node_uncertain":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.uncertain_node_runs.add(key)
                        folded.node_status[event.node_id] = WorkflowNodeStatus.UNCERTAIN.value
                        folded.status = WorkflowRunStatus.UNCERTAIN
                case "checkpoint_created":
                    checkpoint_id = str(payload.get("checkpoint_id", ""))
                    checkpoints[checkpoint_id] = WorkflowCheckpointStatus.OPEN.value
                    folded.status = WorkflowRunStatus.WAITING
                    folded.waiting_checkpoint_id = checkpoint_id
                    folded.waiting_node_id = event.node_id
                    folded.waiting_scope_path = event.scope_path
                    if event.node_id is not None:
                        folded.node_status[event.node_id] = WorkflowNodeStatus.WAITING.value
                case "checkpoint_resumed":
                    checkpoint_id = str(payload.get("checkpoint_id", ""))
                    checkpoints[checkpoint_id] = WorkflowCheckpointStatus.RESUMED.value
                    if folded.waiting_checkpoint_id == checkpoint_id:
                        folded.waiting_checkpoint_id = None
                        folded.waiting_node_id = None
                        folded.waiting_scope_path = ""
                    folded.status = WorkflowRunStatus.RUNNING
                case "state_patched":
                    raw_patch = payload.get("patch", {})
                    folded.state = merge_json_objects(folded.state, json_object(raw_patch, label="state_patched.patch"))
                case "workflow_run_completed":
                    raw_output = payload.get("output", {})
                    folded.output = json_object(raw_output, label="workflow_run_completed.output")
                    folded.status = WorkflowRunStatus.COMPLETED
                case "workflow_run_failed":
                    folded.status = WorkflowRunStatus.FAILED
                case "workflow_run_cancelled":
                    folded.status = WorkflowRunStatus.CANCELLED
                case _:
                    continue

        return folded

    async def upsert_node_run(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        status: WorkflowNodeStatus,
        attempt: int,
        node_input: JsonObject | None = None,
        output: JsonObject | None = None,
        raw_output: JsonObject | None = None,
        idempotency_key: str | None = None,
        lease_expires_at: datetime | None = None,
    ) -> WorkflowNodeRunModel:
        """Create or update a materialized node-run row."""

        await self.setup()
        model = WorkflowNodeRunModel(
            node_run_id=node_run_key(run_id, node_id, scope_path),
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            status=status.value,
            attempt=attempt,
            idempotency_key=idempotency_key,
            input=dict(node_input or {}),
            output=dict(output) if output is not None else None,
            raw_output=dict(raw_output) if raw_output is not None else None,
            lease_expires_at=lease_expires_at,
            updated_at=datetime.now(),
        )
        saved, _ = await self._engine.upsert(model)
        return saved

    async def get_node_run(self, run_id: str, node_id: str, scope_path: str = "") -> WorkflowNodeRunModel | None:
        """Fetch one materialized node run."""

        await self.setup()
        return await self._engine.find_first(
            WorkflowNodeRunModel,
            filters=ComparisonFilter.eq("node_run_id", node_run_key(run_id, node_id, scope_path)),
        )

    async def list_node_runs(self, run_id: str) -> list[WorkflowNodeRunModel]:
        """List node runs for a workflow run."""

        await self.setup()
        return await self._engine.find_many(WorkflowNodeRunModel, filters=ComparisonFilter.eq("run_id", run_id))

    async def create_checkpoint(
        self,
        *,
        checkpoint_id: str,
        run_id: str,
        node_id: str,
        scope_path: str,
        prompt: str,
        checkpoint_input: JsonObject,
        output_schema: JsonObject | None,
    ) -> WorkflowCheckpointModel:
        """Persist an open human checkpoint."""

        await self.setup()
        checkpoint = WorkflowCheckpointModel(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            node_id=node_id,
            scope_path=scope_path,
            prompt=prompt,
            input=dict(checkpoint_input),
            output_schema=dict(output_schema) if output_schema is not None else None,
        )
        return await self._engine.create(checkpoint)

    async def get_checkpoint(self, checkpoint_id: str) -> WorkflowCheckpointModel | None:
        """Fetch a checkpoint by id."""

        await self.setup()
        return await self._engine.find_first(
            WorkflowCheckpointModel,
            filters=ComparisonFilter.eq("checkpoint_id", checkpoint_id),
        )

    async def update_checkpoint(
        self,
        checkpoint: WorkflowCheckpointModel,
        *,
        status: WorkflowCheckpointStatus,
        output: JsonObject | None = None,
    ) -> WorkflowCheckpointModel:
        """Update a checkpoint state."""

        checkpoint.status = status.value
        if output is not None:
            checkpoint.output = dict(output)
        if status == WorkflowCheckpointStatus.RESUMED:
            checkpoint.resumed_at = datetime.now()
        return await self._engine.update(checkpoint)

    async def create_lease(
        self,
        *,
        run_id: str,
        node_id: str,
        scope_path: str,
        attempt: int,
        worker_id: str,
        ttl_seconds: float,
    ) -> WorkflowLeaseModel:
        """Create a worker lease for an in-flight attempt."""

        await self.setup()
        lease = WorkflowLeaseModel(
            lease_id=f"wf_lease_{uuid.uuid4().hex}",
            run_id=run_id,
            node_run_id=node_run_key(run_id, node_id, scope_path),
            node_id=node_id,
            scope_path=scope_path,
            attempt=attempt,
            worker_id=worker_id,
            expires_at=datetime.now() + timedelta(seconds=ttl_seconds),
        )
        return await self._engine.create(lease)

    async def close_leases(self, *, run_id: str, node_id: str, scope_path: str, status: str) -> None:
        """Mark active leases for a node run as closed."""

        leases = await self._engine.find_many(
            WorkflowLeaseModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("run_id", run_id),
                    ComparisonFilter.eq("node_id", node_id),
                    ComparisonFilter.eq("scope_path", scope_path),
                    ComparisonFilter.eq("status", "active"),
                ]
            ),
        )
        for lease in leases:
            lease.status = status
            lease.updated_at = datetime.now()
            await self._engine.update(lease)

    async def list_expired_running_nodes(self, run_id: str) -> list[WorkflowNodeRunModel]:
        """Return running node attempts whose lease expired."""

        nodes = await self._engine.find_many(
            WorkflowNodeRunModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("run_id", run_id),
                    ComparisonFilter.eq("status", WorkflowNodeStatus.RUNNING.value),
                ]
            ),
        )
        now = datetime.now()
        return [node for node in nodes if node.lease_expires_at is not None and node.lease_expires_at < now]

    async def save_state_snapshot(self, *, run_id: str, state: JsonObject) -> WorkflowStateModel:
        """Persist a folded state snapshot."""

        model = WorkflowStateModel(run_id=run_id, state=dict(state), updated_at=datetime.now())
        saved, _ = await self._engine.upsert(model)
        return saved
