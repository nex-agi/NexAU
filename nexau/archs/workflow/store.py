"""Durable workflow event store and fold logic.

RFC-0027: append-only event log is the canonical workflow state. Materialized
run, node, checkpoint, lease, and state rows are maintained for fast lookup,
but recovery folds events first and treats summaries as indexes.
RFC-0029: parallel map item state is also derived from the event log.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Protocol

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
from nexau.archs.workflow.types import JsonObject, JsonValue, json_array, json_object, merge_json_objects


def _empty_json_object() -> JsonObject:
    return {}


class WorkflowEventPublisher(Protocol):
    """Publishes persisted workflow events to optional live subscribers."""

    def publish_workflow_event(self, event: WorkflowEventModel) -> None:
        """Publish one persisted workflow event."""

        ...


def _empty_node_outputs() -> dict[str, JsonObject]:
    return {}


def _empty_node_status() -> dict[str, str]:
    return {}


def _empty_scoped_state() -> dict[str, JsonObject]:
    return {}


def _empty_node_scope_paths() -> dict[str, str]:
    return {}


def _empty_node_ids() -> dict[str, str]:
    return {}


def _empty_completed_node_order() -> list[str]:
    return []


def _empty_string_set() -> set[str]:
    return set()


def _empty_string_list() -> list[str]:
    return []


def _empty_parallel_maps() -> dict[str, FoldedParallelMapState]:
    return {}


def node_run_key(run_id: str, node_id: str, scope_path: str = "") -> str:
    """Return the stable materialized key for one scoped node instance."""

    normalized_scope = scope_path or "$"
    return f"{run_id}:{node_id}:{normalized_scope}"


def event_payload(**items: JsonValue) -> JsonObject:
    """Build a DB JSON payload from JSON-compatible values."""

    return dict(items)


def _payload_state_scope(payload: JsonObject) -> str:
    raw_scope = payload.get("state_scope_path", "")
    return raw_scope if isinstance(raw_scope, str) else ""


def _apply_state_patch(folded: FoldedWorkflowState, *, patch: JsonObject, state_scope_path: str) -> None:
    if state_scope_path == "":
        folded.state = merge_json_objects(folded.state, patch)
        return
    existing = folded.scoped_state.get(state_scope_path, {})
    folded.scoped_state[state_scope_path] = merge_json_objects(existing, patch)


@dataclass
class FoldedParallelMapItem:
    """One RFC-0029 parallel map item reconstructed from events."""

    index: int
    key: str
    item: JsonValue
    scope_path: str
    body_node_id: str
    status: str = "pending"
    output: JsonObject | None = None
    error: JsonObject | None = None


def _empty_parallel_items() -> list[FoldedParallelMapItem]:
    return []


def _empty_completed_item_order() -> list[str]:
    return []


@dataclass
class FoldedParallelMapState:
    """RFC-0029 parallel map state reconstructed from events."""

    node_id: str
    scope_path: str
    body_node_id: str
    max_concurrency: int
    failure_policy: str
    result_order: str
    items: list[FoldedParallelMapItem] = field(default_factory=_empty_parallel_items)
    completed_item_order: list[str] = field(default_factory=_empty_completed_item_order)

    def item_by_scope(self, scope_path: str) -> FoldedParallelMapItem | None:
        """Return the item for a durable item scope."""

        for item in self.items:
            if item.scope_path == scope_path:
                return item
        return None


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
    scoped_state: dict[str, JsonObject] = field(default_factory=_empty_scoped_state)
    node_inputs_by_key: dict[str, JsonObject] = field(default_factory=_empty_node_outputs)
    node_outputs_by_key: dict[str, JsonObject] = field(default_factory=_empty_node_outputs)
    node_status_by_key: dict[str, str] = field(default_factory=_empty_node_status)
    node_scope_paths_by_key: dict[str, str] = field(default_factory=_empty_node_scope_paths)
    node_ids_by_key: dict[str, str] = field(default_factory=_empty_node_ids)
    completed_node_order: list[str] = field(default_factory=_empty_completed_node_order)
    completed_node_runs: set[str] = field(default_factory=_empty_string_set)
    failed_node_runs: set[str] = field(default_factory=_empty_string_set)
    uncertain_node_runs: set[str] = field(default_factory=_empty_string_set)
    parallel_maps: dict[str, FoldedParallelMapState] = field(default_factory=_empty_parallel_maps)
    waiting_checkpoint_ids: list[str] = field(default_factory=_empty_string_list)
    waiting_checkpoint_id: str | None = None
    waiting_node_id: str | None = None
    waiting_scope_path: str = ""
    last_completed_node_id: str | None = None
    last_completed_scope_path: str = ""

    def node_context(self, scope_prefix: str = "") -> JsonObject:
        """Return the ``nodes`` expression context."""

        result: JsonObject = {}
        for key, status in self.node_status_by_key.items():
            node_id = self.node_ids_by_key.get(key)
            scope_path = self.node_scope_paths_by_key.get(key)
            if node_id is None or scope_path is None:
                continue
            if not self._is_top_level_node_scope(scope_prefix, node_id, scope_path):
                continue
            node_entry: JsonObject = {"status": status}
            output = self.node_outputs_by_key.get(key)
            if output is not None:
                node_entry["output"] = output
            result[node_id] = node_entry
        return result

    def state_for_scope(self, state_scope_path: str) -> JsonObject:
        """Return the durable state object for a graph frame."""

        if state_scope_path == "":
            return self.state
        return self.scoped_state.get(state_scope_path, {})

    def output_for_node(self, run_id: str, node_id: str, scope_path: str) -> JsonObject | None:
        """Return a scoped node output when present."""

        return self.node_outputs_by_key.get(node_run_key(run_id, node_id, scope_path))

    def parallel_item_for_scope(self, scope_path: str) -> FoldedParallelMapItem | None:
        """Return a parallel item whose scope matches or prefixes *scope_path*."""

        for parallel_map in self.parallel_maps.values():
            for item in parallel_map.items:
                if scope_path == item.scope_path or scope_path.startswith(f"{item.scope_path}/"):
                    return item
        return None

    def last_completed_node_id_for_scope(self, scope_prefix: str, candidate_node_ids: set[str]) -> str | None:
        """Return the last completed top-level node inside a graph frame."""

        for key in reversed(self.completed_node_order):
            node_id = self.node_ids_by_key.get(key)
            scope_path = self.node_scope_paths_by_key.get(key)
            if node_id is None or scope_path is None or node_id not in candidate_node_ids:
                continue
            if self._is_top_level_node_scope(scope_prefix, node_id, scope_path):
                return node_id
        return None

    @staticmethod
    def _is_top_level_node_scope(scope_prefix: str, node_id: str, scope_path: str) -> bool:
        if scope_prefix == "":
            return scope_path == ""
        return scope_path == f"{scope_prefix}/{node_id}"


def _payload_has_parallel_item(payload: JsonObject) -> bool:
    return isinstance(payload.get("parallel_node_id"), str)


def _payload_int(payload: JsonObject, key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return default


def _payload_str(payload: JsonObject, key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if isinstance(value, str):
        return value
    return default


def _parallel_map_from_payload(node_id: str, scope_path: str, payload: JsonObject) -> FoldedParallelMapState:
    raw_items = payload.get("items", [])
    item_values = json_array(raw_items, label="parallel_map_started.items") if isinstance(raw_items, list) else []
    body_node_id = _payload_str(payload, "body_node_id", default="")
    parallel_map = FoldedParallelMapState(
        node_id=node_id,
        scope_path=scope_path,
        body_node_id=body_node_id,
        max_concurrency=_payload_int(payload, "max_concurrency", default=1),
        failure_policy=_payload_str(payload, "failure_policy", default="fail_fast"),
        result_order=_payload_str(payload, "result_order", default="input"),
    )
    for item_value in item_values:
        if not isinstance(item_value, dict):
            continue
        item_payload = json_object(item_value, label="parallel_map_started.items[]")
        parallel_map.items.append(
            FoldedParallelMapItem(
                index=_payload_int(item_payload, "index", default=0),
                key=_payload_str(item_payload, "key", default=""),
                item=item_payload.get("item"),
                scope_path=_payload_str(item_payload, "scope_path", default=""),
                body_node_id=_payload_str(item_payload, "body_node_id", default=body_node_id),
            )
        )
    return parallel_map


def _update_parallel_item(
    folded: FoldedWorkflowState,
    run_id: str,
    node_id: str | None,
    scope_path: str,
    payload: JsonObject,
    *,
    status: str,
    output: JsonObject | None = None,
    error: JsonObject | None = None,
) -> None:
    if node_id is None:
        return
    parallel_map = folded.parallel_maps.get(node_run_key(run_id, node_id, scope_path))
    if parallel_map is None:
        return
    item_scope = _payload_str(payload, "item_scope_path", default="")
    item = parallel_map.item_by_scope(item_scope)
    if item is None:
        return
    item.status = status
    if output is not None:
        item.output = output
        if item.scope_path not in parallel_map.completed_item_order:
            parallel_map.completed_item_order.append(item.scope_path)
    if error is not None:
        item.error = error


class WorkflowStore:
    """Repository for workflow durable execution models."""

    def __init__(self, engine: DatabaseEngine, live_event_bus: WorkflowEventPublisher | None = None):
        self._engine = engine
        self._initialized = False
        self._live_event_bus = live_event_bus

    def set_live_event_bus(self, live_event_bus: WorkflowEventPublisher | None) -> None:
        """Attach or clear the optional RFC-0030 live event publisher."""

        self._live_event_bus = live_event_bus

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
        waiting_checkpoint_ids: list[str] | None = None,
    ) -> WorkflowRunModel:
        """Update a materialized run summary."""

        if status is not None:
            run.status = status.value
        if output is not None:
            run.output = dict(output)
        if state is not None:
            run.state = dict(state)
        run.waiting_checkpoint_id = waiting_checkpoint_id
        if waiting_checkpoint_ids is not None:
            run.waiting_checkpoint_ids = list(waiting_checkpoint_ids)
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
        saved = await self._engine.create(event)
        if self._live_event_bus is not None:
            self._live_event_bus.publish_workflow_event(saved)
        return saved

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
        checkpoint_nodes: dict[str, tuple[str | None, str]] = {}

        def refresh_waiting_checkpoints() -> None:
            open_ids = [checkpoint_id for checkpoint_id, status in checkpoints.items() if status == WorkflowCheckpointStatus.OPEN.value]
            folded.waiting_checkpoint_ids = open_ids
            folded.waiting_checkpoint_id = open_ids[0] if open_ids else None
            if folded.waiting_checkpoint_id is not None:
                node_id, scope_path = checkpoint_nodes.get(folded.waiting_checkpoint_id, (None, ""))
                folded.waiting_node_id = node_id
                folded.waiting_scope_path = scope_path
                if folded.status not in {
                    WorkflowRunStatus.COMPLETED,
                    WorkflowRunStatus.FAILED,
                    WorkflowRunStatus.CANCELLED,
                    WorkflowRunStatus.UNCERTAIN,
                }:
                    folded.status = WorkflowRunStatus.WAITING
                return
            folded.waiting_node_id = None
            folded.waiting_scope_path = ""
            if folded.status == WorkflowRunStatus.WAITING:
                folded.status = WorkflowRunStatus.RUNNING

        for event in events:
            payload = json_object(event.payload, label="workflow event payload")
            match event.event_type:
                case "workflow_run_started":
                    folded.status = WorkflowRunStatus.RUNNING
                    raw_inputs = payload.get("inputs", {})
                    folded.inputs = json_object(raw_inputs, label="workflow_run_started.inputs")
                case "node_scheduled":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.SCHEDULED.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.SCHEDULED.value
                case "node_started":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.RUNNING.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        raw_input = payload.get("input")
                        if raw_input is not None:
                            folded.node_inputs_by_key[key] = json_object(raw_input, label="node_started.input")
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.RUNNING.value
                case "node_completed":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.completed_node_runs.add(key)
                        folded.failed_node_runs.discard(key)
                        folded.uncertain_node_runs.discard(key)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.COMPLETED.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        if folded.status in {WorkflowRunStatus.FAILED, WorkflowRunStatus.UNCERTAIN}:
                            folded.status = WorkflowRunStatus.RUNNING
                        raw_output = payload.get("output", {})
                        output = json_object(raw_output, label="node_completed.output")
                        folded.node_outputs_by_key[key] = output
                        raw_patch = payload.get("state_patch")
                        if raw_patch is not None:
                            _apply_state_patch(
                                folded,
                                patch=json_object(raw_patch, label="node_completed.state_patch"),
                                state_scope_path=_payload_state_scope(payload),
                            )
                        folded.completed_node_order.append(key)
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.COMPLETED.value
                            folded.node_outputs[event.node_id] = output
                            folded.last_completed_node_id = event.node_id
                            folded.last_completed_scope_path = event.scope_path
                case "node_failed":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.failed_node_runs.add(key)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.FAILED.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.FAILED.value
                        if not _payload_has_parallel_item(payload):
                            folded.status = WorkflowRunStatus.FAILED
                case "node_retry_scheduled":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.SCHEDULED.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.SCHEDULED.value
                        folded.status = WorkflowRunStatus.RUNNING
                case "node_uncertain":
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.uncertain_node_runs.add(key)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.UNCERTAIN.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.UNCERTAIN.value
                        folded.status = WorkflowRunStatus.UNCERTAIN
                case "checkpoint_created":
                    checkpoint_id = str(payload.get("checkpoint_id", ""))
                    checkpoints[checkpoint_id] = WorkflowCheckpointStatus.OPEN.value
                    checkpoint_nodes[checkpoint_id] = (event.node_id, event.scope_path)
                    refresh_waiting_checkpoints()
                    if event.node_id is not None:
                        key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.node_status_by_key[key] = WorkflowNodeStatus.WAITING.value
                        folded.node_scope_paths_by_key[key] = event.scope_path
                        folded.node_ids_by_key[key] = event.node_id
                        if event.scope_path == "":
                            folded.node_status[event.node_id] = WorkflowNodeStatus.WAITING.value
                case "checkpoint_resumed":
                    checkpoint_id = str(payload.get("checkpoint_id", ""))
                    checkpoints[checkpoint_id] = WorkflowCheckpointStatus.RESUMED.value
                    refresh_waiting_checkpoints()
                case "state_patched":
                    raw_patch = payload.get("patch", {})
                    _apply_state_patch(
                        folded,
                        patch=json_object(raw_patch, label="state_patched.patch"),
                        state_scope_path=_payload_state_scope(payload),
                    )
                case "workflow_run_completed":
                    raw_output = payload.get("output", {})
                    folded.output = json_object(raw_output, label="workflow_run_completed.output")
                    folded.status = WorkflowRunStatus.COMPLETED
                case "workflow_run_failed":
                    folded.status = WorkflowRunStatus.FAILED
                case "workflow_run_cancelled":
                    folded.status = WorkflowRunStatus.CANCELLED
                case "parallel_map_started":
                    if event.node_id is not None:
                        parallel_key = node_run_key(run_id, event.node_id, event.scope_path)
                        folded.parallel_maps[parallel_key] = _parallel_map_from_payload(event.node_id, event.scope_path, payload)
                case "parallel_item_scheduled":
                    _update_parallel_item(
                        folded,
                        run_id,
                        event.node_id,
                        event.scope_path,
                        payload,
                        status=WorkflowNodeStatus.SCHEDULED.value,
                    )
                case "parallel_item_completed":
                    _update_parallel_item(
                        folded,
                        run_id,
                        event.node_id,
                        event.scope_path,
                        payload,
                        status=WorkflowNodeStatus.COMPLETED.value,
                        output=json_object(payload.get("output", {}), label="parallel_item_completed.output"),
                    )
                case "parallel_item_failed":
                    _update_parallel_item(
                        folded,
                        run_id,
                        event.node_id,
                        event.scope_path,
                        payload,
                        status=WorkflowNodeStatus.FAILED.value,
                        error=json_object(payload.get("error", {}), label="parallel_item_failed.error"),
                    )
                case "parallel_item_waiting":
                    _update_parallel_item(
                        folded,
                        run_id,
                        event.node_id,
                        event.scope_path,
                        payload,
                        status=WorkflowNodeStatus.WAITING.value,
                    )
                case "parallel_item_uncertain":
                    _update_parallel_item(
                        folded,
                        run_id,
                        event.node_id,
                        event.scope_path,
                        payload,
                        status=WorkflowNodeStatus.UNCERTAIN.value,
                    )
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
