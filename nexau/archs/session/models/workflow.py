"""Workflow durable execution storage models.

RFC-0027: workflow event log, run summary, node runs, checkpoints, leases, and
state snapshots.
RFC-0029: parallel map run summaries can expose multiple open checkpoints.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


def _empty_payload() -> JsonObject:
    return {}


def _empty_string_list() -> list[str]:
    return []


class WorkflowRunStatus(StrEnum):
    """Materialized workflow run states."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNCERTAIN = "uncertain"


class WorkflowNodeStatus(StrEnum):
    """Materialized workflow node states."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"
    UNCERTAIN = "uncertain"


class WorkflowCheckpointStatus(StrEnum):
    """Human checkpoint states."""

    OPEN = "open"
    RESUMED = "resumed"
    CANCELLED = "cancelled"


class WorkflowEventModel(SQLModel, table=True):
    """Append-only canonical workflow event."""

    event_id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    sequence: int = Field(index=True)
    event_type: str = Field(index=True)
    node_id: str | None = Field(default=None, index=True)
    scope_path: str = Field(default="", index=True)
    attempt: int | None = Field(default=None)
    payload: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now, index=True)


class WorkflowRunModel(SQLModel, table=True):
    """Materialized workflow run summary."""

    run_id: str = Field(primary_key=True)
    workflow_name: str = Field(index=True)
    status: str = Field(default=WorkflowRunStatus.PENDING.value, index=True)
    user_id: str | None = Field(default=None, index=True)
    session_id: str | None = Field(default=None, index=True)
    input: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    output: JsonObject | None = Field(default=None, sa_column=Column(JSON))
    state: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    definition_snapshot: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    waiting_checkpoint_id: str | None = Field(default=None, index=True)
    waiting_checkpoint_ids: list[str] = Field(default_factory=_empty_string_list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowNodeRunModel(SQLModel, table=True):
    """Materialized node status and output."""

    node_run_id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    node_id: str = Field(index=True)
    scope_path: str = Field(default="", index=True)
    status: str = Field(default=WorkflowNodeStatus.SCHEDULED.value, index=True)
    attempt: int = Field(default=0)
    idempotency_key: str | None = None
    input: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    output: JsonObject | None = Field(default=None, sa_column=Column(JSON))
    raw_output: JsonObject | None = Field(default=None, sa_column=Column(JSON))
    lease_expires_at: datetime | None = Field(default=None, index=True)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowCheckpointModel(SQLModel, table=True):
    """Durable human/permission checkpoint."""

    checkpoint_id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    node_id: str = Field(index=True)
    scope_path: str = Field(default="", index=True)
    status: str = Field(default=WorkflowCheckpointStatus.OPEN.value, index=True)
    prompt: str
    input: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    output_schema: JsonObject | None = Field(default=None, sa_column=Column(JSON))
    output: JsonObject | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.now)
    resumed_at: datetime | None = None


class WorkflowLeaseModel(SQLModel, table=True):
    """Worker lease for an in-flight node attempt."""

    lease_id: str = Field(primary_key=True)
    run_id: str = Field(index=True)
    node_run_id: str = Field(index=True)
    node_id: str = Field(index=True)
    scope_path: str = Field(default="", index=True)
    attempt: int
    worker_id: str = Field(index=True)
    expires_at: datetime = Field(index=True)
    status: str = Field(default="active", index=True)
    updated_at: datetime = Field(default_factory=datetime.now)


class WorkflowStateModel(SQLModel, table=True):
    """Optional folded workflow state snapshot."""

    run_id: str = Field(primary_key=True)
    state: JsonObject = Field(default_factory=_empty_payload, sa_column=Column(JSON))
    updated_at: datetime = Field(default_factory=datetime.now)


WORKFLOW_MODELS: list[type[SQLModel]] = [
    WorkflowEventModel,
    WorkflowRunModel,
    WorkflowNodeRunModel,
    WorkflowCheckpointModel,
    WorkflowLeaseModel,
    WorkflowStateModel,
]
