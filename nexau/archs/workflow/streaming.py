"""Live workflow stream envelopes and in-process event bus.

RFC-0030: Workflow 与 Agent 事件统一流式输出

Durable workflow events remain persisted in ``WorkflowStore``. Agent events are
live-only by default and are wrapped with workflow scope metadata before they
are sent to HTTP SSE clients.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nexau.archs.llm.llm_aggregators.events import Event
from nexau.archs.session.models.workflow import WorkflowEventModel
from nexau.archs.workflow.types import JsonObject, json_object, json_value

WorkflowStreamEnvelopeType = Literal["workflow_event", "agent_event", "stream_status", "error", "complete"]


def _empty_json_object() -> JsonObject:
    return {}


def _timestamp_ms() -> int:
    return int(datetime.now().timestamp() * 1000)


class WorkflowStreamOptions(BaseModel):
    """Request options for RFC-0030 live workflow streams."""

    model_config = ConfigDict(extra="forbid")

    include_workflow_events: bool = True
    include_agent_events: bool = True
    include_thinking_events: bool = False
    include_usage_events: bool = True
    include_tool_events: bool = True
    include_text_deltas: bool = True
    persist_agent_events: bool = False
    cancel_on_disconnect: bool = False


class WorkflowStreamContext(BaseModel):
    """Workflow scope metadata attached to nested agent events."""

    model_config = ConfigDict(extra="forbid")

    workflow_name: str
    graph_id: str
    node_id: str
    node_type: str
    scope_path: str
    parent_node_id: str | None = None
    subgraph: str | None = None
    depth: int = 0
    parallel_node_id: str | None = None
    item_index: int | None = None
    item_key: str | None = None
    item_scope_path: str | None = None


class WorkflowStreamAgentPayload(BaseModel):
    """Nested Agent/LLM event payload with workflow-local identity."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    agent_run_id: str | None = None
    session_id: str | None = None
    event: JsonObject


class WorkflowStreamEnvelope(BaseModel):
    """Single event sent on the RFC-0030 workflow live SSE connection."""

    model_config = ConfigDict(extra="forbid")

    type: WorkflowStreamEnvelopeType
    stream_sequence: int = 0
    run_id: str
    timestamp: int = Field(default_factory=_timestamp_ms)
    workflow: WorkflowStreamContext | None = None
    workflow_event: JsonObject | None = None
    agent: WorkflowStreamAgentPayload | None = None
    status: str | None = None
    payload: JsonObject = Field(default_factory=_empty_json_object)
    error: str | None = None
    persisted: bool = False


def workflow_event_payload(event: WorkflowEventModel) -> JsonObject:
    """Return the SSE-compatible payload for one persisted workflow event."""

    return {
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
        "parallel_node_id": event.payload.get("parallel_node_id"),
        "item_index": event.payload.get("item_index"),
        "item_key": event.payload.get("item_key"),
        "item_scope_path": event.payload.get("item_scope_path"),
        "body_node_id": event.payload.get("body_node_id"),
        "payload": event.payload,
        "created_at": event.created_at.isoformat(),
    }


def agent_event_payload(event: Event) -> JsonObject:
    """Serialize a unified Agent/LLM event to a JSON object."""

    return json_object(json_value(event.model_dump(mode="json")), label="agent event")


def agent_event_type(event_payload: JsonObject) -> str:
    """Return a normalized event type string from a serialized agent event."""

    raw_type = event_payload.get("type")
    return raw_type if isinstance(raw_type, str) else ""


def agent_event_allowed(event_payload: JsonObject, options: WorkflowStreamOptions) -> bool:
    """Return whether an agent event should be sent for the stream options."""

    event_type = agent_event_type(event_payload)
    if event_type.startswith("THINKING_"):
        return options.include_thinking_events
    if event_type == "TEXT_MESSAGE_CONTENT":
        return options.include_text_deltas
    if event_type.startswith("TOOL_CALL_"):
        return options.include_tool_events
    if event_type == "USAGE_UPDATE":
        return options.include_usage_events
    return True


class WorkflowLiveSubscription:
    """One in-process subscriber for a workflow live stream."""

    def __init__(self, *, run_id: str, options: WorkflowStreamOptions, queue_size: int):
        self.run_id = run_id
        self.options = options
        self.loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[WorkflowStreamEnvelope | None] = asyncio.Queue(maxsize=queue_size)
        self._stream_sequence = 0
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[WorkflowStreamEnvelope]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    def enqueue(self, envelope: WorkflowStreamEnvelope) -> None:
        """Add an envelope to this subscriber queue."""

        if self._closed:
            return
        self._stream_sequence += 1
        queued = envelope.model_copy(update={"stream_sequence": self._stream_sequence})
        try:
            self._queue.put_nowait(queued)
        except asyncio.QueueFull:
            if envelope.type == "workflow_event":
                self._queue.get_nowait()
                self._queue.put_nowait(queued)
                return
            dropped = WorkflowStreamEnvelope(
                type="stream_status",
                run_id=self.run_id,
                status="agent_event_dropped",
                payload={"reason": "subscriber_queue_full"},
            )
            self._stream_sequence += 1
            self._queue.get_nowait()
            self._queue.put_nowait(dropped.model_copy(update={"stream_sequence": self._stream_sequence}))

    def close(self) -> None:
        """Close the subscriber queue."""

        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            self._queue.get_nowait()
            self._queue.put_nowait(None)


class WorkflowLiveEventBus:
    """In-process live event bus for RFC-0030 workflow streams."""

    def __init__(self, *, queue_size: int = 0):
        self._queue_size = queue_size
        self._subscriptions: dict[str, list[WorkflowLiveSubscription]] = defaultdict(list)

    def subscribe(self, run_id: str, *, options: WorkflowStreamOptions | None = None) -> WorkflowLiveSubscription:
        """Subscribe to live events for *run_id*."""

        subscription = WorkflowLiveSubscription(
            run_id=run_id,
            options=options or WorkflowStreamOptions(),
            queue_size=self._queue_size,
        )
        self._subscriptions[run_id].append(subscription)
        return subscription

    def unsubscribe(self, subscription: WorkflowLiveSubscription) -> None:
        """Remove and close one subscription."""

        subscriptions = self._subscriptions.get(subscription.run_id)
        if subscriptions is not None and subscription in subscriptions:
            subscriptions.remove(subscription)
            if not subscriptions:
                self._subscriptions.pop(subscription.run_id, None)
        subscription.close()

    def publish_workflow_event(self, event: WorkflowEventModel) -> None:
        """Publish a persisted workflow event to current live subscribers."""

        envelope = WorkflowStreamEnvelope(
            type="workflow_event",
            run_id=event.run_id,
            workflow_event=workflow_event_payload(event),
            persisted=True,
        )
        self._publish(event.run_id, envelope)

    def publish_agent_event(
        self,
        *,
        run_id: str,
        workflow: WorkflowStreamContext,
        agent_name: str,
        agent_run_id: str | None,
        session_id: str | None,
        event: Event,
    ) -> None:
        """Publish a live-only nested agent event."""

        event_payload = agent_event_payload(event)
        envelope = WorkflowStreamEnvelope(
            type="agent_event",
            run_id=run_id,
            workflow=workflow,
            agent=WorkflowStreamAgentPayload(
                agent_name=agent_name,
                agent_run_id=agent_run_id,
                session_id=session_id,
                event=event_payload,
            ),
        )
        self._publish(run_id, envelope, agent_event=event_payload)

    def publish_status(self, *, run_id: str, status: str, payload: JsonObject | None = None) -> None:
        """Publish a live stream status envelope."""

        self._publish(
            run_id,
            WorkflowStreamEnvelope(
                type="stream_status",
                run_id=run_id,
                status=status,
                payload=payload or {},
            ),
        )

    def publish_error(self, *, run_id: str, error: str) -> None:
        """Publish a live stream error envelope."""

        self._publish(run_id, WorkflowStreamEnvelope(type="error", run_id=run_id, error=error))

    def publish_complete(self, *, run_id: str, status: str, payload: JsonObject | None = None) -> None:
        """Publish a live stream completion envelope."""

        self._publish(
            run_id,
            WorkflowStreamEnvelope(
                type="complete",
                run_id=run_id,
                status=status,
                payload=payload or {},
            ),
        )

    def close(self, run_id: str) -> None:
        """Close all subscriptions for *run_id*."""

        subscriptions = self._subscriptions.pop(run_id, [])
        for subscription in subscriptions:
            self._close_subscription(subscription)

    def _publish(self, run_id: str, envelope: WorkflowStreamEnvelope, *, agent_event: JsonObject | None = None) -> None:
        subscriptions = tuple(self._subscriptions.get(run_id, ()))
        for subscription in subscriptions:
            if envelope.type == "workflow_event" and not subscription.options.include_workflow_events:
                continue
            if envelope.type == "agent_event":
                if not subscription.options.include_agent_events:
                    continue
                if agent_event is not None and not agent_event_allowed(agent_event, subscription.options):
                    continue
            self._enqueue_subscription(subscription, envelope)

    @staticmethod
    def _enqueue_subscription(subscription: WorkflowLiveSubscription, envelope: WorkflowStreamEnvelope) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is subscription.loop:
            subscription.enqueue(envelope)
            return
        subscription.loop.call_soon_threadsafe(subscription.enqueue, envelope)

    @staticmethod
    def _close_subscription(subscription: WorkflowLiveSubscription) -> None:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is subscription.loop:
            subscription.loop.call_soon(subscription.close)
            return
        subscription.loop.call_soon_threadsafe(subscription.close)
