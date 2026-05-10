"""Unified event and aggregator infrastructure for LLM streams.

This module provides:
1. Core Aggregator ABC and type aliases
2. Multimodal event definitions compatible with ag_ui core architecture
3. Pydantic-based event classes following START → CONTENT → END lifecycle patterns

Usage:
    from nexau.archs.llm.llm_aggregators import (
        Aggregator,
        Event,
        ImageMessageStartEvent,
    )
    from nexau.archs.llm.llm_aggregators.openai_responses import (
        OpenAIResponsesAggregator,
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    BaseEvent,
    EventType,
    RunFinishedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    RunErrorEvent as AgUiRunErrorEvent,
)
from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    RunStartedEvent as AgUiRunStartedEvent,
)
from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    TextMessageStartEvent as AgUiTextMessageStartEvent,
)
from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    ThinkingTextMessageContentEvent as AgUiThinkingTextMessageContentEvent,
)
from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    ThinkingTextMessageEndEvent as AgUiThinkingTextMessageEndEvent,
)
from ag_ui.core.events import (  # pyright: ignore[reportMissingTypeStubs]
    ThinkingTextMessageStartEvent as AgUiThinkingTextMessageStartEvent,
)

from nexau.core.usage import TokenUsage

# ============= Core Aggregator Infrastructure =============


class Aggregator[AggregatorInputT, AggregatorOutputT](ABC):
    """
    Abstract base class for aggregators that accumulate input items into a built output.

    This is a generic pattern for processing sequential inputs (like stream chunks)
    and building a final result. It supports reusability through the clear() method.
    """

    @abstractmethod
    def aggregate(self, item: AggregatorInputT) -> None:
        """
        Aggregate a single input item.

        Args:
            item: The input to aggregate

        Raises:
            RuntimeError: If called after build() or on a completed aggregator
        """
        raise NotImplementedError

    @abstractmethod
    def build(self) -> AggregatorOutputT:
        """
        Build the final result from aggregated items.

        Returns:
            The complete aggregated output

        Raises:
            RuntimeError: If called before any items were aggregated
        """
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """
        Reset the aggregator state for reuse.

        This allows the aggregator to be reused for processing a new sequence
        without creating a new instance.
        """
        raise NotImplementedError


# ============= TEXT MESSAGE EVENTS =============


class TextMessageStartEvent(AgUiTextMessageStartEvent):
    """Text message start event with run_id for multi-agent support.

    Attributes:
        run_id: ID of the agent run that produced this event
    """

    run_id: str


# ============= THINKING MESSAGE EVENTS =============


class ThinkingTextMessageStartEvent(AgUiThinkingTextMessageStartEvent):
    """Thinking message start event with parent_message_id, thinking_message_id and run_id.

    Attributes:
        parent_message_id: ID of the parent message/response (for correlation to parent)
        thinking_message_id: Unique identifier for the thinking message (for correlating Start/Content/End)
        run_id: ID of the agent run that produced this event
        is_redacted: True when this thinking block carries an opaque
            ``redacted_thinking`` payload (Anthropic) instead of plaintext —
            consumers should not expect content events.
            RFC-0023 §阶段 ② extension.
    """

    parent_message_id: str
    thinking_message_id: str
    run_id: str
    is_redacted: bool = False


class ThinkingTextMessageContentEvent(AgUiThinkingTextMessageContentEvent):
    """Thinking message content event with thinking_message_id for correlation.

    Attributes:
        thinking_message_id: Unique identifier linking to the start event
    """

    thinking_message_id: str


class ThinkingTextMessageEndEvent(AgUiThinkingTextMessageEndEvent):
    """Thinking message end event with thinking_message_id for correlation.

    Attributes:
        thinking_message_id: Unique identifier linking to the start event
        signature: Optional reasoning signature (Anthropic ``SignatureDelta`` /
            Gemini ``thoughtSignature``) emitted by the provider for replay
            authentication. RFC-0023 §阶段 ② extension.
        redacted_data: Opaque payload for ``redacted_thinking`` blocks
            (Anthropic) — present only when ``is_redacted=True`` was set on
            the matching Start event. RFC-0023 §阶段 ② extension.
    """

    thinking_message_id: str
    signature: str | None = None
    redacted_data: str | None = None


# ============= IMAGE EVENTS =============


class ImageMessageStartEvent(BaseEvent):
    """Event indicating the start of an image message.

    Attributes:
        message_id: Unique identifier for the message
        mime_type: MIME type of the image (default: image/jpeg)
        run_id: ID of the agent run that produced this event
    """

    type: Literal["IMAGE_MESSAGE_START"] = "IMAGE_MESSAGE_START"  # type: ignore[assignment]
    message_id: str
    mime_type: str = "image/jpeg"
    run_id: str


class ImageMessageContentEvent(BaseEvent):
    """Event containing base64-encoded image data.

    Attributes:
        message_id: Unique identifier for the message (links to StartEvent)
        delta: Base64-encoded image data
    """

    type: Literal["IMAGE_MESSAGE_CONTENT"] = "IMAGE_MESSAGE_CONTENT"  # type: ignore[assignment]
    message_id: str
    delta: str


class ImageMessageEndEvent(BaseEvent):
    """Event indicating the end of an image message.

    Attributes:
        message_id: Unique identifier for the message (links to StartEvent)
    """

    type: Literal["IMAGE_MESSAGE_END"] = "IMAGE_MESSAGE_END"  # type: ignore[assignment]
    message_id: str


# ============= TOOL CALL RESULT EVENT =============


class ToolCallResultEvent(BaseEvent):
    """Event for sending tool execution result back to the LLM display system.

    Note: This is a custom event type defined specifically for our aggregators,
    modeled after the AG UI Core ToolCallResultEvent but without the message_id requirement.

    Attributes:
        tool_call_id: Unique identifier for this tool call
        content: Tool execution result (JSON string or plain text)
        role: Role field for compatibility (typically set to None)
    """

    type: Literal[EventType.TOOL_CALL_RESULT] = EventType.TOOL_CALL_RESULT  # type: ignore[reportIncompatibleVariableOverride]
    tool_call_id: str
    content: str
    role: Literal["tool"] | None = "tool"


# ============= RUN LIFECYCLE EVENTS =============


class RunStartedEvent(AgUiRunStartedEvent):
    """Run started event with full tracing IDs.

    Attributes:
        agent_id: ID of the agent
        root_run_id: ID of the root run
        trace_id: W3C trace id (32-hex) from the OTel span context that
            wrapped agent.run(). Optional — None when no OTel span is
            active. RFC-0024: surfaced here so consumers (UI, tooling)
            can wire trace links live without an out-of-band side channel
            (NAC playground used to stamp this in the gateway tap).
    """

    agent_id: str
    # run_id is in base class
    root_run_id: str
    trace_id: str | None = None


class RunErrorEvent(AgUiRunErrorEvent):
    """Run error event with full tracing IDs.

    Attributes:
        run_id: ID of the agent run
    """

    run_id: str


class CompactionStartedEvent(BaseEvent):
    """Event emitted when context compaction starts."""

    type: Literal["COMPACTION_STARTED"] = "COMPACTION_STARTED"  # type: ignore[assignment]
    run_id: str
    phase: Literal["before_model", "after_model", "wrap_model_call"]
    mode: Literal["regular", "emergency"]
    trigger_reason: str | None = None
    original_message_count: int
    original_token_count: int | None = None
    max_context_tokens: int | None = None


class CompactionFinishedEvent(BaseEvent):
    """Event emitted when context compaction finishes (success or failure)."""

    type: Literal["COMPACTION_FINISHED"] = "COMPACTION_FINISHED"  # type: ignore[assignment]
    run_id: str
    phase: Literal["before_model", "after_model", "wrap_model_call"]
    mode: Literal["regular", "emergency"]
    success: bool
    trigger_reason: str | None = None
    original_message_count: int
    compacted_message_count: int | None = None
    original_token_count: int | None = None
    compacted_token_count: int | None = None
    max_context_tokens: int | None = None
    error: str | None = None


class TransportErrorEvent(BaseEvent):
    """Event indicating a transport-level error (e.g. streaming failure).

    This event is used when an error occurs outside the context of a specific agent run,
    or when the run_id is not available/relevant.

    Attributes:
        message: Error description
        timestamp: Unix timestamp
    """

    type: Literal["TRANSPORT_ERROR"] = "TRANSPORT_ERROR"  # type: ignore[assignment]
    message: str
    # BaseEvent already has timestamp: int | None = None


class UserMessageEvent(BaseEvent):
    """Event for user messages sent to an agent during streaming.

    RFC-0002: 用户消息事件

    Attributes:
        content: Message text
        to_agent_id: Target agent ID
    """

    type: Literal["USER_MESSAGE"] = "USER_MESSAGE"  # type: ignore[assignment]
    content: str
    to_agent_id: str


class TeamMessageEvent(BaseEvent):
    """Event for inter-agent messages via the message bus.

    RFC-0002: Agent 间消息事件

    Attributes:
        content: Message text
        from_agent_id: Sender agent ID
        to_agent_id: Receiver agent ID (None for broadcast)
    """

    type: Literal["TEAM_MESSAGE"] = "TEAM_MESSAGE"  # type: ignore[assignment]
    content: str
    from_agent_id: str
    to_agent_id: str | None = None


class UsageUpdateEvent(BaseEvent):
    """Event emitted after an LLM call completes with canonical token usage."""

    type: Literal["USAGE_UPDATE"] = "USAGE_UPDATE"  # type: ignore[assignment]
    run_id: str
    usage: TokenUsage


class ModelCallFinishedEvent(BaseEvent):
    """Sidecar event emitted once per LLM call carrying per-call metadata
    that doesn't belong on any single message-level event.

    RFC-0023 §阶段 ② — closes the Set A weak gaps for ``model_name`` /
    ``stop_reason`` / ``model_call_id``. Set A previously had no event
    carrying these, so consumers (parity tests, agent_events_middleware)
    had to read ``ModelResponse`` from Set B. With this event Set A is
    self-sufficient.

    Emitted at the END of an LLM call (after all content events and
    after ``UsageUpdateEvent``), exactly once per call. ``model_name`` and
    ``stop_reason`` may be None if the provider doesn't surface them on
    a given response (e.g. truncated stream).

    **Token usage is NOT on this event** — ``UsageUpdateEvent`` (already a
    deployed contract carrying the normalized ``TokenUsage``) is the
    canonical token-counts source. Two events for the same data was an
    earlier RFC draft anti-pattern; resolved by removing usage here.

    **Vendor-specific extras are NOT on this event** either. An earlier
    draft had a ``provider_extras: dict`` pocket for OpenAI Chat's
    ``system_fingerprint``/``service_tier`` and OpenAI Responses' ``status``/
    ``incomplete_details``, but no consumer in the codebase actually
    subscribed to it. Per YAGNI, removed. If a real downstream need
    appears, the inherited ``BaseEvent.raw_event: Any`` slot from ag_ui
    is available without changing this class's schema.

    Attributes:
        run_id: Agent run that issued the LLM call (correlation key).
        message_id: Assistant message id this call produced — same id used
            on the matching ``TextMessageStartEvent``.
        model_name: Vendor-side model identifier (e.g. ``claude-sonnet-4-5``,
            ``gpt-5``, ``gemini-3-flash-preview``).
        model_call_id: Vendor-side response id (Anthropic ``message.id``,
            OpenAI ``id``, Gemini ``responseId``). Useful for tracing back
            to provider-side logs.
        stop_reason: Vendor-specific terminator string, **verbatim**:
            Anthropic ``end_turn`` / ``tool_use`` / ``stop_sequence`` /
            ``max_tokens`` / ``pause_turn`` / ``refusal``; OpenAI Chat
            ``stop`` / ``length`` / ``tool_calls`` / ``content_filter``;
            OpenAI Responses ``completed`` / ``incomplete``; Gemini
            ``STOP`` / ``MAX_TOKENS`` / ``SAFETY`` / ``RECITATION`` / ...
            Cross-provider mapping (if needed) is the consumer's job.
    """

    type: Literal["MODEL_CALL_FINISHED"] = "MODEL_CALL_FINISHED"  # type: ignore[assignment]
    run_id: str
    message_id: str
    model_name: str | None = None
    model_call_id: str | None = None
    stop_reason: str | None = None


class RetryEvent(BaseEvent):
    """Event emitted when an LLM request is about to retry after a transient failure."""

    type: Literal["RETRY"] = "RETRY"  # type: ignore[assignment]
    run_id: str | None = None
    api_type: str
    attempt: int
    max_attempts: int
    backoff_seconds: float
    error_message: str


# ============= UNION TYPES =============

# Unified Event type that includes all AG UI core events and multimodal events
Event = (
    # Text message events (StartEvent has run_id, others link via message_id)
    TextMessageStartEvent
    | TextMessageContentEvent
    | TextMessageEndEvent
    # Thinking message events (StartEvent has run_id)
    | ThinkingTextMessageStartEvent
    | ThinkingTextMessageContentEvent
    | ThinkingTextMessageEndEvent
    # Tool call events (StartEvent has parent_message_id to link to message)
    | ToolCallStartEvent
    | ToolCallArgsEvent
    | ToolCallEndEvent
    # Tool result event (has run_id since it's emitted by middleware, not aggregator)
    | ToolCallResultEvent
    # Run lifecycle events
    | RunStartedEvent
    | RunFinishedEvent
    | RunErrorEvent
    | CompactionStartedEvent
    | CompactionFinishedEvent
    | TransportErrorEvent
    # Image events (StartEvent has run_id, others link via message_id)
    | ImageMessageStartEvent
    | ImageMessageContentEvent
    | ImageMessageEndEvent
    # Team message events (RFC-0002)
    | UserMessageEvent
    | TeamMessageEvent
    | UsageUpdateEvent
    | ModelCallFinishedEvent
    | RetryEvent
)

__all__ = [
    # Core infrastructure
    "Aggregator",
    "Event",
    "RunStartedEvent",
    "RunFinishedEvent",
    "RunErrorEvent",
    "CompactionStartedEvent",
    "CompactionFinishedEvent",
    "TransportErrorEvent",
    "UsageUpdateEvent",
    "ModelCallFinishedEvent",
    "RetryEvent",
    # Text message events
    "TextMessageStartEvent",
    "TextMessageContentEvent",
    "TextMessageEndEvent",
    # Thinking message events
    "ThinkingTextMessageStartEvent",
    "ThinkingTextMessageContentEvent",
    "ThinkingTextMessageEndEvent",
    # Tool call events
    "ToolCallStartEvent",
    "ToolCallArgsEvent",
    "ToolCallEndEvent",
    "ToolCallResultEvent",
    # Image events
    "ImageMessageStartEvent",
    "ImageMessageContentEvent",
    "ImageMessageEndEvent",
    # Team message events
    "UserMessageEvent",
    "TeamMessageEvent",
]
