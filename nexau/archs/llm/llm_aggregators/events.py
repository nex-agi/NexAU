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
    """

    parent_message_id: str
    thinking_message_id: str
    run_id: str


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
    """

    thinking_message_id: str


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
    """

    agent_id: str
    # run_id is in base class
    root_run_id: str


class RunErrorEvent(AgUiRunErrorEvent):
    """Run error event with full tracing IDs.

    Attributes:
        run_id: ID of the agent run
    """

    run_id: str


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
    | TransportErrorEvent
    # Image events (StartEvent has run_id, others link via message_id)
    | ImageMessageStartEvent
    | ImageMessageContentEvent
    | ImageMessageEndEvent
)

__all__ = [
    # Core infrastructure
    "Aggregator",
    "Event",
    "RunStartedEvent",
    "RunFinishedEvent",
    "RunErrorEvent",
    "TransportErrorEvent",
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
]
