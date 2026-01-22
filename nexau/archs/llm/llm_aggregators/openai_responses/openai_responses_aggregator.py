from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from openai.types.responses import Response, ResponseStreamEvent
from openai.types.responses.response_content_part_added_event import (
    ResponseContentPartAddedEvent,
)
from openai.types.responses.response_content_part_done_event import (
    ResponseContentPartDoneEvent,
)
from openai.types.responses.response_function_call_arguments_delta_event import ResponseFunctionCallArgumentsDeltaEvent
from openai.types.responses.response_function_call_arguments_done_event import ResponseFunctionCallArgumentsDoneEvent
from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_refusal import ResponseOutputRefusal
from openai.types.responses.response_output_text import ResponseOutputText
from openai.types.responses.response_reasoning_item import ResponseReasoningItem, Summary
from openai.types.responses.response_reasoning_summary_part_added_event import ResponseReasoningSummaryPartAddedEvent
from openai.types.responses.response_reasoning_summary_part_done_event import ResponseReasoningSummaryPartDoneEvent
from openai.types.responses.response_reasoning_summary_text_delta_event import ResponseReasoningSummaryTextDeltaEvent
from openai.types.responses.response_reasoning_summary_text_done_event import ResponseReasoningSummaryTextDoneEvent
from openai.types.responses.response_refusal_delta_event import ResponseRefusalDeltaEvent
from openai.types.responses.response_refusal_done_event import ResponseRefusalDoneEvent
from openai.types.responses.response_text_delta_event import ResponseTextDeltaEvent
from openai.types.responses.response_text_done_event import ResponseTextDoneEvent

from ..events import (
    Aggregator,
    Event,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

logger = logging.getLogger(__name__)


class OpenAIResponsesAggregator(Aggregator[ResponseStreamEvent, Response]):
    """
    Aggregates a stream of response events into a complete response.

    This class maintains state while aggregating stream events and provides reusability
    through the clear() method.
    """

    def __init__(self, *, on_event: Callable[[Event], None], run_id: str) -> None:
        self._on_event = on_event
        self._run_id = run_id
        self._output_aggregators: dict[str, _ReasoningItemAggregator | _FunctionCallItemAggregator | _MessageItemAggregator] = {}
        # Initialize _value with empty Response (output will be built/merged in build())
        self._value = Response(
            id="",
            created_at=0,
            model="",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

    def aggregate(self, item: ResponseStreamEvent) -> None:
        """
        Aggregate a stream event.

        Args:
            item: A single event from the OpenAI responses stream
        """

        # Handle different event types
        if item.type == "response.created":
            # Initialize response with created data
            if item.response:
                if item.response.id:
                    self._value.id = item.response.id
                if item.response.model:
                    self._value.model = item.response.model
                if item.response.created_at:
                    self._value.created_at = item.response.created_at
            return

        if item.type == "response.output_item.added":
            # Create a new aggregator for this output item based on type
            if len(self._value.output) != item.output_index:
                raise ValueError(
                    f"Invalid output_index {item.output_index} for output_item.added event. "
                    f"Expected index {len(self._value.output)} to append to output list. "
                    "output_item.added events must be processed in order of output_index."
                )
            self._value.output.append(item.item)

            # Inline aggregator creation logic
            if item.item.type == "reasoning":
                aggregator = _ReasoningItemAggregator(
                    item_id=item.item.id,
                    output_index=item.output_index,
                    on_event=self._on_event,
                    run_id=self._run_id,
                    response_id=self._value.id,
                )
                aggregator.aggregate(item.item)
                self._output_aggregators[item.item.id] = aggregator
                return

            if item.item.type == "function_call":
                if not item.item.id:
                    raise ValueError("Function call item must have an ID for aggregation")
                # Extract function call name from item data if available
                function_aggregator = _FunctionCallItemAggregator(
                    item_id=item.item.id,
                    output_index=item.output_index,
                    on_event=self._on_event,
                    response_id=self._value.id,
                )
                function_aggregator.aggregate(item.item)
                self._output_aggregators[item.item.id] = function_aggregator
                return

            if item.item.type == "message":
                if not item.item.id:
                    raise ValueError("Message item must have an ID for aggregation")
                # Create message aggregator for text content handling
                message_aggregator = _MessageItemAggregator(
                    item_id=item.item.id,
                    output_index=item.output_index,
                    on_event=self._on_event,
                    run_id=self._run_id,
                )
                message_aggregator.aggregate(item.item)
                self._output_aggregators[item.item.id] = message_aggregator
                return

            return

        if item.type == "response.output_item.done":
            if item.item.type == "reasoning":
                agg = self._output_aggregators.get(item.item.id)
                if agg and isinstance(agg, _ReasoningItemAggregator):
                    agg.aggregate(item.item)
                return

            if item.item.type == "function_call":
                if not item.item.id:
                    raise ValueError("Function call item must have an ID for aggregation")
                agg = self._output_aggregators.get(item.item.id)
                if agg and isinstance(agg, _FunctionCallItemAggregator):
                    agg.aggregate(item.item)
                return

            if item.item.type == "message":
                if not item.item.id:
                    raise ValueError("Message item must have an ID for aggregation")
                agg = self._output_aggregators.get(item.item.id)
                if agg and isinstance(agg, _MessageItemAggregator):
                    agg.aggregate(item.item)
                return

            self._value.output[item.output_index] = item.item

            return

        if (
            item.type == "response.reasoning_summary_text.delta"
            or item.type == "response.reasoning_summary_text.done"
            or item.type == "response.reasoning_summary_part.added"
            or item.type == "response.reasoning_summary_part.done"
        ):
            agg = self._output_aggregators.get(item.item_id)
            if agg and isinstance(agg, _ReasoningItemAggregator):
                agg.aggregate(item)
            return

        if item.type == "response.function_call_arguments.delta" or item.type == "response.function_call_arguments.done":
            agg = self._output_aggregators.get(item.item_id)
            if agg and isinstance(agg, _FunctionCallItemAggregator):
                agg.aggregate(item)
            return

        if (
            item.type == "response.content_part.added"
            or item.type == "response.content_part.done"
            or item.type == "response.output_text.delta"
            or item.type == "response.output_text.done"
            or item.type == "response.refusal.delta"
            or item.type == "response.refusal.done"
        ):
            agg = self._output_aggregators.get(item.item_id)
            if agg and isinstance(agg, _MessageItemAggregator):
                agg.aggregate(item)
            return

        if item.type == "response.completed":
            # Update status in metadata (no action needed for now)
            self._value = item.response
            return

    def build(self) -> Response:
        """
        Build the final response object from aggregated events.

        Returns:
            The complete Response object

        Raises:
            RuntimeError: If no valid events were received
        """
        new_value = self._value.model_copy(deep=True)
        # Build final output items and merge them into _value.output by output_index
        for item_id in sorted(self._output_aggregators.keys(), key=lambda x: self._output_aggregators[x].output_index):
            output_item = self._output_aggregators[item_id].build()
            output_index = self._output_aggregators[item_id].output_index
            # Replace the item at output_index with the built output
            new_value.output[output_index] = output_item

        return new_value

    def clear(self) -> None:
        """
        Reset the aggregator state for reuse.

        This allows the aggregator to be reused for processing a new stream
        without creating a new instance.
        """
        self._output_aggregators.clear()
        # Reset _value completely for a new stream
        self._value = Response(
            id="",
            created_at=0,
            model="",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )


ReasoningItemAggregatorEvent = (
    ResponseReasoningItem
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningSummaryPartDoneEvent
)


class _ReasoningItemAggregator(Aggregator[ReasoningItemAggregatorEvent, ResponseReasoningItem]):
    """
    Aggregates a reasoning output item within a response stream.

    Handles reasoning content and summary generation.
    Emits ThinkingTextMessage events:
    - StartEvent on first summary_part.added
    - ContentEvent on text delta chunks
    - EndEvent on summary_text.done

    The aggregator tracks state internally and builds a ResponseReasoningItem.
    """

    def __init__(
        self,
        *,
        item_id: str,
        output_index: int,
        on_event: Callable[[Event], None],
        run_id: str,
        response_id: str,
    ) -> None:
        self._item_id = item_id
        self.output_index = output_index
        self._on_event = on_event
        self._run_id = run_id
        self._response_id = response_id
        # Initialize _value with empty ResponseReasoningItem
        self._value = ResponseReasoningItem(
            id=self._item_id,
            type="reasoning",
            summary=[],
            content=None,
            encrypted_content=None,
            status=None,
        )

    def aggregate(self, item: ReasoningItemAggregatorEvent) -> None:
        """Aggregate a reasoning content event.

        Args:
            item: A single event for reasoning content (output_item state, summary parts, text deltas, etc.)
        """

        # Handle different event types

        if item.type == "reasoning":
            self._value = item.model_copy(deep=True)
            return

        if item.type == "response.reasoning_summary_part.added":
            # Validate summary_index ordering - must append in order to summary list
            if len(self._value.summary) != item.summary_index:
                raise ValueError(
                    f"Invalid summary_index {item.summary_index} for summary_part.added event. "
                    f"Current summary length: {len(self._value.summary)}. "
                    f"summary_part.added events must be processed in order of summary_index."
                )
            self._value.summary.append(Summary(text="", type="summary_text"))
            # Emit start event - reasoning content begins with first summary part
            self._on_event(
                ThinkingTextMessageStartEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    parent_message_id=self._response_id,
                    thinking_message_id=self._item_id,
                    run_id=self._run_id,
                )
            )
            return

        if item.type == "response.reasoning_summary_text.delta":
            # Add text delta to the current summary text
            summary = self._value.summary[item.summary_index]
            summary.text += item.delta
            # Emit content chunk event for real-time updates (progressive display)
            self._on_event(
                ThinkingTextMessageContentEvent(
                    delta=item.delta,
                    timestamp=int(datetime.now().timestamp() * 1000),
                    thinking_message_id=self._item_id,
                )
            )
            return

        if item.type == "response.reasoning_summary_text.done":
            # Set the final text for this summary part (ensures completeness)
            summary = self._value.summary[item.summary_index]
            summary.text = item.text
            # Emit end event - summary part is complete
            self._on_event(
                ThinkingTextMessageEndEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    thinking_message_id=self._item_id,
                )
            )
            return

        if item.type == "response.reasoning_summary_part.done":
            # No additional action needed - summary part completion is tracked by text.done
            return

    def build(self) -> ResponseReasoningItem:
        """Build the final reasoning output item object.

        Returns:
            A deep copy of the fully aggregated ResponseReasoningItem
        """
        # Return a copy of the aggregated value with current summary
        return self._value.model_copy(deep=True)

    def clear(self) -> None:
        """Reset the aggregator state for reuse.

        Clears all stored summary content and prepares for a new aggregation session.
        """
        # Reset _value to initial state
        self._value = ResponseReasoningItem(
            id=self._item_id,
            type="reasoning",
            summary=[],
            content=None,
            encrypted_content=None,
            status=None,
        )


MessageItemAggregatorEvent = (
    ResponseOutputMessage
    | ResponseContentPartAddedEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseContentPartDoneEvent
)


class _MessageItemAggregator(Aggregator[MessageItemAggregatorEvent, ResponseOutputMessage]):
    """
    Aggregates a message output item within a response stream.

    Handles message content parts and text generation.
    Emits TextMessage events:
    - StartEvent on first content_part.added
    - ContentEvent on text delta chunks
    - EndEvent on output_text.done

    The aggregator tracks state internally and builds a ResponseOutputMessage.
    """

    def __init__(
        self,
        *,
        item_id: str,
        output_index: int,
        on_event: Callable[[Event], None],
        run_id: str,
    ) -> None:
        self._item_id = item_id
        self.output_index = output_index
        self._on_event = on_event
        self._run_id = run_id
        self._started = False
        # Initialize _value with empty ResponseOutputMessage
        self._value = ResponseOutputMessage(
            id=self._item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )

    def aggregate(self, item: MessageItemAggregatorEvent) -> None:
        """Aggregate a message content event.

        Args:
            item: A single event for message content (output_item state, content parts, text deltas, etc.)
        """

        # Handle different event types

        if item.type == "message":
            # Initialize message from output_item.added event data
            self._value = item.model_copy(deep=True)
            return

        if item.type == "response.content_part.added":
            # Validate content_index ordering - must append in order to content list
            if len(self._value.content) != item.content_index:
                raise ValueError(
                    f"Invalid content_index {item.content_index} for content_part.added event. "
                    f"Current content length: {len(self._value.content)}. "
                    f"content_part.added events must be processed in order of content_index."
                )
            # Handle different content types
            if item.part.type == "output_text":
                # Create text content part
                self._value.content.append(ResponseOutputText(type="output_text", text="", annotations=[]))
            elif item.part.type == "refusal":
                # Create refusal content part
                self._value.content.append(ResponseOutputRefusal(type="refusal", refusal=""))

            # Emit start event - message content begins with first content part
            if not self._started:
                self._started = True
                self._on_event(
                    TextMessageStartEvent(
                        timestamp=int(datetime.now().timestamp() * 1000),
                        message_id=self._item_id,
                        role="assistant",
                        run_id=self._run_id,
                    )
                )
            return

        if item.type == "response.output_text.delta" or item.type == "response.refusal.delta":
            # Add text delta to the current content text
            content_part = self._value.content[item.content_index]
            # Cast to ResponseOutputText - type is guaranteed by event structure
            if content_part.type == "output_text":
                content_part.text += item.delta
            if content_part.type == "refusal":
                content_part.refusal += item.delta
            # Emit content chunk event for real-time updates
            self._on_event(
                TextMessageContentEvent(
                    delta=item.delta,
                    timestamp=int(datetime.now().timestamp() * 1000),
                    message_id=self._item_id,
                )
            )
            return

        if item.type == "response.output_text.done":
            # Set the final text for this content part
            content_part = self._value.content[item.content_index]
            # Cast and set final text value
            if content_part.type == "output_text":
                content_part.text = item.text
            # Emit end event - content part is complete
            self._on_event(
                TextMessageEndEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    message_id=self._item_id,
                )
            )
            return

        if item.type == "response.refusal.done":
            # Set the final text for this content part
            content_part = self._value.content[item.content_index]
            # Cast and set final text value
            if content_part.type == "refusal":
                content_part.refusal = item.refusal
            # Emit end event - content part is complete
            self._on_event(
                TextMessageEndEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    message_id=self._item_id,
                )
            )
            return

        if item.type == "response.content_part.done":
            # No additional action needed - content part completion is tracked by text.done
            return

    def build(self) -> ResponseOutputMessage:
        """Build the final message output item object.

        Returns:
            A deep copy of the fully aggregated ResponseOutputMessage
        """
        # Return a copy of the aggregated value with current content
        return self._value.model_copy(deep=True)

    def clear(self) -> None:
        """Reset the aggregator state for reuse.

        Clears all stored message content and prepares for a new aggregation session.
        """
        self._started = False
        # Reset _value to initial state
        self._value = ResponseOutputMessage(
            id=self._item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )


FunctionCallItemAggregatorEvent = (
    ResponseFunctionToolCall | ResponseFunctionCallArgumentsDeltaEvent | ResponseFunctionCallArgumentsDoneEvent
)


class _FunctionCallItemAggregator(Aggregator[FunctionCallItemAggregatorEvent, ResponseFunctionToolCall]):
    """
    Aggregates a function call output item within a response stream.

    Handles function call arguments and emits tool call events.
    Emits ToolCall events:
    - StartEvent on first arguments.delta event
    - ArgsEvent/ChunkEvent on delta events
    - EndEvent on arguments.done event

    The aggregator tracks state internally and builds a ResponseFunctionToolCall.
    """

    def __init__(
        self,
        *,
        item_id: str,
        output_index: int,
        on_event: Callable[[Event], None],
        response_id: str,
    ) -> None:
        self._item_id = item_id
        self.output_index = output_index
        self._on_event = on_event
        self._response_id = response_id
        self._started = False
        # Initialize _value with empty ResponseFunctionToolCall
        self._value = ResponseFunctionToolCall(
            id=self._item_id,
            type="function_call",
            call_id="",
            name="",
            arguments="",  # Arguments will be built from delta events
            status=None,
        )

    def aggregate(self, item: FunctionCallItemAggregatorEvent) -> None:
        """Aggregate a function call arguments event.

        Args:
            item: A single event for function call arguments (output_item state, delta, or done)
        """

        if item.type == "function_call":
            # The raw function call item from output_item.added
            self._value = item.model_copy(deep=True)
            if not self._started:
                self._started = True
                self._on_event(
                    ToolCallStartEvent(
                        timestamp=int(datetime.now().timestamp() * 1000),
                        tool_call_id=self._value.call_id,
                        tool_call_name=self._value.name,
                        parent_message_id=self._response_id,
                    )
                )
            return

        if item.type == "response.function_call_arguments.delta":
            # Add delta to current arguments
            self._value.arguments += item.delta

            # Emit delta content event
            self._on_event(
                ToolCallArgsEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    tool_call_id=self._value.call_id,
                    delta=item.delta,
                )
            )
            return

        if item.type == "response.function_call_arguments.done":
            # Set final arguments from done event
            self._value.arguments = item.arguments

            # Emit end event - function call is complete
            self._on_event(
                ToolCallEndEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    tool_call_id=self._value.call_id,
                )
            )
            return

    def build(self) -> ResponseFunctionToolCall:
        """Build the final function call output item object.

        Returns:
            A deep copy of the fully aggregated ResponseFunctionToolCall
        """
        # Return a copy of the aggregated value with current arguments
        return self._value.model_copy(deep=True)

    def clear(self) -> None:
        """Reset the aggregator state for reuse.

        Clears all stored function call arguments and prepares for a new aggregation session.
        """
        self._started = False
        # Reset _value to initial state and clear started flag
        self._value = ResponseFunctionToolCall(
            id=self._item_id,
            type="function_call",
            call_id="",
            name="",
            arguments="",
            status=None,
        )
