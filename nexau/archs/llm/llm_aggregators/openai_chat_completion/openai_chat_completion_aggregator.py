from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Literal

from openai.types.chat.chat_completion import (
    ChatCompletion,
)
from openai.types.chat.chat_completion import (
    Choice as ChatCompletionChoice,
)
from openai.types.chat.chat_completion import (
    ChoiceLogprobs as ChatCompletionChoiceLogprobs,
)
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChatCompletionChunkChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceLogprobs as ChatCompletionChunkChoiceLogprobs,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion_message_tool_call import (
    Function as ChatCompletionMessageToolCallFunction,
)

from ..events import (
    Aggregator,
    Event,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)

# Re-export types from OpenAI SDK for easy reuse
# These match the Literal types used in OpenAI's type annotations
FINISH_REASON_LITERAL = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
SERVICE_TIER_LITERAL = Literal["auto", "default", "flex", "scale", "priority"]


def _noop_event_handler(_: Event) -> None:
    """No-op event handler for non-first choices to avoid duplicate UI updates."""
    return None


_logger = logging.getLogger(__name__)


class OpenAIChatCompletionAggregator(Aggregator[ChatCompletionChunk, ChatCompletion]):
    """
    Aggregates a stream of chat completion chunks into a complete response.

    This class maintains state while aggregating chunks and provides reusability
    through the clear() method.
    """

    def __init__(self, *, on_event: Callable[[Event], None], run_id: str) -> None:
        self._on_event = on_event
        self._run_id = run_id
        self._choice_aggregators: dict[int, _ChoiceAggregator] = {}
        # Initialize _value with empty ChatCompletion (minus choices which are built later)
        self._value = ChatCompletion(
            id="",
            choices=[],  # Will be replaced in build()
            created=0,
            model="",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=None,
        )

    def aggregate(self, item: ChatCompletionChunk) -> None:
        """
        Aggregate a stream item.

        Args:
            item: A single chunk from the OpenAI chat completion stream
        """
        # Update metadata from the chunk
        self._aggregate_metadata(item)

        # Process each choice in the chunk
        for choice in item.choices:
            aggregator = self._choice_aggregators.setdefault(
                choice.index,
                _ChoiceAggregator(
                    index=choice.index,
                    message_id=self._value.id,
                    on_event=self._on_event,
                    run_id=self._run_id,
                ),
            )
            aggregator.aggregate(choice)

    def build(self) -> ChatCompletion:
        """
        Build the final chat completion object from aggregated chunks.

        Returns:
            The complete ChatCompletion object

        Raises:
            RuntimeError: If no valid chunks were received
        """
        if not self._choice_aggregators or not self._value.id:
            raise RuntimeError("Chat completion stream did not receive any valid chunks from OpenAI")

        # Build final choices
        ordered_indices = sorted(self._choice_aggregators)
        built_choices: list[ChatCompletionChoice] = []
        for idx in ordered_indices:
            built_choices.append(self._choice_aggregators[idx].build())

        # Update _value with final choices and return
        self._value.choices = built_choices
        return self._value.model_copy(deep=True)

    def clear(self) -> None:
        """
        Reset the aggregator state for reuse.

        This allows the aggregator to be reused for processing a new stream
        without creating a new instance.
        """
        self._choice_aggregators.clear()
        # Reset _value to initial state
        self._value = ChatCompletion(
            id="",
            choices=[],
            created=0,
            model="",
            object="chat.completion",
            service_tier=None,
            system_fingerprint=None,
            usage=None,
        )

    def _aggregate_metadata(self, item: ChatCompletionChunk) -> None:
        """Aggregate metadata from a chunk."""
        if item.id and not self._value.id:
            self._value.id = item.id

        if item.created:
            self._value.created = item.created

        if item.model:
            self._value.model = item.model

        if item.service_tier:
            self._value.service_tier = item.service_tier

        if item.system_fingerprint:
            self._value.system_fingerprint = item.system_fingerprint

        if item.usage:
            self._value.usage = item.usage


class _ChoiceAggregator(Aggregator[ChatCompletionChunkChoice, ChatCompletionChoice]):
    """
    Aggregates a single choice within a chat completion stream.

    Handles content, tool calls, and logging events for one choice.
    """

    def __init__(
        self,
        *,
        index: int,
        message_id: str,
        on_event: Callable[[Event], None],
        run_id: str,
    ) -> None:
        self._index = index
        self._message_id = message_id
        self._run_id = run_id
        # Only emit events for the first choice to avoid duplicate UI updates
        self._on_event = on_event if index == 0 else _noop_event_handler
        self._tool_call_aggregators: dict[int, _ToolCallAggregator] = {}
        # Initialize _value with empty ChatCompletionChoice
        self._value = ChatCompletionChoice(
            finish_reason="stop",
            index=index,
            message=ChatCompletionMessage(role="assistant", content=None, refusal=None, tool_calls=None),
            logprobs=None,
        )
        self._started = False

    def aggregate(self, item: ChatCompletionChunkChoice) -> None:
        """
        Aggregate a chunk for this choice.

        Args:
            item: Choice data from a stream chunk
        """
        delta: ChoiceDelta = item.delta

        # Emit message start event on first content (either text or tool calls)
        if not self._started:
            self._started = True
            self._on_event(
                TextMessageStartEvent(
                    message_id=self._message_id,
                    role="assistant",
                    timestamp=int(datetime.now().timestamp() * 1000),
                    run_id=self._run_id,
                )
            )

        # Aggregate content
        if delta.content:
            self._value.message.content = (self._value.message.content or "") + delta.content
            # Emit TextMessageContentEvent
            self._on_event(
                TextMessageContentEvent(
                    message_id=self._message_id,
                    delta=delta.content,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )

        # Aggregate refusal
        if delta.refusal:
            self._value.message.refusal = (self._value.message.refusal or "") + delta.refusal
            self._on_event(
                TextMessageContentEvent(
                    message_id=self._message_id,
                    delta=delta.refusal,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )

        # Aggregate tool calls
        if delta.tool_calls:
            for tool_delta in delta.tool_calls:
                aggregator = self._tool_call_aggregators.setdefault(
                    tool_delta.index,
                    _ToolCallAggregator(
                        on_event=self._on_event,
                        parent_message_id=self._message_id,
                    ),
                )
                aggregator.aggregate(tool_delta)

        # Update finish reason
        if item.finish_reason:
            self._value.finish_reason = item.finish_reason

        # Aggregate logprobs
        if item.logprobs:
            self._append_logprobs(item.logprobs)

        # Emit message end event when choice is complete
        if item.finish_reason is not None:
            # Ensure all tool calls have emitted their start+end events
            for aggregator in self._tool_call_aggregators.values():
                aggregator.ensure_ended()
            self._on_event(
                TextMessageEndEvent(
                    message_id=self._message_id,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )

    def build(self) -> ChatCompletionChoice:
        """
        Build the final choice object.

        Returns:
            The complete ChatCompletionChoice object

        Raises:
            RuntimeError: If aggregate() was never called
        """
        # Check if any content was aggregated (content, refusal, or tool calls)
        if self._value.message.content is None and self._value.message.refusal is None and not self._tool_call_aggregators:
            raise RuntimeError(f"Choice {self._index} was never aggregated with any content")

        # Build final tool calls
        built_tool_calls: list[ChatCompletionMessageToolCallUnion] = []
        if self._tool_call_aggregators:
            for idx in sorted(self._tool_call_aggregators):
                built_tool_calls.append(self._tool_call_aggregators[idx].build())
        # Set tool calls with type assertion to handle union type
        self._value.message.tool_calls = built_tool_calls

        # Return the complete choice
        return self._value.model_copy(deep=True)

    def clear(self) -> None:
        """Reset choice aggregator for reuse."""
        self._tool_call_aggregators.clear()
        # Reset _value to initial state
        self._value = ChatCompletionChoice(
            finish_reason="stop",
            index=self._index,
            message=ChatCompletionMessage(role="assistant", content=None, refusal=None, tool_calls=None),
            logprobs=None,
        )
        self._started = False

    def _append_logprobs(self, chunk_logprobs: ChatCompletionChunkChoiceLogprobs) -> None:
        """Append logprobs from a chunk to the aggregated logprobs."""
        if self._value.logprobs is None:
            self._value.logprobs = ChatCompletionChoiceLogprobs()

        if chunk_logprobs.content:
            if self._value.logprobs.content is None:
                self._value.logprobs.content = []
            self._value.logprobs.content.extend(chunk_logprobs.content)

        if chunk_logprobs.refusal:
            if self._value.logprobs.refusal is None:
                self._value.logprobs.refusal = []
            self._value.logprobs.refusal.extend(chunk_logprobs.refusal)


class _ToolCallAggregator(Aggregator[ChoiceDeltaToolCall, ChatCompletionMessageToolCall]):
    """
    Aggregates a single tool call within a choice.

    Handles tool call arguments and emits appropriate events.
    """

    def __init__(
        self,
        *,
        on_event: Callable[[Event], None],
        parent_message_id: str,
    ) -> None:
        self._on_event = on_event
        self._parent_message_id = parent_message_id
        self._value = ChatCompletionMessageToolCall(
            id="", type="function", function=ChatCompletionMessageToolCallFunction(name="", arguments="")
        )
        self._started = False
        self._ended = False
        self._json_state: Literal["init", "in_object", "complete"] = "init"

    def aggregate(self, item: ChoiceDeltaToolCall) -> None:
        """
        Aggregate a tool call item.

        Args:
            delta: Tool call data from a stream chunk
        """
        fn_args = None
        if item.function and item.function.arguments:
            raw = item.function.arguments
            fn_args = (raw[:80] + "...") if len(raw) > 80 else raw
        _logger.debug(
            "tool_call_chunk id=%s index=%s type=%s fn.name=%s fn.args=%s accumulated_id=%s accumulated_name=%s started=%s",
            item.id,
            item.index,
            item.type,
            item.function.name if item.function else None,
            fn_args,
            self._value.id,
            self._value.function.name,
            self._started,
        )

        # Accumulate tool call ID and type first
        if item.id:
            self._value.id = item.id

        if item.type:
            self._value.type = item.type

        # Aggregate function data
        if item.function:
            fn = item.function

            # Emit tool call start event once we have ID and name
            # Check both current chunk name and accumulated name for cases where
            # name arrives in a different chunk than id
            if not self._started and self._value.id and (fn.name or self._value.function.name):
                self._started = True
                effective_name = fn.name or self._value.function.name
                self._on_event(
                    ToolCallStartEvent(
                        tool_call_id=self._value.id,
                        tool_call_name=effective_name,
                        parent_message_id=self._parent_message_id,
                        timestamp=int(datetime.now().timestamp() * 1000),
                    )
                )

            if fn.name:
                self._value.function.name = fn.name

            if fn.arguments:
                self._value.function.arguments += fn.arguments
                # Emit ToolCallArgsEvent
                self._on_event(
                    ToolCallArgsEvent(
                        tool_call_id=self._value.id,
                        delta=fn.arguments,
                        timestamp=int(datetime.now().timestamp() * 1000),
                    )
                )
                # Update JSON state
                self._update_json_state()

        # Emit tool call end event once when JSON is complete AND start was emitted
        if self._json_state == "complete" and self._started and not self._ended:
            self._ended = True
            self._on_event(
                ToolCallEndEvent(
                    tool_call_id=self._value.id,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )

    def ensure_ended(self) -> None:
        """Ensure ToolCallEndEvent is emitted for this tool call.

        Called by the parent ChoiceAggregator when finish_reason is received,
        to guarantee the frontend always gets a paired end event.
        If ToolCallStartEvent was never emitted (e.g. name arrived late or
        never arrived at all for some model providers), emit it first using
        the accumulated name or 'unknown' as fallback.
        """
        if not self._started and self._value.id:
            self._started = True
            effective_name = self._value.function.name or "unknown"
            _logger.warning(
                "ensure_ended: emitting late TOOL_CALL_START id=%s name=%s accumulated_args_len=%d",
                self._value.id,
                effective_name,
                len(self._value.function.arguments),
            )
            self._on_event(
                ToolCallStartEvent(
                    tool_call_id=self._value.id,
                    tool_call_name=effective_name,
                    parent_message_id=self._parent_message_id,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )
        if self._started and not self._ended:
            self._ended = True
            self._on_event(
                ToolCallEndEvent(
                    tool_call_id=self._value.id,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )

    def _update_json_state(self) -> None:
        """Update JSON state by trying to parse the accumulated arguments.

        This method checks if the accumulated arguments form a valid JSON object.
        """

        try:
            if self._value.function.arguments.strip():
                json.loads(self._value.function.arguments)
                self._json_state = "complete"
            else:
                self._json_state = "init"
        except (json.JSONDecodeError, ValueError):
            self._json_state = "in_object"

    def build(self) -> ChatCompletionMessageToolCall:
        """
        Build the final tool call object.

        Returns:
            The complete ChatCompletionMessageToolCall object

        Raises:
            ValueError: If called without receiving valid tool call data
        """
        if not self._started or not self._value.id or not self._value.function.name:
            raise ValueError(
                f"Tool call aggregator never received valid tool call data (ID={self._value.id}, name={self._value.function.name})"
            )

        # Return complete tool call
        return self._value.model_copy(deep=True)

    def clear(self) -> None:
        """Reset tool call aggregator for reuse."""
        self._value = ChatCompletionMessageToolCall(
            id="", type="function", function=ChatCompletionMessageToolCallFunction(name="", arguments="")
        )
        self._started = False
        self._ended = False
        # Reset JSON tracking
        self._json_state = "init"
        # Note: _parent_message_id is intentionally not reset as it's a constant reference
