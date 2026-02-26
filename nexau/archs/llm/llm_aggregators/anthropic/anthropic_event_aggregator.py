"""Anthropic raw stream event aggregator.

Converts raw RawMessageStreamEvent objects from client.messages.create(stream=True)
into unified Event objects for the agent events middleware pipeline.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import datetime

from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RawMessageStreamEvent,
    TextDelta,
    ThinkingDelta,
)
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock

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

_logger = logging.getLogger(__name__)


class AnthropicEventAggregator(Aggregator[RawMessageStreamEvent, None]):
    """Aggregate raw Anthropic stream events and emit unified Event objects.

    Handles text, thinking, and tool_use content blocks from the Anthropic
    Messages API streaming response.
    """

    def __init__(self, *, on_event: Callable[[Event], None], run_id: str) -> None:
        self._on_event = on_event
        self._run_id = run_id
        self._message_id: str = ""
        self._started = False
        # Track active content blocks by index
        self._block_types: dict[int, str] = {}
        # Tool call state per index
        self._tool_ids: dict[int, str] = {}
        self._tool_names: dict[int, str] = {}
        self._tool_args: dict[int, str] = {}
        self._tool_started: dict[int, bool] = {}
        self._tool_ended: dict[int, bool] = {}
        # Thinking state per index
        self._thinking_ids: dict[int, str] = {}

    def aggregate(self, item: RawMessageStreamEvent) -> None:
        """Process a single raw stream event."""
        match item:
            case RawMessageStartEvent():
                self._handle_message_start(item)
            case RawContentBlockStartEvent():
                self._handle_content_block_start(item)
            case RawContentBlockDeltaEvent():
                self._handle_content_block_delta(item)
            case RawContentBlockStopEvent():
                self._handle_content_block_stop(item)
            case RawMessageDeltaEvent():
                pass  # Usage updates, not needed for frontend events
            case RawMessageStopEvent():
                self._handle_message_stop()

    def build(self) -> None:
        """Not used — events are emitted via on_event callback."""
        return None

    def clear(self) -> None:
        """Reset aggregator state for reuse."""
        self._message_id = ""
        self._started = False
        self._block_types.clear()
        self._tool_ids.clear()
        self._tool_names.clear()
        self._tool_args.clear()
        self._tool_started.clear()
        self._tool_ended.clear()
        self._thinking_ids.clear()

    # ---- Internal handlers ----

    def _ts(self) -> int:
        return int(datetime.now().timestamp() * 1000)

    def _handle_message_start(self, event: RawMessageStartEvent) -> None:
        self._message_id = event.message.id
        if not self._started:
            self._started = True
            self._on_event(
                TextMessageStartEvent(
                    message_id=self._message_id,
                    role="assistant",
                    timestamp=self._ts(),
                    run_id=self._run_id,
                )
            )

    def _handle_content_block_start(self, event: RawContentBlockStartEvent) -> None:
        idx = event.index
        block = event.content_block
        self._block_types[idx] = block.type

        match block:
            case AnthropicToolUseBlock():
                self._tool_ids[idx] = block.id
                self._tool_names[idx] = block.name
                self._tool_args[idx] = ""
                self._tool_started[idx] = True
                self._tool_ended[idx] = False
                self._on_event(
                    ToolCallStartEvent(
                        tool_call_id=block.id,
                        tool_call_name=block.name,
                        parent_message_id=self._message_id,
                        timestamp=self._ts(),
                    )
                )
            case AnthropicThinkingBlock():
                thinking_id = str(uuid.uuid4())
                self._thinking_ids[idx] = thinking_id
                self._on_event(
                    ThinkingTextMessageStartEvent(
                        parent_message_id=self._message_id,
                        thinking_message_id=thinking_id,
                        run_id=self._run_id,
                        timestamp=self._ts(),
                    )
                )
            case _:
                pass

    def _handle_content_block_delta(self, event: RawContentBlockDeltaEvent) -> None:
        idx = event.index

        match event.delta:
            case TextDelta(text=text):
                self._on_event(
                    TextMessageContentEvent(
                        message_id=self._message_id,
                        delta=text,
                        timestamp=self._ts(),
                    )
                )
            case InputJSONDelta(partial_json=fragment):
                tool_id = self._tool_ids.get(idx, "")
                if not tool_id:
                    _logger.warning("Received input_json_delta for unknown tool at index %d", idx)
                self._tool_args[idx] = self._tool_args.get(idx, "") + fragment
                self._on_event(
                    ToolCallArgsEvent(
                        tool_call_id=tool_id,
                        delta=fragment,
                        timestamp=self._ts(),
                    )
                )
            case ThinkingDelta(thinking=thinking):
                thinking_id = self._thinking_ids.get(idx)
                if not thinking_id:
                    _logger.warning("Received thinking_delta for unknown thinking block at index %d", idx)
                    return
                self._on_event(
                    ThinkingTextMessageContentEvent(
                        thinking_message_id=thinking_id,
                        delta=thinking,
                        timestamp=self._ts(),
                    )
                )
            case _:
                pass

    def _handle_content_block_stop(self, event: RawContentBlockStopEvent) -> None:
        idx = event.index
        block_type = self._block_types.get(idx)

        if block_type == "tool_use":
            tool_id = self._tool_ids.get(idx)
            if not tool_id:
                _logger.warning("Received content_block_stop for unknown tool at index %d", idx)
                return
            if not self._tool_ended.get(idx, False):
                self._tool_ended[idx] = True
                self._on_event(
                    ToolCallEndEvent(
                        tool_call_id=tool_id,
                        timestamp=self._ts(),
                    )
                )
        elif block_type == "thinking":
            thinking_id = self._thinking_ids.get(idx)
            if not thinking_id:
                _logger.warning("Received content_block_stop for unknown thinking block at index %d", idx)
                return
            self._on_event(
                ThinkingTextMessageEndEvent(
                    thinking_message_id=thinking_id,
                    timestamp=self._ts(),
                )
            )

    def _handle_message_stop(self) -> None:
        # Ensure all tool calls are ended
        for idx, started in self._tool_started.items():
            if started and not self._tool_ended.get(idx, False):
                tool_id = self._tool_ids.get(idx)
                if not tool_id:
                    _logger.warning("Received message_stop for unknown tool at index %d", idx)
                    continue
                self._tool_ended[idx] = True
                self._on_event(
                    ToolCallEndEvent(
                        tool_call_id=tool_id,
                        timestamp=self._ts(),
                    )
                )
        # Emit message end
        self._on_event(
            TextMessageEndEvent(
                message_id=self._message_id,
                timestamp=self._ts(),
            )
        )
