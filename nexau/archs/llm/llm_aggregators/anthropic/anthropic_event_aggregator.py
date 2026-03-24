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
from anthropic.types import ServerToolUseBlock as AnthropicServerToolUseBlock
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

    Handles text, thinking, tool_use, and server_tool_use content blocks
    from the Anthropic Messages API streaming response.

    With ``eager_input_streaming`` enabled, ``input_json_delta`` events may
    arrive *before* the corresponding ``content_block_start``.  To guarantee
    a consistent ``tool_call_id`` throughout the event stream we **buffer**
    early fragments and only flush them once ``content_block_start`` provides
    the real id/name.  If ``content_block_start`` never arrives (stop event
    comes first), we fall back to a synthetic id so the stream still closes
    cleanly.
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
        # Buffer for input_json_delta fragments that arrive before content_block_start.
        # key = content-block index, value = list of non-empty partial_json strings.
        self._pending_tool_deltas: dict[int, list[str]] = {}

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
        self._pending_tool_deltas.clear()

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

    # ---- Tool registration helpers ----

    def _register_tool_and_flush(self, idx: int, tool_id: str, tool_name: str) -> None:
        """Register a tool block, emit ToolCallStartEvent, then flush any buffered deltas.

        Called from ``_handle_content_block_start`` (normal path) and from
        ``_flush_pending_with_synthetic`` (fallback when start never arrived).
        """
        self._tool_ids[idx] = tool_id
        self._tool_names[idx] = tool_name
        self._tool_args.setdefault(idx, "")
        self._tool_started[idx] = True
        self._tool_ended.setdefault(idx, False)
        self._on_event(
            ToolCallStartEvent(
                tool_call_id=tool_id,
                tool_call_name=tool_name,
                parent_message_id=self._message_id,
                timestamp=self._ts(),
            )
        )
        # Flush any buffered fragments that arrived before this start event
        buffered = self._pending_tool_deltas.pop(idx, None)
        if buffered:
            for fragment in buffered:
                self._tool_args[idx] = self._tool_args.get(idx, "") + fragment
                self._on_event(
                    ToolCallArgsEvent(
                        tool_call_id=tool_id,
                        delta=fragment,
                        timestamp=self._ts(),
                    )
                )

    def _flush_pending_with_synthetic(self, idx: int) -> str:
        """Flush buffered deltas for *idx* using a synthetic tool id.

        Returns the synthetic tool_call_id so callers can emit
        ``ToolCallEndEvent`` with the same id.
        """
        synthetic_id = f"toolu_late_{uuid.uuid4().hex[:12]}"
        _logger.debug(
            "content_block_start never arrived for index %d; flushing buffered deltas with synthetic tool %s",
            idx,
            synthetic_id,
        )
        self._register_tool_and_flush(idx, synthetic_id, "")
        return synthetic_id

    # ---- Event dispatch ----

    def _handle_content_block_start(self, event: RawContentBlockStartEvent) -> None:
        idx = event.index
        block = event.content_block
        self._block_types[idx] = block.type

        match block:
            case AnthropicToolUseBlock():
                self._register_tool_and_flush(idx, block.id, block.name)
            case AnthropicServerToolUseBlock():
                # 服务端工具（web_search、code_execution 等）使用相同的 id/name 接口
                self._register_tool_and_flush(idx, block.id, block.name)
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
                if not text:
                    return
                self._on_event(
                    TextMessageContentEvent(
                        message_id=self._message_id,
                        delta=text,
                        timestamp=self._ts(),
                    )
                )
            case InputJSONDelta(partial_json=fragment):
                if not fragment:
                    return
                tool_id = self._tool_ids.get(idx, "")
                if tool_id:
                    # 正常路径：content_block_start 已到达，直接发射事件
                    self._tool_args[idx] = self._tool_args.get(idx, "") + fragment
                    self._on_event(
                        ToolCallArgsEvent(
                            tool_call_id=tool_id,
                            delta=fragment,
                            timestamp=self._ts(),
                        )
                    )
                else:
                    # eager_input_streaming 下 delta 先于 content_block_start 到达，
                    # 仅缓冲，等 start 带着真实 ID 到达后统一 flush。
                    self._pending_tool_deltas.setdefault(idx, []).append(fragment)
            case ThinkingDelta(thinking=thinking):
                if not thinking:
                    return
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

        # 1. 有缓冲但 content_block_start 始终未到达 → 合成 ID 兜底后再关闭
        if idx in self._pending_tool_deltas:
            self._flush_pending_with_synthetic(idx)
            # tool 已注册，直接走下面的关闭分支

        if block_type in {"tool_use", "server_tool_use"} or (block_type is None and idx in self._tool_ids):
            # tool_use / server_tool_use 共用同一收尾逻辑；
            # block_type 为 None 说明 content_block_start 未到达但 delta 已注册了工具。
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
        # Flush any remaining buffered deltas that never received content_block_start
        for idx in list(self._pending_tool_deltas):
            self._flush_pending_with_synthetic(idx)

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
