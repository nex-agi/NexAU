"""Anthropic raw stream event aggregator.

Converts raw RawMessageStreamEvent objects from client.messages.create(stream=True)
into unified Event objects for the agent events middleware pipeline.

⚠️ PARITY PROTOCOL: This module has a twin in
``nexau/archs/main_sub/execution/llm_caller.py`` (``AnthropicStreamAggregator``)
that parses the same wire format and MUST stay in lock-step until
RFC-0023 §阶段 ③ retires the twin. Any change to this module's parsing or
emission logic requires:

1. Run ``uv run pytest tests/aggregator_parity/`` before commit.
2. If your change handles a new wire pattern (new event / block / field
   type), record a fixture via
   ``tests/aggregator_parity/scripts/record_fixture.py``.
3. If parity surfaces a divergence, fix the buggy side rather than
   xfail — real Set A↔Set B drift = real production bug visible to
   end users (live SSE vs persisted history).

See ``tests/aggregator_parity/README.md`` for the full protocol.
This harness has caught 3+ production bugs that would otherwise have
shipped silently.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import cast

from anthropic.types import (
    ContentBlock,
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RawMessageStreamEvent,
    SignatureDelta,
    StopReason,
    TextDelta,
    ThinkingDelta,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import RedactedThinkingBlock as AnthropicRedactedThinkingBlock
from anthropic.types import ServerToolUseBlock as AnthropicServerToolUseBlock
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
)
from anthropic.types import ThinkingBlock as AnthropicThinkingBlock
from anthropic.types import ToolUseBlock as AnthropicToolUseBlock
from anthropic.types import (
    Usage as AnthropicUsage,
)

from ..events import (
    Aggregator,
    Event,
    ModelCallFinishedEvent,
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


class AnthropicEventAggregator(Aggregator[RawMessageStreamEvent, AnthropicMessage]):
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
        # Per-thinking-block metadata (RFC-0023 §阶段 ②) — captured during
        # the stream, attached to ThinkingTextMessageEndEvent at block close.
        self._thinking_signatures: dict[int, str] = {}
        self._thinking_redacted_data: dict[int, str] = {}
        self._thinking_is_redacted: dict[int, bool] = {}
        # Buffer for input_json_delta fragments that arrive before content_block_start.
        # key = content-block index, value = list of non-empty partial_json strings.
        self._pending_tool_deltas: dict[int, list[str]] = {}
        # Per-call metadata (RFC-0023 §阶段 ②) — captured across the stream,
        # emitted as a single ModelCallFinishedEvent at message_stop.
        self._model_name: str | None = None
        self._model_call_id: str | None = None
        # Keep SDK's narrow Literal type — assigned only from
        # ``RawMessageDeltaEvent.delta.stop_reason`` which IS this type, so
        # the constructor below needs no cast and stays type-safe.
        self._stop_reason: StopReason | None = None
        # Store usage as the SDK ``Usage`` object directly. ``message_start``
        # provides initial input/cache totals; ``message_delta`` ships the
        # cumulative output tokens at the end. We merge by mutating fields
        # in place (Pydantic v2 BaseModels are mutable), which keeps the
        # type strict as ``Usage`` instead of falling back to a dict.
        self._usage: AnthropicUsage | None = None
        # Per-block content accumulators for ``build() -> Message`` (RFC-0023
        # §阶段 ③). Each entry is a strict SDK block type — pydantic v2
        # BaseModels are mutable so we can update fields as deltas arrive
        # without intermediate ``dict[str, object]`` plumbing. Mirrors the
        # OpenAI Chat aggregator's pattern of storing ``ChatCompletionChoice``
        # directly and mutating it in place.
        #
        # Index reuse (e.g. thinking + tool both at idx 0 in
        # rec_single_tool_call) is preserved because each block is sealed at
        # its own content_block_stop and appended to the ordered list before
        # the next ``content_block_start`` at the same index opens a fresh
        # accumulator.
        self._active_payloads: dict[int, ContentBlock] = {}
        self._completed_payloads: list[ContentBlock] = []
        # Tool args still buffered as raw text (input_json_delta arrives as
        # text fragments; parsed at seal time). Kept separate from the
        # ToolUseBlock's typed ``input: dict`` so we don't lose unparsed
        # state mid-stream.
        self._tool_args_per_index: dict[int, str] = {}

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
                self._handle_message_delta(item)
            case RawMessageStopEvent():
                self._handle_message_stop()

    def build(self) -> AnthropicMessage:
        """Construct the final Anthropic Message from accumulated stream state.

        RFC-0023 §阶段 ③ — Set A becomes the canonical aggregator. The
        returned object is a strict ``anthropic.types.Message``; downstream
        code that wants a unified ``ModelResponse`` calls
        ``ModelResponse.from_anthropic_message(aggregator.build())``.

        Block ordering follows wire index (Anthropic streams blocks in
        order with explicit ``index`` fields). Blocks that were started but
        never received a ``content_block_stop`` are still emitted (mirrors
        Set B's ``_flush_active_blocks`` at finalize) so truncated streams
        produce a structurally-valid Message.
        """
        return self._construct_message()

    def _construct_message(self) -> AnthropicMessage:
        # ``_completed_payloads`` already holds typed SDK ContentBlocks in
        # wire order (sealed at content_block_stop); pass through verbatim.
        # Index reuse (e.g. thinking + tool_use both at idx 0 in
        # rec_single_tool_call) is preserved because each block was sealed
        # before the next start at the same index opened a fresh accumulator.
        content_blocks: list[ContentBlock] = list(self._completed_payloads)

        # Usage: ``_usage`` is already the SDK ``Usage`` object (mutated as
        # message_start + message_delta arrived). Default to an empty Usage
        # if the stream lacked both events (e.g. truncated transport).
        usage = self._usage if self._usage is not None else AnthropicUsage(input_tokens=0, output_tokens=0)

        return AnthropicMessage(
            id=self._message_id or self._model_call_id or "",
            type="message",
            role="assistant",
            content=content_blocks,
            model=self._model_name or "",
            stop_reason=self._stop_reason,
            stop_sequence=None,
            usage=usage,
        )

    def _parse_tool_input(self, idx: int) -> dict[str, object]:
        """Parse the buffered ``input_json_delta`` text for a tool block.

        Mirrors Set B's recovery path: try strict json.loads, then
        ``raw_decode`` to extract the first JSON object (eager_input_streaming
        can append junk), then return ``{"_raw": ...}`` as last resort.
        """
        buffer = self._tool_args.get(idx, "")
        if not buffer:
            return {}
        # Type ``parsed`` as ``object`` so mypy/pyright treat narrowing
        # strictly — ``json.loads`` would otherwise propagate ``Any`` and
        # poison downstream type inference.
        parsed: object
        try:
            parsed = json.loads(buffer)
        except json.JSONDecodeError:
            try:
                first_obj, _ = json.JSONDecoder().raw_decode(buffer.lstrip())
                parsed = first_obj
            except (json.JSONDecodeError, ValueError):
                return {"_raw": buffer}
        # JSON spec guarantees string keys, but isinstance(parsed, dict) only
        # narrows to dict[Unknown, Unknown]. Cast to widen-to-strict-shape so
        # the comprehension lands on the SDK's expected ``dict[str, object]``.
        # The cast is justified: the runtime check above proves parsed is dict;
        # the cast only refines the parametric types mypy/pyright can't infer.
        if isinstance(parsed, dict):
            typed_parsed = cast(dict[object, object], parsed)
            return {str(k): v for k, v in typed_parsed.items()}
        return {"_raw": buffer}

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
        self._thinking_signatures.clear()
        self._thinking_redacted_data.clear()
        self._thinking_is_redacted.clear()
        self._pending_tool_deltas.clear()
        self._model_name = None
        self._model_call_id = None
        self._stop_reason = None
        self._usage = None
        self._active_payloads.clear()
        self._completed_payloads.clear()
        self._tool_args_per_index.clear()

    # ---- Internal handlers ----

    def _ts(self) -> int:
        return int(datetime.now().timestamp() * 1000)

    def _handle_message_start(self, event: RawMessageStartEvent) -> None:
        message = cast(AnthropicMessage | None, event.message)
        if message is None:
            _logger.warning("Received Anthropic message_start with null message; using empty message metadata")
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
            return

        self._message_id = message.id
        # Capture per-call metadata for ModelCallFinishedEvent (RFC-0023 §阶段 ②)
        self._model_call_id = message.id
        self._model_name = message.model
        # ``event.message.usage`` is the SDK ``Usage`` (non-Optional). Copy
        # so subsequent message_delta merges don't mutate the source object.
        self._usage = message.usage.model_copy()
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

    def _handle_message_delta(self, event: RawMessageDeltaEvent) -> None:
        """RawMessageDeltaEvent carries the cumulative ``stop_reason`` and the
        final ``usage`` totals — both targets for ``ModelCallFinishedEvent``."""
        # ``event.delta`` is non-Optional in the Anthropic SDK type but its
        # fields can be empty; only stop_reason is what we want here.
        if event.delta.stop_reason:
            self._stop_reason = event.delta.stop_reason
        if event.usage:
            # ``event.usage`` (MessageDeltaUsage) ships the cumulative
            # output_tokens at end-of-stream; merge into our running Usage by
            # field. MessageDeltaUsage's fields overlap Usage but are a strict
            # subset, so we copy through any non-None values.
            if self._usage is None:
                # No message_start usage seen — synthesize a fresh Usage with
                # zero input tokens and the delta-supplied output tokens.
                self._usage = AnthropicUsage(input_tokens=0, output_tokens=event.usage.output_tokens or 0)
            else:
                # Field-level mutation. Only output_tokens / cache_*_tokens are
                # commonly updated by message_delta; copy any non-None field.
                for field in event.usage.model_fields_set:
                    new_value = getattr(event.usage, field, None)
                    if new_value is not None:
                        setattr(self._usage, field, new_value)

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

        # If this index is reused (e.g. thinking → tool_use both at idx 0 in
        # rec_single_tool_call), seal the previous in-flight payload first
        # so it survives in _completed_payloads. Without this we'd silently
        # overwrite the prior block.
        if idx in self._active_payloads:
            self._completed_payloads.append(self._active_payloads.pop(idx))

        match block:
            case AnthropicToolUseBlock():
                self._register_tool_and_flush(idx, block.id, block.name)
                # Pre-allocate the SDK ToolUseBlock; ``input`` filled at
                # content_block_stop after the JSON delta buffer is parsed.
                self._active_payloads[idx] = AnthropicToolUseBlock(
                    type="tool_use",
                    id=block.id,
                    name=block.name,
                    input={},
                )
            case AnthropicServerToolUseBlock():
                # 服务端工具（web_search、code_execution 等）使用相同的 id/name 接口
                self._register_tool_and_flush(idx, block.id, block.name)
                self._active_payloads[idx] = block.model_copy()
            case AnthropicThinkingBlock():
                thinking_id = str(uuid.uuid4())
                self._thinking_ids[idx] = thinking_id
                # Pre-allocate; ``thinking`` text appended via deltas, ``signature``
                # set when SignatureDelta arrives (or inline on this start).
                self._active_payloads[idx] = AnthropicThinkingBlock(
                    type="thinking",
                    thinking=block.thinking or "",
                    signature=block.signature or "",
                )
                if block.signature:
                    self._thinking_signatures[idx] = block.signature
                self._on_event(
                    ThinkingTextMessageStartEvent(
                        parent_message_id=self._message_id,
                        thinking_message_id=thinking_id,
                        run_id=self._run_id,
                        timestamp=self._ts(),
                    )
                )
            case AnthropicRedactedThinkingBlock():
                # Opaque encrypted-thinking block. Synthesize a thinking_id,
                # mark redacted, capture the data; consumers don't expect
                # ContentEvents — only Start (is_redacted=True) and End
                # (with redacted_data set).
                thinking_id = str(uuid.uuid4())
                self._thinking_ids[idx] = thinking_id
                self._thinking_is_redacted[idx] = True
                self._active_payloads[idx] = AnthropicRedactedThinkingBlock(
                    type="redacted_thinking",
                    data=block.data or "",
                )
                if block.data:
                    self._thinking_redacted_data[idx] = block.data
                # Re-tag for the stop handler — RedactedThinkingBlock.type is
                # "redacted_thinking" but our close path keys on "thinking".
                self._block_types[idx] = "thinking"
                self._on_event(
                    ThinkingTextMessageStartEvent(
                        parent_message_id=self._message_id,
                        thinking_message_id=thinking_id,
                        run_id=self._run_id,
                        timestamp=self._ts(),
                        is_redacted=True,
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
                # Mark this index as a text block (covers the case where the
                # corresponding content_block_start hasn't been processed yet).
                self._block_types.setdefault(idx, "text")
                # Retain content for build() (RFC-0023 §阶段 ③). Mutate the
                # SDK TextBlock in place — pre-allocated here if the start
                # event hasn't been processed.
                existing = self._active_payloads.get(idx)
                if isinstance(existing, AnthropicTextBlock):
                    existing.text += text
                else:
                    self._active_payloads[idx] = AnthropicTextBlock(type="text", text=text, citations=None)
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
                    # Eager streaming / wire pathology: thinking_delta arrived
                    # before any content_block_start. Mirror Set B's
                    # AnthropicStreamAggregator behavior — lazily synthesize a
                    # thinking block here so the delta isn't silently dropped.
                    # Same pattern as InputJSONDelta's _pending_tool_deltas
                    # buffering, but for thinking we don't need to buffer
                    # because we have no id/name to wait for.
                    thinking_id = str(uuid.uuid4())
                    self._thinking_ids[idx] = thinking_id
                    self._block_types[idx] = "thinking"
                    self._on_event(
                        ThinkingTextMessageStartEvent(
                            parent_message_id=self._message_id,
                            thinking_message_id=thinking_id,
                            run_id=self._run_id,
                            timestamp=self._ts(),
                        )
                    )
                # Retain content for build() — mutate the SDK ThinkingBlock
                # in place; pre-allocate if start event hasn't run yet.
                existing = self._active_payloads.get(idx)
                if isinstance(existing, AnthropicThinkingBlock):
                    existing.thinking += thinking
                else:
                    self._active_payloads[idx] = AnthropicThinkingBlock(
                        type="thinking",
                        thinking=thinking,
                        signature="",
                    )
                self._on_event(
                    ThinkingTextMessageContentEvent(
                        thinking_message_id=thinking_id,
                        delta=thinking,
                        timestamp=self._ts(),
                    )
                )
            case SignatureDelta(signature=sig):
                # Anthropic emits SignatureDelta near end-of-thinking carrying
                # the replay-auth signature for the block. Stash for End.
                if sig:
                    self._thinking_signatures[idx] = sig
                    existing = self._active_payloads.get(idx)
                    if isinstance(existing, AnthropicThinkingBlock):
                        existing.signature = sig
                    else:
                        # Lazy: ThinkingBlock pre-alloc happens in start handler;
                        # SignatureDelta arriving without a thinking payload is
                        # an unusual wire shape (covered by the lazy synthesis
                        # branch in the ThinkingDelta case above).
                        self._active_payloads[idx] = AnthropicThinkingBlock(
                            type="thinking",
                            thinking="",
                            signature=sig,
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
                    signature=self._thinking_signatures.get(idx),
                    redacted_data=self._thinking_redacted_data.get(idx),
                )
            )

        # Seal the in-flight payload (RFC-0023 §阶段 ③) — this finalizes
        # the block for build(). If the block is a tool, the input JSON
        # gets parsed here.
        self._seal_active_payload(idx)

    def _seal_active_payload(self, idx: int) -> None:
        payload = self._active_payloads.pop(idx, None)
        if payload is None:
            return
        # Parse buffered tool input now (mirrors Set B's _finalize_block).
        if isinstance(payload, AnthropicToolUseBlock | AnthropicServerToolUseBlock):
            payload.input = self._parse_tool_input(idx)
        self._completed_payloads.append(payload)

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
        # Flush any active payloads that didn't see content_block_stop —
        # truncated streams (max_tokens, network cut) leave blocks in-flight.
        for idx in list(self._active_payloads):
            self._seal_active_payload(idx)
        # Emit message end
        self._on_event(
            TextMessageEndEvent(
                message_id=self._message_id,
                timestamp=self._ts(),
            )
        )
        # RFC-0023 §阶段 ② — emit per-call metadata as the closing event so
        # consumers (parity tests, agent_events_middleware) get model_name /
        # stop_reason / model_call_id without peeking at Set B's ModelResponse.
        # Token usage is owned by ``UsageUpdateEvent`` (canonical TokenUsage).
        self._on_event(
            ModelCallFinishedEvent(
                run_id=self._run_id,
                message_id=self._message_id,
                model_name=self._model_name,
                model_call_id=self._model_call_id,
                stop_reason=self._stop_reason,
                timestamp=self._ts(),
            )
        )
