# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Anthropic-specific glue for the parity harness.

Wires Set A (``AnthropicEventAggregator``) into the provider-agnostic
harness in ``parity_helpers``. RFC-0023 §阶段 ③ retired Set B; only the
SDK-typed event path remains.
"""

from __future__ import annotations

from typing import Any

from anthropic.types import RawMessageStopEvent, RawMessageStreamEvent
from pydantic import TypeAdapter

from nexau.archs.llm.llm_aggregators import AnthropicEventAggregator
from nexau.archs.llm.llm_aggregators.events import Event

_ANTHROPIC_EVENT_ADAPTER: TypeAdapter[RawMessageStreamEvent] = TypeAdapter(RawMessageStreamEvent)


def dict_to_anthropic_event(d: dict[str, Any]) -> RawMessageStreamEvent:
    """Convert a loose Anthropic-shaped dict into a strict SDK event type.

    Fills in fields the Anthropic SDK requires but ``test_llm_streaming.py``
    fixtures often omit. Only fields that don't affect aggregator semantics
    are defaulted.
    """
    normalized: dict[str, Any] = dict(d)

    if normalized.get("type") == "message_start":
        msg = dict(normalized.get("message", {}))
        msg.setdefault("id", "msg_synth")
        msg.setdefault("type", "message")
        msg.setdefault("role", "assistant")
        msg.setdefault("content", [])
        msg.setdefault("model", "claude-3")
        msg.setdefault("stop_reason", None)
        msg.setdefault("stop_sequence", None)
        msg.setdefault("usage", {"input_tokens": 0, "output_tokens": 0})
        normalized["message"] = msg

    elif normalized.get("type") == "content_block_start":
        cb = dict(normalized.get("content_block") or {})
        cb_type = cb.get("type")
        if cb_type == "thinking":
            cb.setdefault("thinking", "")
            cb.setdefault("signature", "")
            # Anthropic SDK requires str, not None
            if cb.get("thinking") is None:
                cb["thinking"] = ""
        elif cb_type == "text":
            if cb.get("text") is None:
                cb["text"] = ""
        elif cb_type == "tool_use":
            cb.setdefault("input", {})
        normalized["content_block"] = cb

    return _ANTHROPIC_EVENT_ADAPTER.validate_python(normalized)


def _coerce_to_sdk_events(
    events: list[RawMessageStreamEvent | dict[str, Any]],
) -> list[RawMessageStreamEvent]:
    """Stateful pass-through.

    Production observation (recorded from northgate.xiaobei.top): a real
    Anthropic-compatible gateway can emit DUPLICATE ``content_block_start``
    events for the same index when generating tool_use blocks, with the
    duplicates carrying empty ``id``/``name``. Set B handles this by
    preserving the prior id/name; Set A's strict Pydantic typing on
    ``ToolUseBlock`` rejects ``id``/``name``-less starts entirely.

    To keep parity testable, we replay Set B's "preserve prior id/name"
    logic when normalizing duplicate starts: fill the missing fields from
    the most recent block at that index. Both Sets then see equivalent
    logical events (Set A gets the SDK-strict version with id/name carried
    forward; Set B gets the original loose dict and applies its own merge).

    Where a duplicate adds NO new information (empty input, empty fields),
    it's dropped entirely — this matches Set B's net effect of "prior state
    wins".

    The original divergence remains documented: in the wild, gateway-emitted
    duplicates would never reach Set A in production via the SDK (which would
    raise on parsing). This normalizer is purely a parity-test affordance.
    """
    coerced: list[RawMessageStreamEvent] = []
    last_block_at_index: dict[int, dict[str, Any]] = {}

    for ev in events:
        if not isinstance(ev, dict):
            coerced.append(ev)
            continue

        ev_dict = dict(ev)

        if ev_dict.get("type") == "content_block_start":
            idx = ev_dict.get("index")
            cb = dict(ev_dict.get("content_block") or {})

            # Detect duplicate: index already seen with same type
            prior = last_block_at_index.get(idx) if isinstance(idx, int) else None
            if prior is not None and prior.get("type") == cb.get("type"):
                # Fill in any missing/empty fields from prior block
                merged = dict(prior)
                for key, value in cb.items():
                    if value not in (None, "", {}, []):
                        merged[key] = value
                cb = merged
                # If nothing meaningful changed, skip emitting this duplicate
                if cb == prior:
                    continue

            ev_dict["content_block"] = cb
            if isinstance(idx, int):
                last_block_at_index[idx] = cb

        coerced.append(dict_to_anthropic_event(ev_dict))

    return coerced


def run_set_a_anthropic(events: list[Any]) -> list[Event]:
    """Feed events into Set A's AnthropicEventAggregator and collect emitted events.

    Lifted dict fixtures sometimes don't include a trailing ``RawMessageStopEvent``
    because the original test only cared about content blocks. Without it,
    the aggregator's end-of-call hooks (TextMessageEndEvent, the §阶段 ②
    ModelCallFinishedEvent) never fire. Detect that case and invoke the stop
    handler manually so parity tests see the same finalized event stream
    they would on a real wire.
    """
    sdk_events = _coerce_to_sdk_events(events)
    collected: list[Event] = []
    aggregator = AnthropicEventAggregator(
        on_event=collected.append,
        run_id="parity-test-run",
    )
    saw_stop = False
    for ev in sdk_events:
        if isinstance(ev, RawMessageStopEvent):
            saw_stop = True
        aggregator.aggregate(ev)
    if not saw_stop:
        aggregator._handle_message_stop()  # noqa: SLF001 — synthetic flush for fixtures lacking stop event
    return collected
