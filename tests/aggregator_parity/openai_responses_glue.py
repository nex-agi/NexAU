# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OpenAI Responses-specific glue for the parity harness.

Wires Set A (``OpenAIResponsesAggregator``) and Set B
(``OpenAIResponsesStreamAggregator``) into the harness.

Notable difference from the Anthropic case: Set A's OpenAI Responses
aggregator's ``build()`` returns a structured ``Response`` SDK object (not
None), so we have *two* finalize sources to choose from when shipping the
post-§阶段 ③ unified aggregator. The parity harness compares both to a
common UMP ``Message`` representation.

Set A consumes strict ``ResponseStreamEvent`` SDK types; Set B's ``consume()``
is dict-permissive. Loose dicts are normalized via
``dict_to_response_event`` before reaching Set A.
"""

from __future__ import annotations

from typing import Any

from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

from nexau.archs.llm.llm_aggregators import OpenAIResponsesAggregator
from nexau.archs.llm.llm_aggregators.events import Event

_RESPONSE_EVENT_ADAPTER: TypeAdapter[ResponseStreamEvent] = TypeAdapter(ResponseStreamEvent)


def dict_to_response_event(d: dict[str, Any]) -> ResponseStreamEvent:
    """Convert a loose Responses-shaped dict into a strict SDK event type.

    Fills in fields the OpenAI Responses SDK requires but Set B's permissive
    ``consume()`` is happy without (e.g. ``sequence_number``, ``content_index``).
    """
    normalized: dict[str, Any] = dict(d)
    normalized.setdefault("sequence_number", 0)

    ev_type = normalized.get("type")

    if ev_type == "response.output_item.added":
        normalized.setdefault("output_index", 0)
        item = dict(normalized.get("item") or {})
        if item.get("type") == "message":
            item.setdefault("status", "in_progress")
            item.setdefault("content", [])
            item.setdefault("role", "assistant")
        elif item.get("type") == "function_call":
            item.setdefault("arguments", "")
        elif item.get("type") == "reasoning":
            item.setdefault("summary", [])
        normalized["item"] = item

    elif ev_type == "response.content_part.added":
        normalized.setdefault("output_index", 0)
        part = dict(normalized.get("part") or {})
        if part.get("type") == "output_text":
            part.setdefault("annotations", [])
        normalized["part"] = part

    elif ev_type == "response.output_text.delta":
        normalized.setdefault("output_index", 0)
        normalized.setdefault("logprobs", [])

    elif ev_type == "response.function_call_arguments.delta":
        normalized.setdefault("output_index", 0)

    elif ev_type == "response.reasoning_summary_text.delta":
        normalized.setdefault("output_index", 0)

    elif ev_type == "response.reasoning_summary_text.done":
        normalized.setdefault("output_index", 0)

    elif ev_type == "response.reasoning_summary_part.added":
        normalized.setdefault("output_index", 0)

    elif ev_type == "response.function_call_arguments.done":
        normalized.setdefault("output_index", 0)
        # SDK requires `name` (recovered from prior output_item.added). The
        # stateful coercer would normally fill this in, but to keep
        # ``dict_to_response_event`` callable standalone we accept any sentinel.
        normalized.setdefault("name", "")

    elif ev_type == "response.completed":
        resp = dict(normalized.get("response") or {})
        resp.setdefault("object", "response")
        resp.setdefault("output", [])
        resp.setdefault("parallel_tool_calls", True)
        resp.setdefault("tool_choice", "auto")
        resp.setdefault("tools", [])
        resp.setdefault("created_at", 0)
        # Normalize usage block: SDK requires input_tokens_details, output_tokens_details, total_tokens
        usage = resp.get("usage")
        if isinstance(usage, dict):
            usage = dict(usage)
            usage.setdefault("input_tokens_details", {"cached_tokens": 0})
            usage.setdefault("output_tokens_details", {"reasoning_tokens": 0})
            usage.setdefault(
                "total_tokens",
                int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
            )
            resp["usage"] = usage
        normalized["response"] = resp

    return _RESPONSE_EVENT_ADAPTER.validate_python(normalized)


def _coerce_to_sdk_events(
    events: list[ResponseStreamEvent | dict[str, Any]],
) -> list[ResponseStreamEvent]:
    """Stateful normalization: assign output_index per item_id in order.

    Set A's OpenAI Responses aggregator enforces sequential output_index on
    ``response.output_item.added`` events. Loose dict fixtures from
    ``test_llm_streaming.py`` omit the field. We assign 0, 1, 2, ... in the
    order ``output_item.added`` events appear, and propagate the same value
    to subsequent events that reference the same ``item_id``.
    """
    item_id_to_index: dict[str, int] = {}
    next_index = 0
    coerced: list[ResponseStreamEvent] = []

    for ev in events:
        if not isinstance(ev, dict):
            coerced.append(ev)
            continue

        ev_dict = dict(ev)
        ev_type = ev_dict.get("type")

        if ev_type == "response.output_item.added":
            item = ev_dict.get("item") or {}
            item_id = item.get("id")
            if item_id and "output_index" not in ev_dict:
                if item_id not in item_id_to_index:
                    item_id_to_index[item_id] = next_index
                    next_index += 1
                ev_dict["output_index"] = item_id_to_index[item_id]
        elif ev_type in {
            "response.content_part.added",
            "response.output_text.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.added",
            "response.output_item.done",
        }:
            item_id = ev_dict.get("item_id")
            if item_id and "output_index" not in ev_dict and item_id in item_id_to_index:
                ev_dict["output_index"] = item_id_to_index[item_id]

        coerced.append(dict_to_response_event(ev_dict))

    return coerced


def run_set_a_openai_responses(events: list[Any]) -> tuple[list[Event], Any]:
    """Feed events into Set A and return (collected events, built Response).

    Unlike the Anthropic case, Set A here also produces a structured
    ``Response`` via ``build()``. We return both: the event stream (for
    parity vs Set B's reconstructed Message) and the built Response (for
    direct comparison once §阶段 ② lands).
    """
    sdk_events = _coerce_to_sdk_events(events)
    collected: list[Event] = []
    aggregator = OpenAIResponsesAggregator(
        on_event=collected.append,
        run_id="parity-test-run",
    )
    for ev in sdk_events:
        aggregator.aggregate(ev)
    try:
        built = aggregator.build()
    except RuntimeError:
        # Set A's build() requires a response.created event — synthetic
        # fixtures may skip it. Fall back to None; comparison goes via events.
        built = None
    return collected, built


def run_set_a_openai_responses_events_only(events: list[Any]) -> list[Event]:
    """Adapter for the run_parity() harness which expects events list only."""
    collected, _built = run_set_a_openai_responses(events)
    return collected
