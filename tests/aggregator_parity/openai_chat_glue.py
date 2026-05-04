# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""OpenAI Chat Completions-specific glue for the parity harness.

Wires Set A (``OpenAIChatCompletionAggregator``) and Set B
(``OpenAIChatStreamAggregator``).

Important context: the OpenAI Chat Completions wire format has TWO common
shapes in production:

1. **Standard OpenAI**: ``delta.content: str`` — what the strict
   ``ChatCompletionChunk`` SDK type accepts and what Set A is built for.
2. **OpenRouter / proxy extension**: ``delta.content: list[{type, text}]`` —
   list-of-typed-parts borrowed from Anthropic / OpenAI Responses style.
   Set B's permissive ``consume()`` accepts this; Set A's strict typing
   rejects it.

The recordings under ``fixtures/openai_chat/recordings/`` come from the
gateway northgate.xiaobei.top which uses standard format → both Sets work.
The dict fixtures lifted from ``test_llm_streaming.py`` use the OpenRouter
extension and remain Set-B-only edge cases (not parity-testable).
"""

from __future__ import annotations

from typing import Any

from openai.types.chat import ChatCompletionChunk
from pydantic import TypeAdapter

from nexau.archs.llm.llm_aggregators import OpenAIChatCompletionAggregator
from nexau.archs.llm.llm_aggregators.events import Event

_CHUNK_ADAPTER: TypeAdapter[ChatCompletionChunk] = TypeAdapter(ChatCompletionChunk)


def dict_to_chat_chunk(d: dict[str, Any]) -> ChatCompletionChunk:
    """Convert a chunk dict into the strict ``ChatCompletionChunk`` SDK type.

    The recordings are already in the canonical wire format, so usually no
    fill-in is needed. We only normalize known-permissive fields.
    """
    normalized: dict[str, Any] = dict(d)
    normalized.setdefault("id", "")
    normalized.setdefault("object", "chat.completion.chunk")
    normalized.setdefault("created", 0)
    normalized.setdefault("model", "")
    normalized.setdefault("choices", [])
    return _CHUNK_ADAPTER.validate_python(normalized)


def _coerce_to_sdk_chunks(
    events: list[ChatCompletionChunk | dict[str, Any]],
) -> list[ChatCompletionChunk]:
    return [ev if not isinstance(ev, dict) else dict_to_chat_chunk(ev) for ev in events]


def run_set_a_openai_chat(events: list[Any]) -> list[Event]:
    """Feed events into Set A's OpenAIChatCompletionAggregator and collect events.

    Set A's OAC aggregator emits the §阶段 ② ``ModelCallFinishedEvent`` from
    ``build()``, not from ``aggregate()``. Call build() at the end to
    trigger it; ignore the RuntimeError it raises when no choices were
    received (synthetic fixtures sometimes carry only metadata-shape
    chunks; the metadata event still fires before the validation raises).
    """
    sdk_chunks = _coerce_to_sdk_chunks(events)
    collected: list[Event] = []
    aggregator = OpenAIChatCompletionAggregator(
        on_event=collected.append,
        run_id="parity-test-run",
    )
    for chunk in sdk_chunks:
        aggregator.aggregate(chunk)
    try:
        aggregator.build()
    except RuntimeError:
        pass
    return collected
