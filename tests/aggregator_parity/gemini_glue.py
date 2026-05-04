# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Gemini REST-specific glue for the parity harness.

Wires Set A (``GeminiRestEventAggregator``) and Set B
(``GeminiRestStreamAggregator``).

Both Sets consume plain dicts (the parsed JSON from each ``data:`` line in
the SSE stream); no Pydantic SDK normalization is needed. This is in
contrast to the Anthropic / OpenAI Chat / OpenAI Responses cases where Set
A expects strict SDK types.

Gemini wire format quirks captured by these recordings:

- ``parts`` array can contain heterogeneous part types in a single chunk:
  ``{text: str, thought: bool=False}`` for content text,
  ``{text: str, thought: True}`` for reasoning,
  ``{thoughtSignature: str}`` for Gemini's reasoning signature
  (similar to Anthropic's signature),
  ``{functionCall: {name, args}}`` for tool calls,
  ``{inlineData: {mimeType, data}}`` for image inputs (input-only).
- ``usageMetadata.thoughtsTokenCount`` is Gemini's reasoning-token equivalent.
- ``responseId`` is the per-call provider id (analogous to
  ``ChatCompletion.id`` / Anthropic's ``message.id``).
- ``finishReason: "STOP"`` marks completion in the final chunk.
"""

from __future__ import annotations

from typing import Any, cast

from nexau.archs.llm.llm_aggregators import GeminiRestEventAggregator
from nexau.archs.llm.llm_aggregators.events import Event


def run_set_a_gemini(events: list[dict[str, Any]]) -> list[Event]:
    """Feed chunks into Set A's GeminiRestEventAggregator and collect events."""
    from nexau.archs.llm.llm_aggregators.gemini_rest.gemini_rest_event_aggregator import GeminiResponse

    collected: list[Event] = []
    aggregator = GeminiRestEventAggregator(
        on_event=collected.append,
        run_id="parity-test-run",
    )
    for chunk in events:
        # Wire dicts conform to GeminiResponse shape; cast at the boundary.
        aggregator.aggregate(cast(GeminiResponse, chunk))
    return collected
