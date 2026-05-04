# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fixture-replay regression tests for Set A aggregators.

RFC-0023 §阶段 ③.2 retired the Set B unit suite; the deleted files
(``test_llm_streaming.py``, ``test_long_thinking_fixes.py``,
``test_gemini_rest.py``) carried specific edge-case assertions on the
aggregator → ``build()`` shape. Vendor-truth axis-3 covers happy-path
structural equivalence, but the per-field regressions a future PR
might introduce (e.g. PR #475's ``reasoning_details`` verbatim bug,
PR #395's eager_input_streaming concatenation) need explicit pinning.

This file replays the **already-recorded** SSE fixtures under
``tests/aggregator_parity/fixtures/<provider>/recordings/`` through
Set A's aggregator and asserts on specific ``Message`` / ``ChatCompletion``
/ ``Response`` / ``GeminiResponse`` fields. Pure-replay, deterministic,
~10ms each, no live LLM call.

For edge cases the upstream wouldn't naturally emit (truncated stream,
malformed JSON, duplicate block_start), see ``test_llm_caller_helpers.py``
which uses synthetic SDK objects.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    RawMessageStopEvent,
)
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
)
from anthropic.types import (
    ThinkingBlock as AnthropicThinkingBlock,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import Response as OpenAIResponse
from openai.types.responses import ResponseStreamEvent
from pydantic import TypeAdapter

from nexau.archs.llm.llm_aggregators import (
    AnthropicEventAggregator,
    GeminiRestEventAggregator,
    OpenAIChatCompletionAggregator,
    OpenAIResponsesAggregator,
)
from nexau.archs.llm.llm_aggregators.events import Event
from nexau.archs.llm.llm_aggregators.gemini_rest.gemini_rest_event_aggregator import GeminiResponse
from tests.aggregator_parity.anthropic_glue import _coerce_to_sdk_events as _coerce_anthropic
from tests.aggregator_parity.openai_chat_glue import _coerce_to_sdk_chunks as _coerce_chat
from tests.aggregator_parity.openai_responses_glue import _coerce_to_sdk_events as _coerce_responses
from tests.aggregator_parity.sse_loader import load_recording

_RUN_ID = "replay-test"


# ── Per-provider replay drivers ─────────────────────────────────────


def _replay_anthropic(scenario: str) -> tuple[AnthropicMessage, list[Event]]:
    # ``load_recording`` returns ``list[dict[str, Any]]``; the glue
    # coercer's signature widens that to a Union with the SDK types
    # for cases where SDK objects are passed through unchanged. List
    # invariance means we have to cast the input shape rather than
    # rely on covariance.
    raw_events = cast(list[Any], load_recording("anthropic", scenario))
    sdk_events = _coerce_anthropic(raw_events)
    collected: list[Event] = []
    agg = AnthropicEventAggregator(on_event=collected.append, run_id=_RUN_ID)
    saw_stop = False
    for ev in sdk_events:
        if isinstance(ev, RawMessageStopEvent):
            saw_stop = True
        agg.aggregate(ev)
    if not saw_stop:
        agg._handle_message_stop()  # noqa: SLF001 — synthetic flush mirrors anthropic_glue
    return agg.build(), collected


def _replay_openai_chat(scenario: str):
    raw_chunks = cast(list[Any], load_recording("openai_chat", scenario))
    sdk_chunks = _coerce_chat(raw_chunks)
    collected: list[Event] = []
    agg = OpenAIChatCompletionAggregator(on_event=collected.append, run_id=_RUN_ID)
    for chunk in sdk_chunks:
        agg.aggregate(chunk)
    return agg.build(), collected


def _replay_openai_responses(scenario: str) -> tuple[OpenAIResponse, list[Event]]:
    raw_events = cast(list[Any], load_recording("openai_responses", scenario))
    sdk_events = _coerce_responses(raw_events)
    collected: list[Event] = []
    agg = OpenAIResponsesAggregator(on_event=collected.append, run_id=_RUN_ID)
    for ev in sdk_events:
        agg.aggregate(ev)
    return agg.build(), collected


def _replay_gemini(scenario: str) -> tuple[GeminiResponse, list[Event]]:
    raw_chunks = load_recording("gemini_rest", scenario)
    collected: list[Event] = []
    agg = GeminiRestEventAggregator(on_event=collected.append, run_id=_RUN_ID)
    for chunk in raw_chunks:
        agg.aggregate(cast(GeminiResponse, chunk))
    return agg.build(), collected


def _flatten_text_blocks(blocks: Iterable[Any]) -> str:
    parts: list[str] = []
    for b in blocks:
        if isinstance(b, AnthropicTextBlock):
            parts.append(b.text)
    return "".join(parts)


# ── Anthropic ───────────────────────────────────────────────────────
#
# Targets the deleted ``test_long_thinking_fixes.py`` +
# ``test_anthropic_stream_aggregator_*`` cases.


class TestAnthropicReplay:
    def test_plain_text_builds_text_block(self):
        msg, _ = _replay_anthropic("claude_haiku_plain")
        assert isinstance(msg, AnthropicMessage)
        assert msg.role == "assistant"
        text_blocks = [b for b in msg.content if isinstance(b, AnthropicTextBlock)]
        assert text_blocks, "expected at least one text block"
        assert text_blocks[0].text, "text block should be non-empty"
        assert msg.usage.input_tokens > 0
        assert msg.usage.output_tokens > 0

    def test_stop_reason_propagates_from_message_delta(self):
        """Pre-PR-C.2 Set B test: ``stop_reason`` from message_delta must
        land on the final Message — historically the parser dropped it
        when no message_delta arrived. Plain replay confirms it survives
        the round-trip."""
        msg, _ = _replay_anthropic("claude_opus46_plain")
        assert msg.stop_reason in {"end_turn", "max_tokens", "stop_sequence", "tool_use"}, f"unexpected stop_reason: {msg.stop_reason!r}"

    def test_thinking_block_with_signature_preserved(self):
        """Anthropic thinking block carries a ``signature`` that lets the
        provider verify the chain-of-thought when the next turn re-sends
        it. PR-0395 era bug: signature got dropped during finalize.
        Replay asserts it survives end-to-end.
        """
        msg, _ = _replay_anthropic("claude_thinking_real")
        thinking_blocks = [b for b in msg.content if isinstance(b, AnthropicThinkingBlock)]
        assert thinking_blocks, "expected thinking block in claude_thinking_real recording"
        assert thinking_blocks[0].thinking, "thinking text should be non-empty"
        assert thinking_blocks[0].signature, "thinking signature must propagate through aggregator (regression net for PR-0395)"

    def test_thinking_then_text_preserves_block_order(self):
        """Mixed-block recording: thinking block followed by text block.
        Verifies ``_completed_payloads`` preserves wire-order even with
        index reuse."""
        msg, _ = _replay_anthropic("thinking_then_text")
        assert len(msg.content) >= 2, f"expected ≥2 blocks, got {len(msg.content)}"
        # Thinking should come before text in this recording.
        thinking_idx = next(
            (i for i, b in enumerate(msg.content) if isinstance(b, AnthropicThinkingBlock)),
            -1,
        )
        text_idx = next(
            (i for i, b in enumerate(msg.content) if isinstance(b, AnthropicTextBlock)),
            -1,
        )
        assert thinking_idx >= 0 and text_idx >= 0, "missing thinking or text block"
        assert thinking_idx < text_idx, "thinking should precede text in this recording"

    def test_thinking_then_tool_call(self):
        """Thinking + tool_use combo — both blocks must be sealed and
        survive build(). Tests the deleted Set B 'thinking_then_text'
        + 'tool_use' parallel case."""
        msg, _ = _replay_anthropic("claude_thinking_then_tool")
        thinking = [b for b in msg.content if isinstance(b, AnthropicThinkingBlock)]
        tool_uses = [b for b in msg.content if isinstance(b, AnthropicToolUseBlock)]
        assert thinking, "expected thinking block"
        assert tool_uses, "expected tool_use block"
        assert tool_uses[0].name, "tool name must be set"
        assert isinstance(tool_uses[0].input, dict), "tool input must be a dict"

    def test_complex_tool_args_partial_json_concatenation(self):
        """Eager_input_streaming sends tool args as fragmented partial_json
        across many delta events. PR-0395 was a real regression where
        the concatenation produced invalid JSON. This recording captures
        that scenario; build() must produce a valid input dict."""
        msg, _ = _replay_anthropic("claude_complex_tool_args")
        tool_uses = [b for b in msg.content if isinstance(b, AnthropicToolUseBlock)]
        assert tool_uses, "expected at least one tool_use block"
        # input must be a dict (not the {"_raw": ...} fallback, which
        # would mean parsing failed).
        for tu in tool_uses:
            assert isinstance(tu.input, dict)
            assert "_raw" not in tu.input, f"tool {tu.name!r} fell back to _raw — partial_json reassembly failed"

    def test_parallel_tool_use_all_blocks_present(self):
        """Multiple tool_use blocks in a single response — each at its
        own content_block index. All must survive build()."""
        msg, _ = _replay_anthropic("claude_parallel_tools")
        tool_uses = [b for b in msg.content if isinstance(b, AnthropicToolUseBlock)]
        assert len(tool_uses) >= 2, f"expected ≥2 parallel tools, got {len(tool_uses)}"
        # Each must have distinct id + name.
        ids = {t.id for t in tool_uses}
        assert len(ids) == len(tool_uses), "tool_use ids must be unique"

    def test_truncated_recording_still_builds_valid_message(self):
        """Real recording where the upstream connection dropped — no
        message_stop. build() must still return a structurally valid
        Message (truncated content is OK; corrupt content is not)."""
        msg, _ = _replay_anthropic("claude_truncated")
        # Should not raise and should give us a Message.
        assert isinstance(msg, AnthropicMessage)
        # Content may be partial but must be a list.
        assert isinstance(msg.content, list)


# ── OpenAI Chat ─────────────────────────────────────────────────────
#
# Targets the deleted ``test_openai_chat_stream_aggregator_*`` cases.
# The most consequential is reasoning_details verbatim preservation
# (PR #475 fix — got broken once already, needs an explicit guard).


class TestOpenAIChatReplay:
    def test_plain_text_round_trip(self):
        completion, _ = _replay_openai_chat("plain_text")
        assert completion.choices, "expected at least one choice"
        msg = completion.choices[0].message
        assert msg.content, "content should be non-empty"
        assert msg.role == "assistant"

    def test_single_tool_call_preserved(self):
        completion, _ = _replay_openai_chat("single_tool_call")
        msg = completion.choices[0].message
        assert msg.tool_calls, "expected tool_calls in single_tool_call recording"
        tc = msg.tool_calls[0]
        assert tc.id, "tool_call.id must be set"
        assert tc.function.name, "tool_call.function.name must be set"
        # Arguments must be a non-empty JSON string.
        assert tc.function.arguments
        # Arguments should be parseable (the aggregator preserves them as a
        # string; a regression that splits/reorders chunks would corrupt it).
        import json  # noqa: PLC0415

        args = json.loads(tc.function.arguments)
        assert isinstance(args, dict)

    def test_reasoning_content_preserved_for_deepseek_pro(self):
        """DeepSeek-style ``reasoning_content`` field on ChatCompletionMessage.
        Deleted Set B test:
        ``test_openai_chat_stream_aggregator_preserves_reasoning_details_verbatim``
        guarded against silently dropping the chain-of-thought field.
        """
        completion, _ = _replay_openai_chat("deepseek_pro_thinking")
        msg = completion.choices[0].message
        # reasoning_content is provider-specific; access via getattr since
        # the SDK's strict ChatCompletionMessage doesn't declare it.
        rc = getattr(msg, "reasoning_content", None)
        assert rc, (
            "deepseek_pro_thinking recording should produce non-empty reasoning_content "
            "(regression net for PR #475 'preserve reasoning_details verbatim')"
        )

    def test_openrouter_reasoning_details_preserved(self):
        """OpenRouter wraps reasoning in ``reasoning_details: list[dict]``
        rather than ``reasoning_content: str``. Deleted Set B test
        explicitly preserved this list verbatim through chunk merging.
        """
        completion, _ = _replay_openai_chat("openrouter_nemotron_reasoning")
        msg = completion.choices[0].message
        rd = getattr(msg, "reasoning_details", None)
        assert rd, "openrouter_nemotron_reasoning should yield reasoning_details list"
        assert isinstance(rd, list)

    def test_usage_propagates_through_chunks(self):
        """``usage`` arrives in the final ``[DONE]`` chunk for OpenAI Chat
        (when ``stream_options.include_usage=true``). build() must
        populate ``ChatCompletion.usage`` with non-zero counts.
        """
        completion, _ = _replay_openai_chat("gpt5_plain")
        assert completion.usage is not None, "usage must propagate to final ChatCompletion"
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0


# ── OpenAI Responses ────────────────────────────────────────────────


class TestOpenAIResponsesReplay:
    def test_plain_text_response_built(self):
        response, _ = _replay_openai_responses("plain_text")
        assert isinstance(response, OpenAIResponse)
        assert response.output, "expected output items"
        # First message item should have output_text content.
        msg_items = [it for it in response.output if it.type == "message"]
        assert msg_items, "expected at least one message output item"

    def test_tool_call_output_item_reconstructed(self):
        """function_call output_item.added/done sequence reconstructs
        into a typed function_call output item with arguments populated.
        Deleted Set B test:
        ``test_openai_responses_stream_aggregator_reconstructs_items``."""
        response, _ = _replay_openai_responses("tool_call")
        fn_calls = [it for it in response.output if it.type == "function_call"]
        assert fn_calls, "expected function_call output item"
        fc = fn_calls[0]
        assert fc.name, "function_call.name must be set"
        assert fc.arguments, "function_call.arguments must be set"
        assert fc.call_id, "function_call.call_id must be set"

    def test_parallel_tools_all_present(self):
        response, _ = _replay_openai_responses("gpt5_parallel_tools")
        fn_calls = [it for it in response.output if it.type == "function_call"]
        assert len(fn_calls) >= 2, f"expected ≥2 parallel function_calls, got {len(fn_calls)}"
        ids = {fc.call_id for fc in fn_calls}
        assert len(ids) == len(fn_calls), "call_ids must be unique"

    def test_reasoning_summary_present_when_high_effort(self):
        response, _ = _replay_openai_responses("gpt5_high_reasoning")
        # Reasoning items appear in output for gpt-5 with reasoning effort.
        reasoning_items = [it for it in response.output if it.type == "reasoning"]
        assert reasoning_items, (
            "expected reasoning items in gpt5_high_reasoning recording — regression net for "
            "the OpenAI Responses reasoning summary aggregation path"
        )


# ── Gemini REST ─────────────────────────────────────────────────────
#
# Gemini ``thoughtSignature`` preservation + functionCall reconstruction
# replace the deleted ``test_gemini_rest.py`` aggregator coverage.


class TestGeminiReplay:
    def test_plain_text_basic_shape(self):
        response, _ = _replay_gemini("plain_text")
        assert "candidates" in response, "expected candidates in built response"
        candidates = response["candidates"]
        assert candidates, "candidates list must be non-empty"
        parts = candidates[0]["content"]["parts"]
        assert any(p.get("text") for p in parts), "expected at least one text part"

    def test_thinking_text_part_carries_thought_flag(self):
        """When Gemini emits reasoning AS A TEXT PART (not just usage),
        that part has ``thought: true``. Some Gemini-3 recordings only
        expose thinking via ``thoughtsTokenCount`` without dedicated
        parts; ``thinking_then_text`` is the recording that has both
        the dedicated thought part AND the answer text part — exactly
        what the deleted ``test_response_with_thinking`` exercised."""
        response, _ = _replay_gemini("thinking_then_text")
        candidates = response["candidates"]
        parts = candidates[0]["content"]["parts"]
        thought_parts = [p for p in parts if p.get("thought") is True]
        assert thought_parts, f"expected at least one thought=true part in thinking_then_text recording; got parts={parts!r}"

    def test_thought_signature_persisted(self):
        """``thoughtSignature`` is required for stateful continuation in
        the next turn. Deleted Set B
        ``test_response_with_thought_signature``. ``thinking_then_tool``
        recording carries the signature explicitly (``thinking_then_text``
        only has the ``thought: true`` flag without the signature)."""
        response, _ = _replay_gemini("thinking_then_tool")
        parts = response["candidates"][0]["content"]["parts"]
        sigs = [p.get("thoughtSignature") for p in parts if p.get("thoughtSignature")]
        assert sigs, "expected thoughtSignature on at least one part (regression net)"

    def test_function_call_reconstructed(self):
        response, _ = _replay_gemini("tool_call")
        parts = response["candidates"][0]["content"]["parts"]
        fc_parts = [p for p in parts if "functionCall" in p]
        assert fc_parts, "expected functionCall part in tool_call recording"
        fc = fc_parts[0]["functionCall"]
        assert fc.get("name"), "functionCall.name must be set"
        # args may be {} for parameterless tools, but the field must exist.
        assert "args" in fc

    def test_thinking_then_tool_call_both_present(self):
        """Mixed-content recording: thinking followed by tool call.
        Both must survive build()."""
        response, _ = _replay_gemini("thinking_then_tool")
        parts = response["candidates"][0]["content"]["parts"]
        has_thought = any(p.get("thought") is True for p in parts)
        has_function_call = any("functionCall" in p for p in parts)
        assert has_thought, "expected thought part"
        assert has_function_call, "expected functionCall part"

    def test_usage_metadata_propagates(self):
        response, _ = _replay_gemini("plain_text")
        usage = response.get("usageMetadata")
        assert usage is not None, "expected usageMetadata"
        assert usage.get("totalTokenCount", 0) > 0


# ── Sanity: load_recording smoke for unused glue paths ─────────────
#
# Make sure the SSE loader handles each provider's recording format
# without raising on the ones we DON'T deeply assert on, so a future
# format change shows up as an obvious test failure rather than a
# silent skip.


def _all_recordings(provider: str) -> list[str]:
    from tests.aggregator_parity.sse_loader import list_recordings  # noqa: PLC0415

    return list_recordings(provider)


def test_all_anthropic_recordings_parse_and_build():
    for scenario in _all_recordings("anthropic"):
        msg, _ = _replay_anthropic(scenario)
        assert isinstance(msg, AnthropicMessage), f"{scenario}: build() returned non-Message"


def test_all_oac_recordings_parse_and_build():
    """OpenAI Chat Completions bulk parse smoke.

    Function name uses ``oac`` (matches ``test_non_stream_loaders.py``
    convention) so the conftest auto-llm-marker — which fires on any
    name containing ``openai`` or ``chat`` and gates on ``LLM_API_KEY``
    — doesn't auto-skip this purely synthetic replay test.

    Some recordings (``openrouter_glm_plain``, ``with_logprobs``) are
    known to lack any content chunk for choice 0; the aggregator's
    documented contract is to raise ``RuntimeError`` in that case
    rather than silently fabricate an empty Choice. Match the existing
    ``openai_chat_glue.run_set_a_openai_chat`` semantics by treating
    that specific RuntimeError as a documented outcome.
    """
    from openai.types.chat import ChatCompletion  # noqa: PLC0415

    for scenario in _all_recordings("openai_chat"):
        try:
            completion, _ = _replay_openai_chat(scenario)
        except RuntimeError as exc:
            assert "never aggregated" in str(exc), (
                f"{scenario}: unexpected RuntimeError {exc!r} — only the documented "
                "'Choice N was never aggregated' contract is acceptable here"
            )
            continue
        assert isinstance(completion, ChatCompletion), f"{scenario}: build() returned non-ChatCompletion"


def test_all_oresp_recordings_parse_and_build():
    """OpenAI Responses bulk parse smoke (see ``oac`` rename rationale)."""
    for scenario in _all_recordings("openai_responses"):
        response, _ = _replay_openai_responses(scenario)
        assert isinstance(response, OpenAIResponse), f"{scenario}: build() returned non-Response"


def test_all_gemini_recordings_parse_and_build():
    for scenario in _all_recordings("gemini_rest"):
        response, _ = _replay_gemini(scenario)
        assert isinstance(response, dict), f"{scenario}: build() returned non-dict"
        assert "candidates" in response, f"{scenario}: missing candidates"


# Silence unused-import linters (the type adapters are used inside
# _replay_* helpers but importing them at module top makes the
# dependency explicit for IDEs).
_unused = (TypeAdapter, ChatCompletionChunk, ResponseStreamEvent)
