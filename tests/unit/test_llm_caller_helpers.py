# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mock-based unit tests for ``llm_caller`` helpers + Set A edge cases.

These cover paths that are awkward to exercise via live e2e:

- ``_maybe_wrap_stream_idle_timeout`` exception normalization
- ``_get_event_emitter`` middleware-chain walk
- ``_chat_completion_to_model_response`` adapter shape handling
- ``_process_stream_chunk`` middleware drop semantics
- Set A Anthropic aggregator: duplicate ``content_block_start`` and
  truncated streams (message_stop never arrives)

The corresponding live e2e tests in
``tests/integration/test_aggregator_live_e2e.py`` cover the
positive paths against real provider endpoints; this file pins
the negative / error-handling branches.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest
import requests
from anthropic.types import (
    InputJSONDelta,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
)
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    TextBlock as AnthropicTextBlock,
)
from anthropic.types import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import TypeAdapter

from nexau.archs.llm.llm_aggregators import AnthropicEventAggregator
from nexau.archs.llm.llm_aggregators.events import Event
from nexau.archs.main_sub.execution.hooks import Middleware, MiddlewareManager, ModelCallParams
from nexau.archs.main_sub.execution.llm_caller import (
    StreamIdleTimeoutError,
    _chat_completion_to_model_response,
    _get_event_emitter,
    _maybe_wrap_stream_idle_timeout,
    _process_stream_chunk,
    _resolve_run_id,
)

# ── _maybe_wrap_stream_idle_timeout ──────────────────────────────────


class TestMaybeWrapStreamIdleTimeout:
    def test_wraps_httpx_read_timeout(self):
        exc = httpx.ReadTimeout("read timed out")
        wrapped = _maybe_wrap_stream_idle_timeout(exc, transport_name="anthropic stream", llm_config=None)
        assert isinstance(wrapped, StreamIdleTimeoutError)
        assert "anthropic stream" in str(wrapped)

    def test_wraps_requests_read_timeout(self):
        exc = requests.exceptions.ReadTimeout("read timed out")
        wrapped = _maybe_wrap_stream_idle_timeout(exc, transport_name="gemini stream", llm_config=None)
        assert isinstance(wrapped, StreamIdleTimeoutError)
        assert "gemini stream" in str(wrapped)

    def test_wraps_builtin_timeout_error(self):
        exc = TimeoutError("frame timeout")
        wrapped = _maybe_wrap_stream_idle_timeout(exc, transport_name="openai stream", llm_config=None)
        assert isinstance(wrapped, StreamIdleTimeoutError)

    def test_wraps_duck_typed_class_name_match(self):
        # SDKs sometimes ship their own Timeout class hierarchy that
        # doesn't subclass the stdlib / httpx ones; the helper falls
        # back to a class-name check.
        class APITimeoutError(Exception):
            pass

        exc = APITimeoutError("upstream slow")
        wrapped = _maybe_wrap_stream_idle_timeout(exc, transport_name="x", llm_config=None)
        assert isinstance(wrapped, StreamIdleTimeoutError)

    def test_returns_none_for_unrelated_exception(self):
        exc = ValueError("not a timeout")
        wrapped = _maybe_wrap_stream_idle_timeout(exc, transport_name="x", llm_config=None)
        assert wrapped is None

    def test_returns_none_for_runtime_error(self):
        exc = RuntimeError("aggregator state error")
        wrapped = _maybe_wrap_stream_idle_timeout(exc, transport_name="x", llm_config=None)
        assert wrapped is None


# ── _get_event_emitter ──────────────────────────────────────────────


class TestGetEventEmitter:
    def test_no_manager_returns_noop(self):
        emitter = _get_event_emitter(None)
        # Should be callable with any event without raising.
        emitter(MagicMock())

    def test_empty_manager_returns_noop(self):
        manager = MiddlewareManager(middlewares=[])
        emitter = _get_event_emitter(manager)
        emitter(MagicMock())

    def test_walks_chain_for_first_middleware_with_handler(self):
        events: list[Event] = []

        class _CapturingMiddleware(Middleware):
            def __init__(self, sink: list[Event]) -> None:
                self.on_event = sink.append

        # Two middlewares, only the second has on_event — emitter must
        # find it via ``get_event_handler``.
        plain = Middleware()
        capturing = _CapturingMiddleware(events)
        manager = MiddlewareManager(middlewares=[plain, capturing])

        emitter = _get_event_emitter(manager)
        sentinel = MagicMock()
        emitter(sentinel)

        assert events == [sentinel]


# ── _resolve_run_id ─────────────────────────────────────────────────


class TestResolveRunId:
    def test_none_params_returns_default(self):
        assert _resolve_run_id(None) == "stream"

    def test_no_agent_state_returns_default(self):
        params = ModelCallParams(
            messages=[],
            max_tokens=None,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="auto",
            tools=None,
            api_params={},
        )
        assert _resolve_run_id(params) == "stream"

    def test_agent_state_run_id_returned(self):
        agent_state = MagicMock()
        agent_state.run_id = "run_xyz"
        params = ModelCallParams(
            messages=[],
            max_tokens=None,
            force_stop_reason=None,
            agent_state=agent_state,
            tool_call_mode="auto",
            tools=None,
            api_params={},
        )
        assert _resolve_run_id(params) == "run_xyz"


# ── _chat_completion_to_model_response ──────────────────────────────


class TestChatCompletionToModelResponse:
    def test_extracts_message_and_usage_from_completion(self):
        completion = ChatCompletion(
            id="cmpl_test",
            object="chat.completion",
            created=0,
            model="m",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(role="assistant", content="hi there"),
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
        )

        response = _chat_completion_to_model_response(completion)

        assert response.content == "hi there"
        assert response.usage is not None
        assert response.usage.total_tokens == 5

    def test_empty_choices_raises_value_error(self):
        # Set A's ``build()`` already raises ``RuntimeError`` if no
        # chunks were observed, so a no-choice completion shouldn't
        # reach this helper in normal flow. Document the contract:
        # if it ever does, ``from_openai_message`` raises
        # ``ValueError("message cannot be None")`` rather than
        # silently corrupting state.
        completion = ChatCompletion(
            id="cmpl_test",
            object="chat.completion",
            created=0,
            model="m",
            choices=[],
            usage=None,
        )
        with pytest.raises(ValueError, match="message cannot be None"):
            _chat_completion_to_model_response(completion)


# ── _process_stream_chunk ───────────────────────────────────────────


class TestProcessStreamChunk:
    def test_no_manager_passes_chunk_through_unmodified(self):
        chunk = {"id": "x"}
        result = _process_stream_chunk(chunk, None, None)
        assert result is chunk

    def test_no_params_passes_chunk_through_unmodified(self):
        chunk = {"id": "x"}
        manager = MiddlewareManager(middlewares=[])
        result = _process_stream_chunk(chunk, manager, None)
        assert result is chunk

    def test_middleware_can_drop_chunk_by_returning_none(self):
        class _DroppingMiddleware(Middleware):
            def stream_chunk(self, chunk, params):  # noqa: ARG002 — interface match
                return None

        manager = MiddlewareManager(middlewares=[_DroppingMiddleware()])
        params = ModelCallParams(
            messages=[],
            max_tokens=None,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="auto",
            tools=None,
            api_params={},
        )

        result = _process_stream_chunk({"id": "x"}, manager, params)
        # Manager returns None when any middleware drops; llm_caller's
        # streaming loop honors this with ``if processed_chunk is None:
        # continue`` so the aggregator never sees the dropped chunk.
        assert result is None

    def test_middleware_can_mutate_chunk(self):
        replacement = {"id": "replaced"}

        class _ReplacingMiddleware(Middleware):
            def stream_chunk(self, chunk, params):  # noqa: ARG002
                return replacement

        manager = MiddlewareManager(middlewares=[_ReplacingMiddleware()])
        params = ModelCallParams(
            messages=[],
            max_tokens=None,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="auto",
            tools=None,
            api_params={},
        )

        result = _process_stream_chunk({"id": "original"}, manager, params)
        assert result is replacement


# ── Set A Anthropic aggregator: duplicate content_block_start ────────


# One TypeAdapter per concrete event type. Each helper builds the
# wire-format dict for one specific shape, so a per-type adapter
# returns the exact subtype — no union narrowing required, no
# ``type: ignore`` band-aids. The pydantic adapter does shape
# validation cheaply at construction time.

_MSG_START_ADAPTER: TypeAdapter[RawMessageStartEvent] = TypeAdapter(RawMessageStartEvent)
_MSG_STOP_ADAPTER: TypeAdapter[RawMessageStopEvent] = TypeAdapter(RawMessageStopEvent)
_MSG_DELTA_ADAPTER: TypeAdapter[RawMessageDeltaEvent] = TypeAdapter(RawMessageDeltaEvent)
_BLOCK_START_ADAPTER: TypeAdapter[RawContentBlockStartEvent] = TypeAdapter(RawContentBlockStartEvent)
_BLOCK_DELTA_ADAPTER: TypeAdapter[RawContentBlockDeltaEvent] = TypeAdapter(RawContentBlockDeltaEvent)
_BLOCK_STOP_ADAPTER: TypeAdapter[RawContentBlockStopEvent] = TypeAdapter(RawContentBlockStopEvent)


def _msg_start() -> RawMessageStartEvent:
    return _MSG_START_ADAPTER.validate_python(
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "claude-test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        }
    )


def _msg_stop() -> RawMessageStopEvent:
    return _MSG_STOP_ADAPTER.validate_python({"type": "message_stop"})


def _msg_delta_with_stop_reason(*, output_tokens: int = 5, stop_reason: str = "end_turn") -> RawMessageDeltaEvent:
    return _MSG_DELTA_ADAPTER.validate_python(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
    )


def _text_block_start(idx: int) -> RawContentBlockStartEvent:
    return _BLOCK_START_ADAPTER.validate_python(
        {
            "type": "content_block_start",
            "index": idx,
            "content_block": {"type": "text", "text": ""},
        }
    )


def _text_delta(idx: int, text: str) -> RawContentBlockDeltaEvent:
    return _BLOCK_DELTA_ADAPTER.validate_python(
        {
            "type": "content_block_delta",
            "index": idx,
            "delta": {"type": "text_delta", "text": text},
        }
    )


def _block_stop(idx: int) -> RawContentBlockStopEvent:
    return _BLOCK_STOP_ADAPTER.validate_python({"type": "content_block_stop", "index": idx})


class TestAnthropicAggregatorEdgeCases:
    """Edge cases the deleted Set B unit tests used to cover."""

    def test_truncated_stream_no_message_stop(self):
        """Aggregator should still build a valid Message if message_stop
        never arrives (e.g., upstream connection drops mid-stream).

        Mirrors Set B's ``_flush_active_blocks`` semantics — emit
        whatever was sealed plus anything still active so downstream
        consumers see a consistent (if truncated) Message rather than
        a half-formed pydantic object.
        """
        events: list[Event] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="t1")

        agg.aggregate(_msg_start())
        agg.aggregate(_text_block_start(0))
        agg.aggregate(_text_delta(0, "Hello "))
        agg.aggregate(_text_delta(0, "world"))
        agg.aggregate(_block_stop(0))
        # NO message_delta, NO message_stop — connection drops here.

        message = agg.build()

        assert isinstance(message, AnthropicMessage)
        assert message.role == "assistant"
        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, AnthropicTextBlock)
        assert block.text == "Hello world"
        # No stop_reason was observed; aggregator should not invent one.
        assert message.stop_reason is None

    def test_duplicate_content_block_start_starts_fresh_block(self):
        """A second ``content_block_start`` at the same index after a
        block_stop should open a fresh accumulator, not corrupt the
        sealed first one.

        Index reuse is documented behavior: e.g., Anthropic reuses
        ``index=0`` for both a thinking block and a follow-up tool_use
        in the rec_single_tool_call fixture. The seal-on-stop
        semantics let the aggregator preserve both.
        """
        events: list[Event] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="t2")

        agg.aggregate(_msg_start())

        # First block at index 0: text "alpha"
        agg.aggregate(_text_block_start(0))
        agg.aggregate(_text_delta(0, "alpha"))
        agg.aggregate(_block_stop(0))

        # Second block reusing index 0: text "beta"
        agg.aggregate(_text_block_start(0))
        agg.aggregate(_text_delta(0, "beta"))
        agg.aggregate(_block_stop(0))

        agg.aggregate(_msg_delta_with_stop_reason())
        agg.aggregate(_msg_stop())

        message = agg.build()

        assert len(message.content) == 2, f"expected 2 sealed blocks, got {len(message.content)}"
        assert isinstance(message.content[0], AnthropicTextBlock)
        assert isinstance(message.content[1], AnthropicTextBlock)
        assert message.content[0].text == "alpha"
        assert message.content[1].text == "beta"

    def test_partial_tool_call_input_json_recovers(self):
        """A tool_use block that received fragmented partial_json
        (real-world: providers split JSON across many delta events)
        must still parse to a usable ``input`` dict on build().
        """
        events: list[Event] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="t3")

        agg.aggregate(_msg_start())

        # Open tool_use block.
        agg.aggregate(
            _BLOCK_START_ADAPTER.validate_python(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool_abc",
                        "name": "lookup",
                        "input": {},
                    },
                }
            )
        )
        # Stream the JSON in 3 fragments.
        for fragment in ('{"q":', ' "weather"', "}"):
            agg.aggregate(
                RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=InputJSONDelta(type="input_json_delta", partial_json=fragment),
                )
            )
        agg.aggregate(_block_stop(0))
        agg.aggregate(_msg_delta_with_stop_reason(stop_reason="tool_use"))
        agg.aggregate(_msg_stop())

        message = agg.build()

        assert len(message.content) == 1
        block = message.content[0]
        assert isinstance(block, AnthropicToolUseBlock)
        assert block.name == "lookup"
        assert block.input == {"q": "weather"}

    def test_invalid_tool_input_json_falls_back_to_raw(self):
        """Malformed JSON should not crash; aggregator returns
        ``{"_raw": <buffer>}`` so downstream can still log/inspect.
        """
        events: list[Event] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="t4")

        agg.aggregate(_msg_start())
        agg.aggregate(
            _BLOCK_START_ADAPTER.validate_python(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool_xyz",
                        "name": "broken",
                        "input": {},
                    },
                }
            )
        )
        agg.aggregate(
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=InputJSONDelta(type="input_json_delta", partial_json="not valid json {{{"),
            )
        )
        agg.aggregate(_block_stop(0))
        agg.aggregate(_msg_delta_with_stop_reason(stop_reason="tool_use"))
        agg.aggregate(_msg_stop())

        message = agg.build()
        block = message.content[0]
        assert isinstance(block, AnthropicToolUseBlock)
        assert block.input == {"_raw": "not valid json {{{"}
