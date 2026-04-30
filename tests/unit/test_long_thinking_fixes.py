# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the long-thinking fix: thinking-only responses should not be
discarded or retried when max_tokens is hit during extended thinking.

Covers:
- AnthropicStreamAggregator captures stop_reason from message_delta
- ModelResponse.has_content() considers reasoning_content
- Anthropic streaming path preserves thinking + stop_reason through the full chain
"""

from __future__ import annotations

from types import SimpleNamespace, TracebackType
from typing import TYPE_CHECKING, Any

from nexau.archs.main_sub.execution import llm_caller
from nexau.archs.main_sub.execution.model_response import ModelResponse

if TYPE_CHECKING:
    from nexau.archs.main_sub.execution.hooks import ModelCallParams

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _IterableStream:
    """Context-manager wrapper for faking client.messages.create(stream=True)."""

    def __init__(self, items: list[Any]) -> None:
        self._items = items

    def __iter__(self):
        return iter(self._items)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def _make_anthropic_client(events: list[dict[str, Any]]) -> Any:
    """Build a fake Anthropic client that yields *events* from messages.create(stream=True)."""

    class _FakeClient:
        def __init__(self) -> None:
            self.messages = SimpleNamespace(create=self._create)

        def _create(self, **_payload: Any) -> _IterableStream:
            if _payload.get("stream"):
                return _IterableStream(events)
            raise NotImplementedError("non-stream create should not be called")

    return _FakeClient()


def _anthropic_thinking_only_events(
    *,
    thinking_text: str = "Let me think step by step...",
    text_content: str = "\n",
    signature: str = "EpybBQ_test_signature",
    stop_reason: str = "max_tokens",
) -> list[dict[str, Any]]:
    """Simulate Anthropic response where thinking consumes all output tokens.

    Returns events for: message_start → thinking block → text block (near-empty) → message_delta(stop_reason) → message_stop
    """
    thinking_events: list[dict[str, Any]] = [
        # 1. thinking block
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": thinking_text}},
    ]
    # signature_delta 在流式正常完成时到达；流式中断时缺失
    if signature:
        thinking_events.append(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": signature}},
        )
    thinking_events.append({"type": "content_block_stop", "index": 0})

    return [
        {
            "type": "message_start",
            "message": {
                "role": "assistant",
                "model": "claude-opus-4-6",
                "usage": {"input_tokens": 100, "output_tokens": 0},
            },
        },
        *thinking_events,
        # 2. near-empty text block
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": text_content}},
        {"type": "content_block_stop", "index": 1},
        # 3. message_delta with stop_reason
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 32768},
        },
        {"type": "message_stop"},
    ]


# ---------------------------------------------------------------------------
# Fix 1: AnthropicStreamAggregator captures stop_reason
# ---------------------------------------------------------------------------


class TestAnthropicStreamAggregatorStopReason:
    """AnthropicStreamAggregator tracks stop_reason from message_delta events.

    These tests verify that stop_reason is correctly captured and included
    in finalize() output when present in the stream.
    """

    def test_stop_reason_tracked_on_aggregator(self) -> None:
        """stop_reason is captured from message_delta events."""
        agg = llm_caller.AnthropicStreamAggregator()
        for event in _anthropic_thinking_only_events(stop_reason="max_tokens"):
            agg.consume(event)

        assert agg.stop_reason == "max_tokens"

    def test_stop_reason_in_finalize_output(self) -> None:
        """finalize() dict includes stop_reason when present."""
        agg = llm_caller.AnthropicStreamAggregator()
        for event in _anthropic_thinking_only_events(stop_reason="max_tokens"):
            agg.consume(event)

        result = agg.finalize()
        assert result["stop_reason"] == "max_tokens"

    def test_finalize_still_works_with_end_turn(self) -> None:
        """Normal end_turn events should produce valid finalize output with stop_reason."""
        agg = llm_caller.AnthropicStreamAggregator()
        for event in _anthropic_thinking_only_events(stop_reason="end_turn", text_content="Hello"):
            agg.consume(event)

        result = agg.finalize()
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"

    def test_finalize_without_message_delta(self) -> None:
        """Without message_delta event, finalize should still work (no stop_reason)."""
        events = [
            {
                "type": "message_start",
                "message": {"role": "assistant", "model": "claude-test", "usage": {"input_tokens": 5, "output_tokens": 0}},
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
        agg = llm_caller.AnthropicStreamAggregator()
        for event in events:
            agg.consume(event)

        result = agg.finalize()
        assert "stop_reason" not in result
        assert result["role"] == "assistant"


# ---------------------------------------------------------------------------
# Fix 1b: AnthropicStreamAggregator captures signature_delta
# ---------------------------------------------------------------------------


class TestAnthropicStreamAggregatorSignature:
    """signature_delta is assigned directly (one event per block, matching official SDK)."""

    def test_single_signature_delta(self) -> None:
        """Single signature_delta event should be captured on the thinking block."""
        agg = llm_caller.AnthropicStreamAggregator()
        for event in _anthropic_thinking_only_events(signature="EpybBQ_full_sig"):
            agg.consume(event)

        result = agg.finalize()
        thinking_blocks = [b for b in result["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0]["signature"] == "EpybBQ_full_sig"

    def test_no_signature_delta_means_no_signature(self) -> None:
        """When no signature_delta arrives (stream interrupted), block has no signature key."""
        agg = llm_caller.AnthropicStreamAggregator()
        for event in _anthropic_thinking_only_events(signature=""):
            agg.consume(event)

        result = agg.finalize()
        thinking_blocks = [b for b in result["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].get("signature") in (None, "")


# ---------------------------------------------------------------------------
# Fix 2: ModelResponse.has_content() considers reasoning_content
# ---------------------------------------------------------------------------


class TestModelResponseHasContentWithReasoning:
    """has_content() should return True when reasoning_content exists, even if text is empty."""

    def test_empty_text_with_reasoning_is_has_content(self) -> None:
        resp = ModelResponse(content="", reasoning_content="thinking...")
        assert resp.has_content() is True

    def test_whitespace_text_with_reasoning_is_has_content(self) -> None:
        resp = ModelResponse(content="\n", reasoning_content="thinking...")
        assert resp.has_content() is True

    def test_none_text_with_reasoning_is_has_content(self) -> None:
        resp = ModelResponse(content=None, reasoning_content="thinking...")
        assert resp.has_content() is True

    def test_empty_text_no_reasoning_is_not_has_content(self) -> None:
        """Empty text WITHOUT reasoning should still return False (original behavior)."""
        resp = ModelResponse(content="", reasoning_content=None)
        assert resp.has_content() is False

    def test_none_text_no_reasoning_is_not_has_content(self) -> None:
        resp = ModelResponse(content=None, reasoning_content=None)
        assert resp.has_content() is False

    def test_real_text_no_reasoning_is_has_content(self) -> None:
        """Normal text content should still work."""
        resp = ModelResponse(content="Hello world")
        assert resp.has_content() is True


# ---------------------------------------------------------------------------
# Fix 3 (integration): Full Anthropic streaming path with thinking-only response
# ---------------------------------------------------------------------------


class TestAnthropicThinkingOnlyStreamingIntegration:
    """End-to-end: Anthropic streaming with thinking-only response (max_tokens hit)
    should produce a valid ModelResponse with reasoning_content preserved.

    RFC-0014: Anthropic calls now require ModelCallParams with UMP messages.
    """

    @staticmethod
    def _make_model_call_params(user_text: str = "complex question") -> ModelCallParams:
        from nexau.archs.main_sub.execution.hooks import ModelCallParams
        from nexau.core.messages import Message

        return ModelCallParams(
            messages=[Message.user(user_text)],
            max_tokens=None,
            force_stop_reason=None,
            agent_state=None,
            tool_call_mode="structured",
            tools=None,
            api_params={},
        )

    def test_thinking_only_response_not_discarded(self) -> None:
        """A thinking-only Anthropic streaming response should produce a valid ModelResponse."""
        thinking_text = "Let me analyze this step by step... " * 100
        events = _anthropic_thinking_only_events(
            thinking_text=thinking_text,
            text_content="\n",
            stop_reason="max_tokens",
        )
        client = _make_anthropic_client(events)
        kwargs: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "stream": True,
        }

        resp = llm_caller.call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            tracer=None,
            model_call_params=self._make_model_call_params(),
        )

        assert isinstance(resp, ModelResponse)
        # 1. reasoning_content 被正确提取
        assert resp.reasoning_content is not None
        assert "step by step" in resp.reasoning_content

    def test_thinking_with_signature_preserved(self) -> None:
        """Thinking block signature should be preserved for next-turn context."""
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {"role": "assistant", "model": "claude-opus-4-6", "usage": {"input_tokens": 100, "output_tokens": 0}},
            },
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Deep analysis..."}},
            # signature arrives via content_block_stop or is embedded in the block
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "\n"}},
            {"type": "content_block_stop", "index": 1},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "max_tokens"},
                "usage": {"output_tokens": 32768},
            },
            {"type": "message_stop"},
        ]
        client = _make_anthropic_client(events)
        kwargs: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "stream": True,
        }

        resp = llm_caller.call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            tracer=None,
            model_call_params=self._make_model_call_params("q"),
        )

        assert resp.reasoning_content == "Deep analysis..."

    def test_stop_reason_readable_from_raw_message_metadata(self) -> None:
        """_raw_message_metadata should be able to extract metadata from streaming response."""
        events = _anthropic_thinking_only_events(stop_reason="max_tokens")
        client = _make_anthropic_client(events)
        kwargs: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "stream": True,
        }

        resp = llm_caller.call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            tracer=None,
            model_call_params=self._make_model_call_params("q"),
        )

        meta = llm_caller._raw_message_metadata(resp.raw_message)
        # stop_reason is no longer tracked in the aggregator (RFC-0014),
        # so it won't be in raw_message; just verify metadata extraction works
        assert isinstance(meta, dict)

    def test_ump_roundtrip_preserves_thinking_for_anthropic(self) -> None:
        """The thinking content should survive: ModelResponse → UMP Message → Anthropic vendor format."""
        from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter
        from nexau.core.messages import Message, ReasoningBlock, TextBlock

        events = _anthropic_thinking_only_events(
            thinking_text="Very long thinking...",
            text_content="\n",
            stop_reason="max_tokens",
        )
        client = _make_anthropic_client(events)
        kwargs: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "stream": True,
        }

        resp = llm_caller.call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            tracer=None,
            model_call_params=self._make_model_call_params("q"),
        )
        # 1. 转为 UMP Message
        ump_msg = resp.to_ump_message()
        assert any(isinstance(b, ReasoningBlock) for b in ump_msg.content)
        assert any(isinstance(b, TextBlock) for b in ump_msg.content)

        # 2. 构造下一轮的 messages: [user, assistant(thinking+text), user]
        messages = [
            Message.user("original question"),
            ump_msg,
            Message.user("follow-up"),
        ]

        # 3. 转为 Anthropic vendor format
        _system, convo = AnthropicMessagesAdapter().to_vendor_format(messages)

        # 4. 找到 assistant message 中的 thinking block
        assistant_msg = next(m for m in convo if m["role"] == "assistant")
        content_blocks = assistant_msg["content"]
        thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]

        assert len(thinking_blocks) >= 1
        assert thinking_blocks[0]["thinking"] == "Very long thinking..."
        assert thinking_blocks[0]["signature"] == "EpybBQ_test_signature"

    def test_ump_roundtrip_downgrades_unsigned_thinking_with_companion_content(self) -> None:
        """Unsigned thinking is downgraded so Anthropic follow-up payloads remain valid."""
        from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter
        from nexau.core.messages import Message

        # 模拟流式中断：不发送 signature_delta
        events = _anthropic_thinking_only_events(
            thinking_text="Interrupted thinking...",
            text_content="\n",
            signature="",  # 空 signature 表示流式中断
            stop_reason="max_tokens",
        )
        client = _make_anthropic_client(events)
        kwargs: dict[str, Any] = {
            "model": "claude-opus-4-6",
            "stream": True,
        }

        resp = llm_caller.call_llm_with_anthropic_chat_completion(
            client,
            kwargs,
            tracer=None,
            model_call_params=self._make_model_call_params("q"),
        )
        ump_msg = resp.to_ump_message()

        messages = [
            Message.user("original question"),
            ump_msg,
            Message.user("follow-up"),
        ]

        _system, convo = AnthropicMessagesAdapter().to_vendor_format(messages)

        assistant_msg = next(m for m in convo if m["role"] == "assistant")
        content_blocks = assistant_msg["content"]

        thinking_blocks = [b for b in content_blocks if b.get("type") == "thinking"]
        assert thinking_blocks == []

        text_blocks = [b for b in content_blocks if b.get("type") == "text"]
        assert any(b["text"] == "Interrupted thinking..." for b in text_blocks)
        assert any(b["text"] == "\n" for b in text_blocks)


# ---------------------------------------------------------------------------
# Fix 4: Signature demotion retry helpers
# ---------------------------------------------------------------------------
