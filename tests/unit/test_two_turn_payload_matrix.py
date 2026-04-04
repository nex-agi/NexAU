# Licensed under the Apache License, Version 2.0

"""Two-turn payload matrix tests for provider reasoning history.

These tests are intentionally focused on the provider-boundary conversion layer:

1. Normalize one provider response into UMP via ``ModelResponse.to_ump_message()``
2. Append a second user turn
3. Serialize the full two-turn history into a target provider payload

The goal is to lock down today's behavior and document the desired future behavior
for UMP -> provider payload conversion, especially around reasoning / thinking
artifacts.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter
from nexau.core.adapters.gemini_messages import GeminiMessagesAdapter
from nexau.core.messages import Message
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload
from nexau.core.serializers.openai_responses import prepare_openai_responses_api_input

_TASK_A = "Task A: Compute ((35 * 11) + 57) / 13 and end with Final A: <number>."
_TASK_B = "Task B: Using A, compute (A * 3) + 15 and end with Final B: <number>."


def _completion_source() -> ModelResponse:
    return ModelResponse.from_openai_message(
        {
            "role": "assistant",
            "content": "35 × 11 = 385\n385 + 57 = 442\n442 ÷ 13 = 34\n\nFinal A: 34",
            "reasoning_content": "completion reasoning",
        },
        usage={"prompt_tokens": 37, "completion_tokens": 90, "total_tokens": 127},
    )


def _responses_source() -> ModelResponse:
    return ModelResponse.from_openai_response(
        {
            "output": [
                {
                    "id": "rs_real",
                    "type": "reasoning",
                    "encrypted_content": "encrypted_blob",
                    "summary": [{"type": "summary_text", "text": "responses reasoning summary"}],
                },
                {
                    "id": "msg_real",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Final A: 34"}],
                },
            ],
            "usage": {
                "input_tokens": 43,
                "output_tokens": 73,
                "total_tokens": 116,
                "output_tokens_details": {"reasoning_tokens": 38},
            },
        }
    )


def _claude_source() -> ModelResponse:
    return ModelResponse.from_anthropic_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "claude thinking", "signature": "claude_sig"},
                {"type": "text", "text": "Final A: 34"},
            ],
        },
        usage={"input_tokens": 80, "output_tokens": 89},
    )


def _gemini_source() -> ModelResponse:
    return ModelResponse.from_gemini_rest(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "gemini thought", "thought": True},
                            {"text": "Final A: 34", "thoughtSignature": "gemini_sig"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 31,
                "candidatesTokenCount": 8,
                "thoughtsTokenCount": 12,
                "totalTokenCount": 51,
            },
        }
    )


_SOURCE_FACTORIES: dict[str, Callable[[], ModelResponse]] = {
    "completion": _completion_source,
    "responses": _responses_source,
    "claude": _claude_source,
    "gemini": _gemini_source,
}


def _build_two_turn_history(source_name: str) -> list[Message]:
    source_response = _SOURCE_FACTORIES[source_name]()
    return [
        Message.user(_TASK_A),
        source_response.to_ump_message(),
        Message.user(_TASK_B),
    ]


class TestTwoTurnPayloadMatrix:
    """Matrix tests for source-provider -> target-payload two-turn conversion."""

    @pytest.mark.parametrize(
        ("source_name", "expected_reasoning", "expects_response_items", "expected_signature", "expected_thought_signature"),
        [
            ("completion", "completion reasoning", False, None, None),
            ("responses", "responses reasoning summary", True, None, None),
            ("claude", "claude thinking", False, "claude_sig", None),
            ("gemini", "gemini thought", False, None, "gemini_sig"),
        ],
    )
    def test_two_turn_legacy_completion_payload(
        self,
        source_name,
        expected_reasoning,
        expects_response_items,
        expected_signature,
        expected_thought_signature,
    ):
        history = _build_two_turn_history(source_name)

        payload = serialize_ump_to_openai_chat_payload(history)

        assert payload[0] == {"role": "user", "content": _TASK_A}
        assert payload[2] == {"role": "user", "content": _TASK_B}

        assistant = payload[1]
        assert assistant["role"] == "assistant"
        expected_text = "35 × 11 = 385\n385 + 57 = 442\n442 ÷ 13 = 34\n\nFinal A: 34" if source_name == "completion" else "Final A: 34"
        assert assistant["content"] == expected_text
        assert assistant.get("reasoning_content") == expected_reasoning
        assert ("response_items" in assistant) is expects_response_items
        assert assistant.get("reasoning_signature") == expected_signature
        assert assistant.get("thought_signature") == expected_thought_signature

    @pytest.mark.parametrize(
        ("source_name", "expected_reasoning_text", "expects_encrypted_content"),
        [
            ("completion", "completion reasoning", False),
            ("responses", "responses reasoning summary", True),
            ("claude", "claude thinking", False),
            ("gemini", "gemini thought", False),
        ],
    )
    def test_two_turn_responses_payload_replays_reasoning(self, source_name, expected_reasoning_text, expects_encrypted_content):
        history = _build_two_turn_history(source_name)
        payload = serialize_ump_to_openai_chat_payload(history)

        prepared, instructions = prepare_openai_responses_api_input(payload)

        assert instructions is None
        assert prepared[0] == {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": _TASK_A}],
        }
        assert prepared[-1] == {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": _TASK_B}],
        }

        reasoning_items = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["summary"][0]["text"] == expected_reasoning_text
        if expects_encrypted_content:
            assert reasoning_items[0]["encrypted_content"] == "encrypted_blob"
            assert "content" not in reasoning_items[0]
        else:
            assert "content" not in reasoning_items[0]
            assert "encrypted_content" not in reasoning_items[0]

    @pytest.mark.parametrize(
        ("source_name", "expected_reasoning_text"),
        [
            ("completion", "completion reasoning"),
            ("claude", "claude thinking"),
            ("gemini", "gemini thought"),
        ],
    )
    def test_two_turn_responses_payload_reconstructs_reasoning_for_non_responses_sources(self, source_name, expected_reasoning_text):
        history = _build_two_turn_history(source_name)
        payload = serialize_ump_to_openai_chat_payload(history)

        prepared, _instructions = prepare_openai_responses_api_input(payload)

        reasoning_items = [item for item in prepared if item.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert "content" not in reasoning_items[0]
        assert reasoning_items[0]["summary"][0]["text"] == expected_reasoning_text

    @pytest.mark.parametrize(
        ("source_name", "expects_thinking_block", "expected_first_text"),
        [
            ("completion", False, "completion reasoning"),
            ("responses", False, "responses reasoning summary"),
            ("claude", True, None),
            ("gemini", False, "gemini thought"),
        ],
    )
    def test_two_turn_claude_payload_behavior(self, source_name, expects_thinking_block, expected_first_text):
        history = _build_two_turn_history(source_name)

        system_blocks, convo = AnthropicMessagesAdapter().to_vendor_format(history)

        assert system_blocks == []
        assert convo[0] == {"role": "user", "content": [{"type": "text", "text": _TASK_A}]}
        assert convo[2] == {"role": "user", "content": [{"type": "text", "text": _TASK_B}]}

        assistant_blocks = convo[1]["content"]
        thinking_blocks = [block for block in assistant_blocks if block.get("type") == "thinking"]
        text_blocks = [block for block in assistant_blocks if block.get("type") == "text"]

        if expects_thinking_block:
            assert len(thinking_blocks) == 1
            assert thinking_blocks[0]["thinking"] == "claude thinking"
            assert thinking_blocks[0]["signature"] == "claude_sig"
            assert text_blocks[-1]["text"] == "Final A: 34"
        else:
            assert thinking_blocks == []
            expected_text = "35 × 11 = 385\n385 + 57 = 442\n442 ÷ 13 = 34\n\nFinal A: 34" if source_name == "completion" else "Final A: 34"
            assert text_blocks[0]["text"] == expected_first_text
            assert text_blocks[-1]["text"] == expected_text

    @pytest.mark.parametrize(
        ("source_name", "expected_thought_text", "expected_signature"),
        [
            ("completion", "completion reasoning", None),
            ("responses", "responses reasoning summary", None),
            ("claude", "claude thinking", None),
            ("gemini", "gemini thought", "gemini_sig"),
        ],
    )
    def test_two_turn_gemini_payload_behavior(self, source_name, expected_thought_text, expected_signature):
        history = _build_two_turn_history(source_name)

        contents, system_instruction = GeminiMessagesAdapter().to_vendor_format(history)

        assert system_instruction is None
        assert contents[0] == {"role": "user", "parts": [{"text": _TASK_A}]}
        assert contents[2] == {"role": "user", "parts": [{"text": _TASK_B}]}

        assistant_parts = contents[1]["parts"]
        assert assistant_parts[0]["text"] == expected_thought_text
        assert assistant_parts[0]["thought"] is True
        if expected_signature is None:
            assert "thoughtSignature" not in assistant_parts[0]
        else:
            assert assistant_parts[0]["thoughtSignature"] == expected_signature
        expected_text = "35 × 11 = 385\n385 + 57 = 442\n442 ÷ 13 = 34\n\nFinal A: 34" if source_name == "completion" else "Final A: 34"
        assert assistant_parts[-1]["text"] == expected_text

    @pytest.mark.parametrize(
        ("source_name", "target_name", "expected_policy"),
        [
            ("completion", "completion", "reasoning_content"),
            ("completion", "responses", "reconstructed_reasoning_replay"),
            ("completion", "claude", "downgrade_to_text"),
            ("completion", "gemini", "thought_part"),
            ("responses", "completion", "response_items_plus_reasoning_content"),
            ("responses", "responses", "typed_reasoning_replay"),
            ("responses", "claude", "downgrade_to_text"),
            ("responses", "gemini", "thought_part"),
            ("claude", "completion", "reasoning_content_plus_signature"),
            ("claude", "responses", "reconstructed_reasoning_replay"),
            ("claude", "claude", "signed_thinking"),
            ("claude", "gemini", "thought_part"),
            ("gemini", "completion", "reasoning_content_plus_thought_signature"),
            ("gemini", "responses", "reconstructed_reasoning_replay"),
            ("gemini", "claude", "downgrade_to_text"),
            ("gemini", "gemini", "thought_part_with_signature"),
        ],
    )
    def test_two_turn_full_matrix_policy_table(self, source_name: str, target_name: str, expected_policy: str):
        history = _build_two_turn_history(source_name)

        if target_name == "completion":
            payload = serialize_ump_to_openai_chat_payload(history)
            assistant = payload[1]
            if expected_policy == "reasoning_content":
                assert assistant.get("reasoning_content") == "completion reasoning"
            elif expected_policy == "response_items_plus_reasoning_content":
                assert assistant.get("reasoning_content") == "responses reasoning summary"
                assert assistant.get("response_items")
            elif expected_policy == "reasoning_content_plus_signature":
                assert assistant.get("reasoning_content") == "claude thinking"
                assert assistant.get("reasoning_signature") == "claude_sig"
            elif expected_policy == "reasoning_content_plus_thought_signature":
                assert assistant.get("reasoning_content") == "gemini thought"
                assert assistant.get("thought_signature") == "gemini_sig"
            else:
                raise AssertionError(f"Unexpected completion policy {expected_policy}")
            return

        if target_name == "responses":
            chat_payload = serialize_ump_to_openai_chat_payload(history)
            payload, _instructions = prepare_openai_responses_api_input(chat_payload)
            reasoning_items = [item for item in payload if item.get("type") == "reasoning"]
            if expected_policy == "typed_reasoning_replay":
                assert len(reasoning_items) == 1
                assert reasoning_items[0].get("encrypted_content") == "encrypted_blob"
                assert reasoning_items[0]["summary"][0]["text"] == "responses reasoning summary"
            elif expected_policy == "reconstructed_reasoning_replay":
                assert len(reasoning_items) == 1
                assert "content" not in reasoning_items[0]
                assert reasoning_items[0]["summary"][0]["text"] in {"completion reasoning", "claude thinking", "gemini thought"}
                assert "encrypted_content" not in reasoning_items[0]
            else:
                raise AssertionError(f"Unexpected responses policy {expected_policy}")
            return

        if target_name == "claude":
            _system, payload = AnthropicMessagesAdapter().to_vendor_format(history)
            assistant_blocks = payload[1]["content"]
            thinking_blocks = [block for block in assistant_blocks if block.get("type") == "thinking"]
            text_blocks = [block for block in assistant_blocks if block.get("type") == "text"]
            if expected_policy == "signed_thinking":
                assert len(thinking_blocks) == 1
                assert thinking_blocks[0]["signature"] == "claude_sig"
            elif expected_policy == "downgrade_to_text":
                assert thinking_blocks == []
                assert len(text_blocks) >= 2
                assert text_blocks[0]["text"] in {"completion reasoning", "responses reasoning summary", "gemini thought"}
            else:
                raise AssertionError(f"Unexpected claude policy {expected_policy}")
            return

        if target_name == "gemini":
            payload, _system_instruction = GeminiMessagesAdapter().to_vendor_format(history)
            assistant_parts = payload[1]["parts"]
            if expected_policy == "thought_part":
                assert assistant_parts[0].get("thought") is True
                assert "thoughtSignature" not in assistant_parts[0]
            elif expected_policy == "thought_part_with_signature":
                assert assistant_parts[0].get("thought") is True
                assert assistant_parts[0].get("thoughtSignature") == "gemini_sig"
            else:
                raise AssertionError(f"Unexpected gemini policy {expected_policy}")
            return

        raise AssertionError(f"Unexpected target {target_name}")

    @pytest.mark.parametrize(
        ("source_name", "expected_downgraded_reasoning_text"),
        [
            ("completion", "completion reasoning"),
            ("responses", "responses reasoning summary"),
            ("gemini", "gemini thought"),
        ],
    )
    def test_two_turn_claude_target_downgrades_unsigned_reasoning_to_text(self, source_name: str, expected_downgraded_reasoning_text: str):
        """Anthropic target must never emit unsigned thinking blocks from non-Anthropic sources."""
        history = _build_two_turn_history(source_name)

        _system, payload = AnthropicMessagesAdapter().to_vendor_format(history)
        assistant_blocks = payload[1]["content"]

        thinking_blocks = [block for block in assistant_blocks if block.get("type") == "thinking"]
        assert thinking_blocks == []

        text_blocks = [block for block in assistant_blocks if block.get("type") == "text"]
        assert text_blocks[0]["text"] == expected_downgraded_reasoning_text
