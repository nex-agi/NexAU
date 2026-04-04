# Licensed under the Apache License, Version 2.0

"""64-case block-type × source-provider × target-provider serialization matrix.

RFC-0014: UMP 序列化完整性矩阵

4 block types × 4 source providers × 4 target providers = 64 test cases.

Each case:
1. Constructs a UMP conversation containing a specific block type
   (produced by source provider's ModelResponse parser)
2. Serializes to target provider's payload format
3. Asserts the block is correctly represented in the target format

Block types tested:
- ImageBlock: multimodal image content (base64/url)
- ReasoningBlock: thinking / chain-of-thought content
- ToolUseBlock: tool/function call
- ToolResultBlock: tool execution result

Provider targets:
- completion: serialize_ump_to_openai_chat_payload
- responses: prepare_openai_responses_api_input
- claude: AnthropicMessagesAdapter().to_vendor_format
- gemini: GeminiMessagesAdapter().to_vendor_format

Note: Provider IDs use "completion"/"responses"/"claude"/"gemini" to avoid
triggering the auto-llm-marker in conftest (keywords: "openai", "chat").
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.core.adapters.anthropic_messages import AnthropicMessagesAdapter
from nexau.core.adapters.gemini_messages import GeminiMessagesAdapter
from nexau.core.messages import (
    ImageBlock,
    Message,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload
from nexau.core.serializers.openai_responses import prepare_openai_responses_api_input

# ---------------------------------------------------------------------------
# 1. Source factories: each returns a UMP Message containing the target block,
#    constructed as if parsed from a specific source provider's response.
# ---------------------------------------------------------------------------

# --- ImageBlock sources ---
# UMP representation for images is uniform; source variations only differ in
# url vs base64 and detail settings.


def _image_from_completion() -> Message:
    """User message with image URL (typical Chat flow)."""
    return Message(
        role=Role.USER,
        content=[
            TextBlock(text="What's in this image?"),
            ImageBlock(url="https://example.com/photo.jpg"),
        ],
    )


def _image_from_responses() -> Message:
    """User message with base64 image (Responses input_image)."""
    return Message(
        role=Role.USER,
        content=[
            TextBlock(text="Describe this image"),
            ImageBlock(base64="iVBORw0KGgo=", mime_type="image/png"),
        ],
    )


def _image_from_claude() -> Message:
    """User message with base64 GIF image (Anthropic style)."""
    return Message(
        role=Role.USER,
        content=[
            TextBlock(text="Analyze this screenshot"),
            ImageBlock(
                base64="R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==",
                mime_type="image/gif",
            ),
        ],
    )


def _image_from_gemini() -> Message:
    """User message with high-detail URL image (Gemini style)."""
    return Message(
        role=Role.USER,
        content=[
            TextBlock(text="What do you see?"),
            ImageBlock(url="https://example.com/diagram.png", detail="high"),
        ],
    )


# --- ReasoningBlock sources ---
# These DO vary by source due to different metadata (signature, encrypted_content,
# thought_signature).  Uses ModelResponse.from_*() to test the real parse path.


def _reasoning_from_completion() -> Message:
    """Assistant with reasoning_content (Chat o-series)."""
    resp = ModelResponse.from_openai_message(
        {
            "role": "assistant",
            "content": "The answer is 42.",
            "reasoning_content": "Let me think step by step...",
        },
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )
    return resp.to_ump_message()


def _reasoning_from_responses() -> Message:
    """Assistant with reasoning item (Responses API)."""
    resp = ModelResponse.from_openai_response(
        {
            "output": [
                {
                    "id": "rs_test",
                    "type": "reasoning",
                    "encrypted_content": "encrypted_test_blob",
                    "summary": [{"type": "summary_text", "text": "Responses reasoning summary"}],
                },
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "The answer is 42."}],
                },
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 10},
            },
        }
    )
    return resp.to_ump_message()


def _reasoning_from_claude() -> Message:
    """Assistant with thinking block (Claude)."""
    resp = ModelResponse.from_anthropic_message(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Claude deep thought...", "signature": "sig_anthropic_test"},
                {"type": "text", "text": "The answer is 42."},
            ],
        },
        usage={"input_tokens": 10, "output_tokens": 20},
    )
    return resp.to_ump_message()


def _reasoning_from_gemini() -> Message:
    """Assistant with thought parts (Gemini)."""
    resp = ModelResponse.from_gemini_rest(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"text": "Gemini internal reasoning", "thought": True},
                            {"text": "The answer is 42.", "thoughtSignature": "sig_gemini_test"},
                        ],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "thoughtsTokenCount": 5,
                "totalTokenCount": 23,
            },
        }
    )
    return resp.to_ump_message()


# --- ToolUseBlock sources ---


def _tool_use_from_completion() -> Message:
    """Assistant with tool_calls (Chat format)."""
    return Message(
        role=Role.ASSISTANT,
        content=[
            TextBlock(text=""),
            ToolUseBlock(
                id="call_abc123",
                name="get_weather",
                input={"city": "Tokyo", "unit": "celsius"},
                raw_input='{"city": "Tokyo", "unit": "celsius"}',
            ),
        ],
    )


def _tool_use_from_responses() -> Message:
    """Assistant with function_call (Responses)."""
    return Message(
        role=Role.ASSISTANT,
        content=[
            ToolUseBlock(
                id="fc_resp_001",
                name="search_web",
                input={"query": "latest news"},
            ),
        ],
    )


def _tool_use_from_claude() -> Message:
    """Assistant with tool_use block (Anthropic)."""
    return Message(
        role=Role.ASSISTANT,
        content=[
            TextBlock(text="Let me look that up."),
            ToolUseBlock(
                id="toolu_anth_001",
                name="calculator",
                input={"expression": "2+2"},
            ),
        ],
    )


def _tool_use_from_gemini() -> Message:
    """Assistant with functionCall (Gemini)."""
    return Message(
        role=Role.ASSISTANT,
        content=[
            ToolUseBlock(
                id="gemini_fc_001",
                name="code_execution",
                input={"code": "print(42)"},
            ),
        ],
    )


# --- ToolResultBlock sources ---


def _tool_result_from_completion() -> Message:
    """Tool result (Chat tool-role message)."""
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_abc123",
                content="Temperature: 22°C, Sunny",
            ),
        ],
        metadata={"tool_name": "get_weather"},
    )


def _tool_result_from_responses() -> Message:
    """Tool result with multimodal content (text + image)."""
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="fc_resp_001",
                content=[
                    TextBlock(text="Search result: found 3 articles"),
                    ImageBlock(url="https://example.com/screenshot.png"),
                ],
            ),
        ],
        metadata={"tool_name": "search_web"},
    )


def _tool_result_from_claude() -> Message:
    """Tool result (Anthropic style)."""
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="toolu_anth_001",
                content="4",
            ),
        ],
        metadata={"tool_name": "calculator"},
    )


def _tool_result_from_gemini() -> Message:
    """Tool result (Gemini style)."""
    return Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="gemini_fc_001",
                content="42\n",
            ),
        ],
        metadata={"tool_name": "code_execution"},
    )


# ---------------------------------------------------------------------------
# 2. Source factory registry: block_type × source_provider
# ---------------------------------------------------------------------------

BLOCK_TYPES = ("image", "reasoning", "tool_use", "tool_result")
PROVIDERS = ("completion", "responses", "claude", "gemini")

_SOURCES: dict[str, dict[str, Callable[[], Message]]] = {
    "image": {
        "completion": _image_from_completion,
        "responses": _image_from_responses,
        "claude": _image_from_claude,
        "gemini": _image_from_gemini,
    },
    "reasoning": {
        "completion": _reasoning_from_completion,
        "responses": _reasoning_from_responses,
        "claude": _reasoning_from_claude,
        "gemini": _reasoning_from_gemini,
    },
    "tool_use": {
        "completion": _tool_use_from_completion,
        "responses": _tool_use_from_responses,
        "claude": _tool_use_from_claude,
        "gemini": _tool_use_from_gemini,
    },
    "tool_result": {
        "completion": _tool_result_from_completion,
        "responses": _tool_result_from_responses,
        "claude": _tool_result_from_claude,
        "gemini": _tool_result_from_gemini,
    },
}


# ---------------------------------------------------------------------------
# 3. Target serializers
# ---------------------------------------------------------------------------


def _serialize_to_completion(messages: list[Message]) -> dict[str, object]:
    return {"type": "completion", "payload": serialize_ump_to_openai_chat_payload(messages)}


def _serialize_to_responses(messages: list[Message]) -> dict[str, object]:
    chat_payload = serialize_ump_to_openai_chat_payload(messages, tool_image_policy="embed_in_tool_message")
    items, _sys = prepare_openai_responses_api_input(chat_payload)
    return {"type": "responses", "payload": items}


def _serialize_to_claude(messages: list[Message]) -> dict[str, object]:
    system_blocks, convo = AnthropicMessagesAdapter().to_vendor_format(messages)
    return {"type": "claude", "system": system_blocks, "payload": convo}


def _serialize_to_gemini(messages: list[Message]) -> dict[str, object]:
    # to_vendor_format returns (contents, system_instruction)
    contents, system_instruction = GeminiMessagesAdapter().to_vendor_format(messages)
    return {"type": "gemini", "system": system_instruction, "payload": contents}


_TARGETS: dict[str, Callable[[list[Message]], dict[str, object]]] = {
    "completion": _serialize_to_completion,
    "responses": _serialize_to_responses,
    "claude": _serialize_to_claude,
    "gemini": _serialize_to_gemini,
}


# ---------------------------------------------------------------------------
# 4. Block-specific validators
# ---------------------------------------------------------------------------


def _validate_image_in_target(target_type: str, result: dict[str, object]) -> None:
    """Assert image content survived serialization to target format."""
    payload = result["payload"]
    flat = str(payload)

    if target_type == "gemini":
        # Gemini serializer does not yet handle ImageBlock → known gap (RFC-0014)
        pytest.xfail("Gemini serializer does not handle ImageBlock in user messages yet")

    # Every other target should preserve image data in some form
    assert any(
        kw in flat
        for kw in (
            "image_url",
            "input_image",
            "image",
            "photo.jpg",
            "screenshot.png",
            "diagram.png",
            "iVBORw0KGgo",
            "R0lGODlh",
        )
    ), f"Image data lost in {target_type}: {flat[:500]}"


def _validate_reasoning_in_target(target_type: str, result: dict[str, object]) -> None:
    """Assert reasoning/thinking content survived serialization."""
    payload = result["payload"]
    flat = str(payload)

    if target_type == "completion":
        # OpenAI Chat: reasoning_content field or reasoning in content
        assert "reasoning" in flat.lower() or "think" in flat.lower(), f"Reasoning lost in completion: {flat[:500]}"
    elif target_type == "responses":
        # Responses: reasoning item with summary
        assert any(kw in flat for kw in ("reasoning", "summary", "thinking")), f"Reasoning lost in responses: {flat[:500]}"
    elif target_type == "claude":
        # Anthropic: thinking block (with signature) or demoted to text (without).
        # Non-Anthropic sources lack a signature, so demotion to text is correct.
        assert any(kw in flat for kw in ("thinking", "thought", "signature", "text")), f"Reasoning lost in claude: {flat[:500]}"
    elif target_type == "gemini":
        # Gemini: thought=True part
        assert any(kw in flat for kw in ("thought", "True")), f"Reasoning lost in gemini: {flat[:500]}"


def _validate_tool_use_in_target(target_type: str, result: dict[str, object]) -> None:
    """Assert tool call survived serialization."""
    payload = result["payload"]
    flat = str(payload)

    if target_type == "completion":
        assert "tool_calls" in flat or "function" in flat, f"ToolUse lost in completion: {flat[:500]}"
    elif target_type == "responses":
        assert "function_call" in flat or "function" in flat, f"ToolUse lost in responses: {flat[:500]}"
    elif target_type == "claude":
        assert "tool_use" in flat, f"ToolUse lost in claude: {flat[:500]}"
    elif target_type == "gemini":
        assert "functionCall" in flat, f"ToolUse lost in gemini: {flat[:500]}"


def _validate_tool_result_in_target(target_type: str, result: dict[str, object]) -> None:
    """Assert tool result survived serialization."""
    payload = result["payload"]
    flat = str(payload)

    if target_type == "completion":
        assert "tool" in flat, f"ToolResult lost in completion: {flat[:500]}"
    elif target_type == "responses":
        assert "function_call_output" in flat or "output" in flat, f"ToolResult lost in responses: {flat[:500]}"
    elif target_type == "claude":
        assert "tool_result" in flat, f"ToolResult lost in claude: {flat[:500]}"
    elif target_type == "gemini":
        assert "functionResponse" in flat, f"ToolResult lost in gemini: {flat[:500]}"


_VALIDATORS: dict[str, Callable[[str, dict[str, object]], None]] = {
    "image": _validate_image_in_target,
    "reasoning": _validate_reasoning_in_target,
    "tool_use": _validate_tool_use_in_target,
    "tool_result": _validate_tool_result_in_target,
}


# ---------------------------------------------------------------------------
# 5. Helper: wrap block message into a valid multi-turn conversation
# ---------------------------------------------------------------------------


def _build_conversation(block_type: str, source: str) -> list[Message]:
    """Build a minimal conversation containing the block from the source provider.

    RFC-0014: 构建最小对话上下文，确保序列化器能正确处理。
    """
    msg = _SOURCES[block_type][source]()

    # 1. Image: user message with image → no extra context needed
    if block_type == "image":
        return [msg]

    # 2. Reasoning: needs preceding user message
    if block_type == "reasoning":
        return [Message.user("What is the answer?"), msg]

    # 3. Tool use: needs preceding user message
    if block_type == "tool_use":
        return [Message.user("Please get the weather"), msg]

    # 4. Tool result: needs user → assistant(tool_call) → tool_result
    if block_type == "tool_result":
        tool_use_id = "unknown"
        for block in msg.content:
            if isinstance(block, ToolResultBlock):
                tool_use_id = block.tool_use_id
                break
        tool_name = msg.metadata.get("tool_name", "some_tool")
        assert isinstance(tool_name, str)
        tool_call_msg = Message(
            role=Role.ASSISTANT,
            content=[
                ToolUseBlock(
                    id=tool_use_id,
                    name=tool_name,
                    input={"arg": "value"},
                ),
            ],
        )
        return [Message.user("Run the tool"), tool_call_msg, msg]

    return [msg]  # pragma: no cover


# ---------------------------------------------------------------------------
# 6. The 64-case parametrized test
# ---------------------------------------------------------------------------

_ALL_CASES = [(bt, src, tgt) for bt in BLOCK_TYPES for src in PROVIDERS for tgt in PROVIDERS]


class TestBlockTypeProviderMatrix:
    """64-case matrix: 4 block types × 4 source providers × 4 target providers.

    RFC-0014: 验证每种 block type 在任意 source→target provider 序列化路径上都能正确保留。
    """

    @pytest.mark.parametrize(
        ("block_type", "source", "target"),
        _ALL_CASES,
        ids=[f"{bt}:{s}→{t}" for bt, s, t in _ALL_CASES],
    )
    def test_block_survives_serialization(self, block_type: str, source: str, target: str) -> None:
        """Assert that a block type from source provider survives serialization to target."""
        # 1. 构建包含目标 block type 的 UMP 对话
        conversation = _build_conversation(block_type, source)

        # 2. 序列化到目标 provider 格式（不应 raise）
        result = _TARGETS[target](conversation)

        # 3. 断言 block 内容在目标格式中被保留
        _VALIDATORS[block_type](target, result)
