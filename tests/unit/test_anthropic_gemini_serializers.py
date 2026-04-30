from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock
from nexau.core.serializers.anthropic_messages import (
    apply_anthropic_last_user_cache_control,
    serialize_ump_to_anthropic_messages_payload,
)
from nexau.core.serializers.gemini_messages import serialize_ump_to_gemini_messages_payload


def test_anthropic_serializer_downgrades_unsigned_reasoning_with_companion_content_and_keeps_signed_thinking() -> None:
    system_blocks, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(role=Role.SYSTEM, content=[TextBlock(text="sys")], metadata={"cache": True}),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="unsigned reasoning"),
                    ReasoningBlock(text="signed thinking", signature="sig_1"),
                    TextBlock(text="answer"),
                ],
            ),
        ]
    )

    assert system_blocks == [{"type": "text", "text": "sys", "_cache": True}]
    blocks = convo[0]["content"]
    assert blocks[0] == {"type": "text", "text": "unsigned reasoning"}
    assert blocks[1] == {"type": "thinking", "thinking": "signed thinking", "signature": "sig_1"}
    assert blocks[2] == {"type": "text", "text": "answer"}


def test_anthropic_serializer_downgrades_unsigned_reasoning_only_message() -> None:
    _system_blocks, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(
                role=Role.ASSISTANT,
                content=[ReasoningBlock(text="reasoning only")],
            ),
        ]
    )

    assert convo[0]["content"] == [{"type": "text", "text": "reasoning only"}]


def test_anthropic_serializer_downgrades_unsigned_reasoning_with_tool_call() -> None:
    _system_blocks, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="choose tool"),
                    ToolUseBlock(id="call_1", name="lookup", input={"query": "weather"}),
                ],
            ),
        ]
    )

    blocks = convo[0]["content"]
    assert blocks[0] == {"type": "text", "text": "choose tool"}
    assert blocks[1] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {"query": "weather"},
    }


def test_anthropic_serializer_allows_unsigned_thinking_when_configured() -> None:
    _system_blocks, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="choose tool"),
                    ToolUseBlock(id="call_1", name="lookup", input={"query": "weather"}),
                ],
            ),
        ],
        allow_unsigned_thinking=True,
    )

    blocks = convo[0]["content"]
    assert blocks[0] == {"type": "thinking", "thinking": "choose tool"}
    assert "signature" not in blocks[0]
    assert blocks[1]["type"] == "tool_use"


def test_anthropic_serializer_splits_tool_result_images_and_applies_cache_control() -> None:
    _system, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content=[TextBlock(text="caption"), ImageBlock(url="https://example.com/a.jpg")],
                    ),
                ],
            ),
            Message(role=Role.USER, content=[TextBlock(text="next")]),
        ]
    )

    adjusted = apply_anthropic_last_user_cache_control(convo, system_cache_control_ttl="5m")
    assert adjusted[0]["role"] == "user"
    assert adjusted[0]["content"][0]["type"] == "tool_result"
    assert adjusted[0]["content"][1] == {"type": "image", "source": {"type": "url", "url": "https://example.com/a.jpg"}}
    assert adjusted[1]["role"] == "user"
    assert adjusted[1]["content"][0] == {
        "type": "text",
        "text": "next",
        "cache_control": {"type": "ephemeral", "ttl": "5m"},
    }


def test_gemini_serializer_emits_thought_signature_and_function_response() -> None:
    contents, system_instruction = serialize_ump_to_gemini_messages_payload(
        [
            Message(role=Role.SYSTEM, content=[TextBlock(text="sys")]),
            Message(role=Role.USER, content=[TextBlock(text="question")]),
            Message(
                role=Role.ASSISTANT,
                content=[
                    ReasoningBlock(text="gemini thought"),
                    ToolUseBlock(id="call_1", name="weather", input={"city": "Paris"}),
                ],
                metadata={"thought_signature": "gemini_sig"},
            ),
            Message(role=Role.TOOL, content=[ToolResultBlock(tool_use_id="call_1", content="sunny")]),
        ]
    )

    assert system_instruction == {"parts": [{"text": "sys"}]}
    assert contents[0] == {"role": "user", "parts": [{"text": "question"}]}
    assistant_parts = contents[1]["parts"]
    assert assistant_parts[0] == {"text": "gemini thought", "thought": True}
    assert assistant_parts[1]["functionCall"] == {"name": "weather", "args": {"city": "Paris"}}
    assert assistant_parts[1]["thoughtSignature"] == "gemini_sig"
    assert contents[2] == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "weather", "response": {"result": "sunny"}}}],
    }
