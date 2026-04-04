from typing import Any, cast

from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload


def test_openai_chat_serializer_emits_reasoning_and_tool_calls() -> None:
    msgs = [
        Message.user("hello"),
        Message(
            role=Role.ASSISTANT,
            content=[
                ReasoningBlock(text="reasoning text", signature="sig_1", redacted_data="blob_1"),
                TextBlock(text="done"),
                ToolUseBlock(id="call_1", name="weather", input={"city": "Paris"}),
            ],
            metadata={"response_items": [{"type": "message", "id": "m1"}]},
        ),
    ]

    assert serialize_ump_to_openai_chat_payload(msgs) == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "done",
            "response_items": [{"type": "message", "id": "m1"}],
            "reasoning_content": "reasoning text",
            "reasoning_signature": "sig_1",
            "reasoning_redacted_data": "blob_1",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                },
            ],
        },
    ]


def test_openai_chat_serializer_matches_legacy_tool_image_injection_policy() -> None:
    msgs = [
        Message.user("show tool result"),
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[TextBlock(text="Here"), ImageBlock(url="https://example.com/a.jpg", detail="high")],
                ),
            ],
            metadata={"tool_name": "image_lookup"},
        ),
    ]

    serialized = serialize_ump_to_openai_chat_payload(msgs, tool_image_policy="inject_user_message")
    assert serialized == [
        {"role": "user", "content": "show tool result"},
        {"role": "tool", "tool_call_id": "call_1", "content": "Here<image>", "name": "image_lookup"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Images returned by tool call call_1:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg", "detail": "high"}},
            ],
            "name": "image_lookup",
        },
    ]

    injected_user = next(m for m in serialized if m.get("role") == "user" and isinstance(m.get("content"), list))
    parts = cast(list[dict[str, Any]], injected_user["content"])
    assert any(p.get("type") == "image_url" for p in parts)


def test_openai_chat_serializer_matches_legacy_tool_image_embed_policy() -> None:
    msgs = [
        Message(
            role=Role.TOOL,
            content=[
                ToolResultBlock(
                    tool_use_id="call_1",
                    content=[TextBlock(text="Here"), ImageBlock(url="https://example.com/a.jpg", detail="high")],
                ),
            ],
        ),
    ]

    serialized = serialize_ump_to_openai_chat_payload(msgs, tool_image_policy="embed_in_tool_message")
    assert serialized == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Here"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg", "detail": "high"}},
            ],
        },
    ]
