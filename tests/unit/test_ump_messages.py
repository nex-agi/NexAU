import logging
from typing import Any

from _pytest.logging import LogCaptureFixture

from nexau.archs.main_sub.execution.llm_caller import openai_to_anthropic_message
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat, messages_to_legacy_openai_chat
from nexau.core.messages import Role, TextBlock, ToolResultBlock, ToolUseBlock


def test_legacy_roundtrip_text_only() -> None:
    legacy: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    ump = messages_from_legacy_openai_chat(legacy)
    assert [m.role for m in ump] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert ump[1].get_text_content() == "Hello"

    roundtripped = messages_to_legacy_openai_chat(ump)
    # UMP adds ids; legacy conversion should preserve observable chat shape.
    assert roundtripped == legacy


def test_legacy_roundtrip_tool_call_and_result() -> None:
    legacy: list[dict[str, Any]] = [
        {"role": "user", "content": "What is 2+2? Use the calculator tool."},
        {
            "role": "assistant",
            "content": "Calling tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": '{"expr":"2+2"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4"},
        {"role": "assistant", "content": "Answer: 4"},
    ]

    ump = messages_from_legacy_openai_chat(legacy)
    assert ump[1].role == Role.ASSISTANT
    assert any(
        isinstance(b, ToolUseBlock) and b.id == "call_1" and b.name == "calculator" and b.input.get("expr") == "2+2" for b in ump[1].content
    )

    assert ump[2].role == Role.TOOL
    assert any(isinstance(b, ToolResultBlock) and b.tool_use_id == "call_1" and b.content == "4" for b in ump[2].content)

    roundtripped = messages_to_legacy_openai_chat(ump)
    assert roundtripped == legacy


def test_openai_to_anthropic_message_uses_blocks() -> None:
    legacy: list[dict[str, Any]] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "tool time",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calculator", "arguments": {"expr": "2+2"}},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4"},
    ]

    system, messages = openai_to_anthropic_message(legacy)
    assert system == [{"type": "text", "text": "sys"}]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "hi"}]

    assert messages[1]["role"] == "assistant"
    assistant_blocks = messages[1]["content"]
    assert any(b.get("type") == "text" and b.get("text") == "tool time" for b in assistant_blocks)
    assert any(
        b.get("type") == "tool_use" and b.get("id") == "call_1" and b.get("name") == "calculator" and b.get("input") == {"expr": "2+2"}
        for b in assistant_blocks
    )

    assert messages[2]["role"] == "tool"
    assert messages[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "4", "is_error": False},
    ]


def test_legacy_structured_content_list_is_preserved_as_text_blocks() -> None:
    legacy: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "c"}]},
    ]

    ump = messages_from_legacy_openai_chat(legacy)
    assert [type(b) for b in ump[0].content] == [TextBlock, TextBlock]
    assert ump[0].get_text_content() == "ab"

    roundtripped = messages_to_legacy_openai_chat(ump)
    assert roundtripped == [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "c"},
    ]


def test_legacy_roundtrip_structured_content_with_image_url() -> None:
    legacy: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "see"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "text", "text": "ok"},
            ],
        },
    ]

    ump = messages_from_legacy_openai_chat(legacy)
    assert ump[0].role == Role.USER
    assert [b.type for b in ump[0].content] == ["text", "image", "text"]

    roundtripped = messages_to_legacy_openai_chat(ump)
    assert roundtripped == legacy


def test_unknown_role_logs_warning_and_coerces_to_user(caplog: LogCaptureFixture) -> None:
    legacy: list[dict[str, Any]] = [{"role": "developer", "content": "hi"}]

    with caplog.at_level(logging.WARNING):
        ump = messages_from_legacy_openai_chat(legacy)

    assert ump[0].role == Role.USER
    assert any("Unknown role" in rec.getMessage() and "coercing to user" in rec.getMessage() for rec in caplog.records)
