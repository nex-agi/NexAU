from typing import Any, cast

from nexau.core.adapters.anthropic_messages import anthropic_payload_from_legacy_openai_chat
from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock


def test_chat_completions_tool_result_images_are_injected_as_user_message() -> None:
    """Chat Completions tool messages support text only; images must be provided via user multimodal messages."""

    msgs = [
        Message.user("hi"),
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

    legacy: list[dict[str, Any]] = messages_to_legacy_openai_chat(msgs, tool_image_policy="inject_user_message")
    # Expect tool message contains only text (images replaced by placeholder)
    tool = next(m for m in legacy if m.get("role") == "tool")
    assert tool["tool_call_id"] == "call_1"
    assert isinstance(tool["content"], str)
    assert "<image>" in tool["content"]

    # And an extra user message carries the actual image_url parts
    injected_user = next(m for m in legacy if m.get("role") == "user" and isinstance(m.get("content"), list))
    parts = cast(list[dict[str, Any]], injected_user["content"])
    assert any(p.get("type") == "image_url" for p in parts)
    image_part = next(p for p in parts if p.get("type") == "image_url")
    assert image_part["image_url"]["url"] == "https://example.com/a.jpg"
    assert image_part["image_url"]["detail"] == "high"


def test_responses_policy_embeds_tool_result_images_in_tool_message_parts() -> None:
    """Responses input reconstruction may embed image_url parts inside the tool message content list."""

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

    legacy: list[dict[str, Any]] = messages_to_legacy_openai_chat(msgs, tool_image_policy="embed_in_tool_message")
    tool = next(m for m in legacy if m.get("role") == "tool")
    tool_content = cast(list[dict[str, Any]], tool["content"])
    assert any(p.get("type") == "image_url" for p in tool_content)


def test_anthropic_payload_converts_data_url_images_to_raw_base64() -> None:
    """Anthropic Messages API expects raw base64, not OpenAI-style data URLs."""

    msgs = [
        Message(
            role=Role.USER,
            content=[
                TextBlock(text="describe"),
                # Simulate a base64 string that contains newlines/whitespace (common when users copy/paste).
                ImageBlock(base64="QUJ\nD", mime_type="image/png"),
            ],
        ),
    ]

    legacy: list[dict[str, Any]] = messages_to_legacy_openai_chat(msgs)
    system, convo = anthropic_payload_from_legacy_openai_chat(legacy)

    assert system == []
    assert len(convo) == 1
    assert convo[0]["role"] == "user"

    content = convo[0]["content"]
    assert isinstance(content, list)
    content_list = cast(list[dict[str, Any]], content)
    image_block = next(item for item in content_list if item.get("type") == "image")
    source = cast(dict[str, Any], image_block["source"])
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert source["data"] == "QUJD"
