import pytest

from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock


def test_message_supports_legacy_dict_style_access_with_deprecation_warning() -> None:
    msg = Message.user("hi")

    with pytest.warns(DeprecationWarning):
        assert msg["role"] == "user"

    with pytest.warns(DeprecationWarning):
        assert msg["content"] == "hi"

    with pytest.warns(DeprecationWarning):
        assert msg.get("content") == "hi"

    with pytest.warns(DeprecationWarning):
        assert msg.get("missing_key", "default") == "default"

    with pytest.warns(DeprecationWarning), pytest.raises(KeyError):
        _ = msg["missing_key"]


def test_tool_message_legacy_access_exposes_tool_call_id_and_content() -> None:
    tool_msg = Message(role=Role.TOOL, content=[ToolResultBlock(tool_use_id="call_1", content="4")])

    with pytest.warns(DeprecationWarning):
        assert tool_msg["role"] == "tool"

    with pytest.warns(DeprecationWarning):
        assert tool_msg["tool_call_id"] == "call_1"

    with pytest.warns(DeprecationWarning):
        assert tool_msg["content"] == "4"


def test_tool_message_with_multimodal_tool_result_folds_images_as_placeholders() -> None:
    # When tool result contains only images, legacy tool-role messages still have text content via "<image>" placeholders.
    tool_msg = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_1",
                content=[
                    ImageBlock(url="https://example.com/a.png"),
                ],
            )
        ],
    )

    with pytest.warns(DeprecationWarning):
        assert tool_msg["tool_call_id"] == "call_1"

    with pytest.warns(DeprecationWarning):
        assert tool_msg["content"] == "<image>"


def test_tool_message_with_empty_multimodal_tool_result_uses_tool_output_fallback() -> None:
    tool_msg = Message(
        role=Role.TOOL,
        content=[
            ToolResultBlock(
                tool_use_id="call_1",
                content=[],
            )
        ],
    )

    with pytest.warns(DeprecationWarning):
        assert tool_msg["content"] == "<tool_output>"


def test_assistant_legacy_content_with_images_folds_tool_result_blocks_to_text_parts() -> None:
    # If the assistant message contains any images, legacy conversion uses structured `content` parts.
    msg = Message(
        role=Role.ASSISTANT,
        content=[
            TextBlock(text="a"),
            ToolResultBlock(
                tool_use_id="call_1",
                content=[
                    TextBlock(text="x"),
                    ImageBlock(url="https://example.com/tool.png"),
                ],
            ),
            ImageBlock(url="https://example.com/user.png", detail="high"),
        ],
    )

    with pytest.warns(DeprecationWarning):
        content = msg["content"]

    assert isinstance(content, list)
    # ToolResultBlock should have been folded into a text part with "<image>" placeholders.
    assert any(isinstance(p, dict) and p.get("type") == "text" and p.get("text") == "a" for p in content)
    assert any(isinstance(p, dict) and p.get("type") == "text" and p.get("text") == "x<image>" for p in content)
    # And the actual assistant image should be present as an image_url part.
    assert any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
