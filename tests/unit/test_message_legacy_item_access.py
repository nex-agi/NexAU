import pytest

from nexau.core.messages import Message, Role, ToolResultBlock


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
