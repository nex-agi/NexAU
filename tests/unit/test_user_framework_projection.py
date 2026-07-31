"""Pure unit coverage for provider-facing USER/FRAMEWORK projection."""

from nexau.core.messages import ImageBlock, Message, Role, TextBlock
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload
from nexau.core.serializers.openai_responses import prepare_openai_responses_api_input
from nexau.core.serializers.user_projection import coalesce_user_shaped_messages


def test_projection_coalesces_user_and_framework_without_mutating_history() -> None:
    user = Message(
        role=Role.USER,
        content=[TextBlock(text="question"), ImageBlock(url="https://example.com/input.jpg")],
        metadata={"client": "canonical"},
    )
    framework = Message(role=Role.FRAMEWORK, content=[TextBlock(text="iteration reminder")])
    original_user = user.model_copy(deep=True)
    original_framework = framework.model_copy(deep=True)

    serialized = serialize_ump_to_openai_chat_payload([user, framework])

    assert serialized == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "question"},
                {"type": "image_url", "image_url": {"url": "https://example.com/input.jpg"}},
                {"type": "text", "text": "\n\n"},
                {"type": "text", "text": "iteration reminder"},
            ],
        }
    ]
    assert user == original_user
    assert framework == original_framework
    assert user.role == Role.USER
    assert framework.role == Role.FRAMEWORK


def test_projection_preserves_adjacent_users_when_framework_is_absent() -> None:
    first = Message.user("first")
    second = Message.user("second")

    projected = coalesce_user_shaped_messages([first, second])

    assert projected == [first, second]
    assert projected[0] is first
    assert projected[1] is second


def test_projection_preserves_assistant_boundary_and_standalone_framework() -> None:
    serialized = serialize_ump_to_openai_chat_payload(
        [
            Message.user("first"),
            Message(role=Role.FRAMEWORK, content=[TextBlock(text="first reminder")]),
            Message.assistant("answer"),
            Message(role=Role.FRAMEWORK, content=[TextBlock(text="next reminder")]),
        ]
    )

    assert serialized == [
        {"role": "user", "content": "first\n\nfirst reminder"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "next reminder"},
    ]


def test_empty_framework_is_canonical_but_has_no_provider_turn() -> None:
    first = Message.user("first")
    second = Message.user("second")
    empty_framework = Message(role=Role.FRAMEWORK, content=[TextBlock(text="")])

    projected = coalesce_user_shaped_messages([first, empty_framework, second])
    chat_payload = serialize_ump_to_openai_chat_payload([empty_framework])

    assert projected == [first, second]
    assert chat_payload == []
    assert empty_framework.role == Role.FRAMEWORK


def test_openai_responses_consumes_the_same_coalesced_provider_turn() -> None:
    chat_payload = serialize_ump_to_openai_chat_payload(
        [
            Message.user("question"),
            Message(role=Role.FRAMEWORK, content=[TextBlock(text="reminder")]),
        ]
    )

    response_items, instructions = prepare_openai_responses_api_input(chat_payload)

    assert instructions is None
    assert "question" in str(response_items)
    assert "reminder" in str(response_items)
