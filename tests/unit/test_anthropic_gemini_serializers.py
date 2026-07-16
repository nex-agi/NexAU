from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock
from nexau.core.serializers.anthropic_messages import (
    apply_anthropic_last_user_cache_control,
    serialize_ump_to_anthropic_messages_payload,
)
from nexau.core.serializers.gemini_messages import serialize_ump_to_gemini_messages_payload


def test_anthropic_serializer_drops_unsigned_reasoning_when_companion_content_exists() -> None:
    """Bug fix: unsigned reasoning is DROPPED (not demoted to text) when
    the message has any companion content. Pre-fix this demoted to text,
    causing the agent's internal reasoning to appear as the assistant's
    reply (the Bedrock claude-opus-4.x orphan thinking_delta symptom).
    """
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
    # Unsigned reasoning dropped; only signed thinking + text reply remain.
    assert len(blocks) == 2
    assert blocks[0] == {"type": "thinking", "thinking": "signed thinking", "signature": "sig_1"}
    assert blocks[1] == {"type": "text", "text": "answer"}
    # The unsigned reasoning string MUST NOT appear anywhere in outbound.
    assert "unsigned reasoning" not in str(convo)


def test_anthropic_serializer_stubs_unsigned_reasoning_only_message() -> None:
    """Edge case: when a message has ONLY unsigned reasoning and no
    companion content, emit a neutral `[reasoning omitted]` stub.

    Why stub instead of demoting the reasoning text to a `text` block:
      - Anthropic rejects messages with content=[], so we MUST emit
        something.
      - Demoting the reasoning to text leaks the agent's internal
        stream-of-consciousness into the next-turn LLM context as
        if it were the assistant's reply — re-introducing the exact
        bug Layer 3 was meant to fix, just in a smaller probability
        window.
      - The stub preserves Anthropic's user/assistant alternation
        without any leak.
    """
    _system_blocks, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(
                role=Role.ASSISTANT,
                content=[ReasoningBlock(text="reasoning only")],
            ),
        ]
    )

    assert convo[0]["content"] == [{"type": "text", "text": "[reasoning omitted]"}]
    assert "reasoning only" not in str(convo)


def test_anthropic_serializer_drops_unsigned_reasoning_when_tool_call_companion() -> None:
    """Tool-use companion counts as content — reasoning still dropped."""
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
    # Reasoning dropped; only tool_use remains.
    assert len(blocks) == 1
    assert blocks[0] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {"query": "weather"},
    }
    assert "choose tool" not in str(convo)


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


def test_anthropic_serializer_orders_all_tool_results_before_sibling_images() -> None:
    """Regression: parallel image tool results must not interleave.

    Anthropic only matches the leading run of `tool_result` blocks in the
    next user message: `[TR, IMG, TR, IMG]` makes every tool_use id after
    the first fail with "`tool_use` ids were found without `tool_result`
    blocks immediately after" (400). The flush must emit all tool results
    first, then the hoisted sibling images.
    """
    _system, convo = serialize_ump_to_anthropic_messages_payload(
        [
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content=[ImageBlock(url="https://example.com/a.jpg")],
                    ),
                ],
            ),
            Message(
                role=Role.TOOL,
                content=[
                    ToolResultBlock(
                        tool_use_id="call_2",
                        content=[TextBlock(text="caption"), ImageBlock(url="https://example.com/b.jpg")],
                    ),
                ],
            ),
        ]
    )

    blocks = convo[0]["content"]
    assert [block["type"] for block in blocks] == ["tool_result", "tool_result", "image", "image"]
    assert blocks[0]["tool_use_id"] == "call_1"
    assert blocks[1]["tool_use_id"] == "call_2"
    assert blocks[2]["source"]["url"] == "https://example.com/a.jpg"
    assert blocks[3]["source"]["url"] == "https://example.com/b.jpg"


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
