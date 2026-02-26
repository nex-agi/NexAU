# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Anthropic event aggregator."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from anthropic.types import (
    InputJSONDelta,
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta

from nexau.archs.llm.llm_aggregators import AnthropicEventAggregator
from nexau.archs.llm.llm_aggregators.events import (
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)


def _make_message_start(message_id: str = "msg_01XYZ") -> RawMessageStartEvent:
    """Helper to build a RawMessageStartEvent."""
    return RawMessageStartEvent(
        type="message_start",
        message=Message(
            id=message_id,
            type="message",
            role="assistant",
            content=[],
            model="claude-sonnet-4-20250514",
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=10, output_tokens=0),
        ),
    )


def _make_text_block_start(index: int = 0) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=TextBlock(type="text", text=""),
    )


def _make_tool_use_block_start(
    index: int = 0,
    tool_id: str = "toolu_01ABC",
    name: str = "read_file",
) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=ToolUseBlock(type="tool_use", id=tool_id, name=name, input={}),
    )


def _make_thinking_block_start(index: int = 0) -> RawContentBlockStartEvent:
    return RawContentBlockStartEvent(
        type="content_block_start",
        index=index,
        content_block=ThinkingBlock(type="thinking", thinking="", signature=""),
    )


def _make_text_delta(index: int = 0, text: str = "Hello") -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=TextDelta(type="text_delta", text=text),
    )


def _make_input_json_delta(
    index: int = 0,
    partial_json: str = '{"path":',
) -> RawContentBlockDeltaEvent:
    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=InputJSONDelta(type="input_json_delta", partial_json=partial_json),
    )


def _make_thinking_delta(index: int = 0, thinking: str = "Let me think...") -> RawContentBlockDeltaEvent:
    from anthropic.types import ThinkingDelta

    return RawContentBlockDeltaEvent(
        type="content_block_delta",
        index=index,
        delta=ThinkingDelta(type="thinking_delta", thinking=thinking),
    )


def _make_block_stop(index: int = 0) -> RawContentBlockStopEvent:
    return RawContentBlockStopEvent(type="content_block_stop", index=index)


def _make_message_delta() -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent(
        type="message_delta",
        delta=Delta(stop_reason="end_turn", stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=42),
    )


def _make_message_stop() -> RawMessageStopEvent:
    return RawMessageStopEvent(type="message_stop")


class TestAnthropicEventAggregatorInit:
    """Tests for aggregator initialization and basic lifecycle."""

    def test_initialization(self):
        on_event = Mock()
        agg = AnthropicEventAggregator(on_event=on_event, run_id="run_1")
        assert agg._run_id == "run_1"
        assert agg._message_id == ""
        assert not agg._started
        assert agg._block_types == {}
        assert agg._tool_ids == {}

    def test_build_returns_none(self):
        agg = AnthropicEventAggregator(on_event=Mock(), run_id="run_1")
        assert agg.build() is None

    def test_clear_resets_all_state(self):
        on_event = Mock()
        agg = AnthropicEventAggregator(on_event=on_event, run_id="run_1")

        agg.aggregate(_make_message_start("msg_clear"))
        agg.aggregate(_make_tool_use_block_start(0, "tool_1", "bash"))
        agg.aggregate(_make_thinking_block_start(1))

        agg.clear()

        assert agg._message_id == ""
        assert not agg._started
        assert agg._block_types == {}
        assert agg._tool_ids == {}
        assert agg._tool_names == {}
        assert agg._tool_args == {}
        assert agg._tool_started == {}
        assert agg._tool_ended == {}
        assert agg._thinking_ids == {}


class TestTextMessageFlow:
    """Tests for plain text streaming (no tool calls, no thinking)."""

    def test_simple_text_response(self):
        """message_start → content_block_start(text) → text_delta × N → block_stop → message_stop."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_txt")

        agg.aggregate(_make_message_start("msg_txt"))
        agg.aggregate(_make_text_block_start(0))
        agg.aggregate(_make_text_delta(0, "Hello"))
        agg.aggregate(_make_text_delta(0, " world"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        types = [type(e).__name__ for e in events]
        assert types == [
            "TextMessageStartEvent",
            "TextMessageContentEvent",
            "TextMessageContentEvent",
            "TextMessageEndEvent",
        ]

        start = events[0]
        assert isinstance(start, TextMessageStartEvent)
        assert start.message_id == "msg_txt"
        assert start.role == "assistant"
        assert start.run_id == "run_txt"

        assert isinstance(events[1], TextMessageContentEvent)
        assert events[1].delta == "Hello"
        assert events[1].message_id == "msg_txt"

        assert isinstance(events[2], TextMessageContentEvent)
        assert events[2].delta == " world"

        end = events[3]
        assert isinstance(end, TextMessageEndEvent)
        assert end.message_id == "msg_txt"

    def test_message_start_emitted_only_once(self):
        """Duplicate RawMessageStartEvent should not emit a second TextMessageStartEvent."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_dup")

        agg.aggregate(_make_message_start("msg_dup"))
        agg.aggregate(_make_message_start("msg_dup"))

        start_events = [e for e in events if isinstance(e, TextMessageStartEvent)]
        assert len(start_events) == 1


class TestToolCallFlow:
    """Tests for tool_use content blocks."""

    def test_single_tool_call(self):
        """Full lifecycle of a single tool call."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_tool")

        agg.aggregate(_make_message_start("msg_tool"))
        agg.aggregate(_make_tool_use_block_start(0, "toolu_01", "read_file"))
        agg.aggregate(_make_input_json_delta(0, '{"path":'))
        agg.aggregate(_make_input_json_delta(0, ' "main.py"}'))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        types = [type(e).__name__ for e in events]
        assert types == [
            "TextMessageStartEvent",
            "ToolCallStartEvent",
            "ToolCallArgsEvent",
            "ToolCallArgsEvent",
            "ToolCallEndEvent",
            "TextMessageEndEvent",
        ]

        tool_start = events[1]
        assert isinstance(tool_start, ToolCallStartEvent)
        assert tool_start.tool_call_id == "toolu_01"
        assert tool_start.tool_call_name == "read_file"
        assert tool_start.parent_message_id == "msg_tool"

        args1 = events[2]
        assert isinstance(args1, ToolCallArgsEvent)
        assert args1.tool_call_id == "toolu_01"
        assert args1.delta == '{"path":'

        args2 = events[3]
        assert isinstance(args2, ToolCallArgsEvent)
        assert args2.delta == ' "main.py"}'

        tool_end = events[4]
        assert isinstance(tool_end, ToolCallEndEvent)
        assert tool_end.tool_call_id == "toolu_01"

    def test_tool_args_accumulate(self):
        """_tool_args accumulates fragments across deltas."""
        agg = AnthropicEventAggregator(on_event=Mock(), run_id="run_acc")

        agg.aggregate(_make_message_start("msg_acc"))
        agg.aggregate(_make_tool_use_block_start(0, "toolu_acc", "bash"))
        agg.aggregate(_make_input_json_delta(0, '{"cmd":'))
        agg.aggregate(_make_input_json_delta(0, ' "ls"}'))

        assert agg._tool_args[0] == '{"cmd": "ls"}'

    def test_multiple_tool_calls(self):
        """Two tool_use blocks in the same message."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_multi")

        agg.aggregate(_make_message_start("msg_multi"))

        agg.aggregate(_make_tool_use_block_start(0, "toolu_A", "read_file"))
        agg.aggregate(_make_input_json_delta(0, '{"path": "a.py"}'))
        agg.aggregate(_make_block_stop(0))

        agg.aggregate(_make_tool_use_block_start(1, "toolu_B", "write_file"))
        agg.aggregate(_make_input_json_delta(1, '{"path": "b.py"}'))
        agg.aggregate(_make_block_stop(1))

        agg.aggregate(_make_message_stop())

        tool_starts = [e for e in events if isinstance(e, ToolCallStartEvent)]
        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_starts) == 2
        assert len(tool_ends) == 2
        assert tool_starts[0].tool_call_id == "toolu_A"
        assert tool_starts[1].tool_call_id == "toolu_B"

    def test_tool_end_not_duplicated_on_message_stop(self):
        """If block_stop already emitted ToolCallEndEvent, message_stop should not duplicate it."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_nodup")

        agg.aggregate(_make_message_start("msg_nodup"))
        agg.aggregate(_make_tool_use_block_start(0, "toolu_dup", "bash"))
        agg.aggregate(_make_input_json_delta(0, "{}"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_ends) == 1

    def test_tool_end_emitted_on_message_stop_if_block_stop_missing(self):
        """If block_stop was never received, message_stop should still emit ToolCallEndEvent."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_forced")

        agg.aggregate(_make_message_start("msg_forced"))
        agg.aggregate(_make_tool_use_block_start(0, "toolu_forced", "bash"))
        agg.aggregate(_make_input_json_delta(0, "{}"))
        # Skip block_stop; go straight to message_stop
        agg.aggregate(_make_message_stop())

        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_ends) == 1
        assert tool_ends[0].tool_call_id == "toolu_forced"


class TestThinkingFlow:
    """Tests for thinking content blocks (extended thinking)."""

    def test_thinking_block_lifecycle(self):
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_think")

        agg.aggregate(_make_message_start("msg_think"))
        agg.aggregate(_make_thinking_block_start(0))
        agg.aggregate(_make_thinking_delta(0, "Step 1"))
        agg.aggregate(_make_thinking_delta(0, " then step 2"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        types = [type(e).__name__ for e in events]
        assert types == [
            "TextMessageStartEvent",
            "ThinkingTextMessageStartEvent",
            "ThinkingTextMessageContentEvent",
            "ThinkingTextMessageContentEvent",
            "ThinkingTextMessageEndEvent",
            "TextMessageEndEvent",
        ]

        think_start = events[1]
        assert isinstance(think_start, ThinkingTextMessageStartEvent)
        assert think_start.parent_message_id == "msg_think"
        assert think_start.run_id == "run_think"
        assert think_start.thinking_message_id  # non-empty UUID

        think_content1 = events[2]
        assert isinstance(think_content1, ThinkingTextMessageContentEvent)
        assert think_content1.delta == "Step 1"
        assert think_content1.thinking_message_id == think_start.thinking_message_id

        think_end = events[4]
        assert isinstance(think_end, ThinkingTextMessageEndEvent)
        assert think_end.thinking_message_id == think_start.thinking_message_id


class TestMixedContentBlocks:
    """Tests for responses with thinking + text + tool_use in a single message."""

    def test_thinking_then_text(self):
        """Thinking block followed by a text block."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_mix1")

        agg.aggregate(_make_message_start("msg_mix1"))

        agg.aggregate(_make_thinking_block_start(0))
        agg.aggregate(_make_thinking_delta(0, "hmm"))
        agg.aggregate(_make_block_stop(0))

        agg.aggregate(_make_text_block_start(1))
        agg.aggregate(_make_text_delta(1, "Result"))
        agg.aggregate(_make_block_stop(1))

        agg.aggregate(_make_message_stop())

        types = [type(e).__name__ for e in events]
        assert "ThinkingTextMessageStartEvent" in types
        assert "ThinkingTextMessageEndEvent" in types
        assert "TextMessageContentEvent" in types
        assert types[-1] == "TextMessageEndEvent"

    def test_thinking_then_tool_call(self):
        """Thinking block followed by a tool_use block."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_mix2")

        agg.aggregate(_make_message_start("msg_mix2"))

        agg.aggregate(_make_thinking_block_start(0))
        agg.aggregate(_make_thinking_delta(0, "I should call a tool"))
        agg.aggregate(_make_block_stop(0))

        agg.aggregate(_make_tool_use_block_start(1, "toolu_mix", "bash"))
        agg.aggregate(_make_input_json_delta(1, '{"cmd": "ls"}'))
        agg.aggregate(_make_block_stop(1))

        agg.aggregate(_make_message_stop())

        types = [type(e).__name__ for e in events]
        assert "ThinkingTextMessageStartEvent" in types
        assert "ThinkingTextMessageEndEvent" in types
        assert "ToolCallStartEvent" in types
        assert "ToolCallEndEvent" in types


class TestMessageDeltaIgnored:
    """RawMessageDeltaEvent (usage update) should be silently ignored."""

    def test_message_delta_no_events(self):
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_delta")

        agg.aggregate(_make_message_start("msg_delta"))
        events.clear()

        agg.aggregate(_make_message_delta())

        assert events == []


class TestClearAndReuse:
    """Test that aggregator can be cleared and reused for a new message."""

    def test_reuse_after_clear(self):
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_reuse")

        # First message
        agg.aggregate(_make_message_start("msg_1"))
        agg.aggregate(_make_text_block_start(0))
        agg.aggregate(_make_text_delta(0, "First"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        events.clear()
        agg.clear()

        # Second message
        agg.aggregate(_make_message_start("msg_2"))
        agg.aggregate(_make_text_block_start(0))
        agg.aggregate(_make_text_delta(0, "Second"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        starts_round2 = [e for e in events if isinstance(e, TextMessageStartEvent)]
        assert len(starts_round2) == 1
        assert starts_round2[0].message_id == "msg_2"

        contents_round2 = [e for e in events if isinstance(e, TextMessageContentEvent)]
        assert len(contents_round2) == 1
        assert contents_round2[0].delta == "Second"


class TestTimestamps:
    """All emitted events must carry a positive millisecond timestamp."""

    def test_all_events_have_timestamps(self):
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_ts")

        agg.aggregate(_make_message_start("msg_ts"))
        agg.aggregate(_make_text_block_start(0))
        agg.aggregate(_make_text_delta(0, "x"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        for event in events:
            assert event.timestamp > 0  # type: ignore[union-attr]


class TestEdgeCases:
    """Edge-case scenarios."""

    def test_empty_text_delta_raises_validation_error(self):
        """An empty text delta triggers a Pydantic validation error because delta requires min_length=1."""
        from pydantic import ValidationError

        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_empty")

        agg.aggregate(_make_message_start("msg_empty"))
        agg.aggregate(_make_text_block_start(0))

        with pytest.raises(ValidationError, match="string_too_short"):
            agg.aggregate(_make_text_delta(0, ""))

    def test_unknown_block_type_ignored_on_stop(self):
        """A content_block_stop for an unknown block type should not crash."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_unk")

        agg.aggregate(_make_message_start("msg_unk"))
        agg._block_types[99] = "some_future_type"
        agg.aggregate(_make_block_stop(99))
        agg.aggregate(_make_message_stop())

        # Should only have TextMessageStartEvent + TextMessageEndEvent
        types = [type(e).__name__ for e in events]
        assert "TextMessageStartEvent" in types
        assert "TextMessageEndEvent" in types

    def test_block_stop_for_nonexistent_index(self):
        """block_stop for an index never seen in block_start should be a no-op."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_ghost")

        agg.aggregate(_make_message_start("msg_ghost"))
        agg.aggregate(_make_block_stop(5))
        agg.aggregate(_make_message_stop())

        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_ends) == 0

    def test_tool_delta_with_unknown_index(self):
        """input_json_delta for an index without a prior tool_use start should use fallback."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_orphan")

        agg.aggregate(_make_message_start("msg_orphan"))
        agg.aggregate(_make_input_json_delta(7, '{"x":1}'))

        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(args_events) == 1
        assert args_events[0].tool_call_id == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
