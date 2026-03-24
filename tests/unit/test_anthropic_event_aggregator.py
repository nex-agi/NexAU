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

    def test_empty_thinking_delta_is_ignored(self):
        """Empty thinking deltas should not emit invalid content events."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_think_empty")

        agg.aggregate(_make_message_start("msg_think_empty"))
        agg.aggregate(_make_thinking_block_start(0))
        agg.aggregate(_make_thinking_delta(0, ""))
        agg.aggregate(_make_thinking_delta(0, "Step 1"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        thinking_events = [e for e in events if isinstance(e, ThinkingTextMessageContentEvent)]
        assert len(thinking_events) == 1
        assert thinking_events[0].delta == "Step 1"


class TestEmptyDeltaIgnored:
    """Tests that Anthropic empty deltas do not emit invalid AG-UI events."""

    def test_empty_text_delta_is_ignored(self):
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_text_empty")

        agg.aggregate(_make_message_start("msg_text_empty"))
        agg.aggregate(_make_text_block_start(0))
        agg.aggregate(_make_text_delta(0, ""))
        agg.aggregate(_make_text_delta(0, "Hello"))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        text_events = [e for e in events if isinstance(e, TextMessageContentEvent)]
        assert len(text_events) == 1
        assert text_events[0].delta == "Hello"

    def test_empty_input_json_delta_is_ignored(self):
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_tool_empty")

        agg.aggregate(_make_message_start("msg_tool_empty"))
        agg.aggregate(_make_tool_use_block_start(0, "toolu_empty", "read_file"))
        agg.aggregate(_make_input_json_delta(0, ""))
        agg.aggregate(_make_input_json_delta(0, '{"path": "main.py"}'))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        tool_arg_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(tool_arg_events) == 1
        assert tool_arg_events[0].delta == '{"path": "main.py"}'


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

    def test_empty_text_delta_is_ignored(self):
        """Empty text deltas should be skipped to avoid invalid AG-UI events."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_empty")

        agg.aggregate(_make_message_start("msg_empty"))
        agg.aggregate(_make_text_block_start(0))

        agg.aggregate(_make_text_delta(0, ""))

        content_events = [e for e in events if isinstance(e, TextMessageContentEvent)]
        assert content_events == []

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

    def test_tool_delta_with_unknown_index_buffered(self):
        """input_json_delta without prior content_block_start is buffered, not emitted."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_orphan")

        agg.aggregate(_make_message_start("msg_orphan"))
        agg.aggregate(_make_input_json_delta(7, '{"x":1}'))

        # Nothing emitted yet — fragments are buffered
        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(start_events) == 0
        assert len(args_events) == 0
        assert agg._pending_tool_deltas[7] == ['{"x":1}']

    def test_tool_delta_flushed_on_message_stop(self):
        """Buffered deltas are flushed with a synthetic id on message_stop."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_flush")

        agg.aggregate(_make_message_start("msg_flush"))
        agg.aggregate(_make_input_json_delta(0, '{"x":'))
        agg.aggregate(_make_input_json_delta(0, "1}"))
        agg.aggregate(_make_message_stop())

        # After message_stop: synthetic start + 2 args + end + message_end
        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(start_events) == 1
        assert start_events[0].tool_call_id.startswith("toolu_late_")
        assert len(args_events) == 2
        # All events share the same synthetic id
        synthetic_id = start_events[0].tool_call_id
        assert all(e.tool_call_id == synthetic_id for e in args_events)
        assert len(end_events) == 1
        assert end_events[0].tool_call_id == synthetic_id


class TestServerToolUseBlock:
    """Tests for ServerToolUseBlock (web_search, code_execution, etc.)."""

    def test_server_tool_use_block_registered(self):
        """ServerToolUseBlock should be registered like ToolUseBlock."""
        from anthropic.types import ServerToolUseBlock

        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_server")

        agg.aggregate(_make_message_start("msg_srv"))
        # Simulate a server tool use block start
        agg.aggregate(
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ServerToolUseBlock(
                    type="server_tool_use",
                    id="srvtoolu_01ABC",
                    name="web_search",
                    input={},
                ),
            )
        )
        agg.aggregate(_make_input_json_delta(0, '{"query": "test"}'))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        types = [type(e).__name__ for e in events]
        assert "ToolCallStartEvent" in types
        assert "ToolCallArgsEvent" in types
        assert "ToolCallEndEvent" in types

        tool_start = [e for e in events if isinstance(e, ToolCallStartEvent)][0]
        assert tool_start.tool_call_id == "srvtoolu_01ABC"
        assert tool_start.tool_call_name == "web_search"

    def test_server_tool_use_block_stop(self):
        """content_block_stop for a server_tool_use block should emit ToolCallEndEvent."""
        from anthropic.types import ServerToolUseBlock

        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_srv_stop")

        agg.aggregate(_make_message_start("msg_srv_stop"))
        agg.aggregate(
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ServerToolUseBlock(
                    type="server_tool_use",
                    id="srvtoolu_02DEF",
                    name="code_execution",
                    input={},
                ),
            )
        )
        agg.aggregate(_make_input_json_delta(0, '{"code": "print(1)"}'))
        agg.aggregate(_make_block_stop(0))

        tool_ends = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(tool_ends) == 1
        assert tool_ends[0].tool_call_id == "srvtoolu_02DEF"


class TestLateContentBlockStart:
    """Tests for content_block_start arriving after input_json_delta (buffering path)."""

    def test_late_block_start_flushes_with_real_id(self):
        """If content_block_start arrives after delta, flush buffered fragments with the real id."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_late")

        agg.aggregate(_make_message_start("msg_late"))
        # 1. delta arrives first → buffered, no events emitted
        agg.aggregate(_make_input_json_delta(0, '{"path":'))
        assert len([e for e in events if isinstance(e, ToolCallStartEvent)]) == 0

        # 2. block_start arrives → register real id, flush buffer, then emit
        agg.aggregate(_make_tool_use_block_start(0, "toolu_real", "write_file"))

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].tool_call_id == "toolu_real"
        assert start_events[0].tool_call_name == "write_file"

        # Flushed fragment uses the real id
        args_so_far = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(args_so_far) == 1
        assert args_so_far[0].tool_call_id == "toolu_real"
        assert args_so_far[0].delta == '{"path":'

        # 3. more deltas after start → directly emitted with real id
        agg.aggregate(_make_input_json_delta(0, ' "a.py"}'))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        all_args = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        assert len(all_args) == 2
        assert all(a.tool_call_id == "toolu_real" for a in all_args)

        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].tool_call_id == "toolu_real"

    def test_block_stop_flushes_with_synthetic_id(self):
        """content_block_stop without prior start → flush with synthetic id, all events consistent."""
        events: list[object] = []
        agg = AnthropicEventAggregator(on_event=events.append, run_id="run_no_btype")

        agg.aggregate(_make_message_start("msg_no_btype"))
        # Delta without prior block_start → buffered
        agg.aggregate(_make_input_json_delta(0, '{"x": 1}'))
        agg.aggregate(_make_block_stop(0))
        agg.aggregate(_make_message_stop())

        start_events = [e for e in events if isinstance(e, ToolCallStartEvent)]
        args_events = [e for e in events if isinstance(e, ToolCallArgsEvent)]
        end_events = [e for e in events if isinstance(e, ToolCallEndEvent)]

        assert len(start_events) == 1
        synthetic_id = start_events[0].tool_call_id
        assert synthetic_id.startswith("toolu_late_")

        # All events share the same synthetic id
        assert len(args_events) == 1
        assert args_events[0].tool_call_id == synthetic_id
        assert len(end_events) == 1
        assert end_events[0].tool_call_id == synthetic_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
