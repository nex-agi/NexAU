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

"""Tests for micro-compact: TimeBasedTrigger, ToolResultCompaction compactable_tools,
CompactionConfig new fields, and Message.created_at."""

from datetime import UTC, datetime, timedelta

from nexau.archs.main_sub.execution.middleware.context_compaction import (
    TimeBasedTrigger,
    ToolResultCompaction,
)
from nexau.archs.main_sub.execution.middleware.context_compaction.config import CompactionConfig
from nexau.archs.main_sub.execution.middleware.context_compaction.factory import (
    create_compaction_strategy,
    create_trigger_strategy,
)
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_msg(text: str, created_at: datetime | None = None, tool_calls: list[ToolUseBlock] | None = None) -> Message:
    from nexau.core.messages import BlockType

    blocks: list[BlockType] = [TextBlock(text=text)]
    if tool_calls:
        blocks.extend(tool_calls)
    return Message(role=Role.ASSISTANT, content=blocks, created_at=created_at)


def _tool_result_msg(tool_use_id: str, content: str = "some output") -> Message:
    return Message(
        role=Role.TOOL,
        content=[ToolResultBlock(tool_use_id=tool_use_id, content=content, is_error=False)],
    )


def _user_msg(text: str = "hello") -> Message:
    return Message(role=Role.USER, content=[TextBlock(text=text)])


def _system_msg(text: str = "You are helpful.") -> Message:
    return Message(role=Role.SYSTEM, content=[TextBlock(text=text)])


def _find_tool_result(messages: list[Message], tool_use_id: str) -> ToolResultBlock:
    """Find a ToolResultBlock by tool_use_id in a message list."""
    for m in messages:
        if m.role == Role.TOOL:
            for b in m.content:
                if isinstance(b, ToolResultBlock) and b.tool_use_id == tool_use_id:
                    return b
    raise ValueError(f"ToolResultBlock with tool_use_id={tool_use_id!r} not found")


# ---------------------------------------------------------------------------
# TimeBasedTrigger
# ---------------------------------------------------------------------------


class TestTimeBasedTrigger:
    """Tests for TimeBasedTrigger strategy."""

    def test_init_default(self):
        trigger = TimeBasedTrigger()
        assert trigger.gap_threshold == timedelta(minutes=5)

    def test_init_custom(self):
        trigger = TimeBasedTrigger(gap_threshold_minutes=10)
        assert trigger.gap_threshold == timedelta(minutes=10)

    def test_triggers_when_gap_exceeds_threshold(self):
        """Last assistant message created_at 8 min ago -> triggers."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        msgs = [
            _user_msg(),
            _assistant_msg("hi", created_at=datetime.now(UTC) - timedelta(minutes=8)),
        ]
        should, reason = trigger.should_compact(msgs, 1000, 10000)
        assert should is True
        assert "8." in reason or "7." in reason  # ~8 min gap

    def test_no_trigger_when_below_threshold(self):
        """Last assistant message created_at 2 min ago -> no trigger."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        msgs = [
            _user_msg(),
            _assistant_msg("hi", created_at=datetime.now(UTC) - timedelta(minutes=2)),
        ]
        should, reason = trigger.should_compact(msgs, 1000, 10000)
        assert should is False
        assert reason == ""

    def test_no_trigger_when_no_assistant_messages(self):
        """New session with no assistant messages -> no trigger."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        msgs = [_user_msg()]
        should, reason = trigger.should_compact(msgs, 1000, 10000)
        assert should is False

    def test_no_trigger_when_created_at_is_none(self):
        """Assistant message with created_at=None -> no trigger (safe default)."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        msgs = [_user_msg(), _assistant_msg("hi", created_at=None)]
        should, reason = trigger.should_compact(msgs, 1000, 10000)
        assert should is False

    def test_stateless_same_result(self):
        """Same messages + same time -> same result (statelessness)."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        msgs = [
            _user_msg(),
            _assistant_msg("hi", created_at=datetime.now(UTC) - timedelta(minutes=8)),
        ]
        r1 = trigger.should_compact(msgs, 1000, 10000)
        r2 = trigger.should_compact(msgs, 1000, 10000)
        assert r1[0] == r2[0]

    def test_uses_last_assistant_message(self):
        """Multiple assistant messages -> uses the last one."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        msgs = [
            _user_msg(),
            _assistant_msg("old", created_at=datetime.now(UTC) - timedelta(minutes=30)),
            _user_msg(),
            _assistant_msg("recent", created_at=datetime.now(UTC) - timedelta(minutes=1)),
        ]
        should, _ = trigger.should_compact(msgs, 1000, 10000)
        assert should is False  # last assistant is 1 min ago

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetime (no tzinfo) is treated as UTC for comparison."""
        trigger = TimeBasedTrigger(gap_threshold_minutes=5)
        naive_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(minutes=10)
        msgs = [_assistant_msg("hi", created_at=naive_time)]
        should, _ = trigger.should_compact(msgs, 1000, 10000)
        assert should is True


# ---------------------------------------------------------------------------
# ToolResultCompaction with compactable_tools
# ---------------------------------------------------------------------------


class TestToolResultCompactionWithFilter:
    """Tests for ToolResultCompaction compactable_tools filtering."""

    def _build_messages(self) -> list[Message]:
        """Build a message list with two iterations of tool calls.

        Iteration 1 (old): assistant calls read_file + write_file
        Iteration 2 (recent): assistant calls read_file
        """
        return [
            _system_msg(),
            # --- iteration 1 (old) ---
            _user_msg("read and write"),
            _assistant_msg(
                "I'll read and write.",
                tool_calls=[
                    ToolUseBlock(id="call_1", name="read_file", input={"path": "a.py"}),
                    ToolUseBlock(id="call_2", name="write_file", input={"path": "a.py", "content": "x"}),
                ],
            ),
            _tool_result_msg("call_1", "file content of a.py"),
            _tool_result_msg("call_2", "wrote a.py"),
            # --- iteration 2 (recent, protected) ---
            _user_msg("read again"),
            _assistant_msg(
                "I'll read.",
                tool_calls=[
                    ToolUseBlock(id="call_3", name="read_file", input={"path": "b.py"}),
                ],
            ),
            _tool_result_msg("call_3", "file content of b.py"),
        ]

    def test_compactable_tools_none_compacts_all(self):
        """compactable_tools=None -> compacts all old tool results (backward compat)."""
        strategy = ToolResultCompaction(keep_iterations=1, compactable_tools=None)
        msgs = self._build_messages()
        result = strategy.compact(msgs)

        # iteration 1 tool results should be compacted
        assert _find_tool_result(result, "call_1").content == "Tool call result has been compacted"
        assert _find_tool_result(result, "call_2").content == "Tool call result has been compacted"

    def test_compactable_tools_filters_by_name(self):
        """compactable_tools={"read_file"} -> only compacts read_file results."""
        strategy = ToolResultCompaction(
            keep_iterations=1,
            compactable_tools=frozenset(["read_file"]),
        )
        msgs = self._build_messages()
        result = strategy.compact(msgs)

        # call_1 (read_file, old iteration) -> compacted
        assert _find_tool_result(result, "call_1").content == "Tool call result has been compacted"

        # call_2 (write_file, old iteration) -> NOT compacted (not in compactable_tools)
        assert _find_tool_result(result, "call_2").content == "wrote a.py"

        # call_3 (read_file, recent/protected iteration) -> NOT compacted (protected)
        assert _find_tool_result(result, "call_3").content == "file content of b.py"

    def test_protected_iterations_not_compacted(self):
        """Recent iterations are protected regardless of compactable_tools."""
        strategy = ToolResultCompaction(
            keep_iterations=2,
            compactable_tools=frozenset(["read_file"]),
        )
        msgs = self._build_messages()
        result = strategy.compact(msgs)

        # Both iterations protected (keep_iterations=2) -> nothing compacted
        for m in result:
            if m.role == Role.TOOL:
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        assert b.content != "Tool call result has been compacted"

    def test_empty_compactable_tools_compacts_nothing(self):
        """compactable_tools=frozenset() -> no tools are compactable."""
        strategy = ToolResultCompaction(
            keep_iterations=1,
            compactable_tools=frozenset(),
        )
        msgs = self._build_messages()
        result = strategy.compact(msgs)

        for m in result:
            if m.role == Role.TOOL:
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        assert b.content != "Tool call result has been compacted"


# ---------------------------------------------------------------------------
# Default compactable_tools list (RFC recommended for code agent)
# ---------------------------------------------------------------------------

# RFC 推荐的默认可压缩工具列表（下游 code agent 配置用）
DEFAULT_COMPACTABLE_TOOLS = frozenset(
    [
        "read_file",
        "search_file_content",
        "list_directory",
        "run_shell_command",
        "read_only_shell_command",
        "web_search",
        "web_read",
        "background_task_manage",
    ]
)

# 不可压缩工具（写操作、交互工具、状态工具）
NON_COMPACTABLE_TOOLS = [
    "write_file",
    "replace",
    "ask_user",
    "write_todos",
    "complete_task",
]


class TestDefaultCompactableToolsList:
    """Verify the RFC-recommended default compactable_tools list behavior."""

    def _build_multi_tool_messages(self) -> list[Message]:
        """Build messages with one old iteration containing both compactable and non-compactable tools."""
        tool_calls = [
            ToolUseBlock(id="c_read", name="read_file", input={"path": "a.py"}),
            ToolUseBlock(id="c_search", name="search_file_content", input={"query": "foo"}),
            ToolUseBlock(id="c_ls", name="list_directory", input={"path": "/tmp"}),
            ToolUseBlock(id="c_shell", name="run_shell_command", input={"command": "ls"}),
            ToolUseBlock(id="c_roshell", name="read_only_shell_command", input={"command": "cat x"}),
            ToolUseBlock(id="c_websearch", name="web_search", input={"query": "test"}),
            ToolUseBlock(id="c_webread", name="web_read", input={"url": "http://x"}),
            ToolUseBlock(id="c_bg", name="background_task_manage", input={"action": "list"}),
            ToolUseBlock(id="c_write", name="write_file", input={"path": "b.py", "content": "x"}),
            ToolUseBlock(id="c_replace", name="replace", input={"path": "b.py", "old": "x", "new": "y"}),
            ToolUseBlock(id="c_ask", name="ask_user", input={"question": "ok?"}),
            ToolUseBlock(id="c_todos", name="write_todos", input={"items": ["a"]}),
            ToolUseBlock(id="c_complete", name="complete_task", input={"task": "done"}),
        ]
        return [
            _system_msg(),
            # --- old iteration (should be subject to compaction) ---
            _user_msg("do everything"),
            _assistant_msg("On it.", tool_calls=tool_calls),
            *[_tool_result_msg(tc.id, f"output of {tc.name}") for tc in tool_calls],
            # --- recent iteration (protected) ---
            _user_msg("again"),
            _assistant_msg("Sure.", tool_calls=[ToolUseBlock(id="c_recent", name="read_file", input={"path": "z.py"})]),
            _tool_result_msg("c_recent", "recent read output"),
        ]

    def test_compactable_tools_are_compacted(self):
        """All tools in DEFAULT_COMPACTABLE_TOOLS should be compacted in old iterations."""
        strategy = ToolResultCompaction(
            keep_iterations=1,
            compactable_tools=DEFAULT_COMPACTABLE_TOOLS,
        )
        msgs = self._build_multi_tool_messages()
        result = strategy.compact(msgs)

        compactable_ids = ["c_read", "c_search", "c_ls", "c_shell", "c_roshell", "c_websearch", "c_webread", "c_bg"]
        for tool_id in compactable_ids:
            tr = _find_tool_result(result, tool_id)
            assert tr.content == "Tool call result has been compacted", f"{tool_id} should be compacted but got: {tr.content}"

    def test_non_compactable_tools_are_preserved(self):
        """Tools NOT in DEFAULT_COMPACTABLE_TOOLS should keep original content."""
        strategy = ToolResultCompaction(
            keep_iterations=1,
            compactable_tools=DEFAULT_COMPACTABLE_TOOLS,
        )
        msgs = self._build_multi_tool_messages()
        result = strategy.compact(msgs)

        non_compactable_ids = ["c_write", "c_replace", "c_ask", "c_todos", "c_complete"]
        for tool_id in non_compactable_ids:
            tr = _find_tool_result(result, tool_id)
            assert tr.content != "Tool call result has been compacted", f"{tool_id} should NOT be compacted but was"

    def test_recent_iteration_fully_protected(self):
        """Even compactable tools in the recent iteration should not be compacted."""
        strategy = ToolResultCompaction(
            keep_iterations=1,
            compactable_tools=DEFAULT_COMPACTABLE_TOOLS,
        )
        msgs = self._build_multi_tool_messages()
        result = strategy.compact(msgs)

        tr = _find_tool_result(result, "c_recent")
        assert tr.content == "recent read output", f"Recent iteration's read_file should be preserved, got: {tr.content}"


# ---------------------------------------------------------------------------
# CompactionConfig new fields
# ---------------------------------------------------------------------------


class TestCompactionConfigMicroCompact:
    """Tests for CompactionConfig new fields."""

    def test_default_trigger_is_token_threshold(self):
        config = CompactionConfig()
        assert config.trigger == "token_threshold"
        assert config.gap_threshold_minutes == 5
        assert config.compactable_tools is None

    def test_time_based_trigger_config(self):
        config = CompactionConfig(
            trigger="time_based",
            gap_threshold_minutes=10,
            compaction_strategy="tool_result_compaction",
            compactable_tools=["read_file", "search_file_content"],
        )
        assert config.trigger == "time_based"
        assert config.gap_threshold_minutes == 10
        assert config.compactable_tools == ["read_file", "search_file_content"]

    def test_backward_compat_no_new_fields(self):
        """Config without new fields still works (backward compat)."""
        config = CompactionConfig(
            compaction_strategy="tool_result_compaction",
            keep_iterations=3,
        )
        assert config.trigger == "token_threshold"
        assert config.compactable_tools is None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryMicroCompact:
    """Tests for factory functions with new config fields."""

    def test_create_time_based_trigger(self):
        config = CompactionConfig(trigger="time_based", gap_threshold_minutes=7)
        trigger = create_trigger_strategy(config)
        assert isinstance(trigger, TimeBasedTrigger)
        assert trigger.gap_threshold == timedelta(minutes=7)

    def test_create_token_threshold_trigger_default(self):
        config = CompactionConfig()
        trigger = create_trigger_strategy(config)
        from nexau.archs.main_sub.execution.middleware.context_compaction import TokenThresholdTrigger

        assert isinstance(trigger, TokenThresholdTrigger)

    def test_create_tool_result_strategy_with_compactable_tools(self):
        config = CompactionConfig(
            compaction_strategy="tool_result_compaction",
            compactable_tools=["read_file", "run_shell_command"],
        )
        strategy = create_compaction_strategy(config)
        assert isinstance(strategy, ToolResultCompaction)
        assert strategy.compactable_tools == frozenset(["read_file", "run_shell_command"])

    def test_create_tool_result_strategy_without_compactable_tools(self):
        config = CompactionConfig(compaction_strategy="tool_result_compaction")
        strategy = create_compaction_strategy(config)
        assert isinstance(strategy, ToolResultCompaction)
        assert strategy.compactable_tools is None


# ---------------------------------------------------------------------------
# Message.created_at
# ---------------------------------------------------------------------------


class TestMessageCreatedAt:
    """Tests for Message.created_at auto-setting."""

    def test_factory_user_sets_created_at(self):
        msg = Message.user("hello")
        assert msg.created_at is not None

    def test_factory_assistant_sets_created_at(self):
        msg = Message.assistant("hi")
        assert msg.created_at is not None

    def test_explicit_created_at_preserved(self):
        ts = datetime(2025, 1, 1, 12, 0, 0)
        msg = Message(role=Role.USER, content=[TextBlock(text="hi")], created_at=ts)
        assert msg.created_at == ts

    def test_serialization_preserves_created_at(self):
        ts = datetime(2025, 6, 15, 10, 30, 0)
        msg = Message(role=Role.ASSISTANT, content=[TextBlock(text="hi")], created_at=ts)
        data = msg.model_dump()
        restored = Message.model_validate(data)
        # created_at serializes to isoformat string; after round-trip it should parse back
        assert restored.created_at is not None
