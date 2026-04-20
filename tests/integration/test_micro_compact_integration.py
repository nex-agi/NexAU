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

"""Integration tests for micro-compact: TimeBasedTrigger + ToolResultCompaction
with compactable_tools in a full middleware pipeline.

Two verification approaches:
1. Unit-integration: directly call middleware.before_model() with crafted messages
2. Full E2E: agent.run() with mock LLM, then age timestamps and run again
"""

from __future__ import annotations

import json
import types
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import Mock

from nexau.archs.llm.llm_aggregators.events import CompactionFinishedEvent, CompactionStartedEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.hooks import BeforeModelHookInput
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware
from nexau.archs.main_sub.execution.middleware.context_compaction import ContextCompactionMiddleware
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, ToolUseBlock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleChatModel:
    """Mock model that alternates between tool calls and final responses."""

    def __init__(self) -> None:
        self.call_count = 0

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        last_role = messages[-1].get("role") if messages and isinstance(messages[-1], dict) else ""

        self.call_count += 1

        # After tool result, return final text
        if last_role == "tool":
            msg = types.SimpleNamespace(content="Done.", tool_calls=[])
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)],
                usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            )

        # Otherwise, call read_file and write_file tools
        msg = types.SimpleNamespace(
            content="Let me read and write.",
            tool_calls=[
                {
                    "id": f"call_read_{self.call_count}",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({"path": "/tmp/test.py"}),
                    },
                },
                {
                    "id": f"call_write_{self.call_count}",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({"path": "/tmp/test.py", "content": "print('hello')"}),
                    },
                },
            ],
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=msg)],
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )


def _age_messages(messages: list[Message], minutes: int) -> None:
    """Shift all message created_at timestamps back by `minutes`."""
    delta = timedelta(minutes=minutes)
    for msg in messages:
        if msg.created_at is not None:
            msg.created_at = msg.created_at - delta


# ---------------------------------------------------------------------------
# Test 1: Unit-integration — middleware.before_model() with crafted messages
# ---------------------------------------------------------------------------


def test_micro_compact_middleware_before_model_triggers_on_old_messages():
    """Verify TimeBasedTrigger + ToolResultCompaction with compactable_tools
    works correctly when called through the middleware's before_model() hook."""

    # Create micro-compact CCM instance
    middleware = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=5,
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        compactable_tools=["read_file", "search_file_content"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )

    # Build messages: 2 iterations, all timestamps 10 minutes ago
    old_time = datetime.now(UTC) - timedelta(minutes=10)
    messages = [
        Message(role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")]),
        # --- iteration 1 (old, should be compacted) ---
        Message(role=Role.USER, content=[TextBlock(text="read and write")], created_at=old_time),
        Message(
            role=Role.ASSISTANT,
            content=[
                TextBlock(text="I'll do both."),
                ToolUseBlock(id="call_r1", name="read_file", input={"path": "a.py"}),
                ToolUseBlock(id="call_w1", name="write_file", input={"path": "a.py", "content": "x"}),
            ],
            created_at=old_time,
        ),
        Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="call_r1", content="file content of a.py", is_error=False)],
            created_at=old_time,
        ),
        Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="call_w1", content="wrote a.py successfully", is_error=False)],
            created_at=old_time,
        ),
        # --- iteration 2 (recent, protected by keep_iterations=1) ---
        Message(role=Role.USER, content=[TextBlock(text="read again")], created_at=old_time),
        Message(
            role=Role.ASSISTANT,
            content=[
                TextBlock(text="Reading."),
                ToolUseBlock(id="call_r2", name="read_file", input={"path": "b.py"}),
            ],
            created_at=old_time,
        ),
        Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="call_r2", content="file content of b.py", is_error=False)],
            created_at=old_time,
        ),
    ]

    # Create mock hook input
    from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
    from nexau.archs.main_sub.agent_state import AgentState
    from nexau.archs.tool.tool_registry import ToolRegistry

    agent_state = AgentState(
        agent_name="test",
        agent_id="test_id",
        run_id="run_1",
        root_run_id="run_1",
        context=AgentContext(),
        global_storage=GlobalStorage(),
        tool_registry=ToolRegistry(),
    )

    hook_input = BeforeModelHookInput(
        agent_state=agent_state,
        max_iterations=10,
        current_iteration=0,
        messages=messages,
    )

    result = middleware.before_model(hook_input)

    # Should have triggered compaction (time gap > 5 min)
    assert result.messages is not None, "Compaction should have been triggered"

    compacted = result.messages

    # Verify: iteration 1's read_file result was compacted
    call_r1_msg = next(
        m
        for m in compacted
        if m.role == Role.TOOL and any(isinstance(b, ToolResultBlock) and b.tool_use_id == "call_r1" for b in m.content)
    )
    tr_r1 = next(b for b in call_r1_msg.content if isinstance(b, ToolResultBlock))
    assert tr_r1.content == "Tool call result has been compacted", f"read_file result should be compacted, got: {tr_r1.content}"

    # Verify: iteration 1's write_file result was NOT compacted (not in compactable_tools)
    call_w1_msg = next(
        m
        for m in compacted
        if m.role == Role.TOOL and any(isinstance(b, ToolResultBlock) and b.tool_use_id == "call_w1" for b in m.content)
    )
    tr_w1 = next(b for b in call_w1_msg.content if isinstance(b, ToolResultBlock))
    assert tr_w1.content == "wrote a.py successfully", f"write_file result should NOT be compacted, got: {tr_w1.content}"

    # Verify: iteration 2's read_file result was NOT compacted (protected iteration)
    call_r2_msg = next(
        m
        for m in compacted
        if m.role == Role.TOOL and any(isinstance(b, ToolResultBlock) and b.tool_use_id == "call_r2" for b in m.content)
    )
    tr_r2 = next(b for b in call_r2_msg.content if isinstance(b, ToolResultBlock))
    assert tr_r2.content == "file content of b.py", f"Protected iteration's read_file should NOT be compacted, got: {tr_r2.content}"


def test_micro_compact_does_not_trigger_when_gap_below_threshold():
    """Verify TimeBasedTrigger does NOT fire when time gap is below threshold."""
    middleware = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=5,
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        compactable_tools=["read_file"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )

    # Messages with recent timestamps (1 minute ago)
    recent_time = datetime.now(UTC) - timedelta(minutes=1)
    messages = [
        Message(role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")]),
        Message(role=Role.USER, content=[TextBlock(text="hello")], created_at=recent_time),
        Message(
            role=Role.ASSISTANT,
            content=[
                TextBlock(text="Reading."),
                ToolUseBlock(id="call_1", name="read_file", input={"path": "a.py"}),
            ],
            created_at=recent_time,
        ),
        Message(
            role=Role.TOOL,
            content=[ToolResultBlock(tool_use_id="call_1", content="file content", is_error=False)],
            created_at=recent_time,
        ),
    ]

    from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
    from nexau.archs.main_sub.agent_state import AgentState
    from nexau.archs.tool.tool_registry import ToolRegistry

    agent_state = AgentState(
        agent_name="test",
        agent_id="test_id",
        run_id="run_1",
        root_run_id="run_1",
        context=AgentContext(),
        global_storage=GlobalStorage(),
        tool_registry=ToolRegistry(),
    )

    hook_input = BeforeModelHookInput(
        agent_state=agent_state,
        max_iterations=10,
        current_iteration=0,
        messages=messages,
    )

    result = middleware.before_model(hook_input)

    # Should NOT trigger — gap is only 1 minute
    assert result.messages is None, "Compaction should NOT have been triggered (gap < threshold)"


# ---------------------------------------------------------------------------
# Test 2: Full E2E — agent.run() with mock LLM + timestamp aging
# ---------------------------------------------------------------------------


def test_micro_compact_e2e_with_agent_run(monkeypatch):
    """Full E2E: run agent twice with mock LLM. Age timestamps between runs.
    Verify micro-compact triggers on second run and selectively compacts tool results."""

    model = _SimpleChatModel()
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = model.create
    monkeypatch.setattr(Agent, "_initialize_openai_client", lambda _self: mock_client)

    events: list[Any] = []
    events_middleware = AgentEventsMiddleware(
        session_id="micro_compact_e2e",
        on_event=events.append,
    )

    # Micro-compact CCM (tier-1)
    micro_compact_middleware = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=5,
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        compactable_tools=["read_file"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )

    read_tool = Tool(
        name="read_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        implementation=lambda path: f"content of {path} " + ("X" * 500),
    )
    write_tool = Tool(
        name="write_file",
        description="Write a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        implementation=lambda path, content: f"wrote {path}",
    )

    config = AgentConfig(
        name="micro_compact_e2e_agent",
        system_prompt="Use read_file and write_file tools, then respond.",
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_chat_completion",
            stream=False,
            max_tokens=512,
        ),
        tools=[read_tool, write_tool],
        middlewares=[micro_compact_middleware, events_middleware],
        max_iterations=4,
        retry_attempts=1,
        max_context_tokens=200000,
        tool_call_mode="openai",
    )

    session_manager = SessionManager(engine=InMemoryDatabaseEngine())
    agent = Agent(
        config=config,
        session_manager=session_manager,
        user_id="test_user",
        session_id="test_session",
    )

    # Round 1: normal run, no compaction expected
    response1 = agent.run(message="Please read and write test.py")
    assert isinstance(response1, str)

    compaction_events_after_r1 = [e for e in events if isinstance(e, CompactionStartedEvent) and e.mode == "regular"]
    assert len(compaction_events_after_r1) == 0, "No compaction should trigger on first run"

    # Age all existing messages by 10 minutes to simulate user leaving
    _age_messages(list(agent.history), 10)

    # Round 2: should trigger micro-compact because time gap > 5 min
    response2 = agent.run(message="Read again please")
    assert isinstance(response2, str)

    compaction_started = [e for e in events if isinstance(e, CompactionStartedEvent) and e.mode == "regular"]
    compaction_finished = [e for e in events if isinstance(e, CompactionFinishedEvent) and e.mode == "regular"]

    assert len(compaction_started) >= 1, "Micro-compact should have triggered on second run"
    assert len(compaction_finished) >= 1
    assert any(e.success for e in compaction_finished), "At least one compaction should succeed"

    # Verify the trigger reason mentions time gap
    for e in compaction_started:
        if e.trigger_reason:
            assert "Time gap" in e.trigger_reason or "min" in e.trigger_reason


# ---------------------------------------------------------------------------
# Test 3: Two-layer CCM chain — micro-compact + full-compact
# ---------------------------------------------------------------------------


def test_two_layer_ccm_chain(monkeypatch):
    """Two CCM instances: micro-compact (tier-1) reduces tokens so full-compact (tier-2) doesn't fire."""

    model = _SimpleChatModel()
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = model.create
    monkeypatch.setattr(Agent, "_initialize_openai_client", lambda _self: mock_client)

    events: list[Any] = []
    events_middleware = AgentEventsMiddleware(
        session_id="two_layer_e2e",
        on_event=events.append,
    )

    # Tier-1: micro-compact
    micro_ccm = ContextCompactionMiddleware(
        trigger="time_based",
        gap_threshold_minutes=5,
        compaction_strategy="tool_result_compaction",
        keep_iterations=1,
        compactable_tools=["read_file"],
        auto_compact=True,
        max_context_tokens=200000,
        emergency_compact_enabled=False,
    )

    # Tier-2: full-compact (token threshold very high, should NOT trigger)
    full_ccm = ContextCompactionMiddleware(
        compaction_strategy="tool_result_compaction",
        auto_compact=True,
        max_context_tokens=200000,
        threshold=0.95,
        keep_iterations=1,
        emergency_compact_enabled=False,
    )

    read_tool = Tool(
        name="read_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        implementation=lambda path: f"content of {path}",
    )
    write_tool = Tool(
        name="write_file",
        description="Write a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        implementation=lambda path, content: f"wrote {path}",
    )

    config = AgentConfig(
        name="two_layer_agent",
        system_prompt="Use tools then respond.",
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_type="openai_chat_completion",
            stream=False,
            max_tokens=512,
        ),
        tools=[read_tool, write_tool],
        # micro-compact first, then full-compact
        middlewares=[micro_ccm, full_ccm, events_middleware],
        max_iterations=4,
        retry_attempts=1,
        max_context_tokens=200000,
        tool_call_mode="openai",
    )

    session_manager = SessionManager(engine=InMemoryDatabaseEngine())
    agent = Agent(
        config=config,
        session_manager=session_manager,
        user_id="test_user",
        session_id="test_session_2layer",
    )

    # Round 1
    agent.run(message="Read and write test.py")

    # Age messages
    _age_messages(list(agent.history), 10)

    # Round 2 — micro-compact should fire, full-compact should NOT
    agent.run(message="Read again")

    all_compaction_started = [e for e in events if isinstance(e, CompactionStartedEvent) and e.mode == "regular"]
    # At least micro-compact should have triggered
    assert len(all_compaction_started) >= 1, "At least micro-compact should have triggered"
