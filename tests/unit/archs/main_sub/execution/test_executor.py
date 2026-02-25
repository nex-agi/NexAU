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

"""Unit tests for Executor._wait_for_messages and enqueue_message (team_mode)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.executor import Executor

# --- Helpers ---


def make_executor(team_mode: bool = True) -> Executor:
    """Create a minimal Executor with all heavy dependencies mocked."""
    return Executor(
        agent_name="test-agent",
        agent_id="test-agent-1",
        tool_registry={},
        sub_agents={},
        stop_tools=set(),
        openai_client=MagicMock(),
        llm_config=LLMConfig(model="gpt-4o"),
        team_mode=team_mode,
    )


# --- TestWaitForMessages ---


class TestWaitForMessages:
    def test_returns_true_when_message_arrives(self):
        """Returns True when a message is enqueued before the wait times out."""
        executor = make_executor(team_mode=True)

        from nexau.core.messages import Message, Role, TextBlock

        def enqueue_after_delay() -> None:
            import time

            time.sleep(0.05)
            executor.queued_messages.append(Message(role=Role.USER, content=[TextBlock(text="hello")]))
            executor._message_available.set()

        t = threading.Thread(target=enqueue_after_delay, daemon=True)
        t.start()

        result = executor._wait_for_messages()
        t.join(timeout=1)

        assert result is True

    def test_returns_false_when_stop_signal_set(self):
        """Returns False immediately when stop_signal is already True."""
        executor = make_executor(team_mode=True)
        executor.stop_signal = True

        result = executor._wait_for_messages()

        assert result is False

    def test_sets_is_idle_true_during_wait(self):
        """_is_idle is True while waiting for messages."""
        executor = make_executor(team_mode=True)

        idle_states: list[bool] = []

        def set_stop_after_check() -> None:
            import time

            # Give the wait loop a moment to enter idle state
            time.sleep(0.05)
            idle_states.append(executor._is_idle)
            executor.stop_signal = True
            executor._message_available.set()

        t = threading.Thread(target=set_stop_after_check, daemon=True)
        t.start()
        executor._wait_for_messages()
        t.join(timeout=1)

        assert True in idle_states

    def test_clears_is_idle_after_returning(self):
        """_is_idle is False after _wait_for_messages returns."""
        executor = make_executor(team_mode=True)
        executor.stop_signal = True

        executor._wait_for_messages()

        assert executor._is_idle is False

    def test_returns_true_when_queue_already_has_messages(self):
        """Returns True immediately when queued_messages is non-empty on entry."""
        executor = make_executor(team_mode=True)

        from nexau.core.messages import Message, Role, TextBlock

        executor.queued_messages.append(Message(role=Role.USER, content=[TextBlock(text="pre-queued")]))

        result = executor._wait_for_messages()

        assert result is True

    def test_stop_signal_wins_over_queued_messages(self):
        """Returns False when stop_signal is set even if messages are queued."""
        executor = make_executor(team_mode=True)

        from nexau.core.messages import Message, Role, TextBlock

        executor.stop_signal = True
        executor.queued_messages.append(Message(role=Role.USER, content=[TextBlock(text="ignored")]))

        result = executor._wait_for_messages()

        assert result is False


# --- TestEnqueueMessage ---


class TestEnqueueMessage:
    def test_appends_message_to_queue(self):
        """enqueue_message adds a normalized Message to queued_messages."""
        executor = make_executor(team_mode=True)

        executor.enqueue_message({"role": "user", "content": "hello"})

        assert len(executor.queued_messages) == 1
        msg = executor.queued_messages[0]
        from nexau.core.messages import Role

        assert msg.role == Role.USER

    def test_message_content_preserved(self):
        """The text content is preserved after normalization."""
        executor = make_executor(team_mode=True)

        executor.enqueue_message({"role": "user", "content": "task payload"})

        from nexau.core.messages import TextBlock

        msg = executor.queued_messages[0]
        assert any(isinstance(b, TextBlock) and b.text == "task payload" for b in msg.content)

    def test_sets_message_available_event(self):
        """enqueue_message sets _message_available to wake up the wait loop."""
        executor = make_executor(team_mode=True)
        executor._message_available.clear()

        executor.enqueue_message({"role": "user", "content": "wake"})

        assert executor._message_available.is_set()

    def test_multiple_enqueues_accumulate(self):
        """Multiple calls accumulate messages in order."""
        executor = make_executor(team_mode=True)

        executor.enqueue_message({"role": "user", "content": "first"})
        executor.enqueue_message({"role": "assistant", "content": "second"})

        assert len(executor.queued_messages) == 2

    def test_default_role_is_user(self):
        """Missing 'role' key defaults to 'user'."""
        executor = make_executor(team_mode=True)

        executor.enqueue_message({"content": "no role"})

        from nexau.core.messages import Role

        assert executor.queued_messages[0].role == Role.USER

    def test_enqueue_wakes_wait_for_messages(self):
        """enqueue_message unblocks a concurrent _wait_for_messages call."""
        executor = make_executor(team_mode=True)

        result: list[bool] = []

        def wait_thread() -> None:
            result.append(executor._wait_for_messages())

        t = threading.Thread(target=wait_thread, daemon=True)
        t.start()

        import time

        time.sleep(0.05)
        executor.enqueue_message({"role": "user", "content": "unblock"})
        t.join(timeout=2)

        assert result == [True]


# --- TestIsIdleProperty ---


class TestIsIdleProperty:
    def test_is_idle_false_by_default(self):
        executor = make_executor(team_mode=True)
        assert executor.is_idle is False

    def test_is_idle_reflects_internal_state(self):
        executor = make_executor(team_mode=True)
        executor._is_idle = True
        assert executor.is_idle is True


# --- TestForceStop ---


class TestForceStop:
    def test_force_stop_sets_stop_signal(self):
        executor = make_executor(team_mode=True)
        executor.force_stop()
        assert executor.stop_signal is True

    def test_force_stop_sets_shutdown_event(self):
        executor = make_executor(team_mode=True)
        executor.force_stop()
        assert executor._shutdown_event.is_set()

    def test_force_stop_sets_message_available(self):
        """force_stop wakes the _wait_for_messages loop."""
        executor = make_executor(team_mode=True)
        executor._message_available.clear()
        executor.force_stop()
        assert executor._message_available.is_set()

    def test_force_stop_unblocks_wait_for_messages(self):
        """force_stop causes _wait_for_messages to return False."""
        executor = make_executor(team_mode=True)

        result: list[bool] = []

        def wait_thread() -> None:
            result.append(executor._wait_for_messages())

        t = threading.Thread(target=wait_thread, daemon=True)
        t.start()

        import time

        time.sleep(0.05)
        executor.force_stop()
        t.join(timeout=2)

        assert result == [False]
