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

"""Unit tests for team-mode Executor waiting and stop-tool behavior."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.tool.tool_registry import ToolRegistry

# --- Helpers ---


def make_tool_registry() -> ToolRegistry:
    return ToolRegistry()


def make_executor(team_mode: bool = True) -> Executor:
    """Create a minimal Executor with all heavy dependencies mocked."""
    return Executor(
        agent_name="test-agent",
        agent_id="test-agent-1",
        tool_registry=make_tool_registry(),
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


class TestTeamModeStopTools:
    def _prepare_execute_dependencies(self, executor: Executor) -> None:
        """Stub heavy dependencies so execute() only exercises stop-tool control flow."""
        from nexau.core.messages import Message, Role, TextBlock

        mock_response = MagicMock()
        mock_response.content = "assistant response"
        mock_response.to_ump_message.return_value = Message(
            role=Role.ASSISTANT,
            content=[TextBlock(text="assistant response")],
        )

        executor.token_counter = MagicMock()
        executor.token_counter.count_tokens.return_value = 0
        executor.llm_caller = MagicMock()
        executor.llm_caller.call_llm.return_value = mock_response
        executor.response_parser = MagicMock()

        parsed_response = MagicMock()
        parsed_response.model_response = None
        executor.response_parser.parse_response.return_value = parsed_response

        object.__setattr__(
            executor, "_apply_after_agent_hooks", MagicMock(side_effect=lambda **kwargs: (kwargs["final_response"], kwargs["messages"]))
        )

    def test_ask_user_stop_tool_waits_for_followup_message(self):
        """ask_user should pause in team_mode instead of terminating the executor."""
        executor = make_executor(team_mode=True)
        self._prepare_execute_dependencies(executor)

        def fake_process_xml_calls(hook_input, **_kwargs):  # noqa: ANN001
            executor._last_stop_tool_name = "ask_user"
            return (
                "processed ask_user",
                True,
                '{"content": "Asking user questions, waiting for user answers..."}',
                hook_input.messages,
                [],
            )

        executor._process_xml_calls = MagicMock(side_effect=fake_process_xml_calls)
        executor._wait_for_messages = MagicMock(return_value=False)

        response, _messages = executor.execute(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Need more input."},
            ],
            agent_state=MagicMock(),
        )

        executor._wait_for_messages.assert_called_once()
        assert executor._is_waiting_for_user is True
        assert response == "processed ask_user"

    def test_non_finish_team_stop_tool_waits_in_team_mode(self):
        """Custom/domain stop tools should not terminate a team-mode executor."""
        executor = make_executor(team_mode=True)
        self._prepare_execute_dependencies(executor)

        def fake_process_xml_calls(hook_input, **_kwargs):  # noqa: ANN001
            executor._last_stop_tool_name = "complete_task"
            return (
                "processed complete_task",
                True,
                '{"result": "done"}',
                hook_input.messages,
                [],
            )

        executor._process_xml_calls = MagicMock(side_effect=fake_process_xml_calls)
        executor._wait_for_messages = MagicMock(return_value=False)

        response, _messages = executor.execute(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Finish the work."},
            ],
            agent_state=MagicMock(),
        )

        executor._wait_for_messages.assert_called_once()
        assert executor._is_waiting_for_user is False
        assert response == "processed complete_task"

    def test_finish_team_stop_tool_terminates_in_team_mode(self):
        """finish_team is the only stop tool that should end the team run."""
        executor = make_executor(team_mode=True)
        self._prepare_execute_dependencies(executor)

        def fake_process_xml_calls(hook_input, **_kwargs):  # noqa: ANN001
            executor._last_stop_tool_name = "finish_team"
            return (
                "processed finish_team",
                True,
                '{"result": "done"}',
                hook_input.messages,
                [],
            )

        executor._process_xml_calls = MagicMock(side_effect=fake_process_xml_calls)
        executor._wait_for_messages = MagicMock(return_value=False)

        response, _messages = executor.execute(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Finish the work."},
            ],
            agent_state=MagicMock(),
        )

        executor._wait_for_messages.assert_not_called()
        assert executor._is_waiting_for_user is False
        assert response == '{"result": "done"}'
