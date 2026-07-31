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

"""Unit tests for RoundAndTokenReminderMiddleware."""

import pytest

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import BeforeModelHookInput
from nexau.archs.main_sub.execution.middleware.round_and_token_reminder import (
    RoundAndTokenReminderMiddleware,
)
from nexau.archs.main_sub.history_list import HistoryList
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role


@pytest.fixture
def base_messages():
    """Sample conversation messages."""

    return messages_from_legacy_openai_chat(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    )


def test_before_model_adds_iteration_hint(agent_state: AgentState, base_messages):
    """Middleware appends iteration hint before model call."""

    middleware = RoundAndTokenReminderMiddleware(max_context_tokens=10)
    hook_input = BeforeModelHookInput(
        agent_state=agent_state,
        max_iterations=5,
        current_iteration=4,
        messages=base_messages,
    )

    result = middleware.before_model(hook_input)

    assert result.messages is not None
    assert len(result.messages) == len(base_messages) + 1
    assert result.messages[:-1] == base_messages
    assert result.messages[-1].role == Role.FRAMEWORK

    appended = result.messages[-1].get_text_content()
    assert "iteration 4/5" in appended
    assert "iteration(s) remaining" in appended


def test_before_model_never_mutates_the_last_user_message(agent_state: AgentState):
    """The reminder is a separate FRAMEWORK row even after a real USER turn."""

    user = Message.user("visible question")
    messages = [Message.assistant("previous answer"), user]
    original_user = user.model_copy(deep=True)
    middleware = RoundAndTokenReminderMiddleware(max_context_tokens=1000)
    hook_input = BeforeModelHookInput(
        agent_state=agent_state,
        max_iterations=5,
        current_iteration=2,
        messages=messages,
    )

    result = middleware.before_model(hook_input)

    assert result.messages is not None
    assert result.messages[:-1] == messages
    assert result.messages[-1].role == Role.FRAMEWORK
    assert user == original_user
    assert user.get_text_content() == "visible question"


def test_reminder_is_an_append_in_history(agent_state: AgentState, base_messages):
    """A distinct FRAMEWORK reminder cannot masquerade as compaction."""

    history = HistoryList(base_messages)
    middleware = RoundAndTokenReminderMiddleware(max_context_tokens=1000)
    hook_input = BeforeModelHookInput(
        agent_state=agent_state,
        max_iterations=5,
        current_iteration=2,
        messages=list(history),
    )

    result = middleware.before_model(hook_input)
    assert result.messages is not None
    history.replace_all(result.messages)
    append_messages, replace_messages, _current = history._prepare_flush()  # noqa: SLF001

    assert replace_messages is None
    assert append_messages is not None
    assert len(append_messages) == 1
    assert append_messages[0].role == Role.FRAMEWORK


def test_before_model_adds_token_hint(agent_state: AgentState, base_messages):
    """Middleware appends token warning when enabled."""

    middleware = RoundAndTokenReminderMiddleware(
        max_context_tokens=10,  # small context to force warning branch
        desired_max_tokens=10,
    )
    hook_input = BeforeModelHookInput(
        agent_state=agent_state,
        max_iterations=4,
        current_iteration=2,
        messages=base_messages,
    )

    result = middleware.before_model(hook_input)
    assert result.messages is not None
    appended = result.messages[-1].get_text_content().lower()
    assert "iteration 2/4" in appended
    assert "token usage is approaching the limit" in appended


def test_iteration_hint_variants():
    """Iteration hint messaging matches expected thresholds."""

    # Low remaining (<=1)
    hint_low = RoundAndTokenReminderMiddleware._build_iteration_hint(4, 5, 1)  # type: ignore[attr-defined]
    assert "warning" in hint_low.lower()
    assert "1 iteration(s) remaining" in hint_low

    # Medium remaining (<=3)
    hint_mid = RoundAndTokenReminderMiddleware._build_iteration_hint(4, 6, 2)  # type: ignore[attr-defined]
    assert "iterations remaining" in hint_mid
    assert "mindful" in hint_mid

    # High remaining
    hint_high = RoundAndTokenReminderMiddleware._build_iteration_hint(2, 10, 8)  # type: ignore[attr-defined]
    assert "continue your response" in hint_high.lower()


def test_token_limit_hint_variants():
    """Token limit hint messaging matches expected thresholds."""

    # Low remaining triggers warning
    hint_low = RoundAndTokenReminderMiddleware._build_token_limit_hint(  # type: ignore[attr-defined]
        current_prompt_tokens=9000,
        max_tokens=10000,
        remaining_tokens=1000,
        desired_max_tokens=4000,
    )
    assert "warning" in hint_low.lower()
    assert "1000 tokens left" in hint_low

    # High remaining uses neutral wording
    hint_high = RoundAndTokenReminderMiddleware._build_token_limit_hint(  # type: ignore[attr-defined]
        current_prompt_tokens=5000,
        max_tokens=10000,
        remaining_tokens=5000,
        desired_max_tokens=1000,
    )
    assert "5000 tokens left" in hint_high
    assert "continue your response" in hint_high.lower()
