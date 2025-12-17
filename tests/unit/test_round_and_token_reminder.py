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

from typing import cast

import pytest

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.hooks import BeforeModelHookInput
from nexau.archs.main_sub.execution.middleware.round_and_token_reminder import (
    RoundAndTokenReminderMiddleware,
)


@pytest.fixture
def base_messages() -> list[dict[str, str]]:
    """Sample conversation messages."""

    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def test_before_model_adds_iteration_hint(agent_state: AgentState, base_messages: list[dict[str, str]]):
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

    appended = cast(str, result.messages[-1]["content"])
    assert "iteration 4/5" in appended
    assert "iteration(s) remaining" in appended


def test_before_model_adds_token_hint(agent_state: AgentState, base_messages: list[dict[str, str]]):
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
    appended = cast(str, result.messages[-1]["content"]).lower()
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
