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

"""Unit tests for TeamMessageMiddleware."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from nexau.archs.main_sub.execution.hooks import BeforeModelHookInput
from nexau.archs.main_sub.team.middleware.team_message_middleware import TeamMessageMiddleware
from nexau.archs.session.models.team_message import TeamMessageModel
from nexau.core.messages import Message, Role


def make_team_message(from_agent_id: str, content: str) -> TeamMessageModel:
    msg = MagicMock(spec=TeamMessageModel)
    msg.from_agent_id = from_agent_id
    msg.content = content
    return msg


def make_hook_input(messages: list[Message] | None = None) -> BeforeModelHookInput:
    return BeforeModelHookInput(
        agent_state=MagicMock(),
        max_iterations=10,
        current_iteration=1,
        messages=messages or [],
    )


class TestTeamMessageMiddlewareDrainInbox:
    def test_drain_inbox_populates_pending(self):
        async def run():
            bus = MagicMock()
            bus.drain = AsyncMock(return_value=[make_team_message("leader", "hello")])

            middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")
            await middleware.drain_inbox()

            assert len(middleware._pending) == 1
            assert middleware._pending[0].content == "hello"

        asyncio.run(run())

    def test_drain_inbox_empty(self):
        async def run():
            bus = MagicMock()
            bus.drain = AsyncMock(return_value=[])

            middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")
            await middleware.drain_inbox()

            assert middleware._pending == []

        asyncio.run(run())

    def test_drain_inbox_calls_bus_with_correct_agent_id(self):
        async def run():
            bus = MagicMock()
            bus.drain = AsyncMock(return_value=[])

            middleware = TeamMessageMiddleware(message_bus=bus, agent_id="my-agent")
            await middleware.drain_inbox()

            bus.drain.assert_called_once_with(agent_id="my-agent")

        asyncio.run(run())


class TestTeamMessageMiddlewareBeforeModel:
    def test_before_model_no_pending_returns_no_changes(self):
        bus = MagicMock()
        middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")

        hook_input = make_hook_input(messages=[Message.user("hi")])
        result = middleware.before_model(hook_input)

        assert result.messages is None  # no_changes

    def test_before_model_injects_pending_messages(self):
        bus = MagicMock()
        middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")
        middleware._pending = [make_team_message("leader", "do the task")]

        original_msg = Message.user("hi")
        hook_input = make_hook_input(messages=[original_msg])
        result = middleware.before_model(hook_input)

        assert result.messages is not None
        assert len(result.messages) == 2
        # Original message preserved
        assert result.messages[0] == original_msg
        # Injected message is system role
        injected = result.messages[1]
        assert injected.role == Role.SYSTEM
        assert "[Team Message from leader]: do the task" in injected.get_text_content()

    def test_before_model_clears_pending_after_inject(self):
        bus = MagicMock()
        middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")
        middleware._pending = [make_team_message("leader", "msg")]

        hook_input = make_hook_input()
        middleware.before_model(hook_input)

        assert middleware._pending == []

    def test_before_model_injects_multiple_messages(self):
        bus = MagicMock()
        middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")
        middleware._pending = [
            make_team_message("leader", "msg1"),
            make_team_message("agent2", "msg2"),
        ]

        hook_input = make_hook_input(messages=[])
        result = middleware.before_model(hook_input)

        assert result.messages is not None
        assert len(result.messages) == 2
        assert "leader" in result.messages[0].get_text_content()
        assert "agent2" in result.messages[1].get_text_content()

    def test_before_model_preserves_existing_messages(self):
        bus = MagicMock()
        middleware = TeamMessageMiddleware(message_bus=bus, agent_id="agent1")
        middleware._pending = [make_team_message("leader", "task")]

        existing = [Message.user("original"), Message.assistant("response")]
        hook_input = make_hook_input(messages=existing)
        result = middleware.before_model(hook_input)

        assert result.messages is not None
        assert len(result.messages) == 3
        assert result.messages[0] == existing[0]
        assert result.messages[1] == existing[1]
