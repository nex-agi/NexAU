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

"""Team message injection middleware.

RFC-0002: 迭代边界消息注入

Injects pending team messages into agent history at iteration
boundaries. Messages are pre-drained into a buffer by the AgentTeam
before each model call (async), then read synchronously by the
before_model hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.execution.hooks import (
    BeforeModelHookInput,
    HookResult,
    Middleware,
)
from nexau.core.messages import Message, Role, TextBlock

if TYPE_CHECKING:
    from nexau.archs.main_sub.team.message_bus import TeamMessageBus
    from nexau.archs.session.models.team_message import TeamMessageModel


class TeamMessageMiddleware(Middleware):
    """Injects pending team messages into agent history at iteration boundaries.

    RFC-0002: 迭代边界消息注入

    Messages are pre-drained into a buffer by the AgentTeam before each
    model call. The before_model hook reads from this buffer synchronously.
    """

    def __init__(self, *, message_bus: TeamMessageBus, agent_id: str) -> None:
        self._bus = message_bus
        self._agent_id = agent_id
        self._pending: list[TeamMessageModel] = []

    async def drain_inbox(self) -> None:
        """Pre-drain messages from DB into local buffer.

        RFC-0002: 预拉取消息到本地缓冲

        Called by AgentTeam before each iteration.
        """
        self._pending = await self._bus.drain(agent_id=self._agent_id)

    def before_model(self, hook_input: BeforeModelHookInput) -> HookResult:
        """Inject buffered messages into history.

        RFC-0002: 同步注入缓冲消息到历史
        """
        if not self._pending:
            return HookResult.no_changes()

        new_messages = list(hook_input.messages)
        for msg in self._pending:
            new_messages.append(
                Message(
                    role=Role.SYSTEM,
                    content=[TextBlock(text=f"[Team Message from {msg.from_agent_id}]: {msg.content}")],
                )
            )
        self._pending.clear()
        return HookResult(messages=new_messages)
