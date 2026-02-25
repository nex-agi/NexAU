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

"""message tool — send a point-to-point message to a teammate.

RFC-0002: 队内点对点消息
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import MessageResult, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def message(
    to_agent_id: str,
    content: str,
    agent_state: AgentState,
) -> MessageResult:
    """Send a message to a specific teammate.

    RFC-0002: 队内点对点消息

    消息持久化到 message bus 并通过 enqueue_message 注入目标 agent。
    目标 agent 的 executor 会被 _message_available 事件唤醒。
    """
    ts = require_team_state(agent_state)
    msg = await ts.message_bus.send(
        from_agent_id=agent_state.agent_id,
        to_agent_id=to_agent_id,
        content=content,
    )

    return MessageResult(
        message_id=msg.message_id,
        delivered_to=[to_agent_id],
    )
