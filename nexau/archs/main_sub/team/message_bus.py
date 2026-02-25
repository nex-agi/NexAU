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

"""Team message bus for intra-team communication.

RFC-0002: 队内消息总线

Provides point-to-point and broadcast messaging between team agents,
with persistent storage and delivery tracking.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from nexau.archs.session.models.team_message import TeamMessageModel
from nexau.archs.session.orm import AndFilter, ComparisonFilter

if TYPE_CHECKING:
    from nexau.archs.session.orm import DatabaseEngine

logger = logging.getLogger(__name__)


class TeamMessageBus:
    """DB-backed message bus for team agent communication.

    RFC-0002: 队内消息投递

    Messages are persisted to TeamMessageModel and delivered
    at iteration boundaries via TeamMessageMiddleware.
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        user_id: str,
        session_id: str,
        team_id: str,
    ) -> None:
        self._engine = engine
        self._user_id = user_id
        self._session_id = session_id
        self._team_id = team_id

        # Agent delivery callbacks — set via set_agent_delivery() after construction.
        # Avoids circular import with AgentTeam.
        self._deliver_message: Callable[[str, str, str], None] | None = None
        self._get_broadcast_recipients: Callable[[], list[str]] | None = None

    def set_agent_delivery(
        self,
        *,
        deliver_message: Callable[[str, str, str], None],
        get_broadcast_recipients: Callable[[], list[str]],
    ) -> None:
        """Wire agent delivery so send/broadcast also enqueue messages.

        RFC-0002: 消息投递回调注入

        Args:
            deliver_message: (to_agent_id, content, from_agent_id) -> None
            get_broadcast_recipients: () -> list of teammate agent_ids
        """
        self._deliver_message = deliver_message
        self._get_broadcast_recipients = get_broadcast_recipients

    # --- helpers ---

    def _team_filters(self) -> list[ComparisonFilter]:
        """Return common team-scoped filters."""
        return [
            ComparisonFilter.eq("user_id", self._user_id),
            ComparisonFilter.eq("session_id", self._session_id),
            ComparisonFilter.eq("team_id", self._team_id),
        ]

    # --- public API ---

    async def send(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        content: str,
        message_type: str = "text",
    ) -> TeamMessageModel:
        """Send a point-to-point message and deliver to target agent.

        RFC-0002: 发送点对点消息

        Persists the message to DB, then enqueues it to the target agent
        via the delivery callback (if wired).
        """
        msg = TeamMessageModel(
            user_id=self._user_id,
            session_id=self._session_id,
            team_id=self._team_id,
            message_id=uuid4().hex,
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            content=content,
            message_type=message_type,
        )
        await self._engine.create(msg)
        logger.info(f"Message sent: {msg.message_id} from={from_agent_id} to={to_agent_id}")

        # 通过 enqueue_message 唤醒目标 agent
        if self._deliver_message is not None:
            self._deliver_message(to_agent_id, content, from_agent_id)

        return msg

    async def broadcast(
        self,
        *,
        from_agent_id: str,
        content: str,
    ) -> TeamMessageModel:
        """Broadcast a message to all teammates and deliver immediately.

        RFC-0002: 广播消息

        The broadcast message has to_agent_id=None.
        After persisting, enqueues the message to all teammates
        (excluding sender) via the delivery callback.
        """
        msg = TeamMessageModel(
            user_id=self._user_id,
            session_id=self._session_id,
            team_id=self._team_id,
            message_id=uuid4().hex,
            from_agent_id=from_agent_id,
            to_agent_id=None,
            content=content,
            message_type="text",
        )
        await self._engine.create(msg)
        logger.info(f"Broadcast sent: {msg.message_id} from={from_agent_id}")

        # 通过 enqueue_message 唤醒所有 teammate（排除发送者）
        if self._deliver_message is not None and self._get_broadcast_recipients is not None:
            for agent_id in self._get_broadcast_recipients():
                if agent_id != from_agent_id:
                    self._deliver_message(agent_id, content, from_agent_id)

        return msg

    async def drain(self, *, agent_id: str) -> list[TeamMessageModel]:
        """Fetch undelivered messages for agent, mark as delivered.

        RFC-0002: 拉取未投递消息

        Returns messages targeted to this agent (direct + broadcast,
        excluding self-sent broadcasts).
        """
        # 1. 查找直接发给该 agent 的未投递消息
        direct_msgs = await self._engine.find_many(
            TeamMessageModel,
            filters=AndFilter(
                filters=[
                    *self._team_filters(),
                    ComparisonFilter.eq("to_agent_id", agent_id),
                    ComparisonFilter.eq("delivered", False),
                ]
            ),
        )

        # 2. 查找广播消息（to_agent_id 为 None）
        broadcast_msgs = await self._engine.find_many(
            TeamMessageModel,
            filters=AndFilter(
                filters=[
                    *self._team_filters(),
                    ComparisonFilter.eq("to_agent_id", None),
                    ComparisonFilter.eq("delivered", False),
                ]
            ),
        )
        # 过滤掉自己发送的广播
        broadcast_msgs = [m for m in broadcast_msgs if m.from_agent_id != agent_id]

        all_msgs = direct_msgs + broadcast_msgs

        # 3. 标记为已投递
        now = datetime.now()
        for msg in all_msgs:
            msg.delivered = True
            msg.delivered_at = now
            await self._engine.update(msg)

        if all_msgs:
            logger.info(f"Drained {len(all_msgs)} messages for {agent_id} (direct={len(direct_msgs)}, broadcast={len(broadcast_msgs)})")
        return all_msgs
