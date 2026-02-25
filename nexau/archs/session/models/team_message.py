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

"""Team message data models.

RFC-0002: Agent Team 协作系统

Stores inter-agent messages within a team, including
direct messages and broadcasts.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class TeamMessageModel(SQLModel, table=True):
    """Team message model for inter-agent communication.

    RFC-0002: 团队消息模型

    Attributes:
        user_id: User identifier (primary key).
        session_id: Session identifier (primary key).
        team_id: Team identifier (primary key).
        message_id: Message UUID (primary key).
        from_agent_id: Sender agent ID.
        to_agent_id: Recipient agent ID (None for broadcast).
        content: Message content.
        message_type: Message type (text | idle_notification).
        delivered: Whether message has been delivered.
        delivered_at: Timestamp when message was delivered.
        created_at: Timestamp when message was created.
    """

    __tablename__ = "team_messages"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    message_id: str = Field(primary_key=True)

    from_agent_id: str
    to_agent_id: str | None = Field(default=None)
    content: str
    message_type: str = Field(default="text")
    delivered: bool = Field(default=False)
    delivered_at: datetime | None = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.now)
