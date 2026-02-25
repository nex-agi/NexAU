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

"""Team member data models.

RFC-0002: Agent Team 协作系统

Stores team member state including role and status.
"""

from __future__ import annotations

from datetime import datetime

from sqlmodel import Field, SQLModel


class TeamMemberModel(SQLModel, table=True):
    """Team member model for tracking agent membership in teams.

    RFC-0002: 团队成员模型

    Attributes:
        user_id: User identifier (primary key).
        session_id: Team session identifier (primary key). Scopes team data.
        team_id: Team identifier (primary key).
        agent_id: Agent identifier (primary key).
        member_session_id: Agent's own independent session_id for history/state isolation.
        role_name: Role name assigned to this member.
        status: Member status (idle | running | stopped).
        created_at: Timestamp when member was added.
        updated_at: Timestamp when member was last updated.
    """

    __tablename__ = "team_members"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    agent_id: str = Field(primary_key=True)

    # RFC-0002: agent 独立 session，用于 history/state 隔离和 team 恢复
    member_session_id: str = Field(default="")

    role_name: str
    status: str = Field(default="idle")

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
