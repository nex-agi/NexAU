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

"""Team data models.

RFC-0002: Agent Team 协作系统

Stores team configuration including leader, candidate roles,
and max teammate limits.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class TeamModel(SQLModel, table=True):
    """Team model for agent team collaboration.

    RFC-0002: 团队配置模型

    Attributes:
        user_id: User identifier (primary key).
        session_id: Session identifier (primary key).
        team_id: Team identifier (primary key).
        leader_agent_id: Agent ID of the team leader.
        candidates: Role name to agent config ref mapping.
        max_teammates: Maximum number of teammates allowed.
        created_at: Timestamp when team was created.
        updated_at: Timestamp when team was last updated.
    """

    __tablename__ = "teams"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)

    leader_agent_id: str
    candidates: dict[str, str] = Field(default_factory=dict, sa_column=Column(JSON))
    max_teammates: int = Field(default=10)

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
