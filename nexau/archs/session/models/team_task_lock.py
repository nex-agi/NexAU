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

"""Team task lock data models.

RFC-0002: Agent Team 协作系统

Stores distributed locks for team task assignment.
"""

from __future__ import annotations

from sqlmodel import Field, SQLModel


class TeamTaskLockModel(SQLModel, table=True):
    """Team task lock model for preventing concurrent task assignment.

    RFC-0002: 团队任务锁模型

    Attributes:
        user_id: User identifier (primary key).
        session_id: Session identifier (primary key).
        team_id: Team identifier (primary key).
        task_id: Task identifier (primary key).
        holder_id: Lock holder identifier, format "{pid}:{uuid}".
        acquired_at_ns: Nanosecond timestamp when lock was acquired.
        expires_at_ns: Nanosecond timestamp when lock expires.
    """

    __tablename__ = "team_task_locks"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    task_id: str = Field(primary_key=True)

    holder_id: str
    acquired_at_ns: int
    expires_at_ns: int
