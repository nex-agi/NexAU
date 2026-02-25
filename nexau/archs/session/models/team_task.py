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

"""Team task data models.

RFC-0002: Agent Team 协作系统

Stores team tasks with priority, dependencies, and assignment.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON
from sqlmodel import Column, Field, SQLModel


class TeamTaskModel(SQLModel, table=True):
    """Team task model for tracking work items within a team.

    RFC-0002: 团队任务模型

    Attributes:
        user_id: User identifier (primary key).
        session_id: Session identifier (primary key).
        team_id: Team identifier (primary key).
        task_id: Task identifier, format "T-001" (primary key).
        title: Task title.
        description: Task description.
        priority: Priority level (0=normal, 1=high, 2=critical).
        status: Task status (pending | in_progress | completed).
        dependencies: List of task IDs this task depends on.
        assignee_agent_id: Agent ID assigned to this task.
        result_summary: Summary of task result.
        deliverable_path: Relative path to deliverable document (.nexau/tasks/{task_id}-{slug}.md).
        created_by: Agent ID that created this task.
        created_at: Timestamp when task was created.
        updated_at: Timestamp when task was last updated.
    """

    __tablename__ = "team_tasks"  # type: ignore[assignment]

    user_id: str = Field(primary_key=True)
    session_id: str = Field(primary_key=True)
    team_id: str = Field(primary_key=True)
    task_id: str = Field(primary_key=True)

    title: str
    description: str = ""
    priority: int = Field(default=0)
    status: str = Field(default="pending")
    dependencies: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    assignee_agent_id: str | None = Field(default=None)
    result_summary: str | None = Field(default=None)
    deliverable_path: str | None = Field(default=None)
    created_by: str = ""

    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
