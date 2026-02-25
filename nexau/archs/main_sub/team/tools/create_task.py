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

"""create_task tool — create a new task on the shared task board.

RFC-0002: 创建任务（仅 leader 可调用）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import CreateTaskResult, ToolError, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def create_task(
    title: str,
    agent_state: AgentState,
    description: str = "",
    priority: int = 0,
    dependencies: list[str] | None = None,
) -> CreateTaskResult | ToolError:
    """Create a new task on the shared task board.

    RFC-0002: 创建任务（仅 leader 可调用）
    """
    ts = require_team_state(agent_state)
    if not ts.is_leader:
        return ToolError(
            error="Only leader can create tasks",
            code="permission_denied",
        )

    task = await ts.task_board.create_task(
        title=title,
        description=description,
        priority=priority,
        dependencies=dependencies or [],
        created_by=agent_state.agent_id,
    )
    return CreateTaskResult(
        task_id=task.task_id,
        title=task.title,
        description=task.description,
        priority=task.priority,
        status="created",
        deliverable_path=task.deliverable_path or "",
    )
