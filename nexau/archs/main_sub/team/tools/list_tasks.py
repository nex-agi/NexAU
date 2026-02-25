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

"""list_tasks tool — list tasks on the shared task board.

RFC-0002: 列出共享任务列表
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import TaskInfo, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def list_tasks(
    agent_state: AgentState,
    status: str | None = None,
) -> list[TaskInfo]:
    """List tasks on the shared task board.

    RFC-0002: 列出共享任务列表

    Args:
        status: 可选过滤条件 (pending / in_progress / completed)
    """
    ts = require_team_state(agent_state)
    return await ts.task_board.list_tasks(status=status)
