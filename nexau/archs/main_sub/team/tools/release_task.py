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

"""release_task tool — release a claimed task back to the board.

RFC-0002: 释放任务
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import ReleaseTaskResult, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def release_task(
    task_id: str,
    agent_state: AgentState,
) -> ReleaseTaskResult:
    """Release a claimed task (unassign).

    RFC-0002: 释放任务

    任务状态恢复为 pending，可被其他 teammate 领取。
    """
    ts = require_team_state(agent_state)
    await ts.task_board.release_task(task_id=task_id)
    return ReleaseTaskResult(task_id=task_id, title=task_id, status="released")
