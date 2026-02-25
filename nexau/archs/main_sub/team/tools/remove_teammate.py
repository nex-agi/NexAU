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

"""remove_teammate tool — remove a teammate instance from the team.

RFC-0002: 移除 teammate 实例
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import RemoveTeammateResult, ToolError, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def remove_teammate(
    agent_id: str,
    agent_state: AgentState,
) -> RemoveTeammateResult | ToolError:
    """Remove a teammate instance.

    RFC-0002: 移除 teammate 实例

    仅可移除 idle 状态的 teammate。正在执行任务的 teammate
    需先完成或被 stop 后才能移除。
    """
    ts = require_team_state(agent_state)
    if not ts.is_leader:
        return ToolError(
            error="Only leader can remove teammates",
            code="permission_denied",
        )

    await ts.team.remove_teammate(agent_id)
    return RemoveTeammateResult(agent_id=agent_id, role_name=agent_id)
