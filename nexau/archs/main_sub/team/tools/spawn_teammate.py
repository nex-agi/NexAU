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

"""spawn_teammate tool — instantiate a new teammate from candidates.

RFC-0002: 从 candidates 动态实例化 teammate
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import (
    MaxTeammatesError,
    SpawnResult,
    ToolError,
    require_team_state,
)

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def spawn_teammate(
    role_name: str,
    agent_state: AgentState,
) -> SpawnResult | ToolError:
    """Spawn a new teammate instance from candidates.

    RFC-0002: 从 candidates 动态实例化 teammate

    Leader 根据任务需求调用此工具，从预配置的 candidates 中
    实例化一个新的 teammate agent。同一 role 可多次 spawn
    产生多个实例（如 coder-1, coder-2）。
    """
    ts = require_team_state(agent_state)
    if not ts.is_leader:
        return ToolError(
            error="Only leader can spawn teammates",
            code="permission_denied",
        )

    try:
        agent_id = await ts.team.spawn_teammate(role_name)
    except MaxTeammatesError:
        return ToolError(
            error=f"Max teammates limit reached ({ts.team.max_teammates})",
            code="invalid_state",
        )
    return SpawnResult(agent_id=agent_id, role_name=role_name)
