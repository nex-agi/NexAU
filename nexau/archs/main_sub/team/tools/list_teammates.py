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

"""list_teammates tool — list all teammates and their status.

RFC-0002: 列出队友
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import TeammateInfo, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def list_teammates(
    agent_state: AgentState,
) -> list[TeammateInfo]:
    """List all teammate agents and their current status.

    RFC-0002: 列出队友

    返回所有 teammate 的 agent_id、role_name 和 status。
    """
    ts = require_team_state(agent_state)
    return ts.team.get_teammate_info()
