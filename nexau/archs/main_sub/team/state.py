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

"""Team collaboration context container.

RFC-0002: Team 上下文容器

Lightweight data object stored on AgentState.team_state,
giving team tools typed access to shared services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexau.archs.main_sub.team.agent_team import AgentTeam
    from nexau.archs.main_sub.team.message_bus import TeamMessageBus
    from nexau.archs.main_sub.team.task_board import TaskBoard


class AgentTeamState:
    """Team collaboration context attached to AgentState.

    RFC-0002: Team 上下文容器

    Tools access this via ``agent_state.team_state``.
    """

    __slots__ = ("team", "task_board", "message_bus", "is_leader")

    def __init__(
        self,
        *,
        team: AgentTeam,
        task_board: TaskBoard,
        message_bus: TeamMessageBus,
        is_leader: bool,
    ) -> None:
        self.team = team
        self.task_board = task_board
        self.message_bus = message_bus
        self.is_leader = is_leader
