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

"""Team collaboration module.

RFC-0002: Agent Team 协作系统

Uses lazy imports to avoid circular dependency with session module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .types import (
    ClaimTaskResult,
    CreateTaskResult,
    MaxTeammatesError,
    MessageResult,
    ReleaseTaskResult,
    RemoveTeammateResult,
    SpawnResult,
    TaskBlockedError,
    TaskInfo,
    TeammateInfo,
    ToolError,
    UpdateTaskStatusResult,
)

if TYPE_CHECKING:
    from nexau.archs.session.task_lock_service import LockConflictError as LockConflictError

    from .agent_team import AgentTeam as AgentTeam
    from .message_bus import TeamMessageBus as TeamMessageBus
    from .state import AgentTeamState as AgentTeamState
    from .task_board import TaskBoard as TaskBoard
    from .watchdog import TeammateWatchdog as TeammateWatchdog
    from .watchdog import WatchdogConfig as WatchdogConfig

__all__ = [
    "AgentTeam",
    "AgentTeamState",
    "ClaimTaskResult",
    "CreateTaskResult",
    "LockConflictError",
    "MaxTeammatesError",
    "MessageResult",
    "ReleaseTaskResult",
    "RemoveTeammateResult",
    "SpawnResult",
    "TaskBlockedError",
    "TaskBoard",
    "TaskInfo",
    "TeamMessageBus",
    "TeammateInfo",
    "TeammateWatchdog",
    "ToolError",
    "UpdateTaskStatusResult",
    "WatchdogConfig",
]


def __getattr__(name: str) -> object:
    """Lazy imports to break circular dependency chain."""
    if name == "AgentTeam":
        from .agent_team import AgentTeam

        return AgentTeam
    if name == "AgentTeamState":
        from .state import AgentTeamState

        return AgentTeamState
    if name == "LockConflictError":
        from nexau.archs.session.task_lock_service import LockConflictError

        return LockConflictError
    if name == "TaskBoard":
        from .task_board import TaskBoard

        return TaskBoard
    if name == "TeamMessageBus":
        from .message_bus import TeamMessageBus

        return TeamMessageBus
    if name == "TeammateWatchdog":
        from .watchdog import TeammateWatchdog

        return TeammateWatchdog
    if name == "WatchdogConfig":
        from .watchdog import WatchdogConfig

        return WatchdogConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
