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

"""Team collaboration types and exceptions.

RFC-0002: AgentTeam 协作类型定义

Provides strongly-typed result dataclasses for Team Tools
and custom exceptions for team operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState
    from nexau.archs.main_sub.team.state import AgentTeamState

# --- Helpers ---


def require_team_state(agent_state: AgentState) -> AgentTeamState:
    """Extract team_state or raise if not in a team context.

    RFC-0002: Team 上下文校验辅助函数
    """
    ts = agent_state.team_state
    if ts is None:
        raise RuntimeError("This tool requires a team context (agent_state.team_state is None)")
    return ts


# --- Exceptions ---


class MaxTeammatesError(Exception):
    """Raised when spawn_teammate exceeds max_teammates limit."""


class TaskBlockedError(Exception):
    """Raised when attempting to claim a blocked task."""


# --- Result Types ---


@dataclass(frozen=True)
class TeammateInfo:
    """Teammate instance info.

    RFC-0002: Teammate 状态信息
    """

    agent_id: str
    role_name: str
    status: str  # idle | running | stopped


@dataclass(frozen=True)
class TaskInfo:
    """Task board entry.

    RFC-0002: 任务信息
    """

    task_id: str
    title: str
    description: str
    status: str  # pending | in_progress | completed
    priority: int
    dependencies: list[str]
    assignee_agent_id: str | None
    result_summary: str | None
    created_by: str
    is_blocked: bool
    deliverable_path: str | None


@dataclass(frozen=True)
class SpawnResult:
    """spawn_teammate return value.

    RFC-0002: Teammate 实例化结果
    """

    agent_id: str
    role_name: str


@dataclass(frozen=True)
class RemoveTeammateResult:
    """remove_teammate return value.

    RFC-0002: Teammate 移除结果
    """

    agent_id: str
    role_name: str


@dataclass(frozen=True)
class CreateTaskResult:
    """create_task return value.

    RFC-0002: 任务创建结果
    """

    task_id: str
    title: str
    description: str
    priority: int
    status: str
    deliverable_path: str


@dataclass(frozen=True)
class ClaimTaskResult:
    """claim_task return value.

    RFC-0002: 任务领取结果
    """

    task_id: str
    title: str
    status: str
    assignee_agent_id: str
    deliverable_path: str | None


@dataclass(frozen=True)
class UpdateTaskStatusResult:
    """update_task_status return value.

    RFC-0002: 任务状态更新结果
    """

    task_id: str
    title: str
    status: str
    result_summary: str | None = None


@dataclass(frozen=True)
class ReleaseTaskResult:
    """release_task return value.

    RFC-0002: 任务释放结果
    """

    task_id: str
    title: str
    status: str


@dataclass(frozen=True)
class MessageResult:
    """message / broadcast return value.

    RFC-0002: 消息发送结果
    """

    message_id: str
    delivered_to: list[str] = field(default_factory=lambda: list[str]())


@dataclass(frozen=True)
class FinishTeamResult:
    """finish_team return value.

    RFC-0002: 团队结束结果
    """

    summary: str
    completed_tasks: int
    total_tasks: int


@dataclass(frozen=True)
class ToolError:
    """Tool operation error return.

    RFC-0002: 工具操作错误

    Codes: permission_denied | conflict | blocked | not_found | invalid_state

    ``status`` 固定为 ``"error"``，用于 executor 的 stop-tool 守卫：
    当 stop tool 返回 error 时，executor 不会退出循环。
    """

    error: str
    code: str
    status: str = "error"
