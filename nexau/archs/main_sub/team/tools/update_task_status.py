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

"""update_task_status tool — update task status on the task board.

RFC-0002: 更新任务状态（含交付物验证）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nexau.archs.main_sub.team.types import ToolError, UpdateTaskStatusResult, require_team_state

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent_state import AgentState


async def update_task_status(
    task_id: str,
    status: str,
    agent_state: AgentState,
    result_summary: str | None = None,
) -> UpdateTaskStatusResult | ToolError:
    """Update task status (pending -> in_progress -> completed).

    RFC-0002: 更新任务状态

    当状态变为 completed 时：
    1. 验证交付物文件存在（deliverable_path）
    2. 自动通知 leader agent 以便其检查进度或分配新任务
    """
    ts = require_team_state(agent_state)

    # RFC-0002: 完成任务时验证交付物文件存在
    if status == "completed":
        task_info = await ts.task_board.get_task_info(task_id)
        if task_info.deliverable_path is not None:
            sandbox = agent_state.get_sandbox()
            if sandbox is not None:
                deliverable_exists = sandbox.file_exists(task_info.deliverable_path)
                if not deliverable_exists:
                    return ToolError(
                        error=(
                            f"Deliverable file not found: {task_info.deliverable_path}. "
                            "Please write the deliverable document using write_file "
                            "before completing the task."
                        ),
                        code="invalid_state",
                    )

    await ts.task_board.update_status(
        task_id=task_id,
        status=status,
        result_summary=result_summary,
    )

    # RFC-0002: 任务完成时通知 leader，唤醒其 forever-run 等待循环
    if status == "completed" and not ts.is_leader:
        summary_text = f" Summary: {result_summary}" if result_summary else ""
        ts.team.notify_leader(
            content=f"Task '{task_id}' has been completed by {agent_state.agent_id}.{summary_text}",
            from_agent_id=agent_state.agent_id,
        )

    return UpdateTaskStatusResult(
        task_id=task_id,
        title=task_id,
        status=status,
        result_summary=result_summary,
    )
