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

"""Shared task board for AgentTeam collaboration.

RFC-0002: 共享任务面板

Provides CRUD operations for team tasks with dependency checking
and concurrent-safe claim/release via TaskLockService.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

from nexau.archs.session.models.team_task import TeamTaskModel
from nexau.archs.session.orm import AndFilter, ComparisonFilter
from nexau.archs.session.task_lock_service import LockConflictError

from .types import TaskBlockedError, TaskInfo

if TYPE_CHECKING:
    from nexau.archs.session.orm import DatabaseEngine
    from nexau.archs.session.task_lock_service import TaskLockService

logger = logging.getLogger(__name__)


def _slugify(text: str, max_length: int = 60) -> str:
    """Convert title to URL-safe slug for deliverable file paths.

    RFC-0002: 标题转 slug 用于交付物文件路径
    """
    # 1. 转小写，替换非字母数字为连字符
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    # 2. 去除首尾连字符
    slug = slug.strip("-")
    # 3. 截断到最大长度
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "task"


class TaskBoard:
    """Shared task board with dependency-aware task management.

    RFC-0002: 任务面板

    All task mutations (claim/release/update_status) are protected
    by TaskLockService to ensure concurrent safety.
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        task_lock_service: TaskLockService,
        user_id: str,
        session_id: str,
        team_id: str,
    ) -> None:
        self._engine = engine
        self._lock = task_lock_service
        self._user_id = user_id
        self._session_id = session_id
        self._team_id = team_id

    # --- helpers ---

    def _team_filters(self) -> list[ComparisonFilter]:
        """Return common team-scoped filters."""
        return [
            ComparisonFilter.eq("user_id", self._user_id),
            ComparisonFilter.eq("session_id", self._session_id),
            ComparisonFilter.eq("team_id", self._team_id),
        ]

    async def _next_task_id(self) -> str:
        """Generate next task ID (T-001, T-002, etc.).

        RFC-0002: 自增任务 ID 生成
        """
        existing = await self._engine.count(
            TeamTaskModel,
            filters=AndFilter(filters=self._team_filters()),
        )
        return f"T-{existing + 1:03d}"

    async def _get_task(self, task_id: str) -> TeamTaskModel:
        """Get task by ID, raise ValueError if not found."""
        task = await self._engine.find_first(
            TeamTaskModel,
            filters=AndFilter(
                filters=[
                    *self._team_filters(),
                    ComparisonFilter.eq("task_id", task_id),
                ]
            ),
        )
        if task is None:
            raise ValueError(f"Task {task_id} not found")
        return task

    async def _is_blocked(self, task: TeamTaskModel) -> bool:
        """Check if task has unfinished dependencies.

        RFC-0002: 依赖阻塞检查
        """
        if not task.dependencies:
            return False
        for dep_id in task.dependencies:
            dep = await self._get_task(dep_id)
            if dep.status != "completed":
                return True
        return False

    @staticmethod
    def _to_task_info(task: TeamTaskModel, *, is_blocked: bool) -> TaskInfo:
        """Convert model to TaskInfo dataclass."""
        return TaskInfo(
            task_id=task.task_id,
            title=task.title,
            description=task.description,
            status=task.status,
            priority=task.priority,
            dependencies=task.dependencies,
            assignee_agent_id=task.assignee_agent_id,
            result_summary=task.result_summary,
            created_by=task.created_by,
            is_blocked=is_blocked,
            deliverable_path=task.deliverable_path,
        )

    # --- public API ---

    async def get_task_info(self, task_id: str) -> TaskInfo:
        """Get a single task as TaskInfo.

        RFC-0002: 获取单个任务信息（含交付物路径）
        """
        task = await self._get_task(task_id)
        blocked = await self._is_blocked(task)
        return self._to_task_info(task, is_blocked=blocked)

    async def create_task(
        self,
        *,
        title: str,
        description: str = "",
        priority: int = 0,
        dependencies: list[str] | None = None,
        created_by: str = "",
    ) -> TaskInfo:
        """Create a new task on the board.

        RFC-0002: 创建任务（含交付物路径生成）
        """
        task_id = await self._next_task_id()
        # 生成交付物文件路径
        slug = _slugify(title)
        deliverable_path = f".nexau/tasks/{task_id}-{slug}.md"

        task = TeamTaskModel(
            user_id=self._user_id,
            session_id=self._session_id,
            team_id=self._team_id,
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            dependencies=dependencies or [],
            created_by=created_by,
            deliverable_path=deliverable_path,
        )
        await self._engine.create(task)
        blocked = await self._is_blocked(task)
        logger.info(f"Task created: {task_id} title={title} deliverable={deliverable_path}")
        return self._to_task_info(task, is_blocked=blocked)

    async def list_tasks(self, *, status: str | None = None) -> list[TaskInfo]:
        """List tasks, optionally filtered by status.

        RFC-0002: 列出任务（含阻塞状态计算）
        """
        filters_list = self._team_filters()
        if status is not None:
            filters_list.append(ComparisonFilter.eq("status", status))

        tasks = await self._engine.find_many(
            TeamTaskModel,
            filters=AndFilter(filters=filters_list),
        )

        # 1. 收集已完成任务 ID 集合，用于快速判断阻塞
        completed_ids = {t.task_id for t in tasks if t.status == "completed"}

        # 2. 逐任务计算 is_blocked
        results: list[TaskInfo] = []
        for t in tasks:
            blocked = bool(t.dependencies and not all(d in completed_ids for d in t.dependencies))
            results.append(self._to_task_info(t, is_blocked=blocked))
        return results

    async def claim_task(
        self,
        *,
        task_id: str,
        assignee_agent_id: str,
    ) -> None:
        """Claim a task (assign to agent).

        RFC-0002: 认领任务

        Uses lock for concurrent safety.
        Raises TaskBlockedError if dependencies not met.
        Raises LockConflictError if lock held by another.
        """
        async with self._lock.acquire(
            user_id=self._user_id,
            session_id=self._session_id,
            team_id=self._team_id,
            task_id=task_id,
        ):
            task = await self._get_task(task_id)
            # 1. 检查依赖是否满足
            if await self._is_blocked(task):
                raise TaskBlockedError(f"Task {task_id} has unfinished dependencies")
            # 2. 检查是否已被认领
            if task.assignee_agent_id is not None:
                raise LockConflictError(f"Task {task_id} already assigned to {task.assignee_agent_id}")
            # 3. 分配
            task.assignee_agent_id = assignee_agent_id
            task.status = "in_progress"
            task.updated_at = datetime.now()
            await self._engine.update(task)
            logger.info(f"Task claimed: {task_id} by {assignee_agent_id}")

    async def release_task(self, *, task_id: str) -> None:
        """Release a claimed task (unassign).

        RFC-0002: 释放任务
        """
        async with self._lock.acquire(
            user_id=self._user_id,
            session_id=self._session_id,
            team_id=self._team_id,
            task_id=task_id,
        ):
            task = await self._get_task(task_id)
            task.assignee_agent_id = None
            task.status = "pending"
            task.updated_at = datetime.now()
            await self._engine.update(task)
            logger.info(f"Task released: {task_id}")

    async def update_status(
        self,
        *,
        task_id: str,
        status: str,
        result_summary: str | None = None,
    ) -> None:
        """Update task status.

        RFC-0002: 更新任务状态
        """
        async with self._lock.acquire(
            user_id=self._user_id,
            session_id=self._session_id,
            team_id=self._team_id,
            task_id=task_id,
        ):
            task = await self._get_task(task_id)
            task.status = status
            if result_summary is not None:
                task.result_summary = result_summary
            task.updated_at = datetime.now()
            await self._engine.update(task)
            logger.info(f"Task status updated: {task_id} -> {status}")
