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

"""Task lock service for team task operations.

RFC-0002: 任务锁服务

Short-lived DB-backed TTL lock for task claim/release/update critical sections.
Unlike AgentLockService, no heartbeat is needed (locks are held for < 5s).

Design principles:
- TTL short-hold: default 5s, no heartbeat needed
- No waiting: acquire fails immediately on conflict
- Cross-engine compatible: works with InMemory/SQL/Remote engines
- Fast release: lock is deleted when critical section exits
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from nexau.archs.session.models.team_task_lock import TeamTaskLockModel
from nexau.archs.session.orm import AndFilter, ComparisonFilter

if TYPE_CHECKING:
    from .orm import DatabaseEngine

logger = logging.getLogger(__name__)


class LockConflictError(Exception):
    """Raised when a task lock cannot be acquired due to conflict."""


class TaskLockService:
    """Short-lived DB-backed TTL lock for task operations.

    RFC-0002: 任务操作临界区保护

    Design:
    - TTL short-hold: default 5s, no heartbeat needed
    - No waiting: acquire fails immediately on conflict
    - Cross-engine compatible: works with InMemory/SQL/Remote engines

    Example:
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>> engine = InMemoryDatabaseEngine()
        >>> lock_service = TaskLockService(engine=engine)
        >>> async with lock_service.acquire(
        ...     user_id="u1",
        ...     session_id="s1",
        ...     team_id="t1",
        ...     task_id="task1",
        ... ):
        ...     # Critical section - only one holder at a time
        ...     await claim_task()
    """

    def __init__(self, *, engine: DatabaseEngine, lock_ttl: float = 5.0) -> None:
        """Initialize task lock service.

        Args:
            engine: DatabaseEngine instance for lock storage
            lock_ttl: Lock time-to-live in seconds (default: 5s)
        """
        self._engine = engine
        self._lock_ttl = lock_ttl
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Ensure the locks table is initialized."""
        if not self._initialized:
            logger.debug("Initializing TaskLockService database models")
            await self._engine.setup_models([TeamTaskLockModel])
            self._initialized = True
            logger.info("TaskLockService initialized successfully")

    def _generate_holder_id(self) -> str:
        """Generate unique holder ID for this lock acquisition.

        Format: "{process_id}:{random_uuid}"
        Example: "12345:a1b2c3d4"
        """
        return f"{os.getpid()}:{uuid.uuid4().hex[:8]}"

    def _calculate_expires_at_ns(self) -> int:
        """Calculate expiration timestamp (now + TTL)."""
        return time.time_ns() + int(self._lock_ttl * 1_000_000_000)

    async def _find_valid_lock(
        self,
        user_id: str,
        session_id: str,
        team_id: str,
        task_id: str,
    ) -> TeamTaskLockModel | None:
        """Find a valid (non-expired) lock for the given task.

        Returns:
            Lock if exists and not expired, None otherwise
        """
        lock = await self._engine.find_first(
            TeamTaskLockModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", user_id),
                    ComparisonFilter.eq("session_id", session_id),
                    ComparisonFilter.eq("team_id", team_id),
                    ComparisonFilter.eq("task_id", task_id),
                ]
            ),
        )

        if not lock:
            return None

        # Check if expired
        now = time.time_ns()
        if lock.expires_at_ns <= now:
            logger.debug(f"Found expired task lock for team={team_id}, task={task_id}, holder={lock.holder_id}")
            return None

        return lock

    @asynccontextmanager
    async def acquire(
        self,
        *,
        user_id: str,
        session_id: str,
        team_id: str,
        task_id: str,
    ) -> AsyncGenerator[None, None]:
        """Acquire task lock for critical section.

        RFC-0002: 获取任务锁

        Args:
            user_id: User identifier
            session_id: Session identifier
            team_id: Team identifier
            task_id: Task identifier

        Yields:
            None when lock is acquired

        Raises:
            LockConflictError: If lock is held by another holder
        """
        await self._ensure_initialized()

        holder_id = self._generate_holder_id()
        now = time.time_ns()
        expires_at_ns = self._calculate_expires_at_ns()

        logger.info(f"Attempting to acquire task lock for team={team_id}, task={task_id}, holder={holder_id}")

        # 1. Check existing valid lock
        existing = await self._find_valid_lock(user_id, session_id, team_id, task_id)
        if existing:
            logger.warning(f"Task lock conflict: task {task_id} in team {team_id} is locked by holder={existing.holder_id}")
            raise LockConflictError(f"Task {task_id} in team {team_id} is already locked (holder: {existing.holder_id})")

        # 2. Clean up expired lock and create new lock record
        await self._engine.delete(
            TeamTaskLockModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", user_id),
                    ComparisonFilter.eq("session_id", session_id),
                    ComparisonFilter.eq("team_id", team_id),
                    ComparisonFilter.eq("task_id", task_id),
                    ComparisonFilter.lt("expires_at_ns", now),
                ]
            ),
        )

        lock = TeamTaskLockModel(
            user_id=user_id,
            session_id=session_id,
            team_id=team_id,
            task_id=task_id,
            holder_id=holder_id,
            acquired_at_ns=now,
            expires_at_ns=expires_at_ns,
        )
        await self._engine.create(lock)
        logger.info(f"Task lock acquired for team={team_id}, task={task_id}, holder={holder_id}")

        # 3. yield (critical section)
        try:
            yield
        finally:
            # 4. Delete lock (match holder_id to avoid releasing someone else's lock)
            logger.debug(f"Releasing task lock for team={team_id}, task={task_id}, holder={holder_id}")
            deleted_count = await self._engine.delete(
                TeamTaskLockModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("user_id", user_id),
                        ComparisonFilter.eq("session_id", session_id),
                        ComparisonFilter.eq("team_id", team_id),
                        ComparisonFilter.eq("task_id", task_id),
                        ComparisonFilter.eq("holder_id", holder_id),
                    ]
                ),
            )
            if deleted_count > 0:
                logger.info(f"Task lock released for team={team_id}, task={task_id}, holder={holder_id}")
            else:
                logger.warning(f"Task lock already released or expired for team={team_id}, task={task_id}, holder={holder_id}")
