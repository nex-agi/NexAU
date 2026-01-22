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

"""Agent lock service with automatic heartbeat renewal.

This module provides AgentLockService which uses DatabaseEngine to store locks.
It supports automatic heartbeat renewal and short TTL for fast recovery.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .models.agent_lock import AgentLockModel
from .orm import AndFilter, ComparisonFilter

if TYPE_CHECKING:
    from .orm import DatabaseEngine

logger = logging.getLogger(__name__)


class AgentLockService:
    """Agent lock service with automatic heartbeat renewal.

    Features:
    - Uses DatabaseEngine for storage (works with any engine)
    - Automatic heartbeat every heartbeat_interval seconds
    - Lock expires after lock_ttl seconds without heartbeat
    - Background cleanup task removes expired locks
    - No waiting: fails immediately if agent is busy

    Example:
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>> engine = InMemoryDatabaseEngine()
        >>> lock_service = AgentLockService(engine=engine)
        >>> async with lock_service.acquire("session1", "agent1"):
        ...     # Only one execution per (session_id, agent_id) at a time
        ...     # Lock is automatically renewed every 10s
        ...     await long_running_task()
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        lock_ttl: float = 30.0,
        heartbeat_interval: float = 10.0,
    ):
        """Initialize agent lock service.

        Args:
            engine: DatabaseEngine instance for lock storage
            lock_ttl: Lock time-to-live in seconds (default: 30s)
            heartbeat_interval: Heartbeat interval in seconds (default: 10s)
                Should be < lock_ttl / 2 for safety

        Raises:
            ValueError: If heartbeat_interval >= lock_ttl / 2
        """
        if heartbeat_interval >= lock_ttl / 2:
            raise ValueError(f"heartbeat_interval ({heartbeat_interval}s) must be < lock_ttl/2 ({lock_ttl / 2}s)")

        self._engine = engine
        self._lock_ttl = lock_ttl
        self._heartbeat_interval = heartbeat_interval
        self._cleanup_task: asyncio.Task[None] | None = None
        self._cleanup_lock = asyncio.Lock()
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure the locks table is initialized."""
        if not self._initialized:
            logger.debug("Initializing AgentLockService database models")
            await self._engine.setup_models([AgentLockModel])
            self._initialized = True
            logger.info("AgentLockService initialized successfully")

    def _generate_holder_id(self) -> str:
        """Generate unique holder ID for this lock acquisition.

        Format: "{process_id}:{random_uuid}"
        Example: "12345:a1b2c3d4"
        """
        return f"{os.getpid()}:{uuid.uuid4().hex[:8]}"

    async def _start_cleanup_task(self):
        """Start background cleanup task if not running."""
        async with self._cleanup_lock:
            if self._cleanup_task is None or self._cleanup_task.done():
                logger.debug("Starting background cleanup task")
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self):
        """Background task to clean up expired locks."""
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                await self.cleanup_expired()
        except asyncio.CancelledError:
            pass

    async def cleanup_expired(self) -> int:
        """Clean up expired locks.

        Returns:
            Number of locks cleaned up
        """
        await self._ensure_initialized()

        now = time.time_ns()
        stale_threshold = now - int(self._lock_ttl * 1_000_000_000)

        count = await self._engine.delete(
            AgentLockModel,
            filters=ComparisonFilter.lt("last_heartbeat_ns", stale_threshold),
        )

        if count > 0:
            logger.warning(f"Cleaned up {count} expired lock(s)")

        return count

    async def _heartbeat_loop(
        self,
        session_id: str,
        agent_id: str,
        holder_id: str,
    ):
        """Send periodic heartbeats to keep lock alive."""
        logger.debug(f"Starting heartbeat loop for session={session_id}, agent={agent_id}, holder={holder_id}")
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)

                # Find lock
                lock = await self._engine.find_first(
                    AgentLockModel,
                    filters=AndFilter(
                        filters=[
                            ComparisonFilter.eq("session_id", session_id),
                            ComparisonFilter.eq("agent_id", agent_id),
                            ComparisonFilter.eq("holder_id", holder_id),
                        ]
                    ),
                )

                if not lock:
                    # Lock was released or taken by someone else
                    logger.warning(f"Heartbeat stopped: lock not found for session={session_id}, agent={agent_id}, holder={holder_id}")
                    break

                # Update heartbeat
                lock.last_heartbeat_ns = time.time_ns()
                await self._engine.update(lock)
                logger.debug(f"Heartbeat sent for session={session_id}, agent={agent_id}, holder={holder_id}")

        except asyncio.CancelledError:
            logger.debug(f"Heartbeat loop cancelled for session={session_id}, agent={agent_id}, holder={holder_id}")
            pass

    @asynccontextmanager
    async def acquire(
        self,
        session_id: str,
        agent_id: str,
        *,
        user_id: str | None = None,
        run_id: str | None = None,
    ) -> AsyncGenerator[None, None]:
        """Acquire agent lock with automatic heartbeat renewal.

        Args:
            session_id: Session identifier (primary key)
            agent_id: Agent identifier (primary key)
            user_id: User identifier (metadata, optional)
            run_id: Run identifier (metadata, optional)

        Yields:
            None when lock is acquired

        Raises:
            TimeoutError: If agent is already locked (immediate failure)
        """
        await self._ensure_initialized()
        await self._start_cleanup_task()

        holder_id = self._generate_holder_id()
        now = time.time_ns()

        logger.info(f"Attempting to acquire lock for session={session_id}, agent={agent_id}, holder={holder_id}, run_id={run_id}")

        # Try to create lock first (optimistic approach)
        lock = AgentLockModel(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            run_id=run_id,
            holder_id=holder_id,
            acquired_at_ns=now,
            last_heartbeat_ns=now,
        )

        # Try to create lock, handle race condition
        lock_acquired = False
        try:
            await self._engine.create(lock)
            lock_acquired = True
            logger.info(f"Lock acquired successfully for session={session_id}, agent={agent_id}, holder={holder_id}")
        except Exception as e:
            # Check if it's a unique constraint violation (race condition)
            # Different databases have different exception types
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["unique", "duplicate", "constraint", "integrity"]):
                logger.debug(f"Lock conflict detected for session={session_id}, agent={agent_id}, checking if expired")
                # Lock already exists, check if it's expired
                existing = await self._engine.find_first(
                    AgentLockModel,
                    filters=AndFilter(
                        filters=[
                            ComparisonFilter.eq("session_id", session_id),
                            ComparisonFilter.eq("agent_id", agent_id),
                        ]
                    ),
                )

                if existing and now - existing.last_heartbeat_ns > int(self._lock_ttl * 1_000_000_000):
                    # Lock expired, delete it and retry
                    logger.info(
                        f"Found expired lock for session={session_id}, agent={agent_id}, holder={existing.holder_id}, deleting and retrying"
                    )
                    await self._engine.delete(
                        AgentLockModel,
                        filters=AndFilter(
                            filters=[
                                ComparisonFilter.eq("session_id", session_id),
                                ComparisonFilter.eq("agent_id", agent_id),
                            ]
                        ),
                    )
                    # Retry creating the lock with a new instance
                    lock = AgentLockModel(
                        session_id=session_id,
                        agent_id=agent_id,
                        user_id=user_id,
                        run_id=run_id,
                        holder_id=holder_id,
                        acquired_at_ns=now,
                        last_heartbeat_ns=now,
                    )
                    try:
                        await self._engine.create(lock)
                        lock_acquired = True
                        logger.info(
                            f"Lock acquired after removing expired lock for session={session_id}, agent={agent_id}, holder={holder_id}"
                        )
                    except Exception:
                        # If still fails, another task got it first
                        logger.warning(
                            f"Failed to acquire lock after removing expired lock (race condition) "
                            f"for session={session_id}, agent={agent_id}"
                        )
                        raise TimeoutError(f"Agent {agent_id} in session {session_id} is already locked (race condition detected)") from e
                else:
                    # Lock is still valid or was just acquired by another task
                    holder = existing.holder_id if existing else "unknown"
                    logger.warning(
                        f"Lock acquisition failed: agent {agent_id} in session {session_id} is already locked by holder={holder}"
                    )
                    raise TimeoutError(f"Agent {agent_id} in session {session_id} is already locked (holder: {holder})") from e
            else:
                # Re-raise other exceptions
                logger.error(f"Unexpected error acquiring lock for session={session_id}, agent={agent_id}: {e}")
                raise

        if not lock_acquired:
            raise TimeoutError(f"Agent {agent_id} in session {session_id} could not acquire lock")

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(session_id, agent_id, holder_id))

        try:
            yield
        finally:
            # Stop heartbeat
            logger.debug(f"Releasing lock for session={session_id}, agent={agent_id}, holder={holder_id}")
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            # Release lock (only if we still hold it)
            deleted_count = await self._engine.delete(
                AgentLockModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("session_id", session_id),
                        ComparisonFilter.eq("agent_id", agent_id),
                        ComparisonFilter.eq("holder_id", holder_id),
                    ]
                ),
            )
            if deleted_count > 0:
                logger.info(f"Lock released successfully for session={session_id}, agent={agent_id}, holder={holder_id}")
            else:
                logger.warning(
                    f"Lock was already released or taken by another holder for session={session_id}, agent={agent_id}, holder={holder_id}"
                )

    async def is_locked(
        self,
        session_id: str,
        agent_id: str,
    ) -> bool:
        """Check if agent is currently locked (and not expired).

        Args:
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            True if lock exists and is not expired
        """
        await self._ensure_initialized()

        lock = await self._engine.find_first(
            AgentLockModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("session_id", session_id),
                    ComparisonFilter.eq("agent_id", agent_id),
                ]
            ),
        )

        if not lock:
            logger.debug(f"is_locked check: no lock found for session={session_id}, agent={agent_id}")
            return False

        # Check if expired
        now = time.time_ns()
        if now - lock.last_heartbeat_ns > int(self._lock_ttl * 1_000_000_000):
            logger.debug(f"is_locked check: lock expired for session={session_id}, agent={agent_id}, holder={lock.holder_id}")
            return False

        logger.debug(f"is_locked check: lock is active for session={session_id}, agent={agent_id}, holder={lock.holder_id}")
        return True

    async def stop(self):
        """Stop the cleanup task."""
        if self._cleanup_task and not self._cleanup_task.done():
            logger.info("Stopping AgentLockService cleanup task")
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("AgentLockService cleanup task stopped")
