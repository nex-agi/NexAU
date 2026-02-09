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
It supports automatic heartbeat renewal using expiration-based locking.

Design principles:
- Expiration-based: Lock has expires_at_ns, valid while expires_at_ns > now
- Run-level heartbeat: Heartbeat task runs only during lock acquisition
- No global cleanup: Expired locks are filtered out on query, not deleted periodically
- Fast release: Lock is deleted when run completes
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
    - Expiration-based locking: lock valid while expires_at_ns > now
    - Automatic heartbeat extends expiration during run
    - No background cleanup task - expired locks ignored on query
    - No waiting: fails immediately if agent is busy

    Example:
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>> engine = InMemoryDatabaseEngine()
        >>> lock_service = AgentLockService(engine=engine)
        >>> async with lock_service.acquire("session1", "agent1"):
        ...     # Only one execution per (session_id, agent_id) at a time
        ...     # Lock expiration is automatically extended every heartbeat_interval
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

    def _calculate_expires_at_ns(self) -> int:
        """Calculate expiration timestamp (now + TTL)."""
        return time.time_ns() + int(self._lock_ttl * 1_000_000_000)

    async def _heartbeat_loop(
        self,
        session_id: str,
        agent_id: str,
        holder_id: str,
    ):
        """Send periodic heartbeats to extend lock expiration.

        This runs only during the lock acquisition context, not as a global task.
        Each heartbeat extends expires_at_ns by lock_ttl.
        """
        logger.debug(f"Starting heartbeat loop for session={session_id}, agent={agent_id}, holder={holder_id}")
        try:
            while True:
                await asyncio.sleep(self._heartbeat_interval)

                # Find and update lock in one operation
                # We need to verify we still hold the lock before extending
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

                # Extend expiration
                lock.expires_at_ns = self._calculate_expires_at_ns()
                await self._engine.update(lock)
                logger.debug(f"Heartbeat: extended lock expiration for session={session_id}, agent={agent_id}, holder={holder_id}")

        except asyncio.CancelledError:
            logger.debug(f"Heartbeat loop cancelled for session={session_id}, agent={agent_id}, holder={holder_id}")

    async def _find_valid_lock(
        self,
        session_id: str,
        agent_id: str,
    ) -> AgentLockModel | None:
        """Find a valid (non-expired) lock for the given session and agent.

        Returns:
            Lock if exists and not expired, None otherwise
        """
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
            return None

        # Check if expired
        now = time.time_ns()
        if lock.expires_at_ns <= now:
            logger.debug(f"Found expired lock for session={session_id}, agent={agent_id}, holder={lock.holder_id}")
            return None

        return lock

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

        holder_id = self._generate_holder_id()
        now = time.time_ns()
        expires_at_ns = self._calculate_expires_at_ns()

        logger.info(f"Attempting to acquire lock for session={session_id}, agent={agent_id}, holder={holder_id}, run_id={run_id}")

        # Check for existing valid lock first
        existing = await self._find_valid_lock(session_id, agent_id)
        if existing:
            logger.warning(
                f"Lock acquisition failed: agent {agent_id} in session {session_id} is already locked by holder={existing.holder_id}"
            )
            raise TimeoutError(f"Agent {agent_id} in session {session_id} is already locked (holder: {existing.holder_id})")

        # Try to create or replace lock
        # If there's an expired lock, we can overwrite it
        lock = AgentLockModel(
            session_id=session_id,
            agent_id=agent_id,
            user_id=user_id,
            run_id=run_id,
            holder_id=holder_id,
            acquired_at_ns=now,
            expires_at_ns=expires_at_ns,
        )

        # Try to create lock, handle race condition
        lock_acquired = False
        try:
            # First try to delete any expired lock (atomic cleanup)
            await self._engine.delete(
                AgentLockModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("session_id", session_id),
                        ComparisonFilter.eq("agent_id", agent_id),
                        ComparisonFilter.lt("expires_at_ns", now),  # Only delete if expired
                    ]
                ),
            )

            # Now try to create
            await self._engine.create(lock)
            lock_acquired = True
            logger.info(f"Lock acquired successfully for session={session_id}, agent={agent_id}, holder={holder_id}")
        except Exception as e:
            # Check if it's a unique constraint violation (race condition)
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["unique", "duplicate", "constraint", "integrity"]):
                # Another process got the lock first, check if it's still valid
                existing = await self._find_valid_lock(session_id, agent_id)
                if existing:
                    logger.warning(
                        f"Lock acquisition failed (race condition): agent {agent_id} in session {session_id} "
                        f"is already locked by holder={existing.holder_id}"
                    )
                    raise TimeoutError(f"Agent {agent_id} in session {session_id} is already locked (holder: {existing.holder_id})") from e
                else:
                    # The conflicting lock expired, try again
                    logger.debug(f"Retrying lock acquisition after expired lock cleanup for session={session_id}, agent={agent_id}")
                    await self._engine.delete(
                        AgentLockModel,
                        filters=AndFilter(
                            filters=[
                                ComparisonFilter.eq("session_id", session_id),
                                ComparisonFilter.eq("agent_id", agent_id),
                            ]
                        ),
                    )
                    try:
                        await self._engine.create(lock)
                        lock_acquired = True
                        logger.info(
                            f"Lock acquired after removing expired lock for session={session_id}, agent={agent_id}, holder={holder_id}"
                        )
                    except Exception:
                        logger.warning(f"Failed to acquire lock after retry for session={session_id}, agent={agent_id}")
                        raise TimeoutError(f"Agent {agent_id} in session {session_id} could not acquire lock (race condition)") from e
            else:
                # Re-raise other exceptions
                logger.error(f"Unexpected error acquiring lock for session={session_id}, agent={agent_id}: {e}")
                raise

        if not lock_acquired:
            raise TimeoutError(f"Agent {agent_id} in session {session_id} could not acquire lock")

        # Start heartbeat task (run-level, not global)
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

        lock = await self._find_valid_lock(session_id, agent_id)

        if lock:
            logger.debug(f"is_locked check: lock is active for session={session_id}, agent={agent_id}, holder={lock.holder_id}")
            return True
        else:
            logger.debug(f"is_locked check: no valid lock for session={session_id}, agent={agent_id}")
            return False

    async def stop(self):
        """Stop the lock service.

        Note: This is a no-op now since we no longer have a global cleanup task.
        Kept for API compatibility.
        """
        logger.debug("AgentLockService stop called (no-op, no global cleanup task)")
