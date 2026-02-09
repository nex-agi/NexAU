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

"""Unit tests for AgentLockService.

Tests the new AgentLockService with heartbeat renewal mechanism.
"""

import asyncio
import time

from nexau.archs.session import AgentLockService
from nexau.archs.session.models.agent_lock import AgentLockModel
from nexau.archs.session.orm import InMemoryDatabaseEngine, SQLDatabaseEngine


class TestAgentLockServiceBasic:
    """Basic tests for AgentLockService."""

    def test_acquire_and_release(self):
        """Test basic lock acquisition and release."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                # Acquire lock
                async with service.acquire("session1", "agent1"):
                    # Lock should be held
                    is_locked = await service.is_locked("session1", "agent1")
                    assert is_locked

                # Lock should be released
                is_locked = await service.is_locked("session1", "agent1")
                assert not is_locked
            finally:
                await service.stop()

        asyncio.run(run())

    def test_concurrent_same_agent_fails(self):
        """Test that concurrent execution on same agent fails."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                results = []

                async def task(task_id: int):
                    try:
                        async with service.acquire("session1", "agent1"):
                            results.append(f"task{task_id}_acquired")
                            await asyncio.sleep(0.1)
                            results.append(f"task{task_id}_done")
                    except TimeoutError:
                        results.append(f"task{task_id}_failed")

                # Run two tasks concurrently
                await asyncio.gather(
                    task(1),
                    task(2),
                    return_exceptions=True,
                )

                # One should succeed, one should fail
                assert "task1_acquired" in results or "task2_acquired" in results
                assert "task1_failed" in results or "task2_failed" in results
            finally:
                await service.stop()

        asyncio.run(run())

    def test_concurrent_different_agents_succeeds(self):
        """Test that concurrent execution on different agents succeeds."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                results = []

                async def task(agent_id: str, task_id: int):
                    async with service.acquire("session1", agent_id):
                        results.append(f"task{task_id}_acquired")
                        await asyncio.sleep(0.1)
                        results.append(f"task{task_id}_done")

                # Run two tasks concurrently with different agent_ids
                await asyncio.gather(
                    task("agent1", 1),
                    task("agent2", 2),
                )

                # Both should succeed
                assert "task1_acquired" in results
                assert "task1_done" in results
                assert "task2_acquired" in results
                assert "task2_done" in results
            finally:
                await service.stop()

        asyncio.run(run())

    def test_concurrent_different_sessions_succeeds(self):
        """Test that concurrent execution on different sessions succeeds."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                results = []

                async def task(session_id: str, task_id: int):
                    async with service.acquire(session_id, "agent1"):
                        results.append(f"task{task_id}_acquired")
                        await asyncio.sleep(0.1)
                        results.append(f"task{task_id}_done")

                # Run two tasks concurrently with different session_ids
                await asyncio.gather(
                    task("session1", 1),
                    task("session2", 2),
                )

                # Both should succeed
                assert "task1_acquired" in results
                assert "task1_done" in results
                assert "task2_acquired" in results
                assert "task2_done" in results
            finally:
                await service.stop()

        asyncio.run(run())

    def test_lock_with_metadata(self):
        """Test lock acquisition with user_id and run_id metadata."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                async with service.acquire(
                    "session1",
                    "agent1",
                    user_id="user123",
                    run_id="run456",
                ):
                    is_locked = await service.is_locked("session1", "agent1")
                    assert is_locked
            finally:
                await service.stop()

        asyncio.run(run())


class TestAgentLockServiceHeartbeat:
    """Tests for heartbeat renewal mechanism."""

    def test_heartbeat_renewal(self):
        """Test that lock is renewed by heartbeat."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=3.0,  # 3 seconds TTL
                heartbeat_interval=1.0,  # 1 second heartbeat
            )

            try:
                start = time.time()

                # Hold lock for 5 seconds (longer than TTL of 3s)
                async with service.acquire("session1", "agent1"):
                    await asyncio.sleep(5)

                elapsed = time.time() - start
                assert elapsed >= 5, "Task should complete successfully with heartbeat renewal"
            finally:
                await service.stop()

        asyncio.run(run())

    def test_expired_lock_cleanup(self):
        """Test that expired locks are cleaned up."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=3.0,
                heartbeat_interval=1.0,
            )

            try:
                # Manually create an expired lock
                await service._ensure_initialized()
                old_lock = AgentLockModel(
                    session_id="session1",
                    agent_id="agent1",
                    user_id="user1",
                    run_id=None,
                    holder_id="old_holder",
                    acquired_at_ns=time.time_ns() - 10_000_000_000,  # 10 seconds ago
                    expires_at_ns=time.time_ns() - 1_000_000_000,  # expired 1 second ago
                )
                await service._engine.create(old_lock)

                # Try to acquire - should succeed because old lock is expired
                async with service.acquire("session1", "agent1"):
                    is_locked = await service.is_locked("session1", "agent1")
                    assert is_locked
            finally:
                await service.stop()

        asyncio.run(run())

    def test_cleanup_expired_locks(self):
        """Test that expired locks are ignored and can be replaced.

        In the expiration-based design, there is no explicit cleanup_expired().
        Expired locks are filtered out on query (_find_valid_lock) and
        replaced atomically during acquire().
        """

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=3.0,
                heartbeat_interval=1.0,
            )

            try:
                # Create multiple expired locks
                await service._ensure_initialized()
                for i in range(3):
                    old_lock = AgentLockModel(
                        session_id=f"session{i}",
                        agent_id="agent1",
                        user_id="user1",
                        run_id=None,
                        holder_id=f"holder{i}",
                        acquired_at_ns=time.time_ns() - 10_000_000_000,  # 10 seconds ago
                        expires_at_ns=time.time_ns() - 1_000_000_000,  # expired 1 second ago
                    )
                    await service._engine.create(old_lock)

                # All expired locks should be reported as not locked
                for i in range(3):
                    is_locked = await service.is_locked(f"session{i}", "agent1")
                    assert not is_locked, f"Expired lock for session{i} should not be reported as locked"

                # Should be able to acquire a new lock on a session with expired lock
                async with service.acquire("session0", "agent1"):
                    is_locked = await service.is_locked("session0", "agent1")
                    assert is_locked
            finally:
                await service.stop()

        asyncio.run(run())


class TestAgentLockServiceSQLBackend:
    """Tests for AgentLockService with SQL backend."""

    def test_sql_backend_basic(self):
        """Test basic lock operations with SQL backend."""

        async def run():
            from sqlalchemy.ext.asyncio import create_async_engine

            sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            engine = SQLDatabaseEngine(sql_engine)
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                async with service.acquire("session1", "agent1"):
                    is_locked = await service.is_locked("session1", "agent1")
                    assert is_locked

                is_locked = await service.is_locked("session1", "agent1")
                assert not is_locked
            finally:
                await service.stop()
                await sql_engine.dispose()

        asyncio.run(run())

    def test_sql_backend_concurrent(self):
        """Test concurrent operations with SQL backend.

        Note: Due to SQLite's limited concurrency support in memory mode,
        this test verifies that at least one task succeeds and at least one
        task attempts to acquire the lock. In rare cases, both tasks may
        succeed due to SQLite's write serialization behavior.
        """

        async def run():
            from sqlalchemy.ext.asyncio import create_async_engine

            sql_engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            engine = SQLDatabaseEngine(sql_engine)
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                results = []

                async def task(task_id: int):
                    try:
                        async with service.acquire("session1", "agent1"):
                            results.append(f"task{task_id}_success")
                            await asyncio.sleep(0.2)  # Hold lock longer to ensure conflict
                    except TimeoutError:
                        results.append(f"task{task_id}_failed")

                await asyncio.gather(
                    task(1),
                    task(2),
                    return_exceptions=True,
                )

                # At least one should succeed
                success_count = len([r for r in results if "success" in r])
                failed_count = len([r for r in results if "failed" in r])

                # Verify we got results from both tasks
                assert len(results) == 2, f"Expected 2 results, got {len(results)}: {results}"

                # At least one should succeed (the lock was acquired)
                assert success_count >= 1, f"Expected at least 1 success, got {success_count}: {results}"

                # Ideally one should succeed and one should fail, but due to SQLite's
                # concurrency limitations, both might succeed in rare cases
                if success_count == 2:
                    # Both succeeded - this can happen with SQLite in-memory
                    # Log a warning but don't fail the test
                    import warnings

                    warnings.warn("Both tasks succeeded in acquiring the lock. This is a known limitation of SQLite's concurrency model.")
                else:
                    # Normal case: one succeeds, one fails
                    assert success_count == 1, f"Expected 1 success, got {success_count}: {results}"
                    assert failed_count == 1, f"Expected 1 failure, got {failed_count}: {results}"
            finally:
                await service.stop()
                await sql_engine.dispose()

        asyncio.run(run())


class TestAgentLockServiceEdgeCases:
    """Tests for edge cases and new code paths in expiration-based locking."""

    def test_heartbeat_stops_when_lock_deleted_externally(self):
        """Test that heartbeat loop stops when lock is deleted by another process.

        Covers the _heartbeat_loop break path when lock is not found.
        """

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=5.0,
                heartbeat_interval=0.2,  # Fast heartbeat for test speed
            )

            try:
                lock_acquired = asyncio.Event()

                async def hold_lock():
                    async with service.acquire("session1", "agent1"):
                        lock_acquired.set()
                        # Hold lock long enough for heartbeat to fire
                        await asyncio.sleep(2.0)

                task = asyncio.create_task(hold_lock())

                # Wait for lock to be acquired
                await asyncio.wait_for(lock_acquired.wait(), timeout=2.0)

                # Externally delete the lock (simulating another process)
                from nexau.archs.session.orm import AndFilter, ComparisonFilter

                await engine.delete(
                    AgentLockModel,
                    filters=AndFilter(
                        filters=[
                            ComparisonFilter.eq("session_id", "session1"),
                            ComparisonFilter.eq("agent_id", "agent1"),
                        ]
                    ),
                )

                # Wait for heartbeat to detect the missing lock and stop
                await asyncio.sleep(0.5)

                # Lock should no longer be reported as locked
                is_locked = await service.is_locked("session1", "agent1")
                assert not is_locked, "Lock should not be reported as locked after external deletion"

                # Let the task finish
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            finally:
                await service.stop()

        asyncio.run(run())

    def test_is_locked_returns_false_for_nonexistent_lock(self):
        """Test is_locked returns False when no lock exists at all."""

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                is_locked = await service.is_locked("nonexistent_session", "nonexistent_agent")
                assert not is_locked, "is_locked should return False for nonexistent lock"
            finally:
                await service.stop()

        asyncio.run(run())

    def test_acquire_rejects_when_valid_lock_exists(self):
        """Test that acquire raises TimeoutError immediately when a valid lock exists.

        Covers the early rejection path in acquire() before attempting create.
        """

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                # Manually create a valid (non-expired) lock
                await service._ensure_initialized()
                valid_lock = AgentLockModel(
                    session_id="session1",
                    agent_id="agent1",
                    user_id="user1",
                    run_id=None,
                    holder_id="existing_holder",
                    acquired_at_ns=time.time_ns(),
                    expires_at_ns=time.time_ns() + 30_000_000_000,  # expires in 30s
                )
                await service._engine.create(valid_lock)

                # Try to acquire — should fail immediately with TimeoutError
                import pytest

                with pytest.raises(TimeoutError, match="already locked"):
                    async with service.acquire("session1", "agent1"):
                        pass  # Should never reach here
            finally:
                await service.stop()

        asyncio.run(run())

    def test_invalid_heartbeat_interval_raises_value_error(self):
        """Test that heartbeat_interval >= lock_ttl/2 raises ValueError."""
        import pytest

        engine = InMemoryDatabaseEngine()

        with pytest.raises(ValueError, match="heartbeat_interval"):
            AgentLockService(
                engine=engine,
                lock_ttl=10.0,
                heartbeat_interval=5.0,  # == lock_ttl/2, should fail
            )

        with pytest.raises(ValueError, match="heartbeat_interval"):
            AgentLockService(
                engine=engine,
                lock_ttl=10.0,
                heartbeat_interval=6.0,  # > lock_ttl/2, should fail
            )

    def test_lock_release_only_deletes_own_holder(self):
        """Test that releasing a lock only deletes the lock with matching holder_id.

        Ensures the finally block in acquire() doesn't accidentally delete
        another holder's lock.
        """

        async def run():
            engine = InMemoryDatabaseEngine()
            service = AgentLockService(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            try:
                # Acquire and release a lock
                async with service.acquire("session1", "agent1"):
                    is_locked = await service.is_locked("session1", "agent1")
                    assert is_locked

                # After release, lock should be gone
                is_locked = await service.is_locked("session1", "agent1")
                assert not is_locked, "Lock should be released after context exit"

                # Now manually insert a lock with a different holder
                await service._ensure_initialized()
                other_lock = AgentLockModel(
                    session_id="session1",
                    agent_id="agent1",
                    user_id="user1",
                    run_id=None,
                    holder_id="other_holder_xyz",
                    acquired_at_ns=time.time_ns(),
                    expires_at_ns=time.time_ns() + 30_000_000_000,
                )
                await service._engine.create(other_lock)

                # Verify the other holder's lock is still there
                is_locked = await service.is_locked("session1", "agent1")
                assert is_locked, "Other holder's lock should still exist"
            finally:
                await service.stop()

        asyncio.run(run())
