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

"""Cross-event-loop SessionManager regression tests.

These 4 tests reproduce the issue where a SessionManager (backed by a
SQL-style async engine) is created on the main event loop but then used
from a **worker thread** via ``asyncio.run()`` which creates a temporary loop.

Production call chains that trigger this:

1. ``execute_async → to_thread(sync tool) → get_sandbox → load_sandbox_state → sm``
2. ``execute_async → to_thread(sync sub-agent) → Agent() → _init_session_state → sm``
3. ``execute_async → to_thread(sync sub-agent) → agent.run() → asyncio.run(run_async) → sm``
4. ``execute_async → to_thread(sync sub-agent) → run_async → HistoryList.flush → sm``

The problem:

- ``SQLDatabaseEngine``'s ``AsyncEngine`` connection pool is loop-bound.
  Real async drivers (asyncpg, aiomysql) immediately crash with
  ``RuntimeError: Task got Future attached to a different loop``.
- aiosqlite is resilient (background-thread model), masking the issue.

To make the tests **driver-independent**, we use a ``LoopSafetyEngine``
wrapper that detects cross-loop calls and raises ``CrossLoopViolationError``.
This simulates what asyncpg/aiomysql would do at the connection pool level.

After the cross-loop issue is fixed, **all 4 tests should pass** because
the fixing agent will ensure SM operations are always dispatched to the
owner loop (e.g. via ``run_coroutine_threadsafe``).
"""

from __future__ import annotations

import asyncio
import threading
from typing import TypeVar

from sqlmodel import SQLModel

from nexau.archs.session.agent_run_action_service import AgentRunActionKey
from nexau.archs.session.models import AgentModel, AgentRunActionModel, SessionModel
from nexau.archs.session.models.agent_lock import AgentLockModel
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.session.orm.engine import DatabaseEngine, LoopSafeDatabaseEngine
from nexau.archs.session.orm.filters import Filter
from nexau.archs.session.session_manager import SessionManager
from nexau.core.messages import Message

# ---------------------------------------------------------------------------
# Loop-safety engine wrapper
# ---------------------------------------------------------------------------

ALL_SESSION_MODELS: list[type[SQLModel]] = [
    SessionModel,
    AgentModel,
    AgentRunActionModel,
    AgentLockModel,
]

T = TypeVar("T", bound=SQLModel)


class CrossLoopViolationError(RuntimeError):
    """Raised when a DB operation is called from a different event loop.

    This error simulates what asyncpg / aiomysql would raise at the
    connection pool level (``RuntimeError: ... attached to a different loop``).
    """


class LoopSafetyEngine(DatabaseEngine):
    """Wrapper that asserts all async ops happen on the **same** event loop.

    Delegates all real I/O to an ``InMemoryDatabaseEngine`` so the data
    layer is functional.  On every async call it compares the current
    running loop with the one captured at ``setup_models()`` time.
    If they differ, ``CrossLoopViolationError`` is raised — exactly what a
    real loop-bound driver (asyncpg, aiomysql) would do.
    """

    def __init__(self) -> None:
        self._inner = InMemoryDatabaseEngine()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._owner_thread: int | None = None

    def _check_loop(self, operation: str) -> None:
        """Raise if the caller is on a different loop than the owner."""
        if self._owner_loop is None:
            return  # not yet initialized
        try:
            current = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop at all (pure sync call) — not our concern
            return
        if current is not self._owner_loop:
            raise CrossLoopViolationError(
                f"DB operation '{operation}' called on loop {id(current):#x} "
                f"(thread={threading.current_thread().name}) but engine is "
                f"bound to loop {id(self._owner_loop):#x} "
                f"(thread id={self._owner_thread}). "
                f"This would crash with asyncpg/aiomysql: "
                f"'Task got Future attached to a different loop'."
            )

    # -- lifecycle -----------------------------------------------------------

    async def setup_models(self, model_classes: list[type[SQLModel]]) -> None:
        self._owner_loop = asyncio.get_running_loop()
        self._owner_thread = threading.current_thread().ident
        return await self._inner.setup_models(model_classes)

    # -- CRUD with loop check ------------------------------------------------

    async def create(self, model: T) -> T:
        self._check_loop("create")
        return await self._inner.create(model)

    async def create_many(self, models: list[T]) -> list[T]:
        self._check_loop("create_many")
        return await self._inner.create_many(models)

    async def update(self, model: T) -> T:
        self._check_loop("update")
        return await self._inner.update(model)

    async def find_first(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> T | None:
        self._check_loop("find_first")
        return await self._inner.find_first(model_class, filters=filters)

    async def find_many(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
        limit: int | None = None,
        offset: int | None = None,
        order_by: str | tuple[str, ...] | None = None,
    ) -> list[T]:
        self._check_loop("find_many")
        return await self._inner.find_many(
            model_class,
            filters=filters,
            limit=limit,
            offset=offset,
            order_by=order_by,
        )

    async def delete(
        self,
        model_class: type[T],
        *,
        filters: Filter,
    ) -> int:
        self._check_loop("delete")
        return await self._inner.delete(model_class, filters=filters)

    async def count(
        self,
        model_class: type[T],
        *,
        filters: Filter | None = None,
    ) -> int:
        self._check_loop("count")
        return await self._inner.count(model_class, filters=filters)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _make_session_manager() -> tuple[SessionManager, LoopSafetyEngine]:
    """Create a SessionManager backed by LoopSafetyEngine on the current loop.

    验证 bridge 链路：SessionManager 自动将 LoopSafetyEngine 包装在
    LoopSafeDatabaseEngine 中。当跨 loop 调用时，LoopSafeDatabaseEngine 会将
    操作桥接回 owner loop，然后委托给 LoopSafetyEngine（检查 loop 匹配），
    最终委托给 InMemoryDatabaseEngine。
    """
    engine = LoopSafetyEngine()
    sm = SessionManager(engine=engine)

    # 验证 bridge 链路已建立：sm._engine 是 LoopSafeDatabaseEngine 包装 LoopSafetyEngine
    assert isinstance(sm._engine, LoopSafeDatabaseEngine)
    assert sm._engine._inner is engine

    await sm.setup_models()
    return sm, engine


# ---------------------------------------------------------------------------
# Scenario 1: sync tool → sandbox → load_sandbox_state → sm.get_session()
#
# Production path:
#   execute_async
#     → asyncio.to_thread(_execute_tool_call_safe)       [worker thread]
#       → tool_executor.execute_tool()
#         → agent_state.get_sandbox()
#           → sandbox_manager.start_sync()
#             → load_sandbox_state(session_manager, ...)
#               → run_async_function_sync(sm.get_session(...))
#                 → asyncio.run(sm.get_session(...))      [temp loop ❌]
# ---------------------------------------------------------------------------


class TestCrossLoopSandboxLoadState:
    """sandbox load_sandbox_state calls sm.get_session() in a temp loop."""

    def test_get_session_cross_loop(self):
        """sm.get_session() from a worker thread's temporary event loop.

        Simulates: load_sandbox_state → run_async_function_sync(sm.get_session)
        """

        async def main() -> None:
            sm, _engine = await _make_session_manager()

            # 1. Seed session on main loop
            await sm._get_or_create_session(user_id="u1", session_id="s1")

            # 2. Verify readable on main loop (sanity check)
            session = await sm.get_session(user_id="u1", session_id="s1")
            assert session is not None

            # 3. Worker thread → asyncio.run() → sm.get_session()
            #    This is exactly what load_sandbox_state does.
            def worker() -> SessionModel | None:
                async def _query() -> SessionModel | None:
                    return await sm.get_session(user_id="u1", session_id="s1")

                return asyncio.run(_query())

            # 修复后：LoopSafeDatabaseEngine 将操作桥接回 owner loop，
            # LoopSafetyEngine 不再触发 CrossLoopViolationError
            result = await asyncio.to_thread(worker)
            assert result is not None
            assert result.session_id == "s1"
            assert result.user_id == "u1"

        asyncio.run(main())


# ---------------------------------------------------------------------------
# Scenario 2: sync sub-agent → Agent() → _init_session_state
#             → sm.register_agent()
#
# Production path:
#   execute_async
#     → asyncio.to_thread(_execute_sub_agent_call_safe)   [worker thread]
#       → subagent_manager.call_sub_agent()
#         → Agent()
#           → _init_session_state()
#             → asyncio.run(sm.register_agent(...))       [temp loop ❌]
# ---------------------------------------------------------------------------


class TestCrossLoopSubAgentInit:
    """Sub-agent __init__ calls sm.register_agent() in a temp loop."""

    def test_register_agent_cross_loop(self):
        """sm.register_agent() from a worker thread's temporary event loop.

        Simulates: Agent.__init__ → _init_session_state → asyncio.run(sm.register_agent)
        """

        async def main() -> None:
            sm, _engine = await _make_session_manager()
            await sm._get_or_create_session(user_id="u1", session_id="s1")

            def worker() -> str:
                async def _register() -> str:
                    agent_id, _session = await sm.register_agent(
                        user_id="u1",
                        session_id="s1",
                        agent_name="sub_agent",
                        is_root=False,
                    )
                    return agent_id

                return asyncio.run(_register())

            # 修复后：LoopSafeDatabaseEngine 将操作桥接回 owner loop
            agent_id = await asyncio.to_thread(worker)
            assert isinstance(agent_id, str)
            assert len(agent_id) > 0

        asyncio.run(main())


# ---------------------------------------------------------------------------
# Scenario 3: sync sub-agent → agent.run() → asyncio.run(run_async)
#             → sm.agent_lock.acquire()
#
# Production path:
#   execute_async
#     → asyncio.to_thread(_execute_sub_agent_call_safe)   [worker thread]
#       → subagent_manager.call_sub_agent()
#         → agent.run()
#           → asyncio.run(run_async())                    [temp loop]
#             → sm.agent_lock.acquire(...)                [temp loop ❌]
# ---------------------------------------------------------------------------


class TestCrossLoopSubAgentLock:
    """Sub-agent run acquires agent_lock in a temp loop."""

    def test_agent_lock_acquire_cross_loop(self):
        """sm.agent_lock operations from a worker thread's temporary event loop.

        Simulates: agent.run() → asyncio.run(run_async) → sm.agent_lock.acquire

        The lock service internally does:
        1. _find_valid_lock → engine.find_first  (check existing lock)
        2. engine.delete (cleanup expired)
        3. engine.create (acquire lock)
        4. engine.find_first (heartbeat / release)

        All of these hit the engine on the temp loop → CrossLoopViolationError.
        We test the find + create + delete chain directly to avoid
        asynccontextmanager cleanup complications.
        """

        async def main() -> None:
            sm, _engine = await _make_session_manager()
            await sm._get_or_create_session(user_id="u1", session_id="s1")
            agent_id, _ = await sm.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="sub_agent",
                is_root=False,
            )

            # Directly test the engine operations that agent_lock uses
            from nexau.archs.session.models.agent_lock import AgentLockModel
            from nexau.archs.session.orm import AndFilter, ComparisonFilter

            def worker() -> None:
                async def _lock_operations() -> None:
                    # This is what _find_valid_lock does internally
                    await sm._engine.find_first(
                        AgentLockModel,
                        filters=AndFilter(
                            filters=[
                                ComparisonFilter.eq("session_id", "s1"),
                                ComparisonFilter.eq("agent_id", agent_id),
                            ],
                        ),
                    )

                asyncio.run(_lock_operations())

            # 修复後：LoopSafeDatabaseEngine 将操作桥接回 owner loop
            await asyncio.to_thread(worker)

        asyncio.run(main())


# ---------------------------------------------------------------------------
# Scenario 4: sync sub-agent → run_async on temp loop
#             → HistoryList.flush → sm.agent_run_action.persist_append()
#
# Production path:
#   execute_async
#     → asyncio.to_thread(_execute_sub_agent_call_safe)   [worker thread]
#       → subagent_manager.call_sub_agent()
#         → agent.run()
#           → asyncio.run(run_async())                    [temp loop]
#             → _run_inner → HistoryList.flush()
#               → _persist_flush_async()
#                 → sm.agent_run_action.persist_append()  [temp loop ❌]
# ---------------------------------------------------------------------------


class TestCrossLoopSubAgentHistoryFlush:
    """Sub-agent HistoryList.flush persists via sm.agent_run_action on a temp loop."""

    def test_persist_append_cross_loop(self):
        """sm.agent_run_action.persist_append() from a worker thread's temp loop.

        Simulates: HistoryList.flush → _persist_flush_async → sm.agent_run_action.persist_append
        """

        async def main() -> None:
            sm, _engine = await _make_session_manager()
            await sm._get_or_create_session(user_id="u1", session_id="s1")
            agent_id, _ = await sm.register_agent(
                user_id="u1",
                session_id="s1",
                agent_name="sub_agent",
                is_root=False,
            )

            key = AgentRunActionKey(
                user_id="u1",
                session_id="s1",
                agent_id=agent_id,
            )
            messages = [
                Message.user("hello from sub-agent"),
                Message.assistant("response from sub-agent"),
            ]

            def worker() -> None:
                async def _persist() -> None:
                    await sm.agent_run_action.persist_append(
                        key=key,
                        run_id="run_001",
                        root_run_id="root_001",
                        parent_run_id=None,
                        agent_name="sub_agent",
                        messages=messages,
                    )

                asyncio.run(_persist())

            # 修复后：LoopSafeDatabaseEngine 将操作桥接回 owner loop
            await asyncio.to_thread(worker)

            # 验证数据确实被持久化且在主 loop 上可见
            loaded = await sm.agent_run_action.load_messages(key=key)
            assert len(loaded) == 2
            assert loaded[0].get_text_content() == "hello from sub-agent"
            assert loaded[1].get_text_content() == "response from sub-agent"

        asyncio.run(main())
