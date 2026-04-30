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

"""Reproduce & verify: team lock deadlock when /team/stream is called twice.

RFC-0014: Team Stream 重复调用锁死修复

Bug scenario (reproduced in TestTeamLockConflictReproduction):
1. POST /team/stream starts a team run → leader acquires lock, enters forever loop
2. User sends another POST /team/stream (e.g. "继续") to the same session
3. registry.get_or_create() returns the SAME AgentTeam instance
4. team.run() creates a NEW leader → run_async() → tries to acquire the same lock
5. Lock is still held by first leader (heartbeat keeps renewing) → TimeoutError

Fix verified in TestTeamLockConflictFix:
- team.run() raises RuntimeError when _is_running is True
- /team/stream auto-redirects to enqueue_user_message + subscribe when team is running
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nexau.archs.main_sub.team.agent_team import AgentTeam
from nexau.archs.session import AgentLockService, SessionManager
from nexau.archs.session.models.team import TeamModel
from nexau.archs.session.models.team_member import TeamMemberModel
from nexau.archs.session.models.team_message import TeamMessageModel
from nexau.archs.session.models.team_task import TeamTaskModel
from nexau.archs.session.models.team_task_lock import TeamTaskLockModel
from nexau.archs.session.orm import InMemoryDatabaseEngine

ALL_TEAM_MODELS = [
    TeamModel,
    TeamMemberModel,
    TeamTaskModel,
    TeamTaskLockModel,
    TeamMessageModel,
]


def _make_agent_config(name: str = "leader") -> MagicMock:
    cfg = MagicMock()
    cfg.name = name
    cfg.description = f"A {name} agent"
    cfg.system_prompt = "You are a leader."
    cfg.system_prompt_suffix = None
    cfg.tools = []
    cfg.middlewares = None
    cfg.llm_config = None
    cfg.sandbox_config = None
    cfg.stop_tools = None
    cfg.tracers = []
    cfg.resolved_tracer = None
    cfg.after_model_hooks = None
    cfg.after_tool_hooks = None
    cfg.before_model_hooks = None
    cfg.before_tool_hooks = None
    return cfg


class TestTeamLockConflictReproduction:
    """Reproduce the lock deadlock when team.run() is called concurrently."""

    def test_heartbeat_prevents_lock_expiry(self):
        """Prove that heartbeat keeps the lock alive beyond its TTL.

        层级 1: Lock 机制层
        - lock_ttl=2s, heartbeat_interval=0.5s
        - 持有锁 4 秒（超过 TTL）
        - 第二个 acquire 在 4 秒后仍然失败 → 证明 heartbeat 阻止了过期
        """

        async def _run():
            engine = InMemoryDatabaseEngine()
            lock_service = AgentLockService(
                engine=engine,
                lock_ttl=2.0,
                heartbeat_interval=0.5,
            )

            first_locked = asyncio.Event()

            async def hold_lock():
                async with lock_service.acquire("s1:leader", "leader"):
                    first_locked.set()
                    # 持有锁 4 秒，远超 2s TTL
                    await asyncio.sleep(4.0)

            task = asyncio.create_task(hold_lock())
            await first_locked.wait()

            # 等 3 秒（超过 TTL 的 2s），heartbeat 应该已续期多次
            await asyncio.sleep(3.0)

            # 锁应该仍然有效
            is_locked = await lock_service.is_locked("s1:leader", "leader")
            assert is_locked, "Heartbeat should keep lock alive beyond TTL"

            # 第二个 acquire 应该失败
            with pytest.raises(TimeoutError, match="already locked"):
                async with lock_service.acquire("s1:leader", "leader"):
                    pass

            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

    @pytest.mark.skipif(sys.platform == "win32", reason="reproduction test can block the Windows proactor event loop")
    def test_concurrent_team_run_causes_lock_conflict(self):
        """Reproduce the actual bug: second team.run() fails.

        层级 2: AgentTeam 层
        - 第一次 team.run() 设置 _is_running=True 并阻塞
        - 第二次 team.run() 被 _is_running 守卫拦截（RFC-0014 修复后）
        - 修复前: 会到达 lock 层面得到 TimeoutError
        - 修复后: 在 run() 入口得到 RuntimeError（更快失败，不触碰锁）
        """

        async def _run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            session_manager = SessionManager(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )
            await session_manager.setup_models()

            leader_config = _make_agent_config("leader")
            candidate_config = _make_agent_config("worker")

            team = AgentTeam(
                leader_config=leader_config,
                candidates={"worker": candidate_config},
                engine=engine,
                session_manager=session_manager,
                user_id="devbox-user",
                session_id="devbox-session-xxx",
            )
            team._shared_sandbox_manager = MagicMock()

            first_run_started = asyncio.Event()

            async def fake_run_async_blocking(**kwargs):
                """模拟第一次 run: leader 获取锁后进入 team_mode 永久循环"""
                leader_session_id = "devbox-session-xxx:leader"
                async with session_manager.agent_lock.acquire(leader_session_id, "leader"):
                    first_run_started.set()
                    await asyncio.sleep(999)
                return "done"

            mock_agent = MagicMock()
            mock_agent.run_async = AsyncMock(side_effect=fake_run_async_blocking)
            mock_agent.executor.force_stop = MagicMock()

            # team.run() 调用 await Agent.create(...)，需要 create 是 AsyncMock
            mock_agent_cls = MagicMock()
            mock_agent_cls.create = AsyncMock(return_value=mock_agent)

            with patch(
                "nexau.archs.main_sub.team.agent_team._safe_deepcopy_config",
                side_effect=lambda c: c,
            ):
                with patch(
                    "nexau.archs.main_sub.team.agent_team.Agent",
                    mock_agent_cls,
                ):
                    # 1. 启动第一次 team.run()（在后台运行）
                    first_task = asyncio.create_task(team.run("开始工作"))
                    await first_run_started.wait()

                    # 2. RFC-0014: _is_running 守卫阻止第二次调用
                    with pytest.raises(RuntimeError, match="already running"):
                        await team.run("继续")

                    # 清理
                    first_task.cancel()
                    try:
                        await first_task
                    except (asyncio.CancelledError, Exception):
                        pass

        asyncio.run(_run())

    def test_team_stream_endpoint_no_guard(self):
        """Prove that /team/stream has no is_running guard.

        层级 3: HTTP 路由层
        - registry.get_or_create() 对已有 team 返回同一实例
        - team_stream() 不检查 team.is_running 就调用 run_streaming()
        - 这导致了层级 2 中的锁冲突
        """

        async def _run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)
            session_manager = SessionManager(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            from nexau.archs.transports.http.team_registry import TeamRegistry

            registry = TeamRegistry(engine=engine, session_manager=session_manager)

            leader_config = _make_agent_config("leader")
            candidate_config = _make_agent_config("worker")
            registry.register_config(
                "default",
                leader_config=leader_config,
                candidates={"worker": candidate_config},
            )

            # 第一次 get_or_create 创建新 team
            team1 = registry.get_or_create("u1", "s1")
            team1._is_running = True  # 模拟 team 正在运行

            # 第二次 get_or_create 返回同一实例
            team2 = registry.get_or_create("u1", "s1")
            assert team1 is team2, "Should return the same team instance"
            assert team2.is_running, "Team should still be running"

            # 关键问题: team_stream 端点没有检查 is_running
            # 它直接调用 team.run_streaming()，导致第二次 run() 调用
            # 这就是 bug 的入口

        asyncio.run(_run())


class TestTeamLockConflictFix:
    """Verify the fixes from RFC-0014."""

    def test_run_raises_when_already_running(self):
        """RFC-0014 T1: team.run() rejects concurrent calls with RuntimeError.

        验证 team.run() 在 _is_running=True 时立即抛出 RuntimeError，
        而非尝试获取锁后失败。
        """

        async def _run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            session_manager = MagicMock()
            leader_config = _make_agent_config("leader")
            candidate_config = _make_agent_config("worker")

            team = AgentTeam(
                leader_config=leader_config,
                candidates={"worker": candidate_config},
                engine=engine,
                session_manager=session_manager,
                user_id="u1",
                session_id="s1",
            )

            # 模拟 team 已在运行
            team._is_running = True

            with pytest.raises(RuntimeError, match="already running"):
                await team.run("继续")

        asyncio.run(_run())

    def test_run_works_when_not_running(self):
        """RFC-0014 T1: team.run() works normally when _is_running=False.

        确保并发保护不影响正常的首次调用。
        """

        async def _run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            session_manager = MagicMock()
            leader_config = _make_agent_config("leader")
            candidate_config = _make_agent_config("worker")

            team = AgentTeam(
                leader_config=leader_config,
                candidates={"worker": candidate_config},
                engine=engine,
                session_manager=session_manager,
                user_id="u1",
                session_id="s1",
            )
            team._shared_sandbox_manager = MagicMock()

            mock_leader = MagicMock()
            mock_leader.run_async = AsyncMock(return_value="done")
            mock_leader.executor.force_stop = MagicMock()

            # team.run() 调用 await Agent.create(...)，需要 create 是 AsyncMock
            mock_agent_cls = MagicMock()
            mock_agent_cls.create = AsyncMock(return_value=mock_leader)

            with patch(
                "nexau.archs.main_sub.team.agent_team._safe_deepcopy_config",
                side_effect=lambda c: c,
            ):
                with patch(
                    "nexau.archs.main_sub.team.agent_team.Agent",
                    mock_agent_cls,
                ):
                    result = await team.run("hello")

            assert result == "done"

        asyncio.run(_run())

    def test_team_stream_redirects_when_running(self):
        """RFC-0014 T2: /team/stream enqueues message when team is running.

        验证 team 运行中 POST /team/stream 不再尝试新 run，
        而是将消息 enqueue 到 leader，并返回 SSE subscribe 流。

        使用 FastAPI TestClient 验证 HTTP 层行为。
        team 在 SSE 流开始后立即停止 (is_running=False)，确保 subscribe 循环退出。
        """

        async def _run():
            engine = InMemoryDatabaseEngine()
            await engine.setup_models(ALL_TEAM_MODELS)

            from nexau.archs.session import SessionManager

            session_manager = SessionManager(
                engine=engine,
                lock_ttl=30.0,
                heartbeat_interval=10.0,
            )

            from nexau.archs.transports.http.team_registry import TeamRegistry

            registry = TeamRegistry(engine=engine, session_manager=session_manager)

            leader_config = _make_agent_config("leader")
            candidate_config = _make_agent_config("worker")
            registry.register_config(
                "default",
                leader_config=leader_config,
                candidates={"worker": candidate_config},
            )

            team = registry.get_or_create("u1", "s1")

            # 模拟 team 正在运行，设置 leader agent
            team._is_running = True
            mock_leader = MagicMock()
            mock_leader.enqueue_message = MagicMock()
            team._leader_agent = mock_leader
            team._leader_agent_id = "leader"

            from nexau.archs.transports.http.team_routes import create_team_router

            # get_history 第一次返回空，第二次返回空但 team 已停止
            call_count_hist = 0

            def mock_get_history(user_id: str, session_id: str, after: int) -> list[dict[str, object]]:
                nonlocal call_count_hist
                call_count_hist += 1
                # 第二次调用时停止 team，让 subscribe 循环退出
                if call_count_hist >= 2:
                    team._is_running = False
                return []

            router = create_team_router(
                registry,
                on_stream_event=None,
                get_history=mock_get_history,
                count_events=lambda u, s: 0,
            )

            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            app = FastAPI()
            app.include_router(router)
            client = TestClient(app)

            # 发送请求（team 正在运行）
            response = client.post(
                "/team/stream",
                json={
                    "user_id": "u1",
                    "session_id": "s1",
                    "message": "继续",
                },
            )

            # 1. 应该返回 200（SSE 流），不是 500 错误
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")

            # 2. 消息应该被 enqueue 到 leader
            mock_leader.enqueue_message.assert_called_once()
            call_args = mock_leader.enqueue_message.call_args[0][0]
            assert call_args["role"] == "user"
            assert call_args["content"] == "继续"

            # 3. SSE 流应该包含 complete 信号（subscribe 正常退出）
            assert "complete" in response.text

        asyncio.run(_run())
