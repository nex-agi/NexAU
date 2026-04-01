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

"""End-to-end tests for Agent.stop() with real LLM calls.

RFC-0001: Agent 中断时状态持久化

Replaces the mock-heavy unit/integration tests with real e2e tests that use:
- Real Agent with InMemoryDatabaseEngine + SessionManager
- Real LLM API calls (no mocks, no fakes, no patches)

Tests cover:
1. Graceful stop returns StopResult with USER_INTERRUPTED
2. Force stop returns StopResult
3. stop() persists session state exactly once (no double-persist)
4. stop() during active LLM execution
5. Lock released after stop
6. Normal run without stop persists exactly once
"""

import asyncio
import os
import uuid
from typing import Any

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.execution.stop_result import StopResult
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager

# ── LLM configuration from environment ──────────────────────────────

# 兼容两种 CI 环境：nexau-cloud-runtime 用 NEXAU_SCHEDULER_SIDECAR_LLM_API_KEY，nexau 用 LLM_API_KEY
_LLM_API_KEY = os.environ.get("NEXAU_SCHEDULER_SIDECAR_LLM_API_KEY") or os.environ.get("LLM_API_KEY", "")
_LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://14.103.60.158:3001/v1")
_LLM_MODEL = "nex-agi/deepseek-v3.1-nex-1"

# Short prompt for fast LLM responses
_FAST_PROMPT = "What is 2+2? Answer with just the number."

# Long prompt so the agent runs long enough to be stopped mid-execution
_SLOW_PROMPT = (
    "Write a detailed 2000-word essay about the history and evolution of "
    "distributed systems, covering topics from early mainframes through "
    "modern cloud computing and microservices architectures."
)


def _skip_if_no_llm_key() -> None:
    """Skip test if LLM API key is not configured."""
    if not _LLM_API_KEY:
        pytest.skip("LLM API key not set (NEXAU_SCHEDULER_SIDECAR_LLM_API_KEY or LLM_API_KEY)")


def _looks_like_transient_upstream_error(exc: BaseException) -> bool:
    """Best-effort detection for flaky live-provider failures."""
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "temporarily unavailable",
            "connection reset",
            "read timed out",
            "timeout error",
        )
    )


def _skip_if_transient_upstream_error(exc: BaseException) -> None:
    """Skip live e2e tests when the upstream provider is temporarily unavailable."""
    if _looks_like_transient_upstream_error(exc):
        pytest.skip(f"Transient upstream LLM failure: {exc}")


def _make_llm_config() -> LLMConfig:
    """Create LLMConfig for real LLM calls."""
    return LLMConfig(
        model=_LLM_MODEL,
        base_url=_LLM_BASE_URL,
        api_key=_LLM_API_KEY,
        api_type="openai_chat_completion",
    )


class _CountingSessionManager(SessionManager):
    """SessionManager that counts update_session_state() calls.

    All operations delegate to the real SessionManager; only
    update_session_state() is instrumented to count persist calls.
    """

    def __init__(self, engine: InMemoryDatabaseEngine) -> None:
        super().__init__(engine=engine)
        self.persist_call_count = 0

    async def update_session_state(self, **kwargs: Any) -> Any:
        self.persist_call_count += 1
        return await super().update_session_state(**kwargs)


async def _make_agent(
    session_manager: SessionManager | _CountingSessionManager,
    session_id: str | None = None,
) -> Agent:
    """Build a real Agent with real LLM config and session manager."""
    config = AgentConfig(
        name="test-stop-agent",
        llm_config=_make_llm_config(),
        system_prompt="You are a helpful assistant.",
    )
    return await Agent.create(
        config=config,
        session_manager=session_manager,
        user_id="test-user",
        session_id=session_id or f"test-session-{uuid.uuid4().hex[:8]}",
    )


# ── Test: graceful stop ──────────────────────────────────────────────


class TestGracefulStop:
    """agent.stop(force=False) returns StopResult with USER_INTERRUPTED."""

    @pytest.mark.anyio
    async def test_graceful_stop_returns_stop_result(self) -> None:
        """Start a long-running agent, gracefully stop it, verify StopResult."""
        _skip_if_no_llm_key()

        sm = SessionManager(engine=InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        async def _stop_after_started() -> StopResult:
            await run_started.wait()
            # 1. 给 LLM 一点时间开始执行
            await asyncio.sleep(0.5)
            return await agent.stop(force=False, timeout=30.0)

        try:
            _, result = await asyncio.gather(
                agent.run_async(message=_SLOW_PROMPT),
                _stop_after_started(),
                return_exceptions=False,
            )
        except Exception as exc:
            _skip_if_transient_upstream_error(exc)
            raise

        assert isinstance(result, StopResult)
        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED


# ── Test: force stop ─────────────────────────────────────────────────


class TestForceStop:
    """agent.stop(force=True) returns StopResult with USER_INTERRUPTED."""

    @pytest.mark.anyio
    async def test_force_stop_returns_stop_result(self) -> None:
        """Start a long-running agent, force stop it, verify StopResult."""
        _skip_if_no_llm_key()

        sm = SessionManager(engine=InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        # 1. 启动 run_async 作为独立 task
        run_task = asyncio.create_task(agent.run_async(message=_SLOW_PROMPT))

        # 2. 等待 run 开始后发送 force stop
        await run_started.wait()
        await asyncio.sleep(0.5)
        result = await agent.stop(force=True)

        # 3. Force stop 无法立即终止线程中阻塞的同步 HTTP 调用，
        #    取消 run_task 并限时等待，避免 HTTP 线程不可取消导致无限挂死
        run_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(run_task), timeout=10.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass
        if run_task.done() and not run_task.cancelled():
            exc = run_task.exception()
            if exc is not None:
                _skip_if_transient_upstream_error(exc)
                raise exc

        assert isinstance(result, StopResult)
        assert result.stop_reason == AgentStopReason.USER_INTERRUPTED


# ── Test: no double-persist ──────────────────────────────────────────


class TestNoPersistDoubling:
    """stop() + run_async() concurrently: exactly one DB persist."""

    @pytest.mark.anyio
    async def test_graceful_stop_persists_exactly_once(self) -> None:
        """Graceful stop during execution: _persist called exactly once by stop()."""
        _skip_if_no_llm_key()

        sm = _CountingSessionManager(InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        async def _stop_after_started() -> StopResult:
            await run_started.wait()
            await asyncio.sleep(0.5)
            return await agent.stop(force=False, timeout=30.0)

        try:
            await asyncio.gather(
                agent.run_async(message=_SLOW_PROMPT),
                _stop_after_started(),
            )
        except Exception as exc:
            _skip_if_transient_upstream_error(exc)
            raise

        assert sm.persist_call_count == 1, f"Expected exactly 1 persist call, got {sm.persist_call_count}"

    @pytest.mark.anyio
    async def test_force_stop_persists_exactly_once(self) -> None:
        """Force stop during execution: _persist called exactly once by stop()."""
        _skip_if_no_llm_key()

        sm = _CountingSessionManager(InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        # 1. 启动 run_async 作为独立 task
        run_task = asyncio.create_task(agent.run_async(message=_SLOW_PROMPT))

        # 2. 等待 run 开始后发送 force stop
        await run_started.wait()
        await asyncio.sleep(0.5)
        await agent.stop(force=True)

        # 3. Force stop 无法立即终止线程中阻塞的同步 HTTP 调用，
        #    取消 run_task 并限时等待，避免 HTTP 线程不可取消导致无限挂死
        run_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(run_task), timeout=10.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass
        if run_task.done() and not run_task.cancelled():
            exc = run_task.exception()
            if exc is not None:
                _skip_if_transient_upstream_error(exc)
                raise exc

        assert sm.persist_call_count == 1, f"Expected exactly 1 persist call, got {sm.persist_call_count}"

    @pytest.mark.anyio
    async def test_normal_run_persists_exactly_once(self) -> None:
        """A run that completes normally (no stop) persists exactly once."""
        _skip_if_no_llm_key()

        sm = _CountingSessionManager(InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        try:
            await agent.run_async(message=_FAST_PROMPT)
        except Exception as exc:
            _skip_if_transient_upstream_error(exc)
            raise

        assert sm.persist_call_count == 1, f"Normal run should persist once, got {sm.persist_call_count}"


# ── Test: stop during active LLM execution ──────────────────────────


class TestStopDuringActiveExecution:
    """stop() during active model execution - agent stops and releases lock."""

    @pytest.mark.anyio
    async def test_stop_during_active_run_releases_lock(self) -> None:
        """Stop while model is generating, verify lock is released after."""
        _skip_if_no_llm_key()

        sm = SessionManager(engine=InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        async def _stop_after_started() -> StopResult:
            await run_started.wait()
            # 2. 等待 executor 真正开始 LLM 调用
            await asyncio.sleep(1.0)
            return await agent.stop(force=False, timeout=30.0)

        try:
            await asyncio.gather(
                agent.run_async(message=_SLOW_PROMPT),
                _stop_after_started(),
            )
        except Exception as exc:
            _skip_if_transient_upstream_error(exc)
            raise

        # 3. 验证 lock 已释放
        is_locked = await sm.agent_lock.is_locked(
            session_id=agent._session_id,
            agent_id=agent.agent_id,
        )
        assert not is_locked, "Lock should be released after stop()"


# ── Test: lock released after stop ───────────────────────────────────


class TestLockReleasedAfterStop:
    """Verify lock is released after both graceful and force stop."""

    @pytest.mark.anyio
    async def test_lock_released_after_graceful_stop(self) -> None:
        """After graceful stop, agent lock is released."""
        _skip_if_no_llm_key()

        sm = SessionManager(engine=InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        async def _stop_after_started() -> StopResult:
            await run_started.wait()
            await asyncio.sleep(0.5)
            return await agent.stop(force=False, timeout=30.0)

        try:
            await asyncio.gather(
                agent.run_async(message=_SLOW_PROMPT),
                _stop_after_started(),
            )
        except Exception as exc:
            _skip_if_transient_upstream_error(exc)
            raise

        is_locked = await sm.agent_lock.is_locked(
            session_id=agent._session_id,
            agent_id=agent.agent_id,
        )
        assert not is_locked, "Lock should be released after graceful stop"

    @pytest.mark.anyio
    async def test_lock_released_after_force_stop(self) -> None:
        """After force stop, agent lock is released."""
        _skip_if_no_llm_key()

        sm = SessionManager(engine=InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        run_started = asyncio.Event()
        original_run_inner = agent._run_async_inner

        async def _wrapped_run_inner(**kwargs: Any) -> Any:
            run_started.set()
            return await original_run_inner(**kwargs)

        agent._run_async_inner = _wrapped_run_inner  # type: ignore[method-assign]

        # 1. 启动 run_async 作为独立 task
        run_task = asyncio.create_task(agent.run_async(message=_SLOW_PROMPT))

        # 2. 等待 run 开始后发送 force stop
        await run_started.wait()
        await asyncio.sleep(0.5)
        await agent.stop(force=True)

        # 3. Force stop 无法立即终止线程中阻塞的同步 HTTP 调用，
        #    取消 run_task 并限时等待，避免 HTTP 线程不可取消导致无限挂死
        run_task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(run_task), timeout=10.0)
        except (TimeoutError, asyncio.CancelledError, Exception):
            pass
        if run_task.done() and not run_task.cancelled():
            exc = run_task.exception()
            if exc is not None:
                _skip_if_transient_upstream_error(exc)
                raise exc

        # 4. 验证 lock 已释放
        is_locked = await sm.agent_lock.is_locked(
            session_id=agent._session_id,
            agent_id=agent.agent_id,
        )
        assert not is_locked, "Lock should be released after force stop"

    @pytest.mark.anyio
    async def test_lock_released_after_normal_run(self) -> None:
        """After a normal run completes, agent lock is released."""
        _skip_if_no_llm_key()

        sm = SessionManager(engine=InMemoryDatabaseEngine())
        agent = await _make_agent(sm)

        try:
            await agent.run_async(message=_FAST_PROMPT)
        except Exception as exc:
            _skip_if_transient_upstream_error(exc)
            raise

        is_locked = await sm.agent_lock.is_locked(
            session_id=agent._session_id,
            agent_id=agent.agent_id,
        )
        assert not is_locked, "Lock should be released after normal run"
