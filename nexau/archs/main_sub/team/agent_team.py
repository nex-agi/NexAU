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

"""Team lifecycle manager.

RFC-0002: AgentTeam 生命周期管理

Manages leader + teammate agents, shared task board,
message bus, and concurrent execution.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import Future
from typing import TYPE_CHECKING
from uuid import uuid4

from nexau.archs.llm.llm_aggregators.events import RunErrorEvent, TeamMessageEvent, UserMessageEvent
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware
from nexau.archs.main_sub.team.message_bus import TeamMessageBus
from nexau.archs.main_sub.team.sse.multiplexer import TeamSSEMultiplexer
from nexau.archs.main_sub.team.state import AgentTeamState
from nexau.archs.main_sub.team.task_board import TaskBoard
from nexau.archs.main_sub.team.tools import get_leader_tools, get_teammate_tools
from nexau.archs.main_sub.team.types import MaxTeammatesError, TeammateInfo
from nexau.archs.main_sub.team.watchdog import TeammateWatchdog, WatchdogConfig
from nexau.archs.sandbox import (
    BaseSandbox,
    BaseSandboxManager,
    E2BSandboxConfig,
    E2BSandboxManager,
    LocalSandboxConfig,
    LocalSandboxManager,
)
from nexau.archs.session.models.team import TeamModel
from nexau.archs.session.models.team_member import TeamMemberModel
from nexau.archs.session.models.team_message import TeamMessageModel
from nexau.archs.session.models.team_task import TeamTaskModel
from nexau.archs.session.models.team_task_lock import TeamTaskLockModel
from nexau.archs.session.orm import AndFilter, ComparisonFilter
from nexau.archs.session.task_lock_service import TaskLockService

if TYPE_CHECKING:
    from nexau.archs.main_sub.config import AgentConfig
    from nexau.archs.main_sub.team.sse.envelope import TeamStreamEnvelope
    from nexau.archs.session.orm import DatabaseEngine
    from nexau.archs.session.session_manager import SessionManager

logger = logging.getLogger(__name__)


def _safe_deepcopy_config(config: AgentConfig) -> AgentConfig:
    """Deepcopy AgentConfig while preserving unpicklable fields.

    RFC-0002: 安全深拷贝 AgentConfig

    某些字段包含不可 pickle 的对象（如 tracer 的有状态连接、
    middleware 中的 OpenAI client 持有 httpx 连接和 _thread.RLock、
    hook 中的闭包等），需要在 deepcopy 前暂存并在拷贝后恢复。
    """
    # 1. 暂存不可 pickle 的字段
    saved = {
        "tracers": config.tracers,
        "resolved_tracer": config.resolved_tracer,
        "middlewares": config.middlewares,
        "after_model_hooks": config.after_model_hooks,
        "after_tool_hooks": config.after_tool_hooks,
        "before_model_hooks": config.before_model_hooks,
        "before_tool_hooks": config.before_tool_hooks,
    }

    # 2. 清空不可 pickle 的字段
    config.tracers = []
    config.resolved_tracer = None
    config.middlewares = None
    config.after_model_hooks = None
    config.after_tool_hooks = None
    config.before_model_hooks = None
    config.before_tool_hooks = None

    # 3. 执行 deepcopy
    copied = copy.deepcopy(config)

    # 4. 恢复原始 config 和拷贝后的 config
    for field, value in saved.items():
        setattr(config, field, value)
        setattr(copied, field, value)

    return copied


class AgentTeam:
    """Team lifecycle manager.

    RFC-0002: AgentTeam 生命周期管理

    Manages leader + teammate agents, shared task board,
    message bus, and concurrent execution.
    """

    def __init__(
        self,
        *,
        leader_config: AgentConfig,
        candidates: dict[str, AgentConfig],
        engine: DatabaseEngine,
        session_manager: SessionManager,
        user_id: str,
        session_id: str,
        max_teammates: int = 10,
    ) -> None:
        self._leader_config = leader_config
        self._candidates = candidates
        self._engine = engine
        self._session_manager = session_manager
        self._user_id = user_id
        self._team_session_id = session_id
        self.max_teammates = max_teammates

        # Generated on initialize
        self._team_id: str = ""
        self._leader_agent_id: str = ""
        self._task_board: TaskBoard | None = None
        self._message_bus: TeamMessageBus | None = None
        self._watchdog: TeammateWatchdog | None = None

        # Shared sandbox manager for all agents in the team
        self._shared_sandbox_manager: BaseSandboxManager[BaseSandbox] | None = None

        # Teammate tracking
        self._teammate_agents: dict[str, Agent] = {}
        self._teammate_futures: dict[str, Future[None]] = {}
        self._role_counters: dict[str, int] = {}
        self._errored_agents: set[str] = set()  # agent_ids that exited with error

        # Leader agent reference (set in run())
        self._leader_agent: Agent | None = None

        # Main event loop reference (set in run(), used by spawn_teammate)
        self._loop: asyncio.AbstractEventLoop | None = None

        # SSE multiplexer (set in run() when on_event is provided)
        self._multiplexer: TeamSSEMultiplexer | None = None

        # Run lifecycle tracking (for SSE reconnection support)
        self._is_running: bool = False
        self._on_run_complete: Callable[[], None] | None = None

    @property
    def team_id(self) -> str:
        """Team identifier."""
        return self._team_id

    @property
    def leader_agent_id(self) -> str:
        """Leader agent identifier."""
        return self._leader_agent_id

    @property
    def task_board(self) -> TaskBoard:
        """Shared task board."""
        if self._task_board is None:
            raise RuntimeError("AgentTeam not initialized. Call initialize() first.")
        return self._task_board

    @property
    def message_bus(self) -> TeamMessageBus:
        """Team message bus."""
        if self._message_bus is None:
            raise RuntimeError("AgentTeam not initialized. Call initialize() first.")
        return self._message_bus

    @property
    def is_running(self) -> bool:
        """Whether the team is currently executing a run."""
        return self._is_running

    def set_on_complete(self, callback: Callable[[], None]) -> None:
        """Set callback invoked when the team run completes.

        Used by HTTP transport to clean up registry after run finishes,
        even if the SSE consumer has disconnected.
        """
        self._on_run_complete = callback

    async def initialize(self) -> None:
        """Initialize or restore team state.

        RFC-0002: 初始化或恢复团队状态

        Steps:
        1. Setup database models
        2. Create or restore team record
        3. Create shared services (TaskBoard, MessageBus, Watchdog)
        4. Restore existing teammate counters

        Idempotent: safe to call multiple times.
        """
        # 跳过重复初始化
        if self._task_board is not None:
            return

        # 1. 初始化数据库模型
        await self._engine.setup_models(
            [
                TeamModel,
                TeamMemberModel,
                TeamTaskModel,
                TeamTaskLockModel,
                TeamMessageModel,
            ]
        )

        # 2. 创建或恢复团队记录
        self._team_id = f"team-{uuid4().hex[:8]}"
        self._leader_agent_id = "leader"

        existing = await self._engine.find_first(
            TeamModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", self._user_id),
                    ComparisonFilter.eq("session_id", self._team_session_id),
                ]
            ),
        )
        if existing is not None:
            self._team_id = existing.team_id
            self._leader_agent_id = existing.leader_agent_id
            logger.info(f"Restored team: {self._team_id}")
        else:
            team = TeamModel(
                user_id=self._user_id,
                session_id=self._team_session_id,
                team_id=self._team_id,
                leader_agent_id=self._leader_agent_id,
                candidates={k: v.name or k for k, v in self._candidates.items()},
                max_teammates=self.max_teammates,
            )
            await self._engine.create(team)
            logger.info(f"Created team: {self._team_id}")

        # 3. 创建共享服务
        task_lock = TaskLockService(engine=self._engine)
        self._task_board = TaskBoard(
            engine=self._engine,
            task_lock_service=task_lock,
            user_id=self._user_id,
            session_id=self._team_session_id,
            team_id=self._team_id,
        )
        self._message_bus = TeamMessageBus(
            engine=self._engine,
            user_id=self._user_id,
            session_id=self._team_session_id,
            team_id=self._team_id,
        )
        self._message_bus.set_agent_delivery(
            deliver_message=self.send_message_to_agent,
            get_broadcast_recipients=lambda: [info.agent_id for info in self.get_teammate_info()],
        )
        self._watchdog = TeammateWatchdog(
            config=WatchdogConfig(),
            check_all_idle=self.is_all_idle,
            notify_leader=self.notify_leader,
        )

        # 4. 恢复已有 teammate 计数器
        existing_members = await self._engine.find_many(
            TeamMemberModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", self._user_id),
                    ComparisonFilter.eq("session_id", self._team_session_id),
                    ComparisonFilter.eq("team_id", self._team_id),
                ]
            ),
        )
        for member in existing_members:
            if member.status != "stopped":
                current = self._role_counters.get(member.role_name, 0)
                self._role_counters[member.role_name] = current + 1

    async def spawn_teammate(self, role_name: str) -> str:
        """Spawn a new teammate instance and start it in forever-run mode.

        RFC-0002: 创建 Teammate 实例并启动永久运行循环

        Args:
            role_name: Role name from candidates dict.

        Returns:
            agent_id of the spawned teammate.

        Raises:
            MaxTeammatesError: If max_teammates limit reached.
            ValueError: If role_name not in candidates.
        """
        if len(self._teammate_agents) >= self.max_teammates:
            raise MaxTeammatesError(f"Max teammates limit reached ({self.max_teammates})")

        if role_name not in self._candidates:
            raise ValueError(f"Unknown role: {role_name}")

        # 1. 自增 ID
        count = self._role_counters.get(role_name, 0) + 1
        self._role_counters[role_name] = count
        agent_id = f"{role_name}-{count}"

        # 2. 生成 teammate 独立 session_id
        agent_session_id = f"{self._team_session_id}:{agent_id}"

        # 3. 创建 teammate 记录（含独立 session_id）
        member = TeamMemberModel(
            user_id=self._user_id,
            session_id=self._team_session_id,
            team_id=self._team_id,
            agent_id=agent_id,
            member_session_id=agent_session_id,
            role_name=role_name,
            status="idle",
        )
        await self._engine.create(member)

        # 4. 注入 team 上下文到 teammate system prompt（deepcopy 避免污染原始 config）
        #    浅拷贝会导致 skills 等嵌套对象被共享，_initialize_sandbox 修改 skill.folder 时
        #    会污染原始 candidate config，导致后续 spawn 同角色时 upload_assets 路径错误
        config = _safe_deepcopy_config(self._candidates[role_name])
        # 注入 teammate tools 到 deepcopy 后的 config（避免污染原始 candidate config）
        config.tools = list(config.tools) + get_teammate_tools()
        teammate_lines: list[str] = []
        for aid, a in self._teammate_agents.items():
            future = self._teammate_futures.get(aid)
            status = "running" if future and not future.done() else "idle"
            name = a.config.name or aid
            teammate_lines.append(f"- `{aid}` (role={name}, status={status})")
        team_context = (
            "\n\n# Team Context\n\n"
            f"your_agent_id: {agent_id}\n"
            f"team_id: {self._team_id}\n"
            f"leader_agent_id: {self._leader_agent_id}\n\n"
            "Active teammates:\n"
            + ("\n".join(teammate_lines) if teammate_lines else "- (none yet)")
            + "\n\nUse `list_teammates` and `list_tasks` to get latest status."
        )
        if config.system_prompt:
            config.system_prompt_suffix = team_context
        else:
            config.system_prompt = team_context.lstrip()

        # 5a. 注入 SSE 事件中间件（若 multiplexer 已激活）
        if self._multiplexer is not None:
            handler = self._multiplexer.create_event_handler(agent_id=agent_id, role_name=role_name)
            mw = AgentEventsMiddleware(session_id=agent_session_id, on_event=handler)
            if config.middlewares is None:
                config.middlewares = []
            config.middlewares.append(mw)
            if config.llm_config:
                config.llm_config.stream = True

        # 6. 创建 teammate Agent 实例（独立 session，带 team_state）
        teammate_team_state = AgentTeamState(
            team=self,
            task_board=self.task_board,
            message_bus=self.message_bus,
            is_leader=False,
        )
        agent = Agent(
            agent_id=agent_id,
            config=config,
            session_manager=self._session_manager,
            user_id=self._user_id,
            session_id=agent_session_id,
            is_root=False,
            team_state=teammate_team_state,
            sandbox_manager=self._shared_sandbox_manager,
        )
        self._teammate_agents[agent_id] = agent

        # 6. 在主事件循环上启动 teammate 的永久运行协程
        #    spawn_teammate 在 asyncio.run() 创建的临时事件循环中执行（tool executor），
        #    必须通过 run_coroutine_threadsafe 调度到主循环，否则 task 会随临时循环销毁。
        if self._loop is None:
            raise RuntimeError("AgentTeam._loop not set. Call run() first.")
        teammate_future = asyncio.run_coroutine_threadsafe(
            self._run_teammate_forever(agent_id),
            self._loop,
        )
        self._teammate_futures[agent_id] = teammate_future

        logger.info(f"Spawned teammate: {agent_id} (role={role_name})")
        return agent_id

    async def remove_teammate(self, agent_id: str) -> None:
        """Remove a teammate instance.

        RFC-0002: 移除 Teammate 实例

        Force-stops the agent's executor to break the forever-run loop,
        then cancels the asyncio.Task as safety net.
        """
        agent = self._teammate_agents.pop(agent_id, None)

        # 1. Force-stop executor 以退出永久运行循环
        if agent is not None:
            agent.executor.force_stop()

        # 2. 取消 Future 作为安全兜底
        future = self._teammate_futures.pop(agent_id, None)
        if future is not None and not future.done():
            future.cancel()
            try:
                future.result(timeout=5)
            except (Exception,):
                pass

        # 3. 更新 DB 状态
        member = await self._engine.find_first(
            TeamMemberModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", self._user_id),
                    ComparisonFilter.eq("session_id", self._team_session_id),
                    ComparisonFilter.eq("team_id", self._team_id),
                    ComparisonFilter.eq("agent_id", agent_id),
                ]
            ),
        )
        if member is not None:
            member.status = "stopped"
            await self._engine.update(member)

        # 4. 从 watchdog 注销
        if self._watchdog is not None:
            self._watchdog.unregister(agent_id)

        logger.info(f"Removed teammate: {agent_id}")

    def get_teammate_info(self) -> list[TeammateInfo]:
        """List all teammates and their status.

        RFC-0002: 获取 Teammate 状态列表

        通过 executor.is_idle 判断状态，而非 future.done()。
        executor 在 team_mode 下无 tool call 时进入 idle 等待，
        此时 future 仍在运行，但 agent 实际处于空闲状态。
        """
        results: list[TeammateInfo] = []
        for agent_id, agent in self._teammate_agents.items():
            future = self._teammate_futures.get(agent_id)
            # agent 报错退出 → error
            if agent_id in self._errored_agents:
                status = "error"
            # future 已结束 或 executor 处于 idle 等待 → idle
            elif future is None or future.done() or agent.executor.is_idle:
                status = "idle"
            else:
                status = "running"
            results.append(
                TeammateInfo(
                    agent_id=agent_id,
                    role_name=agent.config.name or agent_id,
                    status=status,
                )
            )
        return results

    def enqueue_user_message(self, to_agent_id: str, content: str) -> None:
        """Enqueue a user message to a specific agent.

        RFC-0002: 向指定 agent 注入用户消息

        与 send_message_to_agent 不同，此方法注入 role=user 的消息，
        用于前端在 stream 期间向 agent 发送后续指令。
        """
        msg = {"role": "user", "content": content}

        # 1. 通过 SSE 通知前端
        if self._multiplexer is not None:
            self._multiplexer.emit(
                agent_id=to_agent_id,
                event=UserMessageEvent(to_agent_id=to_agent_id, content=content),
                role_name="user",
            )

        # 2. 注入消息到目标 agent
        if to_agent_id == self._leader_agent_id:
            if self._leader_agent is not None:
                self._leader_agent.enqueue_message(msg)
                logger.info("Enqueued user message to leader")
            else:
                logger.warning("Cannot enqueue user message: leader agent not set")
        else:
            agent = self._teammate_agents.get(to_agent_id)
            if agent is not None:
                agent.enqueue_message(msg)
                logger.info(f"Enqueued user message to {to_agent_id}")

                # RFC-0002: 若 teammate 的 executor 已退出（报错等），重新启动永久运行循环
                future = self._teammate_futures.get(to_agent_id)
                if future is not None and future.done() and self._loop is not None:
                    logger.info(f"Restarting exited teammate for user message: {to_agent_id}")
                    new_future = asyncio.run_coroutine_threadsafe(
                        self._run_teammate_forever(to_agent_id),
                        self._loop,
                    )
                    self._teammate_futures[to_agent_id] = new_future
            else:
                logger.warning(f"Cannot enqueue user message to unknown agent: {to_agent_id}")

    def send_message_to_agent(
        self,
        to_agent_id: str,
        content: str,
        from_agent_id: str,
    ) -> None:
        """Enqueue a message to a teammate or leader agent.

        RFC-0002: 向 teammate 或 leader 注入消息

        消息通过 executor.enqueue_message 注入，
        executor 的 _message_available 事件会唤醒等待中的永久运行循环。
        """
        enqueue_text = f"[Team Message from {from_agent_id}]: {content}"
        msg = {"role": "user", "content": enqueue_text}

        # 1. 通过 SSE 通知前端
        if self._multiplexer is not None:
            self._multiplexer.emit(
                agent_id=to_agent_id,
                event=TeamMessageEvent(
                    from_agent_id=from_agent_id,
                    to_agent_id=to_agent_id,
                    content=content,
                ),
            )

        # 2. 注入消息到目标 agent
        if to_agent_id == self._leader_agent_id:
            if self._leader_agent is not None:
                self._leader_agent.enqueue_message(msg)
                logger.info(f"Enqueued message to leader from {from_agent_id}")
            else:
                logger.warning("Cannot send message to leader: leader agent not set")
        else:
            agent = self._teammate_agents.get(to_agent_id)
            if agent is not None:
                agent.enqueue_message(msg)
                logger.info(f"Enqueued message to {to_agent_id} from {from_agent_id}")

                # 3. 若 teammate 的 executor 已退出（idle 超时等），重新启动永久运行循环
                future = self._teammate_futures.get(to_agent_id)
                if future is not None and future.done() and self._loop is not None:
                    logger.info(f"Restarting exited teammate: {to_agent_id}")
                    new_future = asyncio.run_coroutine_threadsafe(
                        self._run_teammate_forever(to_agent_id),
                        self._loop,
                    )
                    self._teammate_futures[to_agent_id] = new_future
            else:
                logger.warning(f"Cannot send message to unknown agent: {to_agent_id}")

    def notify_leader(self, content: str, from_agent_id: str) -> None:
        """Send a notification message to the leader agent.

        RFC-0002: 通知 leader agent

        用于任务完成通知、idle 检测等场景，通过 enqueue_message 唤醒 leader。
        """
        self.send_message_to_agent(
            to_agent_id=self._leader_agent_id,
            content=content,
            from_agent_id=from_agent_id,
        )

    def is_all_idle(self) -> bool:
        """Check if all agents (leader + teammates) are idle.

        RFC-0002: 检测所有 agent 是否处于空闲状态

        Returns True when leader and all teammates are in the executor's
        team_mode wait loop. Used by watchdog for deadlock detection.
        """
        # 1. 检查 leader
        if self._leader_agent is not None:
            if not self._leader_agent.executor.is_idle:
                return False
        else:
            return False  # leader 未启动，不算 all-idle

        # 2. 检查所有 teammate
        for agent in self._teammate_agents.values():
            if not agent.executor.is_idle:
                return False

        return True

    async def _run_teammate_forever(self, agent_id: str) -> None:
        """Run teammate in forever-run mode. Exits only on force_stop.

        RFC-0002: 永久运行 Teammate Agent

        Agent 的 executor 在 team_mode 下会持续循环等待消息，
        直到 force_stop() 被调用。
        """
        agent = self._teammate_agents.get(agent_id)
        if agent is None:
            return

        # 更新 DB 状态为 running，清除 error 标记
        await self._update_member_status(agent_id, "running")
        self._errored_agents.discard(agent_id)

        # 注册 watchdog
        if self._watchdog is not None:
            self._watchdog.register(agent_id)

        try:
            # RFC-0002: 不发送激活消息，teammate 启动后直接进入 idle 等待
            await agent.run_async(message=[])
            logger.info(f"Teammate {agent_id} exited normally")
            await self._update_member_status(agent_id, "idle")
        except Exception as e:
            logger.error(f"Teammate {agent_id} exited with error: {e}")
            self._errored_agents.add(agent_id)
            # 1. 通过 SSE 将错误信息 stream 给前端
            if self._multiplexer is not None:
                role_name = agent.config.name or agent_id
                self._multiplexer.emit(
                    agent_id=agent_id,
                    event=RunErrorEvent(
                        run_id=agent_id,
                        message=str(e),
                    ),
                    role_name=role_name,
                )
            # 2. 更新 DB 状态为 error（区别于正常 idle）
            await self._update_member_status(agent_id, "error")
        finally:
            if self._watchdog is not None:
                self._watchdog.unregister(agent_id)

    async def _stop_all_teammates(self) -> None:
        """Force-stop all running teammates.

        RFC-0002: 强制停止所有运行中的 Teammate

        停止所有 teammate 的 executor，等待 Future 完成，
        清理内存引用。DB 记录保留（状态设为 idle），
        下次 run() 时通过 _restore_teammates() 恢复。
        """
        # 1. Force-stop 所有 teammate executor
        stopped_ids = list(self._teammate_agents.keys())
        for agent in self._teammate_agents.values():
            agent.executor.force_stop()

        # 2. 异步等待所有 Future 完成（避免阻塞事件循环）
        for future in self._teammate_futures.values():
            if not future.done():
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=10)
                except (TimeoutError, asyncio.CancelledError, Exception):
                    future.cancel()

        # 3. 确保 DB 状态为 idle（finally 块可能因 CancelledError 未执行）
        for agent_id in stopped_ids:
            await self._update_member_status(agent_id, "idle")

        # 4. 清理内存引用（DB 记录保留，下次 run 恢复）
        self._teammate_agents.clear()
        self._teammate_futures.clear()

    async def _restore_teammates(self) -> None:
        """Restore previously spawned teammates from DB.

        RFC-0002: 恢复之前 spawn 的 teammate

        读取 DB 中状态非 stopped 的 teammate 记录，
        重建 Agent 实例并启动永久运行循环。
        Agent 会通过 session 恢复机制自动恢复对话历史。
        """
        existing_members = await self._engine.find_many(
            TeamMemberModel,
            filters=AndFilter(
                filters=[
                    ComparisonFilter.eq("user_id", self._user_id),
                    ComparisonFilter.eq("session_id", self._team_session_id),
                    ComparisonFilter.eq("team_id", self._team_id),
                ]
            ),
        )

        for member in existing_members:
            # 跳过已停止或已在内存中的 teammate
            if member.status == "stopped":
                continue
            if member.agent_id in self._teammate_agents:
                continue

            role_name = member.role_name
            if role_name not in self._candidates:
                logger.warning(f"Cannot restore {member.agent_id}: unknown role {role_name}")
                continue

            agent_id = member.agent_id
            agent_session_id = member.member_session_id

            # 1. 创建 config 副本并注入 team 上下文（deepcopy 避免污染原始 candidate config）
            config = _safe_deepcopy_config(self._candidates[role_name])
            # 注入 teammate tools 到 deepcopy 后的 config（避免污染原始 candidate config）
            config.tools = list(config.tools) + get_teammate_tools()
            team_context = (
                "\n\n# Team Context\n\n"
                f"your_agent_id: {agent_id}\n"
                f"team_id: {self._team_id}\n"
                f"leader_agent_id: {self._leader_agent_id}\n\n"
                "Use `list_teammates` and `list_tasks` to get latest status."
            )
            if config.system_prompt:
                config.system_prompt_suffix = team_context
            else:
                config.system_prompt = team_context.lstrip()

            # 2. 注入 SSE 事件中间件（若 multiplexer 已激活）
            if self._multiplexer is not None:
                handler = self._multiplexer.create_event_handler(agent_id=agent_id, role_name=role_name)
                mw = AgentEventsMiddleware(session_id=agent_session_id, on_event=handler)
                if config.middlewares is None:
                    config.middlewares = []
                config.middlewares.append(mw)
                if config.llm_config:
                    config.llm_config.stream = True

            # 3. 创建 Agent 实例（session 恢复机制自动恢复对话历史）
            teammate_team_state = AgentTeamState(
                team=self,
                task_board=self.task_board,
                message_bus=self.message_bus,
                is_leader=False,
            )
            agent = Agent(
                agent_id=agent_id,
                config=config,
                session_manager=self._session_manager,
                user_id=self._user_id,
                session_id=agent_session_id,
                is_root=False,
                team_state=teammate_team_state,
                sandbox_manager=self._shared_sandbox_manager,
            )
            self._teammate_agents[agent_id] = agent

            # 4. 在主事件循环上启动永久运行协程
            if self._loop is None:
                raise RuntimeError("AgentTeam._loop not set. Call run() first.")
            teammate_future = asyncio.run_coroutine_threadsafe(
                self._run_teammate_forever(agent_id),
                self._loop,
            )
            self._teammate_futures[agent_id] = teammate_future

            logger.info(f"Restored teammate: {agent_id} (role={role_name})")

    async def stop_all(self) -> None:
        """Force-stop the leader and all running teammates.

        RFC-0002: 外部强制停止整个 Team

        前端 Stop 按钮调用此方法，立即中断 leader 和所有 teammate 的执行循环。
        """
        # 1. 停止 leader
        if self._leader_agent is not None:
            self._leader_agent.executor.force_stop()

        # 2. 停止所有 teammates
        await self._stop_all_teammates()

        # 3. 停止 watchdog
        if self._watchdog is not None:
            self._watchdog.stop()

    async def _update_member_status(self, agent_id: str, status: str) -> None:
        """Update teammate member status in DB.

        RFC-0002: 更新 Teammate 状态
        """
        try:
            member = await self._engine.find_first(
                TeamMemberModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("user_id", self._user_id),
                        ComparisonFilter.eq("session_id", self._team_session_id),
                        ComparisonFilter.eq("team_id", self._team_id),
                        ComparisonFilter.eq("agent_id", agent_id),
                    ]
                ),
            )
            if member is not None:
                member.status = status
                await self._engine.update(member)
        except Exception as e:
            logger.warning(f"Failed to update member status for {agent_id}: {e}")

    async def run(
        self,
        message: str,
    ) -> str:
        """Run the team with the leader agent in forever-run mode.

        RFC-0002: 运行团队（永久运行模式）

        Leader agent 在 team_mode 下持续运行，直到调用 finish_team stop tool。
        Teammate agents 通过 spawn_teammate 按需创建，各自独立运行。
        所有通信通过 enqueue_message 实现。

        SSE 流式输出通过 self._multiplexer 控制：
        - 若 self._multiplexer 已设置（由 run_streaming() 预设），
          自动为 leader 和 teammate 注入 AgentEventsMiddleware。
        - 非流式调用时 self._multiplexer 为 None，不注入中间件。

        Args:
            message: User message to send to the leader.

        Returns:
            Leader agent response string.
        """
        await self.initialize()

        # 保存主事件循环引用，供 spawn_teammate 跨线程调度使用
        self._loop = asyncio.get_running_loop()
        self._is_running = True

        # 1. 启动 watchdog
        watchdog_task: asyncio.Task[None] | None = None
        if self._watchdog is not None:
            watchdog_task = asyncio.create_task(self._watchdog.run())

        try:
            # 2. 注入 candidate 信息到 leader system prompt（deepcopy 避免污染原始 config）
            leader_config = _safe_deepcopy_config(self._leader_config)
            # 注入 team tools 到 deepcopy 后的 leader config（避免污染原始 config）
            leader_config.tools = list(leader_config.tools) + get_leader_tools()
            candidate_lines = [f"- `{name}`: {cfg.description or cfg.name or name}" for name, cfg in self._candidates.items()]
            team_context = (
                "\n\n# Team Context\n\n"
                f"team_id: {self._team_id}\n"
                f"max_teammates: {self.max_teammates}\n\n"
                "Available candidate roles for `spawn_teammate`:\n"
                + "\n".join(candidate_lines)
                + "\n\nYou MUST call `finish_team` when done."
                + "For simple messages that don't need team work, respond and call `finish_team` immediately."
            )
            if leader_config.system_prompt:
                leader_config.system_prompt_suffix = team_context
            else:
                leader_config.system_prompt = team_context.lstrip()

            # 3. 注册 finish_team 为 stop tool
            if leader_config.stop_tools is None:
                leader_config.stop_tools = set()
            leader_config.stop_tools.add("finish_team")

            # 3a. 注入 SSE 事件中间件（若 multiplexer 已激活）
            leader_session_id = f"{self._team_session_id}:leader"
            if self._multiplexer is not None:
                leader_handler = self._multiplexer.create_event_handler(agent_id=self._leader_agent_id, role_name="leader")
                leader_mw = AgentEventsMiddleware(session_id=leader_session_id, on_event=leader_handler)
                if leader_config.middlewares is None:
                    leader_config.middlewares = []
                leader_config.middlewares.append(leader_mw)
                if leader_config.llm_config:
                    leader_config.llm_config.stream = True

            # 4. 创建共享 sandbox manager（所有 agent 共用一个 sandbox 实例）
            if self._shared_sandbox_manager is None:
                leader_sandbox_config = leader_config.sandbox_config
                if leader_sandbox_config is None:
                    leader_sandbox_config = LocalSandboxConfig()
                if isinstance(leader_sandbox_config, E2BSandboxConfig):
                    self._shared_sandbox_manager = E2BSandboxManager(
                        _work_dir=leader_sandbox_config.work_dir,
                        template=leader_sandbox_config.template,
                        timeout=leader_sandbox_config.timeout,
                        api_key=leader_sandbox_config.api_key,
                        api_url=leader_sandbox_config.api_url,
                        metadata=leader_sandbox_config.metadata,
                        envs=leader_sandbox_config.envs,
                    )
                else:
                    self._shared_sandbox_manager = LocalSandboxManager(
                        _work_dir=leader_sandbox_config.work_dir,
                    )
                self._shared_sandbox_manager.prepare_session_context(
                    session_manager=self._session_manager,
                    user_id=self._user_id,
                    session_id=self._team_session_id,
                    sandbox_config=leader_sandbox_config,
                )

            # 5. 创建 team_state 并构建 leader agent（独立 session）
            leader_team_state = AgentTeamState(
                team=self,
                task_board=self.task_board,
                message_bus=self.message_bus,
                is_leader=True,
            )
            leader = Agent(
                agent_id=self._leader_agent_id,
                config=leader_config,
                session_manager=self._session_manager,
                user_id=self._user_id,
                session_id=leader_session_id,
                team_state=leader_team_state,
                sandbox_manager=self._shared_sandbox_manager,
            )
            self._leader_agent = leader

            # 5. 恢复之前 spawn 的 teammate（从 DB 读取，重建 Agent 并启动）
            await self._restore_teammates()

            # 6. 通过 SSE 发送初始用户消息事件
            if self._multiplexer is not None:
                self._multiplexer.emit(
                    agent_id=self._leader_agent_id,
                    event=UserMessageEvent(to_agent_id=self._leader_agent_id, content=message),
                    role_name="user",
                )

            # 7. Leader 在 team_mode 下运行，直到 finish_team 被调用
            raw = await leader.run_async(message=message)
            result = raw[0] if isinstance(raw, tuple) else raw

            return result
        finally:
            # 8. Leader 结束后，强制停止所有 teammate
            await self._stop_all_teammates()

            # 8a. 管理共享 sandbox 生命周期
            if self._shared_sandbox_manager is not None:
                self._shared_sandbox_manager.on_run_complete()
                leader_sandbox_cfg = self._leader_config.sandbox_config
                status_after_run = leader_sandbox_cfg.status_after_run if leader_sandbox_cfg else "stop"
                if status_after_run == "pause":
                    self._shared_sandbox_manager.pause_no_wait()
                elif status_after_run == "stop":
                    self._shared_sandbox_manager.stop()

            # 9. 清理 watchdog
            if watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    pass

            # 10. 关闭 SSE multiplexer（发送流结束信号）
            if self._multiplexer is not None:
                self._multiplexer.close()
                self._multiplexer = None

            # 11. 标记运行结束，触发完成回调（用于 SSE 断连后的注册表清理）
            self._is_running = False
            if self._on_run_complete is not None:
                try:
                    self._on_run_complete()
                except Exception:
                    logger.warning("on_run_complete callback failed", exc_info=True)

    async def run_streaming(
        self,
        message: str,
        on_envelope: Callable[[TeamStreamEnvelope], None] | None = None,
    ) -> AsyncGenerator[TeamStreamEnvelope, None]:
        """Run team with SSE streaming output.

        RFC-0002: Team SSE 流式输出

        创建 TeamSSEMultiplexer，在后台运行 team，
        通过 async generator 输出 TeamStreamEnvelope 事件流。

        Args:
            message: User message to send to the leader.
            on_envelope: Optional callback invoked for each envelope (e.g. for persistence).

        Yields:
            TeamStreamEnvelope events from all agents.
        """
        # 1. 预初始化以获取 team_id
        await self.initialize()

        # 2. 创建 multiplexer 并设置到 self（run() 会检查 self._multiplexer）
        multiplexer = TeamSSEMultiplexer(team_id=self._team_id, on_envelope=on_envelope)
        self._multiplexer = multiplexer

        # 3. 后台运行 team（run() 检测 self._multiplexer 自动注入中间件）
        run_task: asyncio.Task[str] = asyncio.create_task(self.run(message))

        try:
            # 4. 流式输出 envelope 事件
            async for envelope in multiplexer.stream():
                yield envelope

            # 5. 等待 run_task 完成，捕获内部异常
            await run_task
        except Exception:
            multiplexer.close()
            raise
        finally:
            # 6. SSE 断连时不取消 run_task，让 team 继续运行
            # on_envelope 回调会持续将事件存入 EventStore，
            # 前端刷新后可通过 /team/subscribe 重新接收事件。
            # run() 的 finally 块会在执行结束后自动关闭 multiplexer 并触发 on_run_complete。
            pass
