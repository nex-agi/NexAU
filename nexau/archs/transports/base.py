"""Base transport abstractions for Nexau.

This module defines the common interfaces and abstractions for all transport
implementations (HTTP, stdio, WebSocket, gRPC, etc.).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from nexau.archs.llm.llm_aggregators.events import Event
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.context_value import ContextValue
from nexau.archs.main_sub.execution.hooks import Middleware
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware
from nexau.archs.main_sub.execution.stop_result import StopResult
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import DatabaseEngine
from nexau.core.messages import Message

if TYPE_CHECKING:
    from nexau.archs.main_sub.config import AgentConfig

logger = logging.getLogger(__name__)


class TransportBase[TTransportConfig](ABC):
    """Base class for all transport servers using ORM Repository pattern.

    This base class provides common functionality for all transport implementations,
    using the ORM Repository pattern for session and agent model management.

    Architecture:
        - SessionModel: Lightweight, stores agent_ids and context
        - AgentModel: Independent storage, one per agent
        - AgentRunActionModel: Conversation history actions (APPEND/UNDO/REPLACE)
        - ORMRepository[T]: Generic repository with DatabaseEngine
        - DatabaseEngine: InMemoryDatabaseEngine, SQLDatabaseEngine (pluggable)

    Example:
        >>> from nexau.archs.session.orm import SQLDatabaseEngine
        >>>
        >>> engine = SQLDatabaseEngine.from_url("sqlite+aiosqlite:///sessions.db")
        >>>
        >>> transport = MyTransport(
        ...     engine=engine,
        ...     config=config,
        ...     default_agent_config=agent_config,
        ... )
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        config: TTransportConfig,
        default_agent_config: AgentConfig,
        lock_ttl: float = 30.0,
        heartbeat_interval: float = 10.0,
    ):
        """Initialize transport base.

        Args:
            engine: DatabaseEngine for all model storage (SessionModel, AgentModel, AgentRunActionModel)
            config: Transport-specific configuration
            default_agent_config: Default agent configuration
            lock_ttl: Lock time-to-live in seconds (default: 30s)
            heartbeat_interval: Heartbeat interval in seconds (default: 10s)
        """
        self._config = config
        self._default_agent_config = default_agent_config
        self._engine = engine

        # Create SessionManager with shared engine and lock configuration
        self._session_manager = SessionManager(
            engine=engine,
            lock_ttl=lock_ttl,
            heartbeat_interval=heartbeat_interval,
        )

        # RFC-0001 Phase 4: 运行中 Agent 注册表，供 interrupt 查找
        # key: (user_id, session_id, agent_id)
        self._running_agents: dict[tuple[str, str, str], Agent] = {}
        self._running_agents_lock = asyncio.Lock()

    @staticmethod
    def _recursively_apply_middlewares(
        cfg: AgentConfig,
        *middlewares: Middleware,
        enable_stream: bool = False,
    ) -> AgentConfig:
        """Recursively apply multiple middlewares to agent config and all sub-agents.

        Args:
            cfg: Agent configuration to modify
            middlewares: Middleware instances to add
            enable_stream: If True, enable streaming on llm_config for all agents
        """
        # Save stateful objects before deep copy to avoid pickling issues
        # - Tracers (e.g., LangfuseTracer with httpx clients containing thread locks)
        # - Middlewares (e.g., ContextCompactionMiddleware with unpicklable state)
        # These objects contain unpicklable state (thread locks, connections, etc.)
        saved_tracers = cfg.tracers
        saved_resolved_tracer = cfg.resolved_tracer
        saved_middlewares = cfg.middlewares

        # Temporarily clear stateful objects for deep copy
        cfg.tracers = []
        cfg.resolved_tracer = None
        cfg.middlewares = None

        try:
            cfg_copy = cfg.model_copy(deep=True)
        finally:
            # Restore original stateful objects
            cfg.tracers = saved_tracers
            cfg.resolved_tracer = saved_resolved_tracer
            cfg.middlewares = saved_middlewares

        # Assign stateful objects to the copy (shallow - should be shared, not duplicated)
        cfg_copy.tracers = saved_tracers
        cfg_copy.resolved_tracer = saved_resolved_tracer
        cfg_copy.middlewares = list(saved_middlewares) if saved_middlewares else []

        # Add new middlewares
        cfg_copy.middlewares.extend(middlewares)

        # Enable streaming if requested
        if enable_stream and cfg_copy.llm_config:
            cfg_copy.llm_config.stream = True

        if cfg_copy.sub_agents:
            for name, sub_cfg in cfg_copy.sub_agents.items():
                if sub_cfg:
                    cfg_copy.sub_agents[name] = TransportBase._recursively_apply_middlewares(
                        sub_cfg, *middlewares, enable_stream=enable_stream
                    )

        return cfg_copy

    @abstractmethod
    def start(self) -> None:
        """Start the transport server."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the transport server."""

    async def handle_request(
        self,
        *,
        message: str | list[Message],
        user_id: str,
        agent_config: AgentConfig | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        variables: ContextValue | None = None,
    ) -> str:
        """Handle a single request.

        Agent internally handles all session/agent initialization including:
        - Database table initialization
        - Session creation
        - Agent registration with root_agent_id reuse

        Args:
            message: User message or list of messages
            user_id: User ID
            agent_config: Optional agent configuration (uses default if None)
            session_id: Optional session ID (creates new if None)
            context: Optional context dict to merge with session context

        Returns:
            Agent response string
        """
        start_time = datetime.now()
        logger.info("handle_request (user_id: %s, session_id: %s)", user_id, session_id or "new")

        # Generate session_id if not provided
        if session_id is None:
            from nexau.archs.session.id_generator import generate_session_id

            session_id = generate_session_id()

        # Apply middlewares
        events_mw = AgentEventsMiddleware(session_id=session_id, on_event=lambda _: None)
        config_with_middlewares = self._recursively_apply_middlewares(
            agent_config or self._default_agent_config,
            events_mw,
        )

        # Create agent in thread pool (Agent.__init__ does blocking/sync session init
        # via asyncio.run in that thread; avoids event-loop nesting in async handler)
        def create_agent() -> Agent:
            return Agent(
                config=config_with_middlewares,
                session_manager=self._session_manager,
                user_id=user_id,
                session_id=session_id,
                variables=variables,
            )

        agent: Agent = await asyncio.to_thread(create_agent)

        # RFC-0001 Phase 4: 注册 agent 到运行表
        agent_key = (user_id, session_id, agent.agent_id)
        async with self._running_agents_lock:
            self._running_agents[agent_key] = agent

        try:
            # Run agent (agent handles locking and persistence internally)
            response = cast(str, await agent.run_async(message=message, context=context, variables=variables))
        finally:
            # RFC-0001 Phase 4: 从运行表移除
            async with self._running_agents_lock:
                self._running_agents.pop(agent_key, None)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info("handle_request completed in %.2fs (session_id: %s)", duration, session_id)

        return response

    async def handle_streaming_request(
        self,
        *,
        message: str | list[Message],
        user_id: str,
        agent_config: AgentConfig | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        variables: ContextValue | None = None,
    ) -> AsyncGenerator[Event]:
        """Handle a streaming request.

        Agent internally handles all session/agent initialization including:
        - Database table initialization
        - Session creation
        - Agent registration with root_agent_id reuse

        Args:
            message: User message or list of messages
            user_id: User ID
            agent_config: Optional agent configuration (uses default if None)
            session_id: Optional session ID (creates new if None)
            context: Optional context dict to merge with session context

        Yields:
            Event objects for streaming response
        """
        start_time = datetime.now()
        logger.info("handle_streaming_request (user_id: %s, session_id: %s)", user_id, session_id or "new")

        # Generate session_id if not provided
        if session_id is None:
            from nexau.archs.session.id_generator import generate_session_id

            session_id = generate_session_id()

        # Apply middlewares with streaming enabled
        event_queue: asyncio.Queue[Event] = asyncio.Queue()
        events_mw = AgentEventsMiddleware(session_id=session_id, on_event=event_queue.put_nowait)
        config_with_middlewares = self._recursively_apply_middlewares(
            agent_config or self._default_agent_config,
            events_mw,
            enable_stream=True,
        )

        # Create agent in thread pool (Agent.__init__ does blocking/sync session init
        # via asyncio.run in that thread; avoids event-loop nesting in async handler)
        def create_agent() -> Agent:
            return Agent(
                config=config_with_middlewares,
                session_manager=self._session_manager,
                user_id=user_id,
                session_id=session_id,
                variables=variables,
            )

        agent: Agent = await asyncio.to_thread(create_agent)

        # RFC-0001 Phase 4: 注册 agent 到运行表
        agent_key = (user_id, session_id, agent.agent_id)
        async with self._running_agents_lock:
            self._running_agents[agent_key] = agent

        async def run_agent() -> str:
            return cast(str, await agent.run_async(message=message, context=context, variables=variables))

        agent_task = asyncio.create_task(run_agent())

        try:
            # Stream events from queue
            while not agent_task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    yield event
                except TimeoutError:
                    continue

            # Drain remaining events
            while not event_queue.empty():
                yield event_queue.get_nowait()

            # Wait for agent task to complete (RunFinishedEvent is emitted by middleware)
            await agent_task
        finally:
            # RFC-0001 Phase 4: 从运行表移除
            async with self._running_agents_lock:
                self._running_agents.pop(agent_key, None)

        duration = (datetime.now() - start_time).total_seconds()
        logger.info("handle_streaming_request completed in %.2fs (session_id: %s)", duration, session_id)

    async def handle_stop_request(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
        force: bool = False,
        timeout: float = 30.0,
    ) -> StopResult:
        """Handle a stop request for a running agent.

        RFC-0001 Phase 4: Transport 层 stop 端点

        在运行表中查找匹配的 Agent 实例并调用 stop()。

        Args:
            user_id: User ID
            session_id: Session ID
            agent_id: Optional agent ID (if None, stops the first matching agent)
            force: True 立即停止，False 优雅停止
            timeout: Maximum seconds to wait for execution to complete

        Returns:
            StopResult with message snapshot and stop reason

        Raises:
            ValueError: If no matching running agent is found
        """
        logger.info(
            "handle_stop_request (user_id: %s, session_id: %s, agent_id: %s, force: %s)",
            user_id,
            session_id,
            agent_id or "any",
            force,
        )

        # 查找匹配的运行中 Agent
        agent: Agent | None = None
        async with self._running_agents_lock:
            if agent_id:
                agent = self._running_agents.get((user_id, session_id, agent_id))
            else:
                # 未指定 agent_id 时，查找该 session 下任意运行中的 agent
                for key, running_agent in self._running_agents.items():
                    if key[0] == user_id and key[1] == session_id:
                        agent = running_agent
                        break

        if agent is None:
            raise ValueError(f"No running agent found for user_id={user_id}, session_id={session_id}, agent_id={agent_id or 'any'}")

        result = await agent.stop(force=force, timeout=timeout)
        logger.info(
            "handle_stop_request completed (session_id: %s, messages: %d)",
            session_id,
            len(result.messages),
        )
        return result

    def __enter__(self) -> TransportBase[TTransportConfig]:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()
