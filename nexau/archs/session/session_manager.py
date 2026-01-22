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

"""Unified session manager - single entry point for all session operations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nexau.archs.main_sub.agent_context import GlobalStorage

from .agent_lock_service import AgentLockService
from .agent_run_action_service import AgentRunActionService
from .agent_service import AgentService
from .models import AgentModel, AgentRunActionModel, SessionModel
from .models.agent_lock import AgentLockModel
from .orm import AndFilter, ComparisonFilter, DatabaseEngine


class SessionManager:
    """Unified session manager that encapsulates all session-related operations.

    SessionManager provides a clean API for:
    - Agent management (register, get, update metadata)
    - Agent run actions (via agent_run_action service)
    - Agent locking (via agent_lock service)

    Example:
        >>> from nexau.archs.session.orm import InMemoryDatabaseEngine
        >>> engine = InMemoryDatabaseEngine()
        >>> manager = SessionManager(engine=engine)
        >>> await manager.setup_models()
        >>>
        >>> # Register an agent (reuses root_agent_id if exists)
        >>> agent_id = await manager.register_agent(user_id="user1", session_id="sess_123", agent_name="main_agent")
        >>>
        >>> # Use agent lock
        >>> async with manager.agent_lock.acquire("sess_123", agent_id):
        ...     # Only one execution per agent at a time
        ...     pass
    """

    def __init__(
        self,
        *,
        engine: DatabaseEngine,
        lock_ttl: float = 30.0,
        heartbeat_interval: float = 10.0,
    ) -> None:
        """Initialize session manager.

        Args:
            engine: DatabaseEngine instance for data storage
            lock_ttl: Lock time-to-live in seconds (default: 30s)
            heartbeat_interval: Heartbeat interval in seconds (default: 10s)
        """
        self._engine: DatabaseEngine = engine
        self._agent_service = AgentService(engine=engine)
        self._agent_run_action = AgentRunActionService(engine=engine)
        self._agent_lock = AgentLockService(
            engine=engine,
            lock_ttl=lock_ttl,
            heartbeat_interval=heartbeat_interval,
        )
        self._models_initialized = False

    # Public services (exposed for direct use)
    @property
    def agent_run_action(self) -> AgentRunActionService:
        """Get the AgentRunActionService for managing agent run actions."""
        return self._agent_run_action

    @property
    def agent_lock(self) -> AgentLockService:
        """Get the AgentLockService for agent-level locking."""
        return self._agent_lock

    # Session management API
    async def get_session(self, *, user_id: str, session_id: str) -> SessionModel | None:
        """Get a session by user_id and session_id.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            SessionModel if found, None otherwise
        """
        filters = AndFilter(
            filters=[
                ComparisonFilter.eq("user_id", user_id),
                ComparisonFilter.eq("session_id", session_id),
            ]
        )
        return await self._engine.find_first(SessionModel, filters=filters)

    async def _get_or_create_session(self, *, user_id: str, session_id: str) -> SessionModel:
        """Get existing session or create a new one."""
        session = await self.get_session(user_id=user_id, session_id=session_id)
        if session is None:
            session = SessionModel(user_id=user_id, session_id=session_id)
            session = await self._engine.create(session)
        return session

    async def _update_session(self, session: SessionModel) -> SessionModel:
        """Update an existing session."""
        return await self._engine.update(session)

    async def _get_root_agent_id(self, *, user_id: str, session_id: str) -> str | None:
        """Get the root agent ID for a session."""
        session = await self.get_session(user_id=user_id, session_id=session_id)
        return session.root_agent_id if session else None

    # Agent management API
    async def register_agent(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
        agent_name: str = "",
        is_root: bool = True,
    ) -> tuple[str, SessionModel]:
        """Register an agent in the session.

        This method handles agent ID resolution with the following logic:
        1. If agent_id is provided, use it directly
        2. If is_root=True and session already has root_agent_id, reuse it
        3. Otherwise, generate a new agent_id
        4. Register the agent in AgentModel
        5. Update root_agent_id in SessionModel if is_root=True and not already set

        This ensures that root agents maintain consistent IDs across multiple
        requests within the same session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Optional agent ID. If None, may reuse root_agent_id or generate new.
            agent_name: Agent name
            is_root: If True, reuse existing root_agent_id if available

        Returns:
            Tuple of (agent_id, session):
            - agent_id: The registered agent_id (e.g., "agent_a1b2c3d4")
            - session: The SessionModel (for accessing storage without extra query)
        """
        # Get or create session first - this is the only session query needed
        session = await self._get_or_create_session(user_id=user_id, session_id=session_id)

        # Determine the agent_id to use
        agent_id_to_use = agent_id

        # Root agent reuse logic: if no agent_id provided and this is root agent,
        # try to reuse existing root_agent_id from session
        if agent_id_to_use is None and is_root and session.root_agent_id is not None:
            agent_id_to_use = session.root_agent_id

        # Register agent in AgentModel (generates new ID if agent_id_to_use is None)
        agent_id_result = await self._agent_service.register_agent(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id_to_use,
            agent_name=agent_name,
        )

        # Update root_agent_id in SessionModel if needed
        if is_root and session.root_agent_id is None:
            session.root_agent_id = agent_id_result
            session = await self._update_session(session)

        return agent_id_result, session

    async def get_agent(self, *, user_id: str, session_id: str, agent_id: str) -> AgentModel | None:
        """Get agent metadata.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            AgentModel if found, None otherwise
        """
        return await self._agent_service.get_agent(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
        )

    async def update_agent_metadata(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str,
        agent_name: str | None = None,
    ) -> AgentModel | None:
        """Update agent metadata.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier
            agent_name: Optional new agent name

        Returns:
            Updated AgentModel if found, None otherwise
        """
        agent = await self.get_agent(user_id=user_id, session_id=session_id, agent_id=agent_id)
        if agent is None:
            return None

        if agent_name is not None:
            agent.agent_name = agent_name
        agent.last_updated = datetime.now()
        return await self._engine.update(agent)

    # Initialization
    async def setup_models(self) -> None:
        """Initialize all session-related models.

        This method is idempotent - calling it multiple times is safe.
        """
        if self._models_initialized:
            return
        await self._engine.setup_models(
            [
                SessionModel,
                AgentModel,
                AgentRunActionModel,
                AgentLockModel,
            ]
        )
        self._models_initialized = True

    # Session context and storage management
    async def update_session_context(
        self,
        *,
        user_id: str,
        session_id: str,
        context: dict[str, Any],
    ) -> SessionModel:
        """Update session context (full replacement).

        Args:
            user_id: User identifier
            session_id: Session identifier
            context: Context data to store (replaces existing context entirely)

        Returns:
            Updated SessionModel
        """
        session = await self._get_or_create_session(user_id=user_id, session_id=session_id)
        session.context = context
        session.updated_at = datetime.now()
        return await self._update_session(session)

    async def update_session_storage(
        self,
        *,
        user_id: str,
        session_id: str,
        storage: GlobalStorage,
    ) -> SessionModel:
        """Update session storage.

        Args:
            user_id: User identifier
            session_id: Session identifier
            storage: GlobalStorage to persist (will be serialized automatically)

        Returns:
            Updated SessionModel
        """
        session = await self._get_or_create_session(user_id=user_id, session_id=session_id)
        session.storage = storage
        session.updated_at = datetime.now()
        return await self._update_session(session)

    async def update_session_state(
        self,
        *,
        user_id: str,
        session_id: str,
        context: dict[str, Any],
        storage: GlobalStorage,
    ) -> SessionModel:
        """Update both session context and storage in a single operation.

        This method avoids the overhead of two separate session lookups
        when both context and storage need to be updated.

        Args:
            user_id: User identifier
            session_id: Session identifier
            context: Context data to store (replaces existing context entirely)
            storage: GlobalStorage to persist (will be serialized automatically)

        Returns:
            Updated SessionModel
        """
        session = await self._get_or_create_session(user_id=user_id, session_id=session_id)
        session.context = context
        session.storage = storage
        session.updated_at = datetime.now()
        return await self._update_session(session)
