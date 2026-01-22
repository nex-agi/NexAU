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

"""Service for managing agent metadata (AgentModel CRUD operations)."""

from __future__ import annotations

from .id_generator import generate_agent_id
from .models import AgentModel
from .orm import AndFilter, ComparisonFilter, DatabaseEngine


class AgentService:
    """Service for managing agent metadata.

    This service provides methods to:
    - Register agents in a session
    - Check if an agent exists
    - Retrieve agent metadata

    All operations are scoped by (user_id, session_id, agent_id).
    """

    def __init__(self, *, engine: DatabaseEngine) -> None:
        """Initialize the service.

        Args:
            engine: Database engine for persistence
        """
        self._engine = engine

    async def register_agent(
        self,
        *,
        user_id: str,
        session_id: str,
        agent_id: str | None = None,
        agent_name: str = "",
    ) -> str:
        """Register an agent in the session.

        This method is idempotent - if the agent already exists, it returns the existing agent_id.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Optional agent ID. If None, generates a new one.
            agent_name: Agent name

        Returns:
            The registered agent_id (e.g., "agent_a1b2c3d4")
        """
        # Generate agent_id if not provided
        if agent_id is None:
            agent_id = generate_agent_id()

        # Check if agent already exists (idempotent)
        existing = await self.get_agent(user_id=user_id, session_id=session_id, agent_id=agent_id)
        if existing is not None:
            return agent_id

        # Create new agent
        agent = AgentModel(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            agent_name=agent_name,
        )
        await self._engine.create(agent)
        return agent_id

    async def has_agent(self, *, user_id: str, session_id: str, agent_id: str) -> bool:
        """Check if an agent exists in the session.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            True if agent exists, False otherwise
        """
        agent = await self.get_agent(user_id=user_id, session_id=session_id, agent_id=agent_id)
        return agent is not None

    async def get_agent(self, *, user_id: str, session_id: str, agent_id: str) -> AgentModel | None:
        """Get agent metadata.

        Args:
            user_id: User identifier
            session_id: Session identifier
            agent_id: Agent identifier

        Returns:
            AgentModel if found, None otherwise
        """
        filters = AndFilter(
            filters=[
                ComparisonFilter.eq("user_id", user_id),
                ComparisonFilter.eq("session_id", session_id),
                ComparisonFilter.eq("agent_id", agent_id),
            ]
        )
        return await self._engine.find_first(AgentModel, filters=filters)
