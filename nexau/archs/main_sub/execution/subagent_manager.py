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

"""Sub-agent management and lifecycle control."""

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.utils.xml_utils import XMLParser

from ..agent_context import GlobalStorage

if TYPE_CHECKING:
    from nexau.archs.session import SessionManager

logger = logging.getLogger(__name__)


class SubAgentManager:
    """Manages sub-agent lifecycle and delegation."""

    def __init__(
        self,
        agent_name: str,
        sub_agents: dict[str, AgentConfig],
        global_storage: GlobalStorage | None = None,
        session_manager: "SessionManager | None" = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize sub-agent manager.

        Args:
            agent_name: Name of the parent agent
            sub_agents: Dictionary mapping sub-agent names to AgentConfig objects
            global_storage: Optional global storage to share with sub-agents
            session_manager: Optional SessionManager for unified data access
            user_id: Optional user ID for persistence
            session_id: Optional session ID for persistence
        """
        from nexau.archs.main_sub.agent import Agent

        self.agent_name = agent_name
        self.sub_agents: dict[str, AgentConfig] = sub_agents
        self.global_storage = global_storage
        self.session_manager = session_manager
        self.user_id = user_id
        self.session_id = session_id
        self.xml_parser = XMLParser()
        self._shutdown_event = threading.Event()
        self.running_sub_agents: dict[str, Agent] = {}

    def call_sub_agent(
        self,
        sub_agent_name: str,
        message: str,
        sub_agent_id: str | None = None,
        context: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
        parallel_execution_id: str | None = None,
    ) -> str:
        """Call a sub-agent like a tool call.

        Args:
            sub_agent_name: Name of the sub-agent to call
            message: Message to send to the sub-agent
            context: Optional context to pass

        Returns:
            Result from the sub-agent

        Raises:
            RuntimeError: If agent is shutting down
            ValueError: If sub-agent is not found
        """
        from ...main_sub.agent import Agent
        from ..agent_context import get_context

        # Check if agent is shutting down
        if self._shutdown_event.is_set():
            logger.warning(
                f"⚠️ Agent '{self.agent_name}' is shutting down, cannot call sub-agent '{sub_agent_name}'",
            )
            raise RuntimeError(f"Agent '{self.agent_name}' is shutting down")

        logger.info(
            f"🤖➡️🤖 Agent '{self.agent_name}' calling sub-agent '{sub_agent_name}' with message: {message}",
        )

        if sub_agent_name not in self.sub_agents:
            error_msg = f"Sub-agent '{sub_agent_name}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        sub_agent_config = self.sub_agents[sub_agent_name]

        # Recall existing sub-agent by ID (will restore history from agent_repo)
        if sub_agent_id is not None:
            logger.info(
                f"🔄🤖 Recall sub-agent '{sub_agent_name}' with id '{sub_agent_id}' - history will be restored from storage",
            )
            # Create new Agent instance with the same agent_id
            # Agent.run() will restore history from agent_repo automatically
            sub_agent = Agent(
                agent_id=sub_agent_id,
                config=sub_agent_config,
                global_storage=self.global_storage,
                session_manager=self.session_manager,
                user_id=self.user_id,
                session_id=self.session_id,
                is_root=False,
            )
        else:
            # Instantiate a new sub-agent (Agent.__init__ handles registration)
            sub_agent = Agent(
                config=sub_agent_config,
                global_storage=self.global_storage,
                session_manager=self.session_manager,
                user_id=self.user_id,
                session_id=self.session_id,
                is_root=False,
            )

        self.running_sub_agents[sub_agent.agent_id] = sub_agent

        try:
            effective_context = None
            if context:
                effective_context = context
            else:
                # Pass current agent context state, config, and context to sub-agent
                current_context = get_context()
                if current_context:
                    # Use context from current agent context if not explicitly provided
                    effective_context = current_context.context.copy()

            # Store parallel_execution_id in agent_state if provided
            if parallel_execution_id and parent_agent_state:
                parent_agent_state.set_global_value("parallel_execution_id", parallel_execution_id)

            result = sub_agent.run(
                message=message,
                context=effective_context,
                parent_agent_state=parent_agent_state,
                custom_llm_client_provider=custom_llm_client_provider,
            )
            result = (
                f"{result}\n"
                f"Sub-agent finished (sub_agent_name: {sub_agent.agent_name}, "
                f"sub_agent_id: {sub_agent.agent_id}. Recall this agent if needed)."
            )

            logger.info(
                f"✅ Sub-agent '{sub_agent_name}' returned result to agent '{self.agent_name}'",
            )
            # Sub-agent history is persisted via agent_repo, no need to keep in memory
            return str(result)

        except Exception as e:
            logger.error(f"❌ Sub-agent '{sub_agent_name}' failed: {e}")
            raise
        finally:
            self.running_sub_agents.pop(sub_agent.agent_id, None)

    def shutdown(self) -> None:
        """Signal shutdown to prevent new sub-agent tasks."""
        self._shutdown_event.set()
        for sub_agent_id, sub_agent in self.running_sub_agents.items():
            try:
                sub_agent.sync_cleanup()
            except Exception as e:
                logger.error(
                    f"❌ Error shutting down sub-agent {sub_agent_id}: {e}",
                )

    def add_sub_agent(self, name: str, agent_config: AgentConfig) -> None:
        """Add a sub-agent config.

        Args:
            name: Name of the sub-agent
            agent_config: Config to create the sub-agent
        """
        self.sub_agents[name] = agent_config
