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
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.utils.xml_utils import XMLParser

from ..agent_context import GlobalStorage

logger = logging.getLogger(__name__)


class SubAgentManager:
    """Manages sub-agent lifecycle and delegation."""

    def __init__(
        self,
        agent_name: str,
        sub_agent_factories: dict[str, Callable[..., Any]],
        global_storage: GlobalStorage | None = None,
    ):
        """Initialize sub-agent manager.

        Args:
            agent_name: Name of the parent agent
            sub_agent_factories: Dictionary mapping sub-agent names to factory functions
            global_storage: Optional global storage to share with sub-agents
        """
        from nexau.archs.main_sub.agent import Agent

        self.agent_name = agent_name
        self.sub_agent_factories = sub_agent_factories
        self.global_storage = global_storage
        self.xml_parser = XMLParser()
        self._shutdown_event = threading.Event()
        self.running_sub_agents: dict[str, Agent] = {}

    def call_sub_agent(
        self,
        sub_agent_name: str,
        message: str,
        context: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
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

        if sub_agent_name not in self.sub_agent_factories:
            error_msg = f"Sub-agent '{sub_agent_name}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)

        # Instantiate a fresh sub-agent from the factory
        sub_agent_factory = self.sub_agent_factories[sub_agent_name]

        # Try to create sub-agent with global storage if available
        if self.global_storage is not None:
            try:
                # Try to pass global_storage as keyword argument
                sub_agent = sub_agent_factory(
                    global_storage=self.global_storage,
                )
            except TypeError:
                # If factory doesn't support global_storage parameter, create normally and set afterwards
                sub_agent = sub_agent_factory()
                sub_agent.global_storage = self.global_storage
                # Also ensure the sub-agent's executor uses the same global storage
                if hasattr(sub_agent, "executor"):
                    sub_agent.executor.global_storage = self.global_storage
                if hasattr(sub_agent, "executor") and hasattr(
                    sub_agent.executor,
                    "subagent_manager",
                ):
                    sub_agent.executor.subagent_manager.global_storage = self.global_storage
        else:
            sub_agent = sub_agent_factory()
        self.running_sub_agents[sub_agent.config.agent_id] = sub_agent

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

            result = sub_agent.run(
                message,
                context=effective_context,
                parent_agent_state=parent_agent_state,
            )

            logger.info(
                f"✅ Sub-agent '{sub_agent_name}' returned result to agent '{self.agent_name}'",
            )
            self.running_sub_agents.pop(sub_agent.config.agent_id)
            return result

        except Exception as e:
            logger.error(f"❌ Sub-agent '{sub_agent_name}' failed: {e}")
            raise

    def shutdown(self) -> None:
        """Signal shutdown to prevent new sub-agent tasks."""
        self._shutdown_event.set()
        for sub_agent_id, sub_agent in self.running_sub_agents.items():
            try:
                sub_agent.stop()
            except Exception as e:
                logger.error(
                    f"❌ Error shutting down sub-agent {sub_agent_id}: {e}",
                )

    def add_sub_agent(self, name: str, agent_factory: Callable[[], Any]) -> None:
        """Add a sub-agent factory for delegation.

        Args:
            name: Name of the sub-agent
            agent_factory: Factory function that creates the agent
        """
        self.sub_agent_factories[name] = agent_factory
