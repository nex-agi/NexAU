"""Sub-agent management and lifecycle control."""

import logging
import threading
from collections.abc import Callable
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.utils.xml_utils import XMLParser

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse

    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    Langfuse = None


class SubAgentManager:
    """Manages sub-agent lifecycle and delegation."""

    def __init__(
        self,
        agent_name: str,
        sub_agent_factories: dict[str, Callable[[], Any]],
        langfuse_client=None,
        global_storage=None,
        main_tracer=None,
    ):
        """Initialize sub-agent manager.

        Args:
            agent_name: Name of the parent agent
            sub_agent_factories: Dictionary mapping sub-agent names to factory functions
            langfuse_client: Optional Langfuse client for tracing
            global_storage: Optional global storage to share with sub-agents
            main_tracer: Optional main agent's tracer for generating sub-agent trace paths
        """
        from nexau.archs.main_sub.agent import Agent

        self.agent_name = agent_name
        self.sub_agent_factories = sub_agent_factories
        self.langfuse_client = langfuse_client
        self.global_storage = global_storage
        self.main_tracer = main_tracer
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

        parent_trace_id = getattr(parent_agent_state, "langfuse_trace_id", None)
        if parent_trace_id:
            sub_agent.langfuse_trace_id = parent_trace_id
        # Ensure sub-agents reuse the parent's Langfuse client and trace when available
        if self.langfuse_client:
            sub_agent.langfuse_client = self.langfuse_client
            if hasattr(sub_agent, "executor") and sub_agent.executor:
                sub_agent.executor.langfuse_client = self.langfuse_client
                if hasattr(sub_agent.executor, "subagent_manager") and sub_agent.executor.subagent_manager:
                    sub_agent.executor.subagent_manager.langfuse_client = self.langfuse_client

        try:
            # Generate sub-agent trace path if main agent has tracing enabled
            sub_agent_trace_path = None
            # Get the main agent's tracer to generate sub-agent trace path
            if self.main_tracer and self.main_tracer.is_tracing():
                main_trace_path = self.main_tracer.get_dump_path()
                if main_trace_path:
                    # Use the tracer's method to generate sub-agent trace path
                    if hasattr(self.main_tracer, "generate_sub_agent_trace_path"):
                        sub_agent_trace_path = self.main_tracer.generate_sub_agent_trace_path(
                            sub_agent_name,
                            main_trace_path,
                        )
                        if sub_agent_trace_path:
                            logger.info(
                                f"📊 Sub-agent '{sub_agent_name}' will generate trace to: {sub_agent_trace_path}",
                            )

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
                dump_trace_path=sub_agent_trace_path,
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

    def shutdown(self):
        """Signal shutdown to prevent new sub-agent tasks."""
        self._shutdown_event.set()
        for sub_agent_id, sub_agent in self.running_sub_agents.items():
            try:
                sub_agent.stop()
            except Exception as e:
                logger.error(
                    f"❌ Error shutting down sub-agent {sub_agent_id}: {e}",
                )

    def add_sub_agent(self, name: str, agent_factory: Callable[[], Any]):
        """Add a sub-agent factory for delegation.

        Args:
            name: Name of the sub-agent
            agent_factory: Factory function that creates the agent
        """
        self.sub_agent_factories[name] = agent_factory
