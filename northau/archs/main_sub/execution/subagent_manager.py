"""Sub-agent management and lifecycle control."""
import logging
import threading
import xml.etree.ElementTree as ET
from typing import Any
from typing import Callable
from typing import Optional

from ..tracing.tracer import Tracer
from ..utils.xml_utils import XMLParser
from ..utils.xml_utils import XMLUtils

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
    _langfuse_available = True
except ImportError:
    _langfuse_available = False
    Langfuse = None


class SubAgentManager:
    """Manages sub-agent lifecycle and delegation."""

    def __init__(self, agent_name: str, sub_agent_factories: dict[str, Callable[[], Any]], langfuse_client=None, global_storage=None, main_tracer=None):
        """Initialize sub-agent manager.

        Args:
            agent_name: Name of the parent agent
            sub_agent_factories: Dictionary mapping sub-agent names to factory functions
            langfuse_client: Optional Langfuse client for tracing
            global_storage: Optional global storage to share with sub-agents
            main_tracer: Optional main agent's tracer for generating sub-agent trace paths
        """
        self.agent_name = agent_name
        self.sub_agent_factories = sub_agent_factories
        self.langfuse_client = langfuse_client
        self.global_storage = global_storage
        self.main_tracer = main_tracer
        self.xml_parser = XMLParser()
        self._shutdown_event = threading.Event()
        self.running_sub_agents = {}

    def call_sub_agent(self, sub_agent_name: str, message: str, context: Optional[dict[str, Any]] = None) -> str:
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
                if hasattr(sub_agent, 'executor'):
                    sub_agent.executor.global_storage = self.global_storage
                if hasattr(sub_agent, 'executor') and hasattr(sub_agent.executor, 'subagent_manager'):
                    sub_agent.executor.subagent_manager.global_storage = self.global_storage
        else:
            sub_agent = sub_agent_factory()
        self.running_sub_agents[sub_agent.config.agent_id] = sub_agent

        try:
            # Generate sub-agent trace path if main agent has tracing enabled
            sub_agent_trace_path = None
            # Get the main agent's tracer to generate sub-agent trace path
            if self.main_tracer and self.main_tracer.is_tracing():
                main_trace_path = self.main_tracer.get_dump_path()
                if main_trace_path:
                    # Use the tracer's method to generate sub-agent trace path
                    if hasattr(self.main_tracer, 'generate_sub_agent_trace_path'):
                        sub_agent_trace_path = self.main_tracer.generate_sub_agent_trace_path(
                            sub_agent_name, main_trace_path,
                        )
                        if sub_agent_trace_path:
                            logger.info(
                                f"📊 Sub-agent '{sub_agent_name}' will generate trace to: {sub_agent_trace_path}",
                            )

            # Pass current agent context state, config, and context to sub-agent
            current_context = get_context()
            if current_context:
                # Use context from current agent context if not explicitly provided
                effective_context: dict[
                    str,
                    Any,
                ] = context or current_context.context.copy()

                if self.langfuse_client:
                    try:
                        with self.langfuse_client.start_as_current_generation(
                            name=f"subagent_{sub_agent_name}",
                            input=message,
                            metadata={
                                'sub_agent_name': sub_agent_name,
                                'type': 'sub_agent_execution',
                            },
                        ):
                            result = sub_agent.run(
                                message,
                                context=effective_context,
                                dump_trace_path=sub_agent_trace_path,
                            )
                            self.langfuse_client.update_current_generation(
                                output=result,
                            )
                        self.langfuse_client.flush()
                    except Exception as langfuse_error:
                        logger.warning(
                            f"⚠️ Langfuse subagent tracing failed: {langfuse_error}",
                        )
                        result = sub_agent.run(
                            message,
                            context=effective_context,
                            dump_trace_path=sub_agent_trace_path,
                        )
                else:
                    result = sub_agent.run(
                        message,
                        context=effective_context,
                        dump_trace_path=sub_agent_trace_path,
                    )
            else:
                result = sub_agent.run(
                    message, context=context, dump_trace_path=sub_agent_trace_path,
                )

            logger.info(
                f"✅ Sub-agent '{sub_agent_name}' returned result to agent '{self.agent_name}'",
            )
            self.running_sub_agents.pop(sub_agent.config.agent_id)
            return result

        except Exception as e:
            logger.error(f"❌ Sub-agent '{sub_agent_name}' failed: {e}")
            raise

    def execute_sub_agent_from_xml(self, xml_content: str, tracer: Optional[Tracer] = None) -> tuple[str, str]:
        """Execute a sub-agent call from XML content.

        Args:
            xml_content: XML content describing the sub-agent call
            tracer: Optional tracer for logging

        Returns:
            Tuple of (agent_name, result)
        """
        try:
            # Parse XML using robust parsing
            root = self.xml_parser.parse_xml_content(xml_content)

            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                raise ValueError('Missing agent_name in sub-agent XML')

            agent_name = (agent_name_elem.text or '').strip()

            # Get message
            message_elem = root.find('message')
            if message_elem is None:
                raise ValueError('Missing message in sub-agent XML')

            message = (message_elem.text or '').strip()

            # Log sub-agent request to trace if enabled
            if tracer:
                tracer.add_subagent_request(agent_name, message)

            # Execute sub-agent call
            result = self.call_sub_agent(agent_name, message)

            # Log sub-agent response to trace if enabled
            if tracer:
                tracer.add_subagent_response(agent_name, result)

            return agent_name, result

        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")

    def execute_sub_agent_from_xml_safe(self, xml_content: str, tracer: Optional[Tracer] = None) -> tuple[str, str, bool]:
        """Safe wrapper for execute_sub_agent_from_xml that handles exceptions.

        Args:
            xml_content: XML content describing the sub-agent call
            tracer: Optional tracer for logging

        Returns:
            Tuple of (agent_name, result, is_error)
        """
        try:
            agent_name, result = self.execute_sub_agent_from_xml(
                xml_content, tracer,
            )
            return agent_name, result, False
        except Exception as e:
            # Extract agent name for error reporting using more robust parsing
            agent_name = XMLUtils.extract_agent_name_from_xml(xml_content)
            return agent_name, str(e), True

    def shutdown(self):
        """Signal shutdown to prevent new sub-agent tasks."""
        self._shutdown_event.set()
        for sub_agent_id, sub_agent in self.running_sub_agents.items():
            try:
                sub_agent.stop()
            except Exception as e:
                logger.error(f"❌ Error shutting down sub-agent {sub_agent_id}: {e}")

    def add_sub_agent(self, name: str, agent_factory: Callable[[], Any]):
        """Add a sub-agent factory for delegation.

        Args:
            name: Name of the sub-agent
            agent_factory: Factory function that creates the agent
        """
        self.sub_agent_factories[name] = agent_factory
