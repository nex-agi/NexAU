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

"""Refactored Agent implementation for the NexAU framework."""

import logging
import traceback
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import dotenv
import openai
import yaml
from anthropic.types import ToolParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig, ConfigError, ExecutionConfig
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.prompt_builder import PromptBuilder
from nexau.archs.main_sub.sub_agent_naming import build_sub_agent_tool_name
from nexau.archs.main_sub.tool_call_modes import (
    STRUCTURED_TOOL_CALL_MODES,
    normalize_tool_call_mode,
)
from nexau.archs.main_sub.utils.cleanup_manager import cleanup_manager
from nexau.archs.main_sub.utils.token_counter import TokenCounter
from nexau.archs.tool import Tool
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import BaseTracer, SpanType
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock

# Setup logger for agent execution
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class Agent:
    """Lightweight agent container focusing on configuration and delegation."""

    def __init__(
        self,
        config: AgentConfig,
        agent_id: str | None = None,
        global_storage: GlobalStorage | None = None,
    ):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration
            global_storage: Optional global storage instance
        """
        self.config: AgentConfig = config
        self.global_storage = global_storage if global_storage is not None else GlobalStorage()
        if self.config.resolved_tracer is not None:
            self.global_storage.set("tracer", self.config.resolved_tracer)
        # Prefer the tool_call_mode defined on AgentConfig when an ExecutionConfig
        # is not explicitly provided to keep Python-created agents consistent with
        # YAML-created ones.
        self.exec_config = ExecutionConfig.from_agent_config(self.config)

        self.tool_call_mode = normalize_tool_call_mode(self.exec_config.tool_call_mode)
        self.use_structured_tool_calls = self.tool_call_mode in STRUCTURED_TOOL_CALL_MODES

        # Initialize services
        self.openai_client = self._initialize_openai_client()

        # Initialize MCP tools if configured
        if self.config.mcp_servers:
            self._initialize_mcp_tools()

        # Build tool payloads after all tools (including MCP) are loaded
        self.tool_call_payload = self._build_tool_call_payload() if self.use_structured_tool_calls else []

        # Build tool registry for quick lookup
        self.tool_registry = {tool.name: tool for tool in self.config.tools}
        self.serial_tool_name = [tool.name for tool in self.config.tools if tool.disable_parallel]

        # Build skill registry for quick lookup
        self.skill_registry = {skill.name: skill for skill in self.config.skills}
        self.global_storage.set("skill_registry", self.skill_registry)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        self.agent_name = self.config.name or f"agent_{uuid.uuid4().hex}"
        self.agent_id = agent_id or uuid.uuid4().hex[:8]

        # Initialize execution components
        self._initialize_execution_components()

        # Conversation history
        self.history: list[Message] = []

        # Queue for messages to be processed in the next execution cycle
        self.queued_messages: list[Message] = []

        # Register for cleanup
        cleanup_manager.register_agent(self)

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        agent_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        global_storage: GlobalStorage | None = None,
    ) -> "Agent":
        """
        Create agent from YAML file.

        Args:
            config_path: Path to the agent configuration YAML file
            overrides: Dictionary of configuration overrides
            template_context: Context variables for Jinja template rendering
            global_storage: Optional global storage instance

        Returns:
            Configured Agent instance
        """
        if overrides:
            warnings.warn(
                "The overrides parameter is deprecated and will be removed in a future "
                "version. Please use AgentConfig.from_yaml() to load the configuration, "
                "modify attributes directly (e.g., agent_config.key = value), and then "
                "initialize the Agent using Agent(agent_config).",
            )
        try:
            dotenv.load_dotenv()
            if not config_path.exists():
                raise ConfigError(f"Configuration file not found: {config_path}")

            # Load config schema from YAML configuration
            agent_config = AgentConfig.from_yaml(config_path, overrides)

            if global_storage is None:
                global_storage = GlobalStorage()
            if agent_config.global_storage:
                global_storage.update(agent_config.global_storage)

            # if config.get("system_prompt_type") == "jinja" and template_context:

            return cls(config=agent_config, agent_id=agent_id, global_storage=global_storage)

        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parsing error in {config_path}: {e}")
        except Exception as e:
            traceback.print_exc()
            raise ConfigError(
                f"Error loading configuration from {config_path}: {e}",
            )

    def _initialize_openai_client(self) -> Any:
        """Initialize OpenAI client from LLM config."""
        # Guard clause
        llm_config = self.config.llm_config or LLMConfig()

        try:
            if llm_config.api_type == "anthropic_chat_completion":
                client_kwargs = llm_config.to_client_kwargs()
                return anthropic.Anthropic(**client_kwargs)
            if llm_config.api_type in ["openai_responses", "openai_chat_completion"]:
                client_kwargs = llm_config.to_client_kwargs()
                return openai.OpenAI(**client_kwargs)
            raise ValueError(f"Invalid API type: {llm_config.api_type}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            return None

    def _initialize_mcp_tools(self) -> None:
        """Initialize tools from MCP servers."""
        try:
            from ..tool.builtin import sync_initialize_mcp_tools

            logger.info(
                f"Initializing MCP tools from {len(self.config.mcp_servers)} servers",
            )

            mcp_tools = sync_initialize_mcp_tools(self.config.mcp_servers)
            self.config.tools.extend(mcp_tools)

            logger.info(f"Successfully initialized {len(mcp_tools)} MCP tools")

        except ImportError:
            logger.error(
                "MCP client not available. Please install the mcp package.",
            )
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")

    def _build_openai_tool_specs(self) -> list[ChatCompletionToolParam]:
        """Convert configured tools and sub-agents into OpenAI tool definitions."""
        tools_spec: list[ChatCompletionToolParam] = [tool.to_openai() for tool in self.config.tools]

        if not self.config.sub_agents:
            return tools_spec
        for sub_agent_name in (self.config.sub_agents or {}).keys():
            tools_spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": build_sub_agent_tool_name(sub_agent_name),
                        "description": (
                            self.config.sub_agents[sub_agent_name].description or f"Delegate work to sub-agent '{sub_agent_name}'."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "Task or question for the sub-agent.",
                                },
                            },
                            "required": ["message"],
                        },
                    },
                },
            )

        return tools_spec

    def _build_anthropic_tool_specs(self) -> list[ToolParam]:
        """Convert tools and sub-agents into anthropic tool definitions."""
        tools_spec: list[ToolParam] = [tool.to_anthropic() for tool in self.config.tools]

        if not self.config.sub_agents:
            return tools_spec
        for sub_agent_name in (self.config.sub_agents or {}).keys():
            tools_spec.append(
                {
                    "name": build_sub_agent_tool_name(sub_agent_name),
                    "description": self.config.sub_agents[sub_agent_name].description or f"Delegate work to sub-agent '{sub_agent_name}'.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Task or question for the sub-agent.",
                            },
                        },
                        "required": ["message"],
                    },
                },
            )

        return tools_spec

    def _build_tool_call_payload(self) -> list[ChatCompletionToolParam] | list[ToolParam]:
        """Build structured tool definitions for the active tool_call_mode."""
        if self.tool_call_mode == "openai":
            return self._build_openai_tool_specs()
        if self.tool_call_mode == "anthropic":
            return self._build_anthropic_tool_specs()
        return []

    def _initialize_execution_components(self) -> None:
        """Initialize execution components."""
        token_counter = self._resolve_token_counter()
        # Initialize the Executor
        self.executor = Executor(
            agent_name=self.agent_name,
            agent_id=self.agent_id,
            tool_registry=self.tool_registry,
            serial_tool_name=self.serial_tool_name,
            sub_agents=self.config.sub_agents or {},
            stop_tools=self.config.stop_tools or set(),
            openai_client=self.openai_client,
            llm_config=self.config.llm_config or LLMConfig(),
            max_iterations=self.exec_config.max_iterations,
            max_context_tokens=self.exec_config.max_context_tokens,
            max_running_subagents=self.exec_config.max_running_subagents,
            retry_attempts=self.exec_config.retry_attempts,
            token_counter=token_counter,
            after_model_hooks=self.config.after_model_hooks,
            after_tool_hooks=self.config.after_tool_hooks,
            before_model_hooks=self.config.before_model_hooks,
            before_tool_hooks=self.config.before_tool_hooks,
            middlewares=self.config.middlewares,
            global_storage=self.global_storage,
            tool_call_mode=self.tool_call_mode,
            openai_tools=self.tool_call_payload,
        )

    def _resolve_token_counter(self) -> TokenCounter:
        """Cast configured token counter to TokenCounter instance."""
        configured_counter = self.config.token_counter
        if isinstance(configured_counter, TokenCounter):
            return configured_counter

        if callable(configured_counter):
            custom_counter = TokenCounter()
            custom_counter._counter = configured_counter  # type: ignore[attr-defined]
            return custom_counter

        return TokenCounter()

    def run(
        self,
        message: str | list[Message],
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
    ) -> str | tuple[str, dict[str, Any]]:
        """Run agent with a message and return response."""
        logger.info(f"🤖 Agent '{self.config.name}' starting execution")
        message_text_for_logs = (
            message
            if isinstance(message, str)
            else next(
                (m.get_text_content() for m in reversed(message) if m.role == Role.USER and m.get_text_content()),
                f"<{len(message)} Message blocks>",
            )
        )
        logger.info(f"📝 User message: {message_text_for_logs}")

        # Merge initial state/config/context with provided ones
        merged_state = {**(self.config.initial_state or {})}
        if state:
            merged_state.update(state)

        merged_config = {**(self.config.initial_config or {})}
        if config:
            merged_config.update(config)

        merged_context = {**(self.config.initial_context or {})}
        if context:
            merged_context.update(context)

        # Get tracer from global storage
        tracer: BaseTracer | None = self.global_storage.get("tracer")

        # Determine span type based on whether this is a sub-agent
        span_type = SpanType.SUB_AGENT if parent_agent_state else SpanType.AGENT

        # Create agent context
        with AgentContext(context=merged_context) as ctx:
            runtime_client = self.openai_client
            if custom_llm_client_provider:
                try:
                    override_client = custom_llm_client_provider(self.agent_name)
                    if override_client is not None:
                        runtime_client = override_client
                except Exception as exc:  # Defensive: user provided callable
                    logger.warning(f"⚠️ custom_llm_client_provider failed for '{self.agent_name}': {exc}")

            # Build and add system prompt to history
            system_prompt = self.prompt_builder.build_system_prompt(
                agent_config=self.config,
                tools=self.config.tools,
                sub_agents=self.config.sub_agents or {},
                runtime_context=merged_context,
                include_tool_instructions=not self.use_structured_tool_calls,
            )
            if not self.history:
                self.history = [Message(role=Role.SYSTEM, content=[TextBlock(text=system_prompt)])]

            if history:
                self.history.extend(messages_from_legacy_openai_chat(history))

            if isinstance(message, str):
                self.history.append(Message.user(message))
            else:
                self.history.extend(message)

            # Create the AgentState instance
            agent_state = AgentState(
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                context=ctx,
                global_storage=self.global_storage,
                parent_agent_state=parent_agent_state,
                executor=self.executor,
            )

            # Execute with or without tracing
            if tracer:
                return self._run_with_tracing(
                    tracer=tracer,
                    span_type=span_type,
                    message_text_for_logs=message_text_for_logs,
                    agent_state=agent_state,
                    merged_context=merged_context,
                    runtime_client=runtime_client,
                    custom_llm_client_provider=custom_llm_client_provider,
                )
            else:
                return self._run_inner(
                    agent_state,
                    merged_context,
                    runtime_client=runtime_client,
                    custom_llm_client_provider=custom_llm_client_provider,
                )

    def _run_with_tracing(
        self,
        tracer: BaseTracer,
        span_type: SpanType,
        message_text_for_logs: str,
        agent_state: AgentState,
        merged_context: dict[str, Any],
        runtime_client: Any,
        custom_llm_client_provider: Callable[[str], Any] | None,
    ) -> str:
        """Execute agent with tracing enabled.

        Args:
            tracer: The tracer instance to use
            span_type: Type of span (AGENT or SUB_AGENT)
            message: User message
            agent_state: Agent state instance
            merged_context: Merged context dictionary

        Returns:
            Agent response string
        """
        span_name = f"Agent: {self.agent_name}"
        inputs = {
            "message": message_text_for_logs,
            "agent_id": self.agent_id,
        }
        attributes: dict[str, Any] = {
            "agent_name": self.agent_name,
            "model": getattr(self.config.llm_config, "model", None),
        }

        trace_ctx = TraceContext(tracer, span_name, span_type, inputs, attributes)
        with trace_ctx:
            try:
                response = self._run_inner(
                    agent_state,
                    merged_context,
                    runtime_client=runtime_client,
                    custom_llm_client_provider=custom_llm_client_provider,
                )
                # Set outputs - TraceContext will handle ending the span
                trace_ctx.set_outputs({"response": response})
                return response
            except Exception:
                # TraceContext will handle the error, but we still need to re-raise
                raise

    def _run_inner(
        self,
        agent_state: AgentState,
        merged_context: dict[str, Any],
        *,
        runtime_client: Any,
        custom_llm_client_provider: Callable[[str], Any] | None,
    ) -> str:
        """Inner execution logic without tracing wrapper.

        Args:
            agent_state: Agent state instance
            merged_context: Merged context dictionary

        Returns:
            Agent response string
        """
        try:
            # Execute using the executor
            response, updated_messages = self.executor.execute(
                self.history,
                agent_state,
                runtime_client=runtime_client,
                custom_llm_client_provider=custom_llm_client_provider,
            )
            self.history = updated_messages

            # Add final assistant response to history if not already included
            if not self.history or self.history[-1].role != Role.ASSISTANT or self.history[-1].get_text_content() != response:
                self.history.append(Message.assistant(response))

            logger.info(
                f"✅ Agent '{self.config.name}' completed execution",
            )
            return response

        except Exception as e:
            logger.error(
                f"❌ Agent '{self.config.name}' encountered error: {e}",
            )

            if self.config.error_handler:
                error_response = self.config.error_handler(
                    e,
                    self,
                    merged_context,
                )
                self.history.append(Message.assistant(error_response))
                return error_response
            else:
                error_message = f"Error: {str(e)}"
                self.history.append(Message.assistant(error_message))
                raise

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.config.tools.append(tool)
        self.tool_registry[tool.name] = tool
        self.executor.add_tool(tool)

    def add_sub_agent(self, name: str, agent_config: AgentConfig) -> None:
        """Add a sub-agent config."""
        if self.config.sub_agents is None:
            self.config.sub_agents = {}
        self.config.sub_agents[name] = agent_config
        self.executor.add_sub_agent(name, agent_config)

    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be added to the history."""
        self.executor.enqueue_message(message)

    def stop(self) -> None:
        """Clean up this agent and all its running sub-agents."""
        logger.info(
            f"🧹 Cleaning up agent '{self.config.name}' and its sub-agents...",
        )
        self.executor.cleanup()
        logger.info(f"✅ Agent '{self.config.name}' cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup when agent is garbage collected."""
        try:
            self.stop()
        except Exception:
            pass  # Avoid exceptions during garbage collection
