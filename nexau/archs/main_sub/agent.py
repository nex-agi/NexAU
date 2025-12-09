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
import uuid
from collections.abc import Callable
from typing import Any, Literal

import anthropic
import openai
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from anthropic.types import ToolParam

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig, ExecutionConfig
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.prompt_builder import PromptBuilder
from nexau.archs.main_sub.skill import Skill
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
        global_storage: GlobalStorage | None = None,
    ):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration
            global_storage: Optional global storage instance
        """
        self.config = config
        self.global_storage = global_storage if global_storage is not None else GlobalStorage()
        existing_tracer = self.global_storage.get("tracer")
        if existing_tracer is None and self.config.resolved_tracer is not None:
            self.global_storage.set("tracer", self.config.resolved_tracer)
        # Prefer the tool_call_mode defined on AgentConfig when an ExecutionConfig
        # is not explicitly provided to keep Python-created agents consistent with
        # YAML-created ones.
        self.exec_config = ExecutionConfig.from_agent_config(self.config)

        self.tool_call_mode = normalize_tool_call_mode(self.exec_config.tool_call_mode)
        self.use_structured_tool_calls = self.tool_call_mode in STRUCTURED_TOOL_CALL_MODES
        self.tool_call_payload = self._build_tool_call_payload() if self.use_structured_tool_calls else []

        # Initialize services
        self.openai_client = self._initialize_openai_client()

        # Initialize MCP tools if configured
        if self.config.mcp_servers:
            self._initialize_mcp_tools()

        # Build tool registry for quick lookup
        self.tool_registry = {tool.name: tool for tool in self.config.tools}
        self.serial_tool_name = [tool.name for tool in self.config.tools if tool.disable_parallel]

        # Build skill registry for quick lookup
        self.skill_registry = {skill.name: skill for skill in self.config.skills}
        self.global_storage.set("skill_registry", self.skill_registry)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        self._agent_name = self.config.name or f"agent_{uuid.uuid4().hex}"
        self._agent_id = self.config.agent_id or self._agent_name

        # Initialize execution components
        self._initialize_execution_components()

        # Conversation history
        self.history: list[dict[str, Any]] = []

        # Queue for messages to be processed in the next execution cycle
        self.queued_messages: list[dict[str, str]] = []

        # Register for cleanup
        cleanup_manager.register_agent(self)

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

        for sub_agent_name in (self.config.sub_agent_factories or {}).keys():
            tools_spec.append(
                {
                    "type": "function",
                    "function": {
                        "name": build_sub_agent_tool_name(sub_agent_name),
                        "description": f"Delegate work to sub-agent '{sub_agent_name}'.",
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

        for sub_agent_name in (self.config.sub_agent_factories or {}).keys():
            tools_spec.append(
                {
                    "name": build_sub_agent_tool_name(sub_agent_name),
                    "description": f"Delegate work to sub-agent '{sub_agent_name}'.",
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
            agent_name=self._agent_name,
            agent_id=self._agent_id,
            tool_registry=self.tool_registry,
            serial_tool_name=self.serial_tool_name,
            sub_agent_factories=self.config.sub_agent_factories,
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
        message: str,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
    ) -> str | tuple[str, dict[str, Any]]:
        """Run agent with a message and return response."""
        logger.info(f"🤖 Agent '{self.config.name}' starting execution")
        logger.info(f"📝 User message: {message}")

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
            # Build and add system prompt to history
            system_prompt = self.prompt_builder.build_system_prompt(
                agent_config=self.config,
                tools=self.config.tools,
                sub_agent_factories=self.config.sub_agent_factories,
                runtime_context=merged_context,
                include_tool_instructions=not self.use_structured_tool_calls,
            )
            if not self.history:
                self.history = [{"role": "system", "content": system_prompt}]

            if history:
                self.history.extend(history)

            self.history.append({"role": "user", "content": message})

            # Create the AgentState instance
            agent_state = AgentState(
                agent_name=self._agent_name,
                agent_id=self._agent_id,
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
                    message=message,
                    agent_state=agent_state,
                    merged_context=merged_context,
                )
            else:
                return self._run_inner(agent_state, merged_context)

    def _run_with_tracing(
        self,
        tracer: BaseTracer,
        span_type: SpanType,
        message: str,
        agent_state: AgentState,
        merged_context: dict[str, Any],
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
        span_name = f"Agent: {self._agent_name}"
        inputs = {
            "message": message,
            "agent_id": self._agent_id,
        }
        attributes: dict[str, Any] = {
            "agent_name": self._agent_name,
            "model": getattr(self.config.llm_config, "model", None),
        }

        trace_ctx = TraceContext(tracer, span_name, span_type, inputs, attributes)
        with trace_ctx:
            try:
                response = self._run_inner(agent_state, merged_context)
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
            )
            self.history = updated_messages

            # Add final assistant response to history if not already included
            if not self.history or self.history[-1]["role"] != "assistant" or self.history[-1]["content"] != response:
                self.history.append(
                    {"role": "assistant", "content": response},
                )

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
                self.history.append(
                    {"role": "assistant", "content": error_response},
                )
                return error_response
            else:
                error_message = f"Error: {str(e)}"
                self.history.append(
                    {"role": "assistant", "content": error_message},
                )
                raise

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.config.tools.append(tool)
        self.tool_registry[tool.name] = tool
        self.executor.add_tool(tool)

    def add_sub_agent(self, name: str, agent_factory: Callable[[], "Agent"]) -> None:
        """Add a sub-agent factory for delegation."""
        self.config.sub_agent_factories[name] = agent_factory
        self.executor.add_sub_agent(name, agent_factory)

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


# Factory function for agent creation
def create_agent(
    name: str | None = None,
    agent_id: str | None = None,
    tools: list[Tool] | None = None,
    sub_agents: list[tuple[str, Callable[[], "Agent"]]] | None = None,
    skills: list[Skill] | None = None,
    system_prompt: str | None = None,
    system_prompt_type: Literal["string", "file", "jinja"] = "string",
    llm_config: LLMConfig | dict[str, Any] | None = None,
    max_iterations: int = 100,
    max_context_tokens: int = 128000,
    max_running_subagents: int = 5,
    error_handler: Callable[..., Any] | None = None,
    retry_attempts: int = 5,
    timeout: int = 300,
    # Token counting parameters
    token_counter: Callable[[list[dict[str, str]]], int] | None = None,
    # Context parameters
    initial_state: dict[str, Any] | None = None,
    initial_config: dict[str, Any] | None = None,
    initial_context: dict[str, Any] | None = None,
    # MCP parameters
    mcp_servers: list[dict[str, Any]] | None = None,
    # Stop tools parameters
    stop_tools: list[str] | set[str] | None = None,
    # Hook parameters
    after_model_hooks: list[Callable[..., Any]] | None = None,
    after_tool_hooks: list[Callable[..., Any]] | None = None,
    before_model_hooks: list[Callable[..., Any]] | None = None,
    before_tool_hooks: list[Callable[..., Any]] | None = None,
    middlewares: list[Callable[..., Any]] | None = None,
    # Global storage parameter
    global_storage: GlobalStorage | None = None,
    tool_call_mode: str = "xml",
    tracers: list[BaseTracer] | None = None,
    **llm_kwargs: Any,
) -> Agent:
    """Create a new agent with specified configuration."""
    # Handle llm_config creation with backward compatibility
    if llm_config is None and llm_kwargs:
        llm_config = LLMConfig(**llm_kwargs)
    elif llm_config is None:
        raise ValueError("llm_config is required")

    # Create agent configuration
    agent_kwargs: dict[str, Any] = {
        "name": name,
        "agent_id": agent_id if agent_id else str(uuid.uuid4()),
        "system_prompt": system_prompt,
        "system_prompt_type": system_prompt_type,
        "tools": tools or [],
        "sub_agents": sub_agents,
        "skills": skills or [],
        "llm_config": llm_config,
        "stop_tools": set(stop_tools) if stop_tools else None,
        "initial_state": initial_state,
        "initial_config": initial_config,
        "initial_context": initial_context,
        "mcp_servers": mcp_servers or [],
        "after_model_hooks": after_model_hooks,
        "after_tool_hooks": after_tool_hooks,
        "before_model_hooks": before_model_hooks,
        "before_tool_hooks": before_tool_hooks,
        "middlewares": middlewares,
        "error_handler": error_handler,
        "token_counter": token_counter,
        "max_iterations": max_iterations,
        "max_context_tokens": max_context_tokens,
        "max_running_subagents": max_running_subagents,
        "tool_call_mode": tool_call_mode,
        "retry_attempts": retry_attempts,
        "timeout": timeout,
        "tracers": tracers or [],
    }

    agent_config = AgentConfig(**agent_kwargs)

    return Agent(agent_config, global_storage)
