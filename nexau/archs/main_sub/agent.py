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
import os
import traceback
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import anthropic
import dotenv
import openai
import yaml
from anthropic.types import ToolParam
from asyncer import asyncify, syncify
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig, ConfigError, ExecutionConfig
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.main_sub.prompt_builder import PromptBuilder
from nexau.archs.main_sub.sub_agent_naming import build_sub_agent_tool_name
from nexau.archs.main_sub.tool_call_modes import (
    STRUCTURED_TOOL_CALL_MODES,
    normalize_tool_call_mode,
)
from nexau.archs.main_sub.utils.cleanup_manager import cleanup_manager
from nexau.archs.main_sub.utils.token_counter import TokenCounter
from nexau.archs.sandbox import (
    BaseSandboxManager,
    E2BSandboxManager,
    LocalSandboxManager,
    extract_dataclass_init_kwargs,
)
from nexau.archs.session import AgentRunActionKey, SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.tool import Tool
from nexau.archs.tool.builtin.skill_tool import generate_skill_tool_description, load_skill
from nexau.archs.tracer.context import TraceContext
from nexau.archs.tracer.core import BaseTracer, SpanType
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock

# Setup logger for agent execution
logger = logging.getLogger(__name__)


class Agent:
    """Lightweight agent container focusing on configuration and delegation."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        agent_id: str | None = None,
        global_storage: GlobalStorage | None = None,
        session_manager: SessionManager | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        is_root: bool = True,
    ):
        """Initialize agent with configuration.

        Args:
            config: Agent configuration
            agent_id: Optional agent ID (auto-generated if not provided)
            global_storage: Optional global storage instance
            session_manager: Optional SessionManager for unified data access. If None,
                uses the shared in-memory SessionManager (via InMemoryDatabaseEngine.get_shared_instance()).
            user_id: Optional user ID for persistence
            session_id: Optional session ID for persistence
            is_root: Whether this is the root agent (default True). Set to False for sub-agents.
        """
        logger.info("Initializing Agent (%s)", config.name)

        # Store basic config
        self._is_root = is_root
        self.config: AgentConfig = config
        self._user_id = user_id or f"local_user_{uuid.uuid4().hex[:8]}"
        self._session_id = session_id or f"local_{uuid.uuid4().hex[:8]}"

        # Initialize session_manager
        if session_manager is not None:
            self._session_manager = session_manager
            logger.debug("Using provided SessionManager")
        else:
            default_engine = InMemoryDatabaseEngine.get_shared_instance()
            self._session_manager = SessionManager(engine=default_engine)
            logger.debug("Using shared in-memory SessionManager")

        # Async initialization: load storage and register agent in one call
        self.global_storage, self.agent_id = self._init_session_state(
            provided_storage=global_storage,
            proposed_agent_id=agent_id,
        )
        self.agent_name = self.config.name or self.agent_id

        # Set tracer in global storage (with conflict check)
        self._setup_tracer()

        # Prefer the tool_call_mode defined on AgentConfig when an ExecutionConfig
        # is not explicitly provided to keep Python-created agents consistent with
        # YAML-created ones.
        self.exec_config = ExecutionConfig.from_agent_config(self.config)

        self.tool_call_mode = normalize_tool_call_mode(self.exec_config.tool_call_mode)
        self.use_structured_tool_calls = self.tool_call_mode in STRUCTURED_TOOL_CALL_MODES

        # Initialize services
        logger.info("Initializing LLM client (api_type=%s)", self.config.llm_config.api_type if self.config.llm_config else "default")
        self.openai_client = self._initialize_openai_client()

        # Initialize MCP tools if configured
        if self.config.mcp_servers:
            self._initialize_mcp_tools()

        # Build tool payloads after all tools (including MCP) are loaded
        self.tool_call_payload = self._build_tool_call_payload() if self.use_structured_tool_calls else []
        logger.info(
            "Registered %d tools, %d sub_agents", len(self.config.tools), len(self.config.sub_agents) if self.config.sub_agents else 0
        )

        # Initialize sandbox
        self._initialize_sandbox()

        # Add skill tool for skill using
        tools = self.config.tools
        skills = self.config.skills
        nexau_package_path = Path(__file__).parent.parent.parent
        has_skilled_tools = any(tool.as_skill for tool in tools)
        if has_skilled_tools or skills:
            skill_tool = Tool.from_yaml(
                str(nexau_package_path / "archs" / "tool" / "builtin" / "description" / "skill_tool.yaml"),
                binding=load_skill,
                as_skill=False,
            )
            skill_tool.description += generate_skill_tool_description(skills, tools)
            tools.append(skill_tool)

        # Build tool registry for quick lookup
        self.tool_registry = {tool.name: tool for tool in tools}
        self.serial_tool_name = [tool.name for tool in tools if tool.disable_parallel]

        # Build skill registry for quick lookup
        self.skill_registry = {skill.name: skill for skill in skills}
        self.global_storage.set("skill_registry", self.skill_registry)

        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()

        # Initialize execution components
        self._initialize_execution_components()

        # Conversation history (using HistoryList for automatic persistence)
        self._history: HistoryList = HistoryList(
            session_manager=self._session_manager,
            history_key=AgentRunActionKey(
                user_id=self._user_id,
                session_id=self._session_id,
                agent_id=self.agent_id,
            ),
            agent_name=self.agent_name,
        )

        # Queue for messages to be processed in the next execution cycle
        self.queued_messages: list[Message] = []

        # Register for cleanup
        cleanup_manager.register_agent(self)
        logger.info("Agent '%s' initialized (agent_id=%s, session_id=%s)", self.agent_name, self.agent_id, self._session_id)

    @property
    def history(self) -> HistoryList:
        """Get the conversation history."""
        return self._history

    @history.setter
    def history(self, value: list[Message] | HistoryList) -> None:
        """Set the conversation history with smart detection.

        This setter intercepts direct assignment to agent.history and:
        1. If value is already a HistoryList, use it directly
        2. Otherwise, use replace_all() which intelligently detects append vs replace

        Args:
            value: New history (list of messages or HistoryList)
        """
        if isinstance(value, HistoryList):
            self._history = value
        else:
            self._history.replace_all(value)

    def _init_session_state(
        self,
        *,
        provided_storage: GlobalStorage | None,
        proposed_agent_id: str | None,
    ) -> tuple[GlobalStorage, str]:
        """Initialize session state: global_storage and agent_id.

        This method consolidates all async session operations into a single call
        to avoid multiple run_sync overhead.

        Initialization logic:
        1. Initialize database models
        2. Register agent (which also creates/fetches session)
        3. Determine global_storage:
           - If user provides storage: use it directly (override mode)
           - Otherwise: use session.storage directly (restore mode)

        Args:
            provided_storage: User-provided GlobalStorage, or None
            proposed_agent_id: User-proposed agent ID, or None

        Returns:
            Tuple of (global_storage, agent_id)
        """

        async def _init() -> tuple[GlobalStorage, str]:
            # Step 1: Initialize database models
            await self._session_manager.setup_models()
            logger.debug("Session models initialized")

            # Step 2: Register agent - this also returns the session
            # No separate get_session call needed, avoiding overlay
            agent_id, session = await self._session_manager.register_agent(
                user_id=self._user_id,
                session_id=self._session_id,
                agent_id=proposed_agent_id,
                agent_name=self.config.name or "",
                is_root=self._is_root,
            )
            logger.debug("Agent registered with id='%s'", agent_id)

            # Step 3: Determine global_storage
            if provided_storage is not None:
                # Override mode: user explicitly provides storage
                storage = provided_storage
                logger.info(
                    "Using user-provided global_storage (override mode, %d keys)",
                    len(storage.to_dict()),
                )
            else:
                # Restore mode: use session.storage directly (already a GlobalStorage)
                storage = session.storage
                storage_size = len(storage.to_dict())
                if storage_size > 0:
                    logger.info(
                        "Restored global_storage from session '%s' (%d keys)",
                        self._session_id,
                        storage_size,
                    )
                else:
                    logger.debug("Using empty GlobalStorage from session")

            return storage, agent_id

        return syncify(_init, raise_sync_error=False)()

    def _setup_tracer(self) -> None:
        """Set up tracer in global_storage with conflict detection.

        Raises:
            ValueError: If both global_storage and config have conflicting tracers
        """
        existing_tracer = self.global_storage.get("tracer")
        if existing_tracer is not None and self.config.resolved_tracer is not None:
            raise ValueError(
                "Conflicting tracers: global_storage already has a tracer, "
                "but config.resolved_tracer is also provided. "
                "For nested agents, do not set resolved_tracer in config."
            )
        if self.config.resolved_tracer is not None:
            self.global_storage.set("tracer", self.config.resolved_tracer)
            logger.debug("Tracer set from config.resolved_tracer")

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
                "initialize the Agent using Agent(config=agent_config).",
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
            # Import here to avoid circular imports and optional dependency
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
            session_manager=self._session_manager,
            user_id=self._user_id,
            session_id=self._session_id,
            sandbox_manager=self.sandbox_manager,
        )

    def _initialize_sandbox(self) -> None:
        """Initialize sandbox."""
        sandbox_config = self.config.sandbox_config
        if sandbox_config is None:
            sandbox_config = {}
        sandbox_type = sandbox_config.get("type", "local")

        sandbox_manager_cls: type[BaseSandboxManager[Any]]

        if sandbox_type == "local":
            sandbox_manager_cls = LocalSandboxManager
        elif sandbox_type == "e2b":
            sandbox_manager_cls = E2BSandboxManager
        else:
            raise ValueError(f"Unsupported sandbox type: {sandbox_type}")
        sandbox_manager_kwargs = extract_dataclass_init_kwargs(sandbox_manager_cls, sandbox_config)
        self.sandbox_manager = sandbox_manager_cls(**sandbox_manager_kwargs)

        # Upload skill assets to sandbox
        upload_assets: list[tuple[str, str]] = []
        for i, skill in enumerate(self.config.skills):
            if skill.folder:
                local_folder = skill.folder
                skill.folder = os.path.join(self.sandbox_manager.work_dir, ".skills", os.path.basename(local_folder))
                self.config.skills[i] = skill
                upload_assets.append((local_folder, skill.folder))

        self.sandbox_manager.start_no_wait(
            session_manager=self._session_manager,
            user_id=self._user_id,
            session_id=self._session_id,
            sandbox_config=sandbox_config,
            upload_assets=upload_assets,
        )

        cleanup_manager.register_sandbox_manager(self.sandbox_manager)

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

    async def run_async(
        self,
        *,
        message: str | list[Message],
        history: list[dict[str, Any]] | list[Message] | None = None,
        context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
    ) -> str | tuple[str, dict[str, Any]]:
        """Run agent asynchronously with a message and return response.

        This is the async version of run(). Use this when you're already in an async context.

        The agent lock ensures only one execution per (session_id, agent_id) at a time.
        If the agent is already running, this method fails immediately with TimeoutError.

        Lock features:
        - Short TTL (default 30s) with automatic heartbeat renewal
        - Fast recovery: max 30s deadlock time even if release fails
        - No waiting: fails immediately if agent is busy

        Args:
            message: User message or list of messages
            history: Optional conversation history
            context: Optional context dict
            state: Optional state dict
            config: Optional config dict
            parent_agent_state: Optional parent agent state (for sub-agents)
            custom_llm_client_provider: Optional custom LLM client provider

        Returns:
            Agent response string or tuple of (response, state)

        Raises:
            TimeoutError: If agent is already running
        """
        # Generate run_id before acquiring lock
        from nexau.archs.session.id_generator import generate_run_id

        run_id = generate_run_id()

        async with self._session_manager.agent_lock.acquire(
            session_id=self._session_id,
            agent_id=self.agent_id,
            user_id=self._user_id,
            run_id=run_id,
        ):
            return await self._run_async_inner(
                message=message,
                history=history,
                context=context,
                state=state,
                config=config,
                parent_agent_state=parent_agent_state,
                custom_llm_client_provider=custom_llm_client_provider,
                run_id=run_id,
            )

    async def _run_async_inner(
        self,
        *,
        message: str | list[Message],
        history: list[dict[str, Any]] | list[Message] | None = None,
        context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
        run_id: str,
    ) -> str | tuple[str, dict[str, Any]]:
        """Inner implementation of run_async without lock handling.

        This method contains the actual agent execution logic.

        Args:
            run_id: Run ID for this execution (generated by run_async)
        """
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

        # Update agent metadata if needed
        agent_model = await self._session_manager.get_agent(
            user_id=self._user_id,
            session_id=self._session_id,
            agent_id=self.agent_id,
        )
        if agent_model and agent_model.agent_name != self.agent_name:
            await self._session_manager.update_agent_metadata(
                user_id=self._user_id,
                session_id=self._session_id,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
            )

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

            # Build system prompt
            system_prompt = self.prompt_builder.build_system_prompt(
                agent_config=self.config,
                tools=self.config.tools,
                sub_agents=self.config.sub_agents or {},
                runtime_context=merged_context,
                include_tool_instructions=not self.use_structured_tool_calls,
            )

            parent_run_id: str | None

            # Determine root_run_id and parent_run_id
            if parent_agent_state:
                root_run_id = parent_agent_state.root_run_id
                parent_run_id = parent_agent_state.run_id
            else:
                root_run_id = run_id
                parent_run_id = None

            # Update HistoryList context with new run IDs
            self._history.update_context(
                run_id=run_id,
                root_run_id=root_run_id,
                parent_run_id=parent_run_id,
            )

            # Load history from storage if this is the first run (history is empty)
            if not self.history:
                history_key = AgentRunActionKey(
                    user_id=self._user_id,
                    session_id=self._session_id,
                    agent_id=self.agent_id,
                )
                stored_messages = await self._session_manager.agent_run_action.load_messages(key=history_key)
                stored_non_system_messages = [msg for msg in stored_messages if msg.role != Role.SYSTEM]

                if stored_non_system_messages:
                    logger.info(f"📚 Restored {len(stored_non_system_messages)} messages from storage for agent '{self.config.name}'")
                    # Initialize history with system prompt + stored messages
                    # Use update_baseline=True since we're loading from storage
                    self._history.replace_all(
                        [Message(role=Role.SYSTEM, content=[TextBlock(text=system_prompt)])] + stored_non_system_messages,
                        update_baseline=True,
                    )
                else:
                    # Initialize with just system prompt
                    # Use update_baseline=True since this is initial state
                    self._history.replace_all(
                        [Message(role=Role.SYSTEM, content=[TextBlock(text=system_prompt)])],
                        update_baseline=True,
                    )
            else:
                # Update system prompt for existing history
                # Find and replace the system message
                # Use update_baseline=True since we're resetting to known state
                non_system_messages = [msg for msg in self.history if msg.role != Role.SYSTEM]
                self._history.replace_all(
                    [Message(role=Role.SYSTEM, content=[TextBlock(text=system_prompt)])] + non_system_messages,
                    update_baseline=True,
                )

            # Add caller-provided history
            if history:
                # Support both Message objects and legacy dict format
                if history and isinstance(history[0], Message):
                    self.history.extend(cast(list[Message], history))
                else:
                    self.history.extend(messages_from_legacy_openai_chat(cast(list[dict[str, Any]], history)))

            # Add user message (HistoryList will auto-persist)
            if isinstance(message, str):
                user_message = Message.user(message)
                self.history.append(user_message)
            else:
                self.history.extend(message)

            # Create the AgentState instance
            agent_state = AgentState(
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                run_id=run_id,
                root_run_id=root_run_id,
                context=ctx,
                global_storage=self.global_storage,
                parent_agent_state=parent_agent_state,
                executor=self.executor,
            )

            # Execute with or without tracing
            try:
                if tracer:
                    response = await self._run_with_tracing(
                        tracer=tracer,
                        span_type=span_type,
                        message_text_for_logs=message_text_for_logs,
                        agent_state=agent_state,
                        merged_context=merged_context,
                        runtime_client=runtime_client,
                        custom_llm_client_provider=custom_llm_client_provider,
                    )
                else:
                    response = await self._run_inner(
                        agent_state,
                        merged_context,
                        runtime_client=runtime_client,
                        custom_llm_client_provider=custom_llm_client_provider,
                    )

                # Persist context and storage to session
                await self._persist_session_state(ctx.context)

                # Pause sandbox after agent execution if persistence is enabled
                if self.config.sandbox_config and self.config.sandbox_config.get("persist_sandbox", True):
                    self.sandbox_manager.pause_no_wait()
                else:
                    self.sandbox_manager.stop()

                logger.info(f"✅ Agent '{self.config.name}' completed execution")
                return response

            except Exception as e:
                logger.error(f"❌ Agent '{self.config.name}' encountered error: {e}")
                raise

    def run(
        self,
        *,
        message: str | list[Message],
        history: list[dict[str, Any]] | list[Message] | None = None,
        context: dict[str, Any] | None = None,
        state: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
    ) -> str | tuple[str, dict[str, Any]]:
        """Run agent with a message and return response.

        This is the sync version that wraps run_async().

        Args:
            message: User message or list of messages
            history: Optional conversation history
            context: Optional context dict
            state: Optional state dict
            config: Optional config dict
            parent_agent_state: Optional parent agent state (for sub-agents)
            custom_llm_client_provider: Optional custom LLM client provider

        Returns:
            Agent response string or tuple of (response, state)

        Raises:
            TimeoutError: If agent is already running
        """

        async def _run() -> str | tuple[str, dict[str, Any]]:
            return await self.run_async(
                message=message,
                history=history,
                context=context,
                state=state,
                config=config,
                parent_agent_state=parent_agent_state,
                custom_llm_client_provider=custom_llm_client_provider,
            )

        return syncify(_run, raise_sync_error=False)()

    async def _run_with_tracing(
        self,
        tracer: BaseTracer,
        span_type: SpanType,
        message_text_for_logs: str,
        agent_state: AgentState,
        merged_context: dict[str, Any],
        runtime_client: Any,
        custom_llm_client_provider: Callable[[str], Any] | None,
    ) -> str:
        """Execute agent with tracing enabled."""
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
                response = await self._run_inner(
                    agent_state,
                    merged_context,
                    runtime_client=runtime_client,
                    custom_llm_client_provider=custom_llm_client_provider,
                )
                trace_ctx.set_outputs({"response": response})
                return response
            except Exception:
                raise

    async def _run_inner(
        self,
        agent_state: AgentState,
        merged_context: dict[str, Any],
        *,
        runtime_client: Any,
        custom_llm_client_provider: Callable[[str], Any] | None,
    ) -> str:
        """Inner execution logic without tracing wrapper."""
        try:
            # Execute using the executor in a thread pool to avoid blocking the event loop
            # This allows streaming events to be processed in real-time
            # Using asyncer.asyncify for cleaner async/sync conversion
            response, updated_messages = await asyncify(self.executor.execute)(
                self.history,
                agent_state,
                runtime_client=runtime_client,
                custom_llm_client_provider=custom_llm_client_provider,
            )
            # HistoryList will automatically persist any changes made by executor
            self.history = updated_messages

            # Flush pending messages to persistence
            self.history.flush()

            return response

        except Exception as e:
            if self.config.error_handler:
                error_response = self.config.error_handler(
                    e,
                    self,
                    merged_context,
                )
                assistant_error_message = Message.assistant(error_response)
                # HistoryList will automatically persist this message
                self.history.append(assistant_error_message)

                # Flush pending messages to persistence
                self.history.flush()

                return error_response
            else:
                assistant_error = Message.assistant(f"Error: {str(e)}")
                self.history.append(assistant_error)

                # Flush pending messages to persistence
                self.history.flush()

                raise

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the agent."""
        self.config.tools.append(tool)
        self.tool_registry[tool.name] = tool
        self.executor.add_tool(tool)

    async def _persist_session_state(self, context: dict[str, Any]) -> None:
        """Persist context and storage to session.

        This method saves the current agent context and global_storage to the SessionModel
        for persistence across requests. Non-serializable objects (like tracer, skill_registry)
        are automatically filtered out by sanitize_for_serialization in the storage layer.

        Args:
            context: The current agent context to persist
        """
        try:
            # Persist both context and storage in a single operation
            await self._session_manager.update_session_state(
                user_id=self._user_id,
                session_id=self._session_id,
                context=context,
                storage=self.global_storage,
            )
            logger.debug(
                "Persisted session state for session '%s', agent '%s'",
                self._session_id,
                self.agent_id,
            )
        except Exception as e:
            logger.warning(f"Failed to persist session state: {e}")

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
