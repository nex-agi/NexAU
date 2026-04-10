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

"""Main execution orchestrator for agents.

RFC-0006: 中性 Structured Tool Definitions 在执行器中的持有与分发

执行器在 structured 模式下只缓存 neutral structured definitions，真正的
OpenAI / Anthropic / Gemini provider schema 由 LLMCaller 在请求边界延迟适配。
"""

import asyncio
import inspect
import json
import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

from nexau.archs.llm.llm_aggregators.events import RetryEvent
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.hooks import (
    AfterAgentHookInput,
    AfterModelHook,
    AfterModelHookInput,
    AfterToolHook,
    BeforeAgentHookInput,
    BeforeModelHook,
    BeforeModelHookInput,
    BeforeToolHook,
    FunctionMiddleware,
    Middleware,
    MiddlewareManager,
)
from nexau.archs.main_sub.execution.llm_caller import LLMCaller
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.parse_structures import (
    ParsedResponse,
    ToolCall,
)
from nexau.archs.main_sub.execution.response_parser import ResponseParser
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager
from nexau.archs.main_sub.execution.tool_executor import ToolExecutionResult, ToolExecutor
from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.main_sub.token_trace_session import TokenTraceContextOverflowError, TokenTraceSession
from nexau.archs.main_sub.tool_call_modes import (
    STRUCTURED_TOOL_CALL_MODES,
    normalize_tool_call_mode,
)
from nexau.archs.main_sub.utils.token_counter import TokenCounter
from nexau.archs.tool.tool import (
    StructuredToolDefinition,
    Tool,
)
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, coerce_tool_result_content

if TYPE_CHECKING:
    from nexau.archs.sandbox.base_sandbox import BaseSandbox
    from nexau.archs.session import SessionManager

logger = logging.getLogger(__name__)


class _IterationOutcome(Enum):
    """Signal returned by _execute_iteration_async to control the main loop.

    CONTINUE — advance to the next iteration.
    BREAK    — exit the while loop (final_response / force_stop_reason already set on state).
    """

    CONTINUE = auto()
    BREAK = auto()


@dataclass
class _AsyncIterationState:
    """Mutable context shared across helpers during one execute_async() invocation."""

    messages: list[Message]
    final_response: str
    force_stop_reason: AgentStopReason
    iteration: int
    agent_state: "AgentState"
    token_trace_session: TokenTraceSession | None
    framework_context: FrameworkContext
    runtime_client: object | None
    custom_llm_client_provider: Callable[[str], object] | None
    origin_history: list[Message] | list[dict[str, object]]


class Executor:
    """Orchestrates execution of agent tasks with parallel processing support."""

    def __init__(
        self,
        agent_name: str,
        agent_id: str,
        tool_registry: ToolRegistry,
        sub_agents: dict[str, AgentConfig],
        stop_tools: set[str],
        openai_client: Any,
        llm_config: LLMConfig,
        async_openai_client: Any | None = None,
        max_iterations: int = 100,
        max_context_tokens: int = 128000,
        max_running_subagents: int = 5,
        retry_attempts: int = 5,
        retry_backoff_max_seconds: int = 30,
        token_counter: TokenCounter | None = None,
        after_model_hooks: list[AfterModelHook] | None = None,
        before_model_hooks: list[BeforeModelHook] | None = None,
        after_tool_hooks: list[AfterToolHook] | None = None,
        before_tool_hooks: list[BeforeToolHook] | None = None,
        middlewares: list[Middleware] | None = None,
        global_storage: Any = None,
        tool_call_mode: str = "structured",
        team_mode: bool = False,
        structured_tools: Sequence[StructuredToolDefinition] | None = None,
        session_manager: "SessionManager | None" = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize executor.

        Args:
            agent_name: Name of the agent
            agent_id: ID of the agent
            tool_registry: ToolRegistry containing available tools
            sub_agents: Dictionary of sub-agent configs
            stop_tools: Set of tool names that trigger execution stop
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            max_iterations: Maximum iterations per execution
            max_context_tokens: Maximum context token limit
            max_running_subagents: Maximum concurrent sub-agents
            retry_attempts: int of API retry attempts
            token_counter: Optional token counter instance
            before_model_hooks: Optional list of hooks called before parsing LLM response
            after_model_hooks: Optional list of hooks called after parsing LLM response
            before_tool_hooks: Optional list of hooks called before tool execution
            after_tool_hooks: Optional list of hooks called after tool execution
            middlewares: Optional list of middleware objects applied to all phases
            tool_call_mode: Preferred tool call format ('xml' or 'structured')
            team_mode: If True, executor runs in "forever run" mode for team agents
            structured_tools: Vendor-neutral structured tool definitions
            session_manager: Optional SessionManager for unified data access
            user_id: Optional user ID for persistence
            session_id: Optional session ID for persistence
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.max_running_subagents = max_running_subagents

        # Initialize components
        self.middleware_manager = self._build_middleware_manager(
            middlewares or [],
            before_model_hooks or [],
            after_model_hooks or [],
            after_tool_hooks or [],
            before_tool_hooks or [],
        )
        self._event_emitter: Callable[[Any], None] | None = None
        self._wire_middleware_llm_runtime(llm_config, openai_client, session_id=session_id)
        self._wire_middleware_event_emitters()
        self._tool_registry = tool_registry
        self._tool_registry_lock = threading.RLock()
        self.tool_executor = ToolExecutor(
            tool_registry=tool_registry,
            stop_tools=stop_tools,
            middleware_manager=self.middleware_manager,
        )

        self.subagent_manager = SubAgentManager(
            agent_name,
            sub_agents,
            global_storage,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        self.response_parser = ResponseParser()
        self.llm_caller = LLMCaller(
            openai_client,
            llm_config,
            retry_attempts,
            retry_backoff_max_seconds=retry_backoff_max_seconds,
            on_retry=self._build_retry_callback(llm_config),
            middleware_manager=self.middleware_manager,
            global_storage=global_storage,
            session_id=session_id,
            async_openai_client=async_openai_client,
        )

        # Execution parameters
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.global_storage = global_storage

        # Token counting
        self.token_counter = token_counter or TokenCounter(model=llm_config.model)

        # Tool call behavior
        self.tool_call_mode = normalize_tool_call_mode(tool_call_mode)
        self.use_structured_tool_calls = self.tool_call_mode in STRUCTURED_TOOL_CALL_MODES

        # 1. RFC-0006: 执行器内部只保存 neutral structured definitions，避免主状态提前 vendor 化。
        if structured_tools is not None:
            self.structured_tool_definitions = deepcopy(list(structured_tools))
        elif self.use_structured_tool_calls:
            self.structured_tool_definitions = [
                tool.to_structured_definition(
                    description=self._structured_tool_description(tool),
                )
                for tool in self._tool_registry.compute_eager_tools()
            ]
        else:
            self.structured_tool_definitions = []

        if self.use_structured_tool_calls and not self.structured_tool_definitions:
            logger.warning(
                f"⚠️ {self.tool_call_mode.capitalize()} tool call mode enabled but no structured tool definitions were provided.",
            )

        # Process tracking for parallel execution
        self._running_executors: dict[str, ThreadPoolExecutor] = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.stop_signal = False

        # RFC-0001: 追踪 execute() 主循环是否正在运行
        # _execution_done: set 表示 execute() 未运行或已结束，clear 表示正在运行
        self._execution_done = threading.Event()
        self._execution_done.set()  # 初始状态：未运行

        # Message queue for dynamic message enqueueing during execution
        self.queued_messages: list[Message] = []

        # RFC-0002: Team mode — "forever run" loop that waits for messages when idle
        self.team_mode = team_mode
        self._message_available = threading.Event()
        self._is_idle = False  # True when waiting for messages in team_mode
        self._is_waiting_for_user = False  # True when idle due to ask_user stop tool
        self._last_stop_tool_name: str | None = None  # 最近触发 stop 的工具名

    def _wire_middleware_event_emitters(self) -> None:
        """Wire internal middleware emitters to the unified event callback when available."""
        if not self.middleware_manager:
            return

        event_emitter: Callable[[Any], None] | None = None
        for middleware in self.middleware_manager.middlewares:
            handler = middleware.get_event_handler()
            if handler is not None:
                event_emitter = handler
                break

        if event_emitter is None:
            return

        self._event_emitter = event_emitter

        for middleware in self.middleware_manager.middlewares:
            if not middleware.supports_set_event_emitter:
                continue
            try:
                middleware.set_event_emitter(event_emitter)
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to wire event emitter for middleware %s: %s",
                    middleware.__class__.__name__,
                    exc,
                )

    def _build_retry_callback(self, llm_config: LLMConfig) -> Callable[[int, int, float, str], None] | None:
        """Build retry event callback when middleware exposes an event emitter."""
        event_emitter = self._event_emitter
        if event_emitter is None:
            return None

        def emit_retry_event(attempt: int, max_attempts: int, backoff_seconds: float, error_message: str) -> None:
            retry_event = RetryEvent(
                api_type=llm_config.api_type,
                attempt=attempt,
                max_attempts=max_attempts,
                backoff_seconds=backoff_seconds,
                error_message=error_message,
            )
            event_emitter(retry_event)

        return emit_retry_event

    def _wire_middleware_llm_runtime(
        self,
        llm_config: LLMConfig,
        openai_client: Any,
        *,
        session_id: str | None = None,
        global_storage: Any | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        """Inject base LLM runtime into middleware that needs inherited model settings."""
        if not self.middleware_manager:
            return

        for middleware in self.middleware_manager.middlewares:
            if not middleware.supports_set_llm_runtime:
                continue
            try:
                middleware.set_llm_runtime(
                    llm_config,
                    openai_client,
                    session_id=session_id,
                    global_storage=global_storage,
                    max_context_tokens=max_context_tokens,
                )
            except TypeError:
                # Backward compatibility: middleware without max_context_tokens/global_storage/session_id parameter
                try:
                    middleware.set_llm_runtime(llm_config, openai_client, session_id=session_id, global_storage=global_storage)
                except TypeError:
                    try:
                        middleware.set_llm_runtime(llm_config, openai_client, session_id=session_id)
                    except TypeError:
                        middleware.set_llm_runtime(llm_config, openai_client)
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to wire LLM runtime for middleware %s: %s",
                    middleware.__class__.__name__,
                    exc,
                )

    @property
    def shutdown_event(self) -> threading.Event:
        """Public accessor for the shutdown event."""
        return self._shutdown_event

    @property
    def has_running_executors(self) -> bool:
        """Check if there are any running thread pool executors."""
        return bool(self._running_executors)

    @property
    def is_executing(self) -> bool:
        """Check if execute() main loop is currently running.

        RFC-0001: 用于 interrupt() 等待主循环退出。
        """
        return not self._execution_done.is_set()

    @property
    def is_idle(self) -> bool:
        """Check if executor is idle (waiting for messages in team_mode).

        RFC-0002: 用于全员空闲检测。
        """
        return self._is_idle

    @property
    def is_waiting_for_user(self) -> bool:
        """Check if executor is idle because it's waiting for user response (ask_user).

        RFC-0002: 区分 ask_user 导致的 idle 与普通 idle，
        避免 watchdog 误报全员空闲。
        """
        return self._is_waiting_for_user

    def _mark_waiting_for_user(self) -> None:
        """Set waiting-for-user flag if the last stop tool was ask_user."""
        if self._last_stop_tool_name == "ask_user":
            self._is_waiting_for_user = True

    @property
    def execution_done_event(self) -> threading.Event:
        """Public accessor for the _execution_done event.

        RFC-0001: interrupt() 通过 wait() 等待 execute() 退出。
        Event is set when execute() is NOT running, cleared when running.
        """
        return self._execution_done

    def _snapshot_structured_tool_definitions(self) -> list[StructuredToolDefinition]:
        """Return a synchronized snapshot of neutral structured tool definitions.

        RFC-0006: neutral structured definitions 与 ToolRegistry 运行时注入同步

        ToolSearch 会把 deferred tools 注入到 ToolRegistry；这里在每轮模型调用前
        将新增 eager tools / sub-agent 代理补齐到执行器缓存中，确保下一轮 LLM
        可以立即看到新可用的 neutral structured definitions。
        """

        if not self.use_structured_tool_calls:
            return []

        with self._tool_registry_lock:
            definition_names = {definition["name"] for definition in self.structured_tool_definitions}

            # 1. 同步 ToolRegistry 中当前所有 eager tools（含 ToolSearch 注入结果）。
            for tool in self._tool_registry.compute_eager_tools():
                if tool.name in definition_names:
                    continue
                self.structured_tool_definitions.append(
                    tool.to_structured_definition(
                        description=self._structured_tool_description(tool),
                    ),
                )
                definition_names.add(tool.name)

            return deepcopy(self.structured_tool_definitions)

    @property
    def structured_tool_payload(self) -> list[StructuredToolDefinition]:
        """Return a snapshot of neutral structured tool definitions."""
        return self._snapshot_structured_tool_definitions()

    def update_structured_tools(self, definitions: list[StructuredToolDefinition]) -> None:
        """Replace the structured tool definitions.

        P1 async/sync 技术债修复: 支持 async MCP 初始化后更新工具定义

        由 Agent.create() 在异步 MCP 工具初始化完成后调用，
        将新发现的 MCP 工具定义注入到 executor 的 structured tool payload 中。
        """
        with self._tool_registry_lock:
            self.structured_tool_definitions = deepcopy(list(definitions))

    def _wait_for_messages(self) -> bool:
        """Enter idle wait until new messages arrive or stop signal is received.

        RFC-0002: team_mode 下的 idle 等待辅助方法

        Returns:
            True if new messages arrived, False if stop signal received.
        """
        self._is_idle = True
        while not self.stop_signal and len(self.queued_messages) == 0:
            self._message_available.clear()
            self._message_available.wait(timeout=30)
        self._is_idle = False
        self._is_waiting_for_user = False
        return not self.stop_signal

    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be processed during execution.

        Args:
            message: Message dictionary with 'role' and 'content' keys
        """
        # Keep public API stable (dict input), but immediately normalize to UMP Message internally.
        role = Role(message.get("role", "user"))
        content = message.get("content", "")
        self.queued_messages.append(Message(role=role, content=[TextBlock(text=content)]))
        # RFC-0002: 唤醒 team_mode 下等待消息的主循环
        self._message_available.set()
        logger.info(
            f"📝 Message enqueued during execution: {message.get('role', 'unknown')} - {message.get('content', '')[:50]}...",
        )

    def execute(
        self,
        history: list[Message] | list[dict[str, Any]],
        agent_state: "AgentState",
        *,
        runtime_client: Any | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
    ) -> tuple[str, list[Message]]:
        """Execute agent task with full orchestration.

        Args:
            history: Complete conversation history including system prompt and user message
            agent_state: AgentState containing agent context and global storage

        Returns:
            Tuple of (agent_response, updated_messages_history)
        """
        # Reset the stop signal
        self.stop_signal = False
        self._shutdown_event.clear()

        # RFC-0006: 构建 FrameworkContext，供工具函数通过 ctx 参数访问框架服务
        framework_context = FrameworkContext(
            agent_name=self.agent_name,
            agent_id=self.agent_id,
            run_id=agent_state.run_id,
            root_run_id=agent_state.root_run_id,
            _tool_registry=self._tool_registry,
            _shutdown_event=self._shutdown_event,
        )

        # RFC-0001: 标记 execute() 正在运行
        self._execution_done.clear()

        messages: list[Message] = []

        # Keep a reference to the original history (HistoryList) so we can
        # sync the executor's local ``messages`` back before blocking waits.
        # This ensures that if the agent is cancelled (e.g. browser refresh)
        # while waiting, the HistoryList already contains all messages and
        # the ``finally`` block in ``_run_inner`` can flush them to storage.
        _origin_history = history

        force_stop_reason = AgentStopReason.SUCCESS

        # RFC-0009: 从 AgentState 获取 token trace session 引用
        token_trace_session = agent_state.token_trace_session

        try:
            # Use history directly as the single source of truth
            if history and isinstance(history[0], dict):
                messages = messages_from_legacy_openai_chat(cast(list[dict[str, Any]], history))
            else:
                messages = cast(list[Message], history).copy()

            # RFC-0009: 复用外部传入的 session 以支持跨 run 延续 token buffer
            tools_payload = None
            if self.use_structured_tool_calls:
                with self._tool_registry_lock:
                    # Snapshot current structured tool definitions under lock to avoid concurrent mutation
                    tools_payload = deepcopy(self.structured_tool_payload)

            if self.llm_config.api_type == "generate_with_token":
                if token_trace_session is None:
                    token_trace_session = TokenTraceSession(
                        self.llm_config,
                        max_context_tokens=self.max_context_tokens,
                    )
                token_trace_session.initialize_from_messages(
                    messages,
                    tools=cast(list[dict[str, Any]] | None, tools_payload),
                )
                token_trace_session.sync_external_messages(messages)

            if self.middleware_manager:
                before_agent_hook_input = BeforeAgentHookInput(
                    agent_state=agent_state,
                    messages=messages,
                )
                try:
                    messages = self.middleware_manager.run_before_agent(
                        before_agent_hook_input,
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Before-agent middleware execution failed: {e}")

            # Loop until no more tool calls or sub-agent calls are made
            iteration = 0
            final_response = ""

            logger.info(
                f"🔄 Starting iterative execution loop for agent '{self.agent_name}'",
            )

            while iteration < self.max_iterations:
                logger.info(
                    f"🔄 Iteration {iteration + 1}/{self.max_iterations} for agent '{self.agent_name}'",
                )

                logger.info(
                    f"Agent name {self.agent_name} Current stop_signal: {self.stop_signal}",
                )
                if self.stop_signal:
                    logger.info(
                        "❗️ Stop signal received, stopping execution",
                    )
                    stop_response = "Stop signal received."
                    stop_response, messages = self._apply_after_agent_hooks(
                        agent_state=agent_state,
                        messages=messages,
                        final_response=stop_response,
                        stop_reason=AgentStopReason.USER_INTERRUPTED,
                    )
                    return stop_response, messages

                # Process any queued messages
                if self.queued_messages:
                    logger.info(
                        f"📝 Processing {len(self.queued_messages)} queued messages",
                    )
                    messages.extend(self.queued_messages)
                    self.queued_messages = []

                # RFC-0002: team_mode 下，若无用户内容（仅 system prompt），
                # 跳过 LLM 调用，直接进入 idle 等待，避免浪费 token。
                if self.team_mode and not any(m.role != Role.SYSTEM for m in messages):
                    # Sync messages back to HistoryList before blocking wait
                    if isinstance(_origin_history, HistoryList):
                        _origin_history.replace_all(messages)
                    if not self._wait_for_messages():
                        break
                    iteration += 1
                    continue

                # RFC-0002: team_mode 下，若最后一条非 system 消息是 assistant 消息，
                # 跳过 LLM 调用，进入 idle 等待。恢复 session 时历史可能以 assistant
                # 消息结尾，此时调用 LLM 会导致空响应（多数 LLM 不接受 assistant 结尾）。
                if self.team_mode:
                    last_non_system: Message | None = None
                    for m in reversed(messages):
                        if m.role != Role.SYSTEM:
                            last_non_system = m
                            break
                    if last_non_system is not None and last_non_system.role == Role.ASSISTANT:
                        logger.info(
                            f"⏸️ team_mode: last message is assistant, skipping LLM call for '{self.agent_name}'",
                        )
                        # Sync messages back to HistoryList before blocking wait
                        if isinstance(_origin_history, HistoryList):
                            _origin_history.replace_all(messages)
                        if not self._wait_for_messages():
                            break
                        iteration += 1
                        continue

                before_model_hook_input = BeforeModelHookInput(
                    agent_state=agent_state,
                    max_iterations=self.max_iterations,
                    current_iteration=iteration,
                    messages=messages,
                )

                if self.middleware_manager:
                    try:
                        messages = self.middleware_manager.run_before_model(
                            before_model_hook_input,
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Before-model middleware execution failed: {e}")

                tools_payload = None
                if self.use_structured_tool_calls:
                    # 1. RFC-0006: 每轮调用前同步 ToolRegistry 注入结果，再快照 neutral definitions。
                    tools_payload = self._snapshot_structured_tool_definitions()

                # Count current prompt tokens (including tool definitions if present)

                current_prompt_tokens = self.token_counter.count_tokens(
                    messages,
                    tools=tools_payload,
                )

                force_stop_reason = AgentStopReason.SUCCESS
                if current_prompt_tokens > self.max_context_tokens:
                    logger.warning(
                        "⚠️ Prompt tokens (%d) exceed max_context_tokens (%d). Continuing model call.",
                        current_prompt_tokens,
                        self.max_context_tokens,
                    )

                # Calculate max_tokens dynamically based on available budget
                available_tokens = self.max_context_tokens - current_prompt_tokens

                # Get desired max_tokens from LLM config or use reasonable default
                desired_max_tokens = 16384  # Default value
                calculated_max_tokens = min(
                    desired_max_tokens,
                    available_tokens,
                )
                calculated_max_tokens = max(1, calculated_max_tokens)

                # Ensure we have at least some tokens for response
                if available_tokens < 50:
                    logger.warning(
                        "⚠️ Available response budget is %d (<50). Continuing model call.",
                        available_tokens,
                    )

                if iteration == self.max_iterations - 1:
                    logger.error(
                        "❌ Maximum iteration limit reached. Stopping execution.",
                    )
                    final_response += "\\n\\n[Error: Maximum iteration limit reached.]"
                    force_stop_reason = AgentStopReason.MAX_ITERATIONS_REACHED

                logger.info(
                    f"🔢 Token usage: prompt={current_prompt_tokens}, max_tokens={calculated_max_tokens}, available={available_tokens}",
                )

                # Call LLM to get response
                logger.info(
                    f"🧠 Calling LLM for agent '{self.agent_name}' with {calculated_max_tokens} max tokens...",
                )
                model_response = self.llm_caller.call_llm(
                    messages,
                    openai_client=runtime_client,
                    force_stop_reason=force_stop_reason,
                    agent_state=agent_state,
                    tool_call_mode=self.tool_call_mode,
                    tools=tools_payload,
                    shutdown_event=self._shutdown_event,
                    token_trace_session=token_trace_session,
                )
                if model_response is None:
                    break

                assistant_content = model_response.content or ""

                # Store this as the latest response (potential final response)
                final_response = assistant_content

                # Parse response to check for actions
                parsed_response = self.response_parser.parse_response(
                    model_response,
                )

                # Add the assistant's original response to conversation
                assistant_message = model_response.to_ump_message()
                messages.append(assistant_message)
                if token_trace_session is not None:
                    token_trace_session.append_model_response(
                        output_token_ids=model_response.output_token_ids,
                        fallback_messages=[assistant_message],
                    )

                # Process tool calls and sub-agent calls
                logger.info(
                    f"⚙️ Processing tool/sub-agent calls for agent '{self.agent_name}'...",
                )
                after_model_hook_input = AfterModelHookInput(
                    agent_state=agent_state,
                    max_iterations=self.max_iterations,
                    current_iteration=iteration,
                    original_response=assistant_content,
                    parsed_response=parsed_response,
                    messages=messages,
                    model_response=model_response,
                )

                (
                    processed_response,
                    should_stop,
                    stop_tool_result,
                    updated_messages,
                    execution_feedbacks,
                ) = self._process_xml_calls(
                    after_model_hook_input,
                    custom_llm_client_provider=custom_llm_client_provider,
                    framework_context=framework_context,
                )

                # Update messages with any modifications from hooks
                messages = updated_messages

                processed_parsed_response = after_model_hook_input.parsed_response

                # Extract just the tool results from processed_response
                openai_tool_mode = bool(
                    processed_parsed_response
                    and processed_parsed_response.model_response
                    and processed_parsed_response.model_response.tool_calls
                )

                if openai_tool_mode:
                    tool_result_messages: list[Message] = []
                    for feedback in execution_feedbacks:
                        call_obj = feedback.get("call")
                        content = feedback.get("content") or ""
                        output = feedback.get("output")

                        if isinstance(call_obj, ToolCall):
                            call_id = call_obj.tool_call_id
                        else:
                            call_id = None

                        if not call_id:
                            continue

                        tool_result_block = ToolResultBlock(
                            tool_use_id=str(call_id),
                            content=coerce_tool_result_content(
                                feedback.get("llm_tool_output", output if output is not None else content),
                                fallback_text=None,
                            ),
                            is_error=bool(feedback.get("is_error")),
                        )

                        # micro-compact: 设置 created_at 时间戳
                        from datetime import UTC, datetime

                        tool_result_message = Message(
                            role=Role.TOOL,
                            content=[tool_result_block],
                            created_at=datetime.now(UTC),
                        )
                        messages.append(tool_result_message)
                        tool_result_messages.append(tool_result_message)

                    tool_results = ""
                    if token_trace_session is not None and tool_result_messages:
                        token_trace_session.append_messages(tool_result_messages, mask_value=0)
                else:
                    tool_results = processed_response.replace(
                        assistant_content,
                        "",
                        1,
                    ).strip()

                if tool_results:
                    # micro-compact: 设置 created_at 时间戳
                    from datetime import UTC, datetime

                    from nexau.core.messages import TextBlock

                    tool_result_feedback_message = Message(
                        role=Role.USER,
                        content=[TextBlock(text=f"Tool execution results:\n{tool_results}")],
                        created_at=datetime.now(UTC),
                    )
                    messages.append(tool_result_feedback_message)
                    if token_trace_session is not None:
                        token_trace_session.append_messages([tool_result_feedback_message], mask_value=0)

                # Check if a stop tool was executed
                if should_stop and len(self.queued_messages) == 0:
                    # RFC-0002: team_mode 下，仅「无更多 tool call」时继续等待，
                    # stop_tool（如 finish_team）显式调用时必须退出。
                    if self.team_mode:
                        # team_mode 下只有框架级 stop tool `finish_team` 会真正结束执行；
                        # 其他 stop_tools（如 ask_user / complete_task / 自定义 stop tool）
                        # 只用于结束当前这一轮工具调用，然后继续进入等待态。
                        if stop_tool_result is not None and self._last_stop_tool_name == "finish_team":
                            logger.info(
                                "🛑 Stop tool detected in team_mode, exiting executor loop",
                            )
                            force_stop_reason = AgentStopReason.STOP_TOOL_TRIGGERED
                            final_response = stop_tool_result
                            break
                        # RFC-0002: team_mode 下无限等待新消息，不设超时。
                        # Leader 需要等待 teammate 完成工作（可能远超 120s），
                        # watchdog 负责检测全员空闲并唤醒 leader。
                        # 标记 ask_user 导致的 idle，避免 watchdog 误报全员空闲
                        self._mark_waiting_for_user()
                        # Sync messages back to HistoryList before blocking wait
                        if isinstance(_origin_history, HistoryList):
                            _origin_history.replace_all(messages)
                        if not self._wait_for_messages():
                            force_stop_reason = AgentStopReason.NO_MORE_TOOL_CALLS
                            final_response = processed_response
                            break
                        iteration += 1
                        continue
                    # Return the stop tool result directly, formatted as JSON if it's not a string
                    if stop_tool_result is not None:
                        logger.info(
                            "🛑 Stop tool detected, returning stop tool result as final response",
                        )
                        force_stop_reason = AgentStopReason.STOP_TOOL_TRIGGERED
                        final_response = stop_tool_result
                        break
                    else:
                        logger.info("🛑 No more tool calls, stop.")
                        force_stop_reason = AgentStopReason.NO_MORE_TOOL_CALLS
                        # Fallback to the processed response if no specific result
                        final_response = processed_response
                        break

                iteration += 1

            # Add note if max iterations reached
            if iteration >= self.max_iterations:
                force_stop_reason = AgentStopReason.MAX_ITERATIONS_REACHED
                final_response += "\\n\\n[Note: Maximum iteration limit reached]"

            final_response, messages = self._apply_after_agent_hooks(
                agent_state=agent_state,
                messages=messages,
                final_response=final_response,
                stop_reason=force_stop_reason,
            )

            logger.info(
                f"🔄 Force stop reason: {force_stop_reason.name}",
            )
            logger.info(
                f"🔄 Final response for agent '{self.agent_name}': {final_response[:100]}",
            )
            logger.debug(
                "🔍 [HISTORY-DEBUG] executor returning: %d messages, roles=%s",
                len(messages),
                [m.role.value for m in messages],
            )
            self._store_token_trace(token_trace_session)
            return final_response, messages

        except TokenTraceContextOverflowError as e:
            # token trace session 不支持上下文折叠，超限时直接终止
            force_stop_reason = AgentStopReason.CONTEXT_TOKEN_LIMIT
            final_response = f"Error: {e}"
            logger.error("❌ TokenTraceSession context overflow: %s", e)

            final_response, messages = self._apply_after_agent_hooks(
                agent_state=agent_state,
                messages=messages,
                final_response=final_response,
                stop_reason=force_stop_reason,
            )
            self._store_token_trace(token_trace_session)
            return final_response, messages

        except Exception as e:
            force_stop_reason = AgentStopReason.ERROR_OCCURRED
            final_response = f"Error: {str(e)}"

            final_response, messages = self._apply_after_agent_hooks(
                agent_state=agent_state,
                messages=messages,
                final_response=final_response,
                stop_reason=force_stop_reason,
            )

            logger.error(
                f"🔄 Force stop reason: {force_stop_reason.name}",
            )
            logger.error(
                f"🔄 Final response for agent '{self.agent_name}': {final_response}",
            )
            logger.error(
                f"❌ Error in agent execution: {e}",
            )
            self._store_token_trace(token_trace_session)
            # Re-raise with more context
            raise RuntimeError(f"Error in agent execution: {e}") from e

        finally:
            # Sync intermediate iteration messages back to HistoryList so that
            # _run_inner's error/finally flush can persist them (fixes #390)
            if isinstance(_origin_history, HistoryList):
                _origin_history.replace_all(messages)
            self._store_token_trace(token_trace_session)
            # RFC-0009: 重置同步计数以匹配可能被压缩的 messages，确保下次 run 正确同步新消息
            if token_trace_session is not None:
                token_trace_session.synced_message_count = len(messages)
            # RFC-0001: 标记 execute() 已结束，唤醒 interrupt() 的等待
            self._execution_done.set()

    async def execute_async(
        self,
        history: list[Message] | list[dict[str, Any]],
        agent_state: "AgentState",
        *,
        runtime_client: Any | None = None,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
    ) -> tuple[str, list[Message]]:
        """Fully async execution — runs on the main event loop.

        async/sync 技术债修复: 理想状态的 async 执行器

        - LLM 调用: await call_llm_async() (Gemini 走 httpx.AsyncClient，其余走 to_thread 桥接)
        - Tool 执行: asyncio.gather + tool.execute_async() (async tool 直接 await，sync tool to_thread)
        - Middleware hooks: asyncio.to_thread (用户 hooks 是 sync API)
        - Team mode 等待: asyncio.to_thread(_wait_for_messages)
        - History flush: 直接 create_task 或 await flush_async()

        sync execute() 保留给向后兼容的 sync 调用方和测试。
        """

        # 1. 初始化执行状态
        self.stop_signal = False
        self._shutdown_event.clear()
        self._execution_done.clear()

        state = _AsyncIterationState(
            messages=[],
            final_response="",
            force_stop_reason=AgentStopReason.SUCCESS,
            iteration=0,
            agent_state=agent_state,
            token_trace_session=agent_state.token_trace_session,
            framework_context=FrameworkContext(
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                run_id=agent_state.run_id,
                root_run_id=agent_state.root_run_id,
                _tool_registry=self._tool_registry,
                _shutdown_event=self._shutdown_event,
            ),
            runtime_client=runtime_client,
            custom_llm_client_provider=custom_llm_client_provider,
            origin_history=history,
        )

        try:
            # 2. 准备 messages、token trace session 和 before-agent hooks
            await self._prepare_async_execution(state)

            # 3. 主迭代循环
            while state.iteration < self.max_iterations:
                if self.stop_signal:
                    stop_response = "Stop signal received."
                    stop_response, state.messages = await asyncio.to_thread(
                        self._apply_after_agent_hooks,
                        agent_state=agent_state,
                        messages=state.messages,
                        final_response=stop_response,
                        stop_reason=AgentStopReason.USER_INTERRUPTED,
                    )
                    return stop_response, state.messages

                outcome = await self._execute_iteration_async(state)
                if outcome == _IterationOutcome.BREAK:
                    break

            # 4. 循环结束后处理
            if state.iteration >= self.max_iterations:
                state.force_stop_reason = AgentStopReason.MAX_ITERATIONS_REACHED
                state.final_response += "\\n\\n[Note: Maximum iteration limit reached]"

            state.final_response, state.messages = await asyncio.to_thread(
                self._apply_after_agent_hooks,
                agent_state=agent_state,
                messages=state.messages,
                final_response=state.final_response,
                stop_reason=state.force_stop_reason,
            )

            self._store_token_trace(state.token_trace_session)
            return state.final_response, state.messages

        except TokenTraceContextOverflowError as e:
            state.force_stop_reason = AgentStopReason.CONTEXT_TOKEN_LIMIT
            state.final_response = f"Error: {e}"
            state.final_response, state.messages = await asyncio.to_thread(
                self._apply_after_agent_hooks,
                agent_state=agent_state,
                messages=state.messages,
                final_response=state.final_response,
                stop_reason=state.force_stop_reason,
            )
            self._store_token_trace(state.token_trace_session)
            return state.final_response, state.messages

        except Exception as e:
            state.force_stop_reason = AgentStopReason.ERROR_OCCURRED
            state.final_response = f"Error: {str(e)}"
            state.final_response, state.messages = await asyncio.to_thread(
                self._apply_after_agent_hooks,
                agent_state=agent_state,
                messages=state.messages,
                final_response=state.final_response,
                stop_reason=state.force_stop_reason,
            )
            self._store_token_trace(state.token_trace_session)
            raise RuntimeError(f"Error in agent execution: {e}") from e

        finally:
            if isinstance(state.origin_history, HistoryList):
                state.origin_history.replace_all(state.messages)
            self._store_token_trace(state.token_trace_session)
            if state.token_trace_session is not None:
                state.token_trace_session.synced_message_count = len(state.messages)
            self._execution_done.set()

    # ------------------------------------------------------------------
    # execute_async 辅助方法
    # ------------------------------------------------------------------

    async def _prepare_async_execution(self, state: _AsyncIterationState) -> None:
        """Prepare messages, token trace session and run before-agent hooks.

        历史格式转换 → token trace 初始化 → before-agent middleware

        Mutates *state* in-place.
        """

        history = state.origin_history
        if history and isinstance(history[0], dict):
            state.messages = messages_from_legacy_openai_chat(cast(list[dict[str, Any]], history))
        else:
            state.messages = cast(list[Message], history).copy()

        # Token trace session 初始化
        tools_payload = None
        if self.use_structured_tool_calls:
            with self._tool_registry_lock:
                tools_payload = deepcopy(self.structured_tool_payload)

        if self.llm_config.api_type == "generate_with_token":
            if state.token_trace_session is None:
                state.token_trace_session = TokenTraceSession(
                    self.llm_config,
                    max_context_tokens=self.max_context_tokens,
                )
            state.token_trace_session.initialize_from_messages(
                state.messages,
                tools=cast(list[dict[str, Any]] | None, tools_payload),
            )
            state.token_trace_session.sync_external_messages(state.messages)

        # Before-agent middleware hooks
        if self.middleware_manager:
            before_agent_hook_input = BeforeAgentHookInput(
                agent_state=state.agent_state,
                messages=state.messages,
            )
            try:
                state.messages = await asyncio.to_thread(
                    self.middleware_manager.run_before_agent,
                    before_agent_hook_input,
                )
            except Exception as e:
                logger.warning(f"⚠️ Before-agent middleware execution failed: {e}")

    async def _execute_iteration_async(self, state: _AsyncIterationState) -> _IterationOutcome:
        """Execute one iteration of the main async loop.

        async 主循环单次迭代：排队消息 → team_mode 跳过检查 → LLM 调用 →
        工具执行 → 工具结果构建 → 停止条件判断。

        Mutates *state* in-place and returns the loop control signal.
        """

        # 1. 处理排队消息
        if self.queued_messages:
            state.messages.extend(self.queued_messages)
            self.queued_messages = []

        # 2. team_mode: 跳过无用户内容的 LLM 调用
        if self.team_mode and not any(m.role != Role.SYSTEM for m in state.messages):
            if isinstance(state.origin_history, HistoryList):
                state.origin_history.replace_all(state.messages)
            if not await asyncio.to_thread(self._wait_for_messages):
                return _IterationOutcome.BREAK
            state.iteration += 1
            return _IterationOutcome.CONTINUE

        # 3. team_mode: assistant 结尾时跳过 LLM 调用
        if self.team_mode:
            last_non_system: Message | None = None
            for m in reversed(state.messages):
                if m.role != Role.SYSTEM:
                    last_non_system = m
                    break
            if last_non_system is not None and last_non_system.role == Role.ASSISTANT:
                if isinstance(state.origin_history, HistoryList):
                    state.origin_history.replace_all(state.messages)
                if not await asyncio.to_thread(self._wait_for_messages):
                    return _IterationOutcome.BREAK
                state.iteration += 1
                return _IterationOutcome.CONTINUE

        # 4. Before-model middleware hooks
        before_model_hook_input = BeforeModelHookInput(
            agent_state=state.agent_state,
            max_iterations=self.max_iterations,
            current_iteration=state.iteration,
            messages=state.messages,
        )

        if self.middleware_manager:
            try:
                state.messages = await asyncio.to_thread(
                    self.middleware_manager.run_before_model,
                    before_model_hook_input,
                )
            except Exception as e:
                logger.warning(f"⚠️ Before-model middleware execution failed: {e}")

        # 5. 快照工具定义 & token 计算
        tools_payload = None
        if self.use_structured_tool_calls:
            tools_payload = self._snapshot_structured_tool_definitions()

        current_prompt_tokens = self.token_counter.count_tokens(
            state.messages,
            tools=tools_payload,
        )

        state.force_stop_reason = AgentStopReason.SUCCESS
        if current_prompt_tokens > self.max_context_tokens:
            logger.warning(
                "⚠️ Prompt tokens (%d) exceed max_context_tokens (%d). Continuing.",
                current_prompt_tokens,
                self.max_context_tokens,
            )

        available_tokens = self.max_context_tokens - current_prompt_tokens
        desired_max_tokens = 16384
        _ = max(1, min(desired_max_tokens, available_tokens))  # calculated_max_tokens (reserved for future use)

        if state.iteration == self.max_iterations - 1:
            state.final_response += "\\n\\n[Error: Maximum iteration limit reached.]"
            state.force_stop_reason = AgentStopReason.MAX_ITERATIONS_REACHED

        # 6. 异步 LLM 调用 — 直接 await，不再走 to_thread
        model_response = await self.llm_caller.call_llm_async(
            state.messages,
            openai_client=state.runtime_client,
            force_stop_reason=state.force_stop_reason,
            agent_state=state.agent_state,
            tool_call_mode=self.tool_call_mode,
            tools=tools_payload,
            shutdown_event=self._shutdown_event,
            token_trace_session=state.token_trace_session,
        )
        if model_response is None:
            return _IterationOutcome.BREAK

        # 7. 解析响应并追加 assistant 消息
        assistant_content = model_response.content or ""
        state.final_response = assistant_content

        parsed_response = self.response_parser.parse_response(model_response)
        assistant_message = model_response.to_ump_message()
        state.messages.append(assistant_message)
        if state.token_trace_session is not None:
            state.token_trace_session.append_model_response(
                output_token_ids=model_response.output_token_ids,
                fallback_messages=[assistant_message],
            )

        # 8. 执行工具/子代理调用（含 after-model middleware）
        after_model_hook_input = AfterModelHookInput(
            agent_state=state.agent_state,
            max_iterations=self.max_iterations,
            current_iteration=state.iteration,
            original_response=assistant_content,
            parsed_response=parsed_response,
            messages=state.messages,
            model_response=model_response,
        )

        (
            processed_response,
            should_stop,
            stop_tool_result,
            updated_messages,
            execution_feedbacks,
        ) = await self._process_xml_calls_async(
            after_model_hook_input,
            custom_llm_client_provider=state.custom_llm_client_provider,
            framework_context=state.framework_context,
        )

        state.messages = updated_messages

        # 9. 构建工具结果消息
        self._append_tool_result_messages(
            state=state,
            execution_feedbacks=execution_feedbacks,
            processed_parsed_response=after_model_hook_input.parsed_response,
            assistant_content=assistant_content,
            processed_response=processed_response,
        )

        # 10. 停止条件判断
        if should_stop and len(self.queued_messages) == 0:
            return await self._handle_stop_condition_async(
                state,
                stop_tool_result=stop_tool_result,
                processed_response=processed_response,
            )

        state.iteration += 1
        return _IterationOutcome.CONTINUE

    def _append_tool_result_messages(
        self,
        *,
        state: _AsyncIterationState,
        execution_feedbacks: list[dict[str, Any]],
        processed_parsed_response: ParsedResponse | None,
        assistant_content: str,
        processed_response: str,
    ) -> None:
        """Build tool result messages from execution feedbacks and append to state.

        构建工具结果消息：structured tool 模式生成 TOOL 角色消息，
        XML 模式生成 USER 角色文本反馈。

        Mutates *state.messages* in-place. Also appends to token trace session
        when present.
        """

        openai_tool_mode = bool(
            processed_parsed_response and processed_parsed_response.model_response and processed_parsed_response.model_response.tool_calls
        )

        if openai_tool_mode:
            tool_result_messages: list[Message] = []
            for feedback in execution_feedbacks:
                call_obj = feedback.get("call")
                content = feedback.get("content") or ""
                output = feedback.get("output")

                if isinstance(call_obj, ToolCall):
                    call_id = call_obj.tool_call_id
                else:
                    call_id = None

                if not call_id:
                    continue

                tool_result_block = ToolResultBlock(
                    tool_use_id=str(call_id),
                    content=coerce_tool_result_content(
                        feedback.get("llm_tool_output", output if output is not None else content),
                        fallback_text=None,
                    ),
                    is_error=bool(feedback.get("is_error")),
                )
                # micro-compact: 设置 created_at 时间戳
                from datetime import UTC, datetime

                tool_result_message = Message(role=Role.TOOL, content=[tool_result_block], created_at=datetime.now(UTC))
                state.messages.append(tool_result_message)
                tool_result_messages.append(tool_result_message)

            tool_results = ""
            if state.token_trace_session is not None and tool_result_messages:
                state.token_trace_session.append_messages(tool_result_messages, mask_value=0)
        else:
            tool_results = processed_response.replace(assistant_content, "", 1).strip()

        if tool_results:
            tool_result_feedback_message = Message(role=Role.USER, content=[TextBlock(text=f"Tool execution results:\n{tool_results}")])
            state.messages.append(tool_result_feedback_message)
            if state.token_trace_session is not None:
                state.token_trace_session.append_messages([tool_result_feedback_message], mask_value=0)

    async def _handle_stop_condition_async(
        self,
        state: _AsyncIterationState,
        *,
        stop_tool_result: str | None,
        processed_response: str,
    ) -> _IterationOutcome:
        """Determine whether to break or continue when should_stop is True.

        停止条件处理：team_mode 区分 finish_team（退出）和其他 stop tool（等待新消息），
        普通模式直接退出循环。

        Mutates *state.final_response* and *state.force_stop_reason* when breaking.
        """

        if self.team_mode:
            if stop_tool_result is not None and self._last_stop_tool_name == "finish_team":
                state.force_stop_reason = AgentStopReason.STOP_TOOL_TRIGGERED
                state.final_response = stop_tool_result
                return _IterationOutcome.BREAK
            self._mark_waiting_for_user()
            if isinstance(state.origin_history, HistoryList):
                state.origin_history.replace_all(state.messages)
            if not await asyncio.to_thread(self._wait_for_messages):
                state.force_stop_reason = AgentStopReason.NO_MORE_TOOL_CALLS
                state.final_response = processed_response
                return _IterationOutcome.BREAK
            state.iteration += 1
            return _IterationOutcome.CONTINUE

        if stop_tool_result is not None:
            state.force_stop_reason = AgentStopReason.STOP_TOOL_TRIGGERED
            state.final_response = stop_tool_result
            return _IterationOutcome.BREAK

        state.force_stop_reason = AgentStopReason.NO_MORE_TOOL_CALLS
        state.final_response = processed_response
        return _IterationOutcome.BREAK

    async def _process_xml_calls_async(
        self,
        hook_input: AfterModelHookInput,
        *,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
        framework_context: FrameworkContext,
    ) -> tuple[str, bool, str | None, list[Message], list[dict[str, Any]]]:
        """Async version of _process_xml_calls.

        Middleware hooks 通过 to_thread 调用（sync API），
        tool/sub-agent 通过 _execute_parsed_calls_async 异步执行。
        """

        response_payload: str | ModelResponse = hook_input.model_response or hook_input.original_response
        parsed_response: ParsedResponse | None = hook_input.parsed_response or self.response_parser.parse_response(
            response_payload,
        )
        hook_input.parsed_response = parsed_response
        current_messages = hook_input.messages.copy()
        force_continue = False

        if self.middleware_manager:
            try:
                parsed_response, current_messages, force_continue = await asyncio.to_thread(
                    self.middleware_manager.run_after_model, hook_input
                )
            except Exception as e:
                logger.warning(f"⚠️ After-model middleware execution failed: {e}")

        if not parsed_response or not parsed_response.has_calls():
            if force_continue:
                return hook_input.original_response, False, None, current_messages, []
            else:
                return hook_input.original_response, True, None, current_messages, []

        assert parsed_response is not None
        processed_response, should_stop, stop_tool_result, execution_feedbacks = await self._execute_parsed_calls_async(
            parsed_response,
            hook_input.agent_state,
            custom_llm_client_provider=custom_llm_client_provider,
            framework_context=framework_context,
        )
        return processed_response, should_stop, stop_tool_result, current_messages, execution_feedbacks

    async def _execute_parsed_calls_async(
        self,
        parsed_response: ParsedResponse,
        agent_state: "AgentState",
        *,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
        framework_context: FrameworkContext,
    ) -> tuple[str, bool, str | None, list[dict[str, Any]]]:
        """Async parallel tool/sub-agent execution via asyncio.gather.

        async tool → 直接 await tool.execute_async()
        sync tool / sub-agent → asyncio.to_thread 在线程池执行
        """

        processed_response = parsed_response.original_response

        if self._shutdown_event.is_set():
            return processed_response, False, None, []

        if not parsed_response.tool_calls:
            return processed_response, False, None, []

        parallel_execution_id = str(uuid.uuid4())

        # Deduplicate tool_call_ids
        seen_tool_call_ids: defaultdict[str, int] = defaultdict(int)
        for idx, tool_call in enumerate(parsed_response.tool_calls):
            base_id = tool_call.tool_call_id or f"tool_call_{idx}"
            count = seen_tool_call_ids[base_id]
            tool_call.tool_call_id = f"{base_id}_{count}" if count else base_id
            seen_tool_call_ids[base_id] += 1
            tool_call.parallel_execution_id = parallel_execution_id

        serial_tool_names = set(self._tool_registry.compute_serial_tool_names())

        # 构建 async tasks
        async def _run_tool(tc: ToolCall) -> tuple[str, ToolCall, tuple[str, Any, bool]]:
            """Dispatch tool execution based on sync/async implementation type.

            - Sync tool: 复用 _execute_tool_call_safe 在 worker 线程中执行。
              整条链（get_sandbox → before_tool → tool.execute → after_tool）
              都在同一个 worker 线程中运行，没有 running event loop，
              与改造前 ThreadPoolExecutor 行为完全一致。
              用户 sync tool 可以安全调用 agent_state.get_sandbox()、
              run_async_function_sync() 等 sync framework API。

            - Async tool: 在 event loop 上直接 await tool.execute_async()。
              Async tool 应使用 async-native API，不应调用 sync-only 的
              run_async_function_sync()。
            """
            if self._shutdown_event.is_set():
                return ("tool", tc, (tc.tool_name, "Shutdown in progress", True))

            tool_obj = self._tool_registry.get_tool(tc.tool_name)
            if tool_obj is None:
                return ("tool", tc, (tc.tool_name, f"Tool '{tc.tool_name}' not found", True))

            # 检测 async tool: implementation 是 async def，
            # 或子类（如 MCPTool）声明了原生 async execute_async() 覆盖。
            is_async_impl = (
                tool_obj.implementation is not None and inspect.iscoroutinefunction(tool_obj.implementation)
            ) or tool_obj.has_native_async_execute

            if not is_async_impl:
                # ── Sync tool: 整条链在 worker 线程中执行 ──
                # 与改造前 ThreadPoolExecutor.submit(_execute_tool_call_safe) 行为一致
                return (
                    "tool",
                    tc,
                    await asyncio.to_thread(
                        self._execute_tool_call_safe,
                        tc,
                        agent_state,
                        framework_context,
                    ),
                )

            # ── Async tool: event loop 原生路径 ──
            try:
                converted_params: dict[str, Any] = {}
                for pn, pv in tc.parameters.items():
                    converted_params[pn] = self.tool_executor.convert_parameter_type(tc.tool_name, pn, pv)

                # get_sandbox 内部可能调用 sync session API，放到 to_thread 避免阻塞
                sandbox: BaseSandbox | None = None
                if tc.tool_name not in {"LoadSkill"}:
                    sandbox = await asyncio.to_thread(agent_state.get_sandbox)

                # before_tool middleware hook (sync hook → to_thread)
                tool_parameters = converted_params.copy()
                if self.middleware_manager:
                    from nexau.archs.main_sub.execution.hooks import BeforeToolHookInput

                    before_input = BeforeToolHookInput(
                        agent_state=agent_state,
                        sandbox=sandbox,
                        tool_name=tc.tool_name,
                        tool_call_id=tc.tool_call_id or "",
                        tool_input=tool_parameters,
                        parallel_execution_id=tc.parallel_execution_id,
                    )
                    tool_parameters = await asyncio.to_thread(self.middleware_manager.run_before_tool, before_input)

                exec_params: dict[str, Any] = dict(tool_parameters)
                exec_params["agent_state"] = agent_state
                exec_params["sandbox"] = sandbox
                exec_params["ctx"] = framework_context

                # Async tool: 直接 await（不经过 to_thread）
                result = await tool_obj.execute_async(**exec_params)
                execution_result = await asyncio.to_thread(
                    self.tool_executor.finalize_tool_execution,
                    agent_state=agent_state,
                    sandbox=sandbox,
                    tool=tool_obj,
                    tool_name=tc.tool_name,
                    tool_parameters=tool_parameters,
                    tool_call_id=tc.tool_call_id or "",
                    result=result,
                    execution_error=None,
                )

                return ("tool", tc, (tc.tool_name, execution_result, False))
            except Exception as e:
                return ("tool", tc, (tc.tool_name, str(e), True))

        # serial tools 需要顺序执行，其余并行
        serial_tasks: list[ToolCall] = []
        parallel_tool_tasks: list[ToolCall] = []
        for tc in parsed_response.tool_calls:
            if tc.tool_name in serial_tool_names:
                serial_tasks.append(tc)
            else:
                parallel_tool_tasks.append(tc)

        # 结果类型: (call_type, call_obj, (name, result, is_error))
        all_results: list[tuple[str, ToolCall, tuple[str, Any, bool]]] = []

        # 先执行 serial tools（顺序）
        for tc in serial_tasks:
            all_results.append(await _run_tool(tc))

        # 再并行执行剩余 tools
        # 维护 call_origins 与 parallel_coros 索引一一对应，gather 异常时保留调用上下文
        parallel_coros: list[Any] = []
        call_origins: list[tuple[str, ToolCall]] = []
        for tc in parallel_tool_tasks:
            parallel_coros.append(_run_tool(tc))
            call_origins.append(("tool", tc))

        if parallel_coros:
            parallel_results = await asyncio.gather(*parallel_coros, return_exceptions=True)
            for idx, r in enumerate(parallel_results):
                if isinstance(r, BaseException):
                    _, origin_call = call_origins[idx]
                    all_results.append(("tool", origin_call, (origin_call.tool_name, str(r), True)))
                elif isinstance(r, tuple):
                    all_results.append(r)  # type: ignore[arg-type]

        # 收集结果
        tool_results: list[str] = []
        execution_feedbacks: list[dict[str, Any]] = []
        stop_tool_detected = False
        stop_tool_result: str | None = None

        for entry in all_results:
            call_type: str = entry[0]
            call_obj: ToolCall = entry[1]
            result_data: tuple[str, Any, bool] = entry[2]
            tool_name, result, is_error = result_data
            raw_output, llm_tool_output = self._split_tool_outputs(result)
            result_str = self._serialize_llm_tool_output(llm_tool_output)
            execution_feedbacks.append(
                {
                    "call_type": call_type,
                    "call": call_obj,
                    "content": result_str,
                    "output": raw_output,
                    "llm_tool_output": llm_tool_output,
                    "is_error": is_error,
                }
            )

            # ToolCall.source indicates call format; structured tool calls do not append XML tool results.
            should_append_xml = call_obj.source != "structured"
            if is_error:
                if should_append_xml:
                    tool_results.append(
                        f"\n<tool_result>\n<tool_name>{tool_name}</tool_name>\n<error>{result_str}</error>\n</tool_result>\n"
                    )
            else:
                if should_append_xml:
                    tool_results.append(
                        f"\n<tool_result>\n<tool_name>{tool_name}</tool_name>\n<result>{result_str}</result>\n</tool_result>\n"
                    )
                # 检查 stop tool
                try:
                    if isinstance(raw_output, dict):
                        raw_output_dict = cast(dict[str, Any], raw_output)
                    else:
                        raw_output_dict = None

                    if raw_output_dict is not None and raw_output_dict.get("_is_stop_tool"):
                        stop_tool_detected = True
                        self._last_stop_tool_name = tool_name
                        actual_result: dict[str, Any] = {key: value for key, value in raw_output_dict.items() if key != "_is_stop_tool"}
                        if "result" in actual_result and len(actual_result) == 1:
                            stop_tool_result = json.dumps(actual_result["result"], ensure_ascii=False, indent=4)
                        else:
                            stop_tool_result = json.dumps(actual_result or raw_output_dict, ensure_ascii=False, indent=4)
                except TypeError:
                    pass

        if tool_results:
            processed_response += "\n\n" + "\n\n".join(tool_results)

        return processed_response, stop_tool_detected, stop_tool_result, execution_feedbacks

    def _store_token_trace(self, token_trace_session: TokenTraceSession | None) -> None:
        """Persist token trace data into shared trace memory."""
        if token_trace_session is None or self.global_storage is None:
            return

        existing_trace_memory = self.global_storage.get("trace_memory", {})
        trace_memory = cast(dict[str, Any], existing_trace_memory) if isinstance(existing_trace_memory, dict) else {}
        trace_memory.update(token_trace_session.export_trace())
        self.global_storage.set("trace_memory", trace_memory)

    def _apply_after_agent_hooks(
        self,
        *,
        agent_state: "AgentState",
        messages: list[Message],
        final_response: str,
        stop_reason: AgentStopReason | None,
    ) -> tuple[str, list[Message]]:
        """Run after-agent middleware hooks and return possibly updated values."""

        if not self.middleware_manager:
            return final_response, messages

        after_agent_hook_input = AfterAgentHookInput(
            agent_state=agent_state,
            messages=messages,
            agent_response=final_response,
            stop_reason=stop_reason,
        )
        try:
            return self.middleware_manager.run_after_agent(after_agent_hook_input)
        except Exception as exc:
            logger.warning(f"⚠️ After-agent middleware execution failed: {exc}")
            return final_response, messages

    @staticmethod
    def _build_middleware_manager(
        configured_middlewares: list[Middleware],
        before_model_hooks: list[BeforeModelHook],
        after_model_hooks: list[AfterModelHook],
        after_tool_hooks: list[AfterToolHook],
        before_tool_hooks: list[BeforeToolHook],
    ) -> MiddlewareManager | None:
        combined: list[Middleware] = list(configured_middlewares)

        def _hook_name(hook: Callable[..., Any]) -> str:
            return getattr(hook, "__name__", hook.__class__.__name__)

        for bm_hook in before_model_hooks:
            combined.append(
                FunctionMiddleware(
                    before_model_hook=bm_hook,
                    name=f"before_model::{_hook_name(bm_hook)}",
                ),
            )

        for am_hook in after_model_hooks:
            combined.append(
                FunctionMiddleware(
                    after_model_hook=am_hook,
                    name=f"after_model::{_hook_name(am_hook)}",
                ),
            )

        for at_hook in after_tool_hooks:
            combined.append(
                FunctionMiddleware(
                    after_tool_hook=at_hook,
                    name=f"after_tool::{_hook_name(at_hook)}",
                ),
            )

        for bt_hook in before_tool_hooks:
            combined.append(
                FunctionMiddleware(
                    before_tool_hook=bt_hook,
                    name=f"before_tool::{_hook_name(bt_hook)}",
                ),
            )

        return MiddlewareManager(combined)

    def _process_xml_calls(
        self,
        hook_input: AfterModelHookInput,
        *,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
        framework_context: FrameworkContext,
    ) -> tuple[str, bool, str | None, list[Message], list[dict[str, Any]]]:
        """Process XML tool calls and sub-agent calls using two-phase approach.

        Args:
            response: Agent response containing XML calls
            messages: Current conversation history

        Returns:
            Tuple of (processed_response, should_stop, stop_tool_result, updated_messages)
        """
        # Phase 1: Parse the response to extract all calls
        logger.info("📋 Phase 1: Parsing LLM response for all executable calls")
        response_payload: str | ModelResponse = hook_input.model_response or hook_input.original_response
        parsed_response: ParsedResponse | None = hook_input.parsed_response or self.response_parser.parse_response(
            response_payload,
        )
        hook_input.parsed_response = parsed_response

        # Keep track of current messages (may be modified by hooks)
        current_messages = hook_input.messages.copy()
        force_continue = False  # Default: don't force continue

        # Execute middlewares if any are configured (always run even if no calls)
        if self.middleware_manager:
            try:
                parsed_response, current_messages, force_continue = self.middleware_manager.run_after_model(hook_input)
            except Exception as e:
                logger.warning(f"⚠️ After-model middleware execution failed: {e}")

        # If no calls found after hooks, check if we should force continue
        if not parsed_response or not parsed_response.has_calls():
            if force_continue:
                # Hook removed all calls but added feedback, let agent continue
                logger.info(
                    "🎣 No tool calls remaining, but hook requested force_continue. Agent will continue with feedback.",
                )
                return hook_input.original_response, False, None, current_messages, []
            else:
                # Normal behavior: no calls means stop
                logger.info(
                    "🛑 No tool calls remaining, stopping.",
                )
                return hook_input.original_response, True, None, current_messages, []

        # Phase 2: Execute all parsed calls
        logger.info(
            f"⚡ Phase 2: Executing {parsed_response.get_call_summary()}",
        )
        assert parsed_response is not None
        processed_response, should_stop, stop_tool_result, execution_feedbacks = self._execute_parsed_calls(
            parsed_response,
            hook_input.agent_state,
            custom_llm_client_provider=custom_llm_client_provider,
            framework_context=framework_context,
        )
        return processed_response, should_stop, stop_tool_result, current_messages, execution_feedbacks

    def _execute_parsed_calls(
        self,
        parsed_response: ParsedResponse,
        agent_state: "AgentState",
        *,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
        framework_context: FrameworkContext,
    ) -> tuple[str, bool, str | None, list[dict[str, Any]]]:
        """Execute all parsed calls in parallel.

        Args:
            parsed_response: ParsedResponse containing all calls to execute
            agent_state: AgentState containing agent context and global storage

        Returns:
            Tuple of (processed_response, should_stop, stop_tool_result)
        """
        processed_response = parsed_response.original_response

        # Check if agent is shutting down
        if self._shutdown_event.is_set():
            logger.warning(
                f"⚠️ Agent '{self.agent_name}' ({self.agent_id}) is shutting down, skipping new task execution",
            )
            return processed_response, False, None, []

        # Execute tool calls in parallel
        if not parsed_response.tool_calls:
            return processed_response, False, None, []

        executor_id = str(uuid.uuid4())
        parallel_execution_id = str(uuid.uuid4())

        tool_executor = ThreadPoolExecutor()

        # Track executors for cleanup
        with self._executor_lock:
            self._running_executors[f"{executor_id}_tools"] = tool_executor

        # Handle duplicate tool_call_ids by adding suffixes
        seen_tool_call_ids: defaultdict[str, int] = defaultdict(int)
        for idx, tool_call in enumerate(parsed_response.tool_calls):
            base_id = tool_call.tool_call_id or f"tool_call_{idx}"
            count = seen_tool_call_ids[base_id]
            if count:
                tool_call.tool_call_id = f"{base_id}_{count}"
            else:
                tool_call.tool_call_id = base_id
            seen_tool_call_ids[base_id] += 1
            # Set parallel execution ID for grouping
            tool_call.parallel_execution_id = parallel_execution_id

        serial_tool_names = set(self._tool_registry.compute_serial_tool_names())

        try:
            # Batch all context snapshots before any task submission to prevent
            # OTel context pollution between parallel tool threads (fix-span-overlap)
            tool_snapshots = [(copy_context(), tool_call) for tool_call in parsed_response.tool_calls]

            # Submit tool execution tasks using pre-created context snapshots
            tool_futures: dict[Future[tuple[str, Any, bool]], tuple[str, ToolCall]] = {}
            for task_ctx, tool_call in tool_snapshots:
                # RFC-0001: 每次提交前检查 shutdown_event，避免中断后继续执行
                if self._shutdown_event.is_set():
                    logger.info("🛑 Shutdown event detected, skipping remaining tool calls")
                    break

                future = tool_executor.submit(
                    task_ctx.run,
                    self._execute_tool_call_safe,
                    tool_call,
                    agent_state,
                    framework_context,
                )
                tool_futures[future] = ("tool", tool_call)

                if tool_call.tool_name in serial_tool_names:
                    future.result()

            # Combine all futures
            all_futures: dict[Future[tuple[str, Any, bool]], tuple[str, ToolCall]] = {**tool_futures}

            # Collect results as they complete
            tool_results: list[str] = []
            execution_feedbacks: list[dict[str, Any]] = []
            stop_tool_detected = False
            stop_tool_result = None

            for future in as_completed(all_futures):
                call_type, call_obj = all_futures[future]
                try:
                    result_data = future.result()
                    tool_name, result, is_error = result_data
                    raw_output, llm_tool_output = self._split_tool_outputs(result)
                    result_str = self._serialize_llm_tool_output(llm_tool_output)
                    execution_feedbacks.append(
                        {
                            "call_type": "tool",
                            "call": call_obj,
                            "content": result_str,
                            "output": raw_output,
                            "llm_tool_output": llm_tool_output,
                            "is_error": is_error,
                        },
                    )
                    if is_error:
                        logger.error(
                            f"❌ Tool '{tool_name}' error: {result_str}",
                        )
                        should_append_xml = call_obj.source != "structured"
                        if should_append_xml:
                            tool_results.append(
                                f"""
<tool_result>
<tool_name>{tool_name}</tool_name>
<error>{result_str}</error>
</tool_result>
""",
                            )
                    else:
                        logger.info(
                            f"📤 Tool '{tool_name}' result: {result_str[:100]}",
                        )
                        should_append_xml = call_obj.source != "structured"
                        tool_result_xml = f"""
<tool_result>
<tool_name>{tool_name}</tool_name>
<result>{result_str}</result>
</tool_result>
"""
                        if should_append_xml:
                            tool_results.append(tool_result_xml)

                        # Check if this tool result indicates a stop tool was executed
                        try:
                            if isinstance(raw_output, dict):
                                raw_output_dict = cast(dict[str, Any], raw_output)
                            else:
                                raw_output_dict = None

                            if raw_output_dict is not None and raw_output_dict.get("_is_stop_tool"):
                                stop_tool_detected = True
                                self._last_stop_tool_name = tool_name
                                actual_result: dict[str, Any] = {
                                    key: value for key, value in raw_output_dict.items() if key != "_is_stop_tool"
                                }
                                if "result" in actual_result and len(actual_result) == 1:
                                    stop_tool_result = json.dumps(
                                        actual_result["result"],
                                        ensure_ascii=False,
                                        indent=4,
                                    )
                                else:
                                    stop_tool_result = (
                                        json.dumps(
                                            actual_result,
                                            ensure_ascii=False,
                                            indent=4,
                                        )
                                        if actual_result
                                        else json.dumps(
                                            raw_output_dict,
                                            ensure_ascii=False,
                                            indent=4,
                                        )
                                    )
                                logger.info(
                                    f"🛑 Stop tool '{tool_name}' result detected, will terminate after processing",
                                )
                        except TypeError:
                            pass
                except Exception as e:
                    logger.error(
                        f"❌ Unexpected error processing {call_type}: {e}",
                    )
                    tool_results.append(
                        f"""
<tool_result>
<tool_name>unknown</tool_name>
<error>Unexpected error: {str(e)}</error>
</tool_result>
""",
                    )

            # Append tool results to the original response
            if tool_results:
                processed_response += "\n\n" + "\n\n".join(tool_results)

        finally:
            # Clean up executors
            with self._executor_lock:
                try:
                    tool_executor.shutdown(wait=True, cancel_futures=False)
                except Exception as e:
                    logger.error(f"❌ Error shutting down executors: {e}")
                finally:
                    self._running_executors.pop(f"{executor_id}_tools", None)

        return processed_response, stop_tool_detected, stop_tool_result, execution_feedbacks

    def _execute_tool_call_safe(
        self,
        tool_call: ToolCall,
        agent_state: "AgentState",
        framework_context: FrameworkContext,
    ) -> tuple[str, Any, bool]:
        """Safely execute a tool call."""
        try:
            # Convert parameters to correct types and execute
            converted_params: dict[str, Any] = {}
            for param_name, param_value in tool_call.parameters.items():
                converted_params[param_name] = self.tool_executor.convert_parameter_type(
                    tool_call.tool_name,
                    param_name,
                    param_value,
                )

            tool_call_id = tool_call.tool_call_id or f"tool_call_{uuid.uuid4()}"
            result = self.tool_executor.execute_tool_with_llm_output(
                agent_state,
                tool_call.tool_name,
                converted_params,
                tool_call_id=tool_call_id,
                parallel_execution_id=tool_call.parallel_execution_id,
                framework_context=framework_context,
            )

            return (tool_call.tool_name, result, False)

        except Exception as e:
            return tool_call.tool_name, str(e), True

    def force_stop(self) -> None:
        """Force-stop the executor, breaking the team_mode forever-run loop.

        RFC-0002: 强制停止 team_mode 下的永久运行循环

        Sets stop_signal and wakes the message wait so the loop exits immediately.
        """
        self.stop_signal = True
        self._shutdown_event.set()
        self._message_available.set()

    def cleanup(self) -> None:
        """Clean up executor resources."""
        logger.info(f"🧹 Cleaning up executor for agent '{self.agent_name}'...")
        self.stop_signal = True

        # Signal shutdown to prevent new tasks
        self._shutdown_event.set()

        # RFC-0001: 释放 LLM 资源
        # 1. shutdown(wait=False) 释放 middleware 路径的专用线程池
        # 2. close() 关闭 sync/async HTTP client 连接池
        try:
            self.llm_caller.shutdown_thread_pool()
        except Exception as e:
            logger.warning(f"⚠️ Error shutting down LLM thread pool: {e}")

        client = self.llm_caller.openai_client
        if client is not None:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"⚠️ Error closing sync LLM client during cleanup: {e}")

        async_client = self.llm_caller.async_openai_client
        if async_client is not None:
            # AsyncOpenAI/AsyncAnthropic.close() 是 coroutine，sync cleanup 无法 await。
            # 释放引用让 GC 回收即可；async client 的连接会在析构时自动关闭。
            self.llm_caller.async_openai_client = None

        # Shutdown subagent manager
        self.subagent_manager.shutdown()

        # Shutdown all running executors
        with self._executor_lock:
            for executor_id, executor in self._running_executors.items():
                try:
                    logger.info(f"🛑 Shutting down executor {executor_id}")
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    logger.error(
                        f"❌ Error shutting down executor {executor_id}: {e}",
                    )

            self._running_executors.clear()

        logger.info(
            f"✅ Executor cleanup completed for agent '{self.agent_name}'",
        )

    @staticmethod
    def _structured_tool_description(tool: Tool) -> str:
        """Return the description exposed to structured tool-calling models."""

        return tool.get_structured_description()

    @staticmethod
    def _split_tool_outputs(result: Any) -> tuple[Any, Any]:
        """Split raw and llm-facing outputs from the execution result."""

        if isinstance(result, ToolExecutionResult):
            return result.raw_output, result.llm_tool_output
        return result, result

    @staticmethod
    def _serialize_llm_tool_output(value: Any) -> str:
        """Serialize llm-facing tool output for text-only feedback paths."""

        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the executor.

        Args:
            tool: Tool instance to add
        """
        if tool.defer_loading:
            raise ValueError(
                "Runtime-added deferred tools are not supported. Register deferred tools during agent initialization instead.",
            )

        with self._tool_registry_lock:
            # Keep registry and structured definition updates atomic for concurrent readers/writers
            self._tool_registry.add_source("runtime", [tool])
            if self.use_structured_tool_calls:
                self.structured_tool_definitions.append(
                    tool.to_structured_definition(
                        description=self._structured_tool_description(tool),
                    ),
                )

    def add_sub_agent(self, name: str, agent_config: AgentConfig) -> None:
        """Add a sub-agent config.

        Args:
            name: Name of the sub-agent
            agent_config: Config creates the agent
        """
        self.subagent_manager.add_sub_agent(name, agent_config)
