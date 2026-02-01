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

"""Main execution orchestrator for agents."""

import json
import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from anthropic.types import ToolParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.batch_processor import BatchProcessor
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
    BatchAgentCall,
    ParsedResponse,
    SubAgentCall,
    ToolCall,
)
from nexau.archs.main_sub.execution.response_parser import ResponseParser
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager
from nexau.archs.main_sub.execution.tool_executor import ToolExecutor
from nexau.archs.main_sub.tool_call_modes import (
    STRUCTURED_TOOL_CALL_MODES,
    normalize_tool_call_mode,
)
from nexau.archs.main_sub.utils.token_counter import TokenCounter
from nexau.archs.sandbox.base_sandbox import BaseSandboxManager
from nexau.archs.tool.tool import Tool
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock, coerce_tool_result_content

if TYPE_CHECKING:
    from nexau.archs.session import SessionManager

logger = logging.getLogger(__name__)


class Executor:
    """Orchestrates execution of agent tasks with parallel processing support."""

    def __init__(
        self,
        agent_name: str,
        agent_id: str,
        tool_registry: dict[str, Any],
        sub_agents: dict[str, AgentConfig],
        stop_tools: set[str],
        openai_client: Any,
        llm_config: LLMConfig,
        sandbox_manager: BaseSandboxManager[Any],
        max_iterations: int = 100,
        max_context_tokens: int = 128000,
        max_running_subagents: int = 5,
        retry_attempts: int = 5,
        token_counter: TokenCounter | None = None,
        after_model_hooks: list[AfterModelHook] | None = None,
        before_model_hooks: list[BeforeModelHook] | None = None,
        after_tool_hooks: list[AfterToolHook] | None = None,
        before_tool_hooks: list[BeforeToolHook] | None = None,
        middlewares: list[Middleware] | None = None,
        serial_tool_name: list[str] | None = None,
        global_storage: Any = None,
        tool_call_mode: str = "openai",
        openai_tools: list[ChatCompletionToolParam] | list[ToolParam] | None = None,
        session_manager: "SessionManager | None" = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ):
        """Initialize executor.

        Args:
            agent_name: Name of the agent
            agent_id: ID of the agent
            tool_registry: Dictionary of available tools
            serial_tool_name: List of tool names that should be executed serially
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
            tool_call_mode: Preferred tool call format ('xml', 'openai', or 'anthropic')
            openai_tools: Structured tool definitions for OpenAI/anthropic tool calls
            session_manager: Optional SessionManager for unified data access
            user_id: Optional user ID for persistence
            session_id: Optional session ID for persistence
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.max_running_subagents = max_running_subagents

        # Initialize components
        self._tool_registry_lock = threading.RLock()
        self.middleware_manager = self._build_middleware_manager(
            middlewares or [],
            before_model_hooks or [],
            after_model_hooks or [],
            after_tool_hooks or [],
            before_tool_hooks or [],
        )
        self.tool_executor = ToolExecutor(
            tool_registry=tool_registry,
            stop_tools=stop_tools,
            middleware_manager=self.middleware_manager,
            registry_lock=self._tool_registry_lock,
        )

        self.subagent_manager = SubAgentManager(
            agent_name,
            sub_agents,
            global_storage,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )
        self.batch_processor = BatchProcessor(
            self.subagent_manager,
            max_running_subagents,
        )
        self.response_parser = ResponseParser()
        self.llm_caller = LLMCaller(
            openai_client,
            llm_config,
            retry_attempts,
            middleware_manager=self.middleware_manager,
            global_storage=global_storage,
        )

        # Execution parameters
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.global_storage = global_storage

        # Token counting
        self.token_counter = token_counter or TokenCounter()

        # Tool call behavior
        self.serial_tool_name = serial_tool_name or []
        self.tool_call_mode = normalize_tool_call_mode(tool_call_mode)
        self.use_structured_tool_calls = self.tool_call_mode in STRUCTURED_TOOL_CALL_MODES
        self.structured_tool_payload: list[ChatCompletionToolParam] | list[ToolParam] = deepcopy(openai_tools) if openai_tools else []
        if self.use_structured_tool_calls and not self.structured_tool_payload:
            logger.warning(
                f"⚠️ {self.tool_call_mode.capitalize()} tool call mode enabled but no tool definitions were provided.",
            )

        # Process tracking for parallel execution
        self._running_executors: dict[str, ThreadPoolExecutor] = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.stop_signal = False

        # Message queue for dynamic message enqueueing during execution
        self.queued_messages: list[Message] = []

        # Sandbox manager
        self.sandbox_manager = sandbox_manager

    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be processed during execution.

        Args:
            message: Message dictionary with 'role' and 'content' keys
        """
        # Keep public API stable (dict input), but immediately normalize to UMP Message internally.
        role = Role(message.get("role", "user"))
        content = message.get("content", "")
        self.queued_messages.append(Message(role=role, content=[TextBlock(text=content)]))
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

        messages: list[Message] = []

        force_stop_reason = AgentStopReason.SUCCESS

        try:
            # Use history directly as the single source of truth
            if history and isinstance(history[0], dict):
                messages = messages_from_legacy_openai_chat(cast(list[dict[str, Any]], history))
            else:
                messages = cast(list[Message], history).copy()

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
                        stop_reason=None,
                    )
                    return stop_response, messages

                # Process any queued messages
                if self.queued_messages:
                    logger.info(
                        f"📝 Processing {len(self.queued_messages)} queued messages",
                    )
                    messages.extend(self.queued_messages)
                    self.queued_messages = []

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
                    with self._tool_registry_lock:
                        # Snapshot current structured tool definitions under lock to avoid concurrent mutation
                        tools_payload = deepcopy(self.structured_tool_payload)

                # Count current prompt tokens (including tool definitions if present)

                current_prompt_tokens = self.token_counter.count_tokens(
                    messages,
                    tools=cast(list[dict[str, Any]], tools_payload),
                )

                force_stop_reason = AgentStopReason.SUCCESS
                # Check if prompt exceeds max context tokens - force stop if so
                if current_prompt_tokens > self.max_context_tokens:
                    logger.error(
                        f"❌ Prompt tokens ({current_prompt_tokens}) exceed max_context_tokens \
                            ({self.max_context_tokens}). Stopping execution.",
                    )
                    final_response += f"\\n\\n[Error: Prompt too long ({current_prompt_tokens} tokens) exceeds maximum context \
                        ({self.max_context_tokens} tokens). Execution stopped.]"
                    force_stop_reason = AgentStopReason.CONTEXT_TOKEN_LIMIT

                # Calculate max_tokens dynamically based on available budget
                available_tokens = self.max_context_tokens - current_prompt_tokens

                # Get desired max_tokens from LLM config or use reasonable default
                desired_max_tokens = 16384  # Default value
                calculated_max_tokens = min(
                    desired_max_tokens,
                    available_tokens,
                )

                # Ensure we have at least some tokens for response
                if calculated_max_tokens < 50:
                    logger.error(
                        f"❌ Insufficient tokens for response ({calculated_max_tokens}). Stopping execution.",
                    )
                    final_response += f"\\n\\n[Error: Insufficient tokens for response ({calculated_max_tokens} tokens). Context too full.]"
                    force_stop_reason = AgentStopReason.CONTEXT_TOKEN_LIMIT

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
                messages.append(model_response.to_ump_message())

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
                    for feedback in execution_feedbacks:
                        call_obj = feedback.get("call")
                        call_type = feedback.get("call_type")
                        content = feedback.get("content") or ""
                        output = feedback.get("output")

                        if call_type == "tool":
                            call_id = getattr(call_obj, "tool_call_id", None)
                        elif call_type == "sub_agent":
                            call_id = getattr(call_obj, "tool_call_id", None) or getattr(call_obj, "sub_agent_call_id", None)
                        else:
                            call_id = None

                        if not call_id:
                            continue

                        tool_result_block = ToolResultBlock(
                            tool_use_id=str(call_id),
                            content=coerce_tool_result_content(output if output is not None else content, fallback_text=str(content)),
                            is_error=bool(feedback.get("is_error")),
                        )

                        messages.append(
                            Message(
                                role=Role.TOOL,
                                content=[tool_result_block],
                            ),
                        )

                    tool_results = ""
                else:
                    tool_results = processed_response.replace(
                        assistant_content,
                        "",
                        1,
                    ).strip()

                if tool_results:
                    from nexau.core.messages import TextBlock

                    messages.append(Message(role=Role.USER, content=[TextBlock(text=f"Tool execution results:\n{tool_results}")]))

                # Check if a stop tool was executed
                if should_stop and len(self.queued_messages) == 0:
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
            # Re-raise with more context
            raise RuntimeError(f"Error in agent execution: {e}") from e

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
        )
        return processed_response, should_stop, stop_tool_result, current_messages, execution_feedbacks

    def _execute_parsed_calls(
        self,
        parsed_response: ParsedResponse,
        agent_state: "AgentState",
        *,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
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

        # Handle batch agent calls first (they take priority and are not parallelized)
        if parsed_response.batch_agent_calls:
            for batch_call in parsed_response.batch_agent_calls:
                try:
                    batch_result = self._execute_batch_call(batch_call)
                    processed_response += f"""
<tool_result>
<tool_name>batch_agent</tool_name>
<result>{batch_result}</result>
</tool_result>
"""
                except Exception as e:
                    logger.error(f"❌ Batch agent call failed: {e}")
                    processed_response += f"""
<tool_result>
<tool_name>batch_agent</tool_name>
<error>{str(e)}</error>
</tool_result>
"""
            return processed_response, False, None, []

        # Execute tool calls and sub-agent calls in parallel
        if not parsed_response.tool_calls and not parsed_response.sub_agent_calls:
            return processed_response, False, None, []

        # Get current context to pass to sub-agents
        from ..agent_context import get_context

        current_context = get_context()
        context_dict = current_context.context.copy() if current_context else None

        executor_id = str(uuid.uuid4())
        tool_executor = ThreadPoolExecutor()
        subagent_executor = ThreadPoolExecutor(
            max_workers=self.max_running_subagents,
        )

        # Track executors for cleanup
        with self._executor_lock:
            self._running_executors[f"{executor_id}_tools"] = tool_executor
            self._running_executors[f"{executor_id}_subagents"] = subagent_executor

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

        # Handle duplicate sub_agent_call_ids by adding suffixes
        seen_sub_agent_call_ids: defaultdict[str, int] = defaultdict(int)
        for idx, sub_agent_call in enumerate(parsed_response.sub_agent_calls):
            base_id = sub_agent_call.sub_agent_call_id or f"sub_agent_call_{idx}"
            count = seen_sub_agent_call_ids[base_id]
            if count:
                sub_agent_call.sub_agent_call_id = f"{base_id}_{count}"
            else:
                sub_agent_call.sub_agent_call_id = base_id
            seen_sub_agent_call_ids[base_id] += 1

        serial_tool_names = set(self.serial_tool_name)

        try:
            # Submit tool execution tasks
            tool_futures: dict[Future[tuple[str, Any, bool]], tuple[str, ToolCall]] = {}
            for tool_call in parsed_response.tool_calls:
                task_ctx = copy_context()
                future = tool_executor.submit(
                    task_ctx.run,
                    self._execute_tool_call_safe,
                    tool_call,
                    agent_state,
                )
                tool_futures[future] = ("tool", tool_call)

                if tool_call.tool_name in serial_tool_names:
                    future.result()

            # Submit sub-agent execution tasks
            sub_agent_futures: dict[Future[tuple[str, str, bool]], tuple[str, SubAgentCall]] = {}
            for sub_agent_call in parsed_response.sub_agent_calls:
                task_ctx = copy_context()
                future = subagent_executor.submit(
                    task_ctx.run,
                    self._execute_sub_agent_call_safe,
                    sub_agent_call,
                    context_dict,
                    parent_agent_state=agent_state,
                    custom_llm_client_provider=custom_llm_client_provider,
                )
                sub_agent_futures[future] = ("sub_agent", sub_agent_call)

            # Combine all futures
            all_futures: dict[Future[tuple[str, Any, bool]], tuple[str, ToolCall | SubAgentCall]] = {**tool_futures, **sub_agent_futures}

            # Collect results as they complete
            tool_results: list[str] = []
            execution_feedbacks: list[dict[str, Any]] = []
            stop_tool_detected = False
            stop_tool_result = None

            for future in as_completed(all_futures):
                call_type, call_obj = all_futures[future]
                try:
                    result_data = future.result()
                    if call_type == "sub_agent" and isinstance(call_obj, SubAgentCall):
                        agent_name, result, is_error = result_data
                        result_str = result
                        execution_feedbacks.append(
                            {
                                "call_type": "sub_agent",
                                "call": call_obj,
                                "content": result_str,
                                "is_error": is_error,
                            },
                        )
                        if is_error:
                            logger.error(
                                f"❌ Sub-agent '{agent_name}' error: {result}",
                            )
                            should_append_xml = not getattr(call_obj, "tool_call_id", None)
                            if should_append_xml:
                                tool_results.append(
                                    f"""
<tool_result>
<tool_name>{agent_name}_sub_agent</tool_name>
<error>{result}</error>
</tool_result>
""",
                                )
                        else:
                            logger.info(
                                f"📤 Sub-agent '{agent_name}' result: {result}",
                            )
                            should_append_xml = not getattr(call_obj, "tool_call_id", None)
                            if should_append_xml:
                                tool_results.append(
                                    f"""
<tool_result>
<tool_name>{agent_name}_sub_agent</tool_name>
<result>{result}</result>
</tool_result>
""",
                                )
                    elif not isinstance(call_obj, ToolCall):
                        logger.error(
                            f"❌ Unexpected call object type: {type(call_obj)} (call_type={call_type})",
                        )
                        continue
                    elif call_type == "tool":
                        tool_name, result, is_error = result_data
                        result_str = result if isinstance(result, str) else json.dumps(result, indent=2, ensure_ascii=False)
                        execution_feedbacks.append(
                            {
                                "call_type": "tool",
                                "call": call_obj,
                                "content": result_str,
                                "output": result,
                                "is_error": is_error,
                            },
                        )
                        if is_error:
                            logger.error(
                                f"❌ Tool '{tool_name}' error: {result_str}",
                            )
                            should_append_xml = getattr(call_obj, "source", "xml") != "openai"
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
                            should_append_xml = getattr(call_obj, "source", "xml") != "openai"
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
                                parsed_result = json.loads(result_str)
                                if isinstance(parsed_result, dict):
                                    parsed_result_dict = cast(dict[str, Any], parsed_result)
                                    if parsed_result_dict.get("_is_stop_tool"):
                                        stop_tool_detected = True
                                        actual_result: dict[str, Any] = {
                                            key: value for key, value in parsed_result_dict.items() if key != "_is_stop_tool"
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
                                                    parsed_result,
                                                    ensure_ascii=False,
                                                    indent=4,
                                                )
                                            )
                                        logger.info(
                                            f"🛑 Stop tool '{tool_name}' result detected, will terminate after processing",
                                        )
                            except (json.JSONDecodeError, TypeError):
                                pass
                    else:
                        logger.error(
                            f"❌ Unexpected call type '{call_type}' for object {type(call_obj)}",
                        )
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
                    subagent_executor.shutdown(wait=True, cancel_futures=False)
                except Exception as e:
                    logger.error(f"❌ Error shutting down executors: {e}")
                finally:
                    self._running_executors.pop(f"{executor_id}_tools", None)
                    self._running_executors.pop(
                        f"{executor_id}_subagents",
                        None,
                    )

        return processed_response, stop_tool_detected, stop_tool_result, execution_feedbacks

    def _execute_tool_call_safe(
        self,
        tool_call: ToolCall,
        agent_state: "AgentState",
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
            result = self.tool_executor.execute_tool(
                agent_state,
                self.sandbox_manager.instance,
                tool_call.tool_name,
                converted_params,
                tool_call_id=tool_call_id,
            )

            return (tool_call.tool_name, result, False)

        except Exception as e:
            return tool_call.tool_name, str(e), True

    def _execute_sub_agent_call_safe(
        self,
        sub_agent_call: SubAgentCall,
        context: dict[str, Any] | None = None,
        parent_agent_state: AgentState | None = None,
        *,
        custom_llm_client_provider: Callable[[str], Any] | None = None,
    ) -> tuple[str, str, bool]:
        """Safely execute a sub-agent call."""
        try:
            result = self.subagent_manager.call_sub_agent(
                sub_agent_call.agent_name,
                sub_agent_call.message,
                context=context,
                parent_agent_state=parent_agent_state,
                custom_llm_client_provider=custom_llm_client_provider,
            )

            return sub_agent_call.agent_name, result, False

        except Exception as e:
            return sub_agent_call.agent_name, str(e), True

    def _execute_batch_call(self, batch_call: BatchAgentCall) -> str:
        """Execute a batch agent call."""
        return self.batch_processor.process_batch_data(
            batch_call.agent_name,
            batch_call.file_path,
            batch_call.data_format,
            batch_call.message_template,
        )

    def cleanup(self) -> None:
        """Clean up executor resources."""
        logger.info(f"🧹 Cleaning up executor for agent '{self.agent_name}'...")
        self.stop_signal = True

        # Signal shutdown to prevent new tasks
        self._shutdown_event.set()

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

    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the executor.

        Args:
            tool: Tool instance to add
        """
        with self._tool_registry_lock:
            # Keep registry and structured payload updates atomic for concurrent readers/writers
            self.tool_executor.tool_registry[tool.name] = tool
            if self.tool_call_mode == "openai":
                openai_tools = cast(list[ChatCompletionToolParam], self.structured_tool_payload)
                openai_tools.append(tool.to_openai())
            else:
                anthropic_tools = cast(list[ToolParam], self.structured_tool_payload)
                anthropic_tools.append(tool.to_anthropic())

    def add_sub_agent(self, name: str, agent_config: AgentConfig) -> None:
        """Add a sub-agent config.

        Args:
            name: Name of the sub-agent
            agent_config: Config creates the agent
        """
        self.subagent_manager.add_sub_agent(name, agent_config)
