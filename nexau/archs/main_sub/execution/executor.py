"""Main execution orchestrator for agents."""

import json
import logging
import threading
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from copy import deepcopy
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.batch_processor import BatchProcessor
from nexau.archs.main_sub.execution.hooks import (
    AfterModelHook,
    AfterModelHookInput,
    AfterToolHook,
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
from nexau.archs.main_sub.tracing.trace_dumper import TraceDumper
from nexau.archs.main_sub.tracing.tracer import Tracer
from nexau.archs.main_sub.utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class Executor:
    """Orchestrates execution of agent tasks with parallel processing support."""

    def __init__(
        self,
        agent_name: str,
        agent_id: str,
        tool_registry: dict[str, Any],
        sub_agent_factories: dict[str, Callable[[], Any]],
        stop_tools: set[str],
        openai_client: Any,
        llm_config: Any,
        max_iterations: int = 100,
        max_context_tokens: int = 128000,
        max_running_subagents: int = 5,
        retry_attempts: int = 5,
        token_counter: TokenCounter | None = None,
        langfuse_client: Any = None,
        after_model_hooks: list[AfterModelHook] | None = None,
        before_model_hooks: list[BeforeModelHook] | None = None,
        after_tool_hooks: list[AfterToolHook] | None = None,
        before_tool_hooks: list[BeforeToolHook] | None = None,
        middlewares: list[Middleware] | None = None,
        serial_tool_name: list[str] | None = None,
        global_storage: Any = None,
        tool_call_mode: str = "openai",
        openai_tools: list[dict[str, Any]] | None = None,
    ):
        """Initialize executor.

        Args:
            agent_name: Name of the agent
            agent_id: ID of the agent
            tool_registry: Dictionary of available tools
            serial_tool_name: List of tool names that should be executed serially
            sub_agent_factories: Dictionary of sub-agent factories
            stop_tools: Set of tool names that trigger execution stop
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            max_iterations: Maximum iterations per execution
            max_context_tokens: Maximum context token limit
            max_running_subagents: Maximum concurrent sub-agents
            retry_attempts: int of API retry attempts
            token_counter: Optional token counter instance
            langfuse_client: Optional Langfuse client for tracing
            before_model_hooks: Optional list of hooks called before parsing LLM response
            after_model_hooks: Optional list of hooks called after parsing LLM response
            before_tool_hooks: Optional list of hooks called before tool execution
            after_tool_hooks: Optional list of hooks called after tool execution
            middlewares: Optional list of middleware objects applied to all phases
            tool_call_mode: Preferred tool call format ('xml', 'openai', or 'anthorpic')
            openai_tools: Structured tool definitions for OpenAI/Anthorpic tool calls
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
        self.tool_executor = ToolExecutor(
            tool_registry,
            stop_tools,
            langfuse_client,
            middleware_manager=self.middleware_manager,
        )
        self.tracer = Tracer(agent_name)
        self.subagent_manager = SubAgentManager(
            agent_name,
            sub_agent_factories,
            langfuse_client,
            global_storage,
            self.tracer,
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
        self.structured_tool_payload = deepcopy(openai_tools) if openai_tools else []
        if self.use_structured_tool_calls and not self.structured_tool_payload:
            logger.warning(
                f"⚠️ {self.tool_call_mode.capitalize()} tool call mode enabled but no tool definitions were provided.",
            )

        # Process tracking for parallel execution
        self._running_executors = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.stop_signal = False

        # Message queue for dynamic message enqueueing during execution
        self.queued_messages = []

    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be processed during execution.

        Args:
            message: Message dictionary with 'role' and 'content' keys
        """
        self.queued_messages.append(message)
        logger.info(
            f"📝 Message enqueued during execution: {message.get('role', 'unknown')} - {message.get('content', '')[:50]}...",
        )

    def execute(
        self,
        history: list[dict[str, Any]],
        agent_state: "AgentState",
        dump_trace_path: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Execute agent task with full orchestration.

        Args:
            history: Complete conversation history including system prompt and user message
            agent_state: AgentState containing agent context and global storage
            dump_trace_path: Optional path to dump execution trace

        Returns:
            Tuple of (agent_response, updated_messages_history)
        """
        # Reset the stop signal
        self.stop_signal = False

        # Initialize tracing if requested
        if dump_trace_path:
            self.tracer.start_tracing(dump_trace_path)

        try:
            # Use history directly as the single source of truth
            messages = history.copy()

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
                    return "Stop signal received.", messages

                # Process any queued messages
                if self.queued_messages:
                    logger.info(
                        f"📝 Processing {len(self.queued_messages)} queued messages",
                    )
                    messages.extend(self.queued_messages)
                    self.queued_messages = []

                # Count current prompt tokens
                current_prompt_tokens = self.token_counter.count_tokens(
                    messages,
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

                # Log LLM request to trace if enabled
                if dump_trace_path:
                    self.tracer.add_llm_request(
                        iteration + 1,
                        {
                            "messages": messages,
                            "max_tokens": calculated_max_tokens,
                        },
                    )

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

                # Call LLM to get response
                logger.info(
                    f"🧠 Calling LLM for agent '{self.agent_name}' with {calculated_max_tokens} max tokens...",
                )
                model_response = self.llm_caller.call_llm(
                    messages,
                    calculated_max_tokens,
                    force_stop_reason,
                    agent_state,
                    tool_call_mode=self.tool_call_mode,
                    tools=self.structured_tool_payload if self.use_structured_tool_calls else None,
                )
                if model_response is None:
                    break

                assistant_content = model_response.content or ""
                assistant_log_text = model_response.render_text() or assistant_content

                # Log LLM response to trace if enabled
                if dump_trace_path:
                    # Create a mock response object for tracing compatibility
                    self.tracer.add_llm_response(
                        iteration + 1,
                        assistant_log_text,
                    )

                logger.info(
                    f"💬 LLM Response for agent '{self.agent_name}': {assistant_log_text}",
                )

                # Store this as the latest response (potential final response)
                final_response = assistant_content

                # Parse response to check for actions
                parsed_response = self.response_parser.parse_response(
                    model_response,
                )

                # Add the assistant's original response to conversation
                messages.append(model_response.to_message_dict())

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
                    tracer=self.tracer if dump_trace_path else None,
                )

                # Update messages with any modifications from hooks
                messages = updated_messages

                processed_parsed_response = after_model_hook_input.parsed_response

                # Check if a stop tool was executed
                if should_stop and len(self.queued_messages) == 0:
                    # Return the stop tool result directly, formatted as JSON if it's not a string
                    if stop_tool_result is not None:
                        logger.info(
                            "🛑 Stop tool detected, returning stop tool result as final response",
                        )
                        force_stop_reason = AgentStopReason.STOP_TOOL_TRIGGERED
                        if isinstance(stop_tool_result, str):
                            final_response = stop_tool_result
                            break
                        else:
                            import json

                            final_response = json.dumps(
                                stop_tool_result,
                                indent=4,
                                ensure_ascii=False,
                            )
                            break
                    else:
                        logger.info("🛑 No more tool calls, stop.")
                        force_stop_reason = AgentStopReason.NO_MORE_TOOL_CALLS
                        # Fallback to the processed response if no specific result
                        final_response = processed_response
                        break

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

                        if call_type == "tool":
                            call_id = getattr(call_obj, "tool_call_id", None)
                        elif call_type == "sub_agent":
                            call_id = getattr(call_obj, "tool_call_id", None) or getattr(call_obj, "sub_agent_call_id", None)
                        else:
                            call_id = None

                        if not call_id:
                            continue

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": content,
                            },
                        )

                    tool_results = ""
                else:
                    tool_results = processed_response.replace(
                        assistant_content,
                        "",
                        1,
                    ).strip()

                # Add tool results as user feedback with iteration context
                remaining_iterations = self.max_iterations - iteration - 1
                iteration_hint = self._build_iteration_hint(
                    iteration + 1,
                    self.max_iterations,
                    remaining_iterations,
                )

                if tool_results:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"Tool execution results:\n{tool_results}\n\n{iteration_hint}",
                        },
                    )
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{iteration_hint}",
                        },
                    )
                current_prompt_tokens = self.token_counter.count_tokens(
                    messages,
                )

                token_limit_hint = self._build_token_limit_hint(
                    current_prompt_tokens,
                    self.max_context_tokens,
                    available_tokens,
                    desired_max_tokens,
                )
                messages[-1]["content"] += f"\n\n{token_limit_hint}"

                iteration += 1

            # Add note if max iterations reached
            if iteration >= self.max_iterations:
                force_stop_reason = AgentStopReason.MAX_ITERATIONS_REACHED
                final_response += "\\n\\n[Note: Maximum iteration limit reached]"

            logger.info(
                f"🔄 Force stop reason: {force_stop_reason.name}",
            )
            logger.info(
                f"🔄 Final response for agent '{self.agent_name}': {final_response}",
            )
            # Dump trace if enabled
            if dump_trace_path:
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(
                        trace_data,
                        dump_trace_path,
                        self.agent_name,
                    )

            return final_response, messages

        except Exception as e:
            # Add error to trace if enabled
            force_stop_reason = AgentStopReason.ERROR_OCCURRED
            final_response = f"Error: {str(e)}"
            logger.error(
                f"🔄 Force stop reason: {force_stop_reason.name}",
            )
            logger.error(
                f"🔄 Final response for agent '{self.agent_name}': {final_response}",
            )
            logger.error(
                f"❌ Error in agent execution: {e}",
            )
            if dump_trace_path:
                self.tracer.add_error(e)
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(
                        trace_data,
                        dump_trace_path,
                        self.agent_name,
                    )
            # Re-raise with more context
            raise RuntimeError(f"Error in agent execution: {e}") from e
        finally:
            if dump_trace_path:
                self.tracer.stop_tracing()

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

        for hook in before_model_hooks:
            combined.append(
                FunctionMiddleware(
                    before_model_hook=hook,
                    name=f"before_model::{_hook_name(hook)}",
                ),
            )

        for hook in after_model_hooks:
            combined.append(
                FunctionMiddleware(
                    after_model_hook=hook,
                    name=f"after_model::{_hook_name(hook)}",
                ),
            )

        for hook in after_tool_hooks:
            combined.append(
                FunctionMiddleware(
                    after_tool_hook=hook,
                    name=f"after_tool::{_hook_name(hook)}",
                ),
            )

        for hook in before_tool_hooks:
            combined.append(
                FunctionMiddleware(
                    before_tool_hook=hook,
                    name=f"before_tool::{_hook_name(hook)}",
                ),
            )

        return MiddlewareManager(combined)

    def _process_xml_calls(
        self,
        hook_input: AfterModelHookInput,
        tracer: Tracer | None = None,
    ) -> tuple[str, bool, str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Process XML tool calls and sub-agent calls using two-phase approach.

        Args:
            response: Agent response containing XML calls
            messages: Current conversation history
            tracer: Optional tracer for logging

        Returns:
            Tuple of (processed_response, should_stop, stop_tool_result, updated_messages)
        """
        # Phase 1: Parse the response to extract all calls
        logger.info("📋 Phase 1: Parsing LLM response for all executable calls")
        response_payload: str | ModelResponse = hook_input.model_response or hook_input.original_response
        parsed_response = hook_input.parsed_response or self.response_parser.parse_response(
            response_payload,
        )
        hook_input.parsed_response = parsed_response

        # Keep track of current messages (may be modified by hooks)
        current_messages = hook_input.messages.copy()
        force_continue = False  # Default: don't force continue

        # Execute middlewares if any are configured (always run even if no calls)
        if self.middleware_manager:
            try:
                parsed_response, current_messages, force_continue = self.middleware_manager.run_after_model(
                    hook_input,
                )
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
        processed_response, should_stop, stop_tool_result, execution_feedbacks = self._execute_parsed_calls(
            parsed_response,
            hook_input.agent_state,
            tracer=tracer,
        )
        return processed_response, should_stop, stop_tool_result, current_messages, execution_feedbacks

    def _execute_parsed_calls(
        self,
        parsed_response: ParsedResponse,
        agent_state: "AgentState",
        tracer: Tracer | None = None,
    ) -> tuple[str, bool, str | None]:
        """Execute all parsed calls in parallel.

        Args:
            parsed_response: ParsedResponse containing all calls to execute
            agent_state: AgentState containing agent context and global storage
            tracer: Optional tracer for logging

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
        seen_tool_call_ids = defaultdict(int)
        for tool_call in parsed_response.tool_calls:
            if tool_call.tool_call_id in seen_tool_call_ids:
                seen_tool_call_ids[tool_call.tool_call_id] += 1
                tool_call.tool_call_id = f"{tool_call.tool_call_id}_{seen_tool_call_ids[tool_call.tool_call_id]}"

        # Handle duplicate sub_agent_call_ids by adding suffixes
        seen_sub_agent_call_ids = defaultdict(int)
        for sub_agent_call in parsed_response.sub_agent_calls:
            if sub_agent_call.sub_agent_call_id in seen_sub_agent_call_ids:
                seen_sub_agent_call_ids[sub_agent_call.sub_agent_call_id] += 1
                sub_agent_call.sub_agent_call_id = (
                    f"{sub_agent_call.sub_agent_call_id}_{seen_sub_agent_call_ids[sub_agent_call.sub_agent_call_id]}"
                )

        serial_tool_names = set(self.serial_tool_name)

        try:
            # Submit tool execution tasks
            tool_futures = {}
            for tool_call in parsed_response.tool_calls:
                task_ctx = copy_context()
                future = tool_executor.submit(
                    task_ctx.run,
                    self._execute_tool_call_safe,
                    tool_call,
                    agent_state,
                    tracer,
                )
                tool_futures[future] = ("tool", tool_call)

                if tool_call.tool_name in serial_tool_names:
                    future.result()

            # Submit sub-agent execution tasks
            sub_agent_futures = {}
            for sub_agent_call in parsed_response.sub_agent_calls:
                task_ctx = copy_context()
                future = subagent_executor.submit(
                    task_ctx.run,
                    self._execute_sub_agent_call_safe,
                    sub_agent_call,
                    context_dict,
                    tracer,
                    parent_agent_state=agent_state,
                )
                sub_agent_futures[future] = ("sub_agent", sub_agent_call)

            # Combine all futures
            all_futures = {**tool_futures, **sub_agent_futures}

            # Collect results as they complete
            tool_results = []
            execution_feedbacks: list[dict[str, Any]] = []
            stop_tool_detected = False
            stop_tool_result = None

            for future in as_completed(all_futures):
                call_type, call_obj = all_futures[future]
                try:
                    result_data = future.result()
                    if call_type == "tool":
                        tool_name, result, is_error = result_data
                        execution_feedbacks.append(
                            {
                                "call_type": "tool",
                                "call": call_obj,
                                "content": result,
                                "is_error": is_error,
                            },
                        )
                        if is_error:
                            logger.error(
                                f"❌ Tool '{tool_name}' error: {result}",
                            )
                            should_append_xml = getattr(call_obj, "source", "xml") != "openai"
                            if should_append_xml:
                                tool_results.append(
                                    f"""
<tool_result>
<tool_name>{tool_name}</tool_name>
<error>{result}</error>
</tool_result>
""",
                                )
                        else:
                            logger.info(
                                f"📤 Tool '{tool_name}' result: {result}",
                            )
                            should_append_xml = getattr(call_obj, "source", "xml") != "openai"
                            tool_result_xml = f"""
<tool_result>
<tool_name>{tool_name}</tool_name>
<result>{result}</result>
</tool_result>
"""
                            if should_append_xml:
                                tool_results.append(tool_result_xml)

                            # Check if this tool result indicates a stop tool was executed
                            try:
                                parsed_result = json.loads(result)
                                if isinstance(
                                    parsed_result,
                                    dict,
                                ) and parsed_result.get("_is_stop_tool"):
                                    stop_tool_detected = True
                                    actual_result = {k: v for k, v in parsed_result.items() if k != "_is_stop_tool"}
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
                    elif call_type == "sub_agent":
                        agent_name, result, is_error = result_data
                        result_str = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
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
        tracer: Tracer | None = None,
    ) -> tuple[str, str, bool]:
        """Safely execute a tool call."""
        try:
            # Log tool request to trace if enabled
            if tracer:
                tracer.add_tool_request(
                    tool_call.tool_name,
                    tool_call.parameters,
                )

            # Convert parameters to correct types and execute
            converted_params = {}
            for param_name, param_value in tool_call.parameters.items():
                converted_params[param_name] = self.tool_executor._convert_parameter_type(
                    tool_call.tool_name,
                    param_name,
                    param_value,
                )

            result = self.tool_executor.execute_tool(
                agent_state, tool_call.tool_name, converted_params, tool_call_id=tool_call.tool_call_id
            )

            # Log tool response to trace if enabled
            if tracer:
                tracer.add_tool_response(tool_call.tool_name, result)

            return (
                tool_call.tool_name,
                json.dumps(result, indent=2, ensure_ascii=False),
                False,
            )

        except Exception as e:
            return tool_call.tool_name, str(e), True

    def _execute_sub_agent_call_safe(
        self,
        sub_agent_call: SubAgentCall,
        context: dict[str, Any] | None = None,
        tracer: Tracer | None = None,
        parent_agent_state: AgentState | None = None,
    ) -> tuple[str, str, bool]:
        """Safely execute a sub-agent call."""
        try:
            # Log sub-agent request to trace if enabled
            if tracer:
                tracer.add_subagent_request(
                    sub_agent_call.agent_name,
                    sub_agent_call.message,
                )

            result = self.subagent_manager.call_sub_agent(
                sub_agent_call.agent_name,
                sub_agent_call.message,
                context,
                parent_agent_state=parent_agent_state,
            )

            # Log sub-agent response to trace if enabled
            if tracer:
                tracer.add_subagent_response(sub_agent_call.agent_name, result)

            return sub_agent_call.agent_name, result, False

        except Exception as e:
            return sub_agent_call.agent_name, str(e), True

    def _execute_batch_call(self, batch_call: BatchAgentCall) -> str:
        """Execute a batch agent call."""
        return self.batch_processor._process_batch_data(
            batch_call.agent_name,
            batch_call.file_path,
            batch_call.data_format,
            batch_call.message_template,
        )

    def cleanup(self) -> None:
        """Clean up executor resources."""
        logger.info(f"🧹 Cleaning up executor for agent '{self.agent_name}'...")
        self.stop_signal = True

        # Save trace data if available before cleanup
        if self.tracer.is_tracing():
            self.tracer.add_shutdown("Executor cleanup")
            trace_data = self.tracer.get_trace_data()
            dump_path = self.tracer.get_dump_path()
            if trace_data and dump_path:
                TraceDumper.dump_trace_to_file(
                    trace_data,
                    dump_path,
                    self.agent_name,
                )

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

    def _build_iteration_hint(
        self,
        current_iteration: int,
        max_iterations: int,
        remaining_iterations: int,
    ) -> str:
        """Build a hint message for the LLM about iteration status."""
        if remaining_iterations <= 1:
            return (
                f"⚠️ WARNING: This is iteration {current_iteration}/{max_iterations}. "
                f"You have only {remaining_iterations} iteration(s) remaining. "
                f"Please provide a conclusive response and avoid making additional tool calls or sub-agent calls "
                f"unless absolutely critical. Focus on summarizing your findings and providing final recommendations."
            )
        elif remaining_iterations <= 3:
            return (
                f"🔄 Iteration {current_iteration}/{max_iterations} - {remaining_iterations} iterations remaining. "
                f"Please be mindful of the remaining steps and work towards a conclusion."
            )
        else:
            return (
                f"🔄 Iteration {current_iteration}/{max_iterations} - Continue your response if you have more to say, "
                f"or if you need to make additional tool calls or sub-agent calls."
            )

    def _build_token_limit_hint(
        self,
        current_prompt_tokens: int,
        max_tokens: int,
        remaining_tokens: int,
        desired_max_tokens: int,
    ) -> str:
        """Build a hint message for the LLM about token limit."""
        if remaining_tokens < 3 * desired_max_tokens:
            return (
                f"⚠️ WARNING: Token usage is approaching the limit {current_prompt_tokens}/{max_tokens}."
                f"You have only {remaining_tokens} tokens left."
                f"Please be mindful of the token limit and avoid making additional tool calls or sub-agent calls "
                f"unless absolutely critical. Focus on summarizing your findings and providing final recommendations."
            )
        else:
            return (
                f"🔄 Token Usage: {current_prompt_tokens}/{max_tokens} in the current prompt - {remaining_tokens} tokens left."
                f"Continue your response if you have more to say, or if you need to make additional tool calls or sub-agent calls."
            )

    def add_tool(self, tool: Any) -> None:
        """Add a tool to the executor.

        Args:
            tool: Tool instance to add
        """
        self.tool_executor.tool_registry[tool.name] = tool

    def add_sub_agent(self, name: str, agent_factory: Callable[[], Any]) -> None:
        """Add a sub-agent factory to the executor.

        Args:
            name: Name of the sub-agent
            agent_factory: Factory function that creates the agent
        """
        self.subagent_manager.add_sub_agent(name, agent_factory)
