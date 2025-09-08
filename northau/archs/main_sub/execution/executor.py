"""Main execution orchestrator for agents."""

import json
import threading
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, Callable, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..agent_state import AgentState

from .tool_executor import ToolExecutor
from .subagent_manager import SubAgentManager
from .batch_processor import BatchProcessor
from .llm_caller import LLMCaller
from .response_parser import ResponseParser
from .parse_structures import ParsedResponse, ToolCall, SubAgentCall, BatchAgentCall
from .hooks import HookManager, AfterModelHook, ToolHookManager, AfterToolHook, AfterModelHookInput
from ..tracing.tracer import Tracer
from ..tracing.trace_dumper import TraceDumper
from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class Executor:
    """Orchestrates execution of agent tasks with parallel processing support."""
    
    def __init__(self, agent_name: str, agent_id: str, tool_registry: dict[str, Any], sub_agent_factories: dict[str, Callable[[], Any]],
                 stop_tools: set[str], openai_client: Any, llm_config: Any, max_iterations: int = 100,
                 max_context_tokens: int = 128000, max_running_subagents: int = 5, 
                 retry_attempts: int = 5, token_counter: TokenCounter | None = None,
                 langfuse_client: Any = None, after_model_hooks: list[AfterModelHook] | None = None,
                 after_tool_hooks: list[AfterToolHook] | None = None,
                 global_storage: Any = None, custom_llm_generator: Callable[[Any, dict[str, Any]], Any] | None = None):
        """Initialize executor.
        
        Args:
            agent_name: Name of the agent
            agent_id: ID of the agent
            tool_registry: Dictionary of available tools
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
            after_model_hooks: Optional list of hooks called after parsing LLM response
            after_tool_hooks: Optional list of hooks called after tool execution
            custom_llm_generator: Optional custom LLM generator function
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.max_running_subagents = max_running_subagents
        
        # Initialize components
        tool_hook_manager = ToolHookManager(after_tool_hooks) if after_tool_hooks else None
        self.tool_executor = ToolExecutor(tool_registry, stop_tools, langfuse_client, tool_hook_manager)
        self.tracer = Tracer(agent_name)
        self.subagent_manager = SubAgentManager(agent_name, sub_agent_factories, langfuse_client, global_storage, self.tracer)
        self.batch_processor = BatchProcessor(self.subagent_manager, max_running_subagents)
        self.response_parser = ResponseParser()
        self.llm_caller = LLMCaller(openai_client, llm_config, retry_attempts, custom_llm_generator)
        self.hook_manager = HookManager(after_model_hooks)
        
        # Execution parameters
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.global_storage = global_storage
        
        # Token counting
        self.token_counter = token_counter or TokenCounter()
        
        # Process tracking for parallel execution
        self._running_executors = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Message queue for dynamic message enqueueing during execution
        self.queued_messages = []
    
    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be processed during execution.
        
        Args:
            message: Message dictionary with 'role' and 'content' keys
        """
        self.queued_messages.append(message)
        logger.info(f"📝 Message enqueued during execution: {message.get('role', 'unknown')} - {message.get('content', '')[:50]}...")
    
    def execute(self, history: list[dict[str, str]], agent_state: 'AgentState',
               dump_trace_path: str | None = None) -> tuple[str, list[dict[str, str]]]:
        """Execute agent task with full orchestration.
        
        Args:
            history: Complete conversation history including system prompt and user message
            agent_state: AgentState containing agent context and global storage
            dump_trace_path: Optional path to dump execution trace
            
        Returns:
            Tuple of (agent_response, updated_messages_history)
        """
        # Initialize tracing if requested
        if dump_trace_path:
            self.tracer.start_tracing(dump_trace_path)
        
        try:
            # Use history directly as the single source of truth
            messages = history.copy()
            
            # Loop until no more tool calls or sub-agent calls are made
            iteration = 0
            final_response = ""
            
            logger.info(f"🔄 Starting iterative execution loop for agent '{self.agent_name}'")
            
            while iteration < self.max_iterations:
                logger.info(f"🔄 Iteration {iteration + 1}/{self.max_iterations} for agent '{self.agent_name}'")
                
                # Process any queued messages
                if self.queued_messages:
                    logger.info(f"📝 Processing {len(self.queued_messages)} queued messages")
                    messages.extend(self.queued_messages)
                    self.queued_messages = []
                
                # Count current prompt tokens
                current_prompt_tokens = self.token_counter.count_tokens(messages)
                
                force_stop_reason = None
                # Check if prompt exceeds max context tokens - force stop if so
                if current_prompt_tokens > self.max_context_tokens:
                    logger.error(f"❌ Prompt tokens ({current_prompt_tokens}) exceed max_context_tokens ({self.max_context_tokens}). Stopping execution.")
                    final_response += f"\\n\\n[Error: Prompt too long ({current_prompt_tokens} tokens) exceeds maximum context ({self.max_context_tokens} tokens). Execution stopped.]"
                    force_stop_reason = "Prompt too long"
                
                # Calculate max_tokens dynamically based on available budget
                available_tokens = self.max_context_tokens - current_prompt_tokens
                
                # Get desired max_tokens from LLM config or use reasonable default
                desired_max_tokens = 4096  # Default value
                calculated_max_tokens = min(desired_max_tokens, available_tokens)
                
                # Ensure we have at least some tokens for response
                if calculated_max_tokens < 50:
                    logger.error(f"❌ Insufficient tokens for response ({calculated_max_tokens}). Stopping execution.")
                    final_response += f"\\n\\n[Error: Insufficient tokens for response ({calculated_max_tokens} tokens). Context too full.]"
                    force_stop_reason = "Insufficient tokens"
                
                logger.info(f"🔢 Token usage: prompt={current_prompt_tokens}, max_tokens={calculated_max_tokens}, available={available_tokens}")
                
                # Log LLM request to trace if enabled
                if dump_trace_path:
                    self.tracer.add_llm_request(iteration + 1, {
                        'messages': messages,
                        'max_tokens': calculated_max_tokens
                    })
                
                # Call LLM to get response
                logger.info(f"🧠 Calling LLM for agent '{self.agent_name}' with {calculated_max_tokens} max tokens...")
                assistant_response = self.llm_caller.call_llm(messages, calculated_max_tokens, force_stop_reason)
                
                # Log LLM response to trace if enabled
                if dump_trace_path:
                    # Create a mock response object for tracing compatibility
                    mock_response = type('MockResponse', (), {
                        'choices': [type('Choice', (), {
                            'message': type('Message', (), {'content': assistant_response})()
                        })()]
                    })()
                    self.tracer.add_llm_response(iteration + 1, mock_response)
                
                logger.info(f"💬 LLM Response for agent '{self.agent_name}': {assistant_response}")
                
                # Store this as the latest response (potential final response)
                final_response = assistant_response
                
                # Parse response to check for actions
                parsed_response = self.response_parser.parse_response(assistant_response)
                
                # Add the assistant's original response to conversation
                messages.append({"role": "assistant", "content": assistant_response})
                
                # Process tool calls and sub-agent calls
                logger.info(f"⚙️ Processing tool/sub-agent calls for agent '{self.agent_name}'...")
                after_model_hook_input = AfterModelHookInput(
                    agent_state=agent_state,
                    max_iterations=self.max_iterations,
                    current_iteration=iteration,
                    original_response=assistant_response,
                    parsed_response=parsed_response,
                    messages=messages
                )
                
                processed_response, should_stop, stop_tool_result, updated_messages = self._process_xml_calls(
                    after_model_hook_input, tracer=self.tracer if dump_trace_path else None
                )
                
                # Update messages with any modifications from hooks
                messages = updated_messages
                
                # Check if a stop tool was executed
                if should_stop and len(self.queued_messages) == 0:
                    # Return the stop tool result directly, formatted as JSON if it's not a string
                    if stop_tool_result is not None:
                        logger.info(f"🛑 Stop tool detected, returning stop tool result as final response")
                        if isinstance(stop_tool_result, str):
                            return stop_tool_result, messages
                        else:
                            import json
                            return json.dumps(stop_tool_result, indent=4, ensure_ascii=False), messages
                    else:
                        logger.info(f"🛑 No more tool calls, stop.")
                        # Fallback to the processed response if no specific result
                        return processed_response, messages
                
                # Extract just the tool results from processed_response
                tool_results = processed_response.replace(assistant_response, "").strip()
                
                # Add tool results as user feedback with iteration context
                remaining_iterations = self.max_iterations - iteration - 1
                iteration_hint = self._build_iteration_hint(iteration + 1, self.max_iterations, remaining_iterations)
                token_limit_hint = self._build_token_limit_hint(current_prompt_tokens, self.max_context_tokens, available_tokens, desired_max_tokens)
                
                if tool_results:
                    messages.append({
                        "role": "user", 
                        "content": f"Tool execution results:\\n{tool_results}\\n\\n{iteration_hint}\\n\\n{token_limit_hint}"
                    })
                else:
                    messages.append({
                        "role": "user", 
                        "content": f"{iteration_hint}\\n\\n{token_limit_hint}"
                    })
                
                iteration += 1
            
            # Add note if max iterations reached
            if iteration >= self.max_iterations:
                final_response += "\\n\\n[Note: Maximum iteration limit reached]"
            
            # Dump trace if enabled
            if dump_trace_path:
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(trace_data, dump_trace_path, self.agent_name)
            
            return final_response, messages
            
        except Exception as e:
            # Add error to trace if enabled
            if dump_trace_path:
                self.tracer.add_error(e)
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(trace_data, dump_trace_path, self.agent_name)
            # Re-raise with more context
            raise RuntimeError(f"Error in agent execution: {e}") from e
        finally:
            if dump_trace_path:
                self.tracer.stop_tracing()
    
    def _process_xml_calls(self, hook_input: AfterModelHookInput, tracer: Tracer | None = None) -> tuple[str, bool, str | None, list[dict[str, str]]]:
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
        parsed_response = self.response_parser.parse_response(hook_input.original_response)
        hook_input.parsed_response = parsed_response
        
        # Keep track of current messages (may be modified by hooks)
        current_messages = hook_input.messages.copy()
        
        # Execute hooks if any are configured (always run hooks, even if no calls)
        if self.hook_manager:
            try:
                logger.info(f"🎣 Executing {len(self.hook_manager)} after model hooks")
                parsed_response, current_messages = self.hook_manager.execute_hooks(hook_input)
            except Exception as e:
                logger.warning(f"⚠️ Hook execution failed: {e}")
                # Keep original parsed_response if hook fails
        
        # If no calls found after hooks, return original response
        if not parsed_response or not parsed_response.has_calls():
            return hook_input.original_response, True, None, current_messages
        
        # Phase 2: Execute all parsed calls
        logger.info(f"⚡ Phase 2: Executing {parsed_response.get_call_summary()}")
        processed_response, should_stop, stop_tool_result = self._execute_parsed_calls(parsed_response, hook_input.agent_state, tracer = tracer)
        return processed_response, should_stop, stop_tool_result, current_messages
    
    def _execute_parsed_calls(self, parsed_response: ParsedResponse, agent_state: 'AgentState', tracer: Optional[Tracer] = None) -> tuple[str, bool, str | None]:
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
            logger.warning(f"⚠️ Agent '{self.agent_name}' ({self.agent_id}) is shutting down, skipping new task execution")
            return processed_response, False, None
        
        # Handle batch agent calls first (they take priority and are not parallelized)
        if parsed_response.batch_agent_calls:
            for batch_call in parsed_response.batch_agent_calls:
                try:
                    batch_result = self._execute_batch_call(batch_call)
                    processed_response += f"\\n\\n<tool_result>\\n<tool_name>batch_agent</tool_name>\\n<result>{batch_result}</result>\\n</tool_result>"
                except Exception as e:
                    logger.error(f"❌ Batch agent call failed: {e}")
                    processed_response += f"\\n\\n<tool_result>\\n<tool_name>batch_agent</tool_name>\\n<error>{str(e)}</error>\\n</tool_result>"
            return processed_response, False, None
        
        # Execute tool calls and sub-agent calls in parallel
        if not parsed_response.tool_calls and not parsed_response.sub_agent_calls:
            return processed_response, False, None
        
        # Get current context to pass to sub-agents
        from ..agent_context import get_context
        current_context = get_context()
        context_dict = current_context.context.copy() if current_context else None
        
        executor_id = str(uuid.uuid4())
        tool_executor = ThreadPoolExecutor()
        subagent_executor = ThreadPoolExecutor(max_workers=self.max_running_subagents)
        
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
                sub_agent_call.sub_agent_call_id = f"{sub_agent_call.sub_agent_call_id}_{seen_sub_agent_call_ids[sub_agent_call.sub_agent_call_id]}"
        
        try:
            # Submit tool execution tasks
            tool_futures = {}
            for tool_call in parsed_response.tool_calls:
                task_ctx = copy_context()
                future = tool_executor.submit(task_ctx.run, self._execute_tool_call_safe, tool_call, agent_state, tracer)
                tool_futures[future] = ('tool', tool_call)
            
            # Submit sub-agent execution tasks
            sub_agent_futures = {}
            for sub_agent_call in parsed_response.sub_agent_calls:
                task_ctx = copy_context()
                future = subagent_executor.submit(task_ctx.run, self._execute_sub_agent_call_safe, sub_agent_call, context_dict, tracer)
                sub_agent_futures[future] = ('sub_agent', sub_agent_call)
            
            # Combine all futures
            all_futures = {**tool_futures, **sub_agent_futures}
            
            # Collect results as they complete
            tool_results = []
            stop_tool_detected = False
            stop_tool_result = None
            
            for future in as_completed(all_futures):
                call_type, call_obj = all_futures[future]
                try:
                    result_data = future.result()
                    if call_type == 'tool':
                        tool_name, result, is_error = result_data
                        if is_error:
                            logger.error(f"❌ Tool '{tool_name}' error: {result}")
                            tool_results.append(f"<tool_result>\\n<tool_name>{tool_name}</tool_name>\\n<error>{result}</error>\\n</tool_result>")
                        else:
                            logger.info(f"📤 Tool '{tool_name}' result: {result}")
                            tool_result_xml = f"<tool_result>\\n<tool_name>{tool_name}</tool_name>\\n<result>{result}</result>\\n</tool_result>"
                            tool_results.append(tool_result_xml)
                            
                            # Check if this tool result indicates a stop tool was executed
                            try:
                                parsed_result = json.loads(result)
                                if isinstance(parsed_result, dict) and parsed_result.get('_is_stop_tool'):
                                    stop_tool_detected = True
                                    actual_result = {k: v for k, v in parsed_result.items() if k != '_is_stop_tool'}
                                    if 'result' in actual_result and len(actual_result) == 1:
                                        stop_tool_result = str(actual_result['result'])
                                    else:
                                        stop_tool_result = str(actual_result) if actual_result else str(parsed_result)
                                    logger.info(f"🛑 Stop tool '{tool_name}' result detected, will terminate after processing")
                            except (json.JSONDecodeError, TypeError):
                                pass
                    elif call_type == 'sub_agent':
                        agent_name, result, is_error = result_data
                        if is_error:
                            logger.error(f"❌ Sub-agent '{agent_name}' error: {result}")
                            tool_results.append(f"<tool_result>\\n<tool_name>{agent_name}_sub_agent</tool_name>\\n<error>{result}</error>\\n</tool_result>")
                        else:
                            logger.info(f"📤 Sub-agent '{agent_name}' result: {result}")
                            tool_results.append(f"<tool_result>\\n<tool_name>{agent_name}_sub_agent</tool_name>\\n<result>{result}</result>\\n</tool_result>")
                except Exception as e:
                    logger.error(f"❌ Unexpected error processing {call_type}: {e}")
                    tool_results.append(f"<tool_result>\\n<tool_name>unknown</tool_name>\\n<error>Unexpected error: {str(e)}</error>\\n</tool_result>")
            
            # Append tool results to the original response
            if tool_results:
                processed_response += "\\n\\n" + "\\n\\n".join(tool_results)
        
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
                    self._running_executors.pop(f"{executor_id}_subagents", None)
        
        return processed_response, stop_tool_detected, stop_tool_result
    
    def _execute_tool_call_safe(self, tool_call: ToolCall, agent_state: 'AgentState', tracer: Optional[Tracer] = None) -> tuple[str, str, bool]:
        """Safely execute a tool call."""
        try:
            # Log tool request to trace if enabled
            if tracer:
                tracer.add_tool_request(tool_call.tool_name, tool_call.parameters)
            
            # Convert parameters to correct types and execute
            converted_params = {}
            for param_name, param_value in tool_call.parameters.items():
                converted_params[param_name] = self.tool_executor._convert_parameter_type(
                    tool_call.tool_name, param_name, param_value
                )
            
            result = self.tool_executor.execute_tool(agent_state, tool_call.tool_name, converted_params)
            
            # Log tool response to trace if enabled
            if tracer:
                tracer.add_tool_response(tool_call.tool_name, result)
            
            return tool_call.tool_name, json.dumps(result, indent=2, ensure_ascii=False), False
            
        except Exception as e:
            return tool_call.tool_name, str(e), True
    
    def _execute_sub_agent_call_safe(self, sub_agent_call: SubAgentCall, context: Optional[dict[str, Any]] = None, tracer: Tracer | None = None) -> tuple[str, str, bool]:
        """Safely execute a sub-agent call."""
        try:
            # Log sub-agent request to trace if enabled
            if tracer:
                tracer.add_subagent_request(sub_agent_call.agent_name, sub_agent_call.message)
            
            result = self.subagent_manager.call_sub_agent(sub_agent_call.agent_name, sub_agent_call.message, context)
            
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
            batch_call.message_template
        )
    
    def cleanup(self) -> None:
        """Clean up executor resources."""
        logger.info(f"🧹 Cleaning up executor for agent '{self.agent_name}'...")
        
        # Save trace data if available before cleanup
        if self.tracer.is_tracing():
            self.tracer.add_shutdown("Executor cleanup")
            trace_data = self.tracer.get_trace_data()
            dump_path = self.tracer.get_dump_path()
            if trace_data and dump_path:
                TraceDumper.dump_trace_to_file(trace_data, dump_path, self.agent_name)
        
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
                    logger.error(f"❌ Error shutting down executor {executor_id}: {e}")
            
            self._running_executors.clear()
        
        logger.info(f"✅ Executor cleanup completed for agent '{self.agent_name}'")
    
    def _build_iteration_hint(self, current_iteration: int, max_iterations: int, remaining_iterations: int) -> str:
        """Build a hint message for the LLM about iteration status."""
        if remaining_iterations <= 1:
            return (f"⚠️ WARNING: This is iteration {current_iteration}/{max_iterations}. "
                   f"You have only {remaining_iterations} iteration(s) remaining. "
                   f"Please provide a conclusive response and avoid making additional tool calls or sub-agent calls "
                   f"unless absolutely critical. Focus on summarizing your findings and providing final recommendations.")
        elif remaining_iterations <= 3:
            return (f"🔄 Iteration {current_iteration}/{max_iterations} - {remaining_iterations} iterations remaining. "
                   f"Please be mindful of the remaining steps and work towards a conclusion.")
        else:
            return (f"🔄 Iteration {current_iteration}/{max_iterations} - Continue your response if you have more to say, "
                   f"or if you need to make additional tool calls or sub-agent calls.")
    
    def _build_token_limit_hint(self, current_prompt_tokens: int, max_tokens: int, remaining_tokens: int, desired_max_tokens: int) -> str:
        """Build a hint message for the LLM about token limit."""
        if remaining_tokens < 3 * desired_max_tokens:
            return (f"⚠️ WARNING: Token usage is approaching the limit {current_prompt_tokens}/{max_tokens}."
                   f"You have only {remaining_tokens} tokens left."
                   f"Please be mindful of the token limit and avoid making additional tool calls or sub-agent calls "
                   f"unless absolutely critical. Focus on summarizing your findings and providing final recommendations.")
        else:
            return (f"🔄 Token Usage: {current_prompt_tokens}/{max_tokens} in the current prompt - {remaining_tokens} tokens left."
                    f"Continue your response if you have more to say, or if you need to make additional tool calls or sub-agent calls.")
    
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