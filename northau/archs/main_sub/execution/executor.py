"""Main execution orchestrator for agents."""

import json
import threading
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Any, Callable, Optional

from northau.archs.main_sub.agent_context import GlobalStorage

from .tool_executor import ToolExecutor
from .subagent_manager import SubAgentManager
from .batch_processor import BatchProcessor
from .response_generator import ResponseGenerator
from .response_parser import ResponseParser
from .parse_structures import ParsedResponse, ToolCall, SubAgentCall, BatchAgentCall
from .hooks import HookManager, AfterModelHook, AfterModelHookInput
from ..tracing.tracer import Tracer
from ..tracing.trace_dumper import TraceDumper
from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class Executor:
    """Orchestrates execution of agent tasks with parallel processing support."""
    
    def __init__(self, agent_name: str, tool_registry: dict[str, Any], sub_agent_factories: dict[str, Callable],
                 stop_tools: set[str], openai_client: Any, llm_config: Any, max_iterations: int = 100,
                 max_context_tokens: int = 128000, max_running_subagents: int = 5, 
                 retry_attempts: int = 5, token_counter: TokenCounter | None = None,
                 langfuse_client: Any = None, after_model_hooks: list[AfterModelHook] | None = None,
                 global_storage: Any = None):
        """Initialize executor.
        
        Args:
            agent_name: Name of the agent
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
        """
        self.agent_name = agent_name
        self.max_running_subagents = max_running_subagents
        
        # Initialize components
        self.tool_executor = ToolExecutor(tool_registry, stop_tools, langfuse_client)
        self.tracer = Tracer(agent_name)
        self.subagent_manager = SubAgentManager(agent_name, sub_agent_factories, langfuse_client, global_storage, self.tracer)
        self.batch_processor = BatchProcessor(self.subagent_manager, max_running_subagents)
        self.response_parser = ResponseParser()
        self.response_generator = ResponseGenerator(
            agent_name, openai_client, llm_config, max_iterations, 
            max_context_tokens, retry_attempts, global_storage
        )
        self.hook_manager = HookManager(after_model_hooks)
        
        # Token counting
        self.token_counter = token_counter or TokenCounter()
        
        # Process tracking for parallel execution
        self._running_executors = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
    
    def execute(self, history: list[dict[str, str]], 
               dump_trace_path: str | None = None) -> tuple[str, list[dict[str, str]]]:
        """Execute agent task with full orchestration.
        
        Args:
            history: Complete conversation history including system prompt and user message
            dump_trace_path: Optional path to dump execution trace
            
        Returns:
            Tuple of (agent_response, updated_messages_history)
        """
        # Initialize tracing if requested
        if dump_trace_path:
            self.tracer.start_tracing(dump_trace_path)
        
        try:
            # Generate response with tool/sub-agent execution
            response, updated_messages = self.response_generator.generate_response(
                history,
                self.token_counter.count_tokens,
                self._process_xml_calls,
                self.tracer if dump_trace_path else None
            )
            
            # Dump trace if enabled
            if dump_trace_path:
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(trace_data, dump_trace_path, self.agent_name)
            
            return response, updated_messages
            
        except Exception as e:
            # Add error to trace if enabled
            if dump_trace_path:
                self.tracer.add_error(e)
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(trace_data, dump_trace_path, self.agent_name)
            # Return original history even on error so calling code can still update
            raise
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
        
        # If no calls found after hooks, return original response
        if not parsed_response.has_calls():
            return hook_input.original_response, True, None, current_messages
        
        # Phase 2: Execute all parsed calls
        logger.info(f"⚡ Phase 2: Executing {parsed_response.get_call_summary()}")
        processed_response, should_stop, stop_tool_result = self._execute_parsed_calls(parsed_response, hook_input.global_storage, tracer = tracer)
        return processed_response, should_stop, stop_tool_result, current_messages
    
    def _execute_parsed_calls(self, parsed_response: ParsedResponse, global_storage: Optional[GlobalStorage] = None, tracer: Optional[Tracer] = None) -> tuple[str, bool, str | None]:
        """Execute all parsed calls in parallel.
        
        Args:
            parsed_response: ParsedResponse containing all calls to execute
            tracer: Optional tracer for logging
            
        Returns:
            Tuple of (processed_response, should_stop, stop_tool_result)
        """
        processed_response = parsed_response.original_response
        
        # Check if agent is shutting down
        if self._shutdown_event.is_set():
            logger.warning(f"⚠️ Agent '{self.agent_name}' is shutting down, skipping new task execution")
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
        
        executor_id = str(uuid.uuid4())
        tool_executor = ThreadPoolExecutor()
        subagent_executor = ThreadPoolExecutor(max_workers=self.max_running_subagents)
        
        # Track executors for cleanup
        with self._executor_lock:
            self._running_executors[f"{executor_id}_tools"] = tool_executor
            self._running_executors[f"{executor_id}_subagents"] = subagent_executor
        
        try:
            # Submit tool execution tasks
            tool_futures = {}
            for tool_call in parsed_response.tool_calls:
                task_ctx = copy_context()
                future = tool_executor.submit(task_ctx.run, self._execute_tool_call_safe, tool_call, global_storage, tracer)
                tool_futures[future] = ('tool', tool_call)
            
            # Submit sub-agent execution tasks
            sub_agent_futures = {}
            for sub_agent_call in parsed_response.sub_agent_calls:
                task_ctx = copy_context()
                future = subagent_executor.submit(task_ctx.run, self._execute_sub_agent_call_safe, sub_agent_call, tracer)
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
                                        stop_tool_result = actual_result['result']
                                    else:
                                        stop_tool_result = actual_result if actual_result else parsed_result
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
    
    def _execute_tool_call_safe(self, tool_call: ToolCall, global_storage: Optional[GlobalStorage] = None, tracer: Optional[Tracer] = None) -> tuple[str, str, bool]:
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
            
            if global_storage is not None:
                converted_params["global_storage"] = global_storage
            
            result = self.tool_executor.execute_tool(tool_call.tool_name, converted_params)
            
            # Log tool response to trace if enabled
            if tracer:
                tracer.add_tool_response(tool_call.tool_name, result)
            
            return tool_call.tool_name, json.dumps(result, indent=2, ensure_ascii=False), False
            
        except Exception as e:
            return tool_call.tool_name, str(e), True
    
    def _execute_sub_agent_call_safe(self, sub_agent_call: SubAgentCall, tracer: Tracer | None = None) -> tuple[str, str, bool]:
        """Safely execute a sub-agent call."""
        try:
            # Log sub-agent request to trace if enabled
            if tracer:
                tracer.add_subagent_request(sub_agent_call.agent_name, sub_agent_call.message)
            
            result = self.subagent_manager.call_sub_agent(sub_agent_call.agent_name, sub_agent_call.message)
            
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