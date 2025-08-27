"""Main execution orchestrator for agents."""

import json
import re
import threading
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from typing import Dict, Any, List, Optional, Tuple, Callable

from .tool_executor import ToolExecutor
from .subagent_manager import SubAgentManager
from .batch_processor import BatchProcessor
from .response_generator import ResponseGenerator
from ..tracing.tracer import Tracer
from ..tracing.trace_dumper import TraceDumper
from ..utils.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class Executor:
    """Orchestrates execution of agent tasks with parallel processing support."""
    
    def __init__(self, agent_name: str, tool_registry: Dict[str, Any], sub_agent_factories: Dict[str, Callable],
                 stop_tools: set, openai_client, llm_config, max_iterations: int = 100,
                 max_context_tokens: int = 128000, max_running_subagents: int = 5, 
                 retry_attempts: int = 3, token_counter: Optional[TokenCounter] = None,
                 langfuse_client=None):
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
            retry_attempts: Number of API retry attempts
            token_counter: Optional token counter instance
            langfuse_client: Optional Langfuse client for tracing
        """
        self.agent_name = agent_name
        self.max_running_subagents = max_running_subagents
        
        # Initialize components
        self.tool_executor = ToolExecutor(tool_registry, stop_tools, langfuse_client)
        self.subagent_manager = SubAgentManager(agent_name, sub_agent_factories, langfuse_client)
        self.batch_processor = BatchProcessor(self.subagent_manager, max_running_subagents)
        self.response_generator = ResponseGenerator(
            agent_name, openai_client, llm_config, max_iterations, 
            max_context_tokens, retry_attempts
        )
        self.tracer = Tracer(agent_name)
        
        # Token counting
        self.token_counter = token_counter or TokenCounter()
        
        # Process tracking for parallel execution
        self._running_executors = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
    
    def execute(self, message: str, system_prompt: str, history: List[Dict[str, str]], 
               dump_trace_path: Optional[str] = None) -> str:
        """Execute agent task with full orchestration.
        
        Args:
            message: User message
            system_prompt: System prompt to use
            history: Conversation history
            dump_trace_path: Optional path to dump execution trace
            
        Returns:
            Agent response
        """
        # Initialize tracing if requested
        if dump_trace_path:
            self.tracer.start_tracing(dump_trace_path)
        
        try:
            # Generate response with tool/sub-agent execution
            response = self.response_generator.generate_response(
                message, system_prompt, history,
                self.token_counter.count_tokens,
                self._process_xml_calls,
                self.tracer if dump_trace_path else None
            )
            
            # Dump trace if enabled
            if dump_trace_path:
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(trace_data, dump_trace_path, self.agent_name)
            
            return response
            
        except Exception as e:
            # Add error to trace if enabled
            if dump_trace_path:
                self.tracer.add_error(e)
                trace_data = self.tracer.get_trace_data()
                if trace_data:
                    TraceDumper.dump_trace_to_file(trace_data, dump_trace_path, self.agent_name)
            raise
        finally:
            if dump_trace_path:
                self.tracer.stop_tracing()
    
    def _process_xml_calls(self, response: str, tracer: Optional[Tracer] = None) -> Tuple[str, bool, Optional[str]]:
        """Process XML tool calls and sub-agent calls in the response in parallel.
        
        Args:
            response: Agent response containing XML calls
            tracer: Optional tracer for logging
            
        Returns:
            Tuple of (processed_response, should_stop, stop_tool_result)
        """
        processed_response = response
        
        # Check for parallel execution formats and batch processing first
        parallel_tool_calls_pattern = r'<use_parallel_tool_calls>(.*?)</use_parallel_tool_calls>'
        parallel_tool_calls_match = re.search(parallel_tool_calls_pattern, response, re.DOTALL)
        
        parallel_sub_agents_pattern = r'<use_parallel_sub_agents>(.*?)</use_parallel_sub_agents>'
        parallel_sub_agents_match = re.search(parallel_sub_agents_pattern, response, re.DOTALL)
        
        batch_agent_pattern = r'<use_batch_agent>(.*?)</use_batch_agent>'
        batch_agent_match = re.search(batch_agent_pattern, response, re.DOTALL)
        
        # Handle batch processing first
        if batch_agent_match:
            batch_result = self.batch_processor.execute_batch_agent_from_xml(batch_agent_match.group(1))
            processed_response += f"\\n\\n<tool_result>\\n<tool_name>batch_agent</tool_name>\\n<result>{batch_result}</result>\\n</tool_result>"
            return processed_response, False, None
        
        tool_matches = []
        sub_agent_matches = []
        
        if parallel_tool_calls_match:
            # Extract tool calls from within the parallel block using parallel_tool tags
            parallel_content = parallel_tool_calls_match.group(1)
            tool_pattern = r'<parallel_tool>(.*?)</parallel_tool>'
            tool_matches = re.findall(tool_pattern, parallel_content, re.DOTALL)
        elif parallel_sub_agents_match:
            # Extract both tool calls and sub-agent calls from within the parallel block
            parallel_content = parallel_sub_agents_match.group(1)
            tool_pattern = r'<parallel_tool>(.*?)</parallel_tool>'
            sub_agent_pattern = r'<parallel_agent>(.*?)</parallel_agent>'
            tool_matches = re.findall(tool_pattern, parallel_content, re.DOTALL)
            sub_agent_matches = re.findall(sub_agent_pattern, parallel_content, re.DOTALL)
        else:
            # Fall back to original behavior - find individual tool calls and sub-agent calls
            tool_pattern = r'<tool_use>(.*?)</tool_use>'
            tool_matches = re.findall(tool_pattern, response, re.DOTALL)
            
            sub_agent_pattern = r'<sub-agent>(.*?)</sub-agent>'
            sub_agent_matches = re.findall(sub_agent_pattern, response, re.DOTALL)
        
        # If no calls to process, return original response
        if not tool_matches and not sub_agent_matches:
            return processed_response, False, None
        
        # Check if agent is shutting down
        if self._shutdown_event.is_set():
            logger.warning(f"⚠️ Agent '{self.agent_name}' is shutting down, skipping new task execution")
            return processed_response, False, None
        
        # Execute all calls in parallel with separate thread pools for tools and sub-agents
        executor_id = str(uuid.uuid4())
        
        tool_executor = ThreadPoolExecutor()
        subagent_executor = ThreadPoolExecutor(max_workers=self.max_running_subagents)
        
        # Track executors for cleanup
        with self._executor_lock:
            self._running_executors[f"{executor_id}_tools"] = tool_executor
            self._running_executors[f"{executor_id}_subagents"] = subagent_executor
        
        try:
            # Submit tool execution tasks (no limit on concurrent tools)
            tool_futures = {}
            for tool_xml in tool_matches:
                # Propagate current tracing context into the worker thread
                task_ctx = copy_context()
                future = tool_executor.submit(task_ctx.run, self.tool_executor.execute_tool_from_xml_safe, tool_xml, tracer)
                tool_futures[future] = ('tool', tool_xml)
            
            # Submit sub-agent execution tasks (limited by max_running_subagents)
            sub_agent_futures = {}
            for sub_agent_xml in sub_agent_matches:
                # Propagate current tracing context into the worker thread
                task_ctx = copy_context()
                future = subagent_executor.submit(task_ctx.run, self.subagent_manager.execute_sub_agent_from_xml_safe, sub_agent_xml, tracer)
                sub_agent_futures[future] = ('sub_agent', sub_agent_xml)
            
            # Combine all futures
            all_futures = {**tool_futures, **sub_agent_futures}
            
            # Collect results as they complete
            tool_results = []
            stop_tool_detected = False
            stop_tool_result = None
            
            for future in as_completed(all_futures):
                call_type, xml_content = all_futures[future]
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
                                    # Extract the actual result, excluding the marker
                                    actual_result = {k: v for k, v in parsed_result.items() if k != '_is_stop_tool'}
                                    if 'result' in actual_result and len(actual_result) == 1:
                                        # If only 'result' key remains, use its value
                                        stop_tool_result = actual_result['result']
                                    else:
                                        # Otherwise use the full cleaned result
                                        stop_tool_result = actual_result if actual_result else parsed_result
                                    logger.info(f"🛑 Stop tool '{tool_name}' result detected, will terminate after processing")
                            except (json.JSONDecodeError, TypeError):
                                # Result is not JSON, continue normally
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
                    # This should not happen due to safe wrappers, but just in case
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
                    # Remove from tracking
                    self._running_executors.pop(f"{executor_id}_tools", None)
                    self._running_executors.pop(f"{executor_id}_subagents", None)
        
        return processed_response, stop_tool_detected, stop_tool_result
    
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
    
    def add_tool(self, tool) -> None:
        """Add a tool to the executor.
        
        Args:
            tool: Tool instance to add
        """
        self.tool_executor.tool_registry[tool.name] = tool
    
    def add_sub_agent(self, name: str, agent_factory: Callable) -> None:
        """Add a sub-agent factory to the executor.
        
        Args:
            name: Name of the sub-agent
            agent_factory: Factory function that creates the agent
        """
        self.subagent_manager.add_sub_agent(name, agent_factory)