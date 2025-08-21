"""Agent implementation for the Northau framework."""

from typing import Dict, List, Optional, Tuple, Iterator, Callable, Any, Union
import json
import re
import xml.etree.ElementTree as ET
import logging
import os
import signal
import atexit
import threading
import weakref
import uuid
from datetime import datetime
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None
from .prompt_handler import PromptHandler
from ..llm import LLMConfig
from .agent_context import AgentContext
try:
    from langfuse.client import Langfuse
    try:
        from langfuse.openai import openai
    except ImportError:
        import openai
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None
    try:
        import openai
    except ImportError:
        openai = None

# Setup logger for agent execution
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Global registry to track all active agents for cleanup
_active_agents = weakref.WeakSet()
_cleanup_registered = False
_cleanup_lock = threading.Lock()


def _cleanup_all_agents():
    """Clean up all active agents and their running sub-agents."""
    logger.info("🧹 Cleaning up all active agents and sub-agents...")
    agents_to_cleanup = list(_active_agents)
    
    for agent in agents_to_cleanup:
        try:
            if hasattr(agent, '_cleanup_agent'):
                agent._cleanup_agent()
        except Exception as e:
            logger.error(f"❌ Error cleaning up agent {getattr(agent, 'name', 'unknown')}: {e}")
    
    logger.info("✅ Agent cleanup completed")


def _signal_handler(signum, frame):
    """Handle termination signals by cleaning up agents."""
    logger.info(f"🚨 Received signal {signum}, initiating cleanup...")
    _cleanup_all_agents()
    # Re-raise the signal with default handler to ensure proper termination
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _register_cleanup_handlers():
    """Register cleanup handlers for process termination."""
    global _cleanup_registered
    with _cleanup_lock:
        if _cleanup_registered:
            return
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        
        # Register atexit handler as fallback
        atexit.register(_cleanup_all_agents)
        
        _cleanup_registered = True
        logger.debug("🔧 Cleanup handlers registered")


class Agent:
    """Main agent class that handles task execution and sub-agent delegation."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        tools: Optional[List] = None,
        sub_agents: Optional[List[Tuple[str, Callable[[], 'Agent']]]] = None,
        system_prompt: Optional[str] = None,
        system_prompt_type: str = "string",
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
        max_iterations: int = 100,
        max_context_tokens: int = 128000,
        max_running_subagents: int = 5,
        error_handler: Optional[Callable] = None,
        retry_attempts: int = 3,
        timeout: int = 300,
        # Token counting parameters
        token_counter: Optional[Callable[[List[Dict[str, str]]], int]] = None,
        # Context parameters
        initial_state: Optional[Dict[str, Any]] = None,
        initial_config: Optional[Dict[str, Any]] = None,
        # MCP parameters
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
    ):
        """Initialize an agent with specified configuration."""
        self.name = name or f"agent_{id(self)}"
        self.tools = tools or []
        self.sub_agent_factories = dict(sub_agents or [])
        self.system_prompt = system_prompt
        self.system_prompt_type = system_prompt_type
        self.max_context_tokens = max_context_tokens
        self.max_iterations = max_iterations
        self.max_running_subagents = max_running_subagents
        self.error_handler = error_handler
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        
        # Setup token counter
        self.token_counter = token_counter or self._create_default_token_counter()
        
        # Initialize context data
        self.initial_state = initial_state or {}
        self.initial_config = initial_config or {}
        
        # Handle LLM configuration
        self.llm_config = self._setup_llm_config(llm_config)
        
        # Initialize MCP tools if servers are configured
        if mcp_servers:
            self._initialize_mcp_tools(mcp_servers)
        
        # Build tool registry for quick lookup
        self.tool_registry = {tool.name: tool for tool in self.tools}
        
        # Conversation history
        self.history = []
        
        # Initialize prompt handler
        self.prompt_handler = PromptHandler()
        
        # Process system prompt
        self.processed_system_prompt = self._process_system_prompt()
        
        # Initialize OpenAI client
        if openai is not None:
            client_kwargs = self.llm_config.to_client_kwargs()
            self.openai_client = openai.OpenAI(**client_kwargs)
        else:
            self.openai_client = None
        
        if LANGFUSE_AVAILABLE:
            self.langfuse_client = Langfuse()
        else:
            self.langfuse_client = None
        
        # Initialize process tracking for sub-agents
        self._running_executors = {}  # Maps executor_id to ThreadPoolExecutor
        self._executor_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        
        # Register this agent for cleanup and setup handlers
        _active_agents.add(self)
        _register_cleanup_handlers()
    
    def _create_default_token_counter(self) -> Callable[[List[Dict[str, str]]], int]:
        """Create default token counter using tiktoken with gpt-4o model."""
        if not TIKTOKEN_AVAILABLE:
            logger.warning("tiktoken not available, using approximate token counting (chars/4)")
            return self._fallback_message_token_counter
        
        try:
            # Use gpt-4o encoding
            encoding = tiktoken.encoding_for_model("gpt-4o")
            
            def tiktoken_message_counter(messages: List[Dict[str, str]]) -> int:
                """Count tokens in messages using tiktoken."""
                total_tokens = 0
                for message in messages:
                    # Add tokens for role and content
                    total_tokens += len(encoding.encode(message.get('content', '')))
                return total_tokens
            
            return tiktoken_message_counter
        except Exception as e:
            logger.warning(f"Failed to create tiktoken encoder: {e}, using approximate counting")
            return self._fallback_message_token_counter
    
    def _fallback_message_token_counter(self, messages: List[Dict[str, str]]) -> int:
        """Fallback token counter using character approximation."""
        total_tokens = 0
        for message in messages:
            # Add tokens for role and content using chars/4 approximation
            total_tokens += len(message.get('role', '')) // 4
            total_tokens += len(message.get('content', '')) // 4
            # Add overhead tokens for message formatting
            total_tokens += 4
        return max(total_tokens, 1)  # Ensure at least 1 token
    
    def _count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in a list of messages using configured token counter."""
        return self.token_counter(messages)
    
    def _setup_llm_config(
        self, 
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]], 
    ) -> LLMConfig:
        """Setup LLM configuration with backward compatibility."""
        if llm_config is None:
            # Create from deprecated parameters or defaults
            raise ValueError("llm_config is required")
        elif isinstance(llm_config, dict):
            # Create from dictionary
            return LLMConfig(**llm_config)
        elif isinstance(llm_config, LLMConfig):
            # Use provided config
            return llm_config
        else:
            raise ValueError(f"Invalid llm_config type: {type(llm_config)}")
    
    def _initialize_mcp_tools(self, mcp_servers: List[Dict[str, Any]]) -> None:
        """Initialize tools from MCP servers."""
        try:
            from ..tool.builtin import sync_initialize_mcp_tools
            logger.info(f"Initializing MCP tools from {len(mcp_servers)} servers")
            
            mcp_tools = sync_initialize_mcp_tools(mcp_servers)
            self.tools.extend(mcp_tools)
            
            logger.info(f"Successfully initialized {len(mcp_tools)} MCP tools")
            
        except ImportError:
            logger.error("MCP client not available. Please install the mcp package.")
        except Exception as e:
            logger.error(f"Failed to initialize MCP tools: {e}")
    
    def run(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Run agent with a message and return response."""
        logger.info(f"🤖 Agent '{self.name}' starting execution")
        logger.info(f"📝 User message: {message}")
        
        # Merge initial state/config with provided ones
        merged_state = {**self.initial_state}
        if state:
            merged_state.update(state)
            
        merged_config = {**self.initial_config}
        if config:
            merged_config.update(config)
        
        # Create agent context
        with AgentContext(state=merged_state, config=merged_config) as ctx:
            # Setup context modification callback to refresh system prompt
            def on_context_modified():
                # Reset processed system prompt to force regeneration
                self.processed_system_prompt = self._process_system_prompt()
            
            ctx.add_modification_callback(on_context_modified)
            
            if history:
                self.history = history.copy()
            
            # Add user message to history
            self.history.append({"role": "user", "content": message})
            
            try:
                # Generate response using the LLM
                if self.langfuse_client:
                    with self.langfuse_client.start_as_current_span(name=self.name, input=message) as generation:
                        response = self._generate_response(message, context)
                        generation.update(
                            output=response
                        )
                else:
                    response = self._generate_response(message, context)
                
                # Add assistant response to history
                self.history.append({"role": "assistant", "content": response})
                
                logger.info(f"✅ Agent '{self.name}' completed execution")
                return response
                
            except Exception as e:
                logger.error(f"❌ Agent '{self.name}' encountered error: {e}")
                if self.error_handler:
                    error_response = self.error_handler(e, self, context)
                    self.history.append({"role": "assistant", "content": error_response})
                    return error_response
                else:
                    raise
    
    def add_tool(self, tool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        self.tool_registry[tool.name] = tool
    
    def add_sub_agent(self, name: str, agent_factory: Callable[[], 'Agent']) -> None:
        """Add a sub-agent factory for delegation."""
        self.sub_agent_factories[name] = agent_factory
    
    def _cleanup_agent(self) -> None:
        """Clean up this agent and all its running sub-agents."""
        logger.info(f"🧹 Cleaning up agent '{self.name}' and its sub-agents...")
        
        # Signal shutdown to prevent new tasks
        self._shutdown_event.set()
        
        # Shutdown all running executors
        with self._executor_lock:
            for executor_id, executor in self._running_executors.items():
                try:
                    logger.info(f"🛑 Shutting down executor {executor_id}")
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    logger.error(f"❌ Error shutting down executor {executor_id}: {e}")
            
            self._running_executors.clear()
        
        logger.info(f"✅ Agent '{self.name}' cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup when agent is garbage collected."""
        try:
            if not self._shutdown_event.is_set():
                self._cleanup_agent()
        except Exception:
            pass  # Avoid exceptions during garbage collection
    
    def delegate_task(
        self,
        task: str,
        sub_agent_name: str,
        context: Optional[Dict] = None
    ) -> str:
        """Explicitly delegate a task to a sub-agent."""
        return self.call_sub_agent(sub_agent_name, task, context)
    
    def _generate_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate response using OpenAI API with XML-based tool and sub-agent calls."""
        if not self.openai_client:
            raise RuntimeError("OpenAI client is not available. Please check your API configuration.")
        
        try:
            # Build system prompt with available tools and sub-agents
            system_prompt = self._build_system_prompt_with_capabilities(context)
            
            # Prepare initial messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
            
            # Add conversation history
            if self.history:
                # Insert history before the current message
                history_messages = []
                for msg in self.history[:-1]:  # Exclude the current message we just added
                    history_messages.append(msg)
                messages = [messages[0]] + history_messages + [messages[1]]
            
            # Loop until no more tool calls or sub-agent calls are made
            max_iterations = self.max_iterations  # Prevent infinite loops
            iteration = 0
            final_response = ""
            
            logger.info(f"🔄 Starting iterative execution loop for agent '{self.name}'")
            
            while iteration < max_iterations:
                logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations} for agent '{self.name}'")
                
                # Call OpenAI API with LLM config parameters
                api_params = self.llm_config.to_openai_params()
                api_params['messages'] = messages
                
                # Count current prompt tokens
                current_prompt_tokens = self._count_messages_tokens(messages)
                
                # Check if prompt exceeds max context tokens - force stop if so
                if current_prompt_tokens > self.max_context_tokens:
                    logger.error(f"❌ Prompt tokens ({current_prompt_tokens}) exceed max_context_tokens ({self.max_context_tokens}). Stopping execution.")
                    final_response += f"\n\n[Error: Prompt too long ({current_prompt_tokens} tokens) exceeds maximum context ({self.max_context_tokens} tokens). Execution stopped.]"
                    break
                
                # Calculate max_tokens dynamically based on available budget
                available_tokens = self.max_context_tokens - current_prompt_tokens
                
                # Get desired max_tokens from LLM config or use reasonable default
                desired_max_tokens = api_params.get('max_tokens', 4096)
                calculated_max_tokens = min(desired_max_tokens, available_tokens)
                
                # Ensure we have at least some tokens for response
                if calculated_max_tokens < 50:
                    logger.error(f"❌ Insufficient tokens for response ({calculated_max_tokens}). Stopping execution.")
                    final_response += f"\n\n[Error: Insufficient tokens for response ({calculated_max_tokens} tokens). Context too full.]"
                    break
                
                # Set max_tokens based on calculation
                api_params['max_tokens'] = calculated_max_tokens
                
                logger.info(f"🔢 Token usage: prompt={current_prompt_tokens}, max_tokens={api_params['max_tokens']}, available={available_tokens}")
                
                # Add stop sequences for XML closing tags to prevent malformed XML
                xml_stop_sequences = [
                    "</tool_use>",
                    "</sub-agent>", 
                    "</use_parallel_tool_calls>",
                    "</use_parallel_sub_agents>",
                    "</use_batch_agent>"
                ]
                
                # Merge with existing stop sequences if any
                existing_stop = api_params.get('stop', [])
                if isinstance(existing_stop, str):
                    existing_stop = [existing_stop]
                elif existing_stop is None:
                    existing_stop = []
                
                api_params['stop'] = existing_stop + xml_stop_sequences
                
                # Debug logging for LLM messages
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Request Messages for agent '{self.name}':")
                    for i, msg in enumerate(messages):
                        # logger.info(f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
                        logger.info(f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content']}")
                
                logger.info(f"🧠 Calling LLM for agent '{self.name}' with {api_params['max_tokens']} max tokens...")
                response = self.openai_client.chat.completions.create(**api_params)
                assistant_response = response.choices[0].message.content
                
                # Add back XML closing tags if they were removed by stop sequences
                assistant_response = self._restore_xml_closing_tags(assistant_response)
                
                # Debug logging for LLM response
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Response for agent '{self.name}': {assistant_response}")
                
                logger.info(f"💬 LLM Response for agent '{self.name}': {assistant_response}")
                
                # Store this as the latest response (potential final response)
                final_response = assistant_response
                
                # Check if response contains tool calls or sub-agent calls (including parallel formats and batch processing)
                has_tool_calls = bool(re.search(r'<tool_use>.*?</tool_use>', assistant_response, re.DOTALL))
                has_sub_agent_calls = bool(re.search(r'<sub-agent>.*?</sub-agent>', assistant_response, re.DOTALL))
                has_parallel_tool_calls = bool(re.search(r'<use_parallel_tool_calls>.*?</use_parallel_tool_calls>', assistant_response, re.DOTALL))
                has_parallel_sub_agents = bool(re.search(r'<use_parallel_sub_agents>.*?</use_parallel_sub_agents>', assistant_response, re.DOTALL))
                has_batch_agent = bool(re.search(r'<use_batch_agent>.*?</use_batch_agent>', assistant_response, re.DOTALL))
                
                logger.info(f"🔍 Analysis for agent '{self.name}': tool_calls={has_tool_calls}, sub_agent_calls={has_sub_agent_calls}, parallel_tool_calls={has_parallel_tool_calls}, parallel_sub_agents={has_parallel_sub_agents}, batch_agent={has_batch_agent}")
                
                if not has_tool_calls and not has_sub_agent_calls and not has_parallel_tool_calls and not has_parallel_sub_agents and not has_batch_agent:
                    # No more commands to execute, return final response
                    logger.info(f"🏁 No more commands to execute, finishing agent '{self.name}'")
                    break
                
                # Add the assistant's original response to conversation
                messages.append({"role": "assistant", "content": assistant_response})
                
                # Process tool calls and sub-agent calls
                logger.info(f"⚙️ Processing tool/sub-agent calls for agent '{self.name}'...")
                processed_response = self._process_xml_calls(assistant_response)
                
                # Extract just the tool results from processed_response
                tool_results = processed_response.replace(assistant_response, "").strip()
                
                if tool_results:
                    logger.info(f"🔧 Tool execution results for agent '{self.name}':\n{tool_results}")
                
                # Add tool results as user feedback with iteration context
                remaining_iterations = max_iterations - iteration - 1
                iteration_hint = self._build_iteration_hint(iteration + 1, max_iterations, remaining_iterations)
                
                if tool_results:
                    messages.append({
                        "role": "user", 
                        "content": f"Tool execution results:\n{tool_results}\n\n{iteration_hint}"
                    })
                else:
                    messages.append({
                        "role": "user", 
                        "content": f"{iteration_hint}"
                    })
                
                iteration += 1
            
            # Add note if max iterations reached
            if iteration >= max_iterations:
                final_response += "\n\n[Note: Maximum iteration limit reached]"
            
            return final_response
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Error calling OpenAI API: {e}") from e
    
    
    def call_sub_agent(self, sub_agent_name: str, message: str, context: Optional[Dict] = None) -> str:
        """Call a sub-agent like a tool call."""
        from .agent_context import get_context
        
        # Check if agent is shutting down
        if self._shutdown_event.is_set():
            logger.warning(f"⚠️ Agent '{self.name}' is shutting down, cannot call sub-agent '{sub_agent_name}'")
            raise RuntimeError(f"Agent '{self.name}' is shutting down")
        
        logger.info(f"🤖➡️🤖 Agent '{self.name}' calling sub-agent '{sub_agent_name}' with message: {message}")
        
        if sub_agent_name not in self.sub_agent_factories:
            error_msg = f"Sub-agent '{sub_agent_name}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Instantiate a fresh sub-agent from the factory
        sub_agent_factory = self.sub_agent_factories[sub_agent_name]
        sub_agent = sub_agent_factory()
        try:
            # Pass current agent context state and config to sub-agent
            current_context = get_context()
            if current_context:
                if self.langfuse_client:
                    with self.langfuse_client.start_as_current_span(name=f"Sub-agent: {sub_agent_name}", input=message) as generation:
                        result = sub_agent.run(
                            message, 
                            context=context,
                            state=current_context.state.copy(),
                            config=current_context.config.copy()
                        )
                        generation.update(
                            output=result
                        )
                else:
                    result = sub_agent.run(
                        message, 
                        context=context,
                        state=current_context.state.copy(),
                        config=current_context.config.copy()
                    )
            else:
                result = sub_agent.run(message, context=context)
            logger.info(f"✅ Sub-agent '{sub_agent_name}' returned result to agent '{self.name}'")
            return result
        except Exception as e:
            logger.error(f"❌ Sub-agent '{sub_agent_name}' failed: {e}")
            raise
    
    def _execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool with given parameters."""
        logger.info(f"🔧 Executing tool '{tool_name}' with parameters: {parameters}")
        
        if tool_name not in self.tool_registry:
            error_msg = f"Tool '{tool_name}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        tool = self.tool_registry[tool_name]
        try:
            if self.langfuse_client:
                with self.langfuse_client.start_as_current_generation(name=f"Tool: {tool_name}", input=parameters) as generation:
                    result = tool.execute(**parameters)
                    generation.update(
                        output=result,
                    )
            else:
                result = tool.execute(**parameters)
            logger.info(f"✅ Tool '{tool_name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Tool '{tool_name}' execution failed: {e}")
            raise

    
    def _build_system_prompt_with_capabilities(self, runtime_context: Optional[Dict] = None) -> str:
        """Build system prompt including tool and sub-agent capabilities."""
        # Check if context has been modified and refresh the processed system prompt
        from .agent_context import get_context
        current_context = get_context()
        if current_context and current_context.is_modified():
            self.processed_system_prompt = self._process_system_prompt()
            current_context.reset_modification_flag()
        
        # Re-process system prompt with runtime context if provided
        if runtime_context and self.system_prompt:
            try:
                # Merge runtime context with default context
                default_context = self.prompt_handler.get_default_context(self)
                merged_context = {**default_context, **runtime_context}
                base_prompt = self.prompt_handler.create_dynamic_prompt(
                    self.system_prompt, self, additional_context=merged_context
                )
            except Exception as e:
                # Fallback to processed system prompt if runtime processing fails
                logger.error(f"❌ Error processing system prompt: {e}")
                base_prompt = self.processed_system_prompt
        else:
            base_prompt = self.processed_system_prompt

        # Template for tool documentation
        tools_template = """
{% if tools %}

## Available Tools
You can use tools by including XML blocks in your response:
{% for tool in tools %}

### {{ tool.name }}
{{ tool.description or 'No description available' }}
Usage:
<tool_use>
  <tool_name>{{ tool.name }}</tool_name>
  <parameter>
{% if tool.parameters %}
{% for param in tool.parameters %}
    <{{ param.name }}>{{ param.description }}{% if not param.required %} (optional, type: {{ param.type }}{% if param.default %}, default: {{ param.default }}{% endif %}){% else %} (required, type: {{ param.type }}){% endif %}</{{ param.name }}>
{% endfor %}
{% endif %}
  </parameter>
</tool_use>
{% endfor %}
{% endif %}"""

        # Template for sub-agent documentation  
        sub_agents_template = """
{% if sub_agents %}

## Available Sub-Agents
You can delegate tasks to specialized sub-agents:
{% for sub_agent in sub_agents %}

### {{ sub_agent.name }}
{{ sub_agent.description or 'Specialized agent for ' + sub_agent.name + '-related tasks' }}
Usage:
<sub-agent>
  <agent_name>{{ sub_agent.name }}</agent_name>
  <message>task description</message>
</sub-agent>
{% endfor %}
{% endif %}"""

        try:
            # Prepare enhanced context with tool parameters
            context = self.prompt_handler.get_default_context(self)
            
            # Merge with runtime context if provided
            if runtime_context:
                context.update(runtime_context)
            
            # Add detailed tool parameter information
            if context.get('tools'):
                for i, tool_info in enumerate(context['tools']):
                    if hasattr(self.tools[i], 'input_schema'):
                        schema = getattr(self.tools[i], 'input_schema', {})
                        properties = schema.get('properties', {})
                        required_params = schema.get('required', [])
                        
                        parameters = []
                        for param_name, param_info in properties.items():
                            param_type = param_info.get('type', 'string')
                            param_desc = param_info.get('description', '')
                            default_value = param_info.get('default')
                            is_required = param_name in required_params
                            
                            # Map JSON Schema types to Python types
                            type_mapping = {
                                'string': 'str',
                                'integer': 'int', 
                                'number': 'float',
                                'boolean': 'bool',
                                'array': 'list',
                                'object': 'dict'
                            }
                            python_type = type_mapping.get(param_type, 'str')
                            
                            parameters.append({
                                'name': param_name,
                                'description': param_desc,
                                'type': python_type,
                                'required': is_required,
                                'default': default_value
                            })
                        
                        tool_info['parameters'] = parameters
            
            # Render tool documentation
            tools_docs = self.prompt_handler.create_dynamic_prompt(
                tools_template, self, context
            )
            
            # Render sub-agent documentation
            sub_agents_docs = self.prompt_handler.create_dynamic_prompt(
                sub_agents_template, self, context
            )
            
            base_prompt += tools_docs + sub_agents_docs
            
        except Exception:
            # Fallback to original implementation if template rendering fails
            base_prompt = self._build_system_prompt_with_capabilities_fallback(base_prompt)
        
        base_prompt += """\n\nWhen you use tools or sub-agents, include the XML blocks in your response and I will execute them and provide the results.

For parallel execution of multiple tools, use:
<use_parallel_tool_calls>
<parallel_tool>
  <tool_name>tool1</tool_name>
  <parameter>...</parameter>
</parallel_tool>
<parallel_tool>
  <tool_name>tool2</tool_name>
  <parameter>...</parameter>
</parallel_tool>
</use_parallel_tool_calls>

For parallel execution of tools and sub-agents together, use:
<use_parallel_sub_agents>
<parallel_agent>
  <agent_name>agent1</agent_name>
  <message>task description</message>
</parallel_agent>
<parallel_tool>
  <tool_name>tool1</tool_name>
  <parameter>...</parameter>
</parallel_tool>
</use_parallel_sub_agents>

For batch processing multiple data items with an agent, use:
<use_batch_agent>
  <agent_name>agent_name</agent_name>
  <input_data_source>
    <file_name>/absolute/path/to/file.jsonl</file_name>
    <format>jsonl</format>
  </input_data_source>
  <message>Your message template with {variable_placeholders}</message>
</use_batch_agent>

IMPORTANT: When using batch processing:
- Ensure the JSONL file exists or will be created if not found
- Validate that the JSONL file contains valid JSON objects
- Make sure the message template uses valid keys that exist in the JSONL data
- Each line in the JSONL file should be a separate JSON object
- Variable placeholders in the message use {key_name} format from the JSON data
- All agents in the batch will run in parallel for efficient processing"""
        
        return base_prompt
    
    def _build_system_prompt_with_capabilities_fallback(self, base_prompt: str) -> str:
        """Fallback method for building capabilities documentation."""
        # Add tool documentation
        if self.tools:
            tool_docs = []
            tool_docs.append("\n## Available Tools")
            tool_docs.append("You can use tools by including XML blocks in your response:")
            
            for tool in self.tools:
                tool_docs.append(f"\n### {tool.name}")
                tool_docs.append(f"{getattr(tool, 'description', 'No description available')}")
                tool_docs.append("Usage:")
                tool_docs.append("<tool_use>")
                tool_docs.append(f"  <tool_name>{tool.name}</tool_name>")
                tool_docs.append("  <parameter>")
                
                # Add parameter documentation from schema
                schema = getattr(tool, 'input_schema', {})
                properties = schema.get('properties', {})
                required_params = schema.get('required', [])
                
                for param_name, param_info in properties.items():
                    param_type = param_info.get('type', 'string')
                    param_desc = param_info.get('description', '')
                    default_value = param_info.get('default')
                    is_required = param_name in required_params
                    
                    # Map JSON Schema types to Python types
                    type_mapping = {
                        'string': 'str',
                        'integer': 'int', 
                        'number': 'float',
                        'boolean': 'bool',
                        'array': 'list',
                        'object': 'dict'
                    }
                    python_type = type_mapping.get(param_type, 'str')
                    
                    # Build parameter documentation with type and optional info
                    param_parts = [param_desc]
                    if not is_required:
                        param_parts.append(f"(optional, type: {python_type}")
                        if default_value is not None:
                            param_parts.append(f", default: {default_value}")
                        param_parts.append(")")
                    else:
                        param_parts.append(f"(required, type: {python_type})")
                    
                    param_doc = " ".join(param_parts)
                    tool_docs.append(f"    <{param_name}>{param_doc}</{param_name}>")
                
                tool_docs.append("  </parameter>")
                tool_docs.append("</tool_use>")
            
            base_prompt += "\n".join(tool_docs)
            base_prompt += "\n\nIMPORTANT: use </parameter> to end the parameters block."
        
        # Add sub-agent documentation
        if self.sub_agent_factories:
            sub_agent_docs = []
            sub_agent_docs.append("\n## Available Sub-Agents")
            sub_agent_docs.append("You can delegate tasks to specialized sub-agents:")
            
            for name in self.sub_agent_factories.keys():
                sub_agent_docs.append(f"\n### {name}")
                sub_agent_docs.append(f"Specialized agent for {name}-related tasks")
                sub_agent_docs.append("Usage:")
                sub_agent_docs.append("<sub-agent>")
                sub_agent_docs.append(f"  <agent_name>{name}</agent_name>")
                sub_agent_docs.append("  <message>task description</message>")
                sub_agent_docs.append("</sub-agent>")
            
            base_prompt += "\n".join(sub_agent_docs)
        
        return base_prompt
    
    def _process_xml_calls(self, response: str) -> str:
        """Process XML tool calls and sub-agent calls in the response in parallel."""
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
            batch_result = self._execute_batch_agent_from_xml(batch_agent_match.group(1))
            processed_response += f"\n\n<tool_result>\n<tool_name>batch_agent</tool_name>\n<result>{batch_result}</result>\n</tool_result>"
            return processed_response
        
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
            return processed_response
        
        # Check if agent is shutting down
        if self._shutdown_event.is_set():
            logger.warning(f"⚠️ Agent '{self.name}' is shutting down, skipping new task execution")
            return processed_response
        
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
                future = tool_executor.submit(task_ctx.run, self._execute_tool_from_xml_safe, tool_xml)
                tool_futures[future] = ('tool', tool_xml)
            
            # Submit sub-agent execution tasks (limited by max_running_subagents)
            sub_agent_futures = {}
            for sub_agent_xml in sub_agent_matches:
                # Propagate current tracing context into the worker thread
                task_ctx = copy_context()
                future = subagent_executor.submit(task_ctx.run, self._execute_sub_agent_from_xml_safe, sub_agent_xml)
                sub_agent_futures[future] = ('sub_agent', sub_agent_xml)
            
            # Combine all futures
            all_futures = {**tool_futures, **sub_agent_futures}
            
            # Collect results as they complete
            tool_results = []
            for future in as_completed(all_futures):
                call_type, xml_content = all_futures[future]
                try:
                    result_data = future.result()
                    if call_type == 'tool':
                        tool_name, result, is_error = result_data
                        if is_error:
                            logger.error(f"❌ Tool '{tool_name}' error: {result}")
                            tool_results.append(f"<tool_result>\n<tool_name>{tool_name}</tool_name>\n<error>{result}</error>\n</tool_result>")
                        else:
                            logger.info(f"📤 Tool '{tool_name}' result: {result}")
                            tool_results.append(f"<tool_result>\n<tool_name>{tool_name}</tool_name>\n<result>{result}</result>\n</tool_result>")
                    elif call_type == 'sub_agent':
                        agent_name, result, is_error = result_data
                        if is_error:
                            logger.error(f"❌ Sub-agent '{agent_name}' error: {result}")
                            tool_results.append(f"<tool_result>\n<tool_name>{agent_name}_sub_agent</tool_name>\n<error>{result}</error>\n</tool_result>")
                        else:
                            logger.info(f"📤 Sub-agent '{agent_name}' result: {result}")
                            tool_results.append(f"<tool_result>\n<tool_name>{agent_name}_sub_agent</tool_name>\n<result>{result}</result>\n</tool_result>")
                except Exception as e:
                    # This should not happen due to safe wrappers, but just in case
                    logger.error(f"❌ Unexpected error processing {call_type}: {e}")
                    tool_results.append(f"<tool_result>\n<tool_name>unknown</tool_name>\n<error>Unexpected error: {str(e)}</error>\n</tool_result>")
            
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
                    # Remove from tracking
                    self._running_executors.pop(f"{executor_id}_tools", None)
                    self._running_executors.pop(f"{executor_id}_subagents", None)
        
        return processed_response
    
    def _execute_tool_from_xml_safe(self, xml_content: str) -> Tuple[str, str, bool]:
        """Safe wrapper for _execute_tool_from_xml that handles exceptions. Returns (tool_name, result, is_error)."""
        try:
            tool_name, result = self._execute_tool_from_xml(xml_content)
            return tool_name, result, False
        except Exception as e:
            # Extract tool name for error reporting using more robust parsing
            tool_name = self._extract_tool_name_from_xml(xml_content)
            return tool_name, str(e), True
    
    def _extract_tool_name_from_xml(self, xml_content: str) -> str:
        """Extract tool name from potentially malformed XML using multiple strategies."""
        # Strategy 1: Try simple regex extraction
        import re
        tool_name_match = re.search(r'<tool_name>\s*([^<]+)\s*</tool_name>', xml_content, re.IGNORECASE | re.DOTALL)
        if tool_name_match:
            return tool_name_match.group(1).strip()
        
        # Strategy 2: Try parsing as valid XML
        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is not None and tool_name_elem.text:
                return tool_name_elem.text.strip()
        except ET.ParseError:
            pass
        
        # Strategy 3: Try cleaning up common XML issues
        try:
            # Remove potential unclosed tags and extra content
            cleaned_xml = xml_content.strip()
            
            # Try to fix unclosed tags by finding the pattern and closing them
            lines = cleaned_xml.split('\n')
            for i, line in enumerate(lines):
                # Check if this line has an opening tag but no closing tag
                tag_match = re.search(r'<(\w+)>[^<]*$', line.strip())
                if tag_match:
                    tag_name = tag_match.group(1)
                    lines[i] = line.rstrip() + f'</{tag_name}>'
            
            cleaned_xml = '\n'.join(lines)
            root = ET.fromstring(f"<root>{cleaned_xml}</root>")
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is not None and tool_name_elem.text:
                return tool_name_elem.text.strip()
        except (ET.ParseError, AttributeError):
            pass
        
        return "unknown"
    
    def _execute_sub_agent_from_xml_safe(self, xml_content: str) -> Tuple[str, str, bool]:
        """Safe wrapper for _execute_sub_agent_from_xml that handles exceptions. Returns (agent_name, result, is_error)."""
        try:
            agent_name, result = self._execute_sub_agent_from_xml(xml_content)
            return agent_name, result, False
        except Exception as e:
            # Extract agent name for error reporting using more robust parsing
            agent_name = self._extract_agent_name_from_xml(xml_content)
            return agent_name, str(e), True
    
    def _extract_agent_name_from_xml(self, xml_content: str) -> str:
        """Extract agent name from potentially malformed XML using multiple strategies."""
        # Strategy 1: Try simple regex extraction
        import re
        agent_name_match = re.search(r'<agent_name>\s*([^<]+)\s*</agent_name>', xml_content, re.IGNORECASE | re.DOTALL)
        if agent_name_match:
            return agent_name_match.group(1).strip()
        
        # Strategy 2: Try parsing as valid XML
        try:
            root = ET.fromstring(f"<root>{xml_content}</root>")
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is not None and agent_name_elem.text:
                return agent_name_elem.text.strip()
        except ET.ParseError:
            pass
        
        # Strategy 3: Try cleaning up common XML issues
        try:
            # Remove potential unclosed tags and extra content
            cleaned_xml = xml_content.strip()
            
            # Try to fix unclosed tags by finding the pattern and closing them
            lines = cleaned_xml.split('\n')
            for i, line in enumerate(lines):
                # Check if this line has an opening tag but no closing tag
                tag_match = re.search(r'<(\w+)>[^<]*$', line.strip())
                if tag_match:
                    tag_name = tag_match.group(1)
                    lines[i] = line.rstrip() + f'</{tag_name}>'
            
            cleaned_xml = '\n'.join(lines)
            root = ET.fromstring(f"<root>{cleaned_xml}</root>")
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is not None and agent_name_elem.text:
                return agent_name_elem.text.strip()
        except (ET.ParseError, AttributeError):
            pass
        
        return "unknown"
    
    def _parse_xml_content_robust(self, xml_content: str):
        """Parse XML content using multiple strategies to handle malformed XML."""
        import re
        import html
        
        # Strategy 1: Try as-is
        try:
            return ET.fromstring(f"<root>{xml_content}</root>")
        except ET.ParseError as e:
            logger.warning(f"Initial XML parsing failed: {e}. Attempting recovery strategies...")
        
        # Strategy 2: Clean up common issues (unclosed tags, extra whitespace)
        try:
            cleaned_xml = xml_content.strip()
            
            # Fix potential unclosed tags by ensuring proper closing
            lines = cleaned_xml.split('\n')
            corrected_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check for unclosed tags (opening tag without closing)
                tag_matches = re.findall(r'<(\w+)(?:\s+[^>]*)?>([^<]*?)(?:</\1>|$)', line)
                if tag_matches:
                    # Line has proper tag structure
                    corrected_lines.append(line)
                else:
                    # Check if line has opening tag but no closing tag
                    opening_match = re.match(r'<(\w+)(?:\s+[^>]*)?>\s*([^<]*)\s*$', line)
                    if opening_match:
                        tag_name = opening_match.group(1)
                        content = opening_match.group(2)
                        corrected_lines.append(f"<{tag_name}>{content}</{tag_name}>")
                    else:
                        corrected_lines.append(line)
            
            cleaned_xml = '\n'.join(corrected_lines)
            return ET.fromstring(f"<root>{cleaned_xml}</root>")
            
        except ET.ParseError:
            pass
        
        # Strategy 3: Handle JSON content within XML parameters
        try:
            def handle_json_param_content(match):
                param_name = match.group(1)
                param_content = match.group(2).strip()
                
                # Check if content looks like JSON
                if param_content.startswith(('{', '[')):
                    # Wrap JSON content in CDATA to preserve it
                    return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                elif '<' in param_content and '>' in param_content:
                    # Escape HTML/XML content if it contains < >
                    escaped_content = html.escape(param_content)
                    return f"<{param_name}>{escaped_content}</{param_name}>"
                return match.group(0)
            
            # Find and handle parameter content within parameters block
            escaped_xml = xml_content
            params_match = re.search(r'<parameter>(.*?)</parameter>', xml_content, re.DOTALL)
            if params_match:
                params_content = params_match.group(1)
                # Pattern to match individual parameter tags with their content
                param_pattern = r'<(\w+)>(.*?)</\1>'
                escaped_params = re.sub(param_pattern, handle_json_param_content, params_content, flags=re.DOTALL)
                escaped_xml = xml_content.replace(params_match.group(1), escaped_params)
            
            return ET.fromstring(f"<root>{escaped_xml}</root>")
            
        except ET.ParseError:
            pass
        
        # Strategy 4: Fallback - escape all content and selectively unescape XML tags
        try:
            escaped_content = html.escape(xml_content, quote=False)
            # Unescape the XML tags we need
            escaped_content = escaped_content.replace("&lt;", "<").replace("&gt;", ">")
            return ET.fromstring(f"<root>{escaped_content}</root>")
        except ET.ParseError:
            pass
        
        # Strategy 5: Extract content using regex and build minimal XML
        try:
            tool_name_match = re.search(r'<tool_name>\s*([^<]+)\s*</tool_name>', xml_content, re.IGNORECASE | re.DOTALL)
            tool_name = tool_name_match.group(1).strip() if tool_name_match else "unknown"
            
            # Build minimal XML structure
            minimal_xml = f"<tool_name>{tool_name}</tool_name>"
            
            # Try to extract parameters
            params_match = re.search(r'<parameter>(.*?)</parameter>', xml_content, re.DOTALL | re.IGNORECASE)
            if params_match:
                params_content = params_match.group(1).strip()
                minimal_xml += f"<parameter>{params_content}</parameter>"
            
            return ET.fromstring(f"<root>{minimal_xml}</root>")
        except (ET.ParseError, AttributeError):
            pass
        
        # Final fallback: raise with detailed error
        raise ValueError(f"Unable to parse XML content after multiple strategies. Content preview: {xml_content[:200]}...")
    
    def _execute_tool_from_xml(self, xml_content: str) -> Tuple[str, str]:
        """Execute a tool from XML content. Returns (tool_name, result)."""
        try:
            # Parse XML - use improved multi-strategy approach
            root = self._parse_xml_content_robust(xml_content)
            
            # Get tool name
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is None:
                raise ValueError("Missing tool_name in tool_use XML")
            
            tool_name = (tool_name_elem.text or "").strip()
            
            # Get parameters
            parameters = {}
            params_elem = root.find('parameter')
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    
                    # Check if parameter has child elements (nested XML structure)
                    if len(param) > 0:
                        # Parameter has child elements - parse as nested XML into dictionary
                        param_value = self._parse_nested_xml_to_dict(param)
                    else:
                        # Handle both regular text and CDATA content, preserving whitespace for JSON
                        if param.text is not None:
                            param_value = param.text
                            # If there's tail text, include it
                            if param.tail:
                                param_value = ''.join(param.itertext())
                        else:
                            # Handle case where content is in CDATA or mixed content
                            param_value = ''.join(param.itertext()) or ""
                        
                        # Unescape HTML entities in parameter values
                        import html
                        param_value = html.unescape(param_value)
                        
                        # Don't strip whitespace for JSON parameters as it might be significant
                        if param_value.strip().startswith(('{', '[')):
                            # Likely JSON, preserve formatting but clean up excessive whitespace
                            param_value = param_value.strip()
                        else:
                            param_value = param_value.strip()
                    
                    parameters[param_name] = self._convert_parameter_type(
                        tool_name, param_name, param_value
                    )
            
            # Execute tool
            result = self._execute_tool(tool_name, parameters)
            return tool_name, json.dumps(result, indent=2, ensure_ascii=False)
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
    
    def _parse_nested_xml_to_dict(self, element) -> Dict[str, Any]:
        """Parse nested XML element into a dictionary."""
        result = {}
        
        for child in element:
            child_name = child.tag
            
            # Check if child has its own children (nested structure)
            if len(child) > 0:
                # Recursively parse nested elements
                result[child_name] = self._parse_nested_xml_to_dict(child)
            else:
                # Get text content
                if child.text is not None:
                    child_value = child.text.strip()
                else:
                    child_value = ''.join(child.itertext()).strip()
                
                # Try to convert to appropriate Python type
                result[child_name] = self._convert_xml_value_to_python_type(child_value)
        
        return result
    
    def _convert_xml_value_to_python_type(self, value: str):
        """Convert XML string value to appropriate Python type."""
        if not value:
            return ""
        
        # Try boolean conversion first
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON parsing (for arrays or objects)
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # Return as string if no other type matches
        return value
    
    def _convert_parameter_type(self, tool_name: str, param_name: str, param_value):
        """Convert parameter value to the correct type based on tool schema."""
        # If param_value is already a dict (from nested XML parsing), return it as-is for object types
        if isinstance(param_value, dict):
            return param_value
        
        # If param_value is already a list, return it as-is for array types
        if isinstance(param_value, list):
            return param_value
        
        # If param_value is not a string, return as-is (already converted)
        if not isinstance(param_value, str):
            return param_value
        
        if tool_name not in self.tool_registry:
            return param_value  # Return as string if tool not found
        
        tool = self.tool_registry[tool_name]
        schema = getattr(tool, 'input_schema', {})
        properties = schema.get('properties', {})
        
        if param_name not in properties:
            return param_value  # Return as string if parameter not in schema
        
        param_info = properties[param_name]
        param_type = param_info.get('type', 'string')
        
        try:
            if param_type == 'boolean':
                if isinstance(param_value, bool):
                    return param_value
                return param_value.lower() in ('true', '1', 'yes', 'on')
            elif param_type == 'integer':
                return int(param_value)
            elif param_type == 'number':
                return float(param_value)
            elif param_type == 'array':
                # Try to parse as JSON array, fallback to comma-separated
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError as json_err:
                    logger.debug(f"JSON parsing failed for array parameter '{param_name}': {json_err}. Trying comma-separated fallback.")
                    return [item.strip() for item in param_value.split(',')]
            elif param_type == 'object':
                try:
                    return json.loads(param_value)
                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to parse JSON object for parameter '{param_name}': {json_err}")
                    logger.error(f"Parameter value was: {repr(param_value)}")
                    raise ValueError(f"Invalid JSON for parameter '{param_name}': {json_err}")
            else:  # string or unknown type
                return param_value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to convert parameter '{param_name}' of type '{param_type}': {e}")
            logger.warning(f"Parameter value was: {repr(param_value)}")
            return param_value  # Fallback to string value
    
    def _restore_xml_closing_tags(self, response: str) -> str:
        """Restore XML closing tags that may have been removed by stop sequences."""
        # Check for incomplete XML blocks and add missing closing tags
        restored_response = response
        
        # List of tag pairs to check (opening_tag, closing_tag)
        tag_pairs = [
            ('<tool_use>', '</tool_use>'),
            ('<sub-agent>', '</sub-agent>'),
            ('<parallel_tool>', '</parallel_tool>'),
            ('<parallel_agent>', '</parallel_agent>'),
            ('<use_parallel_tool_calls>', '</use_parallel_tool_calls>'),
            ('<use_parallel_sub_agents>', '</use_parallel_sub_agents>'),
            ('<use_batch_agent>', '</use_batch_agent>')
        ]
        
        for open_tag, close_tag in tag_pairs:
            if open_tag in restored_response and not restored_response.rstrip().endswith(close_tag):
                # Count open and close tags
                open_count = restored_response.count(open_tag)
                close_count = restored_response.count(close_tag)
                if open_count > close_count:
                    restored_response += close_tag
        
        return restored_response
    
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
    
    def _execute_batch_agent_from_xml(self, xml_content: str) -> str:
        """Execute batch agent processing from XML content."""
        try:
            # Parse XML using robust parsing
            root = self._parse_xml_content_robust(xml_content)
            
            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                raise ValueError("Missing agent_name in batch agent XML")
            
            agent_name = (agent_name_elem.text or "").strip()
            
            # Get input data source
            input_data_elem = root.find('input_data_source')
            if input_data_elem is None:
                raise ValueError("Missing input_data_source in batch agent XML")
            
            file_name_elem = input_data_elem.find('file_name')
            if file_name_elem is None:
                raise ValueError("Missing file_name in input_data_source")
            
            file_path = (file_name_elem.text or "").strip()
            
            format_elem = input_data_elem.find('format')
            data_format = (format_elem.text or "jsonl").strip() if format_elem is not None else "jsonl"
            
            # Get message template
            message_elem = root.find('message')
            if message_elem is None:
                raise ValueError("Missing message in batch agent XML")
            
            message_template = (message_elem.text or "").strip()
            
            # Validate file exists or create if not exist
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist. Creating empty JSONL file.")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    pass  # Create empty file
                return f"Batch processing completed: 0 items processed (file was created but empty)"
            
            # Execute batch processing
            return self._process_batch_data(agent_name, file_path, data_format, message_template)
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
        except Exception as e:
            raise ValueError(f"Batch processing error: {e}")
    
    def _process_batch_data(self, agent_name: str, file_path: str, data_format: str, message_template: str) -> str:
        """Process batch data from file and execute agent calls in parallel."""
        if data_format.lower() != 'jsonl':
            raise ValueError(f"Unsupported data format: {data_format}. Only 'jsonl' is supported.")
        
        # Read and validate JSONL file
        batch_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            logger.warning(f"Line {line_num}: Expected JSON object, got {type(data).__name__}")
                            continue
                        batch_data.append((line_num, data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num}: Invalid JSON - {e}")
                        continue
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        
        if not batch_data:
            return f"Batch processing completed: 0 items processed (no valid JSON objects found)"
        
        # Validate message template uses valid keys
        template_keys = self._extract_template_keys(message_template)
        sample_data = batch_data[0][1]
        invalid_keys = [key for key in template_keys if key not in sample_data]
        if invalid_keys:
            raise ValueError(f"Message template uses invalid keys: {invalid_keys}. Available keys in data: {list(sample_data.keys())}")
        
        # Execute batch processing in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_running_subagents) as executor:
            # Submit all batch items for parallel execution
            futures = {}
            for line_num, data in batch_data:
                # Render message template with data
                try:
                    rendered_message = self._render_message_template(message_template, data)
                    # Propagate current tracing context into the worker thread
                    task_ctx = copy_context()
                    future = executor.submit(task_ctx.run, self._execute_batch_item_safe, agent_name, rendered_message, line_num)
                    futures[future] = (line_num, data)
                except Exception as e:
                    results.append({
                        'line': line_num,
                        'status': 'error',
                        'error': f"Template rendering failed: {e}",
                        'data': data
                    })
            
            # Collect results as they complete
            for future in as_completed(futures):
                line_num, data = futures[future]
                try:
                    result = future.result()
                    results.append({
                        'line': line_num,
                        'status': 'success',
                        'result': result,
                        'data': data
                    })
                except Exception as e:
                    results.append({
                        'line': line_num,
                        'status': 'error',
                        'error': str(e),
                        'data': data
                    })
        
        # Sort results by line number
        results.sort(key=lambda x: x['line'])
        
        # Generate summary
        total_items = len(results)
        successful_items = len([r for r in results if r['status'] == 'success'])
        failed_items = total_items - successful_items
        
        summary = f"Batch processing completed: {successful_items}/{total_items} items successful, {failed_items} failed"
        
        # Limit detailed results to first 3 items to prevent overly long responses
        displayed_results = results[:3]
        remaining_count = max(0, len(results) - 3)
        
        # Include limited detailed results
        detailed_results = {
            'summary': summary,
            'total_items': total_items,
            'successful_items': successful_items,
            'failed_items': failed_items,
            'displayed_results': displayed_results,
            'remaining_items': remaining_count
        }
        
        if remaining_count > 0:
            detailed_results['note'] = f"Showing first 3 results. {remaining_count} additional results not displayed to keep response concise."
        
        return json.dumps(detailed_results, indent=2, ensure_ascii=False)
    
    def _extract_template_keys(self, template: str) -> List[str]:
        """Extract variable keys from message template."""
        import re
        # Find all {variable_name} patterns
        keys = re.findall(r'\{([^}]+)\}', template)
        return keys
    
    def _render_message_template(self, template: str, data: Dict[str, Any]) -> str:
        """Render message template with data."""
        try:
            return template.format(**data)
        except KeyError as e:
            missing_key = str(e).strip("'\"")
            available_keys = list(data.keys())
            raise ValueError(f"Template key '{missing_key}' not found in data. Available keys: {available_keys}")
    
    def _execute_batch_item_safe(self, agent_name: str, message: str, line_num: int) -> str:
        """Safely execute a single batch item."""
        try:
            logger.info(f"🔄 Processing batch item {line_num} with agent '{agent_name}'")
            result = self.call_sub_agent(agent_name, message)
            logger.info(f"✅ Batch item {line_num} completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Batch item {line_num} failed: {e}")
            raise

    def _execute_sub_agent_from_xml(self, xml_content: str) -> Tuple[str, str]:
        """Execute a sub-agent call from XML content. Returns (agent_name, result)."""
        try:
            # Parse XML using robust parsing
            root = self._parse_xml_content_robust(xml_content)
            
            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                raise ValueError("Missing agent_name in sub-agent XML")
            
            agent_name = (agent_name_elem.text or "").strip()
            
            # Get message
            message_elem = root.find('message')
            if message_elem is None:
                raise ValueError("Missing message in sub-agent XML")
            
            message = (message_elem.text or "").strip()
            
            # Execute sub-agent call
            result = self.call_sub_agent(agent_name, message)
            return agent_name, result
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
    
    def _process_system_prompt(self) -> str:
        """Process the system prompt based on its type."""
        if not self.system_prompt:
            return self._get_default_system_prompt()
        
        try:
            # Use create_dynamic_prompt for enhanced variable rendering
            return self.prompt_handler.create_dynamic_prompt(
                self.system_prompt,
                self
            )
        except Exception as e:
            # Fallback to default prompt if processing fails
            return f"{self._get_default_system_prompt()}\n\nNote: System prompt processing failed: {e}"
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        template = """You are an AI agent named '{{ agent_name }}' built on the Northau framework.

{% if tools %}
You have access to the following tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description or 'No description' }}
{% endfor %}
{% else %}
You currently have no tools available.
{% endif %}

{% if sub_agents %}
You can delegate tasks to the following sub-agents:
{% for sub_agent in sub_agents %}
- {{ sub_agent.name }}: {{ sub_agent.description or 'Specialized agent for ' + sub_agent.name + '-related tasks' }}
{% endfor %}
{% else %}
You currently have no sub-agents available.
{% endif %}

Your goal is to help users accomplish their tasks efficiently by:
1. Understanding the user's request
2. Determining if you can handle it with your available tools
3. Delegating to appropriate sub-agents when their specialized capabilities are needed
4. Executing the necessary actions and providing clear, helpful responses"""
        
        try:
            return self.prompt_handler.create_dynamic_prompt(template, self)
        except Exception:
            # Fallback to simple string formatting if dynamic rendering fails
            return f"""You are an AI agent named '{self.name}' built on the Northau framework.

You have access to the following tools:
{chr(10).join(f"- {tool.name}: {getattr(tool, 'description', 'No description')}" for tool in self.tools)}

You can delegate tasks to the following sub-agents:
{chr(10).join(f"- {name}: Specialized agent for {name}-related tasks" for name in self.sub_agent_factories.keys())}

Your goal is to help users accomplish their tasks efficiently by:
1. Understanding the user's request
2. Determining if you can handle it with your available tools
3. Delegating to appropriate sub-agents when their specialized capabilities are needed
4. Executing the necessary actions and providing clear, helpful responses"""


def create_agent(
    name: Optional[str] = None,
    tools: Optional[List] = None,
    sub_agents: Optional[List[Tuple[str, Callable[[], Agent]]]] = None,
    system_prompt: Optional[str] = None,
    system_prompt_type: str = "string",
    llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
    max_context_tokens: int = 128000,
    max_running_subagents: int = 5,
    error_handler: Optional[Callable] = None,
    retry_attempts: int = 3,
    timeout: int = 300,
    # Token counting parameters
    token_counter: Optional[Callable[[List[Dict[str, str]]], int]] = None,
    # Context parameters
    initial_state: Optional[Dict[str, Any]] = None,
    initial_config: Optional[Dict[str, Any]] = None,
    # MCP parameters
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    # Backward compatibility
    max_context: Optional[int] = None,
    max_response_tokens: Optional[int] = None,
    **llm_kwargs
) -> Agent:
    """Create a new agent with specified configuration."""
    # Handle backward compatibility for max_context
    if max_context is not None:
        logger.warning("max_context is deprecated, use max_context_tokens instead")
        max_context_tokens = max_context
    
    # Handle backward compatibility for max_response_tokens
    if max_response_tokens is not None:
        logger.warning("max_response_tokens is deprecated, use LLMConfig.max_tokens instead")
        if llm_config is None:
            llm_config = {}
        if isinstance(llm_config, dict):
            llm_config.setdefault('max_tokens', max_response_tokens)
        elif hasattr(llm_config, 'max_tokens') and llm_config.max_tokens is None:
            llm_config.max_tokens = max_response_tokens
    
    # Handle llm_config creation with backward compatibility
    if llm_config is None and llm_kwargs:
        # Create LLMConfig from kwargs
        llm_config = LLMConfig(**llm_kwargs)
    
    return Agent(
        name=name,
        tools=tools,
        sub_agents=sub_agents,
        system_prompt=system_prompt,
        system_prompt_type=system_prompt_type,
        llm_config=llm_config,
        max_context_tokens=max_context_tokens,
        max_running_subagents=max_running_subagents,
        error_handler=error_handler,
        retry_attempts=retry_attempts,
        timeout=timeout,
        token_counter=token_counter,
        initial_state=initial_state,
        initial_config=initial_config,
        mcp_servers=mcp_servers,
    )