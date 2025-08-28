"""Refactored Agent implementation for the Northau framework."""

from typing import Dict, List, Optional, Tuple, Callable, Any, Union
import logging
from pathlib import Path

try:
    from langfuse import Langfuse
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

from .prompt_handler import PromptHandler
from ..llm import LLMConfig
from .agent_context import AgentContext
from .execution.executor import Executor
from .utils.token_counter import TokenCounter
from .utils.cleanup_manager import cleanup_manager

# Setup logger for agent execution
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _get_python_type_from_json_schema(json_type: str) -> str:
    """Convert JSON Schema type to Python type string.
    
    Args:
        json_type: JSON Schema type (string, integer, number, boolean, array, object)
        
    Returns:
        Python type string (str, int, float, bool, list, dict)
    """
    type_mapping = {
        'string': 'str',
        'integer': 'int', 
        'number': 'float',
        'boolean': 'bool',
        'array': 'list',
        'object': 'dict'
    }
    return type_mapping.get(json_type, 'str')


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
        retry_attempts: int = 5,
        timeout: int = 300,
        # Token counting parameters
        token_counter: Optional[Callable[[List[Dict[str, str]]], int]] = None,
        # Context parameters
        initial_state: Optional[Dict[str, Any]] = None,
        initial_config: Optional[Dict[str, Any]] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        # MCP parameters
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        # Stop tools parameters
        stop_tools: Optional[List[str]] = None,
        # Hook parameters
        after_model_hooks: Optional[List[Callable]] = None,
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
        
        # Initialize stop tools list
        self.stop_tools = set(stop_tools or [])
        
        # Store the hooks
        self.after_model_hooks = after_model_hooks
        
        # Handle LLM configuration
        self.llm_config = self._setup_llm_config(llm_config)
        
        # Initialize MCP tools if servers are configured
        if mcp_servers:
            self._initialize_mcp_tools(mcp_servers)
        
        # Build tool registry for quick lookup (after MCP tools are added)
        self.tool_registry = {tool.name: tool for tool in self.tools}
        
        # Conversation history
        self.history = []
        
        # Initialize prompt handler
        self.prompt_handler = PromptHandler()
        
        # Process system prompt
        self.processed_system_prompt = str(self._process_system_prompt())
        
        # Initialize context data
        self.initial_state = initial_state or {}
        self.initial_config = initial_config or {}
        self.initial_context = initial_context or {}
        
        # Initialize OpenAI client
        if openai is not None:
            client_kwargs = self.llm_config.to_client_kwargs()
            self.openai_client = openai.OpenAI(**client_kwargs)
        else:
            self.openai_client = None
        
        # Initialize Langfuse client
        self.langfuse_client = None
        if LANGFUSE_AVAILABLE:
            try:
                import os
                # Check if required environment variables are set
                public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
                secret_key = os.getenv('LANGFUSE_SECRET_KEY')
                host = os.getenv('LANGFUSE_HOST')
                
                if public_key and secret_key:
                    # Initialize with explicit parameters to ensure proper configuration
                    self.langfuse_client = Langfuse(
                        public_key=public_key,
                        secret_key=secret_key,
                        host=host
                    )
                    logger.info(f"✅ Langfuse client initialized with host: {host or 'default'}")
                else:
                    logger.warning("⚠️ Langfuse environment variables not found, tracing disabled")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Langfuse client: {e}")
                self.langfuse_client = None
        
        # Setup token counter
        if token_counter:
            # Convert callable to TokenCounter if needed
            if callable(token_counter) and not isinstance(token_counter, TokenCounter):
                class CallableTokenCounter(TokenCounter):
                    def __init__(self, counter_func):
                        self._counter_func = counter_func
                    def count_tokens(self, messages):
                        return self._counter_func(messages)
                self.token_counter = CallableTokenCounter(token_counter)
            else:
                self.token_counter = token_counter
        else:
            self.token_counter = TokenCounter()
        
        # Initialize executor with all components
        self.executor = Executor(
            agent_name=self.name,
            tool_registry=self.tool_registry,
            sub_agent_factories=self.sub_agent_factories,
            stop_tools=self.stop_tools,
            openai_client=self.openai_client,
            llm_config=self.llm_config,
            max_iterations=max_iterations,
            max_context_tokens=max_context_tokens,
            max_running_subagents=max_running_subagents,
            retry_attempts=retry_attempts,
            token_counter=self.token_counter,
            langfuse_client=self.langfuse_client,
            after_model_hooks=self.after_model_hooks
        )
        # Register this agent for cleanup
        cleanup_manager.register_agent(self)
    
    @staticmethod
    def _load_prompt_template(prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        current_dir = Path(__file__).parent
        prompts_dir = current_dir / "prompts"
        prompt_file = prompts_dir / f"{prompt_name}.j2"
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.warning(f"Prompt file not found: {prompt_file}")
            return ""
        except Exception as e:
            logger.error(f"Error loading prompt template {prompt_name}: {e}")
            return ""
    
    def run(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        dump_trace_path: Optional[str] = None,
        return_final_state: bool = False,
    ) -> str | tuple[str, dict[str, Any]]:
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
        
        merged_context = {**self.initial_context}
        if context:
            merged_context.update(context)
        
        # Create agent context
        with AgentContext(state=merged_state, config=merged_config) as ctx:
            # Setup context modification callback to refresh system prompt
            def on_context_modified():
                # Reset processed system prompt to force regeneration
                self.processed_system_prompt = str(self._process_system_prompt())
            
            ctx.add_modification_callback(on_context_modified)
            
            if history:
                self.history = history.copy()
            
            # Add user message to history
            self.history.append({"role": "user", "content": message})
            
            # Add system prompt as first message if history is empty (first run)
            if len(self.history) == 1:  # Only user message exists
                self.history.insert(0, {"role": "system", "content": self._build_system_prompt_with_capabilities(merged_context)})
            
            try:
                # Generate response using the executor
                if self.langfuse_client:
                    try:
                        from datetime import datetime
                        
                        # Create main span for the agent execution
                        with self.langfuse_client.start_as_current_span(
                            name=f"agent_{self.name}",
                            input=message,
                            metadata={
                                "agent_name": self.name,
                                "max_iterations": self.max_iterations,
                                "model": getattr(self.llm_config, 'model', None),
                                "system_prompt_type": self.system_prompt_type,
                                "timestamp": datetime.now().isoformat()
                            }
                        ) as span:
                            logger.info(f"📊 Created Langfuse span for agent: {self.name}")
                            
                            # Execute the agent task - executor now returns response and updated messages
                            response, updated_messages = self.executor.execute(self.history, dump_trace_path)
                            
                            # Update history with all new messages generated during execution
                            self.history = updated_messages
                            
                            # Update span with response
                            self.langfuse_client.update_current_span(
                                output=response,
                                metadata={
                                    "response_length": len(response),
                                    "history_length": len(self.history),
                                    "execution_completed": True
                                }
                            )
                            
                            logger.info(f"📤 Langfuse span completed for agent: {self.name}")
                        
                        # Flush to ensure data is sent
                        self.langfuse_client.flush()
                        logger.info(f"📤 Langfuse data flushed successfully")
                        
                    except Exception as langfuse_error:
                        logger.warning(f"⚠️ Langfuse tracing failed: {langfuse_error}")
                        response, updated_messages = self.executor.execute(self.history, dump_trace_path)
                        # Update history with all new messages even when Langfuse fails
                        self.history = updated_messages
                else:
                    response, updated_messages = self.executor.execute(self.history, dump_trace_path)
                    # Update history with all new messages generated during execution
                    self.history = updated_messages
                
                # Add final assistant response to history if not already included
                if not self.history or self.history[-1]["role"] != "assistant" or self.history[-1]["content"] != response:
                    self.history.append({"role": "assistant", "content": response})
                
                logger.info(f"✅ Agent '{self.name}' completed execution")
                
                if return_final_state:
                    final_state = ctx.state.copy()
                    return response, final_state
                else:
                    return response
                
            except Exception as e:
                logger.error(f"❌ Agent '{self.name}' encountered error: {e}")
                
                if self.error_handler:
                    error_response = self.error_handler(e, self, merged_context)
                    # Always add error response to history
                    self.history.append({"role": "assistant", "content": error_response})
                    if return_final_state:
                        final_state = ctx.state.copy()
                        return error_response, final_state
                    else:
                        return error_response
                else:
                    # Even if no error handler, add an error message to history for completeness
                    error_message = f"Error: {str(e)}"
                    self.history.append({"role": "assistant", "content": error_message})
                    raise
    
    def add_tool(self, tool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        self.tool_registry[tool.name] = tool
        self.executor.add_tool(tool)
    
    def add_sub_agent(self, name: str, agent_factory: Callable[[], 'Agent']) -> None:
        """Add a sub-agent factory for delegation."""
        self.sub_agent_factories[name] = agent_factory
        self.executor.add_sub_agent(name, agent_factory)
    
    def delegate_task(self, task: str, sub_agent_name: str, context: Optional[Dict] = None) -> str:
        """Explicitly delegate a task to a sub-agent."""
        return self.executor.subagent_manager.call_sub_agent(sub_agent_name, task, context)
    
    def call_sub_agent(self, sub_agent_name: str, message: str, context: Optional[Dict] = None) -> str:
        """Call a sub-agent like a tool call."""
        return self.executor.subagent_manager.call_sub_agent(sub_agent_name, message, context)
    
    def _setup_llm_config(self, llm_config: Optional[Union[LLMConfig, Dict[str, Any]]]) -> LLMConfig:
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
            # Don't silently continue - let the caller know there was an issue
            import traceback
            logger.debug(f"MCP initialization traceback: {traceback.format_exc()}")
    
    def _process_system_prompt(self) -> str:
        """Process the system prompt based on its type."""
        if not self.system_prompt:
            return self._get_default_system_prompt()
        
        try:
            # Use create_dynamic_prompt for enhanced variable rendering
            return self.prompt_handler.create_dynamic_prompt(
                self.system_prompt,
                self,
                template_type=self.system_prompt_type
            )
        except Exception as e:
            # Fallback to default prompt if processing fails
            return f"{self._get_default_system_prompt()}\\n\\nNote: System prompt processing failed: {e}"
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for the agent."""
        template = self._load_prompt_template("default_system_prompt")
        
        if not template:
            # Fallback template if file loading fails
            raise ValueError("Default system prompt template not found")
        
        try:
            return self.prompt_handler.create_dynamic_prompt(template, self)
        except Exception:
            # Fallback to simple string formatting if dynamic rendering fails
            fallback_template = self._load_prompt_template("fallback_system_prompt")
            if fallback_template:
                tools_list = chr(10).join(f"- {tool.name}: {getattr(tool, 'description', 'No description')}" for tool in self.tools)
                sub_agents_list = chr(10).join(f"- {name}: Specialized agent for {name}-related tasks" for name in self.sub_agent_factories.keys())
                return fallback_template.replace("{{ agent_name }}", self.name).replace("{{ tools_list }}", tools_list).replace("{{ sub_agents_list }}", sub_agents_list)
            else:
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
    
    def _build_system_prompt_with_capabilities(self, runtime_context: Optional[Dict] = None) -> str:
        """Build system prompt including tool and sub-agent capabilities."""
        # Check if context has been modified and refresh the processed system prompt
        from .agent_context import get_context
        current_context = get_context()
        if current_context and current_context.is_modified():
            self.processed_system_prompt = str(self._process_system_prompt())
            current_context.reset_modification_flag()
        
        # Re-process system prompt with runtime context if provided
        if runtime_context and self.system_prompt:
            try:
                # Merge runtime context with default context
                default_context = self.prompt_handler.get_default_context(self)
                merged_context = {**default_context, **runtime_context}
                base_prompt = str(self.prompt_handler.create_dynamic_prompt(
                    self.system_prompt, self, additional_context=merged_context, template_type=self.system_prompt_type
                ))
            except Exception as e:
                # Fallback to processed system prompt if runtime processing fails
                logger.error(f"❌ Error processing system prompt: {e}")
                base_prompt = str(self.processed_system_prompt)
        else:
            base_prompt = str(self.processed_system_prompt)

        # Load templates from files
        tools_template = self._load_prompt_template("tools_template")
        sub_agents_template = self._load_prompt_template("sub_agents_template")
        
        # Provide fallback templates if loading fails
        if not tools_template:
            raise ValueError("Tools template not found")
        
        if not sub_agents_template:
            sub_agents_template = """
{% if sub_agents %}

## Available Sub-Agents
You can delegate tasks to specialized sub-agents:
{% for sub_agent in sub_agents %}

### {{ sub_agent.name }}
{{ sub_agent.description or 'Specialized agent for ' + sub_agent.name + '-related tasks' }}
Usage:
<sub_agent>
  <agent_name>{{ sub_agent.name }}</agent_name>
  <message>task description</message>
</sub_agent>
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
                            python_type = _get_python_type_from_json_schema(param_type)
                            
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
        
        # Add tool execution instructions from file
        tool_execution_instructions = self._load_prompt_template("tool_execution_instructions")
        if tool_execution_instructions:
            base_prompt += tool_execution_instructions
        else:
            raise ValueError("Tool execution instructions template not found")
        
        return base_prompt
    
    def _build_system_prompt_with_capabilities_fallback(self, base_prompt: str) -> str:
        """Fallback method for building capabilities documentation."""
        # Add tool documentation
        if self.tools:
            tool_docs = []
            tool_docs.append("\\n## Available Tools")
            tool_docs.append("You can use tools by including XML blocks in your response:")
            
            for tool in self.tools:
                tool_docs.append(f"\\n### {tool.name}")
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
                    python_type = _get_python_type_from_json_schema(param_type)
                    
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
            
            base_prompt += "\\n".join(tool_docs)
            base_prompt += "\\n\\nIMPORTANT: use </parameter> to end the parameters block."
            base_prompt += "\\n\\nCRITICAL TOOL EXECUTION INSTRUCTIONS:"
            base_prompt += "\\nIMPORTANT: After outputting any tool call XML block (e.g., <tool_use>, <sub_agent>, etc.), you MUST STOP and WAIT for the tool execution results before continuing your response. Do NOT continue generating text after tool calls until you receive the results."
        
        # Add sub-agent documentation
        if self.sub_agent_factories:
            sub_agent_docs = []
            sub_agent_docs.append("\\n## Available Sub-Agents")
            sub_agent_docs.append("You can delegate tasks to specialized sub-agents:")
            
            for name in self.sub_agent_factories.keys():
                sub_agent_docs.append(f"\\n### {name}")
                sub_agent_docs.append(f"Specialized agent for {name}-related tasks")
                sub_agent_docs.append("Usage:")
                sub_agent_docs.append("<sub_agent>")
                sub_agent_docs.append(f"  <agent_name>{name}</agent_name>")
                sub_agent_docs.append("  <message>task description</message>")
                sub_agent_docs.append("</sub_agent>")
            
            base_prompt += "\\n".join(sub_agent_docs)
            base_prompt += "\\n\\nEXECUTION FLOW REMINDER:"
            base_prompt += "\\n1. When you output XML tool/agent blocks, STOP your response immediately"
            base_prompt += "\\n2. Wait for the execution results to be provided to you"
            base_prompt += "\\n3. Only then continue with analysis of the results and next steps"
            base_prompt += "\\n4. Never generate additional content after XML blocks until results are returned"
        
        return base_prompt
    
    def _cleanup_agent(self) -> None:
        """Clean up this agent and all its running sub-agents."""
        logger.info(f"🧹 Cleaning up agent '{self.name}' and its sub-agents...")
        self.executor.cleanup()
        logger.info(f"✅ Agent '{self.name}' cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup when agent is garbage collected."""
        try:
            self._cleanup_agent()
        except Exception:
            pass  # Avoid exceptions during garbage collection


def create_agent(
    name: Optional[str] = None,
    tools: Optional[List] = None,
    sub_agents: Optional[List[Tuple[str, Callable[[], Agent]]]] = None,
    system_prompt: Optional[str] = None,
    system_prompt_type: str = "string",
    llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
    max_iterations: int = 100,
    max_context_tokens: int = 128000,
    max_running_subagents: int = 5,
    error_handler: Optional[Callable] = None,
    retry_attempts: int = 5,
    timeout: int = 300,
    # Token counting parameters
    token_counter: Optional[Callable[[List[Dict[str, str]]], int]] = None,
    # Context parameters
    initial_state: Optional[Dict[str, Any]] = None,
    initial_config: Optional[Dict[str, Any]] = None,
    initial_context: Optional[Dict[str, Any]] = None,
    # MCP parameters
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    # Backward compatibility
    max_context: Optional[int] = None,
    max_response_tokens: Optional[int] = None,
    # Stop tools parameters
    stop_tools: Optional[List[str]] = None,
    # Hook parameters
    after_model_hooks: Optional[List[Callable]] = None,
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
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        max_running_subagents=max_running_subagents,
        error_handler=error_handler,
        retry_attempts=retry_attempts,
        timeout=timeout,
        token_counter=token_counter,
        initial_state=initial_state,
        initial_config=initial_config,
        initial_context=initial_context,
        mcp_servers=mcp_servers,
        stop_tools=stop_tools,
        after_model_hooks=after_model_hooks,
    )