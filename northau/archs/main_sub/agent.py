"""Agent implementation for the Northau framework."""

from typing import Dict, List, Optional, Tuple, Iterator, Callable, Any, Union
import json
import re
import xml.etree.ElementTree as ET
import logging
import os
from datetime import datetime
from contextvars import copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompt_handler import PromptHandler
from ..llm import LLMConfig
from .agent_context import AgentContext
from langfuse import get_client
from langfuse.openai import openai

# Setup logger for agent execution
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


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
        max_context: int = 100000,
        max_running_subagents: int = 5,
        error_handler: Optional[Callable] = None,
        retry_attempts: int = 3,
        timeout: int = 300,
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
        self.max_context = max_context
        self.max_iterations = max_iterations
        self.max_running_subagents = max_running_subagents
        self.error_handler = error_handler
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        
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
        client_kwargs = self.llm_config.to_client_kwargs()
        self.openai_client = openai.OpenAI(**client_kwargs)
        
        self.langfuse_client = get_client()
    
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
                with self.langfuse_client.start_as_current_span(name=self.name, input=message) as generation:
                    response = self._generate_response(message, context)
                    generation.update(
                        output=response
                    )
                
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
                
                # Set max_tokens if not specified in config
                if 'max_tokens' not in api_params:
                    api_params['max_tokens'] = self.max_context // 4  # Reserve space for context
                
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
                
                logger.info(f"🧠 Calling LLM for agent '{self.name}'...")
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
            with self.langfuse_client.start_as_current_generation(name=f"Tool: {tool_name}", input=parameters) as generation:
                result = tool.execute(**parameters)
                generation.update(
                    output=result,
                )
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
        
        # Execute all calls in parallel with separate thread pools for tools and sub-agents
        with ThreadPoolExecutor() as tool_executor, ThreadPoolExecutor(max_workers=self.max_running_subagents) as subagent_executor:
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
        
        # Strategy 3: Escape HTML/XML content in parameter values
        try:
            def escape_param_content(match):
                param_name = match.group(1)
                param_content = match.group(2)
                
                # Escape HTML/XML content if it contains < >
                if '<' in param_content and '>' in param_content:
                    escaped_content = html.escape(param_content)
                    return f"<{param_name}>{escaped_content}</{param_name}>"
                return match.group(0)
            
            # Find and escape parameter content within parameters block
            escaped_xml = xml_content
            params_match = re.search(r'<parameter>(.*?)</parameter>', xml_content, re.DOTALL)
            if params_match:
                params_content = params_match.group(1)
                # Pattern to match individual parameter tags
                param_pattern = r'<(\w+)>(.*?)</\1>'
                escaped_params = re.sub(param_pattern, escape_param_content, params_content, flags=re.DOTALL)
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
                    
                    # Handle both regular text and CDATA content
                    if param.text is not None:
                        param_value = param.text
                    else:
                        # Handle case where content is in CDATA or mixed content
                        param_value = ''.join(param.itertext()) or ""
                    
                    # Unescape HTML entities in parameter values
                    import html
                    param_value = html.unescape(param_value)
                    parameters[param_name] = self._convert_parameter_type(
                        tool_name, param_name, param_value.strip()
                    )
            
            # Execute tool
            result = self._execute_tool(tool_name, parameters)
            return tool_name, json.dumps(result, indent=2, ensure_ascii=False)
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
    
    def _convert_parameter_type(self, tool_name: str, param_name: str, param_value: str):
        """Convert parameter value to the correct type based on tool schema."""
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
                return param_value.lower() in ('true', '1', 'yes', 'on')
            elif param_type == 'integer':
                return int(param_value)
            elif param_type == 'number':
                return float(param_value)
            elif param_type == 'array':
                # Try to parse as JSON array, fallback to comma-separated
                try:
                    return json.loads(param_value)
                except:
                    return [item.strip() for item in param_value.split(',')]
            elif param_type == 'object':
                return json.loads(param_value)
            else:  # string or unknown type
                return param_value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to convert parameter '{param_name}' of type '{param_type}': {e}")
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
    max_context: int = 100000,
    max_running_subagents: int = 5,
    error_handler: Optional[Callable] = None,
    retry_attempts: int = 3,
    timeout: int = 300,
    # Context parameters
    initial_state: Optional[Dict[str, Any]] = None,
    initial_config: Optional[Dict[str, Any]] = None,
    # MCP parameters
    mcp_servers: Optional[List[Dict[str, Any]]] = None,
    **llm_kwargs
) -> Agent:
    """Create a new agent with specified configuration."""
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
        max_context=max_context,
        max_running_subagents=max_running_subagents,
        error_handler=error_handler,
        retry_attempts=retry_attempts,
        timeout=timeout,
        initial_state=initial_state,
        initial_config=initial_config,
        mcp_servers=mcp_servers,
    )