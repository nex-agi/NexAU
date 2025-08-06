"""Agent implementation for the Northau framework."""

from typing import Dict, List, Optional, Tuple, Iterator, Callable, Any, Union
import json
import re
import xml.etree.ElementTree as ET
import logging
from datetime import datetime
from .prompt_handler import PromptHandler
from ..llm import LLMConfig
from .agent_context import AgentContext

# Setup logger for agent execution
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class Agent:
    """Main agent class that handles task execution and sub-agent delegation."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        tools: Optional[List] = None,
        sub_agents: Optional[List[Tuple[str, 'Agent']]] = None,
        system_prompt: Optional[str] = None,
        system_prompt_type: str = "string",
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
        max_context: int = 100000,
        error_handler: Optional[Callable] = None,
        retry_attempts: int = 3,
        timeout: int = 300,
        # Context parameters
        initial_state: Optional[Dict[str, Any]] = None,
        initial_config: Optional[Dict[str, Any]] = None,
        # Deprecated parameters (for backward compatibility)
        model: Optional[str] = None,
        model_base_url: Optional[str] = None
    ):
        """Initialize an agent with specified configuration."""
        self.name = name or f"agent_{id(self)}"
        self.tools = tools or []
        self.sub_agents = dict(sub_agents or [])
        self.system_prompt = system_prompt
        self.system_prompt_type = system_prompt_type
        self.max_context = max_context
        self.error_handler = error_handler
        self.retry_attempts = retry_attempts
        self.timeout = timeout
        
        # Initialize context data
        self.initial_state = initial_state or {}
        self.initial_config = initial_config or {}
        
        # Handle LLM configuration
        self.llm_config = self._setup_llm_config(llm_config, model, model_base_url)
        
        # Build tool registry for quick lookup
        self.tool_registry = {tool.name: tool for tool in self.tools}
        
        # Conversation history
        self.history = []
        
        # Initialize prompt handler
        self.prompt_handler = PromptHandler()
        
        # Process system prompt
        self.processed_system_prompt = self._process_system_prompt()
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                client_kwargs = self.llm_config.to_client_kwargs()
                self.openai_client = OpenAI(**client_kwargs)
            except Exception:
                # If OpenAI client fails to initialize, continue without it
                pass
    
    def _setup_llm_config(
        self, 
        llm_config: Optional[Union[LLMConfig, Dict[str, Any]]], 
        model: Optional[str], 
        model_base_url: Optional[str]
    ) -> LLMConfig:
        """Setup LLM configuration with backward compatibility."""
        if llm_config is None:
            # Create from deprecated parameters or defaults
            return LLMConfig(
                model=model or "gpt-4",
                base_url=model_base_url
            )
        elif isinstance(llm_config, dict):
            # Create from dictionary
            return LLMConfig(**llm_config)
        elif isinstance(llm_config, LLMConfig):
            # Use provided config
            return llm_config
        else:
            raise ValueError(f"Invalid llm_config type: {type(llm_config)}")
    
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
    
    def stream(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        context: Optional[Dict] = None,
        state: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Iterator[str]:
        """Stream agent response in chunks with real-time tool execution."""
        logger.info(f"🌊 Agent '{self.name}' starting streaming execution")
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
                # Stream the response generation
                for chunk in self._stream_response(message, context):
                    yield chunk
                
                logger.info(f"✅ Agent '{self.name}' completed streaming execution")
                    
            except Exception as e:
                logger.error(f"❌ Agent '{self.name}' streaming error: {e}")
                if self.error_handler:
                    error_response = self.error_handler(e, self, context)
                    self.history.append({"role": "assistant", "content": error_response})
                    yield error_response
                else:
                    raise
    
    def add_tool(self, tool) -> None:
        """Add a tool to the agent."""
        self.tools.append(tool)
        self.tool_registry[tool.name] = tool
    
    def add_sub_agent(self, name: str, agent: 'Agent') -> None:
        """Add a sub-agent for delegation."""
        self.sub_agents[name] = agent
    
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
            max_iterations = 10  # Prevent infinite loops
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
                
                # Debug logging for LLM messages
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Request Messages for agent '{self.name}':")
                    for i, msg in enumerate(messages):
                        # logger.info(f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
                        logger.info(f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content']}")
                
                logger.info(f"🧠 Calling LLM for agent '{self.name}'...")
                response = self.openai_client.chat.completions.create(**api_params)
                assistant_response = response.choices[0].message.content
                
                # Debug logging for LLM response
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Response for agent '{self.name}': {assistant_response}")
                
                logger.info(f"💬 LLM Response for agent '{self.name}': {assistant_response}")
                
                # Store this as the latest response (potential final response)
                final_response = assistant_response
                
                # Check if response contains tool calls or sub-agent calls
                has_tool_calls = bool(re.search(r'<tool_use>.*?</tool_use>', assistant_response, re.DOTALL))
                has_sub_agent_calls = bool(re.search(r'<sub-agent>.*?</sub-agent>', assistant_response, re.DOTALL))
                
                logger.info(f"🔍 Analysis for agent '{self.name}': tool_calls={has_tool_calls}, sub_agent_calls={has_sub_agent_calls}")
                
                if not has_tool_calls and not has_sub_agent_calls:
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
    
    def _stream_response(self, message: str, context: Optional[Dict] = None) -> Iterator[str]:
        """Stream response using OpenAI API with XML-based tool and sub-agent calls."""
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
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                # Call OpenAI API with streaming enabled
                api_params = self.llm_config.to_openai_params()
                api_params['messages'] = messages
                api_params['stream'] = True
                
                # Set max_tokens if not specified in config
                if 'max_tokens' not in api_params:
                    api_params['max_tokens'] = self.max_context // 4  # Reserve space for context
                
                # Debug logging for LLM messages
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Streaming Request Messages for agent '{self.name}':")
                    for i, msg in enumerate(messages):
                        logger.info(f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
                
                response_stream = self.openai_client.chat.completions.create(**api_params)
                
                # Collect streamed response
                assistant_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        assistant_response += content
                        yield content
                
                # Debug logging for complete streamed response
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Streaming Response for agent '{self.name}': {assistant_response}")
                
                # Check if response contains tool calls or sub-agent calls
                has_tool_calls = bool(re.search(r'<tool_use>.*?</tool_use>', assistant_response, re.DOTALL))
                has_sub_agent_calls = bool(re.search(r'<sub-agent>.*?</sub-agent>', assistant_response, re.DOTALL))
                
                if not has_tool_calls and not has_sub_agent_calls:
                    # No more commands to execute, we're done
                    # Add final response to history
                    self.history.append({"role": "assistant", "content": assistant_response})
                    break
                
                # Add the assistant's original response to conversation
                messages.append({"role": "assistant", "content": assistant_response})
                
                # Process tool calls and sub-agent calls
                processed_response = self._process_xml_calls(assistant_response)
                
                # Yield the tool/sub-agent execution results
                tool_results = processed_response.replace(assistant_response, "").strip()
                if tool_results:
                    yield tool_results
                
                # Add final response to history (will be updated in next iteration or at end)
                if iteration == 0:  # First iteration
                    self.history.append({"role": "assistant", "content": processed_response})
                else:  # Update existing entry
                    self.history[-1]["content"] = processed_response
                
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
            
            if iteration >= max_iterations:
                yield "\n\n[Note: Maximum iteration limit reached]"
                
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Error calling OpenAI API: {e}") from e
    
    
    def call_sub_agent(self, sub_agent_name: str, message: str, context: Optional[Dict] = None) -> str:
        """Call a sub-agent like a tool call."""
        from .agent_context import get_context
        
        logger.info(f"🤖➡️🤖 Agent '{self.name}' calling sub-agent '{sub_agent_name}' with message: {message}")
        
        if sub_agent_name not in self.sub_agents:
            error_msg = f"Sub-agent '{sub_agent_name}' not found"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        sub_agent = self.sub_agents[sub_agent_name]
        try:
            # Pass current agent context state and config to sub-agent
            current_context = get_context()
            if current_context:
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
  <parameters>
{% if tool.parameters %}
{% for param in tool.parameters %}
    <{{ param.name }}>{{ param.description }}{% if not param.required %} (optional, type: {{ param.type }}{% if param.default %}, default: {{ param.default }}{% endif %}){% else %} (required, type: {{ param.type }}){% endif %}</{{ param.name }}>
{% endfor %}
{% endif %}
  </parameters>
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
        
        base_prompt += "\n\nWhen you use tools or sub-agents, include the XML blocks in your response and I will execute them and provide the results."
        
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
                tool_docs.append("  <parameters>")
                
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
                
                tool_docs.append("  </parameters>")
                tool_docs.append("</tool_use>")
            
            base_prompt += "\n".join(tool_docs)
        
        # Add sub-agent documentation
        if self.sub_agents:
            sub_agent_docs = []
            sub_agent_docs.append("\n## Available Sub-Agents")
            sub_agent_docs.append("You can delegate tasks to specialized sub-agents:")
            
            for name, sub_agent in self.sub_agents.items():
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
        """Process XML tool calls and sub-agent calls in the response."""
        processed_response = response
        tool_results = []
        
        # Process tool calls
        tool_pattern = r'<tool_use>(.*?)</tool_use>'
        tool_matches = re.findall(tool_pattern, response, re.DOTALL)
        
        for tool_xml in tool_matches:
            try:
                tool_name, result = self._execute_tool_from_xml(tool_xml)
                logger.info(f"📤 Tool '{tool_name}' result: {result}")
                tool_results.append(f"<tool_result>\n<tool_name>{tool_name}</tool_name>\n<result>{result}</result>\n</tool_result>")
            except Exception as e:
                # Extract tool name for error reporting
                try:
                    root = ET.fromstring(f"<root>{tool_xml}</root>")
                    tool_name_elem = root.find('tool_name')
                    tool_name = (tool_name_elem.text or "").strip() if tool_name_elem is not None else "unknown"
                except:
                    tool_name = "unknown"
                logger.error(f"❌ Tool '{tool_name}' error: {e}")
                tool_results.append(f"<tool_result>\n<tool_name>{tool_name}</tool_name>\n<error>{str(e)}</error>\n</tool_result>")
        
        # Process sub-agent calls
        sub_agent_pattern = r'<sub-agent>(.*?)</sub-agent>'
        sub_agent_matches = re.findall(sub_agent_pattern, response, re.DOTALL)
        
        for sub_agent_xml in sub_agent_matches:
            try:
                agent_name, result = self._execute_sub_agent_from_xml(sub_agent_xml)
                logger.info(f"📤 Sub-agent '{agent_name}' result: {result}")
                tool_results.append(f"<tool_result>\n<tool_name>{agent_name}_sub_agent</tool_name>\n<result>{result}</result>\n</tool_result>")
            except Exception as e:
                # Extract agent name for error reporting
                try:
                    root = ET.fromstring(f"<root>{sub_agent_xml}</root>")
                    agent_name_elem = root.find('agent_name')
                    agent_name = (agent_name_elem.text or "").strip() if agent_name_elem is not None else "unknown"
                except:
                    agent_name = "unknown"
                logger.error(f"❌ Sub-agent '{agent_name}' error: {e}")
                tool_results.append(f"<tool_result>\n<tool_name>{agent_name}_sub_agent</tool_name>\n<error>{str(e)}</error>\n</tool_result>")
        
        # Append tool results to the original response
        if tool_results:
            processed_response += "\n\n" + "\n\n".join(tool_results)
        
        return processed_response
    
    def _execute_tool_from_xml(self, xml_content: str) -> Tuple[str, str]:
        """Execute a tool from XML content. Returns (tool_name, result)."""
        try:
            # Parse XML - first try as-is, then try with HTML entity escaping
            try:
                root = ET.fromstring(f"<root>{xml_content}</root>")
            except ET.ParseError as e:
                logger.warning(f"Initial XML parsing failed: {e}. Attempting with CDATA wrapping...")
                
                # Try wrapping parameter content in CDATA to preserve HTML/XML content
                try:
                    # First, extract and wrap parameter contents in CDATA
                    import re
                    
                    # Find parameter blocks and wrap their content in CDATA
                    def wrap_param_content(match):
                        param_name = match.group(1)
                        param_content = match.group(2)
                        # Only wrap in CDATA if content contains XML/HTML-like structures
                        if '<' in param_content and '>' in param_content:
                            return f"<{param_name}><![CDATA[{param_content}]]></{param_name}>"
                        return match.group(0)
                    
                    # Pattern to match parameter tags with content
                    param_pattern = r'<(\w+)>(.*?)</\1>'
                    cdata_wrapped = re.sub(param_pattern, wrap_param_content, xml_content, flags=re.DOTALL)
                    
                    root = ET.fromstring(f"<root>{cdata_wrapped}</root>")
                    
                except ET.ParseError:
                    # Fallback: Try escaping common problematic characters
                    import html
                    escaped_content = html.escape(xml_content, quote=False)
                    # Unescape the XML tags we need
                    escaped_content = escaped_content.replace("&lt;", "<").replace("&gt;", ">")
                    root = ET.fromstring(f"<root>{escaped_content}</root>")
            
            # Get tool name
            tool_name_elem = root.find('tool_name')
            if tool_name_elem is None:
                raise ValueError("Missing tool_name in tool_use XML")
            
            tool_name = (tool_name_elem.text or "").strip()
            
            # Get parameters
            parameters = {}
            params_elem = root.find('parameters')
            if params_elem is not None:
                for param in params_elem:
                    param_name = param.tag
                    param_value = param.text or ""
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
    
    def _execute_sub_agent_from_xml(self, xml_content: str) -> Tuple[str, str]:
        """Execute a sub-agent call from XML content. Returns (agent_name, result)."""
        try:
            # Parse XML
            root = ET.fromstring(f"<root>{xml_content}</root>")
            
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
{chr(10).join(f"- {name}: Specialized agent for {name}-related tasks" for name in self.sub_agents.keys())}

Your goal is to help users accomplish their tasks efficiently by:
1. Understanding the user's request
2. Determining if you can handle it with your available tools
3. Delegating to appropriate sub-agents when their specialized capabilities are needed
4. Executing the necessary actions and providing clear, helpful responses"""


def create_agent(
    name: Optional[str] = None,
    tools: Optional[List] = None,
    sub_agents: Optional[List[Tuple[str, Agent]]] = None,
    system_prompt: Optional[str] = None,
    system_prompt_type: str = "string",
    llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None,
    max_context: int = 100000,
    error_handler: Optional[Callable] = None,
    retry_attempts: int = 3,
    timeout: int = 300,
    # Context parameters
    initial_state: Optional[Dict[str, Any]] = None,
    initial_config: Optional[Dict[str, Any]] = None,
    # Deprecated parameters (for backward compatibility)
    model: Optional[str] = None,
    model_base_url: Optional[str] = None,
    **llm_kwargs
) -> Agent:
    """Create a new agent with specified configuration."""
    # Handle llm_config creation with backward compatibility
    if llm_config is None and (model is not None or model_base_url is not None or llm_kwargs):
        # Create LLMConfig from deprecated parameters and kwargs
        llm_params = {}
        if model is not None:
            llm_params['model'] = model
        if model_base_url is not None:
            llm_params['base_url'] = model_base_url
        llm_params.update(llm_kwargs)
        llm_config = LLMConfig(**llm_params)
    
    return Agent(
        name=name,
        tools=tools,
        sub_agents=sub_agents,
        system_prompt=system_prompt,
        system_prompt_type=system_prompt_type,
        llm_config=llm_config,
        max_context=max_context,
        error_handler=error_handler,
        retry_attempts=retry_attempts,
        timeout=timeout,
        initial_state=initial_state,
        initial_config=initial_config,
        model=model,  # Keep for backward compatibility
        model_base_url=model_base_url  # Keep for backward compatibility
    )