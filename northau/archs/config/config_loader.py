"""Configuration loading system for agents and tools."""

import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from ..main_sub import create_agent, Agent
from ..tool import Tool
from ..llm import LLMConfig
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass


def load_agent_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    template_context: Optional[Dict[str, Any]] = None
) -> Agent:
    """
    Load agent configuration from YAML file.
    
    Args:
        config_path: Path to the agent configuration YAML file
        overrides: Dictionary of configuration overrides
        template_context: Context variables for Jinja template rendering
    
    Returns:
        Configured Agent instance
    """
    try:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
        
        # Load YAML configuration
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ConfigError(f"Empty or invalid configuration file: {config_path}")
        
        # Apply overrides
        if overrides:
            config.update(overrides)
        
        # Extract agent configuration
        agent_name = config.get('name', 'configured_agent')
        max_context = config.get('max_context', 100000)  # Backward compatibility
        max_context_tokens = config.get('max_context_tokens', 128000)
        max_running_subagents = config.get('max_running_subagents', 5)
        system_prompt = config.get('system_prompt')
        system_prompt_type = config.get('system_prompt_type', 'string')
        initial_context = config.get('context', {})
        initial_state = config.get('state', {})
        initial_config = config.get('config', {})
        initial_config.update(initial_context)
        stop_tools = config.get('stop_tools', [])
        
        # Handle after_model_hooks configuration
        after_model_hooks = None
        if 'after_model_hooks' in config:
            hooks_config = config['after_model_hooks']
            after_model_hooks = []
            
            if not isinstance(hooks_config, list):
                raise ConfigError("'after_model_hooks' must be a list")
            
            for i, hook_config in enumerate(hooks_config):
                try:
                    if isinstance(hook_config, str):
                        # Simple import string format
                        hook_func = import_from_string(hook_config)
                        after_model_hooks.append(hook_func)
                    elif isinstance(hook_config, dict):
                        # Dictionary format with import and optional parameters
                        import_string = hook_config.get('import')
                        if not import_string:
                            raise ConfigError(f"Hook {i} missing 'import' field")
                        
                        hook_func = import_from_string(import_string)
                        
                        # Check if there are parameters to pass
                        params = hook_config.get('params', {})
                        if params:
                            # For hooks that are factories (like create_* functions), call them with params
                            if callable(hook_func):
                                hook_instance = hook_func(**params)
                                after_model_hooks.append(hook_instance)
                            else:
                                raise ConfigError(f"Hook {i} is not callable and cannot accept parameters")
                        else:
                            after_model_hooks.append(hook_func)
                    else:
                        raise ConfigError(f"Hook {i} must be a string or dictionary")
                except Exception as e:
                    raise ConfigError(f"Error loading hook {i}: {e}")
        
        # Handle token counter configuration
        token_counter = None
        if 'token_counter' in config:
            token_counter_config = config['token_counter']
            if isinstance(token_counter_config, str):
                # Import string format: "module.path:function_name"
                token_counter = import_from_string(token_counter_config)
            elif isinstance(token_counter_config, dict):
                # Dictionary format with import and optional parameters
                import_string = token_counter_config.get('import')
                if not import_string:
                    raise ConfigError("Token counter configuration missing 'import' field")
                
                # Import the function/class
                token_counter_func = import_from_string(import_string)
                
                # Check if there are parameters to pass
                params = token_counter_config.get('params', {})
                if params:
                    # Create a wrapper function with the parameters
                    def configured_token_counter(messages):
                        return token_counter_func(messages, **params)
                    token_counter = configured_token_counter
                else:
                    token_counter = token_counter_func
            else:
                raise ConfigError("Token counter configuration must be a string or dictionary")
        
        # Handle LLM configuration
        llm_config = None
        if 'llm_config' in config:
            # New format with llm_config section
            llm_config = LLMConfig(**config['llm_config'])
        else:
            # Backward compatibility: extract LLM params from root level
            llm_params = {}
            for key in ['model', 'base_url', 'model_base_url', 'api_key', 'temperature', 
                       'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 
                       'timeout', 'max_retries', 'debug']:
                value = config.get(key)
                if value is not None:
                    # Handle model_base_url -> base_url mapping
                    if key == 'model_base_url':
                        llm_params['base_url'] = value
                    else:
                        llm_params[key] = value
            
            if llm_params:
                llm_config = LLMConfig(**llm_params)
            else:
                # Default LLM config
                llm_config = LLMConfig()
        
        # Load tools
        tools = []
        tool_configs = config.get('tools', [])
        for tool_config in tool_configs:
            try:
                tool = load_tool_from_config(tool_config, path.parent)
                tools.append(tool)
            except Exception as e:
                raise ConfigError(f"Error loading tool '{tool_config.get('name', 'unknown')}': {e}")
        
        # Load sub-agents
        sub_agents = []
        sub_agent_configs = config.get('sub_agents', [])
        for sub_config in sub_agent_configs:
            try:
                sub_agent = load_sub_agent_from_config(sub_config, path.parent)
                sub_agents.append(sub_agent)
            except Exception as e:
                raise ConfigError(f"Error loading sub-agent '{sub_config.get('name', 'unknown')}': {e}")
        
        system_prompt = path.parent / system_prompt
        # Create agent
        agent = create_agent(
            name=agent_name,
            tools=tools,
            sub_agents=sub_agents,
            system_prompt=system_prompt,
            system_prompt_type=system_prompt_type,
            llm_config=llm_config,
            max_context_tokens=max_context_tokens,
            max_running_subagents=max_running_subagents,
            token_counter=token_counter,
            initial_state=initial_state,
            initial_config=initial_config,
            initial_context=initial_context,
            stop_tools=stop_tools,
            after_model_hooks=after_model_hooks
        )
        
        # Apply template context if provided and using Jinja templates
        if system_prompt_type == "jinja":
            # Update the agent's prompt handler context
            if hasattr(agent, 'prompt_handler'):
                agent.processed_system_prompt = agent.prompt_handler.process_prompt(
                    system_prompt, system_prompt_type, template_context
                )
        
        return agent
        
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parsing error in {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration from {config_path}: {e}")


def load_tool_from_config(tool_config: Dict[str, Any], base_path: Path) -> Tool:
    """
    Load a tool from configuration.
    
    Args:
        tool_config: Tool configuration dictionary
        base_path: Base path for resolving relative paths
    
    Returns:
        Configured Tool instance
    """
    name = tool_config.get('name')
    if not name:
        raise ConfigError("Tool configuration missing 'name' field")
    
    yaml_path = tool_config.get('yaml_path')
    binding = tool_config.get('binding')
    
    if not yaml_path:
        raise ConfigError(f"Tool '{name}' missing 'yaml_path' field")
    
    if not binding:
        raise ConfigError(f"Tool '{name}' missing 'binding' field")
    
    # Resolve YAML path
    if not Path(yaml_path).is_absolute():
        yaml_path = base_path / yaml_path
    
    # Import and get the binding function
    binding_func = import_from_string(binding)
    
    # Create tool
    tool = Tool.from_yaml(str(yaml_path), binding_func)
    
    return tool


def load_sub_agent_from_config(
    sub_config: Dict[str, Any], 
    base_path: Path
) -> tuple[str, callable]:
    """
    Load a sub-agent factory from configuration.
    
    Args:
        sub_config: Sub-agent configuration dictionary
        base_path: Base path for resolving relative paths
    
    Returns:
        Tuple of (agent_name, agent_factory)
    """
    name = sub_config.get('name')
    if not name:
        raise ConfigError("Sub-agent configuration missing 'name' field")
    
    config_path = sub_config.get('config_path')
    if not config_path:
        raise ConfigError(f"Sub-agent '{name}' missing 'config_path' field")
    
    # Resolve config path
    if not Path(config_path).is_absolute():
        config_path = base_path / config_path
    
    # Create factory function that loads agent when called
    def agent_factory():
        return load_agent_config(str(config_path))
    
    return (name, agent_factory)


def import_from_string(import_string: str) -> Any:
    """
    Import a function or class from a string specification.
    
    Args:
        import_string: String in format "module.path:function_name"
    
    Returns:
        Imported function or class
    """
    try:
        if ':' not in import_string:
            raise ValueError("Import string must contain ':' separator")
        
        module_path, attr_name = import_string.rsplit(':', 1)
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the attribute
        if not hasattr(module, attr_name):
            raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'")
        
        return getattr(module, attr_name)
        
    except ImportError as e:
        raise ConfigError(f"Could not import module from '{import_string}': {e}")
    except AttributeError as e:
        raise ConfigError(f"Could not import attribute from '{import_string}': {e}")
    except Exception as e:
        raise ConfigError(f"Error importing from '{import_string}': {e}")


def validate_config_schema(config: Dict[str, Any]) -> bool:
    """
    Validate agent configuration schema.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if valid, raises ConfigError if invalid
    """
    required_fields = []  # No strictly required fields for flexibility
    
    # Check for required fields
    for field in required_fields:
        if field not in config:
            raise ConfigError(f"Required field '{field}' missing from configuration")
    
    # Validate tool configurations
    tools = config.get('tools', [])
    if not isinstance(tools, list):
        raise ConfigError("'tools' field must be a list")
    
    for i, tool_config in enumerate(tools):
        if not isinstance(tool_config, dict):
            raise ConfigError(f"Tool configuration {i} must be a dictionary")
        
        if 'name' not in tool_config:
            raise ConfigError(f"Tool configuration {i} missing 'name' field")
    
    # Validate sub-agent configurations
    sub_agents = config.get('sub_agents', [])
    if not isinstance(sub_agents, list):
        raise ConfigError("'sub_agents' field must be a list")
    
    for i, sub_config in enumerate(sub_agents):
        if not isinstance(sub_config, dict):
            raise ConfigError(f"Sub-agent configuration {i} must be a dictionary")
        
        if 'name' not in sub_config:
            raise ConfigError(f"Sub-agent configuration {i} missing 'name' field")
    
    return True


def create_default_config(output_path: str, agent_name: str = "default_agent") -> None:
    """
    Create a default agent configuration file.
    
    Args:
        output_path: Path where to save the configuration
        agent_name: Name for the agent
    """
    default_config = {
        'name': agent_name,
        'max_context_tokens': 128000,
        'max_running_subagents': 5,
        'system_prompt': f'You are an AI agent named {agent_name}. Help users accomplish their tasks efficiently.',
        'system_prompt_type': 'string',
        'llm_config': {
            'model': 'gpt-4',
            'base_url': 'https://api.openai.com/v1',
            'temperature': 0.7,
            'max_tokens': 4096
        },
        # Optional: Custom token counter
        # 'token_counter': {
        #     'import': 'my_module:custom_token_counter',
        #     'params': {
        #         'model_name': 'gpt-4'
        #     }
        # },
        # Optional: After model hooks for processing LLM responses
        # 'after_model_hooks': [
        #     # Simple import string
        #     'northau.archs.main_sub.execution.hooks:create_logging_hook',
        #     # Hook with parameters
        #     {
        #         'import': 'northau.archs.main_sub.execution.hooks:create_filter_hook',
        #         'params': {
        #             'allowed_tools': ['bash', 'file_reader'],
        #             'allowed_agents': ['helper_agent']
        #         }
        #     },
        #     # Rate limiting hook
        #     {
        #         'import': 'northau.archs.main_sub.execution.hooks:create_rate_limit_hook',
        #         'params': {
        #             'max_tool_calls': 3,
        #             'max_sub_agent_calls': 2
        #         }
        #     }
        # ],
        'tools': [
            {
                'name': 'bash',
                'yaml_path': 'tools/Bash.yaml',
                'binding': 'northau.archs.tool.builtin.bash_tool:bash'
            }
        ],
        'sub_agents': []
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)