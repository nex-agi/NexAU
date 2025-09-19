"""Configuration loading system for agents and tools."""
import importlib
import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import dotenv
import yaml

from ..llm import LLMConfig
from ..main_sub import Agent
from ..main_sub import create_agent
from ..main_sub.agent_context import GlobalStorage
from ..tool import Tool

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class AgentBuilder:
    """Builder class for constructing agents from configuration data."""

    def __init__(self, config: dict[str, Any], base_path: Path):
        """Initialize the builder with configuration and base path.

        Args:
            config: The agent configuration dictionary
            base_path: Base path for resolving relative paths
        """
        self.config: dict[str, Any] = config
        self.base_path: Path = base_path
        self.agent_params: dict[str, Any] = {}
        self.overrides: Optional[dict[str, Any]] = None

    def _import_and_instantiate(
        self, hook_config: Union[str, dict[str, Any]],
    ) -> Callable:
        """Import and instantiate a hook from configuration.

        Args:
            hook_config: Hook configuration (string or dict)

        Returns:
            The instantiated hook callable
        """
        if isinstance(hook_config, str):
            # Simple import string format
            return import_from_string(hook_config)
        elif isinstance(hook_config, dict):
            # Dictionary format with import and optional parameters
            import_string = hook_config.get('import')
            if not import_string:
                raise ConfigError("Hook configuration missing 'import' field")

            hook_func = import_from_string(import_string)

            # Check if there are parameters to pass
            params = hook_config.get('params', {})
            if params:
                # For hooks that are factories (like create_* functions), call them with params
                if callable(hook_func):
                    hook_instance = hook_func(**params)
                    return hook_instance
                else:
                    raise ConfigError(
                        'Hook is not callable and cannot accept parameters',
                    )
            else:
                return hook_func
        elif callable(hook_config):
            # Direct callable function (e.g., from overrides)
            return hook_config
        else:
            raise ConfigError('Hook must be a string, dictionary, or callable')

    def build_core_properties(self) -> 'AgentBuilder':
        """Build core agent properties from configuration.

        Returns:
            Self for method chaining
        """
        self.agent_params['name'] = self.config.get('name', 'configured_agent')
        self.agent_params['max_context_tokens'] = self.config.get(
            'max_context_tokens',
            128000,
        )
        self.agent_params['max_running_subagents'] = self.config.get(
            'max_running_subagents',
            5,
        )
        self.agent_params['system_prompt'] = self.config.get('system_prompt')
        self.agent_params['system_prompt_type'] = self.config.get(
            'system_prompt_type',
            'string',
        )
        self.agent_params['initial_context'] = self.config.get('context', {})

        self.agent_params['stop_tools'] = self.config.get('stop_tools', [])
        self.agent_params['max_iterations'] = self.config.get('max_iterations', 100)

        return self

    def build_mcp_servers(self) -> 'AgentBuilder':
        """Build MCP servers configuration from configuration.

        Returns:
            Self for method chaining
        """
        mcp_servers = self.config.get('mcp_servers', [])

        if not isinstance(mcp_servers, list):
            raise ConfigError("'mcp_servers' must be a list")

        # Validate each MCP server configuration
        for i, server_config in enumerate(mcp_servers):
            if not isinstance(server_config, dict):
                raise ConfigError(
                    f"MCP server configuration {i} must be a dictionary",
                )

            # Validate required fields
            if 'name' not in server_config:
                raise ConfigError(
                    f"MCP server configuration {i} missing 'name' field",
                )

            if 'type' not in server_config:
                raise ConfigError(
                    f"MCP server configuration {i} missing 'type' field",
                )

            server_type = server_config['type']
            if server_type not in ['stdio', 'http', 'sse']:
                raise ConfigError(
                    f"MCP server configuration {i} has invalid type '{server_type}'. "
                    'Must be one of: stdio, http, sse',
                )

            # Validate type-specific requirements
            if server_type == 'stdio':
                if 'command' not in server_config:
                    raise ConfigError(
                        f"MCP server configuration {i} of type 'stdio' missing 'command' field",
                    )
            elif server_type in ['http', 'sse']:
                if 'url' not in server_config:
                    raise ConfigError(
                        f"MCP server configuration {i} of type '{server_type}' missing 'url' field",
                    )

        self.agent_params['mcp_servers'] = mcp_servers
        return self

    def build_hooks(self) -> 'AgentBuilder':
        """Build hooks from configuration.

        Returns:
            Self for method chaining
        """
        # Handle after_model_hooks configuration
        after_model_hooks = None
        if 'after_model_hooks' in self.config:
            hooks_config = self.config['after_model_hooks']
            after_model_hooks = []

            if not isinstance(hooks_config, list):
                raise ConfigError("'after_model_hooks' must be a list")

            for i, hook_config in enumerate(hooks_config):
                try:
                    hook_func = self._import_and_instantiate(hook_config)
                    after_model_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading hook {i}: {e}")

        self.agent_params['after_model_hooks'] = after_model_hooks

        # Handle after_tool_hooks configuration
        after_tool_hooks = None
        if 'after_tool_hooks' in self.config:
            hooks_config = self.config['after_tool_hooks']
            after_tool_hooks = []

            if not isinstance(hooks_config, list):
                raise ConfigError("'after_tool_hooks' must be a list")

            for i, hook_config in enumerate(hooks_config):
                try:
                    hook_func = self._import_and_instantiate(hook_config)
                    after_tool_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading tool hook {i}: {e}")

        self.agent_params['after_tool_hooks'] = after_tool_hooks
        
        # Handle before_model_hooks configuration
        before_model_hooks = None
        if 'before_model_hooks' in self.config:
            hooks_config = self.config['before_model_hooks']
            before_model_hooks = []
            
            if not isinstance(hooks_config, list):
                raise ConfigError("'before_model_hooks' must be a list")
            
            for i, hook_config in enumerate(hooks_config):
                try:
                    hook_func = self._import_and_instantiate(hook_config)
                    before_model_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading before model hook {i}: {e}")
                    
        self.agent_params['before_model_hooks'] = before_model_hooks

        return self

    def build_tools(self) -> 'AgentBuilder':
        """Build tools from configuration.

        Returns:
            Self for method chaining
        """
        tools = []
        tool_configs = self.config.get('tools', [])
        for tool_config in tool_configs:
            try:
                tool = load_tool_from_config(tool_config, self.base_path)
                tools.append(tool)
            except Exception as e:
                raise ConfigError(
                    f"Error loading tool '{tool_config.get('name', 'unknown')}': {e}",
                )

        self.agent_params['tools'] = tools
        return self

    def build_sub_agents(self) -> 'AgentBuilder':
        """Build sub-agents from configuration.

        Returns:
            Self for method chaining
        """
        sub_agents = []
        sub_agent_configs = self.config.get('sub_agents', [])
        for sub_config in sub_agent_configs:
            try:
                sub_agent = load_sub_agent_from_config(
                    sub_config,
                    self.base_path,
                    self.overrides,
                )
                sub_agents.append(sub_agent)
            except Exception as e:
                raise ConfigError(
                    f"Error loading sub-agent '{sub_config.get('name', 'unknown')}': {e}",
                )

        self.agent_params['sub_agents'] = sub_agents
        return self

    def build_llm_config(self) -> 'AgentBuilder':
        """Build LLM configuration and related components.

        Returns:
            Self for method chaining
        """
        # Handle LLM configuration
        if 'llm_config' not in self.config:
            raise ConfigError(
                "'llm_config' is required in agent configuration",
            )

        self.agent_params['llm_config'] = LLMConfig(
            **self.config['llm_config'],
        )

        # Handle custom_llm_generator configuration
        custom_llm_generator = None
        if 'custom_llm_generator' in self.config:
            llm_generator_config = self.config['custom_llm_generator']
            if isinstance(llm_generator_config, str):
                # Simple import string format
                custom_llm_generator = import_from_string(llm_generator_config)
            elif isinstance(llm_generator_config, dict):
                # Dictionary format with import and optional parameters
                import_string = llm_generator_config.get('import')
                if not import_string:
                    raise ConfigError(
                        "custom_llm_generator missing 'import' field",
                    )

                llm_generator_func = import_from_string(import_string)

                # Check if there are parameters to pass
                params = llm_generator_config.get('params', {})
                if params:
                    # Create a wrapper function with the parameters
                    def configured_llm_generator(openai_client, kwargs):
                        return llm_generator_func(openai_client, kwargs, **params)

                    custom_llm_generator = configured_llm_generator
                else:
                    custom_llm_generator = llm_generator_func
            else:
                raise ConfigError(
                    'custom_llm_generator configuration must be a string or dictionary',
                )

        self.agent_params['custom_llm_generator'] = custom_llm_generator

        # Handle token counter configuration
        token_counter = None
        if 'token_counter' in self.config:
            token_counter_config = self.config['token_counter']
            if isinstance(token_counter_config, str):
                # Import string format: "module.path:function_name"
                token_counter = import_from_string(token_counter_config)
            elif isinstance(token_counter_config, dict):
                # Dictionary format with import and optional parameters
                import_string = token_counter_config.get('import')
                if not import_string:
                    raise ConfigError(
                        "Token counter configuration missing 'import' field",
                    )

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
                raise ConfigError(
                    'Token counter configuration must be a string or dictionary',
                )

        self.agent_params['token_counter'] = token_counter

        return self

    def build_system_prompt_path(self) -> 'AgentBuilder':
        """Build system prompt path resolution.

        Returns:
            Self for method chaining
        """
        system_prompt = self.agent_params.get('system_prompt')
        system_prompt_type = self.agent_params.get(
            'system_prompt_type',
            'string',
        )

        # Convert system_prompt from relative path to absolute path
        if (
            system_prompt
            and system_prompt_type in ['file', 'jinja']
            and not Path(system_prompt).is_absolute()
        ):
            system_prompt = self.base_path / system_prompt
            if not Path(system_prompt).exists():
                raise ConfigError(
                    f"System prompt file not found: {system_prompt}",
                )
            self.agent_params['system_prompt'] = str(system_prompt)

        return self

    def set_overrides(self, overrides: Optional[dict[str, Any]]) -> 'AgentBuilder':
        """Set overrides for sub-agent loading.

        Args:
            overrides: Configuration overrides

        Returns:
            Self for method chaining
        """
        self.overrides = overrides
        return self

    def get_agent(self, global_storage: Optional[GlobalStorage] = None) -> Agent:
        """Create the final agent instance.

        Args:
            global_storage: Optional global storage instance

        Returns:
            Configured Agent instance
        """
        if global_storage is None:
            global_storage = GlobalStorage()
        logger.info(
            f"updating global_storage from config: {self.config.get('global_storage', {})}",
        )
        global_storage.update(self.config.get('global_storage', {}))
        return create_agent(
            global_storage=global_storage,
            **self.agent_params,
        )


def apply_agent_name_overrides(
    config: dict[str, Any], overrides: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply overrides based on agent names.

    Args:
        config: The agent configuration dictionary
        overrides: Dictionary where keys are agent names and values are override configs

    Returns:
        Updated configuration with overrides applied
    """
    # Make a copy to avoid modifying the original
    config = config.copy()

    # Get the main agent name
    main_agent_name = config.get('name')

    # Apply overrides to main agent if name matches
    if main_agent_name and main_agent_name in overrides:
        agent_overrides = overrides[main_agent_name]
        if isinstance(agent_overrides, dict):
            for key, value in agent_overrides.items():
                if isinstance(value, dict):
                    config[key].update(value)
                else:
                    config[key] = value

    # TODO(hanzhenhua): can sub_agents config be override?

    return config


def load_agent_config(
    config_path: str,
    overrides: Optional[dict[str, Any]] = None,
    template_context: Optional[dict[str, Any]] = None,
    global_storage: Optional[GlobalStorage] = None,
) -> Agent:
    """
    Load agent configuration from YAML file.

    Args:
        config_path: Path to the agent configuration YAML file
        overrides: Dictionary of configuration overrides
        template_context: Context variables for Jinja template rendering
        global_storage: Optional global storage instance

    Returns:
        Configured Agent instance
    """
    try:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        # Load YAML configuration
        with open(path, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not config:
            raise ConfigError(
                f"Empty or invalid configuration file: {config_path}",
            )

        # Apply overrides based on agent name
        if overrides:
            config = apply_agent_name_overrides(config, overrides)

        # Create builder and construct agent
        builder = AgentBuilder(config, path.parent)

        agent = (
            builder.set_overrides(overrides)
            .build_core_properties()
            .build_llm_config()
            .build_mcp_servers()
            .build_hooks()
            .build_tools()
            .build_sub_agents()
            .build_system_prompt_path()
            .get_agent(global_storage)
        )

        # Apply template context if provided and using Jinja templates
        if config.get('system_prompt_type') == 'jinja' and template_context:
            # Template context processing would be handled by the prompt builder
            # during agent execution, so we just pass it through
            pass

        return agent

    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parsing error in {config_path}: {e}")
    except Exception as e:
        raise ConfigError(
            f"Error loading configuration from {config_path}: {e}",
        )


def load_tool_from_config(tool_config: dict[str, Any], base_path: Path) -> Tool:
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
    binding = tool_config.get('binding', None)

    if not yaml_path:
        raise ConfigError(f"Tool '{name}' missing 'yaml_path' field")

    # Resolve YAML path
    if not Path(yaml_path).is_absolute():
        yaml_path = base_path / yaml_path

    # Import and get the binding function
    if binding:
        binding_func = import_from_string(binding)
    else:

        def binding_func(x):
            return x

    # Create tool
    tool = Tool.from_yaml(str(yaml_path), binding_func)

    return tool


def load_sub_agent_from_config(
    sub_config: dict[str, Any],
    base_path: Path,
    overrides: Optional[dict[str, Any]] = None,
) -> tuple[str, Callable[[], Agent]]:
    """
    Load a sub-agent factory from configuration.

    Args:
        sub_config: Sub-agent configuration dictionary
        base_path: Base path for resolving relative paths
        overrides: Dictionary of configuration overrides to pass through

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
    def agent_factory(global_storage: Optional[GlobalStorage] = None):
        return load_agent_config(
            str(config_path), overrides=overrides, global_storage=global_storage,
        )

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
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{attr_name}'",
            )

        return getattr(module, attr_name)

    except ImportError as e:
        raise ConfigError(
            f"Could not import module from '{import_string}': {e}",
        )
    except AttributeError as e:
        raise ConfigError(
            f"Could not import attribute from '{import_string}': {e}",
        )
    except Exception as e:
        raise ConfigError(f"Error importing from '{import_string}': {e}")


def validate_config_schema(config: dict[str, Any]) -> bool:
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
            raise ConfigError(
                f"Required field '{field}' missing from configuration",
            )

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
            raise ConfigError(
                f"Sub-agent configuration {i} must be a dictionary",
            )

        if 'name' not in sub_config:
            raise ConfigError(
                f"Sub-agent configuration {i} missing 'name' field",
            )

    # Validate MCP servers configurations
    mcp_servers = config.get('mcp_servers', [])
    if not isinstance(mcp_servers, list):
        raise ConfigError("'mcp_servers' field must be a list")

    for i, server_config in enumerate(mcp_servers):
        if not isinstance(server_config, dict):
            raise ConfigError(
                f"MCP server configuration {i} must be a dictionary",
            )

        if 'name' not in server_config:
            raise ConfigError(
                f"MCP server configuration {i} missing 'name' field",
            )

        if 'type' not in server_config:
            raise ConfigError(
                f"MCP server configuration {i} missing 'type' field",
            )

    return True
