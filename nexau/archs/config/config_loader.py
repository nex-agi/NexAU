import traceback
import warnings
from pathlib import Path
from typing import Any

import yaml
from typing_extensions import deprecated

from nexau.archs.main_sub import Agent
from nexau.archs.main_sub.agent_state import GlobalStorage


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


@deprecated(
    (
        "This function is deprecated and will be replaced by function Agent.from_yaml() "
        "in v0.4.0. Please use Agent.from_yaml() instead. Example: "
        "agent = Agent.from_yaml('config.yaml')"
    ),
)
def load_agent_config(
    config_path: str,
    overrides: dict[str, Any] | None = None,
    template_context: dict[str, Any] | None = None,
    global_storage: GlobalStorage | None = None,
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
    if overrides:
        warnings.warn(
            "The overrides parameter is deprecated and will be removed in a future "
            "version. Please use AgentConfig.from_yaml() to load the configuration, "
            "modify attributes directly (e.g., agent_config.key = value), and then "
            "initialize the Agent using Agent(agent_config).",
        )
    try:
        path = Path(config_path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        agent = Agent.from_yaml(config_path=path, overrides=overrides, global_storage=global_storage)

        return agent

    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parsing error in {config_path}: {e}")
    except Exception as e:
        traceback.print_exc()
        raise ConfigError(
            f"Error loading configuration from {config_path}: {e}",
        )
