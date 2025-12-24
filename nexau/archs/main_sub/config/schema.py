import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import deprecated

from nexau.archs.main_sub.utils import load_yaml_with_vars

from .base import AgentConfigBase, HookDefinition

YamlValue = dict[str, Any] | list[Any] | str | int | float | bool | None
HookConfig = str | dict[str, Any] | Callable[..., Any]


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


class ToolConfigEntry(BaseModel):
    """Schema for tool entries in agent configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str
    yaml_path: str
    binding: str | None = None
    lazy: bool = False
    as_skill: bool = False
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)


class SubAgentConfigEntry(BaseModel):
    """Schema for sub-agent configuration references."""

    model_config = ConfigDict(extra="forbid")
    name: str
    config_path: str


class MCPServerBaseModel(BaseModel):
    """Shared attributes for MCP server definitions."""

    model_config = ConfigDict(extra="forbid")

    name: str
    timeout: int | None = Field(default=None, gt=0)
    env: dict[str, str] | None = None
    use_cache: bool = False
    disable_parallel: bool = False


class MCPStdIOServer(MCPServerBaseModel):
    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] | None = None


class MCPHttpServer(MCPServerBaseModel):
    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] | None = None


class MCPSseServer(MCPServerBaseModel):
    type: Literal["sse"] = "sse"
    url: str
    headers: dict[str, str] | None = None


MCPServerConfig = MCPStdIOServer | MCPHttpServer | MCPSseServer


def _empty_mcp_server_list() -> list[MCPServerConfig]:
    return []


def _empty_hook_list() -> list[HookDefinition]:
    return []


class AgentConfigSchema(
    AgentConfigBase[ToolConfigEntry, str, list[SubAgentConfigEntry], HookDefinition],
):
    """Top-level schema for agent YAML files."""

    llm_config: dict[str, Any] | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=_empty_mcp_server_list)
    global_storage: dict[str, Any] = Field(default_factory=dict)
    after_model_hooks: list[HookDefinition] | None = None
    after_tool_hooks: list[HookDefinition] | None = None
    before_model_hooks: list[HookDefinition] | None = None
    before_tool_hooks: list[HookDefinition] | None = None
    middlewares: list[HookDefinition] | None = None
    token_counter: HookDefinition | None = None
    tracers: list[HookDefinition] = Field(default_factory=_empty_hook_list)

    @model_validator(mode="after")
    def _require_llm_config(self) -> "AgentConfigSchema":  # type: ignore[override]
        if self.llm_config is None:
            raise ValueError("llm_config is required in agent configuration")
        return self

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        overrides: dict[str, Any] | None = None,
    ) -> "AgentConfigSchema":
        """Load and validate agent configuration from a YAML file."""
        try:
            path = Path(config_path)
            if not path.exists():
                raise ConfigError(f"Configuration file not found: {config_path}")

            config = load_yaml_with_vars(path)
            if not isinstance(config, dict) or not config:
                raise ConfigError(
                    f"Empty or invalid configuration file: {config_path}",
                )

            config = normalize_agent_config_dict(config)

            if overrides:
                warnings.warn(
                    "Overrides will be removed in the v0.4.0, instead use "
                    "agent_config = AgentConfig.from_yaml(...) then agent_config.key = "
                    "value for overrides.",
                )
                config = apply_agent_name_overrides(config, overrides)  # type: ignore[reportDeprecated]

            try:
                return cls.model_validate(config)
            except ValidationError as exc:
                raise ConfigError(
                    f"Invalid agent configuration: {_format_validation_error(exc)}",
                ) from exc

        except yaml.YAMLError as e:
            raise ConfigError(f"YAML parsing error in {config_path}: {e}")
        except Exception as e:
            traceback.print_exc()
            raise ConfigError(
                f"Error loading configuration from {config_path}: {e}",
            )


@deprecated(
    "Overrides will be removed in the v0.4.0, instead use agent_config = "
    "AgentConfig.from_yaml(...) then agent_config.key = value for overrides.",
)
def apply_agent_name_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any],
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
    main_agent_name = config.get("name")

    # Apply overrides to main agent if name matches
    if main_agent_name and main_agent_name in overrides:
        agent_overrides = overrides[main_agent_name]
        if isinstance(agent_overrides, dict):
            agent_overrides_dict = cast(dict[str, Any], agent_overrides)
            for key, value in agent_overrides_dict.items():
                key_str = str(key)
                if isinstance(value, dict) and isinstance(config.get(key_str), dict):
                    existing_section = cast(dict[str, Any], config.get(key_str, {}))
                    value_dict = cast(dict[str, Any], value)
                    existing_section.update(value_dict)
                    config[key_str] = existing_section
                else:
                    config[key_str] = value

    # TODO(hanzhenhua): can sub_agents config be override?

    return config


def normalize_agent_config_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a raw agent config dictionary."""

    try:
        config_model = AgentConfigSchema.model_validate(config)
    except ValidationError as exc:
        raise ConfigError(
            f"Invalid agent configuration: {_format_validation_error(exc)}",
        ) from exc

    return config_model.model_dump(
        mode="python",
        by_alias=True,
        exclude_none=True,
    )


def _format_validation_error(exc: ValidationError) -> str:
    """Return a compact, readable validation error summary."""

    formatted_errors: list[str] = []
    for error in exc.errors():
        location = "->".join(str(segment) for segment in error.get("loc", [])) or "root"
        formatted_errors.append(f"{location}: {error.get('msg')}")
    return "; ".join(formatted_errors)
