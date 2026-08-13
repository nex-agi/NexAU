import traceback
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlsplit

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
    defer_loading: bool = False
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None


class SkillConfigEntry(BaseModel):
    """Schema for skill folder entries in agent configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    path: str
    source_id: str | None = None


class SubAgentConfigEntry(BaseModel):
    """Schema for sub-agent configuration references."""

    model_config = ConfigDict(extra="forbid")
    name: str
    config_path: str
    source_id: str | None = None


class MCPServerBaseModel(BaseModel):
    """Shared attributes for MCP server definitions.

    RFC-0029: MCP 协议与认证配置只描述宿主策略，wire 行为由官方 SDK 负责。
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str
    source_id: str | None = None
    timeout: int | None = Field(default=None, gt=0)
    env: dict[str, str] | None = Field(default=None, repr=False)
    disable_parallel: bool = False
    # RFC-0019: server 级默认权限（None = auto-allow，向后兼容）
    permissions: dict[str, list[str]] | None = None
    # RFC-0019: per-tool 权限覆盖（key=原始工具名，None 值 = auto-allow）
    tool_permissions: dict[str, dict[str, list[str]] | None] | None = None


class MCPStdIOServer(MCPServerBaseModel):
    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] | None = Field(default=None, repr=False)


class MCPEnvSecretRef(BaseModel):
    """Reference a secret without embedding its value in agent configuration.

    RFC-0029: 首版只支持从环境变量解析 secret，后续宿主可注入其他 resolver。
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    source: Literal["env"] = "env"
    key: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class MCPBearerAuth(BaseModel):
    """Static bearer authentication backed by a secret reference."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: Literal["bearer"] = "bearer"
    token: MCPEnvSecretRef


class MCPAuthorizationCodeAuth(BaseModel):
    """Interactive OAuth authorization-code authentication configuration."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: Literal["authorization_code"] = "authorization_code"
    client_name: str = "NexAU"
    scopes: list[str] = Field(default_factory=list)
    client_id: str | None = None
    client_secret: MCPEnvSecretRef | None = None

    @model_validator(mode="after")
    def _validate_registered_client(self) -> "MCPAuthorizationCodeAuth":
        if self.client_secret is not None and not self.client_id:
            raise ValueError("authorization_code client_secret requires client_id")
        return self


class MCPClientCredentialsAuth(BaseModel):
    """Non-interactive OAuth client-credentials authentication configuration."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    type: Literal["client_credentials"] = "client_credentials"
    client_id: str
    client_secret: MCPEnvSecretRef
    scopes: list[str] = Field(default_factory=list)


MCPAuthConfig = Annotated[
    MCPBearerAuth | MCPAuthorizationCodeAuth | MCPClientCredentialsAuth,
    Field(discriminator="type"),
]


class MCPRemoteServer(MCPServerBaseModel):
    """Shared HTTP/SSE configuration and security validation."""

    url: str = Field(repr=False)
    headers: dict[str, str] | None = Field(default=None, repr=False)
    auth: MCPAuthConfig | None = Field(default=None, discriminator="type", repr=False)

    @model_validator(mode="after")
    def _validate_remote_security(self) -> "MCPRemoteServer":
        # RFC-0029: 避免认证凭据经明文网络发送；本地测试允许 loopback HTTP。
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("url must be an absolute HTTP(S) URL")
        loopback_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and parsed.hostname.lower() not in loopback_hosts:
            raise ValueError("remote MCP URLs must use HTTPS; HTTP is allowed only for loopback hosts")

        if self.auth is not None and self.headers is not None:
            if any(name.lower() == "authorization" for name in self.headers):
                raise ValueError("auth cannot be combined with an Authorization header")
        return self


class MCPHttpServer(MCPRemoteServer):
    type: Literal["http"] = "http"


class MCPSseServer(MCPRemoteServer):
    type: Literal["sse"] = "sse"


MCPServerConfig = Annotated[
    MCPStdIOServer | MCPHttpServer | MCPSseServer,
    Field(discriminator="type"),
]


class PluginEntryConfig(BaseModel):
    """Schema for top-level plugin enablement entries."""

    model_config = ConfigDict(extra="forbid")

    use: str
    config: dict[str, Any] = Field(default_factory=dict)


def _empty_mcp_server_list() -> list[MCPServerConfig]:
    return []


def _empty_hook_list() -> list[HookDefinition]:
    return []


def _empty_plugin_entry_list() -> list[PluginEntryConfig]:
    return []


class AgentConfigSchema(
    AgentConfigBase[ToolConfigEntry, str | SkillConfigEntry, list[SubAgentConfigEntry], HookDefinition],
):
    """Top-level schema for agent YAML files."""

    llm_config: dict[str, Any] | None = None
    sandbox_config: dict[str, Any] | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=_empty_mcp_server_list, repr=False)
    global_storage: dict[str, Any] = Field(default_factory=dict)
    after_model_hooks: list[HookDefinition] | None = None
    after_tool_hooks: list[HookDefinition] | None = None
    before_model_hooks: list[HookDefinition] | None = None
    before_tool_hooks: list[HookDefinition] | None = None
    middlewares: list[HookDefinition] | None = None
    token_counter: HookDefinition | None = None
    tracers: list[HookDefinition] = Field(default_factory=_empty_hook_list)
    plugins: list[PluginEntryConfig] = Field(default_factory=_empty_plugin_entry_list)

    @model_validator(mode="after")
    def _require_llm_config(self) -> "AgentConfigSchema":  # type: ignore[override]
        if self.llm_config is None:
            raise ValueError("llm_config is required in agent configuration")
        names = [server.name for server in self.mcp_servers]
        if len(names) != len(set(names)):
            raise ValueError("mcp_servers names must be unique within an Agent")
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
                    "Overrides will be removed in the v0.5.0, instead use "
                    "agent_config = AgentConfig.from_yaml(...) then agent_config.key = "
                    "value for overrides.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                config = apply_agent_name_overrides_to_dict(config, overrides)

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
    "Overrides will be removed in the v0.5.0, instead use agent_config = "
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
    return apply_agent_name_overrides_to_dict(config, overrides)


def apply_agent_name_overrides_to_dict(
    config: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply overrides based on agent names without emitting deprecation diagnostics."""
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
