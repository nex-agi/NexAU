# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration models for the NexAU agent framework."""

from __future__ import annotations

import inspect
import logging
import os
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import dotenv
from pydantic import ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.skill import Skill, build_load_skill_tool, build_tool_skill
from nexau.archs.main_sub.tool_call_modes import normalize_tool_call_mode
from nexau.archs.main_sub.utils import import_from_string
from nexau.archs.sandbox.base_sandbox import (
    E2BSandboxConfig,
    LocalSandboxConfig,
    SandboxConfig,
    parse_sandbox_config,
)
from nexau.archs.tool import Tool
from nexau.archs.tracer.composite import CompositeTracer
from nexau.archs.tracer.core import BaseTracer

from .base import AgentConfigBase, AgentConfigLoadOptions, HookCallable, HookDefinition
from .schema import AgentConfigSchema

if TYPE_CHECKING:
    from nexau.core.messages import Message

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

TTool = TypeVar("TTool", bound=object)
TSkill = TypeVar("TSkill", bound=object)
TSubAgent = TypeVar("TSubAgent", bound=object)
THook = TypeVar("THook", bound=object)

YamlValue = dict[str, Any] | list[Any] | str | int | float | bool | None
HookConfig = str | dict[str, Any] | Callable[..., Any]

_BUILTIN_TOOL_SCHEMA_ROOT = "nexau:archs/tool/builtin/schemas"

# RFC-0028: 聚合 Web 搜索。与上面那些"零配置即可用"的内置工具不同，
# 它必须有搜索服务商密钥才能工作，因此**仅在配置了密钥时才注入**——
# 否则每个 Agent 的工具列表里都会多一个一调就报错的工具，白占上下文还诱导误用。
_CONDITIONAL_BUILTIN_TOOL_BINDINGS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "web_search",
        "nexau.archs.tool.builtin.web_tools:web_search",
        ("SEARCH_API_KEY", "SERPER_API_KEY"),
    ),
)


def _conditional_builtin_bindings() -> tuple[tuple[str, str], ...]:
    """筛出当前环境已具备前置条件的条件式内置工具。

    RFC-0028: 只要任一候选环境变量非空即视为已配置。
    """
    enabled: list[tuple[str, str]] = []
    for name, binding, env_keys in _CONDITIONAL_BUILTIN_TOOL_BINDINGS:
        if any((os.getenv(key) or "").strip() for key in env_keys):
            enabled.append((name, binding))
    return tuple(enabled)


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


def _empty_dict_list() -> list[dict[str, Any]]:
    return []


def _empty_tracer_list() -> list[BaseTracer]:
    return []


def _empty_tool_list() -> list[Tool]:
    return []


def _empty_skill_list() -> list[Skill]:
    return []


def _resolve_config_path(raw_path: str, base_path: Path) -> Path:
    """Resolve a config path that may be a pkg:resource, relative, or absolute path.

    Supports:
      - "some_package:relative/path.yaml"  -> importlib.resources
      - "/absolute/path.yaml"              -> as-is
      - "relative/path.yaml"               -> resolved against base_path

    On Windows absolute paths like "C:\\path" also contain ":", so we
    check ``is_absolute()`` first to avoid misinterpreting them as
    package resources.

    Note: for package resources inside zip archives, the returned Path may not
    exist on the real filesystem.  Use :func:`_resolve_config_resource` instead
    when the path must be immediately opened for reading.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if ":" in raw_path:
        pkg, resource_path = raw_path.split(":", 1)
        from importlib.resources import files

        return Path(str(files(pkg).joinpath(resource_path)))
    return base_path / raw_path


def _resolve_config_resource(raw_path: str, base_path: Path) -> AbstractContextManager[Path]:
    """Resolve a config path, ensuring the result is readable on the filesystem.

    Like :func:`_resolve_config_path`, but wraps package resources with
    ``importlib.resources.as_file`` so that resources inside zip archives are
    materialised to a temporary file for the duration of the context.

    Use this instead of ``_resolve_config_path`` when the resolved path must be
    opened for reading within a well-defined scope (e.g. loading a YAML config).
    """
    path = Path(raw_path)
    if path.is_absolute():
        return nullcontext(path)
    if ":" in raw_path:
        pkg, resource_path = raw_path.split(":", 1)
        from importlib.resources import as_file, files

        return as_file(files(pkg).joinpath(resource_path))
    return nullcontext(base_path / raw_path)


def _require_dict(value: object, *, context: str) -> dict[str, Any]:
    """Ensure a value is a dictionary and return it typed."""
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a dictionary")
    return cast(dict[str, Any], value)


def _inject_builtin_tools(config: dict[str, Any]) -> None:
    """Inject conditional runtime built-in tools without replacing declared tools.

    插件贡献与 agent 自声明的同名工具优先，runtime 仅补齐缺失项。
    """
    tools_raw: object = config.get("tools")
    if tools_raw is None:
        tools: list[object] = []
        config["tools"] = tools
    elif isinstance(tools_raw, list):
        tools = cast(list[object], tools_raw)
    else:
        raise ConfigError("'tools' must be a list")

    # 1. 收集插件展开与 agent 自声明的工具名
    existing_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_entry = cast(dict[object, object], tool)
        name = tool_entry.get("name")
        if isinstance(name, str):
            existing_names.add(name)

    # 2. 按声明顺序补齐已满足前置条件的 runtime 内置工具
    for name, binding in _conditional_builtin_bindings():
        if name in existing_names:
            continue
        tools.append(
            {
                "name": name,
                "yaml_path": f"{_BUILTIN_TOOL_SCHEMA_ROOT}/{name}.tool.yaml",
                "binding": binding,
            },
        )
        existing_names.add(name)


class AgentConfig(
    AgentConfigBase[
        Tool,
        Skill,
        dict[str, "AgentConfig"],
        HookCallable,
    ],
):
    """Configuration for an Agent's definition and behavior."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    tools: list[Tool] = Field(default_factory=_empty_tool_list)
    skills: list[Skill] = Field(default_factory=_empty_skill_list)
    sub_agents: dict[str, AgentConfig] | None = None
    llm_config: LLMConfig | None = None
    sandbox_config: SandboxConfig | None = Field(default=None)

    @field_validator("tool_call_mode", mode="before")
    @classmethod
    def _normalize_tool_call_mode(cls, value: object) -> str:
        if value is None:
            return normalize_tool_call_mode(None)
        if not isinstance(value, str):
            raise ValueError("tool_call_mode must be a string")
        return normalize_tool_call_mode(value)

    @field_validator("sandbox_config", mode="before")
    @classmethod
    def _validate_sandbox_config(cls, value: object) -> SandboxConfig | None:
        if value is None:
            return None
        if isinstance(value, (LocalSandboxConfig, E2BSandboxConfig)):
            return value
        if isinstance(value, dict):
            return parse_sandbox_config(cast(dict[str, Any], value))
        raise ValueError(f"Invalid sandbox_config type: {type(value)}")

    mcp_servers: list[dict[str, Any]] = Field(default_factory=_empty_dict_list)
    after_model_hooks: list[Callable[..., Any]] | None = None
    after_tool_hooks: list[Callable[..., Any]] | None = None
    before_model_hooks: list[Callable[..., Any]] | None = None
    before_tool_hooks: list[Callable[..., Any]] | None = None
    middlewares: list[Any] | None = None
    tracers: list[BaseTracer] = Field(default_factory=_empty_tracer_list)
    resolved_tracer: BaseTracer | None = Field(default=None, exclude=True)
    token_counter: HookDefinition | None = None
    _is_finalized: bool = PrivateAttr(default=False)

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        overrides: dict[str, Any] | None = None,
        options: AgentConfigLoadOptions | None = None,
    ) -> AgentConfig:
        """
        Load a sub-agent factory from configuration.

        Args:
            sub_config: Sub-agent configuration dictionary
            base_path: Base path for resolving relative paths
            overrides: Dictionary of configuration overrides to pass through
            options: Configuration loading options (e.g. strict mode)

        Returns:
            Tuple of (agent_name, agent_factory)
        """
        if overrides:
            warnings.warn(
                "Overrides will be removed in the v0.4.0, instead use agent_config = "
                "AgentConfig.from_yaml(...) then agent_config.key = value for "
                "overrides.",
                DeprecationWarning,
                stacklevel=2,
            )

        agent_config_schema = AgentConfigSchema.from_yaml(str(config_path), overrides)
        return cls._from_schema(
            agent_config_schema,
            base_path=config_path.parent,
            overrides=overrides,
            options=options,
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        base_path: Path,
        overrides: dict[str, Any] | None = None,
        options: AgentConfigLoadOptions | None = None,
    ) -> AgentConfig:
        """Load an AgentConfig from an already parsed config dictionary."""
        if overrides:
            warnings.warn(
                "Overrides will be removed in the v0.4.0, instead use agent_config = "
                "AgentConfig.from_yaml(...) then agent_config.key = value for "
                "overrides.",
                DeprecationWarning,
                stacklevel=2,
            )
            from .schema import apply_agent_name_overrides_to_dict

            config = apply_agent_name_overrides_to_dict(config, overrides)

        try:
            agent_config_schema = AgentConfigSchema.model_validate(config)
        except ValidationError as exc:
            raise ConfigError(f"Invalid agent configuration: {exc}") from exc
        return cls._from_schema(
            agent_config_schema,
            base_path=base_path,
            overrides=overrides,
            options=options,
        )

    @classmethod
    def _from_schema(
        cls,
        agent_config_schema: AgentConfigSchema,
        *,
        base_path: Path,
        overrides: dict[str, Any] | None,
        options: AgentConfigLoadOptions | None,
    ) -> AgentConfig:
        effective_options = options or AgentConfigLoadOptions()

        config_dict = agent_config_schema.model_dump(
            mode="python",
            by_alias=True,
            exclude_none=True,
        )
        if effective_options.expand_plugins:
            from nexau.archs.main_sub.plugin import PluginAdapter, PluginConfigError

            try:
                config_dict = PluginAdapter(
                    config_dict,
                    agent_base_path=base_path,
                    strict=effective_options.strict,
                ).expand()
            except PluginConfigError as exc:
                raise ConfigError(str(exc)) from exc
        else:
            ignored_plugins = config_dict.pop("plugins", [])
            if ignored_plugins:
                logger.info("sub_agent_load plugins_ignored=%d", len(ignored_plugins))

        # 在插件与 agent 工具合并后补齐已满足前置条件的 runtime 内置工具
        _inject_builtin_tools(config_dict)

        agent_builder = AgentConfigBuilder(
            config_dict,
            base_path,
            strict=effective_options.strict,
        )
        agent_config = (
            agent_builder.set_overrides(overrides)
            .build_core_properties()
            .build_llm_config()
            .build_mcp_servers()
            .build_hooks()
            .build_tracers()
            .build_tools()
            .build_sub_agents()
            .build_skills()
            .build_system_prompt_path()
            .build_sandbox()
            .get_agent_config()
        )

        return agent_config

    @field_validator("llm_config", mode="before")
    @classmethod
    def _validate_llm_config(
        cls,
        value: object,
    ) -> LLMConfig | dict[str, Any] | None:
        if value is None:
            return value
        if isinstance(value, LLMConfig):
            return value
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
        raise ValueError(
            f"Invalid llm_config type: {type(value)}",
        )

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _ensure_mcp_servers(
        cls,
        value: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if value is None:
            return []
        return value

    @field_validator("tracers")
    @classmethod
    def _ensure_tracers(
        cls,
        value: object,
    ) -> list[BaseTracer]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Tracers must be provided as a list")
        value_list = cast(list[Any], value)
        typed_tracers: list[BaseTracer] = []
        for tracer in value_list:
            if not isinstance(tracer, BaseTracer):
                raise ValueError("All tracers must inherit from BaseTracer")
            typed_tracers.append(tracer)
        return typed_tracers

    @model_validator(mode="after")
    def _finalize(self):  # type: ignore[override]
        """Finalize configuration by normalizing fields and injecting skill tool.

        RFC-0015: 合并 Sub-agent 工具为统一 Agent 工具

        当 sub_agents 非空时注入 Agent 工具，并在描述中拼接可用子代理列表。
        """
        if self._is_finalized:
            return self
        from nexau.archs.tool.builtin.agent_tool import call_sub_agent

        nexau_package_path = Path(__file__).parent.parent.parent.parent
        if self.sub_agents:
            # 1. 生成可用子代理列表描述后缀
            sub_agent_desc_parts: list[str] = ["\n\nAvailable sub-agents:"]
            for _sa_name, _sa_config in self.sub_agents.items():
                _sa_desc = _sa_config.description or f"Delegate work to sub-agent '{_sa_name}'."
                sub_agent_desc_parts.append(f"\n- **{_sa_name}**: {_sa_desc}")
            sub_agent_description_suffix = "".join(sub_agent_desc_parts)

            # 2. 注册 Agent 工具
            agent_tool = Tool.from_yaml(
                str(nexau_package_path / "archs" / "tool" / "builtin" / "schemas" / "Agent.tool.yaml"),
                binding=call_sub_agent,
                description_suffix=sub_agent_description_suffix,
            )
            self.tools.append(agent_tool)

        # Ensure stop_tools is a set for faster lookup
        if isinstance(self.stop_tools, list):
            self.stop_tools = set(self.stop_tools)
        elif self.stop_tools is None:
            self.stop_tools = set()

        # Handle LLM configuration
        if self.llm_config is None:
            self.llm_config = LLMConfig()

        # Ensure name is set
        if not self.name:
            self.name = f"agent_{id(self)}"

        # Ensure load_skill tool is ingested into tools
        load_skill_tool = build_load_skill_tool(self.tools, self.skills)
        if load_skill_tool:
            self.tools.append(load_skill_tool)

        # Resolve tracer composition
        if len(self.tracers) == 1:
            self.resolved_tracer = self.tracers[0]
        elif len(self.tracers) > 1:
            self.resolved_tracer = CompositeTracer(self.tracers)
        else:
            self.resolved_tracer = None

        self._is_finalized = True
        return self


@dataclass
class ExecutionConfig:
    """Configuration for agent execution environment and behavior."""

    max_iterations: int = 100
    max_context_tokens: int = 128000
    max_running_subagents: int = 5
    retry_attempts: int = 5
    retry_backoff_max_seconds: int = 30
    timeout: int = 300
    tool_call_mode: str = "structured"

    def __post_init__(self) -> None:
        """Validate execution configuration."""
        self.tool_call_mode = normalize_tool_call_mode(self.tool_call_mode)

    @classmethod
    def from_agent_config(cls, agent_config: AgentConfig) -> ExecutionConfig:
        """Create execution configuration derived from an agent configuration."""

        return cls(
            max_iterations=agent_config.max_iterations,
            max_context_tokens=agent_config.max_context_tokens,
            max_running_subagents=agent_config.max_running_subagents,
            retry_attempts=agent_config.retry_attempts,
            retry_backoff_max_seconds=agent_config.retry_backoff_max_seconds,
            timeout=agent_config.timeout,
            tool_call_mode=agent_config.tool_call_mode,
        )


class AgentConfigBuilder:
    """Builder class for constructing agents from configuration data."""

    def __init__(self, config: dict[str, Any], base_path: Path, *, strict: bool = True):
        """Initialize the builder with configuration and base path.

        Args:
            config: The agent configuration dictionary
            base_path: Base path for resolving relative paths
            strict: If True, raise ConfigError on component load failures;
                    if False, log a warning and skip the failed component.
        """
        self.config: dict[str, Any] = config
        self.base_path: Path = base_path
        self.strict: bool = strict
        self.agent_params: dict[str, Any] = {}
        self.overrides: dict[str, Any] | None = None
        self._skipped_components: list[str] = list(cast(list[str], config.pop("_skipped_components", [])))

    def _handle_component_error(self, msg: str, error: Exception | None = None) -> None:
        """Handle a component loading error according to the strict mode.

        In strict mode, raise ConfigError immediately.
        In non-strict mode, log a warning and record for the build summary.
        """
        if self.strict:
            raise ConfigError(msg) from error
        logger.warning(msg)
        self._skipped_components.append(msg)

    def _import_and_instantiate(
        self,
        hook_config: HookConfig,
    ) -> Any:
        """Import and instantiate a hook from configuration.

        Args:
            hook_config: Hook configuration (string or dict)

        Returns:
            The instantiated hook callable
        """
        if isinstance(hook_config, str):
            # Simple import string format
            hook_obj = import_from_string(hook_config)
            return self._instantiate_hook_object(hook_obj, hook_config)
        elif isinstance(hook_config, dict):
            # Dictionary format with import and optional parameters
            hook_config_dict: dict[str, Any] = cast(dict[str, Any], hook_config)
            import_string_value: str | None = hook_config_dict.get("import") if isinstance(hook_config_dict.get("import"), str) else None
            if not import_string_value:
                raise ConfigError("Hook configuration missing 'import' field")

            import_string: str = import_string_value

            hook_obj = import_from_string(import_string)
            params_raw = hook_config_dict.get("params")
            if params_raw is None:
                params: dict[str, Any] = {}
            elif isinstance(params_raw, dict):
                params = cast(dict[str, Any], params_raw)
            else:
                raise ConfigError("Hook configuration 'params' must be a mapping when provided")
            # Resolve relative path params against the YAML file's base directory.
            resolved_params: dict[str, Any] = {}
            for k, v in params.items():
                if isinstance(v, str) and (k.endswith("_path") or k.endswith("_file")):
                    p = Path(v)
                    if not p.is_absolute():
                        v = str(self.base_path / p)
                resolved_params[k] = v
            return self._instantiate_hook_object(hook_obj, import_string, resolved_params)
        elif callable(hook_config):
            # Direct callable function (e.g., from overrides)
            return hook_config
        else:
            raise ConfigError("Hook must be a string, dictionary, or callable")

    def _instantiate_hook_object(
        self,
        hook_obj: Any,
        import_string: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Instantiate hook classes or factory functions with optional params."""

        params_dict: dict[str, Any] = params or {}

        if inspect.isclass(hook_obj):
            try:
                return hook_obj(**params_dict)
            except TypeError as exc:  # pragma: no cover - error path
                raise ConfigError(
                    f"Error instantiating hook '{import_string}': {exc}",
                ) from exc

        if params_dict:
            if callable(hook_obj):
                try:
                    return hook_obj(**params_dict)
                except TypeError as exc:  # pragma: no cover - error path
                    raise ConfigError(
                        f"Error calling hook factory '{import_string}' with params: {exc}",
                    ) from exc
            raise ConfigError(
                f"Hook '{import_string}' is not callable and cannot accept parameters",
            )

        return hook_obj

    def build_core_properties(self) -> AgentConfigBuilder:
        """Build core agent properties from configuration.

        Returns:
            Self for method chaining
        """
        self.agent_params["name"] = self.config.get("name", "configured_agent")
        self.agent_params["source_id"] = self.config.get("source_id")
        self.agent_params["max_context_tokens"] = self.config.get(
            "max_context_tokens",
            128000,
        )
        self.agent_params["max_running_subagents"] = self.config.get(
            "max_running_subagents",
            5,
        )
        self.agent_params["description"] = self.config.get("description")
        self.agent_params["system_prompt"] = self.config.get("system_prompt")
        self.agent_params["system_prompt_type"] = self.config.get(
            "system_prompt_type",
            "string",
        )
        self.agent_params["system_prompt_suffix"] = self.config.get("system_prompt_suffix")
        self.agent_params["initial_context"] = self.config.get("context", {})

        self.agent_params["stop_tools"] = set(self.config.get("stop_tools", []))
        self.agent_params["max_iterations"] = self.config.get("max_iterations", 100)
        self.agent_params["tool_call_mode"] = self.config.get("tool_call_mode", "structured")
        self.agent_params["retry_attempts"] = self.config.get("retry_attempts", 5)
        self.agent_params["retry_backoff_max_seconds"] = self.config.get("retry_backoff_max_seconds", 30)
        self.agent_params["timeout"] = self.config.get("timeout", 300)

        return self

    def build_mcp_servers(self) -> AgentConfigBuilder:
        """Build MCP servers configuration from configuration.

        Returns:
            Self for method chaining
        """
        mcp_servers_raw = self.config.get("mcp_servers", [])

        if not isinstance(mcp_servers_raw, list):
            raise ConfigError("'mcp_servers' must be a list")

        mcp_servers_list: list[Any] = cast(list[Any], mcp_servers_raw)

        # Validate each MCP server configuration
        typed_servers: list[dict[str, Any]] = []
        for i, server_config in enumerate(mcp_servers_list):
            server_config_typed = _require_dict(
                server_config,
                context=f"MCP server configuration {i}",
            )

            # Validate required fields
            if "name" not in server_config_typed:
                raise ConfigError(
                    f"MCP server configuration {i} missing 'name' field",
                )
            server_name = str(server_config_typed["name"])
            if not isinstance(server_config_typed.get("source_id"), str):
                server_config_typed["source_id"] = f"local:mcp_server:{server_name}"

            if "type" not in server_config_typed:
                raise ConfigError(
                    f"MCP server configuration {i} missing 'type' field",
                )

            server_type = str(server_config_typed["type"])
            if server_type not in ["stdio", "http", "sse"]:
                raise ConfigError(
                    f"MCP server configuration {i} has invalid type '{server_type}'. Must be one of: stdio, http, sse",
                )

            # Validate type-specific requirements
            if server_type == "stdio":
                if "command" not in server_config_typed:
                    raise ConfigError(
                        f"MCP server configuration {i} of type 'stdio' missing 'command' field",
                    )
            elif server_type in ["http", "sse"]:
                if "url" not in server_config_typed:
                    raise ConfigError(
                        f"MCP server configuration {i} of type '{server_type}' missing 'url' field",
                    )

            typed_servers.append(server_config_typed)

        self.agent_params["mcp_servers"] = typed_servers
        return self

    def build_hooks(self) -> AgentConfigBuilder:
        """Build hooks from configuration.

        Returns:
            Self for method chaining
        """
        middlewares: list[Any] | None = None
        if "middlewares" in self.config:
            middleware_configs = self.config["middlewares"]
            middlewares = []

            if not isinstance(middleware_configs, list):
                raise ConfigError("'middlewares' must be a list")

            middleware_config_list = cast(list[HookConfig], middleware_configs)

            for i, middleware_config in enumerate(middleware_config_list):
                try:
                    if not isinstance(middleware_config, (str, dict)) and not callable(middleware_config):
                        raise ConfigError(f"Middleware {i} must be a string, dict, or callable")
                    middleware = self._import_and_instantiate(cast(HookConfig, middleware_config))
                    middleware_dict = cast(dict[str, Any], middleware_config) if isinstance(middleware_config, dict) else None
                    if middleware_dict is not None and isinstance(middleware_dict.get("source_id"), str):
                        from nexau.archs.main_sub.execution.hooks import Middleware

                        if not isinstance(middleware, Middleware):
                            raise ConfigError(
                                f"Middleware {i} with source_id must be an instance of Middleware",
                            )
                        middleware.source_id = middleware_dict["source_id"]
                    middlewares.append(middleware)
                except Exception as e:
                    msg = f"Skipped middleware {i}: {e}"
                    self._handle_component_error(msg, e)

        self.agent_params["middlewares"] = middlewares

        # Handle after_model_hooks configuration
        after_model_hooks: list[Callable[..., Any]] | None = None
        if "after_model_hooks" in self.config:
            hooks_config = self.config["after_model_hooks"]
            after_model_hooks = []

            if not isinstance(hooks_config, list):
                raise ConfigError("'after_model_hooks' must be a list")

            hooks_config_list = cast(list[HookConfig], hooks_config)

            for i, hook_config in enumerate(hooks_config_list):
                try:
                    if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                        raise ConfigError(f"after_model_hooks entry {i} must be a string, dict, or callable")
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    after_model_hooks.append(hook_func)
                except Exception as e:
                    msg = f"Skipped after_model_hook {i}: {e}"
                    self._handle_component_error(msg, e)

        self.agent_params["after_model_hooks"] = after_model_hooks

        # Handle after_tool_hooks configuration
        after_tool_hooks: list[Callable[..., Any]] | None = None
        if "after_tool_hooks" in self.config:
            hooks_config = self.config["after_tool_hooks"]
            after_tool_hooks = []

            if not isinstance(hooks_config, list):
                raise ConfigError("'after_tool_hooks' must be a list")

            hooks_config_list = cast(list[HookConfig], hooks_config)

            for i, hook_config in enumerate(hooks_config_list):
                try:
                    if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                        raise ConfigError(f"after_tool_hooks entry {i} must be a string, dict, or callable")
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    after_tool_hooks.append(hook_func)
                except Exception as e:
                    msg = f"Skipped after_tool_hook {i}: {e}"
                    self._handle_component_error(msg, e)

        self.agent_params["after_tool_hooks"] = after_tool_hooks

        # Handle before_model_hooks configuration
        before_model_hooks: list[Callable[..., Any]] | None = None
        if "before_model_hooks" in self.config:
            hooks_config = self.config["before_model_hooks"]
            before_model_hooks = []

            if not isinstance(hooks_config, list):
                raise ConfigError("'before_model_hooks' must be a list")

            hooks_config_list = cast(list[HookConfig], hooks_config)

            for i, hook_config in enumerate(hooks_config_list):
                try:
                    if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                        raise ConfigError(f"before_model_hooks entry {i} must be a string, dict, or callable")
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    before_model_hooks.append(hook_func)
                except Exception as e:
                    msg = f"Skipped before_model_hook {i}: {e}"
                    self._handle_component_error(msg, e)

        self.agent_params["before_model_hooks"] = before_model_hooks

        # Handle before_tool_hooks configuration
        before_tool_hooks: list[Callable[..., Any]] | None = None
        if "before_tool_hooks" in self.config:
            hooks_config = self.config["before_tool_hooks"]
            before_tool_hooks = []

            if not isinstance(hooks_config, list):
                raise ConfigError("'before_tool_hooks' must be a list")

            hooks_config_list = cast(list[HookConfig], hooks_config)

            for i, hook_config in enumerate(hooks_config_list):
                try:
                    if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                        raise ConfigError(f"before_tool_hooks entry {i} must be a string, dict, or callable")
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    before_tool_hooks.append(hook_func)
                except Exception as e:
                    msg = f"Skipped before_tool_hook {i}: {e}"
                    self._handle_component_error(msg, e)

        self.agent_params["before_tool_hooks"] = before_tool_hooks

        return self

    def build_tracers(self) -> AgentConfigBuilder:
        """Build tracer instances from configuration."""
        tracer_configs = self.config.get("tracers", [])
        if tracer_configs is None:
            tracer_configs = []

        if not isinstance(tracer_configs, list):
            raise ConfigError("'tracers' must be a list")

        tracer_config_list = cast(list[BaseTracer | HookConfig], tracer_configs)

        resolved_tracers: list[BaseTracer] = []
        for entry in tracer_config_list:
            if entry is None:
                msg = "Skipped null tracer entry"
                logger.warning(msg)
                self._skipped_components.append(msg)
                continue

            if isinstance(entry, BaseTracer):
                tracer = entry
            elif isinstance(entry, (str, dict)) or callable(entry):
                try:
                    tracer = self._import_and_instantiate(cast(HookConfig, entry))
                except Exception as e:
                    msg = f"Skipped tracer '{entry}': {e}"
                    logger.warning(msg)
                    self._skipped_components.append(msg)
                    continue
            else:
                msg = f"Skipped tracer entry with unsupported type {type(entry)}"
                logger.warning(msg)
                self._skipped_components.append(msg)
                continue

            if not isinstance(tracer, BaseTracer):
                msg = f"Skipped tracer: expected BaseTracer instance, got {type(tracer)}"
                logger.warning(msg)
                self._skipped_components.append(msg)
                continue
            resolved_tracers.append(tracer)

        self.agent_params["tracers"] = resolved_tracers
        return self

    def build_tools(self) -> AgentConfigBuilder:
        """Build tools from configuration.

        Returns:
            Self for method chaining
        """
        tools: list[Tool] = []
        tool_configs = self.config.get("tools", [])
        for tool_config in tool_configs:
            try:
                tool = self._load_tool_from_config(tool_config, self.base_path)
                tools.append(tool)
            except Exception as e:
                tool_name = cast(dict[str, str], tool_config).get("name", "unknown") if isinstance(tool_config, dict) else "unknown"
                self._handle_component_error(f"Skipped tool '{tool_name}': {e}", e)

        self.agent_params["tools"] = tools
        return self

    def build_skills(self) -> AgentConfigBuilder:
        """Build skills from configuration.

        Returns:
            Self for method chaining
        """
        skills: list[Skill] = []

        # build skills from skill folders
        skill_configs_raw: object = self.config.get("skills", [])
        skill_configs = cast(list[object], skill_configs_raw) if isinstance(skill_configs_raw, list) else []
        seen_skill_names: dict[str, str | None] = {}
        for skill_config in skill_configs:
            try:
                configured_name: str | None = None
                source_id: str | None = None
                if isinstance(skill_config, dict):
                    skill_config_dict = cast(dict[str, Any], skill_config)
                    configured_name = cast(str | None, skill_config_dict.get("name"))
                    source_id = cast(str | None, skill_config_dict.get("source_id"))
                    skill_folder_raw = str(skill_config_dict["path"])
                else:
                    skill_folder_raw = str(skill_config)
                skill_folder = _resolve_config_path(skill_folder_raw, self.base_path)
                skill = Skill.from_folder(skill_folder)
                if configured_name:
                    skill.name = configured_name
                skill.source_id = source_id or f"local:skill:{skill.name}"
                previous_source = seen_skill_names.get(skill.name)
                if previous_source is not None:
                    raise ConfigError(f"Skill name conflict for '{skill.name}' between {previous_source} and {skill.source_id}")
                seen_skill_names[skill.name] = skill.source_id
                skills.append(skill)
            except Exception as e:
                self._handle_component_error(f"Skipped skill '{skill_config}': {e}", e)

        # add tool-based skills
        tool_call_mode = self.agent_params.get(
            "tool_call_mode",
            self.config.get("tool_call_mode", "structured"),
        )
        for tool in self.agent_params.get("tools", []):
            if tool.as_skill:
                skills.append(build_tool_skill(tool, tool_call_mode=tool_call_mode))

        self.agent_params["skills"] = skills
        return self

    def build_sub_agents(self) -> AgentConfigBuilder:
        """Build sub-agents from configuration.

        Returns:
            Self for method chaining
        """
        from nexau.archs.main_sub.config.expanded import ExpandedSubAgentConfig

        sub_agents: dict[str, AgentConfig] = {}
        sub_agent_configs_raw: object = self.config.get("sub_agents", [])
        if not isinstance(sub_agent_configs_raw, list):
            raise ConfigError("'sub_agents' must be a list")
        sub_agent_configs = cast(list[object], sub_agent_configs_raw)
        for sub_config in sub_agent_configs:
            try:
                sub_agent_name: str | None
                sub_agent_source_id: str | None
                sub_agent_config_path_raw: str | None
                inline_config: dict[str, Any] | None
                inline_base_path: Path | None
                if isinstance(sub_config, ExpandedSubAgentConfig):
                    sub_agent_name = sub_config.name
                    sub_agent_source_id = sub_config.source_id
                    sub_agent_config_path_raw = sub_config.config_path
                    inline_config = sub_config.inline_config
                    inline_base_path = sub_config.inline_base_path
                elif isinstance(sub_config, dict):
                    sub_config_dict = cast(dict[str, Any], sub_config)
                    sub_agent_name = cast(str | None, sub_config_dict.get("name"))
                    sub_agent_source_id = cast(str | None, sub_config_dict.get("source_id"))
                    sub_agent_config_path_raw = cast(str | None, sub_config_dict.get("config_path"))
                    inline_config = None
                    inline_base_path = None
                else:
                    raise ConfigError("Sub-agent configuration must be a mapping")

                overrides: dict[str, Any] | None = None
                if self.overrides:
                    overrides = self.overrides.copy()
                    if sub_agent_name:
                        overrides["name"] = sub_agent_name

                if not isinstance(sub_agent_config_path_raw, str) or not sub_agent_config_path_raw:
                    raise ConfigError("Sub-agent configuration missing 'config_path' field")

                if inline_config is not None and inline_base_path is not None:
                    sub_agent_config = AgentConfig.from_dict(
                        inline_config,
                        inline_base_path,
                        overrides,
                        options=AgentConfigLoadOptions(strict=self.strict, expand_plugins=False),
                    )
                else:
                    config_path_cm = _resolve_config_resource(sub_agent_config_path_raw, self.base_path)
                    with config_path_cm as config_path:
                        sub_agent_config = AgentConfig.from_yaml(
                            config_path,
                            overrides,
                            options=AgentConfigLoadOptions(strict=self.strict, expand_plugins=False),
                        )

                if sub_agent_config.name is None:
                    raise ConfigError(
                        "Sub-agent configuration must have a name",
                    )
                if sub_agent_name:
                    sub_agent_config.name = sub_agent_name
                if sub_agent_source_id:
                    sub_agent_config.source_id = sub_agent_source_id
                elif sub_agent_config.source_id is None:
                    sub_agent_config.source_id = f"local:sub_agent:{sub_agent_config.name}"
                sub_agent_name_final = sub_agent_config.name
                if sub_agent_name_final in sub_agents:
                    existing_source = sub_agents[sub_agent_name_final].source_id
                    raise ConfigError(
                        f"Sub-agent name conflict for '{sub_agent_name_final}' between {existing_source} and {sub_agent_config.source_id}",
                    )
                sub_agents[sub_agent_name_final] = sub_agent_config
            except Exception as e:
                if isinstance(sub_config, ExpandedSubAgentConfig):
                    sub_name = sub_config.name
                elif isinstance(sub_config, dict):
                    sub_name = str(cast(dict[str, object], sub_config).get("name", "unknown"))
                else:
                    sub_name = "unknown"
                self._handle_component_error(f"Skipped sub-agent '{sub_name}': {e}", e)

        self.agent_params["sub_agents"] = sub_agents
        return self

    def build_llm_config(self) -> AgentConfigBuilder:
        """Build LLM configuration and related components.

        Returns:
            Self for method chaining
        """
        # Handle LLM configuration
        if "llm_config" not in self.config:
            raise ConfigError(
                "'llm_config' is required in agent configuration",
            )

        self.agent_params["llm_config"] = LLMConfig(
            **self.config["llm_config"],
        )

        # Handle token counter configuration
        token_counter = None
        if "token_counter" in self.config:
            token_counter_config = self.config["token_counter"]
            if isinstance(token_counter_config, str):
                # Import string format: "module.path:function_name"
                token_counter = import_from_string(token_counter_config)
            elif isinstance(token_counter_config, dict):
                token_counter_config_dict = cast(dict[str, Any], token_counter_config)
                # Dictionary format with import and optional parameters
                import_string_value: str | None = (
                    token_counter_config_dict.get("import") if isinstance(token_counter_config_dict.get("import"), str) else None
                )
                if not import_string_value:
                    raise ConfigError(
                        "Token counter configuration missing 'import' field",
                    )
                import_string = import_string_value

                # Import the function/class
                token_counter_func = import_from_string(import_string)

                # Check if there are parameters to pass
                params_raw: dict[str, Any] | None = (
                    token_counter_config_dict.get("params", {}) if isinstance(token_counter_config_dict.get("params", {}), dict) else None
                )
                if params_raw is None:
                    raise ConfigError("Token counter params must be a mapping when provided")
                params_dict = params_raw
                if params_dict:
                    try:
                        signature = inspect.signature(token_counter_func)
                    except (TypeError, ValueError):
                        signature = None

                    supports_tools_param = False
                    if signature is not None:
                        supports_tools_param = "tools" in signature.parameters or any(
                            parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
                        )

                    # Create a wrapper function with the parameters
                    def configured_token_counter(
                        messages: Sequence[Message],
                        tools: list[dict[str, Any]] | None = None,
                    ) -> int:
                        call_params = dict(params_dict)
                        if tools is not None and supports_tools_param:
                            call_params["tools"] = tools
                        return int(token_counter_func(messages, **call_params))

                    token_counter = configured_token_counter
                else:
                    token_counter = token_counter_func
            else:
                raise ConfigError(
                    "Token counter configuration must be a string or dictionary",
                )

        self.agent_params["token_counter"] = token_counter

        return self

    def build_system_prompt_path(self) -> AgentConfigBuilder:
        """Build system prompt path resolution.

        Returns:
            Self for method chaining
        """
        system_prompt = self.agent_params.get("system_prompt")
        system_prompt_type = self.agent_params.get(
            "system_prompt_type",
            "string",
        )

        # Convert system_prompt from relative path to absolute path
        # When system_prompt is a list, resolve paths for each item individually
        if system_prompt and system_prompt_type in ["file", "jinja"]:
            if isinstance(system_prompt, list):
                from .base import SystemPromptBlock

                prompt_items = cast(list[str | SystemPromptBlock | dict[str, str | bool]], system_prompt)
                resolved: list[str | SystemPromptBlock] = []
                for item in prompt_items:
                    # Extract the path string from str, SystemPromptBlock, or dict
                    path_str: str
                    cache: bool
                    if isinstance(item, SystemPromptBlock):
                        path_str = item.content
                        cache = item.cache
                    elif isinstance(item, dict):
                        path_str = str(item["content"])
                        cache = bool(item.get("cache", True))
                    else:
                        path_str = str(item)
                        cache = True

                    if not Path(path_str).is_absolute():
                        resolved_path = _resolve_config_path(path_str, self.base_path)
                        abs_path: str = str(resolved_path)
                        if not resolved_path.exists():
                            self._handle_component_error(f"Skipped system prompt file (not found): {abs_path}")
                            continue
                        resolved.append(SystemPromptBlock(content=abs_path, cache=cache))
                    else:
                        # Wrap all items consistently as SystemPromptBlock
                        resolved.append(SystemPromptBlock(content=path_str, cache=cache))
                self.agent_params["system_prompt"] = resolved
            elif not Path(system_prompt).is_absolute():
                system_prompt = _resolve_config_path(str(system_prompt), self.base_path)
                if not system_prompt.exists():
                    self._handle_component_error(f"Skipped system prompt file (not found): {system_prompt}")
                    # non-strict fallback: default to empty string prompt
                    self.agent_params["system_prompt"] = ""
                    self.agent_params["system_prompt_type"] = "string"
                else:
                    self.agent_params["system_prompt"] = str(system_prompt)

        return self

    def build_sandbox(self):
        """Build sandbox configuration.

        Returns:
            Self for method chaining
        """
        sandbox_config = self.config.get("sandbox_config", None)
        self.agent_params["sandbox_config"] = sandbox_config
        return self

    def set_overrides(self, overrides: dict[str, Any] | None) -> AgentConfigBuilder:
        """Set overrides for sub-agent loading.

        Args:
            overrides: Configuration overrides

        Returns:
            Self for method chaining
        """
        self.overrides = overrides
        return self

    def get_agent_config(self) -> AgentConfig:
        """Get the agent configuration.

        Returns:
            Agent configuration dictionary
        """
        if self._skipped_components:
            agent_name = self.agent_params.get("name", "unknown")
            summary = "\n  - ".join(self._skipped_components)
            logger.warning(
                "Agent '%s' initialized with %d skipped component(s):\n  - %s",
                agent_name,
                len(self._skipped_components),
                summary,
            )
        self.agent_params["skipped_components"] = self._skipped_components
        return AgentConfig(**self.agent_params)

    def _load_tool_from_config(self, tool_config: dict[str, Any], base_path: Path) -> Tool:
        """
        Load a tool from configuration.

        Args:
            tool_config: Tool configuration dictionary
            base_path: Base path for resolving relative paths

        Returns:
            Configured Tool instance
        """
        name = tool_config.get("name")
        if not name:
            raise ConfigError("Tool configuration missing 'name' field")

        yaml_path = tool_config.get("yaml_path")
        binding = tool_config.get("binding", None)
        source_id = tool_config.get("source_id") if isinstance(tool_config.get("source_id"), str) else f"local:tool:{name}"
        lazy_raw: object = tool_config.get("lazy", False)
        if not isinstance(lazy_raw, bool):
            raise ConfigError(f"Tool '{name}' field 'lazy' must be a boolean")
        lazy = lazy_raw
        as_skill = tool_config.get("as_skill", False)
        defer_loading: object = tool_config.get("defer_loading")
        if defer_loading is not None and not isinstance(defer_loading, bool):
            raise ConfigError(f"Tool '{name}' field 'defer_loading' must be a boolean")
        extra_kwargs_raw: object | None = tool_config.get("extra_kwargs", {})

        if not yaml_path:
            raise ConfigError(f"Tool '{name}' missing 'yaml_path' field")

        if extra_kwargs_raw is None:
            extra_kwargs_raw = {}
        extra_kwargs = _require_dict(extra_kwargs_raw, context=f"Tool '{name}' extra_kwargs")
        reserved_keys = {"agent_state", "global_storage"}
        conflict_keys = set(extra_kwargs) & reserved_keys
        if conflict_keys:
            raise ConfigError(
                f"Tool '{name}' extra_kwargs contains reserved keys that cannot be overridden: {sorted(conflict_keys)}",
            )

        # RFC-0019: 解析权限配置
        permissions_raw: object | None = tool_config.get("permissions")
        permissions: dict[str, list[str]] | None = None
        if permissions_raw is not None:
            if isinstance(permissions_raw, dict):
                permissions = cast(dict[str, list[str]], permissions_raw)
            else:
                raise ConfigError(f"Tool '{name}' field 'permissions' must be a dict")

        # Resolve YAML path
        yaml_path = str(_resolve_config_path(yaml_path, base_path))

        # Create tool with effective config-level overrides
        return Tool.from_yaml(
            str(yaml_path),
            binding,
            as_skill=as_skill,
            extra_kwargs=extra_kwargs,
            lazy=lazy,
            name=name,
            defer_loading=defer_loading,
            permissions=permissions,
            source_id=source_id,
        )
