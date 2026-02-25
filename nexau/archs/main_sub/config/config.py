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
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import dotenv
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.prompt_builder import PromptBuilder
from nexau.archs.main_sub.skill import Skill, build_load_skill_tool
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

from .base import AgentConfigBase, HookCallable, HookDefinition
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


def _require_dict(value: object, *, context: str) -> dict[str, Any]:
    """Ensure a value is a dictionary and return it typed."""
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be a dictionary")
    return cast(dict[str, Any], value)


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
    ) -> AgentConfig:
        """
        Load a sub-agent factory from configuration.

        Args:
            sub_config: Sub-agent configuration dictionary
            base_path: Base path for resolving relative paths
            overrides: Dictionary of configuration overrides to pass through

        Returns:
            Tuple of (agent_name, agent_factory)
        """
        if overrides:
            warnings.warn(
                "Overrides will be removed in the v0.4.0, instead use agent_config = "
                "AgentConfig.from_yaml(...) then agent_config.key = value for "
                "overrides.",
            )

        agent_config_schema = AgentConfigSchema.from_yaml(str(config_path), overrides)

        agent_builder = AgentConfigBuilder(
            agent_config_schema.model_dump(
                mode="python",
                by_alias=True,
                exclude_none=True,
            ),
            config_path.parent,
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
        """Finalize configuration by normalizing fields and injecting skill tool."""
        if self._is_finalized:
            return self
        from nexau.archs.tool.builtin.recall_sub_agent_tool import recall_sub_agent

        nexau_package_path = Path(__file__).parent.parent.parent.parent
        if self.sub_agents:
            recall_subagent_tool = Tool.from_yaml(
                str(nexau_package_path / "archs" / "tool" / "builtin" / "description" / "recall_sub_agent_tool.yaml"),
                binding=recall_sub_agent,
            )
            self.tools.append(recall_subagent_tool)

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
    timeout: int = 300
    tool_call_mode: str = "openai"

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
            timeout=agent_config.timeout,
            tool_call_mode=agent_config.tool_call_mode,
        )


class AgentConfigBuilder:
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
        self.overrides: dict[str, Any] | None = None

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
        self.agent_params["initial_context"] = self.config.get("context", {})

        self.agent_params["stop_tools"] = set(self.config.get("stop_tools", []))
        self.agent_params["max_iterations"] = self.config.get("max_iterations", 100)
        self.agent_params["tool_call_mode"] = self.config.get("tool_call_mode", "openai")

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
        middlewares: list[Callable[..., Any]] | None = None
        if "middlewares" in self.config:
            middleware_configs = self.config["middlewares"]
            middlewares = []

            if not isinstance(middleware_configs, list):
                raise ConfigError("'middlewares' must be a list")

            middleware_config_list = cast(list[HookConfig], middleware_configs)

            for i, middleware_config in enumerate(middleware_config_list):
                if not isinstance(middleware_config, (str, dict)) and not callable(middleware_config):
                    raise ConfigError(f"Middleware {i} must be a string, dict, or callable")
                try:
                    middleware = self._import_and_instantiate(cast(HookConfig, middleware_config))
                    middlewares.append(middleware)
                except Exception as e:
                    raise ConfigError(f"Error loading middleware {i}: {e}")

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
                if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                    raise ConfigError(f"after_model_hooks entry {i} must be a string, dict, or callable")
                try:
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    after_model_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading hook {i}: {e}")

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
                if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                    raise ConfigError(f"after_tool_hooks entry {i} must be a string, dict, or callable")
                try:
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    after_tool_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading tool hook {i}: {e}")

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
                if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                    raise ConfigError(f"before_model_hooks entry {i} must be a string, dict, or callable")
                try:
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    before_model_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading before model hook {i}: {e}")

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
                if not isinstance(hook_config, (str, dict)) and not callable(hook_config):
                    raise ConfigError(f"before_tool_hooks entry {i} must be a string, dict, or callable")
                try:
                    hook_func = self._import_and_instantiate(cast(HookConfig, hook_config))
                    before_tool_hooks.append(hook_func)
                except Exception as e:
                    raise ConfigError(f"Error loading before tool hook {i}: {e}")

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
                raise ConfigError("Tracer entries cannot be null")

            if isinstance(entry, BaseTracer):
                tracer = entry
            elif isinstance(entry, (str, dict)) or callable(entry):
                try:
                    tracer = self._import_and_instantiate(cast(HookConfig, entry))
                except Exception as e:
                    raise ConfigError(f"Error loading tracer: {e}")
            else:
                raise ConfigError(f"Tracer entry must be BaseTracer, string, dict, or callable, got {type(entry)}")

            if not isinstance(tracer, BaseTracer):
                raise ConfigError(
                    f"Tracer must be an instance of BaseTracer, got {type(tracer)}",
                )
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
                raise ConfigError(
                    f"Error loading tool '{tool_config.get('name', 'unknown')}': {e}",
                )

        self.agent_params["tools"] = tools
        return self

    def build_skills(self) -> AgentConfigBuilder:
        """Build skills from configuration.

        Returns:
            Self for method chaining
        """
        skills: list[Skill] = []

        # build skills from skill folders
        skill_configs = self.config.get("skills", [])
        for skill_folder in skill_configs:
            try:
                if not Path(skill_folder).is_absolute():
                    skill_folder = self.base_path / skill_folder
                skill = Skill.from_folder(skill_folder)
                skills.append(skill)
            except Exception as e:
                raise ConfigError(
                    f"Error loading skill '{skill_folder}': {e}",
                )

        # add skill tool to tools
        prompt_builder = PromptBuilder()
        skill_detail_template = prompt_builder.load_prompt_template("tools_template_for_skill_detail")
        jinja_template = prompt_builder.jinja_env.from_string(skill_detail_template)
        for tool in self.agent_params.get("tools", []):
            if tool.as_skill:
                skill_detail = jinja_template.render({"tool": tool})
                skills.append(Skill(name=tool.name, description=tool.skill_description, detail=skill_detail, folder=""))

        self.agent_params["skills"] = skills
        return self

    def build_sub_agents(self) -> AgentConfigBuilder:
        """Build sub-agents from configuration.

        Returns:
            Self for method chaining
        """
        sub_agents: dict[str, AgentConfig] = {}
        sub_agent_configs = self.config.get("sub_agents", [])
        for sub_config in sub_agent_configs:
            try:
                sub_agent_name = sub_config.get("name", None)

                overrides: dict[str, Any] | None = None
                if self.overrides:
                    overrides = self.overrides.copy()
                    if sub_agent_name:
                        overrides["name"] = sub_agent_name

                sub_agent_config_path_raw = sub_config.get("config_path", None)
                if not isinstance(sub_agent_config_path_raw, str) or not sub_agent_config_path_raw:
                    raise ConfigError("Sub-agent configuration missing 'config_path' field")

                # Support both filesystem paths and importlib resources:
                #   - "some_package:relative/path.yaml"
                #   - "/abs/path.yaml" or "relative/path.yaml"
                if ":" in sub_agent_config_path_raw:
                    pkg, resource_path = sub_agent_config_path_raw.split(":", 1)
                    from importlib.resources import as_file, files

                    resource = files(pkg).joinpath(resource_path)
                    with as_file(resource) as config_path:
                        sub_agent_config = AgentConfig.from_yaml(config_path, overrides)
                else:
                    config_path = Path(sub_agent_config_path_raw)
                    if not config_path.is_absolute():
                        config_path = self.base_path / config_path
                    sub_agent_config = AgentConfig.from_yaml(config_path, overrides)

                if sub_agent_config.name is None:
                    raise ConfigError(
                        "Sub-agent configuration must have a name",
                    )
                sub_agents[sub_agent_config.name] = sub_agent_config
            except Exception as e:
                raise ConfigError(
                    f"Error loading sub-agent '{sub_config.get('name', 'unknown')}': {e}",
                )

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
                    # Create a wrapper function with the parameters
                    def configured_token_counter(messages: Sequence[Message] | Sequence[dict[str, Any]]) -> int:
                        # Internally NexAU uses UMP Messages; preserve backward compatibility
                        # for custom token counters that were written against legacy dict messages.
                        from nexau.core.adapters.legacy import messages_to_legacy_openai_chat
                        from nexau.core.messages import Message

                        if messages and isinstance(messages[0], Message):
                            legacy_messages = messages_to_legacy_openai_chat(list(cast(Sequence[Message], messages)))
                        else:
                            legacy_messages = list(cast(Sequence[dict[str, Any]], messages))
                        return int(token_counter_func(legacy_messages, **params_dict))

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
        if system_prompt and system_prompt_type in ["file", "jinja"] and not Path(system_prompt).is_absolute():
            system_prompt = self.base_path / system_prompt
            if not Path(system_prompt).exists():
                raise ConfigError(
                    f"System prompt file not found: {system_prompt}",
                )
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
        # todo
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
        lazy_raw: object = tool_config.get("lazy", False)
        if not isinstance(lazy_raw, bool):
            raise ConfigError(f"Tool '{name}' field 'lazy' must be a boolean")
        lazy = lazy_raw
        as_skill = tool_config.get("as_skill", False)
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

        # Resolve YAML path
        if not Path(yaml_path).is_absolute():
            if ":" in yaml_path:
                res = yaml_path.split(":")
                from importlib.resources import files

                yaml_path = files(res[0]).joinpath(res[1])
            else:
                yaml_path = base_path / yaml_path

        # Create tool
        tool = Tool.from_yaml(str(yaml_path), binding, as_skill=as_skill, extra_kwargs=extra_kwargs, lazy=lazy)

        # Override tool name with config-provided alias if present
        if name and tool.name != name:
            setattr(tool, "source_name", getattr(tool, "source_name", tool.name))
            tool.name = name

        return tool
