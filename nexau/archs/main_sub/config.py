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

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from ..llm.llm_config import LLMConfig
from ..main_sub.skill import Skill
from ..tool import Tool
from ..tool.builtin.skill_tool import load_skill
from .tool_call_modes import normalize_tool_call_mode

TTool = TypeVar("TTool")
TSkill = TypeVar("TSkill")
TSubAgent = TypeVar("TSubAgent")
THook = TypeVar("THook")


class HookImportConfig(BaseModel):
    """Configuration block for importing a hook callable."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    import_path: str = Field(alias="import")
    params: dict[str, Any] | None = None


HookCallable = Callable[..., Any]
HookDefinition = HookCallable | HookImportConfig | str


class AgentConfigBase[TTool, TSkill, TSubAgent, THook](BaseModel):
    """Generic base for agent configuration structures."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    name: str | None = None
    description: str | None = None
    agent_id: str | None = None
    system_prompt: str | None = None
    system_prompt_type: Literal["string", "file", "jinja"] = "string"
    tools: list[TTool] = Field(default_factory=list)
    sub_agents: list[TSubAgent] | None = None
    skills: list[TSkill] = Field(default_factory=list)
    llm_config: Any | None = None
    stop_tools: list[str] = Field(default_factory=list)
    initial_state: dict[str, Any] | None = None
    initial_config: dict[str, Any] | None = None
    initial_context: dict[str, Any] | None = Field(
        default=None,
        alias="context",
        validation_alias=AliasChoices("context", "initial_context"),
    )
    mcp_servers: list[Any] = Field(default_factory=list)
    after_model_hooks: list[THook] | None = None
    after_tool_hooks: list[THook] | None = None
    before_model_hooks: list[THook] | None = None
    before_tool_hooks: list[THook] | None = None
    middlewares: list[Any] | None = None
    middlewares: list[THook] | None = None
    error_handler: Callable | None = None
    token_counter: Callable | None = None
    global_storage: dict[str, Any] = Field(default_factory=dict)
    max_context_tokens: int = Field(default=128000, ge=1)
    max_running_subagents: int = Field(default=5, ge=0)
    max_iterations: int = Field(default=100, ge=1)
    tool_call_mode: str = "openai"
    retry_attempts: int = Field(default=5, ge=0)
    timeout: int = Field(default=300, ge=1)


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


class AgentConfig(
    AgentConfigBase[
        Tool,
        Skill,
        tuple[str, Callable[[], Any]],
        HookCallable,
    ],
):
    """Configuration for an Agent's definition and behavior."""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    sub_agents: list[tuple[str, Callable[[], Any]]] | None = None
    llm_config: LLMConfig | dict[str, Any] | None = None
    stop_tools: set[str] | None = None
    mcp_servers: list[dict[str, Any]] = Field(default_factory=list)
    after_model_hooks: list[Callable] | None = None
    after_tool_hooks: list[Callable] | None = None
    before_model_hooks: list[Callable] | None = None
    before_tool_hooks: list[Callable] | None = None
    middlewares: list[Any] | None = None
    sub_agent_factories: dict[str, Callable[[], Any]] = Field(
        default_factory=dict,
        exclude=True,
    )

    @field_validator("llm_config", mode="before")
    @classmethod
    def _validate_llm_config(cls, value):
        if value is None:
            return value
        if isinstance(value, (LLMConfig, dict)):
            return value
        raise ValueError(
            f"Invalid llm_config type: {type(value)}",
        )

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _ensure_mcp_servers(cls, value):
        if value is None:
            return []
        return value

    @model_validator(mode="after")
    def _finalize(self):  # type: ignore[override]
        """Finalize configuration by normalizing fields and injecting skill tool."""
        # Convert sub_agents list to dictionary if provided
        if self.sub_agents:
            self.sub_agent_factories = dict(self.sub_agents)
        else:
            self.sub_agent_factories = {}

        nexau_package_path = Path(__file__).parent.parent.parent
        has_skilled_tools = any(tool.as_skill for tool in self.tools)
        if has_skilled_tools or self.skills:
            skill_tool = Tool.from_yaml(
                str(nexau_package_path / "archs" / "tool" / "builtin" / "description" / "skill_tool.yaml"),
                binding=load_skill,
                as_skill=False,
            )
            skill_tool.description += self._generate_skill_description()
            self.tools.append(skill_tool)

        # Ensure stop_tools is a set for faster lookup
        if isinstance(self.stop_tools, list):
            self.stop_tools = set(self.stop_tools)
        elif self.stop_tools is None:
            self.stop_tools = set()

        # Handle LLM configuration
        if self.llm_config is None:
            raise ValueError("llm_config is required")
        if isinstance(self.llm_config, dict):
            self.llm_config = LLMConfig(**self.llm_config)
        elif not isinstance(self.llm_config, LLMConfig):
            raise ValueError(
                f"Invalid llm_config type: {type(self.llm_config)}",
            )

        # Ensure name is set
        if not self.name:
            self.name = f"agent_{id(self)}"

        return self

    def _generate_skill_description(self) -> str:
        """Generate skill description."""
        skill_description = "<Skills>\n"
        for skill in self.skills:
            skill_description += "<SkillBrief>\n"
            skill_description += f"Skill Name: {skill.name}\n"
            skill_description += f"Skill Folder: {skill.folder}\n"
            skill_description += f"Skill Brief Description: {skill.description}\n\n"
            skill_description += "</SkillBrief>\n"

        for tool in self.tools:
            if tool.as_skill:
                skill_description += "<SkillBrief>\n"
                skill_description += f"Skill: {tool.name}\n"
                if not tool.skill_description:
                    raise ValueError(f"Tool {tool.name} has no skill description but is marked as a skill")
                skill_description += f"Skill Brief Description: {tool.skill_description}\n\n"
                skill_description += "</SkillBrief>\n"

        skill_description += "</Skills>\n"
        return skill_description
