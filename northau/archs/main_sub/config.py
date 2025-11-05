"""Configuration models for the Northau agent framework."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.llm_config import LLMConfig
from ..main_sub.skill import Skill
from ..tool import Tool
from ..tool.builtin.skill_tool import load_skill
from .tool_call_modes import normalize_tool_call_mode


@dataclass
class ExecutionConfig:
    """Configuration for agent execution environment and behavior."""

    max_iterations: int = 100
    max_context_tokens: int = 128000
    max_running_subagents: int = 5
    retry_attempts: int = 5
    timeout: int = 300
    tool_call_mode: str = "xml"

    def __post_init__(self) -> None:
        """Validate execution configuration."""
        self.tool_call_mode = normalize_tool_call_mode(self.tool_call_mode)


@dataclass
class AgentConfig:
    """Configuration for an Agent's definition and behavior."""

    name: str | None = None
    agent_id: str | None = None
    system_prompt: str | None = None
    system_prompt_type: str = "string"
    tools: list[Tool] = field(default_factory=list)
    sub_agents: list[tuple[str, Callable[[], Any]]] | None = None
    skills: list[Skill] = field(default_factory=list)
    llm_config: LLMConfig | dict[str, Any] | None = None
    stop_tools: list[str] = field(default_factory=list)

    # Context parameters
    initial_state: dict[str, Any] | None = None
    initial_config: dict[str, Any] | None = None
    initial_context: dict[str, Any] | None = None

    # MCP parameters
    mcp_servers: list[dict[str, Any]] | None = None

    # Hook parameters
    after_model_hooks: list[Callable] | None = None
    after_tool_hooks: list[Callable] | None = None
    before_model_hooks: list[Callable] | None = None

    # Advanced features
    error_handler: Callable | None = None
    token_counter: Callable[[list[dict[str, str]]], int] | None = None
    custom_llm_generator: Callable[[Any, dict[str, Any]], Any] | None = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Convert sub_agents list to dictionary if provided
        if self.sub_agents:
            self.sub_agent_factories = dict(self.sub_agents)
        else:
            self.sub_agent_factories = {}

        northau_package_path = Path(__file__).parent.parent.parent
        has_skilled_tools = False
        for tool in self.tools:
            if tool.as_skill:
                has_skilled_tools = True
                break
        if has_skilled_tools or self.skills:
            skill_tool = Tool.from_yaml(
                str(northau_package_path / "archs" / "tool" / "builtin" / "description" / "skill_tool.yaml"),
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
        elif isinstance(self.llm_config, dict):
            self.llm_config = LLMConfig(**self.llm_config)
        elif not isinstance(self.llm_config, LLMConfig):
            raise ValueError(
                f"Invalid llm_config type: {type(self.llm_config)}",
            )

        # Ensure name is set
        if not self.name:
            self.name = f"agent_{id(self)}"

    def _generate_skill_description(self) -> str:
        """Generate skill description."""
        skill_description = "<Skils>\n"
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
