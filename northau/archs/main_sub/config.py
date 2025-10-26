"""Configuration models for the Northau agent framework."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..llm.llm_config import LLMConfig


@dataclass
class ExecutionConfig:
    """Configuration for agent execution environment and behavior."""

    max_iterations: int = 100
    max_context_tokens: int = 128000
    max_running_subagents: int = 5
    retry_attempts: int = 5
    timeout: int = 300


@dataclass
class AgentConfig:
    """Configuration for an Agent's definition and behavior."""

    name: str | None = None
    agent_id: str | None = None
    system_prompt: str | None = None
    system_prompt_type: str = "string"
    tools: list[Any] = field(default_factory=list)
    sub_agents: list[tuple[str, Callable[[], Any]]] | None = None
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
