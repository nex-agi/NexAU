"""Configuration models for the NexAU agent framework."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeVar

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

TTool = TypeVar("TTool", bound=object)
TSkill = TypeVar("TSkill", bound=object)
TSubAgent = TypeVar("TSubAgent", bound=object)
THook = TypeVar("THook", bound=object)


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

    type: Literal["agent"] | None = Field(default=None)
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    system_prompt_type: Literal["string", "file", "jinja"] = "string"
    tools: list[Any] = Field(default_factory=list)
    sub_agents: TSubAgent | None = None
    skills: list[Any] = Field(default_factory=list)
    llm_config: Any | None = None
    stop_tools: set[str] | None = Field(default_factory=set)
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
    middlewares: list[THook] | None = None
    error_handler: Callable[..., Any] | None = None
    token_counter: HookDefinition | None = None
    global_storage: dict[str, Any] = Field(default_factory=dict)
    max_context_tokens: int = Field(default=128000, ge=1)
    max_running_subagents: int = Field(default=5, ge=0)
    max_iterations: int = Field(default=100, ge=1)
    tool_call_mode: str = "openai"
    retry_attempts: int = Field(default=5, ge=0)
    timeout: int = Field(default=300, ge=1)
    tracers: list[Any] = Field(default_factory=list)
