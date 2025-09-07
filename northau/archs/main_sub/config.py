"""Configuration models for the Northau agent framework."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
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
    name: Optional[str] = None
    agent_id: Optional[str] = None
    system_prompt: Optional[str] = None
    system_prompt_type: str = "string"
    tools: List[Any] = field(default_factory=list)
    sub_agents: Optional[List[Tuple[str, Callable[[], Any]]]] = None
    llm_config: Optional[Union[LLMConfig, Dict[str, Any]]] = None
    stop_tools: List[str] = field(default_factory=list)
    
    # Context parameters
    initial_state: Optional[Dict[str, Any]] = None
    initial_config: Optional[Dict[str, Any]] = None
    initial_context: Optional[Dict[str, Any]] = None
    
    # MCP parameters
    mcp_servers: Optional[List[Dict[str, Any]]] = None
    
    # Hook parameters
    after_model_hooks: Optional[List[Callable]] = None
    after_tool_hooks: Optional[List[Callable]] = None
    
    # Advanced features
    error_handler: Optional[Callable] = None
    token_counter: Optional[Callable[[List[Dict[str, str]]], int]] = None
    custom_llm_generator: Optional[Callable[[Any, Dict[str, Any]], Any]] = None
    
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
            raise ValueError(f"Invalid llm_config type: {type(self.llm_config)}")
        
        # Ensure name is set
        if not self.name:
            self.name = f"agent_{id(self)}"