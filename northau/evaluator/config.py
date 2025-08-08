"""Configuration management for evaluation system."""

import json
import yaml
import copy
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import uuid


class Config:
    """Agent configuration for evaluation purposes."""
    
    def __init__(
        self,
        config_id: Optional[str] = None,
        system_prompts: Dict[str, str] = None,
        tool_descriptions: Dict[str, str] = None,
        llm_config: Dict[str, Any] = None,
        agent_parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ):
        self.config_id = config_id or str(uuid.uuid4())
        self.system_prompts = system_prompts or {}
        self.tool_descriptions = tool_descriptions or {}
        self.llm_config = llm_config or {}
        self.agent_parameters = agent_parameters or {}
        self.metadata = metadata or {}
    
    def copy(self) -> "Config":
        """Create deep copy of configuration."""
        return Config(
            config_id=None,  # Generate new ID
            system_prompts=copy.deepcopy(self.system_prompts),
            tool_descriptions=copy.deepcopy(self.tool_descriptions),
            llm_config=copy.deepcopy(self.llm_config),
            agent_parameters=copy.deepcopy(self.agent_parameters),
            metadata=copy.deepcopy(self.metadata)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "config_id": self.config_id,
            "system_prompts": self.system_prompts,
            "tool_descriptions": self.tool_descriptions,
            "llm_config": self.llm_config,
            "agent_parameters": self.agent_parameters,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(
            config_id=data.get("config_id"),
            system_prompts=data.get("system_prompts", {}),
            tool_descriptions=data.get("tool_descriptions", {}),
            llm_config=data.get("llm_config", {}),
            agent_parameters=data.get("agent_parameters", {}),
            metadata=data.get("metadata", {})
        )
    
    def save_yaml(self, filepath: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
    
    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "Config":
        """Load configuration from YAML file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required fields
        if not self.system_prompts or not any(prompt.strip() for prompt in self.system_prompts.values()):
            issues.append("System prompt dictionary cannot be empty or contain only empty prompts")
        
        # Check LLM config constraints
        if "temperature" in self.llm_config:
            temp = self.llm_config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                issues.append("Temperature must be between 0 and 2")
        
        if "max_tokens" in self.llm_config:
            max_tokens = self.llm_config["max_tokens"]
            if not isinstance(max_tokens, int) or max_tokens < 1:
                issues.append("max_tokens must be positive integer")
        
        # Check agent parameters
        if "retry_attempts" in self.agent_parameters:
            retries = self.agent_parameters["retry_attempts"]
            if not isinstance(retries, int) or retries < 0:
                issues.append("retry_attempts must be non-negative integer")
        
        return issues