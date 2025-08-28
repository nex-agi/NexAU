"""System prompt handling for different prompt types."""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import re


class PromptHandler:
    """Handles different types of system prompts: string, file, and Jinja templates."""
    
    def __init__(self):
        """Initialize the prompt handler."""
        self._jinja_env = None
        self._setup_jinja()
    
    def _setup_jinja(self):
        """Setup Jinja2 environment."""
        from jinja2 import Environment, BaseLoader
        
        # Create a basic environment
        self._jinja_env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def process_prompt(
        self,
        prompt: str,
        prompt_type: str = "string",
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a system prompt based on its type.
        
        Args:
            prompt: The prompt content or file path
            prompt_type: Type of prompt ("string", "file", "jinja")
            context: Context variables for template rendering
        
        Returns:
            Processed prompt string
        """
        if prompt_type == "string":
            return self._process_string_prompt(prompt, context)
        elif prompt_type == "file":
            return self._process_file_prompt(prompt, context)
        elif prompt_type == "jinja":
            return self._process_jinja_prompt(prompt, context)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    def _process_string_prompt(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a string prompt (may contain simple variable substitution)."""
        if not prompt:
            return ""
        
        # Simple variable substitution using {variable} syntax
        if context:
            try:
                return prompt.format(**context)
            except KeyError as e:
                # If some variables are missing, leave them as-is
                return prompt
        
        return prompt
    
    def _process_file_prompt(
        self, 
        file_path: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a file-based prompt."""
        path = Path(file_path)
        
        if not path.exists():
            # Try relative to current working directory
            cwd_path = Path.cwd() / file_path
            if cwd_path.exists():
                path = cwd_path
            else:
                raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply simple variable substitution if context provided
            if context:
                try:
                    content = content.format(**context)
                except KeyError:
                    # If some variables are missing, leave them as-is
                    pass
            
            return content.strip()
            
        except Exception as e:
            raise ValueError(f"Error reading prompt file {file_path}: {e}")
    
    def _process_jinja_prompt(
        self, 
        template_path: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a Jinja template prompt."""
        path = Path(template_path)
        
        if not path.exists():
            # Try relative to current working directory
            cwd_path = Path.cwd() / template_path
            if cwd_path.exists():
                path = cwd_path
            else:
                raise FileNotFoundError(f"Jinja template not found: {template_path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Create template
            template = self._jinja_env.from_string(template_content)
            
            # Render with context
            rendered = template.render(**(context or {}))
            
            return rendered.strip()
            
        except Exception as e:
            raise ValueError(f"Error processing Jinja template {template_path}: {e}")
    
    def validate_prompt_type(self, prompt_type: str) -> bool:
        """Validate if a prompt type is supported."""
        return prompt_type in ["string", "file", "jinja"]
    
    def get_default_context(self, agent) -> Dict[str, Any]:
        """Get default context variables for template rendering."""
        context = {
            "agent_name": getattr(agent, 'name', 'unnamed_agent'),
            "model": getattr(agent, 'model', 'unknown'),
            "tools": [],
            "sub_agents": [],
            "timestamp": self._get_timestamp()
        }
        
        # Add tool information
        if hasattr(agent, 'tools') and agent.tools:
            context["tools"] = [
                {
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "content": getattr(tool, 'content', None),
                }
                for tool in agent.tools
            ]
        
        # Add sub-agent information
        if hasattr(agent, 'sub_agent_factories') and agent.sub_agent_factories:
            context["sub_agents"] = [
                {
                    "name": name,
                    "description": f"Specialized agent for {name}-related tasks"
                }
                for name in agent.sub_agent_factories.keys()
            ]
        
        return context
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_dynamic_prompt(
        self,
        base_template: str,
        agent,
        additional_context: Optional[Dict[str, Any]] = None,
        template_type: str = "string"
    ) -> str:
        """Create a dynamic prompt by combining base template with agent context."""
        context = self.get_default_context(agent)
        
        if additional_context:
            context.update(additional_context)
        
        try:
            if template_type == "jinja":
                # base_template is a path to a jinja template file
                return self._process_jinja_prompt(base_template, context)
            else:
                template = self._jinja_env.from_string(base_template)
                return template.render(**context)
        except Exception as e:
            # Return base template if rendering fails
            return base_template