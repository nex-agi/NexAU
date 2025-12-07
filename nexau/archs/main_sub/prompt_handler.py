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

"""System prompt handling for different prompt types."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from jinja2 import BaseLoader, Environment, Template

if TYPE_CHECKING:
    from nexau.archs.main_sub.agent import Agent
    from nexau.archs.main_sub.config import AgentConfig

AgentContextSource = Union["Agent", "AgentConfig"]


class PromptHandler:
    """Handles different types of system prompts: string, file, and Jinja templates."""

    def __init__(self):
        """Initialize the prompt handler."""
        self._setup_jinja()

    def _setup_jinja(self):
        """Setup Jinja2 environment."""

        self._jinja_env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def process_prompt(
        self,
        prompt: str,
        prompt_type: str = "string",
        context: dict[str, Any] | None = None,
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
        if not self.validate_prompt_type(prompt_type):
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        if prompt_type == "string":
            return self._process_string_prompt(prompt, context)
        elif prompt_type == "file":
            return self._process_file_prompt(prompt, context)
        elif prompt_type == "jinja":
            return self._process_jinja_prompt(prompt, context)
        return ""

    def _process_string_prompt(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Process a string prompt (may contain simple variable substitution)."""
        if not prompt:
            return ""

        # Simple variable substitution using {variable} syntax
        if context:
            try:
                return prompt.format(**context)
            except KeyError:
                # If some variables are missing, leave them as-is
                return prompt

        return prompt

    def _process_file_prompt(
        self,
        file_path: str,
        context: dict[str, Any] | None = None,
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
            with open(path, encoding="utf-8") as f:
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
        context: dict[str, Any] | None = None,
    ) -> str:
        """Process a Jinja template prompt."""
        path = Path(template_path)

        if not path.exists():
            # Try relative to current working directory
            cwd_path = Path.cwd() / template_path
            if cwd_path.exists():
                path = cwd_path
            else:
                raise FileNotFoundError(
                    f"Jinja template not found: {template_path}",
                )

        try:
            with open(path, encoding="utf-8") as f:
                template_content = f.read()

            # Create template
            template: Template = self._jinja_env.from_string(template_content)

            # Render with context
            rendered = template.render(**(context or {}))

            return rendered.strip()

        except Exception as e:
            raise ValueError(
                f"Error processing Jinja template {template_path}: {e}",
            )

    def validate_prompt_type(self, prompt_type: str) -> bool:
        """Validate if a prompt type is supported."""
        return prompt_type in ["string", "file", "jinja"]

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def get_default_context(self, agent: AgentContextSource) -> dict[str, Any]:
        """Get default context for template rendering."""
        context: dict[str, Any] = {
            "agent_name": getattr(agent, "name", "Unknown Agent"),
            "timestamp": self._get_timestamp(),
        }

        config_source = getattr(agent, "config", None) or agent
        context.update(
            {
                "agent_id": getattr(config_source, "agent_id", None),
                "system_prompt_type": getattr(
                    config_source,
                    "system_prompt_type",
                    "string",
                ),
            },
        )

        return context

    def create_dynamic_prompt(
        self,
        base_template: str,
        agent: AgentContextSource,
        additional_context: dict[str, Any] | None = None,
        template_type: str = "string",
    ) -> str:
        """Create a dynamic prompt by combining base template with agent context."""

        if not self.validate_prompt_type(template_type):
            raise ValueError(f"Invalid template type: {template_type}")

        context: dict[str, Any] = self.get_default_context(agent)

        if additional_context:
            context.update(additional_context)

        try:
            if template_type in ["jinja", "file"]:
                # base_template is a path to a jinja template file
                return self._process_jinja_prompt(base_template, context)
            elif template_type == "string":
                template = self._jinja_env.from_string(base_template)
                return template.render(**context)
        except Exception as e:
            # Return base template if rendering fails
            raise ValueError(f"Error creating dynamic prompt: {e}")
        return ""
