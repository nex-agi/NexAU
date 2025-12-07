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

"""System prompt builder for agents."""

import logging
from pathlib import Path
from typing import Any, TypedDict

from jinja2 import Environment, FileSystemLoader

from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.prompt_handler import PromptHandler
from nexau.archs.tool import Tool

logger = logging.getLogger(__name__)


class ToolParameter(TypedDict):
    """Structured representation of a tool parameter for prompt docs."""

    name: str
    description: str
    type: str
    required: bool
    default: Any


class ToolInfo(TypedDict):
    """Structured representation of tool metadata for prompt docs."""

    name: str
    description: str
    template_override: str | None
    parameters: list[ToolParameter]
    as_skill: bool
    skill_description: str | None


def _get_python_type_from_json_schema(json_type: str) -> str:
    """Convert JSON Schema type to Python type string.

    Args:
        json_type: JSON Schema type (string, integer, number, boolean, array, object)

    Returns:
        Python type string (str, int, float, bool, list, dict)
    """
    type_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return type_mapping.get(json_type, "str")


class PromptBuilder:
    """Handles the creation and formatting of system prompts."""

    def __init__(self):
        """Initialize the prompt builder."""
        current_dir = Path(__file__).parent
        self.prompts_dir = current_dir / "prompts"
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.prompt_handler = PromptHandler()

    def build_system_prompt(
        self,
        agent_config: AgentConfig,
        tools: list[Tool] | None = None,
        sub_agent_factories: dict[str, Any] | None = None,
        runtime_context: dict[str, Any] | None = None,
        include_tool_instructions: bool = True,
    ) -> str:
        """Build the complete system prompt including tool and sub-agent docs.

        Args:
            agent_config: Agent configuration
            tools: List of available tools
            sub_agent_factories: Dictionary of sub-agent factories
            runtime_context: Additional runtime context

        Returns:
            Complete system prompt string
        """
        try:
            # Get base system prompt
            base_prompt = self._get_base_system_prompt(agent_config, runtime_context or {})

            if include_tool_instructions:
                # Build capabilities documentation
                capabilities_docs = self._build_capabilities_docs(
                    tools or agent_config.tools,
                    sub_agent_factories or agent_config.sub_agent_factories,
                    runtime_context,
                )

                # Add tool execution instructions
                execution_instructions = ""

                execution_instructions = self._get_tool_execution_instructions() or ""

                return f"{base_prompt}{capabilities_docs}{execution_instructions}"
            else:
                return base_prompt

        except Exception as e:
            logger.error(f"❌ Error building system prompt: {e}")
            raise ValueError("Error building system prompt") from e

    def _get_base_system_prompt(
        self,
        agent_config: AgentConfig,
        runtime_context: dict[str, Any],
    ) -> str:
        """Get the base system prompt from configuration."""
        if not agent_config.system_prompt:
            agent_name = agent_config.name or "agent"
            return self._get_default_system_prompt(agent_name)

        try:
            # Build context for template rendering
            context = self._build_template_context(
                runtime_context,
            )

            # Process the system prompt
            return self.prompt_handler.create_dynamic_prompt(
                agent_config.system_prompt,
                agent_config,  # Pass agent config as agent parameter
                additional_context=context,
                template_type=agent_config.system_prompt_type,
            )
        except Exception as e:
            logger.error(f"❌ Error processing system prompt: {e}")
            raise ValueError("Error processing system prompt") from e

    def _get_default_system_prompt(self, agent_name: str) -> str:
        """Get default system prompt for the agent."""
        try:
            template = self._load_prompt_template("default_system_prompt")
            if template:
                context = {"agent_name": agent_name}
                jinja_template = self.jinja_env.from_string(template)
                return jinja_template.render(**context)
        except Exception as e:
            logger.warning(f"⚠️ Error loading default system prompt: {e}")
            raise ValueError("Error loading default system prompt") from e
        return "You are a helpful assistant."

    def _build_capabilities_docs(
        self,
        tools: list[Tool],
        sub_agent_factories: dict[str, Any],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        """Build documentation for tools and sub-agents."""
        docs: list[str] = []

        # Add tool documentation
        if tools:
            tool_docs = self._build_tools_documentation(tools, runtime_context)
            docs.append(tool_docs)

        # Add sub-agent documentation
        if sub_agent_factories:
            subagent_docs = self._build_subagents_documentation(
                sub_agent_factories,
            )
            docs.append(subagent_docs)

        return "\n".join(docs)

    def _build_tools_documentation(
        self,
        tools: list[Tool],
        runtime_context: dict[str, Any] | None = None,
    ) -> str:
        """Build tools documentation section."""
        try:
            template = self._load_prompt_template("tools_template")
            if not template:
                raise ValueError("Tools template not found")

            # Prepare tool context with enhanced parameter information
            tools_context: list[ToolInfo] = []
            for tool in tools:
                tool_info: ToolInfo = {
                    "name": tool.name,
                    "description": tool.description,
                    "template_override": tool.template_override if tool.template_override else None,
                    "parameters": self._extract_tool_parameters(tool),
                    "as_skill": tool.as_skill,
                    "skill_description": tool.skill_description,
                }
                tools_context.append(tool_info)

            context: dict[str, Any] = {"tools": tools_context}
            if runtime_context:
                context.update(runtime_context)

            jinja_template = self.jinja_env.from_string(template)
            return jinja_template.render(**context)

        except Exception as e:
            logger.warning(f"⚠️ Error building tools documentation: {e}")
            raise ValueError("Error building tools documentation") from e

    def _build_subagents_documentation(
        self,
        sub_agent_factories: dict[str, Any],
    ) -> str:
        """Build sub-agents documentation section."""
        try:
            template = self._load_prompt_template("sub_agents_template")
            if not template:
                raise ValueError("Sub-agents template not found")

            # Prepare sub-agents context
            sub_agents_context = [
                {
                    "name": name,
                    "description": f"Specialized agent for {name}-related tasks",
                }
                for name in sub_agent_factories.keys()
            ]

            context = {"sub_agents": sub_agents_context}
            jinja_template = self.jinja_env.from_string(template)
            return jinja_template.render(**context)

        except Exception as e:
            logger.warning(f"⚠️ Error building sub-agents documentation: {e}")
            raise ValueError("Error building sub-agents documentation") from e

    def _extract_tool_parameters(self, tool: Tool) -> list[ToolParameter]:
        """Extract parameter information from tool schema."""
        if not hasattr(tool, "input_schema"):
            return []

        schema = tool.input_schema
        properties = schema.get("properties", {})
        required_params = schema.get("required", [])

        parameters: list[ToolParameter] = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "")
            default_value = param_info.get("default")
            is_required = param_name in required_params

            python_type = _get_python_type_from_json_schema(param_type)

            parameters.append(
                {
                    "name": param_name,
                    "description": param_desc,
                    "type": python_type,
                    "required": is_required,
                    "default": default_value,
                },
            )

        return parameters

    def _get_tool_execution_instructions(self) -> str | None:
        """Get tool execution instructions."""
        try:
            template = self._load_prompt_template(
                "tool_execution_instructions",
            )
            if template:
                return template
        except Exception as e:
            raise ValueError("Error loading tool execution instructions") from e
        return None

    def _build_template_context(
        self,
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Build template context for prompt rendering."""
        context: dict[str, Any] = {}

        if runtime_context:
            context.update(runtime_context)

        return context

    def load_prompt_template(self, prompt_name: str) -> str:
        """Public wrapper to retrieve prompt templates."""
        return self._load_prompt_template(prompt_name)

    def _load_prompt_template(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        try:
            template_file = self.prompts_dir / f"{prompt_name}.j2"
            if template_file.exists():
                with open(template_file, encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            logger.warning(
                f"⚠️ Error loading prompt template {prompt_name}: {e}",
            )

        return ""
