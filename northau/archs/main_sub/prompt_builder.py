"""System prompt builder for agents."""
import logging
from pathlib import Path
from typing import Any
from typing import Optional

from jinja2 import Environment
from jinja2 import FileSystemLoader

from .config import AgentConfig
from .prompt_handler import PromptHandler

logger = logging.getLogger(__name__)


def _get_python_type_from_json_schema(json_type: str) -> str:
    """Convert JSON Schema type to Python type string.

    Args:
        json_type: JSON Schema type (string, integer, number, boolean, array, object)

    Returns:
        Python type string (str, int, float, bool, list, dict)
    """
    type_mapping = {
        'string': 'str',
        'integer': 'int',
        'number': 'float',
        'boolean': 'bool',
        'array': 'list',
        'object': 'dict',
    }
    return type_mapping.get(json_type, 'str')


class PromptBuilder:
    """Handles the creation and formatting of system prompts."""

    def __init__(self):
        """Initialize the prompt builder."""
        current_dir = Path(__file__).parent
        self.prompts_dir = current_dir / 'prompts'
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.prompts_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.prompt_handler = PromptHandler()

    def build_system_prompt(
        self,
        agent_config: AgentConfig,
        tools: list = None,
        sub_agent_factories: dict[str, Any] = None,
        runtime_context: Optional[dict] = None,
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
            base_prompt = self._get_base_system_prompt(
                agent_config,
                runtime_context,
            )

            # Build capabilities documentation
            capabilities_docs = self._build_capabilities_docs(
                tools or agent_config.tools,
                sub_agent_factories or agent_config.sub_agent_factories,
                runtime_context,
            )

            # Add tool execution instructions
            execution_instructions = self._get_tool_execution_instructions()

            return f"{base_prompt}{capabilities_docs}{execution_instructions}"

        except Exception as e:
            logger.error(f"❌ Error building system prompt: {e}")
            raise ValueError("Error building system prompt") from e

    def _get_base_system_prompt(
        self,
        agent_config: AgentConfig,
        runtime_context: Optional[dict] = None,
    ) -> str:
        """Get the base system prompt from configuration."""
        if not agent_config.system_prompt:
            return self._get_default_system_prompt(agent_config.name)

        try:
            # Build context for template rendering
            context = self._build_template_context(
                agent_config,
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


    def _build_capabilities_docs(
        self,
        tools: list,
        sub_agent_factories: dict[str, Any],
        runtime_context: Optional[dict] = None,
    ) -> str:
        """Build documentation for tools and sub-agents."""
        docs = []

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

        return '\n'.join(docs)

    def _build_tools_documentation(
        self,
        tools: list,
        runtime_context: Optional[dict] = None,
    ) -> str:
        """Build tools documentation section."""
        try:
            template = self._load_prompt_template('tools_template')
            if not template:
                raise ValueError("Tools template not found")

            # Prepare tool context with enhanced parameter information
            tools_context = []
            for tool in tools:
                tool_info = {
                    'name': tool.name,
                    'description': getattr(
                        tool, 'description', 'No description available',
                    ),
                    'template_override': getattr(tool, 'template_override', None),
                    'parameters': self._extract_tool_parameters(tool),
                }
                tools_context.append(tool_info)

            context = {
                'tools': tools_context,
                **(runtime_context or {}),
            }

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
            template = self._load_prompt_template('sub_agents_template')
            if not template:
                raise ValueError("Sub-agents template not found")

            # Prepare sub-agents context
            sub_agents_context = [
                {
                    'name': name,
                    'description': f"Specialized agent for {name}-related tasks",
                }
                for name in sub_agent_factories.keys()
            ]

            context = {'sub_agents': sub_agents_context}
            jinja_template = self.jinja_env.from_string(template)
            return jinja_template.render(**context)

        except Exception as e:
            logger.warning(f"⚠️ Error building sub-agents documentation: {e}")
            raise ValueError("Error building sub-agents documentation") from e

    def _extract_tool_parameters(self, tool) -> list:
        """Extract parameter information from tool schema."""
        if not hasattr(tool, 'input_schema'):
            return []

        schema = tool.input_schema
        properties = schema.get('properties', {})
        required_params = schema.get('required', [])

        parameters = []
        for param_name, param_info in properties.items():
            param_type = param_info.get('type', 'string')
            param_desc = param_info.get('description', '')
            default_value = param_info.get('default')
            is_required = param_name in required_params

            python_type = _get_python_type_from_json_schema(param_type)

            parameters.append(
                {
                    'name': param_name,
                    'description': param_desc,
                    'type': python_type,
                    'required': is_required,
                    'default': default_value,
                },
            )

        return parameters

    def _get_tool_execution_instructions(self) -> str:
        """Get tool execution instructions."""
        try:
            template = self._load_prompt_template(
                'tool_execution_instructions',
            )
            if template:
                return template
        except Exception as e:
            raise ValueError("Error loading tool execution instructions") from e

    def _build_template_context(
        self,
        agent_config: AgentConfig,
        runtime_context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Build template context for prompt rendering."""
        context = {}

        if runtime_context:
            context.update(runtime_context)

        return context

    def _load_prompt_template(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        try:
            template_file = self.prompts_dir / f"{prompt_name}.j2"
            if template_file.exists():
                with open(template_file, encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.warning(
                f"⚠️ Error loading prompt template {prompt_name}: {e}",
            )

        return ''
