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

"""Unit tests for PromptBuilder."""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
from jinja2 import Environment, FileSystemLoader

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.prompt_builder import PromptBuilder, _get_python_type_from_json_schema


class TestGetPythonTypeFromJsonSchema:
    """Tests for _get_python_type_from_json_schema helper function."""

    def test_string_type(self):
        """Test conversion of string type."""
        assert _get_python_type_from_json_schema("string") == "str"

    def test_integer_type(self):
        """Test conversion of integer type."""
        assert _get_python_type_from_json_schema("integer") == "int"

    def test_number_type(self):
        """Test conversion of number type."""
        assert _get_python_type_from_json_schema("number") == "float"

    def test_boolean_type(self):
        """Test conversion of boolean type."""
        assert _get_python_type_from_json_schema("boolean") == "bool"

    def test_array_type(self):
        """Test conversion of array type."""
        assert _get_python_type_from_json_schema("array") == "list"

    def test_object_type(self):
        """Test conversion of object type."""
        assert _get_python_type_from_json_schema("object") == "dict"

    def test_unknown_type(self):
        """Test conversion of unknown type defaults to str."""
        assert _get_python_type_from_json_schema("unknown_type") == "str"

    def test_none_type(self):
        """Test conversion of None type defaults to str."""
        assert _get_python_type_from_json_schema(None) == "str"


class TestPromptBuilderInit:
    """Tests for PromptBuilder initialization."""

    def test_initialization(self):
        """Test successful initialization."""
        builder = PromptBuilder()

        assert builder.prompts_dir is not None
        assert isinstance(builder.prompts_dir, Path)
        assert builder.prompts_dir.name == "prompts"
        assert builder.jinja_env is not None
        assert isinstance(builder.jinja_env, Environment)
        assert builder.prompt_handler is not None

    def test_jinja_env_configuration(self):
        """Test jinja environment is properly configured."""
        builder = PromptBuilder()

        # Check trim_blocks and lstrip_blocks are set
        assert builder.jinja_env.trim_blocks is True
        assert builder.jinja_env.lstrip_blocks is True
        assert isinstance(builder.jinja_env.loader, FileSystemLoader)


class TestBuildSystemPrompt:
    """Tests for build_system_prompt method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock agent config."""
        return AgentConfig(
            name="test_agent",
            llm_config=LLMConfig(
                model="anthropic/claude-3-5-sonnet-20241022",
                provider="anthropic",
            ),
            system_prompt="Test system prompt",
        )

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.template_override = None
        tool.input_schema = {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter",
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter",
                },
            },
            "required": ["param1"],
        }
        return tool

    def test_build_system_prompt_basic(self, mock_config):
        """Test building a basic system prompt."""
        builder = PromptBuilder()

        with patch.object(builder, "_get_base_system_prompt", return_value="Base prompt\n"):
            with patch.object(builder, "_build_capabilities_docs", return_value="Capabilities\n"):
                with patch.object(builder, "_get_tool_execution_instructions", return_value="Instructions\n"):
                    result = builder.build_system_prompt(mock_config)

        assert result == "Base prompt\nCapabilities\nInstructions\n"

    def test_build_system_prompt_with_tools(self, mock_config, mock_tool):
        """Test building system prompt with tools."""
        builder = PromptBuilder()

        with patch.object(builder, "_get_base_system_prompt", return_value="Base\n"):
            with patch.object(builder, "_build_capabilities_docs", return_value="Tools\n"):
                with patch.object(builder, "_get_tool_execution_instructions", return_value="Instructions\n"):
                    result = builder.build_system_prompt(mock_config, tools=[mock_tool])

        assert "Base\n" in result
        assert "Tools\n" in result
        assert "Instructions\n" in result

    def test_build_system_prompt_with_sub_agents(self, mock_config):
        """Test building system prompt with sub-agents."""
        builder = PromptBuilder()
        sub_agent_factories = {"researcher": Mock()}

        with patch.object(builder, "_get_base_system_prompt", return_value="Base\n"):
            with patch.object(builder, "_build_capabilities_docs", return_value="SubAgents\n"):
                with patch.object(builder, "_get_tool_execution_instructions", return_value="Instructions\n"):
                    result = builder.build_system_prompt(
                        mock_config,
                        sub_agent_factories=sub_agent_factories,
                    )

        assert "SubAgents\n" in result

    def test_build_system_prompt_with_runtime_context(self, mock_config):
        """Test building system prompt with runtime context."""
        builder = PromptBuilder()
        runtime_context = {"custom_key": "custom_value"}

        with patch.object(builder, "_get_base_system_prompt", return_value="Base\n") as mock_base:
            with patch.object(builder, "_build_capabilities_docs", return_value="Caps\n") as mock_caps:
                with patch.object(builder, "_get_tool_execution_instructions", return_value="Inst\n"):
                    builder.build_system_prompt(
                        mock_config,
                        runtime_context=runtime_context,
                    )

        # Verify runtime context was passed through
        mock_base.assert_called_once()
        mock_caps.assert_called_once()

    def test_build_system_prompt_without_tool_instructions(self, mock_config):
        """Ensure tool instructions can be skipped when requested."""
        builder = PromptBuilder()

        with patch.object(builder, "_get_base_system_prompt", return_value="Base\n"):
            with patch.object(builder, "_build_capabilities_docs", return_value="Caps\n"):
                with patch.object(builder, "_get_tool_execution_instructions", return_value="ShouldNotAppear\n") as mock_exec:
                    result = builder.build_system_prompt(
                        mock_config,
                        include_tool_instructions=False,
                    )

        mock_exec.assert_not_called()
        assert "ShouldNotAppear" not in result

    def test_build_system_prompt_error_handling(self, mock_config):
        """Test error handling in build_system_prompt."""
        builder = PromptBuilder()

        with patch.object(builder, "_get_base_system_prompt", side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Error building system prompt"):
                builder.build_system_prompt(mock_config)


class TestGetBaseSystemPrompt:
    """Tests for _get_base_system_prompt method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock agent config."""
        return AgentConfig(
            name="test_agent",
            llm_config=LLMConfig(
                model="anthropic/claude-3-5-sonnet-20241022",
                provider="anthropic",
            ),
        )

    def test_get_base_system_prompt_no_custom_prompt(self, mock_config):
        """Test getting base system prompt when no custom prompt is set."""
        builder = PromptBuilder()

        with patch.object(builder, "_get_default_system_prompt", return_value="Default prompt"):
            result = builder._get_base_system_prompt(mock_config)

        assert result == "Default prompt"

    def test_get_base_system_prompt_with_custom_prompt(self, mock_config):
        """Test getting base system prompt with custom prompt."""
        mock_config.system_prompt = "Custom prompt"
        builder = PromptBuilder()

        with patch.object(builder.prompt_handler, "create_dynamic_prompt", return_value="Processed prompt"):
            result = builder._get_base_system_prompt(mock_config)

        assert result == "Processed prompt"

    def test_get_base_system_prompt_with_runtime_context(self, mock_config):
        """Test getting base system prompt with runtime context."""
        mock_config.system_prompt = "Custom prompt"
        builder = PromptBuilder()
        runtime_context = {"key": "value"}

        with patch.object(builder.prompt_handler, "create_dynamic_prompt", return_value="Result") as mock_create:
            with patch.object(builder, "_build_template_context", return_value=runtime_context):
                builder._get_base_system_prompt(mock_config, runtime_context)

        # Verify create_dynamic_prompt was called with context
        mock_create.assert_called_once()

    def test_get_base_system_prompt_error(self, mock_config):
        """Test error handling in _get_base_system_prompt."""
        mock_config.system_prompt = "Custom prompt"
        builder = PromptBuilder()

        with patch.object(builder.prompt_handler, "create_dynamic_prompt", side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Error processing system prompt"):
                builder._get_base_system_prompt(mock_config)


class TestGetDefaultSystemPrompt:
    """Tests for _get_default_system_prompt method."""

    def test_get_default_system_prompt_success(self):
        """Test successfully loading default system prompt."""
        builder = PromptBuilder()

        template_content = "You are an AI agent named '{{ agent_name }}'."
        with patch.object(builder, "_load_prompt_template", return_value=template_content):
            result = builder._get_default_system_prompt("TestAgent")

        assert "TestAgent" in result

    def test_get_default_system_prompt_template_not_found(self):
        """Test behavior when default template is not found returns None."""
        builder = PromptBuilder()

        # When template is empty or None, the method returns None implicitly
        with patch.object(builder, "_load_prompt_template", return_value=None):
            result = builder._get_default_system_prompt("TestAgent")

        assert result is None

    def test_get_default_system_prompt_rendering_error(self):
        """Test error during template rendering."""
        builder = PromptBuilder()

        # Invalid Jinja template
        with patch.object(builder, "_load_prompt_template", return_value="{{ invalid syntax"):
            with pytest.raises(ValueError, match="Error loading default system prompt"):
                builder._get_default_system_prompt("TestAgent")


class TestBuildCapabilitiesDocs:
    """Tests for _build_capabilities_docs method."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool."""
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        return tool

    def test_build_capabilities_docs_with_tools_only(self, mock_tool):
        """Test building capabilities docs with only tools."""
        builder = PromptBuilder()

        with patch.object(builder, "_build_tools_documentation", return_value="Tools doc"):
            result = builder._build_capabilities_docs([mock_tool], None, None)

        assert "Tools doc" in result

    def test_build_capabilities_docs_with_subagents_only(self):
        """Test building capabilities docs with only sub-agents."""
        builder = PromptBuilder()
        sub_agent_factories = {"researcher": Mock()}

        with patch.object(builder, "_build_subagents_documentation", return_value="SubAgents doc"):
            result = builder._build_capabilities_docs([], sub_agent_factories, None)

        assert "SubAgents doc" in result

    def test_build_capabilities_docs_with_both(self, mock_tool):
        """Test building capabilities docs with both tools and sub-agents."""
        builder = PromptBuilder()
        sub_agent_factories = {"researcher": Mock()}

        with patch.object(builder, "_build_tools_documentation", return_value="Tools doc"):
            with patch.object(builder, "_build_subagents_documentation", return_value="SubAgents doc"):
                result = builder._build_capabilities_docs([mock_tool], sub_agent_factories, None)

        assert "Tools doc" in result
        assert "SubAgents doc" in result

    def test_build_capabilities_docs_with_none(self):
        """Test building capabilities docs with no tools or sub-agents."""
        builder = PromptBuilder()

        result = builder._build_capabilities_docs([], None, None)

        assert result == ""


class TestBuildToolsDocumentation:
    """Tests for _build_tools_documentation method."""

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool with schema."""
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.template_override = None
        tool.input_schema = {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter",
                },
            },
            "required": ["param1"],
        }
        return tool

    def test_build_tools_documentation_success(self, mock_tool):
        """Test successful tools documentation building."""
        builder = PromptBuilder()

        template_content = "{% for tool in tools %}{{ tool.name }}{% endfor %}"
        with patch.object(builder, "_load_prompt_template", return_value=template_content):
            result = builder._build_tools_documentation([mock_tool], None)

        assert "test_tool" in result

    def test_build_tools_documentation_with_template_override(self):
        """Test tools documentation with template override."""
        builder = PromptBuilder()
        tool = Mock()
        tool.name = "test_tool"
        tool.description = "Test"
        tool.template_override = "Custom template override"
        tool.input_schema = {"properties": {}, "required": []}

        template_content = "{% for tool in tools %}{{ tool.template_override }}{% endfor %}"
        with patch.object(builder, "_load_prompt_template", return_value=template_content):
            result = builder._build_tools_documentation([tool], None)

        assert "Custom template override" in result

    def test_build_tools_documentation_with_runtime_context(self, mock_tool):
        """Test tools documentation with runtime context."""
        builder = PromptBuilder()
        runtime_context = {"extra_info": "value"}

        template_content = "{{ extra_info }}"
        with patch.object(builder, "_load_prompt_template", return_value=template_content):
            result = builder._build_tools_documentation([mock_tool], runtime_context)

        assert "value" in result

    def test_build_tools_documentation_template_not_found(self, mock_tool):
        """Test error when tools template not found."""
        builder = PromptBuilder()

        with patch.object(builder, "_load_prompt_template", return_value=""):
            with pytest.raises(ValueError, match="Error building tools documentation"):
                builder._build_tools_documentation([mock_tool], None)

    def test_build_tools_documentation_error(self, mock_tool):
        """Test error handling in tools documentation."""
        builder = PromptBuilder()

        with patch.object(builder, "_load_prompt_template", side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Error building tools documentation"):
                builder._build_tools_documentation([mock_tool], None)


class TestBuildSubagentsDocumentation:
    """Tests for _build_subagents_documentation method."""

    def test_build_subagents_documentation_success(self):
        """Test successful sub-agents documentation building."""
        builder = PromptBuilder()
        sub_agent_factories = {
            "researcher": Mock(),
            "writer": Mock(),
        }

        template_content = "{% for sub_agent in sub_agents %}{{ sub_agent.name }}{% endfor %}"
        with patch.object(builder, "_load_prompt_template", return_value=template_content):
            result = builder._build_subagents_documentation(sub_agent_factories)

        assert "researcher" in result
        assert "writer" in result

    def test_build_subagents_documentation_template_not_found(self):
        """Test error when sub-agents template not found."""
        builder = PromptBuilder()
        sub_agent_factories = {"researcher": Mock()}

        with patch.object(builder, "_load_prompt_template", return_value=""):
            with pytest.raises(ValueError, match="Error building sub-agents documentation"):
                builder._build_subagents_documentation(sub_agent_factories)

    def test_build_subagents_documentation_error(self):
        """Test error handling in sub-agents documentation."""
        builder = PromptBuilder()
        sub_agent_factories = {"researcher": Mock()}

        with patch.object(builder, "_load_prompt_template", side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Error building sub-agents documentation"):
                builder._build_subagents_documentation(sub_agent_factories)

    def test_build_subagents_documentation_empty(self):
        """Test sub-agents documentation with empty factories."""
        builder = PromptBuilder()
        sub_agent_factories = {}

        template_content = "{% for sub_agent in sub_agents %}{{ sub_agent.name }}{% endfor %}"
        with patch.object(builder, "_load_prompt_template", return_value=template_content):
            result = builder._build_subagents_documentation(sub_agent_factories)

        # Should be empty since no sub-agents
        assert result.strip() == ""


class TestExtractToolParameters:
    """Tests for _extract_tool_parameters method."""

    def test_extract_tool_parameters_with_schema(self):
        """Test extracting parameters from tool with schema."""
        builder = PromptBuilder()
        tool = Mock()
        tool.input_schema = {
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "First parameter",
                },
                "param2": {
                    "type": "integer",
                    "description": "Second parameter",
                    "default": 10,
                },
            },
            "required": ["param1"],
        }

        result = builder._extract_tool_parameters(tool)

        assert len(result) == 2
        assert result[0]["name"] == "param1"
        assert result[0]["type"] == "str"
        assert result[0]["required"] is True
        assert result[1]["name"] == "param2"
        assert result[1]["type"] == "int"
        assert result[1]["required"] is False
        assert result[1]["default"] == 10

    def test_extract_tool_parameters_all_types(self):
        """Test extracting parameters with all JSON types."""
        builder = PromptBuilder()
        tool = Mock()
        tool.input_schema = {
            "properties": {
                "str_param": {"type": "string"},
                "int_param": {"type": "integer"},
                "float_param": {"type": "number"},
                "bool_param": {"type": "boolean"},
                "list_param": {"type": "array"},
                "dict_param": {"type": "object"},
            },
            "required": [],
        }

        result = builder._extract_tool_parameters(tool)

        assert len(result) == 6
        types_map = {param["name"]: param["type"] for param in result}
        assert types_map["str_param"] == "str"
        assert types_map["int_param"] == "int"
        assert types_map["float_param"] == "float"
        assert types_map["bool_param"] == "bool"
        assert types_map["list_param"] == "list"
        assert types_map["dict_param"] == "dict"

    def test_extract_tool_parameters_no_schema(self):
        """Test extracting parameters from tool without schema."""
        builder = PromptBuilder()
        tool = Mock(spec=["name", "description"])

        result = builder._extract_tool_parameters(tool)

        assert result == []

    def test_extract_tool_parameters_empty_properties(self):
        """Test extracting parameters with empty properties."""
        builder = PromptBuilder()
        tool = Mock()
        tool.input_schema = {
            "properties": {},
            "required": [],
        }

        result = builder._extract_tool_parameters(tool)

        assert result == []

    def test_extract_tool_parameters_missing_type(self):
        """Test extracting parameters with missing type defaults to string."""
        builder = PromptBuilder()
        tool = Mock()
        tool.input_schema = {
            "properties": {
                "param1": {
                    "description": "Parameter without type",
                },
            },
            "required": [],
        }

        result = builder._extract_tool_parameters(tool)

        assert len(result) == 1
        assert result[0]["type"] == "str"  # Default type


class TestGetToolExecutionInstructions:
    """Tests for _get_tool_execution_instructions method."""

    def test_get_tool_execution_instructions_success(self):
        """Test successfully loading tool execution instructions."""
        builder = PromptBuilder()

        expected_instructions = "Tool execution instructions"
        with patch.object(builder, "_load_prompt_template", return_value=expected_instructions):
            result = builder._get_tool_execution_instructions()

        assert result == expected_instructions

    def test_get_tool_execution_instructions_empty(self):
        """Test loading empty tool execution instructions returns None."""
        builder = PromptBuilder()

        with patch.object(builder, "_load_prompt_template", return_value=""):
            result = builder._get_tool_execution_instructions()

        # When template is empty/falsy, the method returns None implicitly
        assert result is None

    def test_get_tool_execution_instructions_error(self):
        """Test error handling when loading instructions."""
        builder = PromptBuilder()

        with patch.object(builder, "_load_prompt_template", side_effect=Exception("Test error")):
            with pytest.raises(ValueError, match="Error loading tool execution instructions"):
                builder._get_tool_execution_instructions()


class TestBuildTemplateContext:
    """Tests for _build_template_context method."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock agent config."""
        return AgentConfig(
            name="test_agent",
            llm_config=LLMConfig(
                model="anthropic/claude-3-5-sonnet-20241022",
                provider="anthropic",
            ),
        )

    def test_build_template_context_no_runtime(self, mock_config):
        """Test building template context without runtime context."""
        builder = PromptBuilder()

        result = builder._build_template_context(mock_config, None)

        assert result == {}

    def test_build_template_context_with_runtime(self, mock_config):
        """Test building template context with runtime context."""
        builder = PromptBuilder()
        runtime_context = {"key1": "value1", "key2": "value2"}

        result = builder._build_template_context(mock_config, runtime_context)

        assert result == runtime_context
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_build_template_context_empty_runtime(self, mock_config):
        """Test building template context with empty runtime context."""
        builder = PromptBuilder()

        result = builder._build_template_context(mock_config, {})

        assert result == {}


class TestLoadPromptTemplate:
    """Tests for _load_prompt_template method."""

    def test_load_prompt_template_success(self):
        """Test successfully loading a prompt template."""
        builder = PromptBuilder()

        template_content = "Test template content"
        with patch("builtins.open", mock_open(read_data=template_content)):
            with patch.object(Path, "exists", return_value=True):
                result = builder._load_prompt_template("test_template")

        assert result == template_content

    def test_load_prompt_template_not_found(self):
        """Test loading non-existent template returns empty string."""
        builder = PromptBuilder()

        with patch.object(Path, "exists", return_value=False):
            result = builder._load_prompt_template("nonexistent")

        assert result == ""

    def test_load_prompt_template_read_error(self):
        """Test error handling when reading template fails."""
        builder = PromptBuilder()

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", side_effect=OSError("Read error")):
                result = builder._load_prompt_template("test_template")

        assert result == ""

    def test_load_prompt_template_encoding(self):
        """Test loading template with UTF-8 encoding."""
        builder = PromptBuilder()

        template_content = "Template with émojis 🎉"
        mock_file = mock_open(read_data=template_content)

        with patch("builtins.open", mock_file):
            with patch.object(Path, "exists", return_value=True):
                builder._load_prompt_template("test_template")

        # Verify open was called with utf-8 encoding
        mock_file.assert_called_once()
        call_args = mock_file.call_args
        assert call_args[1].get("encoding") == "utf-8"


class TestPromptBuilderIntegration:
    """Integration tests for PromptBuilder."""

    @pytest.fixture
    def full_config(self):
        """Create a full agent config for integration testing."""
        return AgentConfig(
            name="integration_test_agent",
            llm_config=LLMConfig(
                model="anthropic/claude-3-5-sonnet-20241022",
                provider="anthropic",
            ),
            system_prompt="You are a helpful assistant.",
        )

    @pytest.fixture
    def full_tool(self):
        """Create a full tool for integration testing."""
        tool = Mock()
        tool.name = "calculator"
        tool.description = "Performs calculations"
        tool.template_override = None
        tool.input_schema = {
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                },
                "precision": {
                    "type": "integer",
                    "description": "Number of decimal places",
                    "default": 2,
                },
            },
            "required": ["expression"],
        }
        return tool

    def test_full_system_prompt_generation(self, full_config, full_tool):
        """Test generating a complete system prompt with all components."""
        builder = PromptBuilder()
        sub_agent_factories = {"researcher": Mock()}

        # Mock all template loads
        templates = {
            "tools_template": "Tools: {% for tool in tools %}{{ tool.name }}{% endfor %}",
            "sub_agents_template": "SubAgents: {% for sa in sub_agents %}{{ sa.name }}{% endfor %}",
            "tool_execution_instructions": "Execute tools properly.",
        }

        def mock_load_template(name):
            return templates.get(name, "")

        with patch.object(builder, "_load_prompt_template", side_effect=mock_load_template):
            with patch.object(builder.prompt_handler, "create_dynamic_prompt", return_value="System Prompt\n"):
                result = builder.build_system_prompt(
                    full_config,
                    tools=[full_tool],
                    sub_agent_factories=sub_agent_factories,
                )

        assert "System Prompt" in result
        assert "calculator" in result
        assert "researcher" in result

    def test_extract_parameters_integration(self, full_tool):
        """Test parameter extraction in a realistic scenario."""
        builder = PromptBuilder()

        params = builder._extract_tool_parameters(full_tool)

        assert len(params) == 2

        # Check expression parameter
        expr_param = next(p for p in params if p["name"] == "expression")
        assert expr_param["type"] == "str"
        assert expr_param["required"] is True
        assert "Mathematical expression" in expr_param["description"]

        # Check precision parameter
        prec_param = next(p for p in params if p["name"] == "precision")
        assert prec_param["type"] == "int"
        assert prec_param["required"] is False
        assert prec_param["default"] == 2
