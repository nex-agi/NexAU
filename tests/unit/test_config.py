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

"""
Unit tests for configuration loading components.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml
from pydantic import ValidationError

from nexau.archs.config.config_loader import (
    AgentBuilder,
    ConfigError,
    apply_agent_name_overrides,
    load_agent_config,
    load_sub_agent_from_config,
    load_tool_from_config,
    validate_config_schema,
)
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.tool.tool import ConfigError as ToolConfigError
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.core import BaseTracer


class TestConfigLoader:
    """Test cases for configuration loading functionality."""

    def test_apply_agent_name_overrides(self):
        """Test applying overrides based on agent names."""
        config = {"name": "main_agent", "llm_config": {"model": "gpt-4"}, "tools": []}

        overrides = {"main_agent": {"llm_config": {"temperature": 0.5}, "max_iterations": 50}}

        result = apply_agent_name_overrides(config, overrides)

        assert result["llm_config"]["model"] == "gpt-4"
        assert result["llm_config"]["temperature"] == 0.5
        assert result["max_iterations"] == 50

    def test_load_yaml_with_vars(self, temp_dir):
        """Test YAML loading with variable substitution."""
        config_content = """
name: test_agent
system_prompt: "Agent in ${this_file_dir}"
llm_config:
  model: gpt-4o-mini
"""

        config_file = Path(temp_dir) / "test_config.yaml"
        config_file.write_text(config_content)

        from nexau.archs.config.config_loader import load_yaml_with_vars

        result = load_yaml_with_vars(str(config_file))

        assert result["name"] == "test_agent"
        assert temp_dir in result["system_prompt"]

    def test_load_yaml_with_yaml_variables(self, temp_dir):
        """YAML variables block should be resolved and removed from final config."""
        config_content = """
variables:
  model_name: gpt-4o
name: test_agent
system_prompt: "Agent in ${variables.model_name}"
llm_config:
  model: ${variables.model_name}
tools: []
"""

        config_file = Path(temp_dir) / "test_config.yaml"
        config_file.write_text(config_content)

        from nexau.archs.config.config_loader import load_yaml_with_vars

        result = load_yaml_with_vars(str(config_file))

        assert result["system_prompt"] == "Agent in gpt-4o"
        assert result["llm_config"]["model"] == "gpt-4o"
        assert "variables" not in result

    def test_variable_errors_for_missing_and_non_string_embeds(self, temp_dir):
        """Missing vars and non-string embedding should raise ConfigError."""
        from nexau.archs.config.config_loader import load_yaml_with_vars

        missing_content = """
variables:
  foo: bar
value: ${variables.not_defined}
"""

        missing_file = Path(temp_dir) / "missing.yaml"
        missing_file.write_text(missing_content)

        with pytest.raises(ConfigError, match="not_defined"):
            load_yaml_with_vars(str(missing_file))

        non_string_content = """
variables:
  mapping:
    key: value
value: "prefix-${variables.mapping}"
"""

        non_string_file = Path(temp_dir) / "non_string.yaml"
        non_string_file.write_text(non_string_content)

        with pytest.raises(ConfigError, match="non-scalar value"):
            load_yaml_with_vars(str(non_string_file))

    def test_load_yaml_with_invalid_variables_block(self, temp_dir):
        """Non-mapping variables block should raise an error."""
        config_content = """
variables: true
name: test_agent
llm_config:
  model: gpt-4o-mini
tools: []
"""

        config_file = Path(temp_dir) / "test_config.yaml"
        config_file.write_text(config_content)

        from nexau.archs.config.config_loader import load_yaml_with_vars

        with pytest.raises(ConfigError, match="must be a mapping"):
            load_yaml_with_vars(str(config_file))

    def test_validate_config_schema_valid(self):
        """Test schema validation with valid configuration."""
        valid_config = {
            "name": "test_agent",
            "llm_config": {"model": "gpt-4"},
            "tools": [{"name": "tool1", "yaml_path": "tool1.yaml"}, {"name": "tool2", "yaml_path": "tool2.yaml"}],
            "sub_agents": [{"name": "sub1", "config_path": "sub1.yaml"}],
        }

        assert validate_config_schema(valid_config)

    def test_validate_config_schema_invalid_tools(self):
        """Test schema validation with invalid tools configuration."""
        invalid_config = {
            "tools": "not_a_list"  # Should be a list
        }

        with pytest.raises(ConfigError, match="tools"):
            validate_config_schema(invalid_config)

    def test_validate_config_schema_invalid_sub_agents(self):
        """Test schema validation with invalid sub-agents configuration."""
        invalid_config = {
            "sub_agents": "not_a_list"  # Should be a list
        }

        with pytest.raises(ConfigError, match="sub_agents"):
            validate_config_schema(invalid_config)

    def test_validate_config_schema_unknown_field(self):
        """Ensure unknown keys are rejected so typos are caught early."""
        invalid_config = {
            "name": "test_agent",
            "llm_config": {"model": "gpt-4"},
            "tools": [],
            "unexpected": True,
        }

        with pytest.raises(ConfigError, match="unexpected"):
            validate_config_schema(invalid_config)


class TestAgentBuilder:
    """Test cases for AgentBuilder functionality."""

    def test_build_core_properties(self):
        """Test building core agent properties."""
        config = {
            "name": "test_agent",
            "max_context_tokens": 16000,
            "max_running_subagents": 3,
            "system_prompt": "Test prompt",
            "context": {"key": "value"},
            "stop_tools": ["stop_tool"],
            "max_iterations": 50,
        }

        builder = AgentBuilder(config, Path("."))
        result = builder.build_core_properties()

        assert result.agent_params["name"] == "test_agent"
        assert result.agent_params["max_context_tokens"] == 16000
        assert result.agent_params["max_running_subagents"] == 3
        assert result.agent_params["system_prompt"] == "Test prompt"
        assert result.agent_params["initial_context"] == {"key": "value"}
        assert result.agent_params["stop_tools"] == ["stop_tool"]
        assert result.agent_params["max_iterations"] == 50

    def test_build_mcp_servers_valid(self):
        """Test building MCP servers with valid configuration."""
        config = {
            "mcp_servers": [
                {"name": "test_server", "type": "stdio", "command": "python", "args": ["server.py"]},
                {"name": "http_server", "type": "http", "url": "http://localhost:8000"},
            ]
        }

        builder = AgentBuilder(config, Path("."))
        result = builder.build_mcp_servers()

        assert len(result.agent_params["mcp_servers"]) == 2
        assert result.agent_params["mcp_servers"][0]["name"] == "test_server"
        assert result.agent_params["mcp_servers"][1]["name"] == "http_server"

    def test_build_mcp_servers_invalid_type(self):
        """Test building MCP servers with invalid type."""
        config = {"mcp_servers": [{"name": "test_server", "type": "invalid_type"}]}

        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="invalid type 'invalid_type'"):
            builder.build_mcp_servers()

    def test_build_mcp_servers_missing_fields(self):
        """Test building MCP servers with missing required fields."""
        config = {
            "mcp_servers": [
                {
                    "name": "test_server",
                    "type": "stdio",
                    # Missing 'command' field
                }
            ]
        }

        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="missing 'command' field"):
            builder.build_mcp_servers()

    @patch("nexau.archs.config.config_loader.import_from_string")
    def test_build_hooks_valid(self, mock_import):
        """Test building hooks with valid configuration."""
        mock_import.return_value = lambda: None

        config = {
            "after_model_hooks": ["module.path:hook_function"],
            "before_model_hooks": ["module.path:before_hook"],
            "after_tool_hooks": ["module.path:tool_hook"],
            "before_tool_hooks": ["module.path:before_tool"],
            "middlewares": ["module.path:middleware"],
        }

        builder = AgentBuilder(config, Path("."))
        result = builder.build_hooks()

        assert len(result.agent_params["after_model_hooks"]) == 1
        assert len(result.agent_params["before_model_hooks"]) == 1
        assert len(result.agent_params["after_tool_hooks"]) == 1
        assert len(result.agent_params["before_tool_hooks"]) == 1
        assert len(result.agent_params["middlewares"]) == 1

    def test_build_hooks_invalid_format(self):
        """Test building hooks with invalid format."""
        config = {"after_model_hooks": ["invalid_format"]}

        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="Import string must contain ':' separator"):
            builder.build_hooks()

    @patch("nexau.archs.config.config_loader.AgentBuilder._import_and_instantiate")
    def test_build_tracers_valid(self, mock_import):
        """Tracer entries should instantiate to BaseTracer objects."""
        tracer_instance = Mock(spec=BaseTracer)
        mock_import.return_value = tracer_instance

        config = {"tracers": ["module.path:Tracer"]}
        builder = AgentBuilder(config, Path("."))

        result = builder.build_tracers()

        mock_import.assert_called_once_with("module.path:Tracer")
        assert result.agent_params["tracers"] == [tracer_instance]

    @patch("nexau.archs.config.config_loader.AgentBuilder._import_and_instantiate")
    def test_build_tracers_invalid_type(self, mock_import):
        """Tracer entries that do not return BaseTracer should fail."""
        mock_import.return_value = object()

        config = {"tracers": ["module.path:Tracer"]}
        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="Tracer must be an instance of BaseTracer"):
            builder.build_tracers()

    def test_build_tracers_accepts_instance(self):
        """Existing tracer instances should pass through unchanged."""
        tracer_instance = Mock(spec=BaseTracer)
        config = {"tracers": [tracer_instance]}

        builder = AgentBuilder(config, Path("."))
        result = builder.build_tracers()

        assert result.agent_params["tracers"] == [tracer_instance]

    @patch("nexau.archs.config.config_loader.AgentBuilder._import_and_instantiate")
    def test_build_tracers_handles_multiple_entries(self, mock_import):
        """Multiple tracer entries should be instantiated."""
        tracer1 = Mock(spec=BaseTracer)
        tracer2 = Mock(spec=BaseTracer)
        mock_import.side_effect = [tracer1, tracer2]

        config = {"tracers": ["module.path:Tracer1", "module.path:Tracer2"]}
        builder = AgentBuilder(config, Path("."))

        result = builder.build_tracers()

        assert result.agent_params["tracers"] == [tracer1, tracer2]

    def test_build_tracers_requires_list(self):
        """Non-list tracers configuration should raise error."""
        config = {"tracers": "not-a-list"}
        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="'tracers' must be a list"):
            builder.build_tracers()

    def test_build_llm_config(self):
        """Test building LLM configuration."""
        config = {
            "llm_config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
                "max_tokens": 1000,
            }
        }

        builder = AgentBuilder(config, Path("."))
        result = builder.build_llm_config()

        assert isinstance(result.agent_params["llm_config"], LLMConfig)
        assert result.agent_params["llm_config"].model == "gpt-4o-mini"
        assert result.agent_params["llm_config"].temperature == 0.1

    def test_build_llm_config_missing(self):
        """Test building LLM config when missing."""
        config = {}

        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="'llm_config' is required"):
            builder.build_llm_config()

    def test_build_skills_from_folders(self, temp_dir):
        """Test building skills from skill folders."""
        # Create two skill folders
        skill1_folder = Path(temp_dir) / "skill1"
        skill1_folder.mkdir()
        skill1_content = """---
name: skill-one
description: First test skill
---

Details for skill one.
"""
        (skill1_folder / "SKILL.md").write_text(skill1_content)

        skill2_folder = Path(temp_dir) / "skill2"
        skill2_folder.mkdir()
        skill2_content = """---
name: skill-two
description: Second test skill
---

Details for skill two.
"""
        (skill2_folder / "SKILL.md").write_text(skill2_content)

        config = {"skills": [str(skill1_folder), str(skill2_folder)]}

        builder = AgentBuilder(config, Path(temp_dir))
        result = builder.build_skills()

        assert len(result.agent_params["skills"]) == 2
        assert result.agent_params["skills"][0].name == "skill-one"
        assert result.agent_params["skills"][1].name == "skill-two"

    def test_build_skills_with_relative_paths(self, temp_dir):
        """Test building skills with relative paths."""
        # Create skill folder
        skill_folder = Path(temp_dir) / "relative_skill"
        skill_folder.mkdir()
        skill_content = """---
name: relative-skill
description: Skill with relative path
---

Details here.
"""
        (skill_folder / "SKILL.md").write_text(skill_content)

        config = {
            "skills": ["relative_skill"]  # Relative path
        }

        builder = AgentBuilder(config, Path(temp_dir))
        result = builder.build_skills()

        assert len(result.agent_params["skills"]) == 1
        assert result.agent_params["skills"][0].name == "relative-skill"

    def test_build_skills_empty_list(self):
        """Test building skills with empty skills list."""
        config = {"skills": []}

        builder = AgentBuilder(config, Path("."))
        result = builder.build_skills()

        assert result.agent_params["skills"] == []

    def test_build_skills_no_skills_config(self):
        """Test building skills when skills key is not in config."""
        config = {}

        builder = AgentBuilder(config, Path("."))
        result = builder.build_skills()

        assert result.agent_params["skills"] == []

    def test_build_skills_invalid_folder(self, temp_dir):
        """Test building skills with invalid folder path."""
        config = {"skills": [str(Path(temp_dir) / "nonexistent")]}

        builder = AgentBuilder(config, Path(temp_dir))

        with pytest.raises(ConfigError, match="Error loading skill"):
            builder.build_skills()


class TestToolExtraKwargs:
    """Behavioral tests for Tool.extra_kwargs and parameter merging."""

    def _base_tool(self, implementation, extra_kwargs=None, input_props=None):
        schema_props = input_props or {"a": {"type": "number"}, "b": {"type": "number"}, "c": {"type": "number"}}
        return Tool(
            name="test_tool",
            description="test",
            input_schema={"type": "object", "properties": schema_props, "additionalProperties": False},
            implementation=implementation,
            extra_kwargs=extra_kwargs,
        )

    def test_extra_kwargs_merge_and_override(self):
        """extra_kwargs are merged and call-time params override preset values."""

        def impl(**kwargs):
            return kwargs

        tool = self._base_tool(impl, extra_kwargs={"a": 1, "b": 2})
        result = tool.execute(b=99, c=3)

        assert result["a"] == 1  # from extra_kwargs
        assert result["b"] == 99  # call-time override
        assert result["c"] == 3

    def test_extra_kwargs_reserved_keys_rejected(self):
        """Reserved keys cannot be supplied via extra_kwargs."""
        with pytest.raises(ToolConfigError, match="reserved keys"):
            self._base_tool(lambda **_: {}, extra_kwargs={"agent_state": "x"})

    def test_extra_kwargs_unknown_param_reaches_function(self):
        """Unknown fields bypass schema and surface as TypeError when the function cannot accept them."""

        def impl(a):
            return {"a": a}

        tool = self._base_tool(impl, extra_kwargs={"x": 1}, input_props={"a": {"type": "number"}})
        response = tool.execute(a=5)

        assert response["error_type"] == "TypeError"
        assert "unexpected keyword argument 'x'" in response["error"]

    def test_extra_kwargs_satisfy_required_fields(self):
        """Required schema fields can be satisfied solely via extra_kwargs."""

        def impl(a, agent_state=None):
            return {"a": a}

        tool = self._base_tool(impl, extra_kwargs={"a": 7}, input_props={"a": {"type": "number"}})
        result = tool.execute()

        assert result["a"] == 7

    def test_extra_kwargs_unknown_with_kwargs_accepted(self):
        """Unknown fields are accepted if the implementation has **kwargs."""

        def impl(a, **kwargs):
            return {"a": a, "extras": kwargs}

        tool = self._base_tool(
            impl,
            extra_kwargs={"x": 1},
            input_props={"a": {"type": "number"}},
        )
        result = tool.execute(a=2)

        assert result["a"] == 2
        assert result["extras"]["x"] == 1

    def test_build_skills_folder_without_skill_md(self, temp_dir):
        """Test building skills from folder without SKILL.md."""
        empty_folder = Path(temp_dir) / "empty"
        empty_folder.mkdir()

        config = {"skills": [str(empty_folder)]}

        builder = AgentBuilder(config, Path(temp_dir))

        with pytest.raises(ConfigError, match="Error loading skill"):
            builder.build_skills()

    @patch("nexau.archs.config.config_loader.PromptBuilder")
    def test_build_skills_from_tools_with_as_skill(self, mock_prompt_builder_class, temp_dir):
        """Test building skills from tools with as_skill=True."""
        from nexau.archs.tool.tool import Tool

        # Mock PromptBuilder
        mock_builder = Mock()
        mock_template = "Tool: {{ tool.name }}"
        mock_builder._load_prompt_template.return_value = mock_template
        mock_builder.jinja_env.from_string.return_value.render.return_value = "Rendered skill detail"
        mock_prompt_builder_class.return_value = mock_builder

        # Create a tool with as_skill=True
        tool = Tool(
            name="skill_tool",
            description="A tool that's also a skill",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=True,
            skill_description="This tool can be used as a skill",
        )

        config = {"skills": []}
        builder = AgentBuilder(config, Path(temp_dir))
        builder.agent_params["tools"] = [tool]

        result = builder.build_skills()

        assert len(result.agent_params["skills"]) == 1
        assert result.agent_params["skills"][0].name == "skill_tool"
        assert result.agent_params["skills"][0].description == "This tool can be used as a skill"
        assert result.agent_params["skills"][0].detail == "Rendered skill detail"
        assert result.agent_params["skills"][0].folder == ""

    @patch("nexau.archs.config.config_loader.PromptBuilder")
    def test_build_skills_mixed_folders_and_tools(self, mock_prompt_builder_class, temp_dir):
        """Test building skills from both folders and tools."""
        from nexau.archs.tool.tool import Tool

        # Create a skill folder
        skill_folder = Path(temp_dir) / "folder_skill"
        skill_folder.mkdir()
        skill_content = """---
name: folder-skill
description: Skill from folder
---

Folder skill details.
"""
        (skill_folder / "SKILL.md").write_text(skill_content)

        # Mock PromptBuilder for tool-based skills
        mock_builder = Mock()
        mock_template = "Tool skill"
        mock_builder._load_prompt_template.return_value = mock_template
        mock_builder.jinja_env.from_string.return_value.render.return_value = "Tool skill detail"
        mock_prompt_builder_class.return_value = mock_builder

        # Create a tool with as_skill=True
        tool = Tool(
            name="tool_skill",
            description="Tool as skill",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=True,
            skill_description="Tool skill description",
        )

        config = {"skills": [str(skill_folder)]}
        builder = AgentBuilder(config, Path(temp_dir))
        builder.agent_params["tools"] = [tool]

        result = builder.build_skills()

        assert len(result.agent_params["skills"]) == 2
        assert result.agent_params["skills"][0].name == "folder-skill"
        assert result.agent_params["skills"][1].name == "tool_skill"

    @patch("nexau.archs.config.config_loader.PromptBuilder")
    def test_build_skills_tool_without_as_skill(self, mock_prompt_builder_class, temp_dir):
        """Test that tools with as_skill=False are not added as skills."""
        from nexau.archs.tool.tool import Tool

        # Mock PromptBuilder
        mock_builder = Mock()
        mock_prompt_builder_class.return_value = mock_builder

        # Create a regular tool (as_skill=False)
        tool = Tool(
            name="regular_tool",
            description="Just a regular tool",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=False,
        )

        config = {"skills": []}
        builder = AgentBuilder(config, Path(temp_dir))
        builder.agent_params["tools"] = [tool]

        result = builder.build_skills()

        # Should not add the tool as a skill
        assert len(result.agent_params["skills"]) == 0


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_load_agent_config_valid(self, temp_dir, mock_llm_config):
        """Test loading valid agent configuration."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {
                "model": "gpt-4o-mini",
                "temperature": 0.1,
            },
            "tools": [],
        }

        config_path = Path(temp_dir) / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = load_agent_config(str(config_path))

            assert agent.config.name == "test_agent"
            assert agent.config.llm_config.model == "gpt-4o-mini"

    def test_load_agent_config_with_type_field(self, temp_dir, mock_llm_config):
        """Agent configs can include an explicit type field."""
        config = {
            "type": "agent",
            "name": "typed_agent",
            "system_prompt": "Typed agent",
            "llm_config": {"model": "gpt-4o-mini"},
            "tools": [],
        }

        config_path = Path(temp_dir) / "typed_agent.yaml"
        config_path.write_text(yaml.dump(config))

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = load_agent_config(str(config_path))

            assert agent.config.name == "typed_agent"
            assert agent.config.llm_config.model == "gpt-4o-mini"

    def test_load_agent_config_invalid_type_field(self, temp_dir):
        """Invalid agent type values should fail validation."""
        config = {
            "type": "tool",
            "name": "bad_type_agent",
            "llm_config": {"model": "gpt-4o-mini"},
            "tools": [],
        }

        config_path = Path(temp_dir) / "bad_type_agent.yaml"
        config_path.write_text(yaml.dump(config))

        with pytest.raises(ConfigError, match="type"):
            load_agent_config(str(config_path))

    def test_load_agent_config_invalid(self, temp_dir):
        """Test loading invalid agent configuration."""
        config = {
            "name": "",  # Invalid
        }

        config_path = Path(temp_dir) / "invalid_agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ConfigError):
            load_agent_config(str(config_path))

    def test_load_agent_config_rejects_max_context_alias(self, temp_dir):
        """Legacy 'max_context' key should now trigger validation errors."""
        config = {
            "name": "test_agent",
            "max_context": 16000,
            "system_prompt": "Alias test",
            "llm_config": {
                "model": "gpt-4o-mini",
            },
            "tools": [],
        }

        config_path = Path(temp_dir) / "agent_alias.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ConfigError):
            load_agent_config(str(config_path))

    def test_sub_agent_configs_validated_eagerly(self, temp_dir):
        """Invalid referenced sub-agent configs should fail during main load."""
        sub_agent_path = Path(temp_dir) / "sub.yaml"
        with open(sub_agent_path, "w") as f:
            yaml.dump({"name": "sub"}, f)

        main_config = {
            "name": "main",
            "llm_config": {"model": "gpt-4o-mini"},
            "tools": [],
            "sub_agents": [
                {"name": "sub", "config_path": str(sub_agent_path)},
            ],
        }

        config_path = Path(temp_dir) / "main.yaml"
        with open(config_path, "w") as f:
            yaml.dump(main_config, f)

        with pytest.raises(ConfigError, match="sub-agent"):
            load_agent_config(str(config_path))

    def test_load_tool_from_config(self, temp_dir):
        """Test loading tool from configuration."""
        # Create a mock tool YAML file
        tool_config = {
            "name": "test_tool",
            "description": "A test tool",
            "input_schema": {"type": "object", "properties": {"param1": {"type": "string"}}},
        }

        tool_path = Path(temp_dir) / "test_tool.yaml"
        with open(tool_path, "w") as f:
            yaml.dump(tool_config, f)

        # Mock tool configuration
        config = {"name": "test_tool", "yaml_path": str(tool_path), "binding": "builtins:print"}

        tool = load_tool_from_config(config, Path(temp_dir))

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    def test_load_tool_from_config_overrides_name(self, temp_dir):
        """Agent config name should override tool YAML name."""
        tool_config = {
            "name": "Finish",
            "description": "A finish tool",
            "input_schema": {"type": "object"},
        }

        tool_path = Path(temp_dir) / "finish.yaml"
        tool_path.write_text(yaml.dump(tool_config))

        config = {"name": "finish", "yaml_path": str(tool_path), "binding": "builtins:print"}

        tool = load_tool_from_config(config, Path(temp_dir))

        assert tool.name == "finish"
        assert getattr(tool, "source_name", None) == "Finish"

    def test_load_tool_from_config_with_type_field(self, temp_dir):
        """Tool YAML files can include a type marker."""
        tool_config = {
            "type": "tool",
            "name": "typed_tool",
            "description": "Typed tool",
            "input_schema": {"type": "object"},
        }

        tool_path = Path(temp_dir) / "typed_tool.yaml"
        tool_path.write_text(yaml.dump(tool_config))

        config = {"name": "typed_tool", "yaml_path": str(tool_path), "binding": "builtins:print"}

        tool = load_tool_from_config(config, Path(temp_dir))

        assert tool.name == "typed_tool"
        assert tool.description == "Typed tool"

    def test_load_tool_from_config_invalid_yaml(self, temp_dir):
        """Tool YAML missing required fields should raise validation errors."""
        tool_path = Path(temp_dir) / "invalid_tool.yaml"
        with open(tool_path, "w") as f:
            yaml.dump({"description": "missing name"}, f)

        config = {"name": "invalid_tool", "yaml_path": str(tool_path), "binding": "builtins:print"}

        with pytest.raises(ValidationError):
            load_tool_from_config(config, Path(temp_dir))

    def test_tool_yaml_invalid_type(self, temp_dir):
        """Invalid tool type values should raise validation errors."""
        tool_config = {
            "type": "agent",
            "name": "bad_tool",
            "description": "Bad type",
            "input_schema": {"type": "object"},
        }

        tool_path = Path(temp_dir) / "bad_tool.yaml"
        tool_path.write_text(yaml.dump(tool_config))

        config = {"name": "bad_tool", "yaml_path": str(tool_path), "binding": "builtins:print"}

        with pytest.raises(ValidationError):
            load_tool_from_config(config, Path(temp_dir))

    def test_load_sub_agent_from_config(self, temp_dir):
        """Test loading sub-agent from configuration."""
        # Create a mock sub-agent config file
        sub_config = {
            "name": "sub_agent",
            "system_prompt": "You are a sub-agent.",
            "llm_config": {
                "model": "gpt-4o-mini",
            },
            "tools": [],
        }

        sub_path = Path(temp_dir) / "sub_agent.yaml"
        with open(sub_path, "w") as f:
            yaml.dump(sub_config, f)

        config = {"name": "sub_agent", "config_path": str(sub_path)}

        name, factory = load_sub_agent_from_config(config, Path(temp_dir))

        assert name == "sub_agent"
        # Factory should be callable
        assert callable(factory)
