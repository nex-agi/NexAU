"""
Unit tests for configuration loading components.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from northau.archs.config.config_loader import (
    AgentBuilder,
    ConfigError,
    apply_agent_name_overrides,
    load_agent_config,
    load_sub_agent_from_config,
    load_tool_from_config,
    validate_config_schema,
)
from northau.archs.llm.llm_config import LLMConfig


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

        from northau.archs.config.config_loader import load_yaml_with_vars

        result = load_yaml_with_vars(str(config_file))

        assert result["name"] == "test_agent"
        assert temp_dir in result["system_prompt"]

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

        with pytest.raises(ConfigError, match="'tools' field must be a list"):
            validate_config_schema(invalid_config)

    def test_validate_config_schema_invalid_sub_agents(self):
        """Test schema validation with invalid sub-agents configuration."""
        invalid_config = {
            "sub_agents": "not_a_list"  # Should be a list
        }

        with pytest.raises(ConfigError, match="'sub_agents' field must be a list"):
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

    @patch("northau.archs.config.config_loader.import_from_string")
    def test_build_hooks_valid(self, mock_import):
        """Test building hooks with valid configuration."""
        mock_import.return_value = lambda: None

        config = {
            "after_model_hooks": ["module.path:hook_function"],
            "before_model_hooks": ["module.path:before_hook"],
            "after_tool_hooks": ["module.path:tool_hook"],
        }

        builder = AgentBuilder(config, Path("."))
        result = builder.build_hooks()

        assert len(result.agent_params["after_model_hooks"]) == 1
        assert len(result.agent_params["before_model_hooks"]) == 1
        assert len(result.agent_params["after_tool_hooks"]) == 1

    def test_build_hooks_invalid_format(self):
        """Test building hooks with invalid format."""
        config = {"after_model_hooks": ["invalid_format"]}

        builder = AgentBuilder(config, Path("."))

        with pytest.raises(ConfigError, match="Import string must contain ':' separator"):
            builder.build_hooks()

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

        with patch("northau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = load_agent_config(str(config_path))

            assert agent.config.name == "test_agent"
            assert agent.config.llm_config.model == "gpt-4o-mini"

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
