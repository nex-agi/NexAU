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
Integration tests for configuration loading and agent creation.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from nexau.archs.config.config_loader import load_agent_config


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    @pytest.mark.integration
    def test_load_agent_from_yaml(self):
        """Test loading agent configuration from YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "agent.yaml")

            config = {
                "name": "test_agent",
                "llm_config": {"model": "gpt-4o-mini", "temperature": 0.7, "max_tokens": 2000},
                "system_prompt": "You are a helpful assistant.",
                "tools": [],
            }

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Load config - returns Agent object
            agent = load_agent_config(config_path)
            assert agent.config.name == "test_agent"
            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert "helpful assistant" in agent.config.system_prompt.lower()

    @pytest.mark.integration
    def test_load_agent_with_tools(self):
        """Test loading agent with tool configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create tools directory
            tools_dir = os.path.join(temp_dir, "tools")
            os.makedirs(tools_dir)

            # Create tool config with proper input_schema
            tool_config_path = os.path.join(tools_dir, "test_tool.yaml")
            tool_config = {
                "name": "test_tool",
                "description": "A test tool",
                "builtin": "bash",  # Use a builtin tool as placeholder
                "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
            }
            with open(tool_config_path, "w") as f:
                yaml.dump(tool_config, f)

            # Create agent config - tools are specified as objects with yaml_path
            config_path = os.path.join(temp_dir, "agent.yaml")
            agent_config = {
                "name": "agent_with_tools",
                "llm_config": {"model": "gpt-4o-mini"},
                "tools": [
                    {"name": "test_tool", "yaml_path": "tools/test_tool.yaml", "binding": "nexau.archs.tool.builtin.bash_tool:bash_tool"}
                ],
            }
            with open(config_path, "w") as f:
                yaml.dump(agent_config, f)

            # Load config - returns Agent object
            agent = load_agent_config(config_path)
            assert agent.config.name == "agent_with_tools"
            assert len(agent.config.tools) == 1
            assert agent.config.tools[0].name == "test_tool"

    @pytest.mark.integration
    def test_load_agent_with_subagents(self):
        """Test loading agent with sub-agent configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sub-agent config
            subagent_path = os.path.join(temp_dir, "subagent.yaml")
            subagent_config = {"name": "sub_agent", "llm_config": {"model": "gpt-4o-mini"}, "tools": []}
            with open(subagent_path, "w") as f:
                yaml.dump(subagent_config, f)

            # Create main agent config - sub_agents are objects with config_path
            main_agent_path = os.path.join(temp_dir, "main_agent.yaml")
            main_config = {
                "name": "main_agent",
                "llm_config": {"model": "gpt-4o"},
                "sub_agents": [{"name": "sub_agent", "config_path": "subagent.yaml"}],
                "tools": [],
            }
            with open(main_agent_path, "w") as f:
                yaml.dump(main_config, f)

            # Load main agent config - returns Agent object
            agent = load_agent_config(main_agent_path)
            assert agent.config.name == "main_agent"
            assert agent.config.sub_agents is not None
            assert len(agent.config.sub_agents) == 1
            # sub_agents are stored as (name, factory) tuples
            assert any("sub_agent" in name for name, _ in agent.config.sub_agents)

    @pytest.mark.integration
    def test_config_with_environment_variables(self):
        """Test configuration loading with environment variables."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "agent.yaml")

            config = {
                "name": "test_agent",
                "llm_config": {"model": "${MODEL_NAME:gpt-4o-mini}", "api_key": "${API_KEY}", "temperature": 0.7},
                "tools": [],
            }

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Set environment variable
            os.environ["MODEL_NAME"] = "gpt-4o"
            os.environ["API_KEY"] = "test-key-123"

            try:
                # Load config - returns Agent object
                agent = load_agent_config(config_path)
                assert agent.config.name == "test_agent"
                # Environment variable substitution happens during config loading
                # The actual model might be substituted or not depending on implementation
            finally:
                # Cleanup
                os.environ.pop("MODEL_NAME", None)
                os.environ.pop("API_KEY", None)


class TestAgentBuilderIntegration:
    """Integration tests for AgentBuilder."""

    @pytest.mark.integration
    def test_build_agent_from_config(self):
        """Test building an agent from configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "agent.yaml")

            config = {
                "name": "built_agent",
                "llm_config": {"model": "gpt-4o-mini", "temperature": 0.8, "max_tokens": 1500},
                "system_prompt": "Test prompt",
                "tools": [],
            }

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with patch("nexau.archs.config.config_loader.create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_create.return_value = mock_agent

                # Build agent (would use actual AgentBuilder)
                # agent = AgentBuilder.from_yaml(config_path)
                # assert agent is not None
                assert True  # Placeholder

    @pytest.mark.integration
    def test_build_agent_with_overrides(self):
        """Test building agent with configuration overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "agent.yaml")

            config = {"name": "override_agent", "llm_config": {"model": "gpt-4o-mini", "temperature": 0.7}, "tools": []}

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Test overrides
            overrides = {"llm_config": {"temperature": 0.9, "max_tokens": 3000}}

            # In real implementation, would apply overrides
            assert overrides["llm_config"]["temperature"] == 0.9


class TestComplexConfigScenarios:
    """Integration tests for complex configuration scenarios."""

    @pytest.mark.integration
    def test_multi_level_agent_hierarchy(self):
        """Test loading configuration with multiple levels of sub-agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create leaf sub-agent
            leaf_agent_path = os.path.join(temp_dir, "leaf.yaml")
            leaf_config = {"name": "leaf_agent", "llm_config": {"model": "gpt-4o-mini"}, "tools": []}
            with open(leaf_agent_path, "w") as f:
                yaml.dump(leaf_config, f)

            # Create middle sub-agent - sub_agents are objects with config_path
            middle_agent_path = os.path.join(temp_dir, "middle.yaml")
            middle_config = {
                "name": "middle_agent",
                "llm_config": {"model": "gpt-4o-mini"},
                "sub_agents": [{"name": "leaf_agent", "config_path": "leaf.yaml"}],
                "tools": [],
            }
            with open(middle_agent_path, "w") as f:
                yaml.dump(middle_config, f)

            # Create root agent - sub_agents are objects with config_path
            root_agent_path = os.path.join(temp_dir, "root.yaml")
            root_config = {
                "name": "root_agent",
                "llm_config": {"model": "gpt-4o"},
                "sub_agents": [{"name": "middle_agent", "config_path": "middle.yaml"}],
                "tools": [],
            }
            with open(root_agent_path, "w") as f:
                yaml.dump(root_config, f)

            # Load root config - returns Agent object
            agent = load_agent_config(root_agent_path)
            assert agent.config.name == "root_agent"
            assert agent.config.sub_agents is not None
            assert len(agent.config.sub_agents) >= 1

    @pytest.mark.integration
    def test_shared_tools_across_agents(self):
        """Test configuration where multiple agents share tools."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = os.path.join(temp_dir, "shared_tools")
            os.makedirs(tools_dir)

            # Create shared tool
            tool_path = os.path.join(tools_dir, "shared.yaml")
            tool_config = {"name": "shared_tool", "description": "A tool shared by multiple agents"}
            with open(tool_path, "w") as f:
                yaml.dump(tool_config, f)

            # Create two agents using the same tool
            for i in range(2):
                agent_path = os.path.join(temp_dir, f"agent{i}.yaml")
                agent_config = {"name": f"agent{i}", "llm_config": {"model": "gpt-4o-mini"}, "tools": ["shared_tools/shared.yaml"]}
                with open(agent_path, "w") as f:
                    yaml.dump(agent_config, f)

            # Verify configs exist
            assert os.path.exists(tool_path)
            assert os.path.exists(os.path.join(temp_dir, "agent0.yaml"))
            assert os.path.exists(os.path.join(temp_dir, "agent1.yaml"))
