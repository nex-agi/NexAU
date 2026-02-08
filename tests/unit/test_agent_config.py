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
Unit tests for AgentConfig and ExecutionConfig classes, focusing on skill-related functionality.
"""

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config import AgentConfig, ExecutionConfig
from nexau.archs.main_sub.skill import Skill
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.composite import CompositeTracer
from nexau.archs.tracer.core import BaseTracer, Span, SpanType


def make_llm_config(**overrides):
    """Helper to create consistent LLMConfig instances for tests."""
    params = {"model": "gpt-4o-mini"}
    params.update(overrides)
    return LLMConfig(**params)


class DummyTracer(BaseTracer):
    """Minimal tracer implementation for tests."""

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict | None = None,
        parent_span: Span | None = None,
        attributes: dict | None = None,
    ) -> Span:
        return Span(id="dummy", name=name, type=span_type)

    def end_span(self, span: Span, outputs=None, error=None, attributes=None) -> None:
        return None


class TestExecutionConfig:
    """Test cases for ExecutionConfig dataclass."""

    def test_execution_config_defaults(self):
        """Test ExecutionConfig with default values."""
        config = ExecutionConfig()

        assert config.max_iterations == 100
        assert config.max_context_tokens == 128000
        assert config.max_running_subagents == 5
        assert config.retry_attempts == 5
        assert config.timeout == 300

    def test_execution_config_custom_values(self):
        """Test ExecutionConfig with custom values."""
        config = ExecutionConfig(max_iterations=50, max_context_tokens=64000, max_running_subagents=3, retry_attempts=3, timeout=180)

        assert config.max_iterations == 50
        assert config.max_context_tokens == 64000
        assert config.max_running_subagents == 3
        assert config.retry_attempts == 3
        assert config.timeout == 180
        assert config.tool_call_mode == "openai"

    def test_execution_config_supports_anthropic_mode(self):
        """ExecutionConfig should normalize anthropic mode."""
        config = ExecutionConfig(tool_call_mode="anthropic")

        assert config.tool_call_mode == "anthropic"

    def test_execution_config_invalid_tool_call_mode(self):
        """Invalid tool_call_mode values should raise ValueError."""
        with pytest.raises(ValueError):
            ExecutionConfig(tool_call_mode="json")


class TestAgentConfigSkills:
    """Test cases for skill-related functionality in AgentConfig."""

    def test_agent_config_with_skills(self):
        """Test AgentConfig initialization with skills."""
        skill = Skill(name="test-skill", description="Test skill description", detail="Skill details", folder="/path/to/skill")

        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), skills=[skill])

        assert len(config.skills) == 1
        assert config.skills[0].name == "test-skill"

    def test_agent_config_with_multiple_skills(self):
        """Test AgentConfig with multiple skills."""
        skills = [Skill(name=f"skill-{i}", description=f"Skill {i}", detail=f"Detail {i}", folder=f"/path/{i}") for i in range(3)]

        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), skills=skills)

        assert len(config.skills) == 3

    def test_agent_config_with_tool_as_skill(self):
        """Test AgentConfig with a tool marked as skill."""
        tool = Tool(
            name="tool_skill",
            description="A tool that's also a skill",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=True,
            skill_description="This tool can be used as a skill",
        )

        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), tools=[tool])

        # Original tool should still be there
        assert any(t.name == "tool_skill" for t in config.tools)

    def test_agent_config_with_multiple_skilled_tools(self):
        """Test AgentConfig with multiple tools marked as skills."""
        tools = [
            Tool(
                name=f"skill_tool_{i}",
                description=f"Skill tool {i}",
                input_schema={"type": "object"},
                implementation=lambda: None,
                as_skill=True,
                skill_description=f"Skill description {i}",
            )
            for i in range(3)
        ]

        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), tools=tools)

        # All original tools should still be there
        assert len([t for t in config.tools if "skill_tool_" in t.name]) == 3

    def test_agent_config_with_skills_and_skilled_tools(self):
        """Test AgentConfig with both skills and tools marked as skills."""
        skill = Skill(name="folder-skill", description="Skill from folder", detail="Details", folder="/path/to/skill")

        tool = Tool(
            name="tool_skill",
            description="Tool as skill",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=True,
            skill_description="Tool skill description",
        )

        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), skills=[skill], tools=[tool])

        assert len(config.skills) == 1
        # Should have the original tool
        assert any(t.name == "tool_skill" for t in config.tools)

    def test_agent_config_empty_tools_and_skills(self):
        """Test AgentConfig with empty tools and skills."""
        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), tools=[], skills=[])

        # Should NOT have added skill_tool
        assert len(config.tools) == 0
        assert len(config.skills) == 0


class TestAgentConfigTracing:
    """Tests for tracer aggregation on AgentConfig."""

    def test_agent_config_single_tracer(self):
        tracer = DummyTracer()
        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), tracers=[tracer])

        assert config.tracers == [tracer]
        assert config.resolved_tracer is tracer

    def test_agent_config_composes_multiple_tracers(self):
        tracer1 = DummyTracer()
        tracer2 = DummyTracer()

        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), tracers=[tracer1, tracer2])

        assert isinstance(config.resolved_tracer, CompositeTracer)
        assert config.resolved_tracer.tracers == [tracer1, tracer2]

    def test_agent_config_rejects_invalid_tracer_entries(self):
        with pytest.raises(ValueError):
            AgentConfig(name="test_agent", llm_config=make_llm_config(), tracers=["not-a-tracer"])


class TestAgentConfigPostInit:
    """Test cases for AgentConfig __post_init__ method."""

    def test_post_init_no_sub_agents(self):
        """Test that sub_agents is empty dict when no sub_agents."""
        config = AgentConfig(name="test_agent", llm_config=make_llm_config())

        assert hasattr(config, "sub_agents")
        assert config.sub_agents is None

    def test_post_init_stop_tools_list_to_set(self):
        """Test that stop_tools list is converted to set."""
        config = AgentConfig(
            name="test_agent",
            llm_config=make_llm_config(),
            stop_tools=["tool1", "tool2", "tool1"],  # List with duplicate
        )

        assert isinstance(config.stop_tools, set)
        assert len(config.stop_tools) == 2
        assert "tool1" in config.stop_tools
        assert "tool2" in config.stop_tools

    def test_post_init_stop_tools_none(self):
        """Test that None stop_tools becomes empty set."""
        config = AgentConfig(name="test_agent", llm_config=make_llm_config(), stop_tools=None)

        assert isinstance(config.stop_tools, set)
        assert len(config.stop_tools) == 0

    def test_post_init_llm_config_dict_conversion(self):
        """Test that LLMConfig overrides are preserved."""
        config = AgentConfig(name="test_agent", llm_config=make_llm_config(temperature=0.5))

        assert isinstance(config.llm_config, LLMConfig)
        assert config.llm_config.model == "gpt-4o-mini"
        assert config.llm_config.temperature == 0.5

    def test_post_init_llm_config_object(self):
        """Test that LLMConfig object is preserved."""
        llm_config = LLMConfig(model="gpt-4o-mini", temperature=0.3)

        config = AgentConfig(name="test_agent", llm_config=llm_config)

        assert isinstance(config.llm_config, LLMConfig)
        assert config.llm_config == llm_config

    def test_post_init_llm_config_none(self):
        """Test that a default LLMConfig is created when none is provided."""
        config = AgentConfig(name="test_agent", llm_config=None)

        assert isinstance(config.llm_config, LLMConfig)

    def test_post_init_llm_config_invalid_type(self):
        """Test that invalid llm_config type raises error."""
        with pytest.raises(ValueError, match="Invalid llm_config type"):
            AgentConfig(
                name="test_agent",
                llm_config="invalid",  # String is invalid
            )

    def test_post_init_name_generation(self):
        """Test that name is generated if not provided."""
        config = AgentConfig(name=None, llm_config=make_llm_config())

        assert config.name is not None
        assert config.name.startswith("agent_")

    def test_post_init_name_preserved(self):
        """Test that provided name is preserved."""
        config = AgentConfig(name="my_agent", llm_config=make_llm_config())

        assert config.name == "my_agent"


class TestAgentConfigIntegration:
    """Integration tests for AgentConfig with skills."""

    def test_full_skill_workflow(self, temp_dir):
        """Test complete workflow with skills from creation to description."""
        # Create actual skills
        skill1 = Skill(name="analysis-skill", description="Data analysis skill", detail="Analyzes data", folder="/analysis")

        skill2 = Skill(name="reporting-skill", description="Report generation skill", detail="Generates reports", folder="/reporting")

        # Create tools with as_skill
        tool_skill = Tool(
            name="code_skill",
            description="Coding skill",
            input_schema={"type": "object"},
            implementation=lambda: None,
            as_skill=True,
            skill_description="Can write code",
        )

        regular_tool = Tool(
            name="calculator", description="Calculator", input_schema={"type": "object"}, implementation=lambda: None, as_skill=False
        )

        # Create agent config
        config = AgentConfig(
            name="multi_skill_agent", llm_config=make_llm_config(), skills=[skill1, skill2], tools=[tool_skill, regular_tool]
        )

        # Verify skills are registered
        assert len(config.skills) == 2

        # Verify tools include LoadSkill, tool_skill, and regular_tool
        assert len(config.tools) == 3
        tool_names = [t.name for t in config.tools]
        assert "LoadSkill" in tool_names
        assert "code_skill" in tool_names
        assert "calculator" in tool_names
