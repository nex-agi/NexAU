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

"""Integration tests for structured-mode tool-based skills."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from nexau import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.skill import load_skill


def brief_search_impl(query: str) -> dict[str, str]:
    """Simple binding used by the YAML-backed integration test."""
    return {"query": query}


class _SkillRegistryState:
    """Minimal agent-state stand-in for load_skill integration checks."""

    def __init__(self, skill_registry: Mapping[str, object]):
        self._skill_registry = skill_registry

    def get_global_value(self, key: str, default=None):  # type: ignore[override]
        if key == "skill_registry":
            return self._skill_registry
        return default


class TestStructuredAsSkillIntegration:
    """Integration coverage for YAML + structured as_skill behavior."""

    @pytest.mark.integration
    @pytest.mark.parametrize("tool_call_mode", ["openai", "anthropic"])
    def test_tool_based_skill_uses_brief_description_up_front_and_loadskill_for_detail(
        self,
        tool_call_mode: str,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            tools_dir = base_dir / "tools"
            tools_dir.mkdir(parents=True)

            tool_yaml = tools_dir / "BriefSearch.yaml"
            tool_yaml.write_text(
                textwrap.dedent(
                    """
                    type: tool
                    name: brief_search
                    description: >-
                      FULL TOOL DESCRIPTION: Search internal knowledge sources with ranking
                      rules, fallback heuristics, and post-processing guidance.
                    skill_description: >-
                      BRIEF SKILL DESCRIPTION: Search internal knowledge sources.
                    input_schema:
                      type: object
                      properties:
                        query:
                          type: string
                          description: Search query
                      required:
                        - query
                    """,
                ).lstrip(),
            )

            agent_yaml = base_dir / "agent.yaml"
            agent_yaml.write_text(
                textwrap.dedent(
                    f"""
                    type: agent
                    name: structured_skill_agent
                    tool_call_mode: {tool_call_mode}
                    llm_config:
                      model: gpt-4o-mini
                    tools:
                      - name: brief_search
                        yaml_path: ./tools/BriefSearch.yaml
                        binding: tests.integration.test_as_skill_structured_modes:brief_search_impl
                        as_skill: true
                    """,
                ).lstrip(),
            )

            config = AgentConfig.from_yaml(agent_yaml)
            assert any(skill.name == "brief_search" for skill in config.skills)
            assert [tool.name for tool in config.tools].count("LoadSkill") == 1

            with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock()
                agent = Agent(config=config)

            assert [tool.name for tool in agent.config.tools].count("LoadSkill") == 1
            assert "brief_search" in agent.skill_registry

            if tool_call_mode == "openai":
                payload_by_name = {spec["function"]["name"]: spec for spec in cast(list[dict[str, Any]], agent.tool_call_payload)}
                assert (
                    payload_by_name["brief_search"]["function"]["description"]
                    == "BRIEF SKILL DESCRIPTION: Search internal knowledge sources."
                )
                assert payload_by_name["brief_search"]["function"]["parameters"]["properties"]["query"]["type"] == "string"
            else:
                payload_by_name = {spec["name"]: spec for spec in cast(list[dict[str, Any]], agent.tool_call_payload)}
                assert payload_by_name["brief_search"]["description"] == "BRIEF SKILL DESCRIPTION: Search internal knowledge sources."
                assert payload_by_name["brief_search"]["input_schema"]["properties"]["query"]["type"] == "string"

            skill_result = load_skill(
                "brief_search",
                cast(Any, _SkillRegistryState(agent.skill_registry)),
            )
            assert "FULL TOOL DESCRIPTION" in skill_result
            assert "BRIEF SKILL DESCRIPTION" in skill_result
