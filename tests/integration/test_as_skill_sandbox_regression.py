"""Integration regression coverage for as_skill loading and sandbox uploads."""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nexau import Agent
from nexau.archs.main_sub.agent_context import AgentContext
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import AgentConfig


def memory_lookup_impl(query: str) -> dict[str, str]:
    """Simple tool binding used to materialize a tool-based skill."""
    return {"query": query}


def inspect_skill_upload(expected_skill_file: str, sandbox, agent_state=None) -> dict[str, object]:
    """Check whether a folder-based skill asset was uploaded into the sandbox."""
    return {
        "skill_file_exists": sandbox.file_exists(expected_skill_file),
        "work_dir": str(sandbox.work_dir),
    }


def _build_agent_state(agent: Agent) -> AgentState:
    """Create a minimal AgentState for direct ToolExecutor integration checks."""
    return AgentState(
        agent_name=agent.config.name or "agent",
        agent_id=agent.agent_id,
        run_id="integration-run",
        root_run_id="integration-run",
        context=AgentContext({}),
        global_storage=agent.global_storage,
        executor=agent.executor,
        sandbox_manager=agent.sandbox_manager,
    )


class TestAsSkillSandboxRegression:
    """Regression tests for issue 311-style as_skill + sandbox interactions."""

    @pytest.mark.integration
    def test_loadskill_stays_in_memory_and_reused_config_keeps_folder_skill_upload_source(self, tmp_path: Path) -> None:
        base_dir = tmp_path
        tools_dir = base_dir / "tools"
        tools_dir.mkdir(parents=True)

        skill_dir = base_dir / "skills" / "folder_skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            textwrap.dedent(
                """
                ---
                name: folder-skill
                description: Folder-based skill used for upload regression coverage
                ---

                Refer to payload.txt for additional instructions.
                """,
            ).lstrip(),
        )
        (skill_dir / "payload.txt").write_text("folder skill payload\n")

        (tools_dir / "MemoryLookup.yaml").write_text(
            textwrap.dedent(
                """
                type: tool
                name: memory_lookup
                description: >-
                  FULL MEMORY TOOL DESCRIPTION: Resolve information from an in-memory
                  tool-based skill without requiring sandbox assets.
                skill_description: >-
                  BRIEF MEMORY TOOL DESCRIPTION: Load in-memory tool skill guidance.
                input_schema:
                  type: object
                  properties:
                    query:
                      type: string
                      description: Query string
                  required:
                    - query
                """,
            ).lstrip(),
        )
        (tools_dir / "InspectSkillUpload.yaml").write_text(
            textwrap.dedent(
                """
                type: tool
                name: inspect_skill_upload
                description: Verify folder-based skill assets are available in the sandbox.
                input_schema:
                  type: object
                  properties:
                    expected_skill_file:
                      type: string
                      description: Relative path to the uploaded skill asset inside the sandbox
                  required:
                    - expected_skill_file
                """,
            ).lstrip(),
        )

        sandbox_dir = base_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        agent_yaml = base_dir / "agent.yaml"
        agent_yaml.write_text(
            textwrap.dedent(
                f"""
                type: agent
                name: as_skill_sandbox_regression
                tool_call_mode: openai
                llm_config:
                  model: gpt-4o-mini
                sandbox_config:
                  type: local
                  work_dir: {sandbox_dir}
                  status_after_run: none
                tools:
                  - name: memory_lookup
                    yaml_path: ./tools/MemoryLookup.yaml
                    binding: tests.integration.test_as_skill_sandbox_regression:memory_lookup_impl
                    as_skill: true
                  - name: inspect_skill_upload
                    yaml_path: ./tools/InspectSkillUpload.yaml
                    binding: tests.integration.test_as_skill_sandbox_regression:inspect_skill_upload
                skills:
                  - ./skills/folder_skill
                """,
            ).lstrip(),
        )

        config = AgentConfig.from_yaml(agent_yaml)
        original_skill_folder = config.skills[0].folder
        uploaded_asset_relative_path = f".skills/{skill_dir.name}/payload.txt"

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock(name="openai_client")

            agent1 = Agent(config=config)
            try:
                agent1_state = _build_agent_state(agent1)
                assert agent1.sandbox_manager._instance is None

                load_skill_result = agent1.executor.tool_executor.execute_tool(
                    agent_state=agent1_state,
                    tool_name="LoadSkill",
                    parameters={"skill_name": "memory_lookup"},
                    tool_call_id="load_skill_call_1",
                )

                assert "FULL MEMORY TOOL DESCRIPTION" in load_skill_result["result"]
                assert agent1.sandbox_manager._instance is None

                upload_check_1 = agent1.executor.tool_executor.execute_tool(
                    agent_state=agent1_state,
                    tool_name="inspect_skill_upload",
                    parameters={"expected_skill_file": uploaded_asset_relative_path},
                    tool_call_id="inspect_upload_call_1",
                )

                assert upload_check_1["skill_file_exists"] is True
                assert config.skills[0].folder == original_skill_folder

                uploaded_skill_dir = sandbox_dir / ".skills" / skill_dir.name
                if uploaded_skill_dir.exists():
                    shutil.rmtree(uploaded_skill_dir)
                assert not uploaded_skill_dir.exists()

                agent2 = Agent(config=config)
                try:
                    agent2_state = _build_agent_state(agent2)
                    assert config.skills[0].folder == original_skill_folder
                    assert agent2.sandbox_manager._instance is None

                    load_skill_result_2 = agent2.executor.tool_executor.execute_tool(
                        agent_state=agent2_state,
                        tool_name="LoadSkill",
                        parameters={"skill_name": "memory_lookup"},
                        tool_call_id="load_skill_call_2",
                    )

                    assert "FULL MEMORY TOOL DESCRIPTION" in load_skill_result_2["result"]
                    assert agent2.sandbox_manager._instance is None

                    upload_check_2 = agent2.executor.tool_executor.execute_tool(
                        agent_state=agent2_state,
                        tool_name="inspect_skill_upload",
                        parameters={"expected_skill_file": uploaded_asset_relative_path},
                        tool_call_id="inspect_upload_call_2",
                    )

                    assert upload_check_2["skill_file_exists"] is True
                    assert config.skills[0].folder == original_skill_folder
                finally:
                    agent2.sandbox_manager.stop()
            finally:
                agent1.sandbox_manager.stop()
