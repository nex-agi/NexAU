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

"""Integration tests for apply_patch when loaded through the example code agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from nexau import Agent, AgentConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState


class TestExampleCodeAgentApplyPatch:
    """Verify the examples/code_agent configuration exposes a working apply_patch tool."""

    def test_example_code_agent_loads_and_executes_apply_patch_tool(self, tmp_path):
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "examples" / "code_agent" / "code_agent_codex.yaml"

        with patch.dict("os.environ", {"SANDBOX_WORK_DIR": str(tmp_path)}, clear=False):
            config = AgentConfig.from_yaml(config_path)
            tool_names = [tool.name for tool in config.tools]
            assert "apply_patch" in tool_names

            apply_patch_tool = next(tool for tool in config.tools if tool.name == "apply_patch")
            assert apply_patch_tool.implementation_import_path == "nexau.archs.tool.builtin.file_tools:apply_patch"

            with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock(name="openai_client")
                agent = Agent(config=config)

            try:
                agent_state = AgentState(
                    agent_name=agent.config.name,
                    agent_id=agent.agent_id,
                    run_id="test-run",
                    root_run_id="test-run",
                    context=AgentContext({}),
                    global_storage=GlobalStorage(),
                    executor=Mock(),
                    sandbox_manager=agent.sandbox_manager,
                )

                patch_body = "*** Begin Patch\n*** Add File: code_agent_apply_patch_check.txt\n+hello from code agent\n*** End Patch"
                result = apply_patch_tool.execute(input=patch_body, agent_state=agent_state)

                assert result["content"] == "Success. Updated the following files:\nA code_agent_apply_patch_check.txt\n"

                sandbox = agent_state.get_sandbox()
                assert sandbox is not None
                read_res = sandbox.read_file("code_agent_apply_patch_check.txt", encoding="utf-8", binary=False)
                assert read_res.status.name == "SUCCESS"
                assert read_res.content == "hello from code agent\n"
            finally:
                agent.sandbox_manager.stop()
