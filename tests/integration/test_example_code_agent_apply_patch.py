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

"""Integration tests for loading the example code agent configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from nexau import Agent, AgentConfig


class TestExampleCodeAgentConfig:
    """Verify the examples/code_agent configuration loads correctly."""

    def test_example_code_agent_loads_tools(self, tmp_path):
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "examples" / "code_agent" / "code_agent.yaml"

        with patch.dict(
            "os.environ",
            {
                "SANDBOX_WORK_DIR": str(tmp_path),
                "LANGFUSE_PUBLIC_KEY": "test-public-key",
                "LANGFUSE_SECRET_KEY": "test-secret-key",
                "LANGFUSE_HOST": "https://localhost",
            },
            clear=False,
        ):
            config = AgentConfig.from_yaml(config_path)
            tool_names = [tool.name for tool in config.tools]

            # Verify core tools are loaded
            assert "read_file" in tool_names
            assert "search_file_content" in tool_names
            assert "run_shell_command" in tool_names

            # Verify agent can be instantiated
            with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
                mock_openai.OpenAI.return_value = Mock(name="openai_client")
                agent = Agent(config=config)

            try:
                assert agent.config.name == "nexau_code_agent"
                assert len(config.tools) > 0
            finally:
                agent.sandbox_manager.stop()
