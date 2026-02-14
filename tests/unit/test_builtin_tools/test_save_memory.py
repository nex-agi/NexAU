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

"""Unit tests for save_memory builtin tool."""

from pathlib import Path
from unittest.mock import Mock

from nexau.archs.sandbox import SandboxStatus
from nexau.archs.tool.builtin.session_tools import save_memory


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestSaveMemoryValidation:
    """Test save_memory validation (no sandbox writes)."""

    def test_empty_fact_returns_error(self):
        """Should return error when fact is empty."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp")
        agent_state = _make_agent_state(sandbox)
        result = save_memory(fact="", agent_state=agent_state)
        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "non-empty" in result["content"].lower()

    def test_whitespace_only_fact_returns_error(self):
        """Should return error when fact is only whitespace."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp")
        agent_state = _make_agent_state(sandbox)
        result = save_memory(fact="   \n\t  ", agent_state=agent_state)
        assert "error" in result
        assert result["error"]["type"] == "INVALID_PARAMETER"


class TestSaveMemoryWithMockedSandbox:
    """Test save_memory with mocked sandbox file operations."""

    def test_successful_save_appends_to_memory_file(self):
        """Should append fact to GEMINI.md and return success."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = False
        write_result = Mock()
        write_result.status = SandboxStatus.SUCCESS
        sandbox.write_file.return_value = write_result
        agent_state = _make_agent_state(sandbox)
        result = save_memory(fact="User prefers dark mode", agent_state=agent_state)
        assert "error" not in result or result.get("error") is None
        assert "Okay, I've remembered" in result["returnDisplay"]
        sandbox.write_file.assert_called_once()
        call_args = sandbox.write_file.call_args
        written_content = call_args[0][1]
        assert "User prefers dark mode" in written_content
        assert "## Gemini Added Memories" in written_content

    def test_save_appends_to_existing_file(self):
        """Should append to existing memory section."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True
        read_result = Mock()
        read_result.status = SandboxStatus.SUCCESS
        read_result.content = "# Context\n\n## Gemini Added Memories\n- Existing fact\n"
        sandbox.read_file.return_value = read_result
        write_result = Mock()
        write_result.status = SandboxStatus.SUCCESS
        sandbox.write_file.return_value = write_result
        agent_state = _make_agent_state(sandbox)
        result = save_memory(fact="New fact to remember", agent_state=agent_state)
        assert "error" not in result or result.get("error") is None
        call_args = sandbox.write_file.call_args
        written_content = call_args[0][1]
        assert "Existing fact" in written_content
        assert "New fact to remember" in written_content
