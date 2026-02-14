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

"""Unit tests for read_many_files builtin tool."""

from pathlib import Path
from unittest.mock import Mock

from nexau.archs.sandbox import SandboxStatus
from nexau.archs.tool.builtin.file_tools import read_many_files


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestReadManyFilesLineLengthTruncation:
    """Test read_many_files single-line length truncation (MAX_LINE_LENGTH = 2000)."""

    def test_truncates_long_lines_in_read_many_files(self):
        """Should truncate lines exceeding 2000 chars in concatenated file content."""
        base_dir = "/tmp/work"
        sandbox = Mock()
        sandbox.work_dir = Path(base_dir)
        sandbox.file_exists.return_value = False  # No .gitignore

        # One text file with a long line
        file_path = f"{base_dir}/test.txt"
        long_line = "x" * 2500
        content = f"normal line\n{long_line}\nshort"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = content
        read_res.truncated = False
        sandbox.read_file.return_value = read_res

        # Glob returns one match
        sandbox.glob.return_value = [file_path]
        info = Mock()
        info.is_file = True
        sandbox.get_file_info.return_value = info

        agent_state = _make_agent_state(sandbox)
        result = read_many_files(include=["*.txt"], agent_state=agent_state)

        assert "error" not in result or result.get("error") is None
        result_content = result["content"]
        if isinstance(result_content, list):
            result_str = "\n".join(str(p) for p in result_content if isinstance(p, str))
        else:
            result_str = str(result_content)
        # Should contain truncation suffix
        assert "... [truncated]" in result_str
        # Long line should be truncated to 2000 + suffix
        assert "x" * 2500 not in result_str
        assert "x" * 2000 + "... [truncated]" in result_str
        # Should have the warning about truncated file
        assert "WARNING" in result_str or "truncated" in result_str.lower()
