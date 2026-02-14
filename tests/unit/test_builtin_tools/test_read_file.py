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

"""Unit tests for read_file builtin tool."""

from pathlib import Path
from unittest.mock import Mock

from nexau.archs.sandbox import SandboxStatus
from nexau.archs.tool.builtin.file_tools import read_file


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestReadFileLineLengthTruncation:
    """Test read_file single-line length truncation (MAX_LINE_LENGTH = 2000)."""

    def test_truncates_line_longer_than_2000_chars(self):
        """Should truncate lines exceeding 2000 chars and append '... [truncated]'."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        # Create content with one line > 2000 chars
        long_line = "a" * 2500
        content = f"short line\n{long_line}\nanother short"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = content
        read_res.truncated = False

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return Mock(status=SandboxStatus.SUCCESS, content=b"")
            return read_res

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="test.txt", agent_state=agent_state)

        assert "error" not in result or result.get("error") is None
        content = result["content"]
        # Truncated line should be 2000 + suffix, not 2500
        assert "... [truncated]" in content
        # Should not contain the full 2500-char line
        assert "a" * 2500 not in content
        # Should contain truncated version
        assert "a" * 2000 + "... [truncated]" in content
        assert "some lines were shortened" in result["returnDisplay"]

    def test_no_truncation_when_all_lines_under_limit(self):
        """Should not truncate when all lines are under 2000 chars."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        content = "line1\nline2\nline3"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = content
        read_res.truncated = False

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return Mock(status=SandboxStatus.SUCCESS, content=b"")
            return read_res

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="test.txt", agent_state=agent_state)

        assert "... [truncated]" not in result["content"]
        assert "some lines were shortened" not in result["returnDisplay"]


class TestReadFileBinaryImage:
    """Test read_file binary image returns nexau-supported format (result + type=image)."""

    def test_image_returns_result_with_image_block_format(self):
        """Image files should return result with type=image, image_url=data:... for coerce."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True

        info = Mock()
        info.is_directory = False
        info.size = 100
        sandbox.get_file_info.return_value = info

        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        read_res = Mock()
        read_res.status = SandboxStatus.SUCCESS
        read_res.content = png_bytes

        def read_file_side_effect(path, encoding=None, binary=False):
            if binary:
                return read_res
            return Mock(status=SandboxStatus.SUCCESS, content="", truncated=False)

        sandbox.read_file.side_effect = read_file_side_effect

        agent_state = _make_agent_state(sandbox)
        result = read_file(file_path="x.png", agent_state=agent_state)

        assert "error" not in result or result.get("error") is None
        assert "content" in result
        block = result["content"]
        assert block["type"] == "image"
        assert "image_url" in block
        assert block["image_url"].startswith("data:image/")
        assert "base64," in block["image_url"]
        assert block["detail"] == "auto"
