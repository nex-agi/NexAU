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

"""Unit tests for search_file_content builtin tool."""

import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools import search_file_content


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestSearchFileContent:
    """Test search_file_content tool functionality."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp(prefix="grep-tool-test-")

        with open(os.path.join(self.temp_dir, "fileA.txt"), "w") as f:
            f.write("hello world\nsecond line with world\n")
        with open(os.path.join(self.temp_dir, "fileB.js"), "w") as f:
            f.write("const x = 1;\nfunction hello() {}\n")

        subdir = os.path.join(self.temp_dir, "sub")
        os.makedirs(subdir)
        with open(os.path.join(subdir, "fileC.txt"), "w") as f:
            f.write("another world in sub dir\n")

        sandbox = LocalSandbox(sandbox_id="test", _work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_find_matches_for_simple_pattern(self):
        """Should find matches for a simple pattern in all files."""
        result = search_file_content(pattern="world", dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "Found" in result["content"]
        assert "match" in result["content"]
        assert "fileA.txt" in result["content"]
        assert "world" in result["content"]

    def test_find_matches_in_specific_path(self):
        """Should find matches only in specified path."""
        subdir = os.path.join(self.temp_dir, "sub")
        result = search_file_content(pattern="world", dir_path=subdir, agent_state=self.agent_state)

        assert "Found" in result["content"]
        assert "fileC.txt" in result["content"]
        assert "fileA.txt" not in result["content"]

    def test_find_matches_with_include_glob(self):
        """Should filter files by include glob pattern."""
        result = search_file_content(
            pattern="hello",
            dir_path=self.temp_dir,
            include="*.txt",
            agent_state=self.agent_state,
        )

        assert "fileA.txt" in result["content"]
        assert "fileB.js" not in result["content"]

    def test_return_no_matches_found(self):
        """Should return appropriate message when no matches found."""
        result = search_file_content(
            pattern="nonexistent_pattern_xyz",
            dir_path=self.temp_dir,
            agent_state=self.agent_state,
        )

        assert "No matches found" in result["content"]

    def test_handle_regex_special_characters(self):
        """Should handle regex special characters in pattern."""
        with open(os.path.join(self.temp_dir, "special.txt"), "w") as f:
            f.write("test (parentheses) and [brackets]\n")

        result = search_file_content(
            pattern=r"\(parentheses\)",
            dir_path=self.temp_dir,
            agent_state=self.agent_state,
        )

        assert "special.txt" in result["content"]


class TestSearchFileContentValidation:
    """Test search_file_content parameter validation."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="grep-tool-test-")
        sandbox = LocalSandbox(sandbox_id="test", _work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_regex_pattern(self):
        """Should return error for invalid regex pattern."""
        result = search_file_content(pattern="[invalid", dir_path=self.temp_dir, agent_state=self.agent_state)

        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PATTERN"

    def test_path_does_not_exist(self):
        """Should return error when path doesn't exist."""
        result = search_file_content(
            pattern="test",
            dir_path="/nonexistent/path",
            agent_state=self.agent_state,
        )

        assert result.get("error") is not None
        assert result["error"]["type"] == "FILE_NOT_FOUND"

    def test_path_is_not_directory(self):
        """Should return error when path is not a directory."""
        file_path = os.path.join(self.temp_dir, "file.txt")
        with open(file_path, "w") as f:
            f.write("content")

        result = search_file_content(pattern="test", dir_path=file_path, agent_state=self.agent_state)

        assert result.get("error") is not None
        assert "directory" in result["content"].lower() or "directory" in result.get("error", {}).get("type", "").lower()


class TestSearchFileContentOutputFormat:
    """Test search_file_content output format."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp(prefix="grep-tool-test-")
        with open(os.path.join(self.temp_dir, "test.txt"), "w") as f:
            f.write("line one\nline two\nline three\n")
        sandbox = LocalSandbox(sandbox_id="test", _work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_llm_content_format(self):
        """Should format llmContent correctly."""
        result = search_file_content(pattern="line", dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "Found" in result["content"]
        assert "match" in result["content"]
        assert "File:" in result["content"]
        assert "L" in result["content"]

    def test_return_display_format(self):
        """Should format returnDisplay correctly."""
        result = search_file_content(pattern="line", dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "Found" in result["returnDisplay"]
        assert "match" in result["returnDisplay"]

    def test_error_format(self):
        """Should format error correctly."""
        result = search_file_content(pattern="[invalid", dir_path=self.temp_dir, agent_state=self.agent_state)

        assert result.get("error") is not None
        assert "message" in result["error"]
        assert "type" in result["error"]
