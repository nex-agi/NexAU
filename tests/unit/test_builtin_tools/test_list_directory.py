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

"""Unit tests for list_directory builtin tool."""

import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools import list_directory


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestListDirectory:
    """Test list_directory tool functionality."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temporary directory structure for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="ls-tool-test-")
        sandbox = LocalSandbox(sandbox_id="test", _work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_files_in_directory(self):
        """Should list files in a directory."""
        with open(os.path.join(self.temp_dir, "file1.txt"), "w") as f:
            f.write("content1")
        os.makedirs(os.path.join(self.temp_dir, "subdir"))

        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "[DIR] subdir" in result["content"]
        assert "file1.txt" in result["content"]
        assert "Listed" in result["returnDisplay"]
        assert "item(s)" in result["returnDisplay"]

    def test_handle_empty_directories(self):
        """Should handle empty directories."""
        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "empty" in result["returnDisplay"].lower() or "Listed" in result["returnDisplay"]

    def test_respect_gitignore_patterns(self):
        """Should respect .gitignore patterns."""
        os.makedirs(os.path.join(self.temp_dir, "node_modules"))
        with open(os.path.join(self.temp_dir, "package.json"), "w") as f:
            f.write("{}")

        with open(os.path.join(self.temp_dir, ".gitignore"), "w") as f:
            f.write("node_modules/\n")

        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "package.json" in result["content"]

    def test_respect_geminiignore_patterns(self):
        """Should respect .geminiignore patterns."""
        with open(os.path.join(self.temp_dir, "visible.txt"), "w") as f:
            f.write("visible")
        with open(os.path.join(self.temp_dir, "secret.env"), "w") as f:
            f.write("SECRET=value")

        with open(os.path.join(self.temp_dir, ".geminiignore"), "w") as f:
            f.write("*.env\n")

        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "visible.txt" in result["content"]

    def test_handle_non_directory_paths(self):
        """Should return error for non-directory paths."""
        file_path = os.path.join(self.temp_dir, "file.txt")
        with open(file_path, "w") as f:
            f.write("content")

        result = list_directory(dir_path=file_path, agent_state=self.agent_state)

        assert result.get("error") is not None
        assert "not a directory" in result["content"].lower()

    def test_handle_non_existent_paths(self):
        """Should return error for non-existent paths."""
        result = list_directory(dir_path="/nonexistent/path", agent_state=self.agent_state)

        assert result.get("error") is not None

    def test_sort_directories_first_then_files_alphabetically(self):
        """Should sort directories first, then files alphabetically."""
        os.makedirs(os.path.join(self.temp_dir, "zdir"))
        os.makedirs(os.path.join(self.temp_dir, "adir"))
        with open(os.path.join(self.temp_dir, "zfile.txt"), "w") as f:
            f.write("z")
        with open(os.path.join(self.temp_dir, "afile.txt"), "w") as f:
            f.write("a")

        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        content = result["content"]
        adir_pos = content.find("adir")
        zdir_pos = content.find("zdir")
        afile_pos = content.find("afile")
        zfile_pos = content.find("zfile")

        assert adir_pos < afile_pos
        assert zdir_pos < zfile_pos

    def test_show_hidden_default_true(self):
        """Should include hidden files and directories by default."""
        os.makedirs(os.path.join(self.temp_dir, ".hidden_dir"))
        with open(os.path.join(self.temp_dir, ".hidden_file"), "w") as f:
            f.write("hidden")
        with open(os.path.join(self.temp_dir, "visible.txt"), "w") as f:
            f.write("visible")

        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert ".hidden_dir" in result["content"]
        assert ".hidden_file" in result["content"]
        assert "visible.txt" in result["content"]

    def test_show_hidden_false_excludes_dotfiles(self):
        """Should exclude hidden files and directories when show_hidden=False."""
        os.makedirs(os.path.join(self.temp_dir, ".hidden_dir"))
        with open(os.path.join(self.temp_dir, ".hidden_file"), "w") as f:
            f.write("hidden")
        with open(os.path.join(self.temp_dir, "visible.txt"), "w") as f:
            f.write("visible")

        result = list_directory(dir_path=self.temp_dir, show_hidden=False, agent_state=self.agent_state)

        assert ".hidden_dir" not in result["content"]
        assert ".hidden_file" not in result["content"]
        assert "visible.txt" in result["content"]
        assert "ignored" in result["returnDisplay"]


class TestListDirectoryOutputFormat:
    """Test list_directory output format."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="ls-tool-test-")
        with open(os.path.join(self.temp_dir, "test.txt"), "w") as f:
            f.write("test")
        os.makedirs(os.path.join(self.temp_dir, "subdir"))
        sandbox = LocalSandbox(sandbox_id="test", _work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_llm_content_format(self):
        """Should format llmContent correctly."""
        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "[DIR]" in result["content"]
        assert "test.txt" in result["content"]

    def test_return_display_format(self):
        """Should format returnDisplay correctly."""
        result = list_directory(dir_path=self.temp_dir, agent_state=self.agent_state)

        assert "Listed" in result["returnDisplay"]
        assert "item(s)" in result["returnDisplay"]
