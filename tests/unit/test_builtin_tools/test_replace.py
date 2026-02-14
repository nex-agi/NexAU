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

"""Unit tests for replace (edit) builtin tool."""

import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools.replace import replace


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestReplace:
    """Test replace tool functionality."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        """Create a temporary directory with test files."""
        self.temp_dir = tempfile.mkdtemp(prefix="replace-tool-test-")
        self.sandbox = LocalSandbox(sandbox_id="test", _work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(self.sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_exact_replacement(self):
        """Should replace exact string in file."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("hello world\nfoo bar\nhello again\n")
        result = replace(
            file_path=path,
            old_string="hello world",
            new_string="hi universe",
            agent_state=self.agent_state,
        )
        assert "Successfully modified" in result["content"]
        assert result["returnDisplay"]["occurrences"] == 1
        assert result["returnDisplay"]["strategy"] == "exact"
        with open(path) as f:
            assert "hi universe" in f.read()
            assert "hello world" not in f.read()

    def test_create_new_file(self):
        """Should create new file when old_string is empty and file does not exist."""
        path = os.path.join(self.temp_dir, "new_file.txt")
        result = replace(
            file_path=path,
            old_string="",
            new_string="new content here",
            agent_state=self.agent_state,
        )
        assert "Created new file" in result["content"]
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "new content here"

    def test_file_not_found(self):
        """Should return error when file does not exist and not creating new file."""
        path = os.path.join(self.temp_dir, "nonexistent.txt")
        result = replace(
            file_path=path,
            old_string="something",
            new_string="other",
            agent_state=self.agent_state,
        )
        assert "File not found" in result["content"]
        assert "error" in result
        assert result["error"]["type"] == "FILE_NOT_FOUND"

    def test_create_existing_file_error(self):
        """Should return error when trying to create file that already exists."""
        path = os.path.join(self.temp_dir, "existing.txt")
        with open(path, "w") as f:
            f.write("exists")
        result = replace(
            file_path=path,
            old_string="",
            new_string="new",
            agent_state=self.agent_state,
        )
        assert "already exists" in result["content"]
        assert result["error"]["type"] == "ATTEMPT_TO_CREATE_EXISTING_FILE"

    def test_no_occurrence_found(self):
        """Should return error when old_string not found."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("hello\n")
        result = replace(
            file_path=path,
            old_string="xyz_not_in_file",
            new_string="replacement",
            agent_state=self.agent_state,
        )
        assert "could not find" in result["content"].lower() or "0 occurrences" in result["content"]
        assert result["error"]["type"] == "EDIT_NO_OCCURRENCE_FOUND"

    def test_flexible_replacement(self):
        """Should use flexible (whitespace-insensitive) replacement when exact fails."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("  def foo():\n    return 1\n")
        result = replace(
            file_path=path,
            old_string="def foo():\nreturn 1",
            new_string="def bar():\nreturn 2",
            agent_state=self.agent_state,
        )
        assert "Successfully modified" in result["content"]
        assert result["returnDisplay"]["strategy"] == "flexible"
        with open(path) as f:
            content = f.read()
            assert "bar" in content
            assert "return 2" in content

    def test_multiple_replacements(self):
        """Should replace all occurrences when expected_replacements allows."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("foo\nfoo\nfoo\n")
        result = replace(
            file_path=path,
            old_string="foo",
            new_string="bar",
            expected_replacements=3,
            agent_state=self.agent_state,
        )
        assert result["returnDisplay"]["occurrences"] == 3
        with open(path) as f:
            assert f.read().count("bar") == 3

    def test_expected_occurrence_mismatch(self):
        """Should return error when occurrences != expected_replacements."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("foo\nfoo\n")
        result = replace(
            file_path=path,
            old_string="foo",
            new_string="bar",
            expected_replacements=3,
            agent_state=self.agent_state,
        )
        assert "expected" in result["content"].lower()
        assert result["error"]["type"] == "EDIT_EXPECTED_OCCURRENCE_MISMATCH"

    def test_no_change_old_equals_new(self):
        """Should return error when old_string and new_string are identical."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("same\n")
        result = replace(
            file_path=path,
            old_string="same",
            new_string="same",
            agent_state=self.agent_state,
        )
        assert "identical" in result["content"].lower()
        assert result["error"]["type"] == "EDIT_NO_CHANGE"

    def test_modified_by_user_appends_message(self):
        """Should append user modification note when modified_by_user is True."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "w") as f:
            f.write("old\n")
        result = replace(
            file_path=path,
            old_string="old",
            new_string="new",
            modified_by_user=True,
            agent_state=self.agent_state,
        )
        assert "User modified" in result["content"]

    def test_crlf_file_replacement(self):
        """Should successfully replace in file with CRLF line endings."""
        path = os.path.join(self.temp_dir, "file.txt")
        with open(path, "wb") as f:
            f.write(b"line1\r\nline2\r\n")
        result = replace(
            file_path=path,
            old_string="line1",
            new_string="replaced",
            agent_state=self.agent_state,
        )
        assert result["returnDisplay"]["occurrences"] == 1
        with open(path) as f:
            content = f.read()
            assert "replaced" in content
            assert "line2" in content
