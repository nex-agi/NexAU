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

"""
Unit tests for file manipulation tools.
"""

import json
import os
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools.file_edit_tool import file_edit_tool
from nexau.archs.tool.builtin.file_tools.file_read_tool import file_read_tool
from nexau.archs.tool.builtin.file_tools.file_write_tool import file_write_tool
from nexau.archs.tool.builtin.file_tools.glob_tool import glob_tool
from nexau.archs.tool.builtin.file_tools.grep_tool import grep_tool
from nexau.archs.tool.builtin.ls_tool import ls_tool


@pytest.fixture
def agent_state():
    agent_state = Mock()
    agent_state.get_sandbox = lambda: LocalSandbox()
    return agent_state


class TestFileEditTool:
    """Test cases for file editing functionality."""

    def test_create_new_file(self, temp_dir, agent_state):
        """Test creating a new file."""
        file_path = os.path.join(temp_dir, "new_file.py")
        old_string = ""
        new_string = 'print("Hello, World!")'

        result = file_edit_tool(file_path, old_string, new_string, agent_state=agent_state)

        # Parse JSON result
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["operation"] == "create"
        assert result_data["file_path"] == file_path
        assert result_data["num_lines"] == 1

        # Verify file was created
        assert os.path.exists(file_path)
        with open(file_path) as f:
            content = f.read()
        assert content == new_string

    def test_update_existing_file(self, temp_file, agent_state):
        """Test updating an existing file."""

        old_content = "test content"  # This is what the fixture writes
        new_content = 'print("Hello, Python!")'

        result = file_edit_tool(temp_file, old_content, new_content, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["operation"] == "update"

        # Verify file was updated
        with open(temp_file) as f:
            content = f.read()
        assert content == new_content

    def test_remove_content(self, temp_file, agent_state):
        """Test removing content from file."""
        from nexau.archs.sandbox import LocalSandbox
        from nexau.archs.tool.builtin.file_tools.file_edit_tool import mark_file_as_read

        original_content = "line1\nline2\nline3"
        with open(temp_file, "w") as f:
            f.write(original_content)

        # Get sandbox and mark file as read after modification
        sandbox = LocalSandbox()
        mark_file_as_read(temp_file, sandbox)

        result = file_edit_tool(temp_file, "line2\n", "", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is True

        # Verify content was removed
        with open(temp_file) as f:
            content = f.read()
        assert content == "line1\nline3"

    def test_relative_path_error(self, agent_state):
        """Test error handling for relative paths."""
        result = file_edit_tool("relative/path.py", "", "content", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "absolute path" in result_data["error"]

    def test_nonexistent_file_error(self, agent_state):
        """Test error handling for nonexistent files."""
        nonexistent_path = "/tmp/nonexistent_file_12345.py"
        result = file_edit_tool(nonexistent_path, "old", "new", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "does not exist" in result_data["error"]

    def test_create_existing_file_error(self, temp_file, agent_state):
        """Test error when trying to create existing file."""
        result = file_edit_tool(temp_file, "", "content", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "already exists" in result_data["error"]

    def test_multiple_matches_error(self, temp_file, agent_state):
        """Test error when old_string matches multiple times."""
        from nexau.archs.sandbox import LocalSandbox
        from nexau.archs.tool.builtin.file_tools.file_edit_tool import mark_file_as_read

        content = "line\nline\nline"
        with open(temp_file, "w") as f:
            f.write(content)

        # Get sandbox and mark file as read after modification
        sandbox = LocalSandbox()
        mark_file_as_read(temp_file, sandbox)

        result = file_edit_tool(temp_file, "line", "replacement", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is False
        # Check for the "matches" keyword in the error (the exact message says "Found 3 matches")
        assert "matches" in result_data["error"].lower()
        assert "3 matches" in result_data["error"].lower() or "multiple" in result_data["error"].lower()

    def test_no_change_error(self, temp_file, agent_state):
        """Test error when old_string equals new_string."""
        content = "test content"
        with open(temp_file, "w") as f:
            f.write(content)

        result = file_edit_tool(temp_file, content, content, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is False
        assert "no changes" in result_data["error"].lower()


class TestFileReadTool:
    """Test cases for file reading functionality."""

    def test_read_text_file(self, temp_file, agent_state):
        """Test reading a text file."""
        content = "line 1\nline 2\nline 3"
        with open(temp_file, "w") as f:
            f.write(content)

        result = file_read_tool(temp_file, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["type"] == "text"
        assert result_data["success"] is True  # Assuming the JSON parsing means it worked
        assert "line 1" in result_data["content"]

    def test_read_with_offset_and_limit(self, temp_file, agent_state):
        """Test reading with offset and limit."""
        content = "\n".join([f"line {i}" for i in range(1, 11)])
        with open(temp_file, "w") as f:
            f.write(content)

        result = file_read_tool(temp_file, offset=3, limit=5, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["start_line"] == 3
        assert result_data["lines_read"] == 5
        assert "line 3" in result_data["content"]
        assert "line 7" in result_data["content"]
        assert "line 8" not in result_data["content"]

    def test_read_nonexistent_file(self, agent_state):
        """Test reading nonexistent file."""
        result = file_read_tool("/tmp/nonexistent_file_12345.py", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["error"] is not None
        assert "does not exist" in result_data["error"]

    def test_read_directory_error(self, temp_dir, agent_state):
        """Test error when trying to read directory."""
        result = file_read_tool(temp_dir, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["error"] is not None
        assert "directory" in result_data["error"].lower()

    def test_read_large_file_truncation(self, temp_file, agent_state):
        """Test large file content truncation."""
        # Create a large content that exceeds MAX_OUTPUT_SIZE
        large_content = "x" * (1024 * 1024)  # 1MB of content
        with open(temp_file, "w") as f:
            f.write(large_content)

        result = file_read_tool(temp_file, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["error"] is not None
        assert "exceeds maximum" in result_data["error"].lower()


class TestFileWriteTool:
    """Test cases for file writing functionality."""

    def test_write_new_file(self, temp_dir, agent_state):
        """Test writing to a new file."""
        file_path = os.path.join(temp_dir, "new_file.txt")
        content = "This is new content\nwith multiple lines"

        result = file_write_tool(file_path, content, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["operation_type"] == "create"
        assert result_data["num_lines"] == 2

        # Verify file was created
        assert os.path.exists(file_path)
        with open(file_path) as f:
            written_content = f.read()
        assert written_content == content

    def test_write_existing_file(self, temp_file, agent_state):
        """Test writing to existing file."""
        from nexau.archs.sandbox import LocalSandbox
        from nexau.archs.tool.builtin.file_tools.file_edit_tool import mark_file_as_read

        original_content = "original content"
        new_content = "updated content"

        # Write original content
        with open(temp_file, "w") as f:
            f.write(original_content)

        # Get sandbox and mark file as read after modification
        sandbox = LocalSandbox()
        mark_file_as_read(temp_file, sandbox)

        result = file_write_tool(temp_file, new_content, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["operation_type"] == "update"
        assert result_data["has_changes"] is True

        # Verify file was updated
        with open(temp_file) as f:
            written_content = f.read()
        assert written_content == new_content

    def test_write_no_changes(self, temp_file, agent_state):
        """Test writing identical content."""
        from nexau.archs.sandbox import LocalSandbox
        from nexau.archs.tool.builtin.file_tools.file_edit_tool import mark_file_as_read

        content = "same content"
        with open(temp_file, "w") as f:
            f.write(content)

        # Get sandbox and mark file as read after modification
        sandbox = LocalSandbox()
        mark_file_as_read(temp_file, sandbox)

        result = file_write_tool(temp_file, content, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["success"] is True
        assert result_data["has_changes"] is False


class TestGlobTool:
    """Test cases for glob pattern matching."""

    def test_glob_python_files(self, temp_dir, agent_state):
        """Test globbing Python files."""
        # Create test files
        test_files = ["test1.py", "test2.py", "script.js", "data.txt"]
        for filename in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(f"content of {filename}")

        result = glob_tool("*.py", temp_dir, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["num_files"] == 2
        assert len(result_data["filenames"]) == 2
        assert any("test1.py" in f for f in result_data["filenames"])
        assert any("test2.py" in f for f in result_data["filenames"])

    def test_glob_with_limit(self, temp_dir, agent_state):
        """Test globbing with result limit."""
        # Create many test files
        for i in range(10):
            file_path = os.path.join(temp_dir, f"file_{i}.txt")
            with open(file_path, "w") as f:
                f.write(f"content {i}")

        result = glob_tool("*.txt", temp_dir, limit=3, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["num_files"] == 3
        assert result_data["truncated"] is True

    def test_glob_nonexistent_directory(self, agent_state):
        """Test globbing in nonexistent directory."""
        result = glob_tool("*.py", "/nonexistent/directory", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["error"] is not None
        assert "does not exist" in result_data["error"]


class TestLSTool:
    """Test cases for directory listing."""

    def test_ls_directory(self, temp_dir, agent_state):
        """Test listing directory contents."""
        # Create test files and directories
        test_items = ["file1.txt", "file2.py", "script.js", "subdir1/file.txt", "subdir2/script.py"]

        for item in test_items:
            item_path = os.path.join(temp_dir, item)
            os.makedirs(os.path.dirname(item_path), exist_ok=True)
            with open(item_path, "w") as f:
                f.write(f"content of {item}")

        result = ls_tool(temp_dir, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["total_items"] == 5  # 3 files + 2 directories at top level
        assert result_data["files"] == 3
        assert result_data["directories"] == 2

    def test_ls_with_ignore_patterns(self, temp_dir, agent_state):
        """Test listing with ignore patterns."""
        # Create test files
        test_files = ["file1.py", "file2.pyc", "script.js", ".hidden"]
        for filename in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(f"content of {filename}")

        result = ls_tool(temp_dir, ignore=["*.pyc", ".hidden"], agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["total_items"] == 2  # Should ignore .pyc and .hidden files
        assert result_data["ignored_items"] == 2

    def test_ls_nonexistent_directory(self, agent_state):
        """Test listing nonexistent directory."""
        result = ls_tool("/nonexistent/directory", agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["status"] == "error"
        assert "does not exist" in result_data["error"]

    def test_ls_file_instead_of_directory(self, temp_file, agent_state):
        """Test listing file instead of directory."""
        result = ls_tool(temp_file, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["status"] == "error"
        assert "not a directory" in result_data["error"]


class TestGrepTool:
    """Test cases for grep search functionality."""

    def test_grep_search_in_files(self, temp_dir, agent_state):
        """Test searching for text in files."""
        # Create test files with different content
        test_files = [
            ("search1.py", 'import os\ndef search_function():\n    return "found"'),
            ("search2.py", 'def another_function():\n    return "not found"'),
            ("data.txt", "some random text without search terms"),
        ]

        for filename, content in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        result = grep_tool("import", temp_dir, agent_state=agent_state)
        result_data = json.loads(result)

        # Skip test if ripgrep is not installed
        if "error" in result_data and "ripgrep" in result_data["error"]:
            pytest.skip("ripgrep (rg) is not installed or not available in PATH")

        assert result_data["num_files"] == 1
        assert any("search1.py" in f for f in result_data["filenames"])

    def test_grep_with_file_pattern(self, temp_dir, agent_state):
        """Test searching with file pattern filter."""
        # Create files with search terms
        test_files = [
            ("test1.py", 'print("python")'),
            ("test2.js", 'console.log("javascript")'),
            ("test3.py", "import os"),
        ]

        for filename, content in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, "w") as f:
                f.write(content)

        result = grep_tool("print", temp_dir, glob="*.py", agent_state=agent_state)
        result_data = json.loads(result)

        # Skip test if ripgrep is not installed
        if "error" in result_data and "ripgrep" in result_data["error"]:
            pytest.skip("ripgrep (rg) is not installed or not available in PATH")

        assert result_data["num_files"] == 1
        assert "test1.py" in result_data["filenames"][0]

    def test_grep_no_matches(self, temp_dir, agent_state):
        """Test searching for non-existent pattern."""
        # Create a test file
        file_path = os.path.join(temp_dir, "test.py")
        with open(file_path, "w") as f:
            f.write("some content")

        result = grep_tool("nonexistent_pattern", temp_dir, agent_state=agent_state)
        result_data = json.loads(result)

        assert result_data["num_files"] == 0


# Integration tests for file tools
class TestFileToolsIntegration:
    """Integration tests for file tools working together."""

    def test_read_edit_write_cycle(self, temp_dir, agent_state):
        """Test the complete cycle of reading, editing, and writing."""
        # Create initial file
        file_path = os.path.join(temp_dir, "cycle_test.py")
        initial_content = 'def old_function():\n    return "old"'

        with open(file_path, "w") as f:
            f.write(initial_content)

        # Read file
        read_result = file_read_tool(file_path, agent_state=agent_state)
        read_data = json.loads(read_result)
        assert read_data["type"] == "text"

        # Edit file
        edit_result = file_edit_tool(file_path, "old_function", "new_function", agent_state=agent_state)
        edit_data = json.loads(edit_result)
        assert edit_data["success"] is True

        # Verify changes
        with open(file_path) as f:
            final_content = f.read()

        assert "new_function" in final_content
        assert "old_function" not in final_content

    def test_file_operations_with_metadata(self, temp_file, agent_state):
        """Test file operations preserve metadata."""
        from nexau.archs.sandbox import LocalSandbox
        from nexau.archs.tool.builtin.file_tools.file_edit_tool import mark_file_as_read

        initial_stat = os.stat(temp_file)

        # Mark file as read first (temp_file fixture creates a file with content)
        sandbox = LocalSandbox()
        mark_file_as_read(temp_file, sandbox)

        # Write content
        new_content = "new content with\nmultiple lines"
        write_result = file_write_tool(temp_file, new_content, agent_state=agent_state)
        write_data = json.loads(write_result)
        assert write_data["success"] is True

        # Verify content and metadata
        with open(temp_file) as f:
            content = f.read()
        assert content == new_content

        # Check if modification time was updated
        final_stat = os.stat(temp_file)
        assert final_stat.st_mtime >= initial_stat.st_mtime


# Performance tests
class TestFileToolsPerformance:
    """Performance tests for file tools."""

    def test_large_file_handling(self, temp_dir, agent_state):
        """Test handling of large files."""
        file_path = os.path.join(temp_dir, "large_file.txt")

        # Create a large file (1MB)
        large_content = "x" * (1024 * 1024)
        with open(file_path, "w") as f:
            f.write(large_content)

        # Test reading with limits
        result = file_read_tool(file_path, offset=1, limit=10, agent_state=agent_state)
        result_data = json.loads(result)

        # Should handle large files gracefully
        if "error" in result_data:
            assert "exceeds maximum" in result_data["error"].lower()
        else:
            assert result_data["lines_read"] <= 10
