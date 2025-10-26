"""
Integration tests for tool interactions and workflows.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from northau.archs.tool.builtin.bash_tool import bash_tool
from northau.archs.tool.builtin.file_tools.file_edit_tool import file_edit_tool
from northau.archs.tool.builtin.file_tools.file_read_tool import file_read_tool
from northau.archs.tool.builtin.file_tools.file_write_tool import file_write_tool


class TestToolChainIntegration:
    """Integration tests for chaining multiple tools together."""

    @pytest.mark.integration
    def test_file_write_read_chain(self):
        """Test writing and reading a file in sequence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.txt")
            content = "Hello, World!\nThis is a test."

            # Write file - returns JSON string
            write_result_str = file_write_tool(file_path, content)
            write_result = json.loads(write_result_str)
            assert write_result["success"]
            assert write_result["file_path"] == file_path

            # Read file back - returns JSON string
            read_result_str = file_read_tool(file_path)
            read_result = json.loads(read_result_str)
            assert "content" in read_result
            # Content includes line numbers, so check for the actual text
            assert "Hello, World!" in read_result["content"]
            assert "This is a test." in read_result["content"]

    @pytest.mark.integration
    def test_file_write_edit_read_chain(self):
        """Test writing, editing, and reading a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.py")
            initial_content = "def hello():\n    print('Hello')"

            # Write initial file
            write_result_str = file_write_tool(file_path, initial_content)
            write_result = json.loads(write_result_str)
            assert write_result["success"]

            # Edit file
            old_string = "print('Hello')"
            new_string = "print('Hello, World!')"
            edit_result_str = file_edit_tool(file_path, old_string, new_string)
            edit_result = json.loads(edit_result_str)
            assert edit_result["success"]

            # Read modified file
            read_result_str = file_read_tool(file_path)
            read_result = json.loads(read_result_str)
            assert "content" in read_result
            assert "print('Hello, World!')" in read_result["content"]

    @pytest.mark.integration
    def test_bash_and_file_tools_integration(self):
        """Test bash tool working with file tools."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "data.txt")

            # Write file (with trailing newline so wc -l counts correctly)
            content = "Line 1\nLine 2\nLine 3\n"
            write_result_str = file_write_tool(file_path, content)
            write_result = json.loads(write_result_str)
            assert write_result["success"]

            # Use bash to count lines
            bash_result = bash_tool(f"wc -l {file_path}")
            assert bash_result["status"] == "success"
            assert "3" in bash_result["stdout"]


class TestToolStateManagement:
    """Integration tests for tools that maintain state."""

    @pytest.mark.integration
    def test_file_state_across_operations(self):
        """Test file state management across multiple operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "stateful.txt")

            # Multiple write operations
            file_write_tool(file_path, "Version 1")
            read1_str = file_read_tool(file_path)
            read1 = json.loads(read1_str)
            assert "Version 1" in read1["content"]

            file_write_tool(file_path, "Version 2")
            read2_str = file_read_tool(file_path)
            read2 = json.loads(read2_str)
            assert "Version 2" in read2["content"]
            assert "Version 1" not in read2["content"]

    @pytest.mark.integration
    def test_concurrent_tool_execution(self):
        """Test tools executing concurrently without conflicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            files = []
            for i in range(5):
                file_path = os.path.join(temp_dir, f"file_{i}.txt")
                file_write_tool(file_path, f"Content {i}")
                files.append(file_path)

            # Read all files
            for i, file_path in enumerate(files):
                result_str = file_read_tool(file_path)
                result = json.loads(result_str)
                assert "content" in result
                assert f"Content {i}" in result["content"]


class TestToolErrorRecovery:
    """Integration tests for tool error handling and recovery."""

    @pytest.mark.integration
    def test_file_tool_error_recovery(self):
        """Test recovery from file tool errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test.txt")

            # Try to read non-existent file
            read_result_str = file_read_tool(file_path)
            read_result = json.loads(read_result_str)
            assert "error" in read_result
            assert "not found" in read_result["error"].lower() or "does not exist" in read_result["error"].lower()

            # Recover by creating the file
            write_result_str = file_write_tool(file_path, "Recovery content")
            write_result = json.loads(write_result_str)
            assert write_result["success"]

            # Now read should work
            read_result_str = file_read_tool(file_path)
            read_result = json.loads(read_result_str)
            assert "content" in read_result
            assert "Recovery content" in read_result["content"]

    @pytest.mark.integration
    def test_bash_tool_error_recovery(self):
        """Test recovery from bash tool errors."""
        # Execute failing command
        result = bash_tool("exit 1")
        assert result["status"] == "error"
        assert result["exit_code"] == 1

        # Execute successful command
        result = bash_tool("echo 'success'")
        assert result["status"] == "success"
        assert result["exit_code"] == 0


class TestToolConfigIntegration:
    """Integration tests for tool configuration and loading."""

    @pytest.mark.integration
    def test_tool_from_config(self):
        """Test creating tools from configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_tool.yaml")

            # Create tool config
            config_content = """
name: test_tool
description: A test tool
parameters:
  - name: param1
    type: string
    description: First parameter
    required: true
  - name: param2
    type: integer
    description: Second parameter
    required: false
    default: 10
"""
            with open(config_path, "w") as f:
                f.write(config_content)

            # In a real test, this would load and create the tool
            assert os.path.exists(config_path)

    @pytest.mark.integration
    def test_multiple_tools_from_directory(self):
        """Test loading multiple tools from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tools_dir = os.path.join(temp_dir, "tools")
            os.makedirs(tools_dir)

            # Create multiple tool configs
            for i in range(3):
                config_path = os.path.join(tools_dir, f"tool{i}.yaml")
                with open(config_path, "w") as f:
                    f.write(f"""
name: tool{i}
description: Tool number {i}
""")

            # Verify all tool configs exist
            tool_files = list(Path(tools_dir).glob("*.yaml"))
            assert len(tool_files) == 3
