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
Unit tests for bash tool functionality.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from nexau.archs.sandbox import LocalSandbox
from nexau.archs.tool.builtin.bash_tool import bash_tool


@pytest.fixture
def agent_state():
    sandbox = LocalSandbox()
    agent_state = Mock()
    agent_state.get_sandbox = lambda: sandbox
    return agent_state


class TestBashTool:
    """Test cases for bash tool functionality."""

    def test_bash_tool_success(self, agent_state):
        """Test successful bash command execution."""
        result = bash_tool("echo 'Hello World'", agent_state=agent_state)

        assert result["status"] == "success"
        assert result["command"] == "echo 'Hello World'"
        assert "Hello World" in result["stdout"]
        assert result["exit_code"] == 0
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    def test_bash_tool_with_description(self, agent_state):
        """Test bash command with description."""
        result = bash_tool("echo 'test'", description="Test echo command", agent_state=agent_state)

        assert result["status"] == "success"
        assert result["description"] == "Test echo command"

    def test_bash_tool_error(self, agent_state):
        """Test bash command that returns error."""
        result = bash_tool("ls /nonexistent_directory_12345", agent_state=agent_state)

        assert result["status"] == "error"
        assert result["exit_code"] != 0
        assert "No such file or directory" in result["stderr"]

    def test_bash_tool_timeout(self, agent_state):
        """Test bash command timeout."""
        result = bash_tool("sleep 2", timeout=500, agent_state=agent_state)  # 500ms timeout

        assert result["status"] == "timeout"
        # Exit code can be -9 (SIGKILL) or -1 or None depending on the system
        assert result["exit_code"] in [None, -1, -9]

    def test_bash_tool_empty_command(self, agent_state):
        """Test empty command handling."""
        result = bash_tool("", agent_state=agent_state)

        assert result["status"] == "error"
        assert "Command cannot be empty" in result["error"]

    def test_bash_tool_dangerous_command(self, agent_state):
        """Test dangerous command blocking."""
        dangerous_commands = [
            "rm -rf /",
            "sudo rm -rf /tmp",
            "rm -rf ~",
            "mkfs /dev/sda",
            "dd if=/dev/zero of=/dev/sda",
            "> /dev/null",
            "shutdown -h now",
        ]

        for command in dangerous_commands:
            result = bash_tool(command, agent_state=agent_state)
            assert result["status"] == "error"
            assert "dangerous" in result["error"].lower()

    def test_bash_tool_working_directory(self, temp_dir, agent_state):
        """Test working directory handling."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Use absolute path since bash_tool uses workspace from agent_state or cwd
        result = bash_tool(f"ls {test_file}", description="List test file", agent_state=agent_state)

        assert result["status"] == "success"
        assert "test.txt" in result["stdout"]
        # The working_directory will be the current working directory, not temp_dir
        assert "working_directory" in result

    def test_bash_tool_permission_error(self, agent_state):
        """Test permission error handling."""
        # Try to write to a read-only location
        result = bash_tool("touch /root/test_file", agent_state=agent_state)

        # Should either succeed (if running as root) or fail with permission/access error
        if result["status"] == "error":
            # Permission errors appear in stderr, could be "Permission denied" or "No such file or directory"
            stderr = result.get("stderr", "")
            assert "Permission denied" in stderr or "No such file or directory" in stderr

    def test_bash_tool_long_output_truncation(self, agent_state):
        """Test long output truncation."""
        # Generate long output
        result = bash_tool("python -c 'print(\"x\" * 50000)'", agent_state=agent_state)

        assert result["status"] == "success"
        assert len(result["stdout"]) <= 30000  # MAX_OUTPUT_LENGTH
        assert result["stdout_truncated"] is True
        # print adds a newline, so output is 50001 characters
        assert result["stdout_original_length"] >= 50000

    def test_bash_tool_stderr_capture(self, agent_state):
        """Test stderr capture."""
        # Don't redirect stderr to stdout with 2>&1, so stderr is captured properly
        result = bash_tool("ls /nonexistent", description="Test stderr capture", agent_state=agent_state)

        assert result["status"] == "error"
        assert result["stderr"]
        assert "No such file or directory" in result["stderr"]


class TestBashToolBackground:
    """Test cases for bash tool background execution."""

    def test_bash_tool_background_returns_pid(self, agent_state):
        """Test background execution returns a pid."""
        result = bash_tool("sleep 10", background=True, agent_state=agent_state)

        assert result["status"] == "success"
        assert "background_pid" in result
        assert result["background_pid"] is not None
        assert result["background_pid"] > 0

        # Cleanup
        sandbox = agent_state.get_sandbox()
        sandbox.kill_background_task(result["background_pid"])

    def test_bash_tool_background_task_tracked_in_sandbox(self, agent_state):
        """Test background task is tracked in sandbox._background_tasks."""
        result = bash_tool("sleep 10", background=True, agent_state=agent_state)

        assert result["status"] == "success"
        pid = result["background_pid"]

        # Verify task is tracked in sandbox
        sandbox = agent_state.get_sandbox()
        bg_tasks = sandbox.list_background_tasks()
        assert pid in bg_tasks
        assert bg_tasks[pid]["command"] == "sleep 10"

        # Cleanup
        sandbox.kill_background_task(pid)

    def test_bash_tool_background_empty_command(self, agent_state):
        """Test background with empty command still fails."""
        result = bash_tool("", background=True, agent_state=agent_state)
        assert result["status"] == "error"
        assert "Command cannot be empty" in result["error"]

    def test_bash_tool_background_dangerous_command(self, agent_state):
        """Test background with dangerous command still blocked."""
        result = bash_tool("rm -rf /", background=True, agent_state=agent_state)
        assert result["status"] == "error"
        assert "dangerous" in result["error"].lower()


class TestBashToolIntegration:
    """Integration tests for bash tool."""

    def test_bash_tool_real_command_execution(self, agent_state):
        """Test actual command execution (integration test)."""
        # Skip if not on Unix-like system
        if os.name == "nt":
            pytest.skip("Bash tool tests require Unix-like system")

        # Test basic commands that should work on any Unix system
        test_commands = [
            ("echo 'test'", "success"),
            ("pwd", "success"),
            ("date", "success"),
            ("whoami", "success"),
        ]

        for command, expected_status in test_commands:
            result = bash_tool(command, agent_state=agent_state)
            assert result["status"] == expected_status, f"Command '{command}' failed: {result}"

    def test_bash_tool_environment_variables(self, agent_state):
        """Test environment variable handling."""
        # Set a test environment variable
        test_var = "TEST_VAR_12345"

        result = bash_tool(f"echo ${test_var}", description="Test env var", agent_state=agent_state)

        # The variable won't be set in the shell, so this should return empty
        assert result["status"] == "success"
        assert result["stdout"].strip() == ""

    def test_bash_tool_path_resolution(self, agent_state):
        """Test path resolution in commands."""
        # Test with absolute paths
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            result = bash_tool(f"cat {temp_file}", agent_state=agent_state)
            assert result["status"] == "success"
            assert "test content" in result["stdout"]
        finally:
            os.unlink(temp_file)


# Fixtures for testing
@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    # Cleanup
    import shutil

    shutil.rmtree(temp_path)


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for testing dangerous commands."""
    with patch("subprocess.Popen") as mock_popen:
        # Mock successful execution
        mock_process = Mock()
        mock_process.communicate.return_value = ("output", "error")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        yield mock_popen
