"""
Unit tests for bash tool functionality.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from northau.archs.tool.builtin.bash_tool import BashTool, bash_tool


class TestBashTool:
    """Test cases for bash tool functionality."""

    def test_bash_tool_success(self):
        """Test successful bash command execution."""
        result = bash_tool("echo 'Hello World'")

        assert result["status"] == "success"
        assert result["command"] == "echo 'Hello World'"
        assert "Hello World" in result["stdout"]
        assert result["exit_code"] == 0
        assert "duration_ms" in result
        assert result["duration_ms"] >= 0

    def test_bash_tool_with_description(self):
        """Test bash command with description."""
        result = bash_tool("echo 'test'", description="Test echo command")

        assert result["status"] == "success"
        assert result["description"] == "Test echo command"

    def test_bash_tool_error(self):
        """Test bash command that returns error."""
        result = bash_tool("ls /nonexistent_directory_12345")

        assert result["status"] == "error"
        assert result["exit_code"] != 0
        assert "No such file or directory" in result["stderr"]

    def test_bash_tool_timeout(self):
        """Test bash command timeout."""
        result = bash_tool("sleep 2", timeout=500)  # 500ms timeout

        assert result["status"] == "timeout"
        # Exit code can be -9 (SIGKILL) or -1 or None depending on the system
        assert result["exit_code"] in [None, -1, -9]

    def test_bash_tool_empty_command(self):
        """Test empty command handling."""
        result = bash_tool("")

        assert result["status"] == "error"
        assert "Command cannot be empty" in result["error"]

    def test_bash_tool_dangerous_command(self):
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
            result = bash_tool(command)
            assert result["status"] == "error"
            assert "dangerous" in result["error"].lower()

    def test_bash_tool_working_directory(self, temp_dir):
        """Test working directory handling."""
        test_file = os.path.join(temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        # Use absolute path since bash_tool uses workspace from agent_state or cwd
        result = bash_tool(f"ls {test_file}", description="List test file")

        assert result["status"] == "success"
        assert "test.txt" in result["stdout"]
        # The working_directory will be the current working directory, not temp_dir
        assert "working_directory" in result

    def test_bash_tool_permission_error(self):
        """Test permission error handling."""
        # Try to write to a read-only location
        result = bash_tool("touch /root/test_file")

        # Should either succeed (if running as root) or fail with permission/access error
        if result["status"] == "error":
            # Permission errors appear in stderr, could be "Permission denied" or "No such file or directory"
            stderr = result.get("stderr", "")
            assert "Permission denied" in stderr or "No such file or directory" in stderr

    def test_bash_tool_long_output_truncation(self):
        """Test long output truncation."""
        # Generate long output
        result = bash_tool("python -c 'print(\"x\" * 50000)'")

        assert result["status"] == "success"
        assert len(result["stdout"]) <= 30000  # MAX_OUTPUT_LENGTH
        assert result["stdout_truncated"] is True
        # print adds a newline, so output is 50001 characters
        assert result["stdout_original_length"] >= 50000

    def test_bash_tool_stderr_capture(self):
        """Test stderr capture."""
        # Don't redirect stderr to stdout with 2>&1, so stderr is captured properly
        result = bash_tool("ls /nonexistent", description="Test stderr capture")

        assert result["status"] == "error"
        assert result["stderr"]
        assert "No such file or directory" in result["stderr"]


class TestBashToolClass:
    """Test cases for BashTool class."""

    def test_bash_tool_class_initialization(self):
        """Test BashTool class initialization."""
        tool = BashTool()

        assert tool.max_output_length == 30000
        assert tool.default_timeout == 120000

    def test_bash_tool_class_execute(self, temp_dir):
        """Test BashTool execute method."""
        tool = BashTool()
        test_file = os.path.join(temp_dir, "class_test.txt")

        result = tool.execute(f"echo 'class test' > {test_file}")

        assert result["status"] == "success"
        assert os.path.exists(test_file)

        with open(test_file) as f:
            content = f.read()
        assert content.strip() == "class test"

    def test_bash_tool_class_multiple_commands(self):
        """Test executing multiple commands."""
        tool = BashTool()

        commands = ["echo 'command 1'", "echo 'command 2'", "echo 'command 3'"]

        results = tool.execute_multiple(commands)

        assert len(results) == 3
        assert all(result["status"] == "success" for result in results)

        for i, result in enumerate(results):
            assert f"command {i + 1}" in result["stdout"]

    def test_bash_tool_class_stop_on_error(self):
        """Test stopping on first error."""
        tool = BashTool()

        commands = [
            "echo 'success'",
            "false",  # This will fail
            "echo 'should not reach'",
        ]

        results = tool.execute_multiple(commands)

        assert len(results) == 2  # Should stop after first error
        assert results[0]["status"] == "success"
        assert results[1]["status"] == "error"

    def test_bash_tool_class_with_custom_timeout(self):
        """Test BashTool with custom timeout."""
        tool = BashTool(default_timeout=1000)  # 1 second timeout

        # This should timeout
        result = tool.execute("sleep 2")

        assert result["status"] == "timeout"
        # Allow small overhead for process management (timeout + ~200ms tolerance)
        assert result["duration_ms"] < 2200


class TestBashToolIntegration:
    """Integration tests for bash tool."""

    def test_bash_tool_real_command_execution(self):
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
            result = bash_tool(command)
            assert result["status"] == expected_status, f"Command '{command}' failed: {result}"

    def test_bash_tool_environment_variables(self):
        """Test environment variable handling."""
        # Set a test environment variable
        test_var = "TEST_VAR_12345"

        result = bash_tool(f"echo ${test_var}", description="Test env var")

        # The variable won't be set in the shell, so this should return empty
        assert result["status"] == "success"
        assert result["stdout"].strip() == ""

    def test_bash_tool_path_resolution(self):
        """Test path resolution in commands."""
        # Test with absolute paths
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_file = f.name

        try:
            result = bash_tool(f"cat {temp_file}")
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
