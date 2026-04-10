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

"""Unit tests for run_shell_command builtin tool."""

import threading
from pathlib import Path
from unittest.mock import Mock, patch

from nexau.archs.main_sub.framework_context import FrameworkContext
from nexau.archs.sandbox import SandboxStatus
from nexau.archs.tool.builtin.shell_tools.run_shell_command import (
    _truncate_shell_output,
    run_shell_command,
)


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestTruncateShellOutput:
    """Test _truncate_shell_output helper (keeps last N lines when output exceeds threshold)."""

    def test_returns_unchanged_when_under_threshold(self):
        """Should return content unchanged when under TRUNCATE_OUTPUT_THRESHOLD."""
        content = "line1\nline2\nline3"
        result = _truncate_shell_output(content)
        assert result == content

    @patch(
        "nexau.archs.tool.builtin.shell_tools.run_shell_command.TRUNCATE_OUTPUT_THRESHOLD",
        50,
    )
    def test_truncates_multi_line_keeping_last_n(self):
        """Should keep last TRUNCATE_OUTPUT_LINES when multi-line output exceeds threshold."""
        # Create content > 50 chars with multiple lines
        lines = [f"Line {i}" for i in range(20)]
        content = "\n".join(lines)
        assert len(content) > 50

        result = _truncate_shell_output(content)

        assert "Output too large" in result
        assert "Showing the last" in result
        assert "of 20 lines" in result
        # Last N lines (1000, but we only have 20) should be preserved
        assert "Line 0" not in result or "lines" in result
        assert "Line 19" in result

    @patch(
        "nexau.archs.tool.builtin.shell_tools.run_shell_command.TRUNCATE_OUTPUT_THRESHOLD",
        50,
    )
    def test_truncates_single_massive_line_keeping_last_chars(self):
        """Should keep last MAX_TRUNCATED_CHARS when single massive line exceeds threshold."""
        content = "a" * 5000  # Single line, > 50 chars
        result = _truncate_shell_output(content)

        assert "Output too large" in result
        assert "4,000" in result or "4000" in result
        # Should contain last 4000 chars
        assert result.endswith("a" * 4000)
        assert "a" * 5000 not in result


class TestRunShellCommandIntegration:
    """Test run_shell_command with mocked sandbox (basic flow)."""

    def test_basic_success_returns_output(self):
        """Should return output on successful command."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True
        info = Mock()
        info.is_directory = True
        sandbox.get_file_info.return_value = info

        start_result = Mock()
        start_result.background_pid = 123
        sandbox.execute_bash.return_value = start_result

        cmd_result = Mock()
        cmd_result.stdout = "hello world"
        cmd_result.stderr = ""
        cmd_result.exit_code = 0
        cmd_result.error = None
        cmd_result.status = SandboxStatus.SUCCESS
        cmd_result.output_dir = "/tmp/out"
        cmd_result.stdout_file = "/tmp/out/stdout.txt"
        cmd_result.stderr_file = "/tmp/out/stderr.txt"
        cmd_result.truncated = False
        cmd_result.original_stdout_length = None
        cmd_result.original_stderr_length = None
        sandbox.get_background_task_status.return_value = cmd_result

        agent_state = _make_agent_state(sandbox)
        ctx = FrameworkContext.for_testing()
        result = run_shell_command("echo hello", agent_state=agent_state, ctx=ctx)

        assert "error" not in result or result.get("error") is None
        assert "hello world" in result["content"]
        assert "hello world" in result["returnDisplay"]
        assert result["stdout"] == "hello world"
        assert result["stderr"] == ""
        assert result["interrupted"] is False
        assert result["timed_out"] is False

    @patch("nexau.archs.tool.builtin.shell_tools.run_shell_command.time.sleep", return_value=None)
    def test_stop_request_kills_running_foreground_command(self, _sleep_mock):
        """Should kill the sandbox task when shutdown_event is set during polling."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        sandbox.file_exists.return_value = True
        info = Mock()
        info.is_directory = True
        sandbox.get_file_info.return_value = info

        start_result = Mock()
        start_result.background_pid = 456
        sandbox.execute_bash.return_value = start_result

        running_result = Mock()
        running_result.status = SandboxStatus.RUNNING
        running_result.stdout = "partial output"
        running_result.stderr = ""
        running_result.exit_code = -1
        running_result.error = None
        running_result.output_dir = "/tmp/out"
        running_result.stdout_file = "/tmp/out/stdout.txt"
        running_result.stderr_file = "/tmp/out/stderr.txt"
        running_result.truncated = False
        running_result.original_stdout_length = None
        running_result.original_stderr_length = None

        shutdown_event = threading.Event()

        def _status_side_effect(pid: int):
            assert pid == 456
            shutdown_event.set()
            return running_result

        sandbox.get_background_task_status.side_effect = _status_side_effect

        agent_state = _make_agent_state(sandbox)
        ctx = FrameworkContext.for_testing(shutdown_event=shutdown_event)

        result = run_shell_command("echo hello", agent_state=agent_state, ctx=ctx)

        sandbox.kill_background_task.assert_called_once_with(456)
        assert result["error"]["message"] == "Command interrupted by stop request"
        assert "Interrupted: command stopped due to stop request." in result["content"]
        assert result["returnDisplay"] == "partial output"
        assert result["exit_code"] == -1
        assert result["stdout"] == "partial output"
        assert result["stderr"] == ""
        assert result["interrupted"] is True
        assert result["timed_out"] is False
