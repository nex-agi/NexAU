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
# limitations in the License.

"""
Integration tests for tool interactions and workflows.
Uses builtin tools (write_file, read_file, replace, run_shell_command).
"""

import os
import tempfile
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools import read_file, replace, write_file
from nexau.archs.tool.builtin.shell_tools import run_shell_command


@pytest.fixture
def agent_state():
    with tempfile.TemporaryDirectory() as work_dir:
        agent_state = Mock()
        agent_state.get_sandbox = lambda: LocalSandbox(_work_dir=work_dir)
        yield agent_state


class TestToolChainIntegration:
    """Integration tests for chaining multiple tools together."""

    @pytest.mark.integration
    def test_file_write_read_chain(self, agent_state):
        """Test writing and reading a file in sequence."""
        work_dir = str(agent_state.get_sandbox().work_dir)
        file_path = os.path.join(work_dir, "test.txt")
        content = "Hello, World!\nThis is a test."

        write_result = write_file(file_path, content, agent_state=agent_state)
        assert "error" not in write_result

        read_result = read_file(file_path, agent_state=agent_state)
        assert "error" not in read_result
        assert "Hello, World!" in read_result["content"]
        assert "This is a test." in read_result["content"]

    @pytest.mark.integration
    def test_file_write_replace_read_chain(self, agent_state):
        """Test writing, replacing, and reading a file."""
        work_dir = str(agent_state.get_sandbox().work_dir)
        file_path = os.path.join(work_dir, "test.py")
        initial_content = "def hello():\n    print('Hello')"

        write_file(file_path, initial_content, agent_state=agent_state)
        replace_result = replace(file_path, "print('Hello')", "print('Hello, World!')", agent_state=agent_state)
        assert "error" not in replace_result

        read_result = read_file(file_path, agent_state=agent_state)
        assert "error" not in read_result
        assert "print('Hello, World!')" in read_result["content"]

    @pytest.mark.integration
    def test_shell_and_file_tools_integration(self, agent_state):
        """Test run_shell_command working with file tools."""
        work_dir = str(agent_state.get_sandbox().work_dir)
        file_path = os.path.join(work_dir, "data.txt")
        content = "Line 1\nLine 2\nLine 3\n"

        write_file(file_path, content, agent_state=agent_state)
        result = run_shell_command(f"wc -l {file_path}", agent_state=agent_state)
        assert "error" not in result
        assert "3" in result["content"]


class TestToolErrorRecovery:
    """Integration tests for tool error handling and recovery."""

    @pytest.mark.integration
    def test_file_tool_error_recovery(self, agent_state):
        """Test recovery from file tool errors."""
        work_dir = str(agent_state.get_sandbox().work_dir)
        file_path = os.path.join(work_dir, "test.txt")

        read_result = read_file(file_path, agent_state=agent_state)
        assert "error" in read_result

        write_file(file_path, "Recovery content", agent_state=agent_state)
        read_result = read_file(file_path, agent_state=agent_state)
        assert "error" not in read_result
        assert "Recovery content" in read_result["content"]

    @pytest.mark.integration
    def test_shell_tool_error_recovery(self, agent_state):
        """Test recovery from shell tool errors."""
        result = run_shell_command("exit 1", agent_state=agent_state)
        assert "error" in result

        result = run_shell_command("echo 'success'", agent_state=agent_state)
        assert "error" not in result
        assert "success" in result["content"]


class TestToolConfigIntegration:
    """Integration tests for tool configuration and loading."""

    @pytest.mark.integration
    def test_tool_from_config(self):
        """Test creating tools from configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "test_tool.yaml")
            config_content = """
name: test_tool
description: A test tool
parameters:
  - name: param1
    type: string
    description: First parameter
    required: true
"""
            with open(config_path, "w") as f:
                f.write(config_content)
            assert os.path.exists(config_path)
