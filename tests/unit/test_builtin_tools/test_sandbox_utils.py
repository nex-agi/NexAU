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

"""Unit tests for _sandbox_utils."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox import SandboxError
from nexau.archs.tool.builtin._sandbox_utils import get_sandbox, resolve_path


class TestGetSandbox:
    """Test get_sandbox helper."""

    def test_returns_sandbox_when_agent_state_has_sandbox(self):
        """Should return sandbox from agent_state.get_sandbox()."""
        mock_sandbox = Mock()
        agent_state = Mock()
        agent_state.get_sandbox.return_value = mock_sandbox
        assert get_sandbox(agent_state) is mock_sandbox

    def test_raises_when_agent_state_is_none(self):
        """Should raise SandboxError when agent_state is None."""
        with pytest.raises(SandboxError, match="Sandbox not found"):
            get_sandbox(None)

    def test_raises_when_get_sandbox_returns_none(self):
        """Should raise SandboxError when get_sandbox returns None."""
        agent_state = Mock()
        agent_state.get_sandbox.return_value = None
        with pytest.raises(SandboxError, match="Sandbox not found"):
            get_sandbox(agent_state)


class TestResolvePath:
    """Test resolve_path helper."""

    def test_absolute_path_unchanged(self):
        """Should return absolute path as-is."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        assert resolve_path("/absolute/path", sandbox) == "/absolute/path"

    def test_relative_path_resolved_against_work_dir(self):
        """Should resolve relative path against sandbox work_dir."""
        sandbox = Mock()
        sandbox.work_dir = Path("/tmp/work")
        resolved = resolve_path("sub/file.txt", sandbox)
        assert resolved == str(Path("/tmp/work") / "sub" / "file.txt")

    def test_empty_relative_path_resolves_to_work_dir(self):
        """Should resolve empty path to work_dir."""
        sandbox = Mock()
        sandbox.work_dir = Path("/home/user")
        resolved = resolve_path("", sandbox)
        assert resolved == str(Path("/home/user"))
