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

"""Unit tests for the apply_patch builtin tool."""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools.apply_patch import apply_patch


def _make_agent_state(sandbox):
    """Create mock agent_state with sandbox."""
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestApplyPatch:
    """Test Codex-style patch application through NexAU sandbox APIs."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp(prefix="apply-patch-tool-test-")
        self.sandbox = LocalSandbox(sandbox_id="test", work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(self.sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_apply_multiple_operations(self):
        """Should add, update, and delete files in a single patch."""
        modify_path = os.path.join(self.temp_dir, "modify.txt")
        delete_path = os.path.join(self.temp_dir, "delete.txt")
        with open(modify_path, "w", encoding="utf-8") as f:
            f.write("line1\nline2\n")
        with open(delete_path, "w", encoding="utf-8") as f:
            f.write("obsolete\n")

        patch = (
            "*** Begin Patch\n"
            "*** Add File: nested/new.txt\n"
            "+created\n"
            "*** Delete File: delete.txt\n"
            "*** Update File: modify.txt\n"
            "@@\n"
            "-line2\n"
            "+changed\n"
            "*** End Patch"
        )

        result = apply_patch(input=patch, agent_state=self.agent_state)

        assert result["content"] == ("Success. Updated the following files:\nA nested/new.txt\nM modify.txt\nD delete.txt\n")
        with open(os.path.join(self.temp_dir, "nested/new.txt"), encoding="utf-8") as f:
            assert f.read() == "created\n"
        with open(modify_path, encoding="utf-8") as f:
            assert f.read() == "line1\nchanged\n"
        assert not os.path.exists(delete_path)

    def test_move_file_to_new_directory(self):
        """Should support Update File with Move to rename semantics."""
        original_path = os.path.join(self.temp_dir, "old", "name.txt")
        os.makedirs(os.path.dirname(original_path), exist_ok=True)
        with open(original_path, "w", encoding="utf-8") as f:
            f.write("old content\n")

        patch = (
            "*** Begin Patch\n"
            "*** Update File: old/name.txt\n"
            "*** Move to: renamed/dir/name.txt\n"
            "@@\n"
            "-old content\n"
            "+new content\n"
            "*** End Patch"
        )

        result = apply_patch(input=patch, agent_state=self.agent_state)

        assert result["content"] == "Success. Updated the following files:\nM renamed/dir/name.txt\n"
        assert not os.path.exists(original_path)
        with open(os.path.join(self.temp_dir, "renamed", "dir", "name.txt"), encoding="utf-8") as f:
            assert f.read() == "new content\n"

    def test_add_overwrites_existing_file(self):
        """Add File should overwrite an existing file like the official Codex implementation."""
        path = os.path.join(self.temp_dir, "duplicate.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("old content\n")

        patch = "*** Begin Patch\n*** Add File: duplicate.txt\n+new content\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)

        assert result["content"] == "Success. Updated the following files:\nA duplicate.txt\n"
        with open(path, encoding="utf-8") as f:
            assert f.read() == "new content\n"

    def test_reports_missing_context_and_keeps_original_content(self):
        """Should fail cleanly when the update hunk context cannot be located."""
        path = os.path.join(self.temp_dir, "modify.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("line1\nline2\n")

        patch = "*** Begin Patch\n*** Update File: modify.txt\n@@\n-missing\n+changed\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)

        assert result["error"]["type"] == "APPLY_PATCH_FAILED"
        assert result["content"] == "Failed to find expected lines in modify.txt:\nmissing"
        with open(path, encoding="utf-8") as f:
            assert f.read() == "line1\nline2\n"

    def test_rejects_absolute_and_parent_relative_paths(self):
        """Patch paths must remain sandbox-relative for safety."""
        absolute_patch = "*** Begin Patch\n*** Add File: /tmp/escape.txt\n+hello\n*** End Patch"
        absolute_result = apply_patch(input=absolute_patch, agent_state=self.agent_state)
        assert absolute_result["error"]["type"] == "INVALID_PATCH"
        assert "never absolute" in absolute_result["content"]

        parent_patch = "*** Begin Patch\n*** Add File: ../escape.txt\n+hello\n*** End Patch"
        parent_result = apply_patch(input=parent_patch, agent_state=self.agent_state)
        assert parent_result["error"]["type"] == "INVALID_PATCH"
        assert "within the sandbox working directory" in parent_result["content"]

    def test_failure_after_partial_success_leaves_prior_changes(self):
        """Sequential application should preserve earlier successful changes on later failure."""
        patch = "*** Begin Patch\n*** Add File: created.txt\n+hello\n*** Update File: missing.txt\n@@\n-old\n+new\n*** End Patch"

        result = apply_patch(input=patch, agent_state=self.agent_state)

        assert result["error"]["type"] == "APPLY_PATCH_FAILED"
        assert "Failed to read file to update missing.txt" in result["content"]
        with open(os.path.join(self.temp_dir, "created.txt"), encoding="utf-8") as f:
            assert f.read() == "hello\n"

    def test_delete_missing_file_reports_delete_failure(self):
        """Delete File should use delete-specific failure wording for missing targets."""
        patch = "*** Begin Patch\n*** Delete File: missing.txt\n*** End Patch"

        result = apply_patch(input=patch, agent_state=self.agent_state)

        assert result["error"]["type"] == "APPLY_PATCH_FAILED"
        assert result["content"] == "Failed to delete file missing.txt"
