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

"""Application-level tests mirroring official Codex apply-patch lib.rs tests.

These verify end-to-end patch application via the NexAU builtin tool by
writing files to a local sandbox, running apply_patch, and asserting on
both the tool result and the resulting file contents.

Official refs:
  - codex-rs/apply-patch/src/lib.rs  test_empty_patch_returns_error
  - codex-rs/apply-patch/src/lib.rs  test_add_file_creates_new_file
  - codex-rs/apply-patch/src/lib.rs  test_delete_file_removes_file
  - codex-rs/apply-patch/src/lib.rs  test_update_file_hunk_modifies_content
  - codex-rs/apply-patch/src/lib.rs  test_update_file_hunk_can_move_file
  - codex-rs/apply-patch/src/lib.rs  test_multiple_update_chunks_apply_to_single_file
  - codex-rs/apply-patch/src/lib.rs  test_update_file_hunk_interleaved_changes
  - codex-rs/apply-patch/src/lib.rs  test_pure_addition_chunk_followed_by_removal
  - codex-rs/apply-patch/src/lib.rs  test_update_line_with_unicode_dash
  - codex-rs/apply-patch/src/lib.rs  test_update_file_deletion_only (via scenario 021)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools.apply_patch import apply_patch


def _make_agent_state(sandbox):
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestApplyPatchApplication:
    """End-to-end patch application tests mirroring official lib.rs tests."""

    @pytest.fixture(autouse=True)
    def setup_temp_dir(self):
        self.temp_dir = tempfile.mkdtemp(prefix="apply-patch-app-test-")
        self.sandbox = LocalSandbox(sandbox_id="test", work_dir=self.temp_dir)
        self.agent_state = _make_agent_state(self.sandbox)
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, rel_path: str, content: str):
        full = os.path.join(self.temp_dir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)

    def _read(self, rel_path: str) -> str:
        with open(os.path.join(self.temp_dir, rel_path), encoding="utf-8") as f:
            return f.read()

    def _exists(self, rel_path: str) -> bool:
        return os.path.exists(os.path.join(self.temp_dir, rel_path))

    # ------------------------------------------------------------------
    # test_empty_patch_returns_error
    # ------------------------------------------------------------------
    def test_empty_patch_returns_error(self):
        """Official: empty patch body -> EMPTY_PATCH error type."""
        result = apply_patch(
            input="*** Begin Patch\n*** End Patch",
            agent_state=self.agent_state,
        )
        assert result["error"]["type"] == "EMPTY_PATCH"
        assert "No files were modified" in result["content"]

    # ------------------------------------------------------------------
    # test_add_file_creates_new_file
    # ------------------------------------------------------------------
    def test_add_file_creates_new_file(self):
        """Official: AddFile hunk creates a new file on disk."""
        patch = "*** Begin Patch\n*** Add File: bar.md\n+hello\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert "A bar.md" in result["content"]
        assert self._read("bar.md") == "hello\n"

    # ------------------------------------------------------------------
    # test_delete_file_removes_file
    # ------------------------------------------------------------------
    def test_delete_file_removes_file(self):
        """Official: DeleteFile hunk removes the file."""
        self._write("obsolete.txt", "gone\n")
        patch = "*** Begin Patch\n*** Delete File: obsolete.txt\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert "D obsolete.txt" in result["content"]
        assert not self._exists("obsolete.txt")

    # ------------------------------------------------------------------
    # test_update_file_hunk_modifies_content
    # ------------------------------------------------------------------
    def test_update_file_modifies_content(self):
        """Official: simple update replaces a line."""
        self._write("update.txt", "foo\nbar\n")
        patch = "*** Begin Patch\n*** Update File: update.txt\n@@\n foo\n-bar\n+baz\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert "M update.txt" in result["content"]
        assert self._read("update.txt") == "foo\nbaz\n"

    # ------------------------------------------------------------------
    # test_update_file_hunk_can_move_file
    # ------------------------------------------------------------------
    def test_update_file_can_move(self):
        """Official: Update File with Move to renames the file."""
        self._write("src.txt", "line\n")
        patch = "*** Begin Patch\n*** Update File: src.txt\n*** Move to: dst.txt\n@@\n-line\n+line2\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert "M dst.txt" in result["content"]
        assert not self._exists("src.txt")
        assert self._read("dst.txt") == "line2\n"

    # ------------------------------------------------------------------
    # test_multiple_update_chunks_apply_to_single_file
    # ------------------------------------------------------------------
    def test_multiple_update_chunks_single_file(self):
        """Official: two @@ chunks modify different parts of the same file."""
        self._write("multi.txt", "foo\nbar\nbaz\nqux\n")
        patch = "*** Begin Patch\n*** Update File: multi.txt\n@@\n foo\n-bar\n+BAR\n@@\n baz\n-qux\n+QUX\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("multi.txt") == "foo\nBAR\nbaz\nQUX\n"

    # ------------------------------------------------------------------
    # test_update_file_hunk_interleaved_changes
    # ------------------------------------------------------------------
    def test_interleaved_changes(self):
        """Official: replace + replace + append at EOF across three chunks."""
        self._write("interleaved.txt", "a\nb\nc\nd\ne\nf\n")
        patch = (
            "*** Begin Patch\n"
            "*** Update File: interleaved.txt\n"
            "@@\n"
            " a\n"
            "-b\n"
            "+B\n"
            "@@\n"
            " c\n"
            " d\n"
            "-e\n"
            "+E\n"
            "@@\n"
            " f\n"
            "+g\n"
            "*** End of File\n"
            "*** End Patch"
        )
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("interleaved.txt") == "a\nB\nc\nd\nE\nf\ng\n"

    # ------------------------------------------------------------------
    # test_pure_addition_chunk_followed_by_removal
    # ------------------------------------------------------------------
    def test_pure_addition_chunk_followed_by_removal(self):
        """Official: a pure addition chunk then a removal chunk."""
        self._write("panic.txt", "line1\nline2\nline3\n")
        patch = (
            "*** Begin Patch\n"
            "*** Update File: panic.txt\n"
            "@@\n"
            "+after-context\n"
            "+second-line\n"
            "@@\n"
            " line1\n"
            "-line2\n"
            "-line3\n"
            "+line2-replacement\n"
            "*** End Patch"
        )
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("panic.txt") == "line1\nline2-replacement\nafter-context\nsecond-line\n"

    # ------------------------------------------------------------------
    # test_update_line_with_unicode_dash
    # ------------------------------------------------------------------
    def test_update_line_with_unicode_dash(self):
        """Official: ASCII patch can update lines containing Unicode punctuation."""
        # Original uses EN DASH and NON-BREAKING HYPHEN
        original = "import asyncio  # local import \u2013 avoids top\u2011level dep\n"
        self._write("unicode.py", original)
        # Patch uses plain ASCII
        patch = (
            "*** Begin Patch\n"
            "*** Update File: unicode.py\n"
            "@@\n"
            "-import asyncio  # local import - avoids top-level dep\n"
            "+import asyncio  # HELLO\n"
            "*** End Patch"
        )
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert "M unicode.py" in result["content"]
        assert self._read("unicode.py") == "import asyncio  # HELLO\n"

    # ------------------------------------------------------------------
    # test_update_file_deletion_only (scenario 021)
    # ------------------------------------------------------------------
    def test_update_file_deletion_only(self):
        """Official scenario 021: update that only removes lines."""
        self._write("lines.txt", "keep\nremove\n")
        patch = "*** Begin Patch\n*** Update File: lines.txt\n@@\n keep\n-remove\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("lines.txt") == "keep\n"

    # ------------------------------------------------------------------
    # test_update_file_end_of_file_marker (scenario 022)
    # ------------------------------------------------------------------
    def test_update_file_end_of_file_marker(self):
        """Official scenario 022: *** End of File marker anchors to tail."""
        self._write("tail.txt", "first\nmiddle\nlast\n")
        patch = "*** Begin Patch\n*** Update File: tail.txt\n@@\n middle\n-last\n+LAST\n*** End of File\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("tail.txt") == "first\nmiddle\nLAST\n"

    # ------------------------------------------------------------------
    # test_update_file_appends_trailing_newline (scenario 014)
    # ------------------------------------------------------------------
    def test_update_file_appends_trailing_newline(self):
        """Official: files without trailing newlines get one after patching."""
        self._write("no_newline.txt", "no newline at end")
        patch = "*** Begin Patch\n*** Update File: no_newline.txt\n@@\n-no newline at end\n+first line\n+second line\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        contents = self._read("no_newline.txt")
        assert contents.endswith("\n")
        assert contents == "first line\nsecond line\n"

    # ------------------------------------------------------------------
    # test_pure_addition_update_chunk (scenario 016)
    # ------------------------------------------------------------------
    def test_pure_addition_update_chunk(self):
        """Official scenario 016: chunk with only + lines appends to end."""
        self._write("input.txt", "existing\n")
        patch = "*** Begin Patch\n*** Update File: input.txt\n@@\n+appended\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("input.txt") == "existing\nappended\n"

    # ------------------------------------------------------------------
    # test_move_overwrites_existing_destination (scenario 010)
    # ------------------------------------------------------------------
    def test_move_overwrites_existing_destination(self):
        """Official scenario 010: move overwrites an existing file at the dest."""
        self._write("old/name.txt", "from\n")
        self._write("renamed/dir/name.txt", "existing\n")
        patch = "*** Begin Patch\n*** Update File: old/name.txt\n*** Move to: renamed/dir/name.txt\n@@\n-from\n+new\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert not self._exists("old/name.txt")
        assert self._read("renamed/dir/name.txt") == "new\n"

    # ------------------------------------------------------------------
    # test_add_overwrites_existing_file (scenario 011)
    # ------------------------------------------------------------------
    def test_add_overwrites_existing_file(self):
        """Official scenario 011: Add File replaces existing file content."""
        self._write("duplicate.txt", "old content\n")
        patch = "*** Begin Patch\n*** Add File: duplicate.txt\n+new content\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("duplicate.txt") == "new content\n"

    # ------------------------------------------------------------------
    # test_unicode_simple (scenario 019)
    # ------------------------------------------------------------------
    def test_unicode_simple(self):
        """Official scenario 019: Unicode file contents handled correctly."""
        self._write("foo.txt", "héllo wörld\n")
        patch = "*** Begin Patch\n*** Update File: foo.txt\n@@\n-héllo wörld\n+héllo 世界\n*** End Patch"
        result = apply_patch(input=patch, agent_state=self.agent_state)
        assert "error" not in result
        assert self._read("foo.txt") == "héllo 世界\n"
