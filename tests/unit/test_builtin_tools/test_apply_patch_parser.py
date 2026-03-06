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

"""Parser-level tests mirroring official Codex apply-patch parser.rs tests.

These verify that the patch text is correctly parsed into hunk structures
without touching the filesystem.

Official refs:
  - codex-rs/apply-patch/src/parser.rs  test_parse_patch
  - codex-rs/apply-patch/src/parser.rs  test_parse_patch_lenient
  - codex-rs/apply-patch/src/parser.rs  test_parse_one_hunk
  - codex-rs/apply-patch/src/parser.rs  test_update_file_chunk
"""

from __future__ import annotations

import pytest

from nexau.archs.tool.builtin.file_tools.apply_patch import (
    AddFileHunk,
    DeleteFileHunk,
    InvalidHunkError,
    InvalidPatchError,
    UpdateFileHunk,
    _parse_one_hunk,
    _parse_patch_text,
    _parse_update_file_chunk,
    _strip_lenient_heredoc,
)


class TestParsePatchBoundaries:
    """Verify Begin/End Patch marker validation."""

    def test_missing_begin_marker(self):
        with pytest.raises(InvalidPatchError, match="first line.*Begin Patch"):
            _parse_patch_text("not a patch\n*** End Patch")

    def test_missing_end_marker(self):
        with pytest.raises(InvalidPatchError, match="last line.*End Patch"):
            _parse_patch_text("*** Begin Patch\nnot an end")

    def test_empty_input(self):
        with pytest.raises(InvalidPatchError):
            _parse_patch_text("")


class TestParsePatchAddFile:
    """Mirrors parser.rs test_parse_patch – AddFile parsing."""

    def test_add_single_file(self):
        patch = "*** Begin Patch\n*** Add File: bar.md\n+hello\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        hunk = hunks[0]
        assert isinstance(hunk, AddFileHunk)
        assert hunk.path == "bar.md"
        assert hunk.contents == "hello\n"

    def test_add_file_multiple_lines(self):
        patch = "*** Begin Patch\n*** Add File: new.txt\n+line1\n+line2\n+line3\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        assert isinstance(hunks[0], AddFileHunk)
        assert hunks[0].contents == "line1\nline2\nline3\n"


class TestParsePatchDeleteFile:
    """Mirrors parser.rs test_parse_patch – DeleteFile parsing."""

    def test_delete_file(self):
        patch = "*** Begin Patch\n*** Delete File: obsolete.txt\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        assert isinstance(hunks[0], DeleteFileHunk)
        assert hunks[0].path == "obsolete.txt"


class TestParsePatchUpdateFile:
    """Mirrors parser.rs test_parse_patch – UpdateFile parsing."""

    def test_update_with_context_marker(self):
        patch = "*** Begin Patch\n*** Update File: file.txt\n@@\n context\n-old\n+new\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        hunk = hunks[0]
        assert isinstance(hunk, UpdateFileHunk)
        assert hunk.path == "file.txt"
        assert hunk.move_path is None
        assert len(hunk.chunks) == 1
        chunk = hunk.chunks[0]
        assert chunk.change_context is None  # bare @@ → None
        assert chunk.old_lines == ["context", "old"]
        assert chunk.new_lines == ["context", "new"]

    def test_update_with_named_context(self):
        patch = "*** Begin Patch\n*** Update File: file.txt\n@@ some_function\n-old\n+new\n*** End Patch"
        hunks = _parse_patch_text(patch)
        chunk = hunks[0].chunks[0]
        assert chunk.change_context == "some_function"

    def test_update_with_move_to(self):
        patch = "*** Begin Patch\n*** Update File: old/path.txt\n*** Move to: new/path.txt\n@@\n-old\n+new\n*** End Patch"
        hunks = _parse_patch_text(patch)
        hunk = hunks[0]
        assert isinstance(hunk, UpdateFileHunk)
        assert hunk.path == "old/path.txt"
        assert hunk.move_path == "new/path.txt"

    def test_update_with_multiple_chunks(self):
        """Official: two @@ chunks inside a single Update File hunk."""
        patch = "*** Begin Patch\n*** Update File: multi.txt\n@@\n-line2\n+changed2\n@@\n-line4\n+changed4\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        assert len(hunks[0].chunks) == 2

    def test_update_hunk_followed_by_other_hunks(self):
        """Official parser.rs: Update hunk followed by Add and Delete hunks."""
        patch = (
            "*** Begin Patch\n"
            "*** Update File: file2.py\n"
            "@@\n"
            "-import foo\n"
            "+import foo\n"
            "+bar\n"
            "*** Add File: bar.md\n"
            "+hello\n"
            "*** Delete File: obsolete.txt\n"
            "*** End Patch"
        )
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 3
        assert isinstance(hunks[0], UpdateFileHunk)
        assert isinstance(hunks[1], AddFileHunk)
        assert isinstance(hunks[2], DeleteFileHunk)

    def test_first_chunk_without_explicit_context_marker(self):
        """Official parser.rs: first chunk can omit the @@ marker."""
        patch = "*** Begin Patch\n*** Update File: file2.py\n-import foo\n+import foo\n+bar\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        hunk = hunks[0]
        assert isinstance(hunk, UpdateFileHunk)
        assert len(hunk.chunks) == 1
        chunk = hunk.chunks[0]
        assert chunk.change_context is None
        assert chunk.old_lines == ["import foo"]
        assert chunk.new_lines == ["import foo", "bar"]


class TestParsePatchWhitespacePadding:
    """Official parser.rs: whitespace-padded markers should parse correctly."""

    def test_whitespace_padded_begin_end(self):
        patch = "  *** Begin Patch  \n*** Add File: foo.txt\n+hello\n  *** End Patch  "
        hunks = _parse_patch_text(patch)
        assert len(hunks) == 1
        assert isinstance(hunks[0], AddFileHunk)

    def test_whitespace_padded_hunk_header(self):
        """Official scenario 017_whitespace_padded_hunk_header."""
        patch = "*** Begin Patch\n  *** Update File: foo.txt  \n@@\n-old\n+new\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert isinstance(hunks[0], UpdateFileHunk)
        assert hunks[0].path == "foo.txt"

    def test_whitespace_padded_delete_hunk(self):
        patch = "*** Begin Patch\n  *** Delete File: old.txt  \n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert isinstance(hunks[0], DeleteFileHunk)
        assert hunks[0].path == "old.txt"


class TestParsePatchLenientHeredoc:
    """Mirrors parser.rs test_parse_patch_lenient – heredoc wrapper stripping."""

    INNER_PATCH = "*** Begin Patch\n*** Update File: file2.py\n@@\n-import foo\n+import foo\n+bar\n*** End Patch"

    def test_bare_heredoc(self):
        """<<EOF … EOF wrapper should be stripped."""
        wrapped = f"<<EOF\n{self.INNER_PATCH}\nEOF\n"
        hunks = _parse_patch_text(wrapped)
        assert len(hunks) == 1
        assert isinstance(hunks[0], UpdateFileHunk)

    def test_single_quoted_heredoc(self):
        """<<'EOF' … EOF wrapper."""
        wrapped = f"<<'EOF'\n{self.INNER_PATCH}\nEOF\n"
        hunks = _parse_patch_text(wrapped)
        assert len(hunks) == 1

    def test_double_quoted_heredoc(self):
        """<<\"EOF\" … EOF wrapper."""
        wrapped = f'<<"EOF"\n{self.INNER_PATCH}\nEOF\n'
        hunks = _parse_patch_text(wrapped)
        assert len(hunks) == 1

    def test_mismatched_quotes_not_stripped(self):
        """Mismatched quotes like <<\"EOF' should NOT be stripped."""
        wrapped = f"<<\"EOF'\n{self.INNER_PATCH}\nEOF\n"
        # _strip_lenient_heredoc should not strip, and then the bare text
        # might or might not parse depending on boundary check. The key is
        # the mismatched delimiter is NOT accepted as a heredoc.
        result = _strip_lenient_heredoc(wrapped)
        # Should return original text unchanged (not stripped)
        assert result == wrapped

    def test_missing_closing_heredoc(self):
        """<<EOF without a matching closing EOF – parser should error."""
        wrapped = "<<EOF\n*** Begin Patch\n*** Update File: file2.py\nEOF\n"
        # After stripping, this becomes "*** Begin Patch\n*** Update File: file2.py"
        # which should fail because the last line is not "*** End Patch"
        with pytest.raises(InvalidPatchError, match="End Patch"):
            _parse_patch_text(wrapped)


class TestParseOneHunk:
    """Mirrors parser.rs test_parse_one_hunk."""

    def test_invalid_hunk_header(self):
        with pytest.raises(InvalidHunkError, match="not a valid hunk header"):
            _parse_one_hunk(["bad"], 234)

    def test_frobnicate_invalid_header(self):
        """Official error message for unknown *** markers."""
        with pytest.raises(InvalidHunkError, match="Frobnicate File: foo"):
            _parse_one_hunk(["*** Frobnicate File: foo"], 2)


class TestParseUpdateFileChunk:
    """Mirrors parser.rs test_update_file_chunk edge cases."""

    def test_missing_context_marker_strict(self):
        """When allow_missing_context=False, a non-@@ first line errors."""
        with pytest.raises(InvalidHunkError, match="Expected update hunk to start with a @@"):
            _parse_update_file_chunk(["bad"], 123, allow_missing_context=False)

    def test_empty_hunk_after_context_marker(self):
        """@@ followed by nothing is an error."""
        with pytest.raises(InvalidHunkError, match="does not contain any lines"):
            _parse_update_file_chunk(["@@"], 123, allow_missing_context=False)

    def test_bad_line_after_context(self):
        """@@ followed by a line without +/- / space prefix errors."""
        with pytest.raises(InvalidHunkError, match="Unexpected line found"):
            _parse_update_file_chunk(["@@", "bad"], 123, allow_missing_context=False)

    def test_eof_marker_without_content(self):
        """*** End of File directly after @@ is an error."""
        with pytest.raises(InvalidHunkError, match="does not contain any lines"):
            _parse_update_file_chunk(["@@", "*** End of File"], 123, allow_missing_context=False)

    def test_chunk_with_context_empty_lines_and_interleaved(self):
        """Official: full chunk with change_context, empty lines, +/- interleaving."""
        lines = [
            "@@ change_context",
            "",  # empty line = context on both sides
            " context",
            "-remove",
            "+add",
            " context2",
            "*** End Patch",  # acts as break
        ]
        chunk, consumed = _parse_update_file_chunk(lines, 123, allow_missing_context=False)
        assert chunk.change_context == "change_context"
        assert chunk.old_lines == ["", "context", "remove", "context2"]
        assert chunk.new_lines == ["", "context", "add", "context2"]
        assert not chunk.is_end_of_file
        assert consumed == 6  # @@ + 5 content lines (break at "*** End Patch")

    def test_chunk_with_eof_marker(self):
        """Official: +line followed by *** End of File."""
        lines = ["@@", "+line", "*** End of File"]
        chunk, consumed = _parse_update_file_chunk(lines, 123, allow_missing_context=False)
        assert chunk.change_context is None
        assert chunk.old_lines == []
        assert chunk.new_lines == ["line"]
        assert chunk.is_end_of_file is True
        assert consumed == 3

    def test_empty_update_hunk_for_path(self):
        """Official scenario 008: Update File with no chunks raises InvalidHunkError."""
        patch = "*** Begin Patch\n*** Update File: foo.txt\n*** End Patch"
        with pytest.raises(InvalidHunkError, match="Update file hunk for path 'foo.txt' is empty"):
            _parse_patch_text(patch)


class TestParseEmptyPatch:
    """Official: empty patch body returns an empty list of hunks."""

    def test_empty_patch(self):
        patch = "*** Begin Patch\n*** End Patch"
        hunks = _parse_patch_text(patch)
        assert hunks == []
