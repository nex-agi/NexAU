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

"""Seek-sequence matching tests mirroring official Codex seek_sequence.rs tests.

These verify the progressively-more-permissive line matching algorithm that
powers apply_patch's fuzzy context matching.

Official refs:
  - codex-rs/apply-patch/src/seek_sequence.rs  test_exact_match_finds_sequence
  - codex-rs/apply-patch/src/seek_sequence.rs  test_rstrip_match_ignores_trailing_whitespace
  - codex-rs/apply-patch/src/seek_sequence.rs  test_trim_match_ignores_leading_and_trailing_whitespace
  - codex-rs/apply-patch/src/seek_sequence.rs  test_pattern_longer_than_input_returns_none
"""

from __future__ import annotations

from nexau.archs.tool.builtin.file_tools.apply_patch import (
    _normalize_unicode_line,
    _seek_sequence,
)


class TestSeekSequenceExactMatch:
    """Mirrors seek_sequence.rs test_exact_match_finds_sequence."""

    def test_exact_match_finds_sequence(self):
        lines = ["foo", "bar", "baz"]
        pattern = ["bar", "baz"]
        assert _seek_sequence(lines, pattern, 0, False) == 1

    def test_exact_match_at_start(self):
        lines = ["alpha", "beta", "gamma"]
        assert _seek_sequence(lines, ["alpha", "beta"], 0, False) == 0

    def test_exact_match_single_line(self):
        lines = ["one", "two", "three"]
        assert _seek_sequence(lines, ["two"], 0, False) == 1

    def test_no_match(self):
        lines = ["foo", "bar", "baz"]
        assert _seek_sequence(lines, ["qux"], 0, False) is None


class TestSeekSequenceRstripMatch:
    """Mirrors seek_sequence.rs test_rstrip_match_ignores_trailing_whitespace."""

    def test_rstrip_match_ignores_trailing_whitespace(self):
        lines = ["foo   ", "bar\t\t"]
        pattern = ["foo", "bar"]
        assert _seek_sequence(lines, pattern, 0, False) == 0

    def test_rstrip_only_trailing(self):
        """Leading whitespace should NOT match on rstrip pass alone."""
        lines = ["   foo", "bar"]
        pattern = ["foo", "bar"]
        # rstrip won't match "   foo", but trim will
        result = _seek_sequence(lines, pattern, 0, False)
        assert result == 0  # found via trim pass


class TestSeekSequenceTrimMatch:
    """Mirrors seek_sequence.rs test_trim_match_ignores_leading_and_trailing_whitespace."""

    def test_trim_match_ignores_both_sides(self):
        lines = ["    foo   ", "   bar\t"]
        pattern = ["foo", "bar"]
        assert _seek_sequence(lines, pattern, 0, False) == 0


class TestSeekSequencePatternLongerThanInput:
    """Mirrors seek_sequence.rs test_pattern_longer_than_input_returns_none."""

    def test_pattern_longer_than_input_returns_none(self):
        lines = ["just one line"]
        pattern = ["too", "many", "lines"]
        assert _seek_sequence(lines, pattern, 0, False) is None


class TestSeekSequenceEmptyPattern:
    """Empty pattern should return start position (no-op match)."""

    def test_empty_pattern(self):
        assert _seek_sequence(["a", "b"], [], 0, False) == 0
        assert _seek_sequence(["a", "b"], [], 1, False) == 1


class TestSeekSequenceEOFFlag:
    """When eof=True the search starts from the end of the file."""

    def test_eof_matches_at_end(self):
        lines = ["a", "b", "c", "b", "c"]
        # With eof=True, should prefer the match at index 3 (from the end)
        assert _seek_sequence(lines, ["b", "c"], 0, True) == 3

    def test_eof_single_line_at_end(self):
        lines = ["first", "middle", "last"]
        assert _seek_sequence(lines, ["last"], 0, True) == 2


class TestSeekSequenceStartOffset:
    """Verify that the start parameter is respected."""

    def test_start_skips_earlier_match(self):
        lines = ["dup", "other", "dup"]
        assert _seek_sequence(lines, ["dup"], 1, False) == 2


class TestSeekSequenceUnicodeNormalization:
    """Verify the final unicode normalization pass works like the official Rust version."""

    def test_en_dash_to_ascii_dash(self):
        """EN DASH (\\u2013) in the file, plain dash in the pattern."""
        lines = ["import asyncio  # local import \u2013 avoids top\u2011level dep"]
        pattern = ["import asyncio  # local import - avoids top-level dep"]
        assert _seek_sequence(lines, pattern, 0, False) == 0

    def test_smart_quotes_normalization(self):
        """Fancy quotes in the file matched by plain quotes in the pattern."""
        lines = ["\u201cHello\u201d said \u2018World\u2019"]
        pattern = ["\"Hello\" said 'World'"]
        assert _seek_sequence(lines, pattern, 0, False) == 0

    def test_non_breaking_space_normalization(self):
        """Non-breaking space \\u00A0 in file matched by regular space in pattern."""
        lines = ["hello\u00a0world"]
        pattern = ["hello world"]
        assert _seek_sequence(lines, pattern, 0, False) == 0


class TestNormalizeUnicodeLine:
    """Direct tests for the normalization helper."""

    def test_dashes(self):
        for ch in "\u2010\u2011\u2012\u2013\u2014\u2015\u2212":
            assert _normalize_unicode_line(f"a{ch}b") == "a-b"

    def test_single_quotes(self):
        for ch in "\u2018\u2019\u201a\u201b":
            assert _normalize_unicode_line(f"a{ch}b") == "a'b"

    def test_double_quotes(self):
        for ch in "\u201c\u201d\u201e\u201f":
            assert _normalize_unicode_line(f"a{ch}b") == 'a"b'

    def test_spaces(self):
        for ch in "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000":
            assert _normalize_unicode_line(f"a{ch}b") == "a b"

    def test_strip_applied(self):
        assert _normalize_unicode_line("  hello  ") == "hello"
