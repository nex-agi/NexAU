# Tests for shell output cleaning utilities.
#
# Issue #498: Verify ANSI stripping, CR resolution, repetitive-line collapse,
# and the combined clean_shell_output pipeline.

from __future__ import annotations

from nexau.archs.sandbox.output_utils import (
    clean_shell_output,
    collapse_repetitive,
    resolve_cr,
    strip_ansi,
)

# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


class TestStripAnsi:
    def test_plain_text_passthrough(self) -> None:
        """Plain text with no escapes is returned unchanged."""
        text = "Hello, World! No escapes here."
        assert strip_ansi(text) == text

    def test_strips_csi_color_codes(self) -> None:
        """CSI colour codes like SGR are removed."""
        assert strip_ansi("\x1b[31mred\x1b[0m") == "red"

    def test_strips_csi_cursor_movement(self) -> None:
        """CSI cursor movement sequences are removed."""
        assert strip_ansi("\x1b[2K\x1b[0A\x1b[0E") == ""

    def test_strips_osc_sequences(self) -> None:
        """OSC (Operating System Command) sequences are removed."""
        # OSC terminated by BEL (\x07)
        assert strip_ansi("\x1b]0;title\x07text") == "text"
        # OSC terminated by ST (ESC \)
        assert strip_ansi("\x1b]0;title\x1b\\text") == "text"

    def test_strips_other_escape_sequences(self) -> None:
        """Single-character escape sequences (e.g., ESC=, ESC>) are removed."""
        assert strip_ansi("\x1b=normal") == "normal"
        assert strip_ansi("\x1b>content") == "content"

    def test_mixed_ansi_and_real_content(self) -> None:
        """All non-ANSI text is preserved while escape sequences are stripped."""
        raw = "\x1b[1m\x1b[32mSuccess:\x1b[0m file saved to /tmp/out.txt"
        assert strip_ansi(raw) == "Success: file saved to /tmp/out.txt"

    def test_multiline_with_ansi(self) -> None:
        """Multi-line output with various ANSI codes on different lines."""
        raw = "\x1b[31mError on line 1\x1b[0m\n\x1b[33mWarning on line 2\x1b[0m\n\x1b[2K\x1b[0AProgress line\nClean line"
        expected = "Error on line 1\nWarning on line 2\nProgress line\nClean line"
        assert strip_ansi(raw) == expected

    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert strip_ansi("") == ""


# ---------------------------------------------------------------------------
# resolve_cr
# ---------------------------------------------------------------------------


class TestResolveCr:
    def test_no_cr_passthrough(self) -> None:
        """Text without \\r is returned unchanged."""
        text = "no carriage returns here"
        assert resolve_cr(text) == text

    def test_simple_cr_overwrite(self) -> None:
        """The last segment after \\r on a line wins."""
        assert resolve_cr("old\rnew") == "new"

    def test_progress_bar_simulation(self) -> None:
        """Multiple \\r-separated segments keep only the final one."""
        line = "Downloading: 10%\rDownloading: 50%\rDownloading: 100%"
        assert resolve_cr(line) == "Downloading: 100%"

    def test_multiline_with_cr(self) -> None:
        """Each line is resolved independently."""
        text = "aaa\rbbb\nccc\rddd"
        assert resolve_cr(text) == "bbb\nddd"

    def test_cr_at_end_of_line(self) -> None:
        """A trailing \\r leaves the line empty (overwritten with nothing)."""
        assert resolve_cr("text\r") == ""

    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert resolve_cr("") == ""


# ---------------------------------------------------------------------------
# collapse_repetitive
# ---------------------------------------------------------------------------


class TestCollapseRepetitive:
    def test_short_text_passthrough(self) -> None:
        """Fewer lines than min_run → returned unchanged."""
        text = "line1\nline2\nline3"
        assert collapse_repetitive(text) == text

    def test_no_repetition_passthrough(self) -> None:
        """All different lines (no numeric-only diff) → unchanged."""
        text = "apple\nbanana\ncherry\norange\npear"
        assert collapse_repetitive(text) == text

    def test_collapses_progress_percentage(self) -> None:
        """10 lines like 'Downloading: N%' are collapsed to first+summary+last."""
        lines = [f"Downloading: {i}%" for i in range(1, 11)]
        text = "\n".join(lines)
        result = collapse_repetitive(text)

        result_lines = result.split("\n")
        assert result_lines[0] == "Downloading: 1%"
        assert result_lines[-1] == "Downloading: 10%"
        assert "similar lines collapsed" in result_lines[1]

    def test_min_run_boundary(self) -> None:
        """Exactly min_run-1 lines → not collapsed; exactly min_run → collapsed."""
        three = "\n".join(f"Step {i}" for i in range(3))
        assert collapse_repetitive(three, min_run=4) == three

        four = "\n".join(f"Step {i}" for i in range(4))
        result = collapse_repetitive(four, min_run=4)
        assert "similar lines collapsed" in result

    def test_custom_min_run(self) -> None:
        """Custom min_run=2 collapses even 2 similar lines."""
        text = "row 1\nrow 2"
        result = collapse_repetitive(text, min_run=2)
        assert "similar lines collapsed" in result
        # first + summary + last
        result_lines = result.split("\n")
        assert result_lines[0] == "row 1"
        assert result_lines[-1] == "row 2"

    def test_preserves_non_repetitive_context(self) -> None:
        """Mixed repetitive and non-repetitive lines: only the run is collapsed."""
        lines = [
            "Starting download...",
            "Progress: 1%",
            "Progress: 2%",
            "Progress: 3%",
            "Progress: 4%",
            "Progress: 5%",
            "Download complete.",
        ]
        text = "\n".join(lines)
        result = collapse_repetitive(text)

        assert "Starting download..." in result
        assert "Download complete." in result
        assert "similar lines collapsed" in result
        # Non-repetitive lines kept verbatim
        result_lines = result.split("\n")
        assert result_lines[0] == "Starting download..."
        assert result_lines[-1] == "Download complete."

    def test_multiple_groups(self) -> None:
        """Two separate groups of repetitive lines, both collapsed."""
        group1 = [f"Upload: {i}%" for i in range(1, 6)]
        separator = ["--- separator ---"]
        group2 = [f"Download: {i}%" for i in range(1, 6)]
        text = "\n".join(group1 + separator + group2)
        result = collapse_repetitive(text)

        # Both groups should be collapsed
        assert result.count("similar lines collapsed") == 2
        assert "Upload: 1%" in result
        assert "Upload: 5%" in result
        assert "--- separator ---" in result
        assert "Download: 1%" in result
        assert "Download: 5%" in result

    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert collapse_repetitive("") == ""

    def test_collapsed_count_is_correct(self) -> None:
        """The collapsed count N should equal total_run_length - 2."""
        # 8 lines → collapsed count should be 6
        lines = [f"Building module {i}/100" for i in range(1, 9)]
        text = "\n".join(lines)
        result = collapse_repetitive(text)

        assert "... [6 similar lines collapsed]" in result


# ---------------------------------------------------------------------------
# clean_shell_output
# ---------------------------------------------------------------------------


class TestCleanShellOutput:
    def test_empty_string(self) -> None:
        """Empty input returns empty output."""
        assert clean_shell_output("") == ""

    def test_plain_text_passthrough(self) -> None:
        """Clean text with no ANSI, CR, or repetition is returned unchanged."""
        text = "Hello World\nAll good."
        assert clean_shell_output(text) == text

    def test_full_pipeline_progress_bar(self) -> None:
        """Realistic progress bar with ANSI + CR + repetition is fully cleaned."""
        # Simulate: each progress step overwrites via \r, final state has no trailing \r
        parts = []
        for pct in range(0, 10):
            parts.append(f"\x1b[2K\x1b[0ARetrieving image: {pct}%\r")
        # Final state (what the terminal actually shows) — no trailing \r
        parts.append("\x1b[2K\x1b[0ARetrieving image: 10%")
        raw = "".join(parts)

        result = clean_shell_output(raw)

        # After ANSI strip: "Retrieving image: 0%\r...Retrieving image: 9%\rRetrieving image: 10%"
        # After CR resolve: only "Retrieving image: 10%" remains (single line)
        assert result == "Retrieving image: 10%"
        # ANSI codes are gone
        assert "\x1b" not in result

    def test_full_pipeline_multiline_progress(self) -> None:
        """Multi-line progress with ANSI + repetition collapsed properly."""
        lines = []
        for pct in range(1, 9):
            lines.append(f"\x1b[32mStep {pct}/100\x1b[0m")
        raw = "\n".join(lines)

        result = clean_shell_output(raw)

        # ANSI stripped, repetitive lines collapsed
        assert "\x1b" not in result
        assert "Step 1/100" in result
        assert "Step 8/100" in result
        assert "similar lines collapsed" in result

    def test_ansi_only_no_cr_no_repetition(self) -> None:
        """Only ANSI cleaning happens when there's no CR or repetition."""
        raw = "\x1b[1mBold\x1b[0m and \x1b[4munderline\x1b[0m"
        assert clean_shell_output(raw) == "Bold and underline"

    def test_pipeline_order_matters(self) -> None:
        """ANSI is stripped before CR resolution.

        Important: ANSI codes containing CR-like bytes should not be
        misinterpreted as actual carriage returns.
        """
        # CSI sequence followed by a real \r:
        # If CR were resolved first, \r might split inside the ANSI sequence.
        raw = "\x1b[31mhello\x1b[0m\rworld"
        result = clean_shell_output(raw)
        # After strip_ansi: "hello\rworld"
        # After resolve_cr: "world"
        assert result == "world"


# ---------------------------------------------------------------------------
# Integration: clean_shell_output + smart_truncate_output
# ---------------------------------------------------------------------------


class TestIntegrationWithSmartTruncate:
    def test_ansi_cleaned_before_truncation_check(self) -> None:
        """Output with ANSI that is over threshold raw but under after cleaning.

        After ANSI stripping inside smart_truncate_output, the combined length
        should fall below the threshold, so no truncation occurs.
        """
        from nexau.archs.sandbox.base_sandbox import smart_truncate_output

        # Build a string whose raw length (with ANSI) exceeds threshold but
        # whose cleaned length is well below it.
        # Each chunk: ~30 bytes of ANSI overhead + short text
        ansi_prefix = "\x1b[38;5;196m"  # 11 chars
        ansi_suffix = "\x1b[0m"  # 4 chars
        content_word = "ok"  # 2 chars real
        # One chunk raw = 17 chars, cleaned = 2 chars
        # Need raw > 10_000 but cleaned < 10_000
        # 10_000 / 17 ≈ 589 chunks → raw ≈ 10_013, cleaned ≈ 1_178
        chunks = 600
        raw_stdout = "\n".join(f"{ansi_prefix}{content_word}{ansi_suffix}" for _ in range(chunks))

        # Verify precondition: raw is over threshold
        assert len(raw_stdout) > 10_000

        stdout, stderr, was_truncated, _, _ = smart_truncate_output(raw_stdout, "", "/tmp/test_output_utils")

        # After cleaning, the text should be short enough → no truncation
        assert not was_truncated
        # The returned stdout is the *cleaned* version
        assert "\x1b" not in stdout


# ---------------------------------------------------------------------------
# Edge cases from review (BLOCKER 2, BLOCKER 3, WARNING 1)
# ---------------------------------------------------------------------------


class TestResolveCrCrlf:
    """BLOCKER 2: CRLF handling — \\r\\n must not be destroyed."""

    def test_crlf_preserved_as_newlines(self) -> None:
        """CRLF line endings are normalized to LF, content preserved."""
        assert resolve_cr("line1\r\nline2\r\n") == "line1\nline2\n"

    def test_crlf_mixed_with_bare_cr(self) -> None:
        """CRLF is preserved while bare \\r still triggers overwrite."""
        text = "old\rnew\r\nnext line\r\n"
        assert resolve_cr(text) == "new\nnext line\n"

    def test_crlf_only(self) -> None:
        """Pure CRLF text is just normalized to LF."""
        assert resolve_cr("a\r\nb\r\nc\r\n") == "a\nb\nc\n"


class TestStripAnsiPrivateCsi:
    """BLOCKER 3: Private CSI sequences like \\x1b[?25l must be stripped."""

    def test_strips_hide_cursor(self) -> None:
        """\\x1b[?25l (hide cursor) is stripped."""
        assert strip_ansi("a\x1b[?25lb") == "ab"

    def test_strips_show_cursor(self) -> None:
        """\\x1b[?25h (show cursor) is stripped."""
        assert strip_ansi("a\x1b[?25hb") == "ab"

    def test_strips_hide_and_show_cursor_around_content(self) -> None:
        """Full hide/show cursor wrapping is stripped, content preserved."""
        assert strip_ansi("\x1b[?25lhidden\x1b[?25h") == "hidden"

    def test_strips_dec_private_mode_set_reset(self) -> None:
        """DEC private mode set/reset sequences are stripped."""
        # \x1b[?1049h = enable alt screen, \x1b[?1049l = disable
        assert strip_ansi("\x1b[?1049hcontent\x1b[?1049l") == "content"


class TestFormatterStderrDefense:
    """WARNING 1: Formatter _build_error_text should clean stderr."""

    def test_stderr_ansi_cleaned_by_formatter(self) -> None:
        """Stderr with ANSI codes is cleaned in the formatter output."""
        from nexau.archs.tool.formatters.shell import format_run_shell_command_output

        # Minimal ToolFormatterContext-like input
        class FakeContext:
            tool_output = {
                "stdout": "",
                "stderr": "\x1b[31mERROR: something failed\x1b[0m",
                "content": "",
            }

        result = format_run_shell_command_output(FakeContext())  # type: ignore[arg-type]
        assert "\x1b" not in str(result)
        assert "ERROR: something failed" in str(result)

    def test_stderr_cr_cleaned_by_formatter(self) -> None:
        """Stderr with CR overwrites is cleaned in the formatter output."""
        from nexau.archs.tool.formatters.shell import format_run_shell_command_output

        class FakeContext:
            tool_output = {
                "stdout": "",
                "stderr": "old error\rFINAL error",
                "content": "",
            }

        result = format_run_shell_command_output(FakeContext())  # type: ignore[arg-type]
        assert "old error" not in str(result)
        assert "FINAL error" in str(result)
