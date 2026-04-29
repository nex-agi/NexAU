# Shell output cleaning utilities for LLM-friendly formatting.
#
# Issue #498: Strip ANSI escape codes and collapse progress-bar output
#
# Provides a pipeline to clean raw subprocess output before it reaches the LLM:
# 1. strip_ansi()           — remove ANSI CSI/OSC/control escape sequences
# 2. resolve_cr()           — simulate carriage-return line overwrites
# 3. collapse_repetitive()  — collapse consecutive similar lines
# 4. clean_shell_output()   — combined pipeline (strip → resolve → collapse)

from __future__ import annotations

import re

# --- ANSI escape sequence patterns ---

# CSI sequences: ESC[ <params> <intermediate bytes> <final byte>
# Covers standard (e.g. \x1b[31m) and private (e.g. \x1b[?25l) CSI sequences
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

# OSC sequences: ESC] ... ST (ST = ESC\ or BEL)
_ANSI_OSC_RE = re.compile(r"\x1b\].*?(?:\x1b\\|\x07)")

# Remaining single-char escapes (e.g., ESC=, ESC>)
_ANSI_OTHER_RE = re.compile(r"\x1b[^[\]]")

# --- Repetitive-line detection ---

# Replaces digit sequences with a placeholder for similarity comparison
_DIGITS_RE = re.compile(r"\d+")

# Default minimum run length to trigger collapse
_DEFAULT_MIN_RUN = 4


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Issue #498: Strip ANSI escape sequences

    Strips CSI (Control Sequence Introducer), OSC (Operating System Command),
    and other single-character escape sequences that are noise for LLMs.
    """
    text = _ANSI_CSI_RE.sub("", text)
    text = _ANSI_OSC_RE.sub("", text)
    text = _ANSI_OTHER_RE.sub("", text)
    return text


def resolve_cr(text: str) -> str:
    """Resolve carriage-return overwrites, keeping only the last version of each line.

    Issue #498: Simulate carriage-return overwrites, keep only the final line state

    Progress bars use ``\\r`` to overwrite the current line. After ANSI stripping,
    this simulates the terminal behavior by keeping only the final segment after
    the last ``\\r`` on each line.

    CRLF (``\\r\\n``) line endings are normalized to ``\\n`` first so that
    Windows-style output is not destroyed.
    """
    # 1. Normalize CRLF to LF to protect normal line endings
    text = text.replace("\r\n", "\n")

    # 2. Process bare \r as carriage-return overwrites
    out_lines: list[str] = []
    for line in text.split("\n"):
        if "\r" in line:
            # Keep only the segment after the last \r (what the terminal displays)
            line = line.rsplit("\r", 1)[-1]
        out_lines.append(line)
    return "\n".join(out_lines)


def _normalize_for_similarity(line: str) -> str:
    """Replace digit sequences with a placeholder for similarity comparison."""
    return _DIGITS_RE.sub("N", line)


def collapse_repetitive(text: str, *, min_run: int = _DEFAULT_MIN_RUN) -> str:
    """Collapse consecutive similar lines into a summary.

    Issue #498: Collapse repetitive consecutive lines (e.g. progress-bar output)

    When ``min_run`` or more consecutive lines differ only in numeric values
    (e.g. a percentage or counter), keep the first and last lines and insert
    a ``... [N similar lines collapsed]`` summary in between.

    Args:
        text: Input text with potential repetitive lines.
        min_run: Minimum number of consecutive similar lines to trigger collapse.
            Defaults to 4.
    """
    lines = text.split("\n")
    if len(lines) < min_run:
        return text

    result: list[str] = []
    i = 0
    while i < len(lines):
        normalized = _normalize_for_similarity(lines[i])
        # Find the range of consecutive similar lines
        j = i + 1
        while j < len(lines) and _normalize_for_similarity(lines[j]) == normalized:
            j += 1

        run_length = j - i
        if run_length >= min_run:
            # Keep first and last, replace middle with summary
            result.append(lines[i])
            collapsed_count = run_length - 2
            result.append(f"... [{collapsed_count} similar lines collapsed]")
            result.append(lines[j - 1])
        else:
            result.extend(lines[i:j])

        i = j

    return "\n".join(result)


def clean_shell_output(text: str) -> str:
    """Full cleaning pipeline for shell output.

    Issue #498: Shell output cleaning pipeline

    Applies in order:
    1. Strip ANSI escape sequences
    2. Resolve carriage-return overwrites
    3. Collapse repetitive consecutive lines

    This should be called **before** length truncation so that cleaned text
    is shorter and truncation is more accurate.
    """
    if not text:
        return text
    # 1. Strip ANSI escape sequences
    text = strip_ansi(text)
    # 2. Resolve carriage-return overwrites
    text = resolve_cr(text)
    # 3. Collapse repetitive lines
    text = collapse_repetitive(text)
    return text
