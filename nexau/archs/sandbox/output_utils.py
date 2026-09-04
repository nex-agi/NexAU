# Shell output cleaning utilities for LLM-friendly formatting.
#
# Strip ANSI escape codes and resolve carriage returns from shell output.
#
# Provides a pipeline to clean raw subprocess output before it reaches the LLM:
# 1. strip_ansi()           — remove ANSI CSI/OSC/control escape sequences
# 2. resolve_cr()           — simulate carriage-return line overwrites
# 3. clean_shell_output()   — combined pipeline (strip → resolve)

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


def clean_shell_output(text: str) -> str:
    """Full cleaning pipeline for shell output.

    Applies in order:
    1. Strip ANSI escape sequences (colors, cursor control)
    2. Resolve carriage-return overwrites (progress bar overwrites)
    """
    if not text:
        return text
    # 1. Strip ANSI escape sequences
    text = strip_ansi(text)
    # 2. Resolve carriage-return overwrites
    text = resolve_cr(text)
    return text
