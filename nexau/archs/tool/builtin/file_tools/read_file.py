# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""
read_file tool - Reads and returns the content of a specified text file.

Based on gemini-cli's read-file.ts implementation.
Handles text files only. For images and videos, use read_visual_file.
"""

import logging
from pathlib import Path
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.sandbox import BaseSandbox, SandboxStatus
from nexau.archs.tool.builtin._sandbox_utils import get_sandbox, resolve_path

logger = logging.getLogger(__name__)

# Configuration constants matching gemini-cli
DEFAULT_LINE_LIMIT = 2000
MAX_LINE_LENGTH = 2000  # Truncate lines longer than this with '... [truncated]'
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Total output budget — prevents 2000 lines × 2000 chars/line ≈ 4M chars from
# blowing up the context window.  When exceeded, only the first lines that fit
# within the budget are kept; the rest are truncated.
MAX_TOTAL_OUTPUT_CHARS = 20_000

# Audio extensions - returned as placeholder text (nexau has no AudioBlock)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aac", ".ogg", ".flac"}
PDF_EXTENSION = ".pdf"

# Image and video extensions - NOT handled here; redirect to read_visual_file
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
    ".svg",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

# Binary extensions that cannot be read as text - return error without attempting read
BINARY_SKIP_EXTENSIONS = {".db", ".sqlite", ".db-shm", ".db-wal", ".pyc", ".pyo"}


def _head_truncate_lines(
    lines: list[str],
    max_chars: int = MAX_TOTAL_OUTPUT_CHARS,
) -> tuple[list[str], int, int]:
    """Keep only the first lines that fit within *max_chars*, drop the rest.

    Returns:
        (result_lines, omitted_line_count, omitted_char_count)
        When no truncation is needed, omitted counts are both 0.
    """
    n = len(lines)
    # +1 per line accounts for the joining newline
    total_chars = sum(len(line) for line in lines) + max(n - 1, 0)
    if total_chars <= max_chars:
        return lines, 0, 0

    # 从头部按行累加，保留能放进预算的行
    head_count = 0
    head_chars = 0
    for line in lines:
        cost = len(line) + (1 if head_count > 0 else 0)  # newline between lines
        if head_chars + cost > max_chars:
            break
        head_chars += cost
        head_count += 1

    # 保证至少保留 1 行（单行超大时 head_count 可能为 0）
    if head_count == 0 and n > 0:
        head_count = 1
        head_chars = len(lines[0])

    omitted_count = n - head_count
    omitted_chars = total_chars - head_chars

    return lines[:head_count], omitted_count, omitted_chars


def _add_line_numbers(content: str, start_line: int = 1) -> str:
    """Add line numbers to content."""
    lines = content.splitlines()
    if not lines:
        return content

    max_line_num = start_line + len(lines) - 1
    width = len(str(max_line_num))

    numbered_lines: list[str] = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:>{width}}| {line}")

    return "\n".join(numbered_lines)


def _read_text_lossy(file_path: str, sandbox: BaseSandbox) -> str:
    """Read file as raw bytes and decode as UTF-8 with lossy replacement.

    Follows the same approach as OpenAI Codex (from_utf8_lossy): always read
    raw bytes and decode as UTF-8, replacing invalid byte sequences with U+FFFD.
    This avoids unreliable chardet encoding detection — especially when the file
    is large and the first 10KB sample contains only ASCII, causing chardet to
    misidentify the encoding as ascii or utf-7.
    """
    read_res = sandbox.read_file(file_path, binary=True)
    if read_res.status != SandboxStatus.SUCCESS:
        raise RuntimeError(read_res.error or "Failed to read file")

    if isinstance(read_res.content, (bytes, bytearray)):
        return bytes(read_res.content).decode("utf-8", errors="replace")

    # sandbox already returned str (e.g. remote sandbox decoded it)
    return read_res.content if isinstance(read_res.content, str) else ""


def _is_visual_file(file_path: str) -> bool:
    """Check if file is an image or video file (should use read_visual_file)."""
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext in VIDEO_EXTENSIONS


def _is_audio_or_pdf(file_path: str) -> bool:
    """Check if file is audio or PDF."""
    ext = Path(file_path).suffix.lower()
    return ext in AUDIO_EXTENSIONS or ext == PDF_EXTENSION


def read_file(
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
    agent_state: AgentState | None = None,
) -> dict[str, Any]:
    """
    Reads and returns the content of a specified text file.

    If the file is large, the content will be truncated. The tool's response
    will clearly indicate if truncation has occurred and will provide details
    on how to read more of the file using the 'offset' and 'limit' parameters.

    Handles text and PDF files. For images and videos, use the
    read_visual_file tool instead.

    Args:
        file_path: The path to the file to read
        offset: Optional 0-based line number to start reading from
        limit: Optional maximum number of lines to read

    Returns:
        Dict with content and returnDisplay matching gemini-cli format
    """
    try:
        sandbox = get_sandbox(agent_state)

        # Resolve path (relative -> sandbox work_dir)
        resolved_path = resolve_path(file_path, sandbox)

        # Check if file exists
        if not sandbox.file_exists(resolved_path):
            error_msg = f"File not found: {file_path}"
            return {
                "content": error_msg,
                "returnDisplay": "File not found.",
                "error": {
                    "message": error_msg,
                    "type": "FILE_NOT_FOUND",
                },
            }

        # Check if it's a directory
        info = sandbox.get_file_info(resolved_path)
        if info.is_directory:
            error_msg = f"Path is a directory, not a file: {file_path}"
            return {
                "content": error_msg,
                "returnDisplay": "Path is a directory.",
                "error": {
                    "message": error_msg,
                    "type": "PATH_IS_DIRECTORY",
                },
            }

        # Redirect image/video files to read_visual_file
        if _is_visual_file(resolved_path):
            ext = Path(resolved_path).suffix.lower()
            error_msg = (
                f"File '{file_path}' is an image/video file ({ext}). "
                f"Use the read_visual_file tool instead of read_file for image and video files."
            )
            return {
                "content": error_msg,
                "returnDisplay": f"Image/video file ({ext}) — use read_visual_file.",
                "error": {
                    "message": error_msg,
                    "type": "USE_READ_VISUAL_FILE",
                },
            }

        # Check file size
        file_size = int(info.size or 0)
        if file_size > MAX_FILE_SIZE_BYTES:
            error_msg = f"File too large ({file_size} bytes). Maximum size is {MAX_FILE_SIZE_BYTES} bytes."
            return {
                "content": error_msg,
                "returnDisplay": "File too large.",
                "error": {
                    "message": error_msg,
                    "type": "FILE_TOO_LARGE",
                },
            }

        # Handle audio/PDF - return text placeholder
        if _is_audio_or_pdf(resolved_path):
            ext = Path(resolved_path).suffix.lower()
            return {
                "content": f"Read binary file ({ext}) - content not displayed to model",
                "returnDisplay": f"Read {ext} file: {file_path}",
            }

        # Reject known binary types that cannot be read as text (avoids UnicodeDecodeError)
        ext = Path(resolved_path).suffix.lower()
        if ext in BINARY_SKIP_EXTENSIONS:
            error_msg = f"File type {ext} is binary and cannot be read as text: {file_path}"
            return {
                "content": error_msg,
                "returnDisplay": f"Binary file ({ext}), cannot read as text.",
                "error": {"message": error_msg, "type": "BINARY_FILE"},
            }

        # Read text file — always UTF-8 lossy, matching Codex's from_utf8_lossy approach
        content_str = _read_text_lossy(resolved_path, sandbox)
        all_lines = content_str.splitlines()
        total_lines = len(all_lines)

        # Apply offset and limit
        start_line = int(offset) if offset is not None and offset >= 0 else 0
        line_limit = int(limit) if limit is not None and limit > 0 else DEFAULT_LINE_LIMIT
        end_line = start_line + line_limit

        selected_lines = all_lines[start_line:end_line]

        # Truncate long lines (matching gemini-cli MAX_LINE_LENGTH_TEXT_FILE)
        lines_were_truncated_in_length = False
        formatted_lines: list[str] = []
        for line in selected_lines:
            if len(line) > MAX_LINE_LENGTH:
                formatted_lines.append(line[:MAX_LINE_LENGTH] + "... [truncated]")
                lines_were_truncated_in_length = True
            else:
                formatted_lines.append(line)

        # ④ Head-only truncation — cap total output chars to protect context window
        formatted_lines, omitted_count, omitted_chars = _head_truncate_lines(formatted_lines)

        # Check if more content exists beyond the selected range
        actual_end = min(end_line, total_lines)
        was_line_truncated = actual_end < total_lines

        # Build numbered content — when trailing lines were omitted, only
        # show the kept head portion with a truncation marker.
        if omitted_count > 0:
            kept_count = len(formatted_lines)
            head_text = _add_line_numbers(
                "\n".join(formatted_lines),
                start_line=start_line + 1,
            )
            first_omitted = start_line + kept_count + 1
            last_omitted = start_line + kept_count + omitted_count
            marker = (
                f"\n\n... [{omitted_count} lines, ~{omitted_chars} chars truncated"
                f" (lines {first_omitted}-{last_omitted})"
                f" — use offset/limit to read further] ...\n"
            )
            content_with_lines = head_text + marker
        else:
            content_with_lines = _add_line_numbers(
                "\n".join(formatted_lines),
                start_line=start_line + 1,
            )

        # Build truncation notice and return display string
        notices: list[str] = []
        display_parts: list[str] = []

        if was_line_truncated:
            notices.append(f"Status: Showing lines {start_line + 1}-{actual_end} of {total_lines} total lines.")
            display_parts.append(f"Read lines {start_line + 1}-{actual_end} of {total_lines}")
        else:
            display_parts.append(f"Read {actual_end - start_line} lines")

        if omitted_count > 0:
            notices.append(
                f"Note: {omitted_count} trailing lines"
                f" (~{omitted_chars} chars) were truncated"
                f" to fit the output budget ({MAX_TOTAL_OUTPUT_CHARS} chars)."
            )
            display_parts.append("(output truncated)")

        if lines_were_truncated_in_length:
            display_parts.append("(some lines were shortened)")

        return_display = " ".join(display_parts)

        if notices:
            notices.insert(0, "IMPORTANT: The file content has been truncated.")
            notices.append("Action: Use 'offset' and 'limit' parameters to read specific sections.")
            truncation_notice = "\n\n" + "\n".join(notices) + "\n\n--- FILE CONTENT (truncated) ---\n"
            llm_content = truncation_notice + content_with_lines
        else:
            llm_content = content_with_lines

        return {
            "content": llm_content,
            "returnDisplay": return_display,
        }

    except PermissionError:
        error_msg = f"Permission denied: {file_path}"
        return {
            "content": error_msg,
            "returnDisplay": "Permission denied.",
            "error": {
                "message": error_msg,
                "type": "PERMISSION_DENIED",
            },
        }
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        return {
            "content": error_msg,
            "returnDisplay": "Error reading file.",
            "error": {
                "message": error_msg,
                "type": "READ_ERROR",
            },
        }
