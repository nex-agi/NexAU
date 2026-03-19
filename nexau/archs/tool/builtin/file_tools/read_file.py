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


# BOM signatures — UTF-32 must precede UTF-16 (prefix overlap)
_BOM_ENCODINGS: list[tuple[bytes, str]] = [
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
]

_CHARDET_SAMPLE_SIZE = 50_000


def _detect_encoding(file_path: str, sandbox: BaseSandbox) -> str:
    """Detect file encoding using a layered strategy with fallback to utf-8.

    分层编码检测：
    1. BOM 检测（确定性，最高优先级）
    2. UTF-8 strict 验证（覆盖绝大多数现代源码文件）
    3. chardet 启发式检测（处理真正的非 UTF-8 文件）
    4. 回退到 utf-8
    """
    try:
        read_res = sandbox.read_file(file_path, binary=True)
        if read_res.status != SandboxStatus.SUCCESS or not read_res.content:
            return "utf-8"

        raw: bytes
        if isinstance(read_res.content, (bytes, bytearray)):
            raw = bytes(read_res.content)
        else:
            return "utf-8"

        # 1. BOM 检测 — 确定性判断，最高优先级
        for bom, encoding in _BOM_ENCODINGS:
            if raw.startswith(bom):
                return encoding

        # 2. UTF-8 strict 验证 — 文件大小已由调用方限制在 10MB 内
        #    ASCII 是 UTF-8 的严格子集，此步骤同时涵盖了纯 ASCII 文件
        try:
            raw.decode("utf-8", errors="strict")
            return "utf-8"
        except UnicodeDecodeError:
            pass

        # 3. chardet 启发式检测 — 仅对确认非 UTF-8 的文件使用
        try:
            import chardet

            sample = raw[:_CHARDET_SAMPLE_SIZE]
            result = chardet.detect(sample)
            detected = str(result.get("encoding") or "").lower()
            confidence: float = float(result.get("confidence", 0))

            if detected == "ascii":
                return "utf-8"
            if detected and confidence > 0.7:
                return detected
        except ImportError:
            pass

    except Exception:
        pass
    return "utf-8"


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

        # Read text file
        encoding = _detect_encoding(resolved_path, sandbox)

        read_res = sandbox.read_file(resolved_path, encoding=encoding, binary=False)
        if read_res.status == SandboxStatus.SUCCESS:
            content_str = read_res.content if isinstance(read_res.content, str) else ""
        else:
            # 编码读取失败（如 UnicodeDecodeError），用二进制 + UTF-8 replace 兜底
            logger.warning(
                "Failed to read %s with encoding %s, falling back to utf-8 with errors=replace",
                file_path,
                encoding,
            )
            bin_res = sandbox.read_file(resolved_path, binary=True)
            if bin_res.status != SandboxStatus.SUCCESS or not bin_res.content:
                raise RuntimeError(read_res.error or "Failed to read file")
            if isinstance(bin_res.content, (bytes, bytearray)):
                content_str = bytes(bin_res.content).decode("utf-8", errors="replace")
            else:
                content_str = str(bin_res.content)
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

        content_with_lines = _add_line_numbers(
            "\n".join(formatted_lines),
            start_line=start_line + 1,
        )

        # Check if more content exists - add truncation notice
        actual_end = min(end_line, total_lines)
        was_truncated = actual_end < total_lines

        if was_truncated:
            truncation_notice = (
                f"\n\nIMPORTANT: The file content has been truncated.\n"
                f"Status: Showing lines {start_line + 1}-{actual_end} of {total_lines} total lines.\n"
                f"Action: To read more of the file, you can use the 'offset' and 'limit' parameters in a subsequent\n"
                f"'read_file' call. For example, to read the next section of the file, use offset: {actual_end}.\n"
                f"\n--- FILE CONTENT (truncated) ---\n"
            )
            llm_content = truncation_notice + content_with_lines

            return_display = f"Read lines {start_line + 1}-{actual_end} of {total_lines}"
            if lines_were_truncated_in_length:
                return_display += " (some lines were shortened)"
        else:
            llm_content = content_with_lines
            return_display = f"Read {total_lines} lines"
            if lines_were_truncated_in_length:
                return_display += " (some lines were shortened)"

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
