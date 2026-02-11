# SPDX-License-Identifier: Apache-2.0
"""
read_file tool - Reads and returns the content of a specified file.

Based on gemini-cli's read-file.ts implementation.
Handles text, images, audio files, and PDF files.
"""

import base64
import mimetypes
from pathlib import Path
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.sandbox import BaseSandbox, SandboxStatus

from .sandbox_utils import get_sandbox, resolve_path

# Configuration constants matching gemini-cli
DEFAULT_LINE_LIMIT = 2000
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Supported media types for binary files
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aac", ".ogg", ".flac"}
PDF_EXTENSION = ".pdf"


def _add_line_numbers(content: str, start_line: int = 1) -> str:
    """Add line numbers to content."""
    lines = content.splitlines()
    if not lines:
        return content

    max_line_num = start_line + len(lines) - 1
    width = len(str(max_line_num))

    numbered_lines = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:>{width}}| {line}")

    return "\n".join(numbered_lines)


def _detect_encoding(file_path: str, sandbox: BaseSandbox) -> str:
    """Detect file encoding with fallback to utf-8 (via sandbox read)."""
    try:
        import chardet

        read_res = sandbox.read_file(file_path, binary=True)
        if read_res.status == SandboxStatus.SUCCESS and read_res.content:
            raw: bytes
            if isinstance(read_res.content, (bytes, bytearray)):
                raw = bytes(read_res.content)[:10000]
            elif isinstance(read_res.content, str):
                raw = read_res.content.encode("utf-8", errors="replace")[:10000]
            else:
                return "utf-8"

            result = chardet.detect(raw)
            if result.get("encoding") and result.get("confidence", 0) > 0.7:
                return str(result["encoding"])
    except (ImportError, Exception):
        pass
    return "utf-8"


def _is_binary_file(file_path: str) -> bool:
    """Check if file is a binary file based on extension."""
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext in AUDIO_EXTENSIONS or ext == PDF_EXTENSION


def _detect_file_type(file_path: str) -> str:
    """Detect file type based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext == PDF_EXTENSION:
        return "pdf"
    return "text"


def _read_binary_file(file_path: str, sandbox: BaseSandbox) -> dict[str, Any]:
    """Read binary file and return as inline data part for LLM (via sandbox read)."""
    ext = Path(file_path).suffix.lower()

    res = sandbox.read_file(file_path, binary=True)
    if res.status != SandboxStatus.SUCCESS:
        raise RuntimeError(res.error or "Failed to read binary file")

    content: bytes
    if isinstance(res.content, (bytes, bytearray)):
        content = bytes(res.content)
    elif isinstance(res.content, str):
        content = res.content.encode("utf-8", errors="replace")
    else:
        content = b""

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        if ext in IMAGE_EXTENSIONS:
            mime_type = f"image/{ext[1:]}"
        elif ext in AUDIO_EXTENSIONS:
            mime_type = f"audio/{ext[1:]}"
        elif ext == PDF_EXTENSION:
            mime_type = "application/pdf"
        else:
            mime_type = "application/octet-stream"

    # Return in a format suitable for LLM (inline_data part)
    return {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(content).decode("utf-8"),
        }
    }


def read_file(
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
    agent_state: AgentState | None = None,
) -> dict[str, Any]:
    """
    Reads and returns the content of a specified file.

    If the file is large, the content will be truncated. The tool's response
    will clearly indicate if truncation has occurred and will provide details
    on how to read more of the file using the 'offset' and 'limit' parameters.

    Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), audio files
    (MP3, WAV, AIFF, AAC, OGG, FLAC), and PDF files.

    Args:
        file_path: The path to the file to read
        offset: Optional 0-based line number to start reading from
        limit: Optional maximum number of lines to read

    Returns:
        Dict with llmContent and returnDisplay matching gemini-cli format
    """
    try:
        sandbox = get_sandbox(agent_state)

        # Resolve path (relative -> sandbox work_dir)
        resolved_path = resolve_path(file_path, sandbox)

        # Check if file exists
        if not sandbox.file_exists(resolved_path):
            error_msg = f"File not found: {file_path}"
            return {
                "llmContent": error_msg,
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
                "llmContent": error_msg,
                "returnDisplay": "Path is a directory.",
                "error": {
                    "message": error_msg,
                    "type": "PATH_IS_DIRECTORY",
                },
            }

        # Check file size
        file_size = int(info.size or 0)
        if file_size > MAX_FILE_SIZE_BYTES:
            error_msg = f"File too large ({file_size} bytes). Maximum size is {MAX_FILE_SIZE_BYTES} bytes."
            return {
                "llmContent": error_msg,
                "returnDisplay": "File too large.",
                "error": {
                    "message": error_msg,
                    "type": "FILE_TOO_LARGE",
                },
            }

        # Handle binary files (images, audio, PDF)
        if _is_binary_file(resolved_path):
            binary_content = _read_binary_file(resolved_path, sandbox)
            file_type = _detect_file_type(resolved_path)
            return {
                "llmContent": binary_content,
                "returnDisplay": f"Read {file_type} file: {file_path}",
            }

        # Read text file
        encoding = _detect_encoding(resolved_path, sandbox)

        read_res = sandbox.read_file(resolved_path, encoding=encoding, binary=False)
        if read_res.status != SandboxStatus.SUCCESS:
            raise RuntimeError(read_res.error or "Failed to read file")

        content_str = read_res.content if isinstance(read_res.content, str) else ""
        all_lines = content_str.splitlines()
        total_lines = len(all_lines)

        # Apply offset and limit
        start_line = int(offset) if offset is not None and offset >= 0 else 0
        line_limit = int(limit) if limit is not None and limit > 0 else DEFAULT_LINE_LIMIT
        end_line = start_line + line_limit

        selected_lines = all_lines[start_line:end_line]
        content = "\n".join(selected_lines)
        lines_shown = len(selected_lines)

        # Check if truncated
        is_truncated = (end_line < total_lines) or bool(read_res.truncated)

        # Add line numbers
        content_with_lines = _add_line_numbers(content, start_line + 1)

        # Build result matching gemini-cli format
        if is_truncated:
            next_offset = start_line + lines_shown
            llm_content = f"""
IMPORTANT: The file content has been truncated.
Status: Showing lines {start_line + 1}-{start_line + lines_shown} of {total_lines} total lines.
Action: To read more of the file, you can use the 'offset' and 'limit' parameters in a subsequent
'read_file' call. For example, to read the next section of the file, use offset: {next_offset}.

--- FILE CONTENT (truncated) ---
{content_with_lines}"""
            return_display = f"Showing lines {start_line + 1}-{start_line + lines_shown} of {total_lines}"
        else:
            llm_content = content_with_lines
            return_display = f"Read {total_lines} lines"

        return {
            "llmContent": llm_content,
            "returnDisplay": return_display,
        }

    except PermissionError:
        error_msg = f"Permission denied: {file_path}"
        return {
            "llmContent": error_msg,
            "returnDisplay": "Permission denied.",
            "error": {
                "message": error_msg,
                "type": "PERMISSION_DENIED",
            },
        }
    except Exception as e:
        error_msg = f"Error reading file: {str(e)}"
        return {
            "llmContent": error_msg,
            "returnDisplay": "Error reading file.",
            "error": {
                "message": error_msg,
                "type": "READ_ERROR",
            },
        }
