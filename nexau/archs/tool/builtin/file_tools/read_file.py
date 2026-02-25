# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""
read_file tool - Reads and returns the content of a specified file.

Based on gemini-cli's read-file.ts implementation.
Handles text, images, audio files, video files, and PDF files.
Video files are processed by extracting key frames via ffmpeg in the sandbox.
"""

import base64
import logging
import mimetypes
import shlex
import uuid
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

# Supported media types for binary files (matching read_file_legacy / file_read_tool)
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
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aiff", ".aac", ".ogg", ".flac"}
PDF_EXTENSION = ".pdf"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

# Video frame extraction settings
VIDEO_FRAME_INTERVAL_SEC = 5  # 每隔 5 秒提取一帧
VIDEO_MAX_FRAMES = 10  # 最多返回 10 帧

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


def _detect_encoding(file_path: str, sandbox: BaseSandbox) -> str:
    """Detect file encoding with fallback to utf-8 (via sandbox read)."""
    try:
        import chardet

        read_res = sandbox.read_file(file_path, binary=True)
        if read_res.status == SandboxStatus.SUCCESS and read_res.content:
            raw: bytes
            if isinstance(read_res.content, (bytes, bytearray)):
                raw = bytes(read_res.content)[:10000]
            else:
                # content is str when not bytes/bytearray (sandbox returns str|bytes)
                raw = read_res.content.encode("utf-8", errors="replace")[:10000]

            result = chardet.detect(raw)
            if result.get("encoding") and result.get("confidence", 0) > 0.7:
                return str(result["encoding"])
    except (ImportError, Exception):
        pass
    return "utf-8"


def _is_binary_file(file_path: str) -> bool:
    """Check if file is a binary file based on extension."""
    ext = Path(file_path).suffix.lower()
    return ext in IMAGE_EXTENSIONS or ext in AUDIO_EXTENSIONS or ext == PDF_EXTENSION or ext in VIDEO_EXTENSIONS


def _detect_file_type(file_path: str) -> str:
    """Detect file type based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    elif ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext == PDF_EXTENSION:
        return "pdf"
    return "text"


def _read_video_frames(
    file_path: str,
    sandbox: BaseSandbox,
    frame_interval: int = VIDEO_FRAME_INTERVAL_SEC,
    max_frames: int = VIDEO_MAX_FRAMES,
    frame_width: int | None = None,
) -> list[dict[str, str]]:
    """Extract key frames from video using ffmpeg via sandbox.

    通过 sandbox 执行 ffmpeg 提取视频关键帧，返回 JPEG base64 图片列表。

    Args:
        file_path: 视频文件路径（sandbox 内）
        sandbox: sandbox 实例
        frame_interval: 帧提取间隔（秒）
        max_frames: 最大返回帧数
        frame_width: 输出帧宽度（像素），保持宽高比缩放。None 表示保持原始尺寸。

    Returns:
        list of image dicts for coerce_tool_result_content
    """
    # 1. 创建临时目录
    tmp_dir = f"/tmp/nexau_video_frames_{uuid.uuid4().hex[:12]}"
    sandbox.execute_bash(f"mkdir -p {shlex.quote(tmp_dir)}")

    out_pattern = f"{tmp_dir}/frame_%04d.jpg"

    # 2. 使用 ffmpeg 提取帧（可选缩放）
    vf_filters = [f"fps=1/{frame_interval}"]
    if frame_width is not None and frame_width > 0:
        # scale=W:-2 保持宽高比，-2 确保高度为偶数（ffmpeg 要求）
        vf_filters.append(f"scale={frame_width}:-2")
    vf_str = ",".join(vf_filters)

    ffmpeg_cmd = f"ffmpeg -i {shlex.quote(file_path)} -vf {shlex.quote(vf_str)} -q:v 2 {shlex.quote(out_pattern)} -y 2>&1"
    cmd_result = sandbox.execute_bash(ffmpeg_cmd, timeout=60_000)

    if cmd_result.status != SandboxStatus.SUCCESS or cmd_result.exit_code != 0:
        # 3. 清理临时目录
        sandbox.execute_bash(f"rm -rf {shlex.quote(tmp_dir)}")
        stderr = cmd_result.stderr or cmd_result.stdout or ""
        if "not found" in stderr.lower() or cmd_result.exit_code == 127:
            raise RuntimeError("ffmpeg not found in sandbox. Install ffmpeg to process video files.")
        raise RuntimeError(f"ffmpeg failed (exit {cmd_result.exit_code}): {stderr[:500]}")

    # 4. 列出提取的帧文件
    ls_result = sandbox.execute_bash(f"ls -1 {shlex.quote(tmp_dir)}/frame_*.jpg 2>/dev/null | sort")
    if ls_result.status != SandboxStatus.SUCCESS or not ls_result.stdout.strip():
        sandbox.execute_bash(f"rm -rf {shlex.quote(tmp_dir)}")
        raise RuntimeError("No frames extracted from video")

    frame_paths = [p.strip() for p in ls_result.stdout.strip().splitlines() if p.strip()]

    # 5. 均匀采样（帧数超过上限时）
    if len(frame_paths) > max_frames:
        step = len(frame_paths) / max_frames
        frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

    # 6. 读取每帧并转为 base64
    results: list[dict[str, str]] = []
    for i, fpath in enumerate(frame_paths):
        res = sandbox.read_file(fpath, binary=True)
        if res.status != SandboxStatus.SUCCESS or not res.content:
            continue

        raw: bytes
        if isinstance(res.content, (bytes, bytearray)):
            raw = bytes(res.content)
        else:
            raw = res.content.encode("utf-8", errors="replace")

        b64 = base64.b64encode(raw).decode("utf-8")

        # 从文件名推算时间戳
        fname = Path(fpath).stem
        try:
            frame_num = int(fname.split("_")[-1]) - 1
        except ValueError:
            frame_num = i
        timestamp = frame_num * frame_interval

        results.append(
            {
                "type": "image",
                "image_url": f"data:image/jpeg;base64,{b64}",
                "detail": "auto",
                "label": f"Frame {i + 1} / ~{timestamp}s",
            }
        )

    # 7. 清理临时目录
    sandbox.execute_bash(f"rm -rf {shlex.quote(tmp_dir)}")

    if not results:
        raise RuntimeError("Failed to read any extracted frames from video")

    return results


def _resize_image_in_sandbox(
    file_path: str,
    sandbox: BaseSandbox,
    max_width: int,
) -> bytes | None:
    """Resize image via ffmpeg in sandbox, returning JPEG bytes or None on failure.

    通过 sandbox 中的 ffmpeg 缩放图片，保持宽高比。
    仅当图片宽度超过 max_width 时才缩放。
    """
    tmp_out = f"/tmp/nexau_resized_{uuid.uuid4().hex[:12]}.jpg"
    # scale='min(max_width,iw)':-2 只在宽度超过阈值时缩小，-2 保证偶数高度
    scale_expr = f"scale='min({max_width},iw)':-2"
    cmd = f"ffmpeg -i {shlex.quote(file_path)} -vf {shlex.quote(scale_expr)} -q:v 2 {shlex.quote(tmp_out)} -y 2>&1"
    result = sandbox.execute_bash(cmd, timeout=30_000)
    if result.status != SandboxStatus.SUCCESS or result.exit_code != 0:
        sandbox.execute_bash(f"rm -f {shlex.quote(tmp_out)}")
        return None

    res = sandbox.read_file(tmp_out, binary=True)
    sandbox.execute_bash(f"rm -f {shlex.quote(tmp_out)}")
    if res.status != SandboxStatus.SUCCESS or not res.content:
        return None

    if isinstance(res.content, (bytes, bytearray)):
        return bytes(res.content)
    return res.content.encode("utf-8", errors="replace")


def _read_binary_file(
    file_path: str,
    sandbox: BaseSandbox,
    image_detail: str = "auto",
    image_max_size: int | None = None,
) -> dict[str, Any]:
    """Read binary file and return in nexau-supported format for LLM (via sandbox read).

    For images: returns {"type": "image", "image_url": "data:...;base64,...", "detail": "..."}
    so coerce_tool_result_content can convert to ImageBlock.
    For audio/PDF: returns text placeholder (nexau has no AudioBlock/PDFBlock).

    Args:
        file_path: 文件路径（sandbox 内）
        sandbox: sandbox 实例
        image_detail: 图片 detail 级别，传递给 LLM（"low"/"high"/"auto"）
        image_max_size: 图片最大宽度（像素），通过 ffmpeg 缩放。None 表示保持原始尺寸。
    """
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

    b64_str = base64.b64encode(content).decode("utf-8")

    if ext in IMAGE_EXTENSIONS:
        # 可选：通过 ffmpeg 缩放图片以减少 token 消耗
        if image_max_size is not None and image_max_size > 0 and ext != ".svg":
            resized = _resize_image_in_sandbox(file_path, sandbox, image_max_size)
            if resized is not None:
                b64_str = base64.b64encode(resized).decode("utf-8")
                # 缩放后统一输出为 JPEG
                mime_type = "image/jpeg"

        return {
            "type": "image",
            "image_url": f"data:{mime_type};base64,{b64_str}",
            "detail": image_detail,
        }
    # Audio/PDF: nexau coerce has no block type; return placeholder
    return {"type": "text", "text": f"Read binary file ({ext}) - content not displayed to model"}


def read_file(
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
    image_detail: str | None = None,
    image_max_size: int | None = None,
    video_frame_interval: int | None = None,
    video_max_frames: int | None = None,
    video_frame_width: int | None = None,
    agent_state: AgentState | None = None,
) -> dict[str, Any]:
    """
    Reads and returns the content of a specified file.

    If the file is large, the content will be truncated. The tool's response
    will clearly indicate if truncation has occurred and will provide details
    on how to read more of the file using the 'offset' and 'limit' parameters.

    Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), audio files
    (MP3, WAV, AIFF, AAC, OGG, FLAC), video files (MP4, AVI, MOV, MKV,
    WEBM, FLV, WMV, M4V), and PDF files.

    Video files are processed by extracting key frames via ffmpeg in the
    sandbox, returning them as a sequence of JPEG images.

    Args:
        file_path: The path to the file to read
        offset: Optional 0-based line number to start reading from
        limit: Optional maximum number of lines to read
        image_detail: Image detail level for LLM ("low", "high", "auto"). Default "auto".
        image_max_size: Max image width in pixels; images wider than this are
            downscaled via ffmpeg (preserving aspect ratio). None keeps original.
        video_frame_interval: Seconds between extracted video frames. Default 5.
        video_max_frames: Maximum number of video frames to return. Default 10.
        video_frame_width: Width in pixels for extracted video frames. None keeps
            original resolution.

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

        # Check file size (skip for video - ffmpeg streams without loading into memory)
        file_size = int(info.size or 0)
        file_type = _detect_file_type(resolved_path)
        if file_type != "video" and file_size > MAX_FILE_SIZE_BYTES:
            error_msg = f"File too large ({file_size} bytes). Maximum size is {MAX_FILE_SIZE_BYTES} bytes."
            return {
                "content": error_msg,
                "returnDisplay": "File too large.",
                "error": {
                    "message": error_msg,
                    "type": "FILE_TOO_LARGE",
                },
            }

        # Handle video files (extract key frames via ffmpeg)
        if file_type == "video":
            v_interval = video_frame_interval if video_frame_interval is not None else VIDEO_FRAME_INTERVAL_SEC
            v_max = video_max_frames if video_max_frames is not None else VIDEO_MAX_FRAMES
            frames = _read_video_frames(
                resolved_path,
                sandbox,
                frame_interval=v_interval,
                max_frames=v_max,
                frame_width=video_frame_width,
            )
            num_frames = len(frames)
            # 在帧列表前插入描述文本
            content_parts: list[dict[str, str]] = [
                {
                    "type": "text",
                    "text": (f"Video: {file_path} ({num_frames} key frames extracted, 1 frame every {v_interval}s)"),
                },
                *frames,
            ]
            return {
                "content": content_parts,
                "returnDisplay": f"Read video file: {file_path} ({num_frames} frames)",
            }

        # Handle binary files (images, audio, PDF)
        if _is_binary_file(resolved_path):
            binary_content = _read_binary_file(
                resolved_path,
                sandbox,
                image_detail=image_detail or "auto",
                image_max_size=image_max_size,
            )
            # Use "content" so coerce_tool_result_content extracts and converts to ImageBlock
            return {
                "content": binary_content,
                "returnDisplay": f"Read {file_type} file: {file_path}",
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

        # Truncate long lines (matching gemini-cli MAX_LINE_LENGTH_TEXT_FILE)
        lines_were_truncated_in_length = False
        formatted_lines: list[str] = []
        for line in selected_lines:
            if len(line) > MAX_LINE_LENGTH:
                lines_were_truncated_in_length = True
                formatted_lines.append(line[:MAX_LINE_LENGTH] + "... [truncated]")
            else:
                formatted_lines.append(line)

        content = "\n".join(formatted_lines)
        lines_shown = len(selected_lines)

        # Check if truncated (by line count or by line length)
        is_truncated = (end_line < total_lines) or bool(read_res.truncated) or lines_were_truncated_in_length

        # Add line numbers
        content_with_lines = _add_line_numbers(content, start_line + 1)

        # Build result matching gemini-cli format
        content_range_truncated = end_line < total_lines or bool(read_res.truncated)
        if is_truncated:
            next_offset = start_line + lines_shown
            llm_content = f"""
IMPORTANT: The file content has been truncated.
Status: Showing lines {start_line + 1}-{start_line + lines_shown} of {total_lines} total lines.
Action: To read more of the file, you can use the 'offset' and 'limit' parameters in a subsequent
'read_file' call. For example, to read the next section of the file, use offset: {next_offset}.

--- FILE CONTENT (truncated) ---
{content_with_lines}"""
            if content_range_truncated:
                return_display = f"Showing lines {start_line + 1}-{start_line + lines_shown} of {total_lines}"
            else:
                return_display = f"Read all {total_lines} lines from {file_path}"
            if lines_were_truncated_in_length:
                return_display += " (some lines were shortened)"
        else:
            llm_content = content_with_lines
            return_display = f"Read {total_lines} lines"

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
