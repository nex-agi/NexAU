# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""
read_visual_file tool - Reads image and video files for multimodal LLMs.

Handles images (PNG, JPG, GIF, WEBP, SVG, BMP) and video files
(MP4, AVI, MOV, MKV, WEBM, FLV, WMV, M4V).
Video files are processed by extracting key frames via ffmpeg in the sandbox.
SVG files are converted to PNG via Inkscape because multimodal LLMs cannot
reliably consume raw SVG images.

For text files, use the read_file tool instead.
"""

import base64
import logging
import math
import mimetypes
import shlex
import uuid
from pathlib import Path
from typing import Any, Final

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.utils.image_probe import probe_dimensions
from nexau.archs.sandbox import BaseSandbox, CommandResult, SandboxStatus
from nexau.archs.tool.builtin._sandbox_utils import get_sandbox, resolve_path

logger = logging.getLogger(__name__)

# Max file size for images (videos are streamed by ffmpeg, so no size limit)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB

# Incident fix (session bf6ef5c923ce; Rust counterpart nexau-rs#94): a
# full-resolution image can push a single prompt past the model's context
# window, so when the caller doesn't pass `image_max_size`, images are
# downscaled to a pixel-area cap derived from a per-image token budget — the
# same shape as Claude Code's `readImageWithTokenBudget` (budget → physical
# size cap), except the budget→size exchange rate is pixels, not base64
# bytes, because providers bill images by pixel area.
#
# The official Anthropic vision formula is patch-based: an image costs
# ceil(width/28) x ceil(height/28) visual tokens — one token per 28x28-pixel
# patch, i.e. 784 pixels per token (older docs' pixels/750 was an
# approximation; the patch formula matched northgate measurements to ±1 token
# across five sizes on 2026-07-07). Per-image ceilings are tier-dependent:
# standard-tier models (Sonnet 4.6, Haiku, older) are server-downscaled to
# ≤1568px long edge / ≤1_568 tokens; high-resolution models (Opus 4.7/4.8,
# Fable 5) to ≤2576px / ≤4_784 tokens. Gateway channels implement those caps
# inconsistently, so we enforce the bound client-side. Budgets are
# denominated in official-formula tokens — the same patch formula
# token_counter charges context cost with, so a budgeted image and its
# accounted cost agree.
OFFICIAL_PIXELS_PER_TOKEN: Final[int] = 784

# Per-image token budget (official-formula tokens) applied when the caller
# passes neither `image_max_size` nor `image_token_budget`. 4_784 matches the
# official high-resolution tier's own per-image ceiling — the tier the
# production model (claude-opus-4-8) is in: 4_784 x 784 ≈ 3.75 megapixels,
# e.g. a ~2582x1452 16:9 screenshot at full fidelity. Standard-tier models
# server-downscale anything above ~1_568 tokens themselves, so the generous
# default costs nothing extra there. A full-cap image is both budgeted and
# accounted at ~4_784 official-formula tokens — deployments wanting image-heavy
# histories to compact sooner can lower this via the `image_token_budget`
# argument (binding `extra_kwargs`); accounting stays consistent for any
# setting because token_counter uses the same patch formula.
# Keep in sync with the Rust `DEFAULT_IMAGE_TOKEN_BUDGET` in nexau-rs.
DEFAULT_IMAGE_TOKEN_BUDGET: Final[int] = 4_784

# Pixel-area ceiling derived from the default budget. Bounding area — not
# edge length — matches how token cost actually scales: under a shared edge
# cap a square image costs ~4x a panoramic one, while an area cap prices
# every aspect ratio identically.
DEFAULT_IMAGE_MAX_PIXELS: Final[int] = DEFAULT_IMAGE_TOKEN_BUDGET * OFFICIAL_PIXELS_PER_TOKEN

# The only image media types the Anthropic API documents as supported.
# Anything else must be transcoded before reaching the model even when its
# pixel count is within bound. The public API rejects unknown media types with
# a 400 ("media_type: Input should be 'image/jpeg', 'image/png', 'image/gif'
# or 'image/webp'"). Behind the production gateway the symptom varies by
# upstream channel — measured live (northgate, 2026-07): raw image/tiff and
# image/bmp came back as a hard 400 on one channel, a 503 on another, and on a
# third a 200 whose image was silently dropped (model reports seeing nothing).
# All three mean the same thing: a non-whitelist format never reaches the
# model, so we re-encode to JPEG before sending.
_API_SAFE_IMAGE_MIME_TYPES: Final[frozenset[str]] = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})

# Pure-transcode scale expression: keeps dimensions (even-floored — ffmpeg's
# default mjpeg 4:2:0 pixel format requires even width/height) while
# re-encoding to JPEG. Used for within-bound images whose *format* is the
# problem rather than their size.
_EVEN_DIMENSIONS_SCALE_EXPR: Final[str] = "scale='trunc(iw/2)*2':'trunc(ih/2)*2'"

# Supported visual media types
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

# Video frame extraction settings
VIDEO_FRAME_INTERVAL_SEC = 5  # Extract one frame every 5 seconds
VIDEO_MAX_FRAMES = 10  # Return at most 10 frames

SVG_CONVERSION_HINT = (
    "SVG files cannot be read directly. Install Inkscape in the sandbox/host so read_visual_file can convert SVG to PNG, then retry."
)


class SvgConversionUnavailableError(RuntimeError):
    """Raised when SVG rasterization cannot run because Inkscape is unavailable."""


def _detect_file_type(file_path: str) -> str:
    """Detect visual file type based on extension."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def _command_output_text(result: CommandResult) -> str:
    """Combine command output streams for diagnostic matching."""
    parts: list[str] = []
    if result.stderr:
        parts.append(result.stderr)
    if result.stdout:
        parts.append(result.stdout)
    if result.error:
        parts.append(result.error)
    return "\n".join(parts)


def _is_missing_inkscape(result: CommandResult) -> bool:
    """Return True if a failed command result indicates Inkscape is unavailable."""
    output = _command_output_text(result).lower()
    if result.exit_code == 127:
        return True
    if "inkscape" not in output:
        return False
    missing_hints = (
        "command not found",
        "not found",
        "not recognized as an internal or external command",
        "is not recognized",
        "no such file or directory",
    )
    return any(hint in output for hint in missing_hints)


def _convert_svg_to_png_in_sandbox(
    file_path: str,
    sandbox: BaseSandbox,
) -> bytes:
    """Convert an SVG file to PNG bytes using Inkscape in the sandbox."""
    tmp_out = sandbox.join_path(sandbox.get_temp_dir(), f"nexau_svg_{uuid.uuid4().hex[:12]}.png")
    try:
        cmd = (
            f"inkscape {shlex.quote(sandbox.to_shell_path(file_path))} "
            f"--export-type=png "
            f"--export-filename={shlex.quote(sandbox.to_shell_path(tmp_out))} 2>&1"
        )
        result = sandbox.execute_shell(cmd, timeout=30_000)
        if result.status != SandboxStatus.SUCCESS or result.exit_code != 0:
            if _is_missing_inkscape(result):
                raise SvgConversionUnavailableError(SVG_CONVERSION_HINT)
            diagnostics = _command_output_text(result)
            raise RuntimeError(f"Inkscape SVG conversion failed (exit {result.exit_code}): {diagnostics[:500]}")

        res = sandbox.read_file(tmp_out, binary=True)
        if res.status != SandboxStatus.SUCCESS or not res.content:
            raise RuntimeError(res.error or "Failed to read converted SVG PNG")

        if isinstance(res.content, (bytes, bytearray)):
            return bytes(res.content)
        return res.content.encode("utf-8", errors="replace")
    finally:
        sandbox.delete_file(tmp_out)


def _read_video_frames(
    file_path: str,
    sandbox: BaseSandbox,
    frame_interval: int = VIDEO_FRAME_INTERVAL_SEC,
    max_frames: int = VIDEO_MAX_FRAMES,
    frame_width: int | None = None,
) -> list[dict[str, str]]:
    """Extract key frames from video using ffmpeg via sandbox.

    Args:
        file_path: Video file path (inside sandbox)
        sandbox: Sandbox instance
        frame_interval: Seconds between extracted frames
        max_frames: Maximum number of frames to return
        frame_width: Output frame width in pixels (preserving aspect ratio).
            None keeps original resolution.

    Returns:
        list of image dicts for coerce_tool_result_content

    Raises:
        ValueError: If numeric parameters are not positive integers.
    """
    # Sanitize numeric parameters to prevent ffmpeg filter injection
    frame_interval = int(frame_interval)
    max_frames = int(max_frames)
    if frame_interval <= 0:
        raise ValueError(f"frame_interval must be a positive integer, got {frame_interval}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be a positive integer, got {max_frames}")
    if frame_width is not None:
        frame_width = int(frame_width)
        if frame_width <= 0:
            raise ValueError(f"frame_width must be a positive integer, got {frame_width}")

    # 1. Create temp directory using sandbox-native path helpers.
    tmp_dir = sandbox.join_path(sandbox.get_temp_dir(), f"nexau_video_frames_{uuid.uuid4().hex[:12]}")
    sandbox.create_directory(tmp_dir, parents=True)

    try:
        out_pattern = sandbox.join_path(tmp_dir, "frame_%04d.jpg")

        # 2. Use ffmpeg to extract frames (with optional scaling)
        vf_filters = [f"fps=1/{frame_interval}"]
        if frame_width is not None and frame_width > 0:
            # scale=W:-2 preserves aspect ratio, -2 ensures even height (ffmpeg requirement)
            vf_filters.append(f"scale={frame_width}:-2")
        vf_str = ",".join(vf_filters)

        ffmpeg_cmd = (
            f"ffmpeg -i {shlex.quote(sandbox.to_shell_path(file_path))} "
            f"-vf {shlex.quote(vf_str)} -q:v 2 {shlex.quote(sandbox.to_shell_path(out_pattern))} -y 2>&1"
        )
        cmd_result = sandbox.execute_shell(ffmpeg_cmd, timeout=60_000)

        if cmd_result.status != SandboxStatus.SUCCESS or cmd_result.exit_code != 0:
            stderr = cmd_result.stderr or cmd_result.stdout or ""
            if "not found" in stderr.lower() or cmd_result.exit_code == 127:
                raise RuntimeError("ffmpeg not found in sandbox. Install ffmpeg to process video files.")
            raise RuntimeError(f"ffmpeg failed (exit {cmd_result.exit_code}): {stderr[:500]}")

        # 3. List extracted frame files via sandbox filesystem APIs.
        frame_infos = sandbox.list_files(tmp_dir, recursive=False, pattern="frame_*.jpg")
        frame_paths = sorted(info.path for info in frame_infos if info.is_file)
        if not frame_paths:
            raise RuntimeError("No frames extracted from video")

        # 4. Uniform sampling (when frame count exceeds limit)
        if len(frame_paths) > max_frames:
            step = len(frame_paths) / max_frames
            frame_paths = [frame_paths[int(i * step)] for i in range(max_frames)]

        # 5. Read each frame and convert to base64
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

            # Estimate timestamp from filename
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

        if not results:
            raise RuntimeError("Failed to read any extracted frames from video")

        return results
    finally:
        # Always cleanup temp directory, even on unexpected exceptions.
        sandbox.delete_file(tmp_dir)


def _floor_even_dimension(value: float) -> int:
    """Floor ``value`` to an even pixel count (minimum 2).

    ffmpeg's default mjpeg pixel format uses 4:2:0 chroma subsampling, which
    requires even dimensions — an odd target would make the encode fail and
    silently skip the downscale via the graceful-fallback path. Flooring
    (never rounding up) keeps every cap guarantee intact: an even-floored
    dimension is <= the exact scaled dimension.
    """
    floored = math.floor(value)
    return max(2, floored - (floored % 2))


def _edge_capped_dimensions(width: int, height: int, max_edge: int) -> tuple[int, int] | None:
    """Target dimensions for a longest-edge cap, or ``None`` if within bound.

    The governed (longest) edge lands on the cap (even-floored); the short
    side scales proportionally, so both dimensions stay <= the cap. Bounding
    the longest edge (rather than just the width, as the pre-incident
    `scale='min(max_width,iw)':-2` did) also caps tall-narrow images — e.g. a
    long mobile screenshot — whose pixel area, and therefore provider token
    cost, would otherwise stay unbounded.
    """
    if max(width, height) <= max_edge:
        return None
    if width >= height:
        return _floor_even_dimension(max_edge), _floor_even_dimension(height * max_edge / width)
    return _floor_even_dimension(width * max_edge / height), _floor_even_dimension(max_edge)


def _area_capped_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int] | None:
    """Target dimensions for a pixel-area cap, or ``None`` if within bound.

    Both dimensions scale by ``sqrt(max_pixels / area)`` and floor to even, so
    the resulting area can never exceed ``max_pixels`` — i.e. the image's real
    token cost can never exceed the budget the cap was derived from.
    """
    pixels = width * height
    if pixels <= max_pixels:
        return None
    scale = math.sqrt(max_pixels / pixels)
    return _floor_even_dimension(width * scale), _floor_even_dimension(height * scale)


def _longest_edge_scale_expr(max_edge: int) -> str:
    """Shrink-only, aspect-preserving ffmpeg expression capping the longest edge.

    Used only when the image's dimensions can't be probed locally (e.g.
    webp/tiff, which ffmpeg can still decode): on a landscape image bound the
    width (height auto via -2), otherwise bound the height.
    """
    # Sanitize numeric parameter to prevent ffmpeg filter injection
    max_edge = int(max_edge)
    if max_edge <= 0:
        raise ValueError(f"max_edge must be a positive integer, got {max_edge}")
    return f"scale='if(gt(iw,ih),min({max_edge},iw),-2)':'if(gt(iw,ih),-2,min({max_edge},ih))'"


def _ffmpeg_scale_in_sandbox(
    file_path: str,
    sandbox: BaseSandbox,
    scale_expr: str,
) -> bytes | None:
    """Re-encode image via ffmpeg in sandbox, returning JPEG bytes or None on failure."""
    tmp_out = sandbox.join_path(sandbox.get_temp_dir(), f"nexau_resized_{uuid.uuid4().hex[:12]}.jpg")
    try:
        cmd = (
            f"ffmpeg -i {shlex.quote(sandbox.to_shell_path(file_path))} "
            f"-vf {shlex.quote(scale_expr)} -q:v 2 {shlex.quote(sandbox.to_shell_path(tmp_out))} -y 2>&1"
        )
        result = sandbox.execute_shell(cmd, timeout=30_000)
        if result.status != SandboxStatus.SUCCESS or result.exit_code != 0:
            return None

        res = sandbox.read_file(tmp_out, binary=True)
        if res.status != SandboxStatus.SUCCESS or not res.content:
            return None

        if isinstance(res.content, (bytes, bytearray)):
            return bytes(res.content)
        return res.content.encode("utf-8", errors="replace")
    finally:
        sandbox.delete_file(tmp_out)


def _downscale_image_content(
    file_path: str,
    content: bytes,
    sandbox: BaseSandbox,
    image_max_size: int | None,
    image_token_budget: int | None,
    mime_type: str,
) -> bytes | None:
    """Downscale image bytes to their token bound via ffmpeg in the sandbox.

    Incident fix (Rust counterpart nexau-rs#94): 读图默认按 token 预算降采样封顶。

    Returns the downscaled JPEG bytes, or ``None`` meaning "keep the original
    bytes/mime unchanged" — either because the image is already within bound
    AND in an API-safe format (true no-op: such images are never re-encoded)
    or because downscaling can't proceed safely (graceful degradation; a read
    must never fail over an image-processing hiccup).

    The target size is computed locally from the image header
    (`probe_dimensions`) and passed to ffmpeg as an exact ``scale=W:H``, so
    the bound guarantees don't depend on ffmpeg expression arithmetic. Only
    when the header can't be parsed does it fall back to a shrink-only ffmpeg
    longest-edge expression — ffmpeg decodes formats the prober doesn't cover.

    Within-bound images whose format is outside `_API_SAFE_IMAGE_MIME_TYPES`
    (e.g. a probeable BMP) are still transcoded — without resizing — because
    the model can't see them otherwise (northgate silently drops
    `image/tiff`; the public API 400s).
    """
    dimensions = probe_dimensions(content)
    if image_max_size is not None:
        # Explicit caller bound: longest-edge semantics (schema parity).
        max_edge = int(image_max_size)
        if max_edge <= 0:
            return None
        if dimensions is None:
            scale_expr = _longest_edge_scale_expr(max_edge)
        else:
            edge_target = _edge_capped_dimensions(dimensions[0], dimensions[1], max_edge)
            if edge_target is not None:
                scale_expr = f"scale={edge_target[0]}:{edge_target[1]}"
            elif mime_type not in _API_SAFE_IMAGE_MIME_TYPES:
                scale_expr = _EVEN_DIMENSIONS_SCALE_EXPR
            else:
                return None
    else:
        # Default bound: pixel-area cap derived from the per-image token
        # budget (configured `image_token_budget` or the built-in default) —
        # identical worst-case cost for every aspect ratio. A budget of 0 is
        # the same explicit escape hatch as `image_max_size: 0`.
        if image_token_budget is not None:
            max_pixels = int(image_token_budget) * OFFICIAL_PIXELS_PER_TOKEN
            if max_pixels <= 0:
                return None
        else:
            max_pixels = DEFAULT_IMAGE_MAX_PIXELS
        if dimensions is None:
            # Unknown size: cap the longest edge at the square-equivalent
            # edge; both edges <= isqrt(A) bounds area at <= A for any aspect
            # ratio (modulo -2's even-rounding, ~0.1% slack at worst).
            scale_expr = _longest_edge_scale_expr(math.isqrt(max_pixels))
        else:
            area_target = _area_capped_dimensions(dimensions[0], dimensions[1], max_pixels)
            if area_target is not None:
                scale_expr = f"scale={area_target[0]}:{area_target[1]}"
            elif mime_type not in _API_SAFE_IMAGE_MIME_TYPES:
                scale_expr = _EVEN_DIMENSIONS_SCALE_EXPR
            else:
                return None

    # Downscale must never fail the read: since this runs by default on every
    # image, any ffmpeg/sandbox hiccup falls back to the original bytes rather
    # than surfacing a tool error (graceful-degradation parity with the Rust
    # `downscale_image_bytes`).
    try:
        return _ffmpeg_scale_in_sandbox(file_path, sandbox, scale_expr)
    except Exception:
        logger.warning("Image downscale failed for %s; using original bytes", file_path, exc_info=True)
        return None


def _read_image_file(
    file_path: str,
    sandbox: BaseSandbox,
    image_detail: str = "auto",
    image_max_size: int | None = None,
    image_token_budget: int | None = None,
) -> dict[str, Any]:
    """Read image file and return in nexau-supported format for LLM.

    Returns {"type": "image", "image_url": "data:...;base64,...", "detail": "..."}
    so coerce_tool_result_content can convert to ImageBlock.
    """
    ext = Path(file_path).suffix.lower()
    content: bytes
    mime_type: str

    if ext == ".svg":
        content = _convert_svg_to_png_in_sandbox(file_path, sandbox)
        mime_type = "image/png"
    else:
        res = sandbox.read_file(file_path, binary=True)
        if res.status != SandboxStatus.SUCCESS:
            raise RuntimeError(res.error or "Failed to read image file")

        if isinstance(res.content, (bytes, bytearray)):
            content = bytes(res.content)
        elif isinstance(res.content, str):
            content = res.content.encode("utf-8", errors="replace")
        else:
            content = b""

        guessed_mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = guessed_mime_type or f"image/{ext[1:]}"

    b64_str = base64.b64encode(content).decode("utf-8")

    # Downscale to bound token consumption (incident fix): by default images
    # are capped to `DEFAULT_IMAGE_MAX_PIXELS`; an explicit `image_max_size`
    # caps the longest edge instead. SVG is exempt: the rasterized PNG exists
    # only in memory, not at `file_path`, so ffmpeg (which reads the file)
    # can't rescale it — token accounting still charges its true pixel area.
    if ext != ".svg":
        resized = _downscale_image_content(file_path, content, sandbox, image_max_size, image_token_budget, mime_type)
        if resized is not None:
            b64_str = base64.b64encode(resized).decode("utf-8")
            # Resized output is always JPEG
            mime_type = "image/jpeg"

    return {
        "type": "image",
        "image_url": f"data:{mime_type};base64,{b64_str}",
        "detail": image_detail,
    }


def read_visual_file(
    file_path: str,
    image_detail: str | None = None,
    image_max_size: int | None = None,
    image_token_budget: int | None = None,
    video_frame_interval: int | None = None,
    video_max_frames: int | None = None,
    video_frame_width: int | None = None,
    agent_state: AgentState | None = None,
) -> dict[str, Any]:
    """
    Reads image and video files, returning visual content for multimodal LLMs.

    Supports images (PNG, JPG, GIF, WEBP, SVG, BMP) and video files
    (MP4, AVI, MOV, MKV, WEBM, FLV, WMV, M4V). SVG files are rasterized to
    PNG via Inkscape; when Inkscape is unavailable, the tool returns an
    SVG_REQUIRES_INKSCAPE error with an actionable hint. Video files are
    processed by extracting key frames via ffmpeg in the sandbox.

    For text files, use the read_file tool instead.

    Args:
        file_path: The path to the image or video file to read.
        image_detail: Image detail level for LLM ("low", "high", "auto"). Default "auto".
        image_max_size: Cap on the image's longest edge in pixels; larger
            images are downscaled via ffmpeg (preserving aspect ratio). None
            applies the token-budget pixel-area cap instead of keeping the
            original resolution.
        image_token_budget: Per-image token budget in official Anthropic
            formula tokens (28x28-pixel patches, 784 px/token); images are
            downscaled so their area is at most budget x 784 pixels. Default
            4_784 — the official high-resolution tier's per-image ceiling
            (≈3.75 megapixels). Intended for deployment configuration via
            binding extra_kwargs; 0 disables downscaling. Ignored when
            `image_max_size` is given.
        video_frame_interval: Seconds between extracted video frames. Default 5.
        video_max_frames: Maximum number of video frames to return. Default 10.
        video_frame_width: Width in pixels for extracted video frames. None keeps
            original resolution.

    Returns:
        Dict with content and returnDisplay for the agent framework.
    """
    try:
        sandbox = get_sandbox(agent_state)

        # Resolve path (relative -> sandbox work_dir)
        resolved_path = resolve_path(file_path, sandbox)

        # Sanitize and validate numeric parameters to prevent ffmpeg filter injection
        try:
            if video_frame_interval is not None:
                video_frame_interval = int(video_frame_interval)
            if video_max_frames is not None:
                video_max_frames = int(video_max_frames)
            if video_frame_width is not None:
                video_frame_width = int(video_frame_width)
            if image_max_size is not None:
                image_max_size = int(image_max_size)
            if image_token_budget is not None:
                image_token_budget = int(image_token_budget)
        except (TypeError, ValueError) as e:
            error_msg = f"Invalid parameter value (expected integer): {e}"
            return {
                "content": error_msg,
                "returnDisplay": "Invalid parameter.",
                "error": {
                    "message": error_msg,
                    "type": "INVALID_PARAMETER",
                },
            }

        if image_detail is not None and image_detail not in ("low", "high", "auto"):
            error_msg = f"Invalid image_detail value: {image_detail!r}. Must be 'low', 'high', or 'auto'."
            return {
                "content": error_msg,
                "returnDisplay": "Invalid parameter.",
                "error": {
                    "message": error_msg,
                    "type": "INVALID_PARAMETER",
                },
            }

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

        file_type = _detect_file_type(resolved_path)

        # Reject non-visual files
        if file_type == "unknown":
            ext = Path(resolved_path).suffix.lower()
            error_msg = f"File '{file_path}' ({ext}) is not an image or video file. Use the read_file tool for text files."
            return {
                "content": error_msg,
                "returnDisplay": f"Not a visual file ({ext}) — use read_file.",
                "error": {
                    "message": error_msg,
                    "type": "NOT_VISUAL_FILE",
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
            # Prepend description text before frame list
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

        # Handle image files
        if file_type == "image":
            # Check file size for images
            file_size = int(info.size or 0)
            if file_size > MAX_FILE_SIZE_BYTES:
                error_msg = f"Image file too large ({file_size} bytes). Maximum size is {MAX_FILE_SIZE_BYTES} bytes."
                return {
                    "content": error_msg,
                    "returnDisplay": "Image file too large.",
                    "error": {
                        "message": error_msg,
                        "type": "FILE_TOO_LARGE",
                    },
                }

            image_content = _read_image_file(
                resolved_path,
                sandbox,
                image_detail=image_detail or "auto",
                image_max_size=image_max_size,
                image_token_budget=image_token_budget,
            )
            return {
                "content": image_content,
                "returnDisplay": f"Read image file: {file_path}",
            }

        # Should not reach here
        error_msg = f"Unexpected file type for: {file_path}"
        return {
            "content": error_msg,
            "returnDisplay": "Unexpected file type.",
            "error": {
                "message": error_msg,
                "type": "UNEXPECTED_TYPE",
            },
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
    except SvgConversionUnavailableError as e:
        error_msg = str(e)
        return {
            "content": error_msg,
            "returnDisplay": "SVG cannot be read directly; install Inkscape to convert it to PNG.",
            "error": {
                "message": error_msg,
                "type": "SVG_REQUIRES_INKSCAPE",
            },
        }
    except Exception as e:
        error_msg = f"Error reading visual file: {str(e)}"
        return {
            "content": error_msg,
            "returnDisplay": "Error reading visual file.",
            "error": {
                "message": error_msg,
                "type": "READ_ERROR",
            },
        }
