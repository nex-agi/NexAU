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
from nexau.archs.main_sub.utils.image_probe import (
    DEFAULT_IMAGE_MAX_PIXELS,
    OFFICIAL_PIXELS_PER_TOKEN,
    OVERSIZED_IMAGE_FILE_SIZE_BYTES,
    OVERSIZED_IMAGE_PIXELS,
    area_capped_dimensions,
    floor_even_dimension,
    probe_dimensions,
)
from nexau.archs.sandbox import BaseSandbox, CommandResult, SandboxStatus
from nexau.archs.tool.builtin._sandbox_utils import get_sandbox, resolve_path

logger = logging.getLogger(__name__)

# Oversized-image thresholds live in `image_probe` (single source, shared with
# the persistence-entry omit gate and mirrored by Rust `nexau-rs`): a file
# above `OVERSIZED_IMAGE_FILE_SIZE_BYTES` (20MB) is never pulled into process
# memory — it is compressed by ffmpeg inside the sandbox first — and an image
# probed above `OVERSIZED_IMAGE_PIXELS` (60MP) must likewise be compressed
# before it may reach the model. When ffmpeg is unavailable in the sandbox,
# reading such an image fails with a structured error instead of falling back
# to the original bytes (which would blow the context window or be rejected
# by the provider API). Only SVG keeps a plain size rejection: its rasterized
# PNG exists in memory, not at a sandbox path, so ffmpeg cannot compress it.
#
# Videos are streamed by ffmpeg (frame extraction), so they carry no size cap.

# Incident fix (session bf6ef5c923ce; Rust counterpart nexau-rs#94): a
# full-resolution image can push a single prompt past the model's context
# window, so when the caller doesn't pass `image_max_size`, images are
# downscaled to a pixel-area cap derived from a per-image token budget — the
# same shape as Claude Code's `readImageWithTokenBudget` (budget → physical
# size cap), except the budget→size exchange rate is pixels, not base64
# bytes, because providers bill images by pixel area.
#
# The budget constants and the area-cap geometry
# (`OFFICIAL_PIXELS_PER_TOKEN`, `DEFAULT_IMAGE_TOKEN_BUDGET`,
# `DEFAULT_IMAGE_MAX_PIXELS`, `floor_even_dimension`, `area_capped_dimensions`)
# live in `image_probe` — the shared low-level image module — so this
# ffmpeg-based read downscale, the Pillow-based persistence downscale
# (`resize_base64_image_if_oversized`) and the token counter all cap against the
# same exchange rate. They are imported above; the tests still import them from
# here (re-export) for calibration parity assertions.

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
FFMPEG_INSTALL_HINT = "Install ffmpeg into the sandbox image/template (or on the host for LocalSandbox), then retry."


class SvgConversionUnavailableError(RuntimeError):
    """Raised when SVG rasterization cannot run because Inkscape is unavailable."""


class FfmpegUnavailableError(RuntimeError):
    """Raised when an oversized image must be compressed but the sandbox has no ffmpeg."""


class OversizedImageCompressionError(RuntimeError):
    """Raised when an oversized image's mandatory ffmpeg compression failed.

    Oversized images (> ``OVERSIZED_IMAGE_FILE_SIZE_BYTES`` or
    > ``OVERSIZED_IMAGE_PIXELS``) have no graceful fallback: sending the
    original bytes would blow the context window or be rejected by the
    provider, so a failed compression surfaces as a structured tool error
    rather than degrading to the original image.
    """


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


_MISSING_PROGRAM_HINTS = (
    "command not found",
    "not found",
    "not recognized as an internal or external command",
    "is not recognized",
    "no such file or directory",
)


def _is_missing_program(result: CommandResult, program: str) -> bool:
    """Return True if a failed command result indicates *program* is unavailable.

    Exit code 127 is the POSIX shell's command-not-found signal. The textual
    hints (Windows cmd and friends) must co-occur with the program name on the
    same line: a shell reports a missing binary as e.g. ``ffmpeg: command not
    found``, whereas the program's own failure output (which for ffmpeg always
    includes a banner line containing its name) keeps hints like ``No such
    file or directory`` on lines that don't mention the program.
    """
    if result.exit_code == 127:
        return True
    output = _command_output_text(result).lower()
    return any(program in line and any(hint in line for hint in _MISSING_PROGRAM_HINTS) for line in output.splitlines())


def _is_missing_inkscape(result: CommandResult) -> bool:
    """Return True if a failed command result indicates Inkscape is unavailable."""
    return _is_missing_program(result, "inkscape")


def _is_missing_ffmpeg(result: CommandResult) -> bool:
    """Return True if a failed command result indicates ffmpeg is unavailable."""
    return _is_missing_program(result, "ffmpeg")


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
            if _is_missing_ffmpeg(cmd_result):
                raise RuntimeError(f"ffmpeg not found in sandbox. {FFMPEG_INSTALL_HINT}")
            diagnostics = _command_output_text(cmd_result)
            raise RuntimeError(f"ffmpeg failed (exit {cmd_result.exit_code}): {diagnostics[:500]}")

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
        return floor_even_dimension(max_edge), floor_even_dimension(height * max_edge / width)
    return floor_even_dimension(width * max_edge / height), floor_even_dimension(max_edge)


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
    single_frame: bool = False,
) -> bytes:
    """Re-encode image via ffmpeg in sandbox, returning the compressed JPEG bytes.

    Raises:
        FfmpegUnavailableError: ffmpeg is not installed in the sandbox.
        RuntimeError: ffmpeg ran but failed, or its output could not be read.

    The default-downscale caller (`_downscale_image_content`) catches all of
    these and degrades to the original bytes; the oversized-image caller
    (`_read_image_file`) lets them surface as structured tool errors because
    an oversized original must never reach the model.

    ``single_frame`` adds ``-frames:v 1``: without it, an animated GIF/WebP
    input makes ffmpeg fail against the single-file ``.jpg`` output ("Cannot
    write more than one file with the same name"). The mandatory oversized
    path sets it — first frame beats a hard error for an image that must be
    compressed to be readable at all. The graceful default path deliberately
    does NOT: there the ffmpeg failure falls back to the original bytes,
    preserving the animation (persistence-side "never flatten" semantics).
    """
    tmp_out = sandbox.join_path(sandbox.get_temp_dir(), f"nexau_resized_{uuid.uuid4().hex[:12]}.jpg")
    try:
        frames_arg = "-frames:v 1 " if single_frame else ""
        cmd = (
            f"ffmpeg -i {shlex.quote(sandbox.to_shell_path(file_path))} "
            f"-vf {shlex.quote(scale_expr)} {frames_arg}-q:v 2 {shlex.quote(sandbox.to_shell_path(tmp_out))} -y 2>&1"
        )
        result = sandbox.execute_shell(cmd, timeout=30_000)
        if result.status != SandboxStatus.SUCCESS or result.exit_code != 0:
            if _is_missing_ffmpeg(result):
                raise FfmpegUnavailableError("ffmpeg is not available in the sandbox")
            diagnostics = _command_output_text(result)
            raise RuntimeError(f"ffmpeg image compression failed (exit {result.exit_code}): {diagnostics[:500]}")

        res = sandbox.read_file(tmp_out, binary=True)
        if res.status != SandboxStatus.SUCCESS or not res.content:
            raise RuntimeError(res.error or "Failed to read ffmpeg-compressed image output")

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
            area_target = area_capped_dimensions(dimensions[0], dimensions[1], max_pixels)
            if area_target is not None:
                scale_expr = f"scale={area_target[0]}:{area_target[1]}"
            elif mime_type not in _API_SAFE_IMAGE_MIME_TYPES:
                scale_expr = _EVEN_DIMENSIONS_SCALE_EXPR
            else:
                return None

    # Downscale must never fail the read: since this runs by default on every
    # image, any ffmpeg/sandbox hiccup falls back to the original bytes rather
    # than surfacing a tool error (graceful-degradation parity with the Rust
    # `downscale_image_bytes`). Oversized images never take this path — their
    # compression is mandatory and raises instead (see `_read_image_file`).
    try:
        return _ffmpeg_scale_in_sandbox(file_path, sandbox, scale_expr)
    except Exception:
        logger.warning("Image downscale failed for %s; using original bytes", file_path, exc_info=True)
        return None


def _oversized_scale_expr(
    dimensions: tuple[int, int] | None,
    image_max_size: int | None,
    image_token_budget: int | None,
) -> str:
    """ffmpeg scale expression for an oversized image's mandatory compression.

    Mirrors `_downscale_image_content`'s bound resolution (explicit
    `image_max_size` wins, else the token-budget area cap), with two
    differences required by the oversized contract:

    - The `0`-disables-downscaling escape hatch does not apply: an image over
      the oversized thresholds must be compressed to be readable at all, so a
      non-positive bound falls back to the default token budget instead of
      disabling the resize.
    - There is no "within bound → None" outcome: a >20MB file whose pixel
      count is already within bound (e.g. an uncompressed BMP/TIFF) still
      needs the ffmpeg re-encode to shrink its byte size, so the
      even-dimensions pure-transcode expression is returned for that case.

    When `dimensions` is None (file too large to pull into memory for header
    probing, or an unprobeable format), the shrink-only longest-edge
    expression lets ffmpeg bound the size without knowing it up front.
    """

    # 不变量:强制压缩的产物永远 <= OVERSIZED_IMAGE_PIXELS。一个显式的大
    # `image_max_size`/`image_token_budget`(如 30000 / 巨大预算)会让 edge/area
    # 目标判定"界内"落到纯转码 —— 但 >60MP 的产物随后必被持久化 omit 成占位,
    # 工具却报成功。因此纯转码 fallback 前先按 60MP 硬上限回夹一次。
    def _transcode_or_hard_cap() -> str:
        if dimensions is not None:
            hard_target = area_capped_dimensions(dimensions[0], dimensions[1], OVERSIZED_IMAGE_PIXELS)
            if hard_target is not None:
                return f"scale={hard_target[0]}:{hard_target[1]}"
        return _EVEN_DIMENSIONS_SCALE_EXPR

    if image_max_size is not None and image_max_size > 0:
        if dimensions is None:
            return _longest_edge_scale_expr(image_max_size)
        edge_target = _edge_capped_dimensions(dimensions[0], dimensions[1], image_max_size)
        if edge_target is not None:
            return f"scale={edge_target[0]}:{edge_target[1]}"
        return _transcode_or_hard_cap()

    if image_token_budget is not None and image_token_budget > 0:
        max_pixels = image_token_budget * OFFICIAL_PIXELS_PER_TOKEN
    else:
        max_pixels = DEFAULT_IMAGE_MAX_PIXELS
    if dimensions is None:
        return _longest_edge_scale_expr(math.isqrt(max_pixels))
    area_target = area_capped_dimensions(dimensions[0], dimensions[1], max_pixels)
    if area_target is not None:
        return f"scale={area_target[0]}:{area_target[1]}"
    return _transcode_or_hard_cap()


def _compress_oversized_image(
    file_path: str,
    sandbox: BaseSandbox,
    dimensions: tuple[int, int] | None,
    image_max_size: int | None,
    image_token_budget: int | None,
    oversized_reason: str,
) -> bytes:
    """Mandatory sandbox-ffmpeg compression for an oversized image.

    Unlike the default downscale there is no original-bytes fallback: an
    oversized original must never reach the model, so a missing ffmpeg or a
    failed compression surfaces as a structured error.

    Raises:
        FfmpegUnavailableError: with the user-facing "install ffmpeg" message.
        OversizedImageCompressionError: ffmpeg ran but compression failed.
    """
    scale_expr = _oversized_scale_expr(dimensions, image_max_size, image_token_budget)
    try:
        return _ffmpeg_scale_in_sandbox(file_path, sandbox, scale_expr, single_frame=True)
    except FfmpegUnavailableError:
        raise FfmpegUnavailableError(
            f"Image file '{file_path}' is too large to read directly ({oversized_reason}). "
            "Reading it requires compressing it with ffmpeg inside the sandbox, but ffmpeg "
            f"is not available there — the image cannot be read. {FFMPEG_INSTALL_HINT} "
            "Alternatively, provide a smaller image."
        ) from None
    except Exception as e:
        raise OversizedImageCompressionError(
            f"Image file '{file_path}' is too large to read directly ({oversized_reason}) "
            f"and compressing it with ffmpeg in the sandbox failed: {e}"
        ) from e


def _read_image_file(
    file_path: str,
    sandbox: BaseSandbox,
    file_size: int,
    image_detail: str = "auto",
    image_max_size: int | None = None,
    image_token_budget: int | None = None,
) -> dict[str, Any]:
    """Read image file and return in nexau-supported format for LLM.

    Returns {"type": "image", "image_url": "data:...;base64,...", "detail": "..."}
    so coerce_tool_result_content can convert to ImageBlock.

    Oversized images (> `OVERSIZED_IMAGE_FILE_SIZE_BYTES` on disk, or probed
    > `OVERSIZED_IMAGE_PIXELS`) must be compressed by ffmpeg in the sandbox
    before anything reaches the model; a file over the byte threshold is never
    pulled into process memory at all — only the compressed output is read
    back. Within-bound images keep the graceful default downscale.
    """
    ext = Path(file_path).suffix.lower()
    content: bytes
    mime_type: str

    if ext == ".svg":
        content = _convert_svg_to_png_in_sandbox(file_path, sandbox)
        mime_type = "image/png"
    else:
        if file_size > OVERSIZED_IMAGE_FILE_SIZE_BYTES:
            # Over the byte threshold: never pull the original into process
            # memory — compress at the sandbox path and read back only the
            # (small) JPEG output. Dimensions stay unprobed (probing needs
            # bytes); the shrink-only ffmpeg expression bounds them anyway.
            compressed = _compress_oversized_image(
                file_path,
                sandbox,
                None,
                image_max_size,
                image_token_budget,
                f"{file_size} bytes > {OVERSIZED_IMAGE_FILE_SIZE_BYTES} bytes limit",
            )
            return {
                "type": "image",
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(compressed).decode('utf-8')}",
                "detail": image_detail,
            }

        res = sandbox.read_file(file_path, binary=True)
        if res.status != SandboxStatus.SUCCESS:
            raise RuntimeError(res.error or "Failed to read image file")

        if isinstance(res.content, (bytes, bytearray)):
            content = bytes(res.content)
        elif isinstance(res.content, str):
            content = res.content.encode("utf-8", errors="replace")
        else:
            content = b""

        dimensions = probe_dimensions(content)
        if dimensions is not None and dimensions[0] * dimensions[1] > OVERSIZED_IMAGE_PIXELS:
            compressed = _compress_oversized_image(
                file_path,
                sandbox,
                dimensions,
                image_max_size,
                image_token_budget,
                f"{dimensions[0]}x{dimensions[1]} = {dimensions[0] * dimensions[1]} pixels > {OVERSIZED_IMAGE_PIXELS} pixels limit",
            )
            return {
                "type": "image",
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(compressed).decode('utf-8')}",
                "detail": image_detail,
            }

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

    Oversized images (file > 20MB, or > 60 megapixels) are mandatorily
    compressed via ffmpeg in the sandbox before anything reaches the model —
    a >20MB file is never pulled into process memory, only its compressed
    output is. When the sandbox has no ffmpeg, reading such an image fails
    with an OVERSIZED_IMAGE_REQUIRES_FFMPEG error instead of falling back to
    the original bytes.

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
            file_size = int(info.size or 0)
            # SVG keeps a plain size rejection: it is rasterized via Inkscape
            # into in-memory PNG bytes, which ffmpeg (a file-path consumer)
            # cannot compress afterwards. Every other format goes through the
            # oversized-compression path inside `_read_image_file` instead of
            # being rejected up front.
            if Path(resolved_path).suffix.lower() == ".svg" and file_size > OVERSIZED_IMAGE_FILE_SIZE_BYTES:
                error_msg = (
                    f"SVG file too large ({file_size} bytes). Maximum size is {OVERSIZED_IMAGE_FILE_SIZE_BYTES} bytes"
                    " — oversized SVG cannot be compressed via ffmpeg because its rasterized form only exists in memory."
                )
                return {
                    "content": error_msg,
                    "returnDisplay": "SVG file too large.",
                    "error": {
                        "message": error_msg,
                        "type": "FILE_TOO_LARGE",
                    },
                }

            image_content = _read_image_file(
                resolved_path,
                sandbox,
                file_size,
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
    except FfmpegUnavailableError as e:
        error_msg = str(e)
        return {
            "content": error_msg,
            "returnDisplay": "Image too large; ffmpeg is unavailable in the sandbox so it cannot be compressed.",
            "error": {
                "message": error_msg,
                "type": "OVERSIZED_IMAGE_REQUIRES_FFMPEG",
            },
        }
    except OversizedImageCompressionError as e:
        error_msg = str(e)
        return {
            "content": error_msg,
            "returnDisplay": "Image too large; ffmpeg compression failed.",
            "error": {
                "message": error_msg,
                "type": "FILE_TOO_LARGE",
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
