"""Unit tests for read_visual_file ffmpeg degradation paths."""

from __future__ import annotations

import base64
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from nexau.archs.main_sub.utils.image_probe import OVERSIZED_IMAGE_FILE_SIZE_BYTES
from nexau.archs.sandbox.base_sandbox import BaseSandbox, CommandResult, FileOperationResult, SandboxStatus
from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools.read_visual_file import (
    _EVEN_DIMENSIONS_SCALE_EXPR,
    _convert_svg_to_png_in_sandbox,
    _is_missing_ffmpeg,
    _is_missing_inkscape,
    _oversized_scale_expr,
    _read_image_file,
    _read_video_frames,
    read_visual_file,
)


class TestReadVisualFileFfmpegDegradation:
    def test_video_frame_extraction_reports_missing_ffmpeg(self) -> None:
        """RFC-0020: ffmpeg 缺失时视频路径返回可诊断错误。"""
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="ffmpeg: command not found",
            exit_code=127,
        )

        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            _read_video_frames("/videos/sample.mp4", sandbox)

        sandbox.delete_file.assert_called_once()

    def test_video_frame_extraction_uses_sandbox_file_apis_and_sorts_frames(self) -> None:
        """RFC-0020: frame directory handling stays backend-neutral on Windows."""
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = r"C:\Temp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/\\')}\\{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.list_files.return_value = [
            Mock(path=r"C:\Temp\nexau_video_frames_test\frame_0002.jpg", is_file=True),
            Mock(path=r"C:\Temp\nexau_video_frames_test\frame_0001.jpg", is_file=True),
            Mock(path=r"C:\Temp\nexau_video_frames_test\note.txt", is_file=False),
        ]
        sandbox.read_file.side_effect = [
            FileOperationResult(status=SandboxStatus.SUCCESS, file_path="frame_0001.jpg", content=b"one", size=3),
            FileOperationResult(status=SandboxStatus.SUCCESS, file_path="frame_0002.jpg", content=b"two", size=3),
        ]

        result = _read_video_frames(r"C:\videos\sample.mp4", sandbox, frame_interval=5, max_frames=10)

        sandbox.create_directory.assert_called_once()
        created_dir = sandbox.create_directory.call_args.args[0]
        sandbox.list_files.assert_called_once_with(created_dir, recursive=False, pattern="frame_*.jpg")
        sandbox.delete_file.assert_called_once_with(created_dir)
        assert [item["image_url"] for item in result] == [
            f"data:image/jpeg;base64,{base64.b64encode(b'one').decode('utf-8')}",
            f"data:image/jpeg;base64,{base64.b64encode(b'two').decode('utf-8')}",
        ]

    def test_image_resize_missing_ffmpeg_falls_back_to_original_image(self) -> None:
        """RFC-0020: ffmpeg 缺失时图片缩放降级为读取原图。"""
        original = b"fake-image-bytes"
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/images/source.png",
            content=original,
            size=len(original),
        )
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="ffmpeg: command not found",
            exit_code=127,
        )

        result = _read_image_file("/images/source.png", sandbox, len(original), image_max_size=320)

        assert result["image_url"] == f"data:image/png;base64,{base64.b64encode(original).decode('utf-8')}"
        assert result["detail"] == "auto"

    def test_e2b_executable_eio_is_not_misclassified_as_missing_ffmpeg(self) -> None:
        """E2B executable-layer EIO means infrastructure failure, not absent ffmpeg."""
        result = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="/bin/bash: line 1: /usr/bin/ffmpeg: Input/output error",
            exit_code=126,
        )

        assert _is_missing_ffmpeg(result) is False

    def test_video_failure_includes_transport_error_field(self) -> None:
        """E2B diagnostics may arrive in CommandResult.error rather than stderr."""
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            error="envd request failed: connection reset",
            exit_code=126,
        )

        with pytest.raises(RuntimeError, match="envd request failed: connection reset"):
            _read_video_frames("/videos/sample.mp4", sandbox)

        sandbox.delete_file.assert_called_once()


class TestReadVisualFileSvgConversion:
    def test_missing_inkscape_detection_handles_non_inkscape_errors(self) -> None:
        """Only missing-Inkscape diagnostics should use the install hint path."""
        result = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="permission denied while exporting",
            exit_code=1,
        )

        assert _is_missing_inkscape(result) is False

    def test_missing_inkscape_detection_handles_windows_message(self) -> None:
        """Windows command-not-recognized output should be treated as missing Inkscape."""
        result = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="'inkscape' is not recognized as an internal or external command",
            exit_code=1,
        )

        assert _is_missing_inkscape(result) is True

    def test_svg_is_converted_to_png_with_inkscape(self) -> None:
        """SVG files are rasterized to PNG before returning an image block."""
        png_bytes = b"converted-png-bytes"
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/tmp/converted.png",
            content=png_bytes,
            size=len(png_bytes),
        )

        result = _read_image_file("/images/icon.svg", sandbox, 256)

        assert result["image_url"] == f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"
        assert result["detail"] == "auto"
        cmd = sandbox.execute_shell.call_args.args[0]
        assert cmd.startswith("inkscape ")
        assert "--export-type=png" in cmd
        assert "--export-filename=" in cmd
        sandbox.delete_file.assert_called_once()

    def test_svg_missing_inkscape_returns_actionable_tool_error(self) -> None:
        """When Inkscape is unavailable, SVG reads return a clear hint instead of raw SVG."""
        sandbox = Mock()
        sandbox.work_dir = "/work"
        sandbox.file_exists.return_value = True
        info = Mock()
        info.is_directory = False
        info.size = 256
        sandbox.get_file_info.return_value = info
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="inkscape: command not found",
            exit_code=127,
        )
        agent_state = Mock()
        agent_state.get_sandbox.return_value = sandbox

        result = read_visual_file("diagram.svg", agent_state=agent_state)

        assert result["error"]["type"] == "SVG_REQUIRES_INKSCAPE"
        assert "SVG files cannot be read directly" in result["content"]
        assert "Install Inkscape" in result["content"]
        sandbox.read_file.assert_not_called()
        sandbox.delete_file.assert_called_once()

    def test_svg_conversion_failure_reports_inkscape_diagnostics(self) -> None:
        """Non-missing Inkscape failures should preserve conversion diagnostics."""
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="inkscape export failed: invalid svg",
            exit_code=2,
        )

        with pytest.raises(RuntimeError, match="Inkscape SVG conversion failed"):
            _convert_svg_to_png_in_sandbox("/images/broken.svg", sandbox)

        sandbox.read_file.assert_not_called()
        sandbox.delete_file.assert_called_once()

    def test_svg_conversion_readback_failure_reports_error(self) -> None:
        """Successful Inkscape execution still fails if the PNG cannot be read back."""
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.ERROR,
            file_path="/tmp/converted.png",
            error="converted file missing",
        )

        with pytest.raises(RuntimeError, match="converted file missing"):
            _convert_svg_to_png_in_sandbox("/images/icon.svg", sandbox)

        sandbox.delete_file.assert_called_once()

    def test_svg_conversion_accepts_text_png_readback(self) -> None:
        """String readback content is encoded defensively like other image reads."""
        sandbox = Mock()
        sandbox.get_temp_dir.return_value = "/tmp"
        sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
        sandbox.to_shell_path.side_effect = lambda path: str(path)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/tmp/converted.png",
            content="png-text",
            size=8,
        )

        assert _convert_svg_to_png_in_sandbox("/images/icon.svg", sandbox) == b"png-text"
        sandbox.delete_file.assert_called_once()


def _png_header_bytes(width: int, height: int) -> bytes:
    """Minimal PNG prefix whose IHDR reports the given dimensions (probe-parseable)."""
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def _oversized_sandbox(file_size: int) -> Mock:
    sandbox = Mock()
    sandbox.work_dir = "/work"
    sandbox.file_exists.return_value = True
    info = Mock()
    info.is_directory = False
    info.size = file_size
    sandbox.get_file_info.return_value = info
    sandbox.get_temp_dir.return_value = "/tmp"
    sandbox.join_path.side_effect = lambda base, child: f"{base.rstrip('/')}/{child}"
    sandbox.to_shell_path.side_effect = lambda path: str(path)
    return sandbox


def _agent_state_for(sandbox: BaseSandbox | Mock) -> Mock:
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


class TestOversizedScaleExpr:
    """Rust parity: mirrors nexau-rs `oversized_scale_expr_matches_python_semantics`."""

    def test_probed_dimensions_produce_exact_area_target(self) -> None:
        # sqrt(DEFAULT_IMAGE_MAX_PIXELS / 81e6) * 9000 = 1936.6 -> floor-even 1936.
        assert _oversized_scale_expr((9000, 9000), None, None) == "scale=1936:1936"

    def test_unknown_dimensions_fall_back_to_shrink_only_expression(self) -> None:
        expr = _oversized_scale_expr(None, None, None)
        assert "if(gt(iw,ih)" in expr
        assert "1936" in expr

    def test_explicit_max_edge_keeps_caller_semantics(self) -> None:
        assert _oversized_scale_expr((9000, 4500), 1000, None) == "scale=1000:500"

    def test_explicit_max_edge_with_unknown_dimensions_uses_shrink_only(self) -> None:
        expr = _oversized_scale_expr(None, 1000, None)
        assert "if(gt(iw,ih)" in expr
        assert "1000" in expr

    def test_within_bound_pixels_still_transcode_for_byte_size(self) -> None:
        # Oversized by bytes only (e.g. uncompressed BMP): pure transcode expr.
        assert _oversized_scale_expr((800, 600), None, None) == _EVEN_DIMENSIONS_SCALE_EXPR
        assert _oversized_scale_expr((800, 600), 2048, None) == _EVEN_DIMENSIONS_SCALE_EXPR

    def test_zero_budget_escape_hatch_does_not_apply_to_oversized(self) -> None:
        assert _oversized_scale_expr((9000, 9000), None, 0) == "scale=1936:1936"
        assert _oversized_scale_expr((9000, 9000), 0, 0) == "scale=1936:1936"

    def test_generous_explicit_cap_still_bounds_output_to_hard_pixel_limit(self) -> None:
        """A huge image_max_size must not let a >60MP product through to be
        omitted at persistence — the transcode fallback re-caps at 60MP."""
        expr = _oversized_scale_expr((9000, 8000), 30_000, None)
        assert expr.startswith("scale=")
        parts = expr.removeprefix("scale=").split(":")
        target_w, target_h = int(parts[0]), int(parts[1])
        assert target_w * target_h <= 60_000_000
        assert max(target_w, target_h) <= 8000


class TestReadVisualFileOversizedImages:
    """>20MB / >60MP images must be ffmpeg-compressed in the sandbox — no original fallback."""

    def test_compressed_output_readback_failure_surfaces_as_file_too_large(self) -> None:
        """ffmpeg succeeds but its output cannot be read back — still a structured error."""
        sandbox = _oversized_sandbox(OVERSIZED_IMAGE_FILE_SIZE_BYTES + 1)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.ERROR,
            file_path="/tmp/nexau_resized_test.jpg",
            error="output vanished",
        )

        result = read_visual_file("/images/huge.png", agent_state=_agent_state_for(sandbox))

        assert result["error"]["type"] == "FILE_TOO_LARGE"
        assert "output vanished" in result["content"]

    def test_oversized_by_bytes_compresses_via_ffmpeg_without_reading_original(self) -> None:
        """A >20MB image is compressed at its sandbox path; only the JPEG output is read back."""
        compressed = b"compressed-jpeg-bytes"
        sandbox = _oversized_sandbox(OVERSIZED_IMAGE_FILE_SIZE_BYTES + 1)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/tmp/nexau_resized_test.jpg",
            content=compressed,
            size=len(compressed),
        )

        result = read_visual_file("/images/huge.png", agent_state=_agent_state_for(sandbox))

        assert "error" not in result
        assert result["content"]["image_url"] == f"data:image/jpeg;base64,{base64.b64encode(compressed).decode('utf-8')}"
        # The original file must never be pulled into process memory: the only
        # read_file call is for the ffmpeg output in the temp dir.
        sandbox.read_file.assert_called_once()
        read_path = sandbox.read_file.call_args.args[0]
        assert read_path.startswith("/tmp/nexau_resized_")
        cmd = sandbox.execute_shell.call_args.args[0]
        assert cmd.startswith("ffmpeg ")
        # Dimensions are unknown (original never read), so the shrink-only
        # longest-edge expression must be used.
        assert "if(gt(iw,ih)" in cmd
        # Mandatory compression takes the first frame: without -frames:v 1 an
        # animated GIF/WebP input fails against the single .jpg output.
        assert "-frames:v 1" in cmd

    def test_oversized_by_bytes_missing_ffmpeg_returns_structured_error(self) -> None:
        """Without ffmpeg in the sandbox, an oversized image read fails with an actionable error."""
        sandbox = _oversized_sandbox(OVERSIZED_IMAGE_FILE_SIZE_BYTES + 1)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="ffmpeg: command not found",
            exit_code=127,
        )

        result = read_visual_file("/images/huge.png", agent_state=_agent_state_for(sandbox))

        assert result["error"]["type"] == "OVERSIZED_IMAGE_REQUIRES_FFMPEG"
        assert "too large" in result["content"]
        assert "Install ffmpeg" in result["content"]
        # Neither the original nor any output was read.
        sandbox.read_file.assert_not_called()

    def test_oversized_by_bytes_compression_failure_returns_file_too_large(self) -> None:
        """ffmpeg present but failing surfaces as FILE_TOO_LARGE, not a silent original fallback."""
        sandbox = _oversized_sandbox(OVERSIZED_IMAGE_FILE_SIZE_BYTES + 1)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="Error while decoding stream: unsupported pixel format",
            exit_code=1,
        )

        result = read_visual_file("/images/huge.png", agent_state=_agent_state_for(sandbox))

        assert result["error"]["type"] == "FILE_TOO_LARGE"
        assert "too large" in result["content"]
        assert "failed" in result["content"]
        sandbox.read_file.assert_not_called()

    def test_oversized_e2b_executable_eio_is_compression_failure_not_missing_ffmpeg(self) -> None:
        """A present-but-unreadable executable must retain EIO diagnostics, not an install hint."""
        sandbox = _oversized_sandbox(OVERSIZED_IMAGE_FILE_SIZE_BYTES + 1)
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="/bin/bash: line 1: /usr/bin/ffmpeg: Input/output error",
            exit_code=126,
        )

        result = read_visual_file("/images/huge.png", agent_state=_agent_state_for(sandbox))

        assert result["error"]["type"] == "FILE_TOO_LARGE"
        assert "Input/output error" in result["content"]
        assert "Install ffmpeg" not in result["content"]
        sandbox.read_file.assert_not_called()

    def test_oversized_by_pixels_forces_compression_with_exact_scale(self) -> None:
        """A small-byte but >60MP image is probed and compressed to the budget area cap."""
        original = _png_header_bytes(9000, 9000)  # 81MP, well over the 60MP limit
        compressed = b"compressed-jpeg-bytes"
        sandbox = _oversized_sandbox(len(original))
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.SUCCESS,
            stdout="",
            stderr="",
            exit_code=0,
        )
        sandbox.read_file.side_effect = [
            FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path="/images/dense.png",
                content=original,
                size=len(original),
            ),
            FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path="/tmp/nexau_resized_test.jpg",
                content=compressed,
                size=len(compressed),
            ),
        ]

        result = read_visual_file("/images/dense.png", agent_state=_agent_state_for(sandbox))

        assert "error" not in result
        assert result["content"]["image_url"] == f"data:image/jpeg;base64,{base64.b64encode(compressed).decode('utf-8')}"
        cmd = sandbox.execute_shell.call_args.args[0]
        # Probed dimensions produce an exact area-capped target:
        # sqrt(DEFAULT_IMAGE_MAX_PIXELS / 81e6) * 9000 = 1936.6 -> floor-even 1936.
        assert "scale=1936:1936" in cmd

    def test_oversized_by_pixels_missing_ffmpeg_errors_instead_of_falling_back(self) -> None:
        """>60MP images must not degrade to original bytes when ffmpeg is unavailable."""
        original = _png_header_bytes(9000, 9000)
        sandbox = _oversized_sandbox(len(original))
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/images/dense.png",
            content=original,
            size=len(original),
        )
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="ffmpeg: command not found",
            exit_code=127,
        )

        result = read_visual_file("/images/dense.png", agent_state=_agent_state_for(sandbox))

        assert result["error"]["type"] == "OVERSIZED_IMAGE_REQUIRES_FFMPEG"
        assert "9000x9000" in result["content"]
        assert "Install ffmpeg" in result["content"]

    def test_within_bounds_probeable_image_keeps_graceful_fallback(self) -> None:
        """Images over the token budget but under 60MP keep the degrade-to-original behavior."""
        original = _png_header_bytes(3000, 3000)  # 9MP: over the ~3.75MP budget, under 60MP
        sandbox = _oversized_sandbox(len(original))
        sandbox.read_file.return_value = FileOperationResult(
            status=SandboxStatus.SUCCESS,
            file_path="/images/small.png",
            content=original,
            size=len(original),
        )
        sandbox.execute_shell.return_value = CommandResult(
            status=SandboxStatus.ERROR,
            stderr="ffmpeg: command not found",
            exit_code=127,
        )

        result = read_visual_file("/images/small.png", agent_state=_agent_state_for(sandbox))

        assert "error" not in result
        assert result["content"]["image_url"] == f"data:image/png;base64,{base64.b64encode(original).decode('utf-8')}"

    def test_oversized_svg_is_rejected_without_compression_attempt(self) -> None:
        """SVG cannot be ffmpeg-compressed (rasterized in memory), so oversized SVG is rejected."""
        sandbox = _oversized_sandbox(OVERSIZED_IMAGE_FILE_SIZE_BYTES + 1)

        result = read_visual_file("/images/huge.svg", agent_state=_agent_state_for(sandbox))

        assert result["error"]["type"] == "FILE_TOO_LARGE"
        assert "SVG file too large" in result["content"]
        sandbox.execute_shell.assert_not_called()
        sandbox.read_file.assert_not_called()

    def test_real_ffmpeg_compresses_6000_square_bmp_over_20_mib(self, tmp_path: Path) -> None:
        """Real LocalSandbox smoke: a 6000x6000 BMP is compressed before full readback."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            pytest.skip("ffmpeg is not installed on this test host")

        source = tmp_path / "large-6000x6000.bmp"
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=6000x6000",
                "-frames:v",
                "1",
                str(source),
            ],
            check=True,
            capture_output=True,
        )
        source_size = source.stat().st_size
        assert source_size > OVERSIZED_IMAGE_FILE_SIZE_BYTES

        sandbox = LocalSandbox(sandbox_id="large-image-real-ffmpeg", work_dir=tmp_path)
        result = read_visual_file(source.name, agent_state=_agent_state_for(sandbox))

        assert "error" not in result
        image_url = result["content"]["image_url"]
        assert image_url.startswith("data:image/jpeg;base64,")
        compressed = base64.b64decode(image_url.partition(",")[2])
        assert 0 < len(compressed) < source_size
