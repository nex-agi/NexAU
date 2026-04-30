"""Unit tests for read_visual_file ffmpeg degradation paths."""

from __future__ import annotations

import base64
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.base_sandbox import CommandResult, FileOperationResult, SandboxStatus
from nexau.archs.tool.builtin.file_tools.read_visual_file import _read_image_file, _read_video_frames


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

        result = _read_image_file("/images/source.png", sandbox, image_max_size=320)

        assert result["image_url"] == f"data:image/png;base64,{base64.b64encode(original).decode('utf-8')}"
        assert result["detail"] == "auto"
