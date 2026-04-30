"""Tests for platform-aware test helpers.

RFC-0020: 测试基建平台化与 shell backend 差异治理

These tests verify the centralized test helper behavior for platform detection,
newline normalization, and temp/output/script directory generation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.utils.platform import (
    build_test_output_dir_root,
    build_test_script_dir_root,
    default_python_command,
    detect_test_platform,
    normalize_newlines,
)


def _make_which(mapping: dict[str, str]):
    def _resolve(name: str) -> str | None:
        return mapping.get(name)

    return _resolve


class TestDetectTestPlatform:
    def test_detect_posix_platform_uses_bash_and_python3(self) -> None:
        detected = detect_test_platform(
            "Linux",
            env={},
            which=_make_which({"bash": "/usr/bin/bash", "python3": "/usr/bin/python3", "rg": "/usr/bin/rg"}),
            temp_dir="/var/tmp/nexau-tests",
        )

        assert detected.is_windows is False
        assert detected.is_posix is True
        assert detected.git_bash_path == "/usr/bin/bash"
        assert detected.python_command == "/usr/bin/python3"
        assert detected.rg_path == "/usr/bin/rg"
        assert detected.temp_dir == Path("/var/tmp/nexau-tests")

    def test_detect_windows_platform_prefers_env_override_for_optional_git_bash(self) -> None:
        detected = detect_test_platform(
            "Windows",
            env={"NEXAU_TEST_GIT_BASH": r"C:\Git\bin\bash.exe"},
            which=_make_which({"python": r"C:\Python312\python.exe"}),
            temp_dir=r"C:/Users/test/AppData/Local/Temp",
        )

        assert detected.is_windows is True
        assert detected.is_posix is False
        assert detected.git_bash_path == r"C:\Git\bin\bash.exe"
        assert detected.python_command == r"C:\Python312\python.exe"
        assert detected.temp_dir == Path(r"C:/Users/test/AppData/Local/Temp")

    def test_detect_windows_platform_uses_discovered_optional_tools(self) -> None:
        detected = detect_test_platform(
            "Windows",
            env={},
            which=_make_which(
                {
                    "bash.exe": r"C:\Program Files\Git\bin\bash.exe",
                    "python": r"C:\Python312\python.exe",
                    "ffmpeg": r"C:\ffmpeg\bin\ffmpeg.exe",
                }
            ),
        )

        assert detected.git_bash_path == r"C:\Program Files\Git\bin\bash.exe"
        assert detected.ffmpeg_path == r"C:\ffmpeg\bin\ffmpeg.exe"


class TestPlatformPathBuilders:
    def test_output_dir_root_uses_tempfile_root_without_tmp_hardcode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tempfile.gettempdir", lambda: "/var/custom-temp")
        result = build_test_output_dir_root()
        assert result == Path("/var/custom-temp") / "nexau_bash_tool_results"

    def test_script_dir_root_uses_tempfile_root_without_tmp_hardcode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tempfile.gettempdir", lambda: r"C:/TempRoot")
        result = build_test_script_dir_root()
        assert result == Path(r"C:/TempRoot") / "nexau_bash_scripts"


class TestPlatformTextHelpers:
    def test_normalize_newlines_collapses_crlf_and_cr(self) -> None:
        assert normalize_newlines("a\r\nb\rc\n") == "a\nb\nc\n"

    def test_default_python_command_matches_platform(self) -> None:
        assert default_python_command("Windows") == "python"
        assert default_python_command("Linux") == "python3"
