#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for RFC-0019 cross-platform path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexau.archs.platform import path_helpers


def test_native_path_to_shell_path_returns_posix_on_unix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_helpers.sys, "platform", "linux")

    assert path_helpers.native_path_to_shell_path(Path("/tmp/work/file.txt")) == "/tmp/work/file.txt"


def test_native_path_to_shell_path_converts_windows_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_helpers.sys, "platform", "win32")

    assert path_helpers.native_path_to_shell_path(r"C:\Users\alice\repo") == "/c/Users/alice/repo"


def test_native_path_to_shell_path_handles_mixed_windows_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_helpers.sys, "platform", "win32")

    assert path_helpers.native_path_to_shell_path(r"D:/work\repo/file.txt") == "/d/work/repo/file.txt"


def test_native_path_to_shell_path_preserves_git_bash_absolute_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_helpers.sys, "platform", "win32")

    assert path_helpers.native_path_to_shell_path(r"/c/Users\alice") == "/c/Users/alice"


def test_native_path_to_shell_path_handles_unc_like_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_helpers.sys, "platform", "win32")

    assert path_helpers.native_path_to_shell_path(r"\\server\share\repo") == "//server/share/repo"


def test_native_path_to_shell_path_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(path_helpers.sys, "platform", "win32")

    assert path_helpers.native_path_to_shell_path("") == ""


def test_get_local_temp_root_uses_tempfile_gettempdir(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC-0020: production temp root uses tempfile, not a hardcoded /tmp path."""
    custom_temp = Path("C:/Users/alice/AppData/Local/Temp")
    monkeypatch.setattr(path_helpers.tempfile, "gettempdir", lambda: str(custom_temp))

    assert path_helpers.get_local_temp_root() == custom_temp


def test_get_local_bash_tool_results_dir_uses_temp_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC-0020: shell result artifacts are created under the detected temp root."""
    custom_temp = Path("D:/nexau-temp")
    monkeypatch.setattr(path_helpers.tempfile, "gettempdir", lambda: str(custom_temp))

    assert path_helpers.get_local_bash_tool_results_dir() == custom_temp / "nexau_bash_tool_results"


def test_get_local_tool_output_dir_uses_temp_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC-0020: long tool output artifacts are created under the detected temp root."""
    custom_temp = Path("D:/nexau-temp")
    monkeypatch.setattr(path_helpers.tempfile, "gettempdir", lambda: str(custom_temp))

    assert path_helpers.get_local_tool_output_dir() == custom_temp / "nexau_tool_outputs"


def test_get_local_cli_sessions_dir_uses_temp_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """RFC-0020: CLI session snapshots are created under the detected temp root."""
    custom_temp = Path("D:/nexau-temp")
    monkeypatch.setattr(path_helpers.tempfile, "gettempdir", lambda: str(custom_temp))

    assert path_helpers.get_local_cli_sessions_dir() == custom_temp / "nexau" / "cli-sessions"
