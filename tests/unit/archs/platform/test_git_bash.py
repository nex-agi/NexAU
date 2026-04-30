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

"""Tests for RFC-0019 Git Bash discovery helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from nexau.archs.platform import git_bash
from nexau.archs.platform.git_bash import MissingGitBashError


def _make_bash(tmp_path: Path, name: str = "bash.exe") -> Path:
    bash_path = tmp_path / name
    bash_path.write_text("#!/bin/sh\n", encoding="utf-8")
    return bash_path


def test_detect_git_bash_returns_none_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_bash.sys, "platform", "linux")

    assert git_bash.detect_git_bash() is None


def test_detect_git_bash_prefers_valid_configured_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bash_path = _make_bash(tmp_path)
    monkeypatch.setattr(git_bash.sys, "platform", "win32")
    monkeypatch.setenv("NEXAU_GIT_BASH_PATH", str(bash_path))
    monkeypatch.setattr(git_bash.shutil, "which", lambda _command: None)
    monkeypatch.setattr(git_bash, "_common_install_dir_candidates", lambda: [])

    result = git_bash.detect_git_bash()

    assert result is not None
    assert result.bash_path == bash_path
    assert result.source == "configured"
    assert result.version is None


def test_detect_git_bash_skips_invalid_configured_path_and_uses_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bash_path = _make_bash(tmp_path)
    monkeypatch.setattr(git_bash.sys, "platform", "win32")
    monkeypatch.setenv("NEXAU_GIT_BASH_PATH", str(tmp_path / "missing-bash.exe"))
    monkeypatch.setattr(git_bash.shutil, "which", lambda _command: str(bash_path))
    monkeypatch.setattr(git_bash, "_common_install_dir_candidates", lambda: [])

    result = git_bash.detect_git_bash()

    assert result is not None
    assert result.bash_path == bash_path
    assert result.source == "path"


def test_detect_git_bash_does_not_run_health_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discovery must not execute bash --version on the hot path."""
    bash_path = _make_bash(tmp_path)
    monkeypatch.setattr(git_bash.sys, "platform", "win32")
    monkeypatch.delenv("NEXAU_GIT_BASH_PATH", raising=False)
    monkeypatch.delenv("NEXAU_BASH_PATH", raising=False)
    monkeypatch.setattr(git_bash.shutil, "which", lambda _command: str(bash_path))
    monkeypatch.setattr(git_bash, "_common_install_dir_candidates", lambda: [])
    run_mock = Mock(side_effect=AssertionError("bash --version should not be probed"))
    monkeypatch.setattr(git_bash, "subprocess", run_mock, raising=False)

    assert git_bash.detect_git_bash() is not None
    run_mock.assert_not_called()


def test_detect_git_bash_prefers_existing_configured_without_health_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured candidate wins by path without executing a health check."""
    configured = _make_bash(tmp_path, name="configured-bash.exe")
    path_candidate = _make_bash(tmp_path, name="path-bash.exe")
    monkeypatch.setattr(git_bash.sys, "platform", "win32")
    monkeypatch.setenv("NEXAU_GIT_BASH_PATH", str(configured))
    monkeypatch.setattr(git_bash.shutil, "which", lambda _command: str(path_candidate))
    monkeypatch.setattr(git_bash, "_common_install_dir_candidates", lambda: [])

    result = git_bash.detect_git_bash()

    assert result is not None
    assert result.bash_path == configured
    assert result.source == "configured"
    assert result.version is None


def test_ensure_git_bash_raises_unusable_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When a configured path is not a file, error mentions the path."""
    unusable = tmp_path / "bash.exe"
    unusable.mkdir()
    monkeypatch.setattr(git_bash.sys, "platform", "win32")
    monkeypatch.setattr(git_bash, "detect_git_bash", lambda: None)
    monkeypatch.setattr(git_bash, "_find_unusable_bash_paths", lambda: [unusable])

    with pytest.raises(MissingGitBashError, match="not an executable file") as exc_info:
        git_bash.ensure_git_bash()

    assert str(unusable) in str(exc_info.value)


def test_detect_git_bash_uses_common_install_dir_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bash_path = _make_bash(tmp_path)
    monkeypatch.setattr(git_bash.sys, "platform", "win32")
    monkeypatch.delenv("NEXAU_GIT_BASH_PATH", raising=False)
    monkeypatch.delenv("NEXAU_BASH_PATH", raising=False)
    monkeypatch.setattr(git_bash.shutil, "which", lambda _command: None)
    monkeypatch.setattr(git_bash, "_common_install_dir_candidates", lambda: [bash_path])

    result = git_bash.detect_git_bash()

    assert result is not None
    assert result.bash_path == bash_path
    assert result.source == "common-install-dir"


def test_ensure_git_bash_raises_clear_error_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(git_bash, "detect_git_bash", lambda: None)

    with pytest.raises(MissingGitBashError, match="Git Bash"):
        git_bash.ensure_git_bash()
