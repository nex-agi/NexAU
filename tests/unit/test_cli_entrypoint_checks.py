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

"""Tests for CLI entrypoint prerequisite checks."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nexau.archs.platform.git_bash import GitBashInstallation, MissingGitBashError
from nexau.cli.entrypoint_checks import (
    ensure_default_windows_shell_for_entrypoint,
    ensure_git_bash_for_entrypoint,
    handoff_git_bash_install_hint,
)


class TestEntrypointChecks:
    def test_handoff_hint_mentions_upper_level_workflow(self) -> None:
        hint = handoff_git_bash_install_hint()
        assert "upper-level agent" in hint
        assert "Git for Windows" in hint

    def test_non_windows_skips_git_bash_detection(self) -> None:
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "linux"),
            patch("nexau.cli.entrypoint_checks.ensure_git_bash") as mock_ensure,
        ):
            result = ensure_git_bash_for_entrypoint()

        assert result is None
        mock_ensure.assert_not_called()

    def test_windows_missing_git_bash_prints_handoff_and_reraises(self) -> None:
        stderr = StringIO()
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch(
                "nexau.cli.entrypoint_checks.ensure_git_bash",
                side_effect=MissingGitBashError("Git Bash is required"),
            ),
        ):
            with pytest.raises(MissingGitBashError, match="Git Bash is required"):
                ensure_git_bash_for_entrypoint(stderr=stderr)

        output = stderr.getvalue()
        assert "Error: Git Bash is required" in output
        assert "Handoff hint:" in output

    def test_windows_returns_detected_installation(self) -> None:
        installation = Mock()
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch("nexau.cli.entrypoint_checks.ensure_git_bash", return_value=installation) as mock_ensure,
        ):
            result = ensure_git_bash_for_entrypoint()

        assert result is installation
        mock_ensure.assert_called_once_with()

    def test_windows_default_entrypoint_uses_default_shell_not_git_bash(self) -> None:
        installation = Mock()
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch("nexau.cli.entrypoint_checks.configured_windows_shell_preference", return_value="default"),
            patch(
                "nexau.cli.entrypoint_checks.ensure_default_windows_shell",
                return_value=installation,
            ) as mock_default_shell,
            patch("nexau.cli.entrypoint_checks.ensure_git_bash") as mock_git_bash,
        ):
            result = ensure_default_windows_shell_for_entrypoint()

        assert result is installation
        mock_default_shell.assert_called_once_with()
        mock_git_bash.assert_not_called()

    def test_windows_explicit_git_bash_entrypoint_fails_fast_with_handoff(self) -> None:
        stderr = StringIO()
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch("nexau.cli.entrypoint_checks.configured_windows_shell_preference", return_value="git-bash"),
            patch(
                "nexau.cli.entrypoint_checks.ensure_git_bash",
                side_effect=MissingGitBashError("Git Bash is required"),
            ),
            patch("nexau.cli.entrypoint_checks.ensure_default_windows_shell") as mock_default_shell,
        ):
            with pytest.raises(MissingGitBashError, match="Git Bash is required"):
                ensure_default_windows_shell_for_entrypoint(stderr=stderr)

        output = stderr.getvalue()
        assert "Error: Git Bash is required" in output
        assert "Handoff hint:" in output
        mock_default_shell.assert_not_called()

    def test_windows_explicit_git_bash_entrypoint_returns_installation(self) -> None:
        installation = GitBashInstallation(
            bash_path=Path(r"C:\Program Files\Git\bin\bash.exe"),
            source="test",
            version="GNU bash",
        )
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch("nexau.cli.entrypoint_checks.configured_windows_shell_preference", return_value="git-bash"),
            patch("nexau.cli.entrypoint_checks.ensure_git_bash", return_value=installation) as mock_git_bash,
            patch("nexau.cli.entrypoint_checks.ensure_default_windows_shell") as mock_default_shell,
        ):
            result = ensure_default_windows_shell_for_entrypoint()

        assert result is installation
        mock_git_bash.assert_called_once_with()
        mock_default_shell.assert_not_called()

    def test_windows_entrypoint_can_reprobe_default_after_explicit_git_bash_failure(self) -> None:
        installation = Mock()
        preferences = iter(["git-bash", "default"])
        stderr = StringIO()
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch("nexau.cli.entrypoint_checks.configured_windows_shell_preference", side_effect=lambda: next(preferences)),
            patch(
                "nexau.cli.entrypoint_checks.ensure_git_bash",
                side_effect=MissingGitBashError("Git Bash is required"),
            ),
            patch(
                "nexau.cli.entrypoint_checks.ensure_default_windows_shell",
                return_value=installation,
            ) as mock_default_shell,
        ):
            with pytest.raises(MissingGitBashError):
                ensure_default_windows_shell_for_entrypoint(stderr=stderr)

            result = ensure_default_windows_shell_for_entrypoint(stderr=stderr)

        assert result is installation
        mock_default_shell.assert_called_once_with()

    def test_windows_default_entrypoint_reports_missing_shell(self) -> None:
        stderr = StringIO()
        with (
            patch("nexau.cli.entrypoint_checks.sys.platform", "win32"),
            patch("nexau.cli.entrypoint_checks.configured_windows_shell_preference", return_value="default"),
            patch(
                "nexau.cli.entrypoint_checks.ensure_default_windows_shell",
                side_effect=RuntimeError("Unable to locate pwsh.exe"),
            ),
        ):
            with pytest.raises(RuntimeError, match="Unable to locate pwsh.exe"):
                ensure_default_windows_shell_for_entrypoint(stderr=stderr)

        assert "Error: Unable to locate pwsh.exe" in stderr.getvalue()
