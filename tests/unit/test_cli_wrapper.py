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

"""Unit tests for the backward-compatible nexau-cli wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from nexau.archs.platform.git_bash import MissingGitBashError
from nexau.cli.entrypoint_checks import MISSING_WINDOWS_SHELL_EXIT_CODE
from nexau.cli_wrapper import main, resolve_node_cli


class TestCliWrapperMain:
    """Test cases for cli_wrapper.main."""

    def test_main_exits_with_missing_windows_shell_code(self) -> None:
        with patch(
            "nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint",
            side_effect=RuntimeError("PowerShell is unavailable"),
        ):
            with pytest.raises(SystemExit, match=str(MISSING_WINDOWS_SHELL_EXIT_CODE)):
                main()

    def test_main_exits_with_missing_dependency_code_for_explicit_git_bash(self) -> None:
        with patch(
            "nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint",
            side_effect=MissingGitBashError("Git Bash is required"),
        ):
            with pytest.raises(SystemExit, match=str(MISSING_WINDOWS_SHELL_EXIT_CODE)):
                main()

    def test_main_proceeds_when_default_shell_check_passes(self) -> None:
        cli_path = Path("/tmp/cli.js")
        node_version_result = Mock(returncode=0)
        cli_run_result = Mock(returncode=0)

        with (
            patch("nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint") as mock_check,
            patch("nexau.cli_wrapper.resolve_node_cli", return_value=cli_path) as mock_resolve,
            patch("nexau.cli_wrapper.sys.argv", ["nexau-cli", "agent.yaml"]),
            patch(
                "nexau.cli_wrapper.subprocess.run",
                side_effect=[node_version_result, cli_run_result],
            ) as mock_run,
        ):
            with pytest.raises(SystemExit, match="0"):
                main()

        mock_check.assert_called_once_with()
        mock_resolve.assert_called_once_with()
        assert mock_run.call_count == 2
        assert mock_run.call_args_list[0].args[0] == ["node", "--version"]
        assert mock_run.call_args_list[1].args[0] == ["node", str(cli_path), "agent.yaml"]

    def test_main_reports_source_tree_build_failure(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint"),
            patch(
                "nexau.cli_wrapper.resolve_node_cli",
                side_effect=RuntimeError("npm is not installed or not in PATH"),
            ),
        ):
            with pytest.raises(SystemExit, match="1"):
                main()

        captured = capsys.readouterr()
        assert "npm is not installed or not in PATH" in captured.err

    def test_main_reports_missing_node_cli(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint"),
            patch("nexau.cli_wrapper.resolve_node_cli", return_value=None),
        ):
            with pytest.raises(SystemExit, match="1"):
                main()

        captured = capsys.readouterr()
        assert "Could not locate nexau Node.js CLI executable" in captured.err

    def test_main_reports_missing_node_binary(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint"),
            patch("nexau.cli_wrapper.resolve_node_cli", return_value=Path("/tmp/cli.js")),
            patch("nexau.cli_wrapper.subprocess.run", side_effect=FileNotFoundError),
        ):
            with pytest.raises(SystemExit, match="1"):
                main()

        captured = capsys.readouterr()
        assert "Node.js is not installed" in captured.err

    def test_main_returns_keyboard_interrupt_code(self, capsys: pytest.CaptureFixture[str]) -> None:
        node_version_result = Mock(returncode=0)

        with (
            patch("nexau.cli_wrapper.ensure_default_windows_shell_for_entrypoint"),
            patch("nexau.cli_wrapper.resolve_node_cli", return_value=Path("/tmp/cli.js")),
            patch("nexau.cli_wrapper.sys.argv", ["nexau-cli", "agent.yaml"]),
            patch("nexau.cli_wrapper.subprocess.run", side_effect=[node_version_result, KeyboardInterrupt]),
        ):
            with pytest.raises(SystemExit, match="130"):
                main()

        captured = capsys.readouterr()
        assert "Interrupted by user" in captured.err


class TestResolveNodeCli:
    def test_resolve_node_cli_returns_existing_cli(self, tmp_path: Path) -> None:
        cli_path = tmp_path / "cli.js"

        with patch("nexau.cli_wrapper.find_node_cli", return_value=cli_path):
            assert resolve_node_cli() == cli_path

    def test_resolve_node_cli_auto_builds_source_tree_dist(self, tmp_path: Path) -> None:
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        dist_path = cli_dir / "dist" / "cli.js"

        with (
            patch("nexau.cli_wrapper.find_node_cli", return_value=None),
            patch("nexau.cli_wrapper._source_tree_cli_paths", return_value=(cli_dir, dist_path)),
            patch("nexau.cli_wrapper.ensure_cli_built", return_value=dist_path) as mock_build,
        ):
            result = resolve_node_cli()

        assert result == dist_path
        mock_build.assert_called_once_with(cli_dir=cli_dir, cli_dist=dist_path)

    def test_resolve_node_cli_returns_none_without_existing_or_source_tree_cli(self) -> None:
        with (
            patch("nexau.cli_wrapper.find_node_cli", return_value=None),
            patch("nexau.cli_wrapper._source_tree_cli_paths", return_value=None),
        ):
            assert resolve_node_cli() is None
