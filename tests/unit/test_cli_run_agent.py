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

"""Tests for the cross-platform run-agent launcher."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from nexau.archs.platform.git_bash import MissingGitBashError
from nexau.cli import run_agent
from nexau.cli.entrypoint_checks import MISSING_WINDOWS_SHELL_EXIT_CODE
from nexau.cli.run_agent import cli_needs_build, ensure_cli_built, main, run_node_cli


class TestCliNeedsBuild:
    def test_returns_true_when_dist_missing(self, tmp_path: Path) -> None:
        source = tmp_path / "cli-source.js"
        source.write_text("console.log('hi')", encoding="utf-8")

        assert cli_needs_build(tmp_path / "missing.js", (source,)) is True

    def test_returns_true_when_source_newer_than_dist(self, tmp_path: Path) -> None:
        dist = tmp_path / "cli.js"
        source = tmp_path / "source.js"
        dist.write_text("old", encoding="utf-8")
        source.write_text("new", encoding="utf-8")
        os.utime(dist, ns=(1_000_000_000, 1_000_000_000))
        os.utime(source, ns=(2_000_000_000, 2_000_000_000))

        assert cli_needs_build(dist, (source,)) is True

    def test_returns_false_when_dist_is_up_to_date(self, tmp_path: Path) -> None:
        dist = tmp_path / "cli.js"
        source = tmp_path / "source.js"
        source.write_text("src", encoding="utf-8")
        dist.write_text("dist", encoding="utf-8")
        dist.touch()

        assert cli_needs_build(dist, (source,)) is False


class TestEnsureCliBuilt:
    def test_raises_when_cli_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="CLI directory not found"):
            ensure_cli_built(cli_dir=tmp_path / "missing", cli_dist=tmp_path / "dist" / "cli.js")

    def test_builds_missing_dist_with_npm(self, tmp_path: Path) -> None:
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()
        cli_dist = cli_dir / "dist" / "cli.js"
        npm_path = tmp_path / "npm"

        def fake_run(command: list[str], *, cwd: Path, check: bool) -> None:
            assert command[0] == str(npm_path)
            assert cwd == cli_dir
            assert check is True
            if command == [str(npm_path), "run", "build"]:
                cli_dist.parent.mkdir()
                cli_dist.write_text("built", encoding="utf-8")

        with (
            patch("nexau.cli.run_agent._resolve_npm_binary", return_value=str(npm_path)),
            patch("nexau.cli.run_agent.subprocess.run", side_effect=fake_run) as mock_run,
        ):
            result = ensure_cli_built(cli_dir=cli_dir, cli_dist=cli_dist)

        assert result == cli_dist
        assert mock_run.call_count == 2

    def test_raises_when_build_does_not_produce_dist(self, tmp_path: Path) -> None:
        cli_dir = tmp_path / "cli"
        cli_dir.mkdir()

        with (
            patch("nexau.cli.run_agent._resolve_npm_binary", return_value="npm"),
            patch("nexau.cli.run_agent.subprocess.run"),
        ):
            with pytest.raises(RuntimeError, match="CLI build did not produce expected artifact"):
                ensure_cli_built(cli_dir=cli_dir, cli_dist=cli_dir / "dist" / "cli.js")


class TestRunNodeCli:
    def test_run_node_cli_invokes_node_with_project_root(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        cli_path = tmp_path / "cli.js"
        run_result = type("RunResult", (), {"returncode": 7})()

        with (
            patch("nexau.cli.run_agent.load_dotenv") as mock_load_dotenv,
            patch("nexau.cli.run_agent._resolve_node_binary", return_value="/usr/bin/node"),
            patch("nexau.cli.run_agent.subprocess.run", return_value=run_result) as mock_run,
        ):
            result = run_node_cli(config_path, cli_path)

        assert result == 7
        mock_load_dotenv.assert_called_once_with(dotenv_path=run_agent._PROJECT_ROOT / ".env", override=False)
        mock_run.assert_called_once()
        assert mock_run.call_args.args[0] == ["/usr/bin/node", str(cli_path), str(config_path)]
        assert mock_run.call_args.kwargs["cwd"] == run_agent._PROJECT_ROOT
        assert mock_run.call_args.kwargs["check"] is False


class TestRunAgentMain:
    def test_returns_one_for_missing_config(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        result = main([str(tmp_path / "missing.yaml")])

        captured = capsys.readouterr()
        assert result == 1
        assert "not found" in captured.err

    def test_returns_missing_windows_shell_exit_code(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        with (
            patch(
                "nexau.cli.run_agent.ensure_default_windows_shell_for_entrypoint",
                side_effect=RuntimeError("PowerShell is unavailable"),
            ),
            patch("nexau.cli.run_agent.ensure_cli_built") as mock_build,
        ):
            result = main([str(config_path)])

        assert result == MISSING_WINDOWS_SHELL_EXIT_CODE
        mock_build.assert_not_called()

    def test_returns_missing_dependency_code_for_explicit_git_bash_preflight(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        with (
            patch(
                "nexau.cli.run_agent.ensure_default_windows_shell_for_entrypoint",
                side_effect=MissingGitBashError("Git Bash is required"),
            ),
            patch("nexau.cli.run_agent.ensure_cli_built") as mock_build,
        ):
            result = main([str(config_path)])

        assert result == MISSING_WINDOWS_SHELL_EXIT_CODE
        mock_build.assert_not_called()

    def test_runs_cli_with_resolved_config_path(self, tmp_path: Path) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")
        cli_path = tmp_path / "cli.js"
        cli_path.write_text("console.log('ok')", encoding="utf-8")

        with (
            patch("nexau.cli.run_agent.ensure_default_windows_shell_for_entrypoint"),
            patch("nexau.cli.run_agent.ensure_cli_built", return_value=cli_path),
            patch("nexau.cli.run_agent.run_node_cli", return_value=0) as mock_run,
        ):
            result = main([str(config_path)])

        assert result == 0
        mock_run.assert_called_once_with(config_path.resolve(), cli_path)

    def test_returns_build_failure_exit_code(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        with (
            patch("nexau.cli.run_agent.ensure_default_windows_shell_for_entrypoint"),
            patch(
                "nexau.cli.run_agent.ensure_cli_built",
                side_effect=run_agent.subprocess.CalledProcessError(9, ["npm", "run", "build"]),
            ),
        ):
            result = main([str(config_path)])

        captured = capsys.readouterr()
        assert result == 9
        assert "CLI build failed" in captured.err

    def test_returns_one_for_runtime_build_error(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        config_path = tmp_path / "agent.yaml"
        config_path.write_text("name: test\n", encoding="utf-8")

        with (
            patch("nexau.cli.run_agent.ensure_default_windows_shell_for_entrypoint"),
            patch("nexau.cli.run_agent.ensure_cli_built", side_effect=RuntimeError("CLI directory not found")),
        ):
            result = main([str(config_path)])

        captured = capsys.readouterr()
        assert result == 1
        assert "CLI directory not found" in captured.err
