# Copyright (c) Nex-AGI. All rights reserved.
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

"""Unit tests for CLI agent runner stdio configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.config.config import AgentConfig
from nexau.archs.main_sub.runtime_context import build_runtime_prompt_context
from nexau.archs.sandbox.base_sandbox import LocalSandboxConfig
from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.cli.agent_runner import CliAgentRuntime, _configure_stdio_for_utf8


class TestConfigureStdioForUtf8:
    def test_reconfigures_text_stdio_streams(self) -> None:
        stdin = SimpleNamespace(reconfigure=Mock())
        stdout = SimpleNamespace(reconfigure=Mock())
        stderr = SimpleNamespace(reconfigure=Mock())

        with (
            patch("nexau.cli.agent_runner.sys.stdin", stdin),
            patch("nexau.cli.agent_runner.sys.stdout", stdout),
            patch("nexau.cli.agent_runner.sys.stderr", stderr),
        ):
            _configure_stdio_for_utf8()

        for stream in (stdin, stdout, stderr):
            stream.reconfigure.assert_called_once_with(encoding="utf-8", errors="replace")

    def test_ignores_streams_without_reconfigure(self) -> None:
        with (
            patch("nexau.cli.agent_runner.sys.stdin", object()),
            patch("nexau.cli.agent_runner.sys.stdout", object()),
            patch("nexau.cli.agent_runner.sys.stderr", object()),
        ):
            _configure_stdio_for_utf8()


class TestRuntimePromptContext:
    def test_describes_existing_windows_powershell_sandbox(self) -> None:
        sandbox = SimpleNamespace(
            work_dir="C:\\workspace",
            _shell_backend=type("WindowsPowerShellBackend", (), {})(),
        )

        context = build_runtime_prompt_context(sandbox)

        assert context["working_directory"] == "C:\\workspace"
        assert context["shell_tool_backend"] == "Windows PowerShell backend"
        assert "PowerShell command syntax" in context["shell_tool_guidance"]
        assert "Write-Output" in context["shell_tool_guidance"]
        assert context["env_content"]["shell_tool_backend"] == context["shell_tool_backend"]

    def test_describes_windows_default_backend_before_sandbox_exists(self) -> None:
        installation = SimpleNamespace(kind="powershell")

        with (
            patch("nexau.archs.main_sub.runtime_context.sys.platform", "win32"),
            patch("nexau.archs.main_sub.runtime_context.os.getcwd", return_value="C:\\repo"),
            patch("nexau.archs.platform.shell_backend.configured_windows_shell_preference", return_value="default"),
            patch("nexau.archs.platform.shell_backend.ensure_default_windows_shell", return_value=installation),
        ):
            context = build_runtime_prompt_context(None)

        assert context["working_directory"] == "C:\\repo"
        assert context["platform"] == "win32"
        assert context["shell_tool_backend"] == "Windows PowerShell backend (powershell)"
        assert "PowerShell command syntax" in context["shell_tool_guidance"]


class TestCliAgentRuntimeBuild:
    def test_build_agent_from_config_does_not_prepend_cli_hooks_to_config(self) -> None:
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        runtime.config_path = Path("agent.yaml")
        runtime._llm_override = None
        runtime._restored_snapshot = None
        runtime._cli_progress_hook = Mock(name="cli_progress_hook")
        runtime._cli_tool_hook = Mock(name="cli_tool_hook")
        runtime.user_id = "user"

        user_after_model_hook = Mock(name="user_after_model_hook")
        user_after_tool_hook = Mock(name="user_after_tool_hook")
        captured_config: dict[str, object] = {}

        class FakeBuilder:
            def __init__(self, config, _base_path):  # type: ignore[no-untyped-def]
                captured_config.update(config)

            def __getattr__(self, _name):  # type: ignore[no-untyped-def]
                return lambda *args, **kwargs: self

            def get_agent_config(self):  # type: ignore[no-untyped-def]
                return "agent-config"

        fake_agent = Mock(name="agent")

        with (
            patch("nexau.cli.agent_runner.load_yaml_with_vars", return_value={"name": "agent"}),
            patch(
                "nexau.cli.agent_runner.normalize_agent_config_dict",
                return_value={
                    "name": "agent",
                    "after_model_hooks": [user_after_model_hook],
                    "after_tool_hooks": [user_after_tool_hook],
                },
            ),
            patch("nexau.cli.agent_runner.AgentConfigBuilder", FakeBuilder),
            patch("nexau.cli.agent_runner.Agent", return_value=fake_agent) as agent_cls,
            patch.object(runtime, "_build_restored_global_storage", return_value=None),
        ):
            agent = runtime._build_agent_from_config(session_id="session")

        assert agent is fake_agent
        assert captured_config["after_model_hooks"] == [user_after_model_hook]
        assert captured_config["after_tool_hooks"] == [user_after_tool_hook]
        agent_cls.assert_called_once()

    def test_git_bash_sandbox_start_does_not_probe_bash_from_cli_runtime(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RFC-0020: CLI/Agent startup must not run Git Bash discovery probes."""
        bash_path = tmp_path / "bash.exe"
        bash_path.write_text("", encoding="utf-8")
        work_dir = tmp_path / "work"
        runtime = CliAgentRuntime.__new__(CliAgentRuntime)
        runtime.config_path = tmp_path / "agent.yaml"
        runtime._llm_override = None
        runtime._restored_snapshot = None
        runtime._cli_progress_hook = Mock(name="cli_progress_hook")
        runtime._cli_tool_hook = Mock(name="cli_tool_hook")
        runtime.user_id = "user"

        agent_config = AgentConfig(
            name="git_bash_cli_no_probe",
            llm_config=LLMConfig(model="gpt-4o-mini"),
            sandbox_config=LocalSandboxConfig(work_dir=str(work_dir)),
        )

        class FakeBuilder:
            def __init__(self, config: dict[str, object], base_path: Path) -> None:
                self._config = config
                self._base_path = base_path

            def build_core_properties(self) -> FakeBuilder:
                return self

            def build_llm_config(self) -> FakeBuilder:
                return self

            def build_mcp_servers(self) -> FakeBuilder:
                return self

            def build_hooks(self) -> FakeBuilder:
                return self

            def build_tracers(self) -> FakeBuilder:
                return self

            def build_tools(self) -> FakeBuilder:
                return self

            def build_sub_agents(self) -> FakeBuilder:
                return self

            def build_skills(self) -> FakeBuilder:
                return self

            def build_system_prompt_path(self) -> FakeBuilder:
                return self

            def build_sandbox(self) -> FakeBuilder:
                return self

            def get_agent_config(self) -> AgentConfig:
                return agent_config

        probe_mock = Mock(side_effect=AssertionError("Git Bash discovery must not call subprocess.run"))
        with (
            monkeypatch.context() as patcher,
            patch("nexau.cli.agent_runner.load_yaml_with_vars", return_value={"name": "agent"}),
            patch("nexau.cli.agent_runner.normalize_agent_config_dict", return_value={"name": "agent"}),
            patch("nexau.cli.agent_runner.AgentConfigBuilder", FakeBuilder),
            patch.object(runtime, "_build_restored_global_storage", return_value=None),
            patch("nexau.archs.main_sub.agent.openai") as mock_openai,
            patch("subprocess.run", probe_mock),
        ):
            patcher.setenv("NEXAU_WINDOWS_SHELL_BACKEND", "git-bash")
            patcher.setenv("NEXAU_GIT_BASH_PATH", str(bash_path))
            patcher.setattr("nexau.archs.platform.shell_backend.sys.platform", "win32")
            patcher.setattr("nexau.archs.platform.git_bash.sys.platform", "win32")
            patcher.setattr("nexau.archs.platform.git_bash.shutil.which", lambda _command: None)
            patcher.setattr("nexau.archs.platform.git_bash._common_install_dir_candidates", lambda: [])
            mock_openai.OpenAI.return_value = Mock()
            mock_openai.AsyncOpenAI.return_value = Mock()

            agent = runtime._build_agent_from_config(session_id="session")
            sandbox = agent.sandbox_manager.start_sync()

        assert isinstance(sandbox, LocalSandbox)
        assert type(sandbox._shell_backend).__name__ == "WindowsGitBashBackend"
        probe_mock.assert_not_called()
