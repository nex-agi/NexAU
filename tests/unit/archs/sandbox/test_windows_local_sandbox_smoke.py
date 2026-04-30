"""Windows LocalSandbox smoke tests from the RFC-0019 manual checklist.

RFC-0020: Absorb W-16 through W-19 into the Windows target test suite.
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.execution.middleware.runtime_environment import RuntimeEnvironmentMiddleware
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.platform.git_bash import detect_git_bash
from nexau.archs.platform.shell_backend import WindowsPowerShellBackend
from nexau.archs.sandbox.base_sandbox import LocalSandboxConfig, SandboxError, SandboxStatus
from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool import Tool
from nexau.archs.tool.builtin import run_shell_command
from nexau.core.messages import Message, Role, ToolResultBlock, ToolUseBlock


def _windows_smoke_shell_preferences() -> list[object]:
    """Return shell backend preferences covered by Windows smoke tests.

    RFC-0019/RFC-0020: Windows smoke coverage always exercises PowerShell and
    opportunistically exercises Git Bash when it is installed on the runner.
    """
    preferences: list[object] = [pytest.param("powershell", id="powershell")]
    if detect_git_bash() is not None:
        preferences.append(pytest.param("git-bash", id="git-bash"))
    return preferences


@pytest.fixture(params=_windows_smoke_shell_preferences())
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> LocalSandbox:
    shell_preference = request.param
    assert isinstance(shell_preference, str)
    monkeypatch.setenv("NEXAU_WINDOWS_SHELL_BACKEND", shell_preference)
    return LocalSandbox(sandbox_id=f"windows_smoke_{shell_preference}", work_dir=tmp_path)


def _backend_name(sandbox: LocalSandbox) -> str:
    return type(sandbox._shell_backend).__name__


def _scripted_agent_shell_preferences() -> list[object]:
    """Return shell preferences for scripted agent Windows smoke coverage."""
    preferences: list[object] = [pytest.param(None, None, id="default")]
    if detect_git_bash() is not None:
        preferences.append(pytest.param("git-bash", "WindowsGitBashBackend", id="git-bash"))
    return preferences


_QUICK_SORT_SOURCE = """def quick_sort(items):
    if len(items) <= 1:
        return list(items)
    pivot = items[len(items) // 2]
    left = [item for item in items if item < pivot]
    middle = [item for item in items if item == pivot]
    right = [item for item in items if item > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""

_QUICK_SORT_TESTS = """from quick_sort import quick_sort


def test_quick_sort_numbers():
    assert quick_sort([3, 1, 2, 3, -1]) == [-1, 1, 2, 3, 3]


def test_quick_sort_empty():
    assert quick_sort([]) == []


def test_quick_sort_strings():
    assert quick_sort([\"b\", \"a\", \"c\"]) == [\"a\", \"b\", \"c\"]
"""


def _python_writer_command(sandbox: LocalSandbox, marker: str) -> str:
    payload = {
        "quick_sort.py": base64.b64encode(_QUICK_SORT_SOURCE.encode()).decode(),
        "test_quick_sort.py": base64.b64encode(_QUICK_SORT_TESTS.encode()).decode(),
    }
    script = (
        "import base64, pathlib; "
        f"payload={payload!r}; "
        "[pathlib.Path(name).write_text(base64.b64decode(data).decode(), encoding='utf-8') for name, data in payload.items()]"
    )
    return f'echo SHELL_BACKEND={marker} && {sandbox.get_python_command()} -c "{script}"'


def _quicksort_task_command(sandbox: LocalSandbox) -> tuple[str, str]:
    python_command = sandbox.get_python_command()
    match _backend_name(sandbox):
        case "WindowsGitBashBackend":
            return (
                "\n".join(
                    [
                        "set -e",
                        "printf 'SHELL_BACKEND=git-bash\\n'",
                        "cat > quick_sort.py <<'PY'",
                        _QUICK_SORT_SOURCE.rstrip(),
                        "PY",
                        "cat > test_quick_sort.py <<'PY'",
                        _QUICK_SORT_TESTS.rstrip(),
                        "PY",
                        f"{python_command} -m pytest test_quick_sort.py -q --no-cov",
                    ]
                ),
                "git-bash",
            )
        case "WindowsCmdBackend":
            return (
                f"{_python_writer_command(sandbox, 'cmd')} && {python_command} -m pytest test_quick_sort.py -q --no-cov",
                "cmd",
            )
        case _:
            return (
                "\n".join(
                    [
                        "$ErrorActionPreference = 'Stop'",
                        "Write-Output 'SHELL_BACKEND=powershell'",
                        "@'",
                        _QUICK_SORT_SOURCE.rstrip(),
                        "'@ | Set-Content -Encoding UTF8 quick_sort.py",
                        "@'",
                        _QUICK_SORT_TESTS.rstrip(),
                        "'@ | Set-Content -Encoding UTF8 test_quick_sort.py",
                        f"{python_command} -m pytest test_quick_sort.py -q --no-cov",
                    ]
                ),
                "powershell",
            )


def _run_shell_tool() -> Tool:
    return Tool(
        name="run_shell_command",
        description="Execute a shell command in the active sandbox shell backend.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute."},
                "timeout_ms": {"type": "integer", "description": "Timeout in milliseconds."},
                "dir_path": {"type": "string", "description": "Optional working directory."},
                "is_background": {"type": "boolean", "description": "Whether to run in the background."},
            },
            "required": ["command"],
        },
        implementation=run_shell_command,
    )


def _message_text(messages: list[Message]) -> str:
    return "\n".join(message.get_text_content() for message in messages)


def _tool_results(messages: list[Message]) -> list[str]:
    results: list[str] = []
    for message in messages:
        if message.role != Role.TOOL:
            continue
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                results.append(block.content if isinstance(block.content, str) else str(block.content))
    return results


@pytest.mark.windows_only
class TestWindowsLocalSandboxSmoke:
    def test_background_task_lifecycle(self, sandbox: LocalSandbox) -> None:
        """W-16: background task start, status, and kill work on Windows."""
        match _backend_name(sandbox):
            case "WindowsCmdBackend":
                command = "ping 127.0.0.1 -n 120 > NUL"
            case "WindowsGitBashBackend":
                command = "sleep 120"
            case _:
                command = "Start-Sleep -Seconds 120"

        result = sandbox.execute_bash(command, background=True)
        pid = result.background_pid

        assert result.status == SandboxStatus.SUCCESS
        assert pid is not None

        time.sleep(1)
        status = sandbox.get_background_task_status(pid)
        assert status.status == SandboxStatus.RUNNING

        kill_result = sandbox.kill_background_task(pid)
        assert kill_result.status == SandboxStatus.SUCCESS

    def test_foreground_timeout_terminates_long_command(self, sandbox: LocalSandbox) -> None:
        """RFC-0019: long foreground commands time out on a real Windows runner."""
        match _backend_name(sandbox):
            case "WindowsCmdBackend":
                command = "ping 127.0.0.1 -n 120 > NUL"
            case "WindowsGitBashBackend":
                command = "sleep 120"
            case _:
                command = "Start-Sleep -Seconds 120"

        result = sandbox.execute_bash(command, timeout=500)

        assert result.status == SandboxStatus.TIMEOUT
        assert result.exit_code != 0
        assert "timed out after 500ms" in (result.error or "")

    def test_basic_cwd_and_env_commands(self, sandbox: LocalSandbox) -> None:
        """W-17: basic command, cwd, and env propagation work through the active backend."""
        match _backend_name(sandbox):
            case "WindowsCmdBackend":
                commands = {
                    "basic": ("echo TOOL_SHELL_OK", "TOOL_SHELL_OK"),
                    "cwd": ("cd", Path(tempfile.gettempdir()).name),
                    "env": ("echo %MY_VAR%", "NEXAU_ENV_TEST"),
                }
            case "WindowsGitBashBackend":
                commands = {
                    "basic": ("echo TOOL_SHELL_OK", "TOOL_SHELL_OK"),
                    "cwd": ("pwd", "/tmp"),
                    "env": ("printf '%s\\n' \"$MY_VAR\"", "NEXAU_ENV_TEST"),
                }
            case _:
                commands = {
                    "basic": ("Write-Output TOOL_SHELL_OK", "TOOL_SHELL_OK"),
                    "cwd": ("Get-Location", Path(tempfile.gettempdir()).name),
                    "env": ("Write-Output $env:MY_VAR", "NEXAU_ENV_TEST"),
                }

        for name, (command, expected) in commands.items():
            result = sandbox.execute_bash(command, cwd=tempfile.gettempdir(), envs={"MY_VAR": "NEXAU_ENV_TEST"})
            assert result.status == SandboxStatus.SUCCESS
            assert result.exit_code == 0
            assert expected.lower() in result.stdout.lower()

    def test_local_search_smoke(self, sandbox: LocalSandbox) -> None:
        """W-18: file search works with backend-appropriate Windows shell syntax."""
        work_dir_raw = sandbox.work_dir
        assert work_dir_raw is not None
        work_dir = Path(work_dir_raw)
        (work_dir / "test.txt").write_text("hello nexau\nfoo bar\nnexau windows\n", encoding="utf-8")

        match _backend_name(sandbox):
            case "WindowsCmdBackend":
                command = "findstr /n /i nexau test.txt"
            case "WindowsGitBashBackend":
                command = "grep -n -i nexau test.txt"
            case _:
                command = r"Select-String -Path .\test.txt -Pattern nexau"
        result = sandbox.execute_bash(command, cwd=str(work_dir))

        assert result.status == SandboxStatus.SUCCESS
        assert result.exit_code == 0
        assert "nexau" in result.stdout.lower()

    @pytest.mark.timeout(10)
    def test_shell_task_writes_quicksort_and_runs_pytest(self, sandbox: LocalSandbox) -> None:
        """RFC-0020: real Windows shell task writes code and validates it with pytest."""
        command, expected_marker = _quicksort_task_command(sandbox)
        started_at = time.monotonic()

        result = sandbox.execute_bash(command, timeout=10000)

        elapsed = time.monotonic() - started_at
        work_dir = Path(str(sandbox.work_dir))
        assert elapsed < 10
        assert result.status == SandboxStatus.SUCCESS
        assert result.exit_code == 0
        assert f"SHELL_BACKEND={expected_marker}" in result.stdout
        assert "passed" in result.stdout
        assert (work_dir / "quick_sort.py").exists()
        assert (work_dir / "test_quick_sort.py").exists()

    def test_python_command_uses_sys_executable_for_powershell(self, sandbox: LocalSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
        """RFC-0020: Windows Python subprocesses use sys.executable, not python3."""
        sandbox._shell_backend = WindowsPowerShellBackend(Path(r"C:\Program Files\PowerShell\7\pwsh.exe"), "pwsh")
        monkeypatch.setattr("nexau.archs.sandbox.local_sandbox.sys.executable", r"C:\Program Files\Python312\python.exe")

        command = sandbox.get_python_command()

        assert command == r"& 'C:\Program Files\Python312\python.exe'"
        assert "python3" not in command

    def test_windows_heredoc_policy_matches_active_backend(self, sandbox: LocalSandbox) -> None:
        """RFC-0019: Windows heredoc behavior follows the active shell backend."""
        backend_name = _backend_name(sandbox)
        command = "cat > heredoc_output.txt <<'EOF'\nhello from heredoc\nEOF"

        if backend_name == "WindowsCmdBackend":
            with pytest.raises(SandboxError) as exc_info:
                sandbox.prepare_shell_command(command)

            message = str(exc_info.value)
            assert "PowerShell syntax" in message
            assert "NEXAU_WINDOWS_SHELL_BACKEND=git-bash" in message
            return

        result = sandbox.execute_bash(command, cwd=str(sandbox.work_dir))

        assert result.status == SandboxStatus.SUCCESS
        output_path = Path(str(sandbox.work_dir)) / "heredoc_output.txt"
        assert output_path.read_text(encoding="utf-8") == "hello from heredoc\n"

        if backend_name != "WindowsGitBashBackend":
            with pytest.raises(SandboxError, match="PowerShell syntax"):
                sandbox.prepare_shell_command("python - <<'PY'\nprint('hello')\nPY")

    @pytest.mark.timeout(10)
    @pytest.mark.parametrize(("shell_preference", "expected_backend"), _scripted_agent_shell_preferences())
    def test_scripted_agent_completes_quicksort_task_with_expected_shell_trajectory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        shell_preference: str | None,
        expected_backend: str | None,
    ) -> None:
        """RFC-0020: Agent/tool/sandbox trajectory follows the selected Windows shell."""
        if shell_preference is None:
            monkeypatch.delenv("NEXAU_WINDOWS_SHELL_BACKEND", raising=False)
        else:
            monkeypatch.setenv("NEXAU_WINDOWS_SHELL_BACKEND", shell_preference)

        config = AgentConfig(
            name=f"windows_shell_task_{shell_preference or 'default'}",
            llm_config=LLMConfig(model="gpt-4o-mini"),
            tools=[_run_shell_tool()],
            middlewares=[RuntimeEnvironmentMiddleware()],
            sandbox_config=LocalSandboxConfig(work_dir=str(tmp_path)),
            tool_call_mode="openai",
            max_iterations=3,
            timeout=10,
        )
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()
            mock_openai.AsyncOpenAI.return_value = Mock()
            agent = Agent(config=config)

        sandbox = agent.sandbox_manager.start_sync()
        assert isinstance(sandbox, LocalSandbox)
        backend_name = _backend_name(sandbox)
        if expected_backend is not None:
            assert backend_name == expected_backend
        command, expected_marker = _quicksort_task_command(sandbox)
        command_args = {"command": command, "timeout_ms": 10000}
        llm_messages: list[list[Message]] = []

        async def scripted_call(messages: list[Message], **_: object) -> ModelResponse:
            llm_messages.append(list(messages))
            if len(llm_messages) == 1:
                prompt_text = _message_text(messages)
                if expected_marker == "git-bash":
                    assert "Git Bash" in prompt_text
                    assert "SHELL_BACKEND=git-bash" in command
                elif expected_marker == "cmd":
                    assert "cmd.exe" in prompt_text
                    assert "SHELL_BACKEND=cmd" in command
                else:
                    assert "PowerShell" in prompt_text
                    assert "SHELL_BACKEND=powershell" in command
                return ModelResponse(
                    content="I will create the quicksort files and run their tests through run_shell_command.",
                    tool_calls=[
                        ModelToolCall(
                            call_id="call_quicksort_task",
                            name="run_shell_command",
                            arguments=command_args,
                            raw_arguments=json.dumps(command_args),
                        )
                    ],
                )

            tool_output = "\n".join(_tool_results(messages))
            assert f"SHELL_BACKEND={expected_marker}" in tool_output
            assert "passed" in tool_output
            return ModelResponse(content=f"done with {expected_marker}")

        started_at = time.monotonic()
        with patch.object(agent.executor.llm_caller, "call_llm_async", new_callable=AsyncMock, side_effect=scripted_call):
            response = agent.run(message="Write a quicksort implementation and test it.")
        elapsed = time.monotonic() - started_at

        assert elapsed < 10
        assert response == f"done with {expected_marker}"
        assert (tmp_path / "quick_sort.py").exists()
        assert (tmp_path / "test_quick_sort.py").exists()

        assistant_tool_uses = [
            block
            for message in agent.history
            if message.role == Role.ASSISTANT
            for block in message.content
            if isinstance(block, ToolUseBlock)
        ]
        assert len(assistant_tool_uses) == 1
        tool_use = assistant_tool_uses[0]
        assert tool_use.name == "run_shell_command"
        assert f"SHELL_BACKEND={expected_marker}" in str(tool_use.input.get("command", ""))
        assert any(f"SHELL_BACKEND={expected_marker}" in result for result in _tool_results(list(agent.history)))
