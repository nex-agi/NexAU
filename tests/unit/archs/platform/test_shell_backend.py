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

"""Tests for RFC-0019 shell backend selection and launch configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexau.archs.platform import shell_backend
from nexau.archs.platform.git_bash import GitBashInstallation
from nexau.archs.platform.shell_backend import (
    UnixShellBackend,
    WindowsCmdBackend,
    WindowsGitBashBackend,
    WindowsPowerShellBackend,
    WindowsShellInstallation,
)


def test_unix_shell_backend_builds_explicit_bash_argv() -> None:
    bash_path = Path("bin/bash")
    backend = UnixShellBackend(bash_path)

    config = backend.build_launch_config("echo hello")

    assert config.argv == (str(bash_path), "-c", "echo hello")
    assert config.start_new_session is True
    assert config.creationflags == 0


def test_windows_git_bash_backend_builds_process_group_config() -> None:
    backend = WindowsGitBashBackend(Path(r"C:\Program Files\Git\bin\bash.exe"))

    config = backend.build_launch_config("echo hello")

    assert config.argv == (r"C:\Program Files\Git\bin\bash.exe", "-c", "echo hello")
    assert config.start_new_session is False
    assert config.creationflags == shell_backend._CREATE_NEW_PROCESS_GROUP


def test_windows_git_bash_backend_formats_native_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_backend.sys, "platform", "win32")
    backend = WindowsGitBashBackend(Path(r"C:\Git\bin\bash.exe"))

    assert backend.format_path_for_shell(r"C:\repo\file.txt") == "/c/repo/file.txt"


def test_windows_powershell_backend_builds_explicit_argv() -> None:
    backend = WindowsPowerShellBackend(Path(r"C:\Program Files\PowerShell\7\pwsh.exe"), "pwsh")

    config = backend.build_launch_config("Write-Output hello")

    assert config.argv == (
        r"C:\Program Files\PowerShell\7\pwsh.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Write-Output hello",
    )
    assert config.start_new_session is False
    assert config.creationflags == shell_backend._CREATE_NEW_PROCESS_GROUP


def test_windows_powershell_backend_formats_native_path() -> None:
    backend = WindowsPowerShellBackend(Path(r"C:\Program Files\PowerShell\7\pwsh.exe"), "pwsh")

    assert backend.format_path_for_shell(r"C:\repo\file.txt") == r"C:\repo\file.txt"
    assert backend.format_executable_for_shell(r"C:\Program Files\Python\python.exe") == (r"& 'C:\Program Files\Python\python.exe'")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"C:\Program Files\Python\python.exe", r"& 'C:\Program Files\Python\python.exe'"),
        (r"C:\Users\O'Brien\AppData\Local\Programs\Python\python.exe", r"& 'C:\Users\O''Brien\AppData\Local\Programs\Python\python.exe'"),
        (r"\\server\share with spaces\tool.exe", r"& '\\server\share with spaces\tool.exe'"),
        (r"C:\Tools\percent%value\tool.exe", r"& 'C:\Tools\percent%value\tool.exe'"),
    ],
)
def test_windows_powershell_backend_quotes_executable_edge_cases(path: str, expected: str) -> None:
    backend = WindowsPowerShellBackend(Path(r"C:\Program Files\PowerShell\7\pwsh.exe"), "pwsh")

    assert backend.format_executable_for_shell(path) == expected


def test_windows_cmd_backend_builds_last_resort_argv() -> None:
    backend = WindowsCmdBackend(Path(r"C:\Windows\System32\cmd.exe"))

    config = backend.build_launch_config("echo hello")

    assert config.argv == (r"C:\Windows\System32\cmd.exe", "/d", "/s", "/c", "echo hello")
    assert config.creationflags == shell_backend._CREATE_NEW_PROCESS_GROUP
    assert backend.format_path_for_shell(r"C:\repo\file.txt") == r"C:\repo\file.txt"
    assert backend.format_executable_for_shell(r'C:\Program Files\App "quoted"\app.exe') == (r'"C:\Program Files\App \"quoted\"\app.exe"')


def test_windows_powershell_backend_builds_windows_powershell_argv() -> None:
    backend = WindowsPowerShellBackend(Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"), "powershell")

    config = backend.build_launch_config("Write-Output hello")

    assert "-ExecutionPolicy" in config.argv
    assert "Bypass" in config.argv


def test_version_for_windows_shell_handles_cmd_without_subprocess() -> None:
    assert shell_backend._version_for_windows_shell(Path(r"C:\Windows\System32\cmd.exe"), "cmd") is None


def test_version_for_windows_shell_returns_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = type("Completed", (), {"stdout": "7.4.0\n", "stderr": ""})()

    monkeypatch.setattr(shell_backend.subprocess, "run", lambda *_args, **_kwargs: completed)

    assert shell_backend._version_for_windows_shell(Path("pwsh.exe"), "pwsh") == "7.4.0"


def test_version_for_windows_shell_returns_none_on_subprocess_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_error(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError

    monkeypatch.setattr(shell_backend.subprocess, "run", raise_error)

    assert shell_backend._version_for_windows_shell(Path("pwsh.exe"), "pwsh") is None


def test_windows_shell_path_candidates_uses_discovered_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_backend.shutil, "which", lambda name: rf"C:\Tools\{name}")

    assert shell_backend._windows_shell_path_candidates("pwsh") == [Path(r"C:\Tools\pwsh.exe")]


def test_windows_shell_path_candidates_uses_comspec_for_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_backend.shutil, "which", lambda _name: None)
    monkeypatch.setenv("ComSpec", r"C:\Windows\System32\cmd.exe")

    assert shell_backend._windows_shell_path_candidates("cmd") == [Path(r"C:\Windows\System32\cmd.exe")]


def test_create_shell_backend_uses_windows_powershell_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    installation = WindowsShellInstallation(
        executable_path=Path(r"C:\Program Files\PowerShell\7\pwsh.exe"),
        kind="pwsh",
        source="test",
        version="7.4.0",
    )
    monkeypatch.setattr(shell_backend.sys, "platform", "win32")
    monkeypatch.delenv("NEXAU_WINDOWS_SHELL_BACKEND", raising=False)
    monkeypatch.setattr(shell_backend, "ensure_default_windows_shell", lambda: installation)

    backend = shell_backend.create_shell_backend()

    assert isinstance(backend, WindowsPowerShellBackend)
    assert backend.build_launch_config("pwd").argv[0] == str(installation.executable_path)


def test_create_shell_backend_uses_windows_git_bash_when_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    bash_path = Path(r"C:\Git\bin\bash.exe")
    installation = GitBashInstallation(bash_path=bash_path, source="test", version="GNU bash")
    monkeypatch.setattr(shell_backend.sys, "platform", "win32")
    monkeypatch.setenv("NEXAU_WINDOWS_SHELL_BACKEND", "git-bash")
    monkeypatch.setattr(shell_backend, "ensure_git_bash", lambda: installation)

    backend = shell_backend.create_shell_backend()

    assert isinstance(backend, WindowsGitBashBackend)
    assert backend.build_launch_config("pwd").argv[0] == str(bash_path)


def test_detect_powershell_backend_prefers_pwsh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pwsh_path = tmp_path / "pwsh.exe"
    powershell_path = tmp_path / "powershell.exe"
    pwsh_path.write_text("", encoding="utf-8")
    powershell_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(shell_backend.sys, "platform", "win32")
    monkeypatch.setattr(
        shell_backend,
        "_windows_shell_path_candidates",
        lambda kind: [pwsh_path] if kind == "pwsh" else [powershell_path],
    )
    monkeypatch.setattr(shell_backend, "_version_for_windows_shell", lambda _path, _kind: "7.4.0")

    installation = shell_backend.detect_powershell_backend()

    assert installation is not None
    assert installation.kind == "pwsh"
    assert installation.executable_path == pwsh_path


def test_detect_powershell_backend_falls_back_to_cmd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_backend.sys, "platform", "win32")
    monkeypatch.setattr(
        shell_backend,
        "_windows_shell_path_candidates",
        lambda kind: [Path(r"C:\Windows\System32\cmd.exe")] if kind == "cmd" else [],
    )

    installation = shell_backend.detect_powershell_backend()

    assert installation is not None
    assert installation.kind == "cmd"


def test_detect_powershell_backend_returns_none_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_backend.sys, "platform", "linux")

    assert shell_backend.detect_powershell_backend() is None


def test_ensure_default_windows_shell_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shell_backend, "detect_powershell_backend", lambda: None)

    with pytest.raises(RuntimeError, match="Unable to locate"):
        shell_backend.ensure_default_windows_shell()


def test_create_windows_shell_backend_rejects_missing_powershell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        shell_backend,
        "detect_powershell_backend",
        lambda: WindowsShellInstallation(
            executable_path=Path(r"C:\Windows\System32\cmd.exe"),
            kind="cmd",
            source="test",
            version=None,
        ),
    )

    with pytest.raises(RuntimeError, match="PowerShell backend"):
        shell_backend._create_windows_shell_backend("powershell")


def test_create_windows_shell_backend_uses_requested_pwsh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pwsh_path = tmp_path / "pwsh.exe"
    pwsh_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        shell_backend,
        "detect_powershell_backend",
        lambda: WindowsShellInstallation(
            executable_path=pwsh_path,
            kind="pwsh",
            source="test",
            version="7.4.0",
        ),
    )
    monkeypatch.setattr(shell_backend, "_windows_shell_path_candidates", lambda kind: [pwsh_path] if kind == "pwsh" else [])

    backend = shell_backend._create_windows_shell_backend("pwsh")

    assert isinstance(backend, WindowsPowerShellBackend)


def test_configured_windows_shell_preference_rejects_invalid_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXAU_WINDOWS_SHELL_BACKEND", "zsh")

    with pytest.raises(RuntimeError, match="Unsupported NEXAU_WINDOWS_SHELL_BACKEND"):
        shell_backend.configured_windows_shell_preference()


def test_create_shell_backend_uses_unix_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    bash_path = Path("/usr/bin/bash")
    monkeypatch.setattr(shell_backend.sys, "platform", "linux")
    monkeypatch.setattr(shell_backend, "_detect_unix_bash", lambda: bash_path)

    backend = shell_backend.create_shell_backend()

    assert isinstance(backend, UnixShellBackend)
    assert backend.build_launch_config("pwd").argv[0] == str(bash_path)
