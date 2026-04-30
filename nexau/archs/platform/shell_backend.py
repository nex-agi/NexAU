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

"""Shell backend selection and launch configuration.

RFC-0019: Windows support with PowerShell default and optional Git Bash

The local sandbox launches an explicit shell backend instead of relying on
``shell=True`` and the host default shell.  Windows defaults to PowerShell
(``pwsh.exe`` -> ``powershell.exe`` -> ``cmd.exe``) while Git Bash remains an
explicit opt-in backend for bash-compatible commands.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .git_bash import ensure_git_bash
from .path_helpers import native_path_to_shell_path

if sys.platform == "win32":
    _CREATE_NEW_PROCESS_GROUP: int = subprocess.CREATE_NEW_PROCESS_GROUP
else:
    _CREATE_NEW_PROCESS_GROUP = 0


@dataclass(frozen=True)
class ShellLaunchConfig:
    """Subprocess launch configuration for a shell command."""

    argv: tuple[str, ...]
    creationflags: int = 0
    start_new_session: bool = False


WindowsShellKind = Literal["pwsh", "powershell", "cmd"]
WindowsShellPreference = Literal["default", "powershell", "pwsh", "cmd", "git-bash", "bash"]


@dataclass(frozen=True)
class WindowsShellInstallation:
    """Resolved Windows default shell metadata."""

    executable_path: Path
    kind: WindowsShellKind
    source: str
    version: str | None


class ShellBackend(Protocol):
    """Interface implemented by platform-specific shell backends."""

    def build_launch_config(self, command: str) -> ShellLaunchConfig:
        """Build subprocess launch config for *command*."""
        ...

    def format_path_for_shell(self, path: str | Path) -> str:
        """Convert a native local path into a shell-consumable path."""
        ...

    def format_executable_for_shell(self, path: str | Path) -> str:
        """Format an executable path so it can be invoked by this backend."""
        ...


class UnixShellBackend:
    """Unix bash backend using an explicit bash executable path."""

    def __init__(self, bash_path: Path) -> None:
        self._bash_path = bash_path

    def build_launch_config(self, command: str) -> ShellLaunchConfig:
        return ShellLaunchConfig(argv=(str(self._bash_path), "-c", command), start_new_session=True)

    def format_path_for_shell(self, path: str | Path) -> str:
        return native_path_to_shell_path(path)

    def format_executable_for_shell(self, path: str | Path) -> str:
        return shlex_quote(native_path_to_shell_path(path))


class WindowsGitBashBackend:
    """Windows backend that launches commands through Git Bash."""

    def __init__(self, bash_path: Path) -> None:
        self._bash_path = bash_path

    def build_launch_config(self, command: str) -> ShellLaunchConfig:
        return ShellLaunchConfig(
            argv=(str(self._bash_path), "-c", command),
            creationflags=_CREATE_NEW_PROCESS_GROUP,
            start_new_session=False,
        )

    def format_path_for_shell(self, path: str | Path) -> str:
        return native_path_to_shell_path(path)

    def format_executable_for_shell(self, path: str | Path) -> str:
        return shlex_quote(native_path_to_shell_path(path))


class WindowsPowerShellBackend:
    """Windows default backend that launches commands through PowerShell."""

    def __init__(self, executable_path: Path, kind: Literal["pwsh", "powershell"]) -> None:
        self._executable_path = executable_path
        self._kind = kind

    def build_launch_config(self, command: str) -> ShellLaunchConfig:
        execution_policy_args: tuple[str, ...]
        if self._kind == "powershell":
            execution_policy_args = ("-ExecutionPolicy", "Bypass")
        else:
            execution_policy_args = ()
        return ShellLaunchConfig(
            argv=(
                str(self._executable_path),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                *execution_policy_args,
                "-Command",
                command,
            ),
            creationflags=_CREATE_NEW_PROCESS_GROUP,
            start_new_session=False,
        )

    def format_path_for_shell(self, path: str | Path) -> str:
        return str(path)

    def format_executable_for_shell(self, path: str | Path) -> str:
        return f"& {_quote_powershell(str(path))}"


class WindowsCmdBackend:
    """Last-resort Windows backend for basic command execution diagnostics."""

    def __init__(self, executable_path: Path) -> None:
        self._executable_path = executable_path

    def build_launch_config(self, command: str) -> ShellLaunchConfig:
        return ShellLaunchConfig(
            argv=(str(self._executable_path), "/d", "/s", "/c", command),
            creationflags=_CREATE_NEW_PROCESS_GROUP,
            start_new_session=False,
        )

    def format_path_for_shell(self, path: str | Path) -> str:
        return str(path)

    def format_executable_for_shell(self, path: str | Path) -> str:
        return _quote_cmd(str(path))


def shlex_quote(value: str) -> str:
    """Return POSIX shell quoting without importing shlex at call sites."""
    import shlex

    return shlex.quote(value)


def _quote_powershell(value: str) -> str:
    """Quote a literal for PowerShell single-quoted strings."""
    return "'" + value.replace("'", "''") + "'"


def _quote_cmd(value: str) -> str:
    """Quote a literal for cmd.exe command lines."""
    escaped = value.replace('"', r"\"")
    return f'"{escaped}"'


def _version_for_windows_shell(executable_path: Path, kind: WindowsShellKind) -> str | None:
    """Return a best-effort version string for a Windows shell candidate."""
    if kind == "cmd":
        return None

    try:
        result = subprocess.run(
            [
                str(executable_path),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "$PSVersionTable.PSVersion.ToString()",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None

    output = result.stdout.strip() or result.stderr.strip()
    return output or None


def _windows_shell_path_candidates(kind: WindowsShellKind) -> list[Path]:
    """Return executable candidates for a Windows shell kind."""
    executable_name = {
        "pwsh": "pwsh.exe",
        "powershell": "powershell.exe",
        "cmd": "cmd.exe",
    }[kind]
    discovered = shutil.which(executable_name)
    if discovered:
        return [Path(discovered)]
    if kind == "cmd":
        import os

        comspec = os.environ.get("ComSpec")
        if comspec:
            return [Path(comspec)]
    return []


def detect_powershell_backend() -> WindowsShellInstallation | None:
    """Detect the RFC-0019 default Windows shell backend.

    RFC-0019: 默认 PowerShell backend 探测

    Detection follows ``pwsh.exe`` -> ``powershell.exe`` -> ``cmd.exe``.  The
    final ``cmd.exe`` fallback is intentionally marked by its ``kind`` so
    callers do not mistake it for full PowerShell semantics.
    """
    if sys.platform != "win32":
        return None

    shell_kinds: tuple[WindowsShellKind, ...] = ("pwsh", "powershell", "cmd")
    for kind in shell_kinds:
        for candidate in _windows_shell_path_candidates(kind):
            if kind == "cmd" or candidate.exists():
                return WindowsShellInstallation(
                    executable_path=candidate,
                    kind=kind,
                    source="path",
                    version=_version_for_windows_shell(candidate, kind),
                )
    return None


def ensure_default_windows_shell() -> WindowsShellInstallation:
    """Return the default Windows shell backend or raise a clear error."""
    installation = detect_powershell_backend()
    if installation is None:
        raise RuntimeError("Unable to locate pwsh.exe, powershell.exe, or cmd.exe for LocalSandbox")
    return installation


def _windows_backend_from_installation(installation: WindowsShellInstallation) -> ShellBackend:
    """Build a shell backend from resolved Windows shell metadata."""
    if installation.kind == "cmd":
        return WindowsCmdBackend(installation.executable_path)
    return WindowsPowerShellBackend(installation.executable_path, installation.kind)


def configured_windows_shell_preference() -> WindowsShellPreference:
    """Return the explicitly configured Windows shell backend preference."""
    import os

    raw = os.environ.get("NEXAU_WINDOWS_SHELL_BACKEND", "default").strip().lower()
    match raw:
        case "" | "default":
            return "default"
        case "powershell" | "ps":
            return "powershell"
        case "pwsh":
            return "pwsh"
        case "cmd" | "cmd.exe":
            return "cmd"
        case "git-bash" | "git_bash" | "bash":
            return "git-bash"
        case _:
            raise RuntimeError("Unsupported NEXAU_WINDOWS_SHELL_BACKEND value. Use default, powershell, pwsh, cmd, or git-bash.")


def _create_windows_shell_backend(preference: WindowsShellPreference) -> ShellBackend:
    """Create a Windows shell backend according to an explicit preference."""
    if preference in {"git-bash", "bash"}:
        git_bash_installation = ensure_git_bash()
        return WindowsGitBashBackend(git_bash_installation.bash_path)

    if preference == "default":
        return _windows_backend_from_installation(ensure_default_windows_shell())

    windows_shell_installation = detect_powershell_backend()
    if windows_shell_installation is None:
        raise RuntimeError("Unable to locate a Windows shell backend")
    if preference == "powershell":
        if windows_shell_installation.kind == "cmd":
            raise RuntimeError("Unable to locate pwsh.exe or powershell.exe for the PowerShell backend")
        return _windows_backend_from_installation(windows_shell_installation)
    if preference == "pwsh":
        for candidate in _windows_shell_path_candidates("pwsh"):
            if candidate.exists():
                return WindowsPowerShellBackend(candidate, "pwsh")
        raise RuntimeError("Unable to locate pwsh.exe for the requested Windows shell backend")
    if preference == "cmd":
        for candidate in _windows_shell_path_candidates("cmd"):
            return WindowsCmdBackend(candidate)
        raise RuntimeError("Unable to locate cmd.exe for the requested Windows shell backend")

    raise RuntimeError("Unsupported Windows shell backend preference")


def _detect_unix_bash() -> Path:
    """Detect the bash executable on Unix-like hosts."""
    import os

    for env_name in ("NEXAU_BASH_PATH", "NEXAU_GIT_BASH_PATH"):
        configured = os.environ.get(env_name)
        if configured:
            return Path(configured).expanduser()

    discovered = shutil.which("bash")
    if discovered:
        return Path(discovered)

    for fallback in (Path("/bin/bash"), Path("/usr/bin/bash"), Path("/bin/sh")):
        if fallback.exists():
            return fallback

    raise RuntimeError("Unable to locate a bash-compatible executable for LocalSandbox")


def create_shell_backend() -> ShellBackend:
    """Create the shell backend for the current host platform."""
    if sys.platform == "win32":
        return _create_windows_shell_backend(configured_windows_shell_preference())
    return UnixShellBackend(_detect_unix_bash())
