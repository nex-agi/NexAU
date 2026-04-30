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

"""CLI entrypoint prerequisite checks.

RFC-0019: Windows support with PowerShell default and optional Git Bash

Centralizes user-facing CLI entrypoint checks.  The default Windows path only
requires a usable PowerShell/cmd backend; Git Bash checks are reserved for
explicit bash-compatible backend requests.
"""

from __future__ import annotations

import sys
from typing import TextIO

from nexau.archs.platform.git_bash import GitBashInstallation, MissingGitBashError, ensure_git_bash
from nexau.archs.platform.shell_backend import (
    WindowsShellInstallation,
    configured_windows_shell_preference,
    ensure_default_windows_shell,
)

MISSING_GIT_BASH_EXIT_CODE = 2
MISSING_WINDOWS_SHELL_EXIT_CODE = 2
EntrypointShellInstallation = WindowsShellInstallation | GitBashInstallation


def handoff_git_bash_install_hint() -> str:
    """Return the stable handoff hint shown by CLI entrypoints.

    RFC-0019: Git Bash 缺失依赖接管提示

    The hint is intentionally phrased so both humans and upstream wrappers can
    understand the next step: an outer workflow should guide the user through
    installing Git for Windows, then retry the same NexAU entrypoint.
    """
    return (
        "Ask the calling workflow or upper-level agent to guide the user through "
        "installing Git for Windows (Git Bash), then retry the same NexAU command. "
        "If no outer workflow exists, manually install Git for Windows and ensure "
        "bash.exe is discoverable before retrying."
    )


def ensure_git_bash_for_entrypoint(*, stderr: TextIO | None = None) -> GitBashInstallation | None:
    """Fail fast when an explicit Git Bash backend is unavailable.

    RFC-0019: 显式 Git Bash backend 缺失依赖闭环

    This helper is intentionally not called by default CLI entrypoints anymore.
    It remains available for wrappers that explicitly request the optional
    bash-compatible backend.
    """
    if sys.platform != "win32":
        return None

    try:
        return ensure_git_bash()
    except MissingGitBashError as exc:
        stream = sys.stderr if stderr is None else stderr
        print(f"Error: {exc}", file=stream)
        print(f"Handoff hint: {handoff_git_bash_install_hint()}", file=stream)
        raise


def ensure_default_windows_shell_for_entrypoint(
    *,
    stderr: TextIO | None = None,
) -> EntrypointShellInstallation | None:
    """Ensure the default Windows shell backend is available.

    RFC-0019: 默认 PowerShell 入口检查

    Missing Git Bash must not block Windows entrypoints.  On Windows this
    checks the default backend order ``pwsh.exe`` -> ``powershell.exe`` ->
    ``cmd.exe`` unless the optional Git Bash backend is explicitly requested.
    Returns ``None`` only on non-Windows hosts.
    """
    if sys.platform != "win32":
        return None

    if configured_windows_shell_preference() == "git-bash":
        return ensure_git_bash_for_entrypoint(stderr=stderr)

    try:
        return ensure_default_windows_shell()
    except RuntimeError as exc:
        stream = sys.stderr if stderr is None else stderr
        print(f"Error: {exc}", file=stream)
        raise
