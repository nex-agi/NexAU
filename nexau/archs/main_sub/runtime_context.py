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

"""Runtime platform context shared by all agent entrypoints."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def get_runtime_date() -> str:
    """Return the current local date-time string used in prompts."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def current_username() -> str:
    """Return a best-effort local username for prompt context."""
    return os.getenv("USER") or os.getenv("USERNAME") or "user"


def describe_shell_backend(sandbox: Any | None = None) -> tuple[str, str]:
    """Describe the command syntax expected by run_shell_command."""
    backend = getattr(sandbox, "_shell_backend", None)
    backend_name = type(backend).__name__ if backend is not None else ""

    if backend_name == "WindowsPowerShellBackend":
        return (
            "Windows PowerShell backend",
            "Use PowerShell command syntax for run_shell_command. PowerShell commands such as Write-Output are valid.",
        )
    if backend_name == "WindowsCmdBackend":
        return (
            "Windows cmd.exe backend",
            "Use cmd.exe command syntax for run_shell_command. PowerShell-specific commands require an explicit powershell invocation.",
        )
    if backend_name == "WindowsGitBashBackend":
        return (
            "Windows Git Bash backend",
            "Use bash-compatible command syntax for run_shell_command because Git Bash was explicitly selected.",
        )
    if backend_name == "UnixShellBackend":
        return (
            "Unix bash-compatible backend",
            "Use bash-compatible command syntax for run_shell_command.",
        )

    if sys.platform == "win32":
        try:
            from nexau.archs.platform.shell_backend import configured_windows_shell_preference, ensure_default_windows_shell

            preference = configured_windows_shell_preference()
            if preference == "git-bash":
                return (
                    "Windows Git Bash backend",
                    "Use bash-compatible command syntax for run_shell_command because Git Bash was explicitly selected.",
                )
            if preference == "cmd":
                return (
                    "Windows cmd.exe backend",
                    "Use cmd.exe command syntax for run_shell_command. "
                    "PowerShell-specific commands require an explicit powershell invocation.",
                )

            installation = ensure_default_windows_shell()
            if installation.kind == "cmd":
                return (
                    "Windows cmd.exe backend",
                    "Use cmd.exe command syntax for run_shell_command. "
                    "PowerShell-specific commands require an explicit powershell invocation.",
                )
            return (
                f"Windows PowerShell backend ({installation.kind})",
                "Use PowerShell command syntax for run_shell_command. PowerShell commands such as Write-Output are valid.",
            )
        except Exception as exc:
            return (
                f"Windows shell backend (unresolved: {exc})",
                "run_shell_command uses the configured Windows shell backend; verify with a small diagnostic command if needed.",
            )

    return (
        "Unix bash-compatible backend",
        "Use bash-compatible command syntax for run_shell_command.",
    )


def build_runtime_prompt_context(
    sandbox: Any | None = None,
    *,
    working_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Build prompt context that tells agents where and how commands run."""
    resolved_working_directory = str(working_directory or getattr(sandbox, "work_dir", None) or os.getcwd())
    shell_backend, shell_guidance = describe_shell_backend(sandbox)
    now = get_runtime_date()
    username = current_username()
    operating_system = platform.platform()
    platform_name = sys.platform
    env_content = {
        "date": now,
        "username": username,
        "working_directory": resolved_working_directory,
        "operating_system": operating_system,
        "platform": platform_name,
        "shell_tool_backend": shell_backend,
        "shell_tool_guidance": shell_guidance,
    }
    return {
        **env_content,
        "env_content": env_content,
    }
