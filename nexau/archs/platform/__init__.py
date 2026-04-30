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

"""Cross-platform runtime helpers.

RFC-0019: Windows support with PowerShell default and optional Git Bash

Centralizes platform-specific shell, process, and path helpers so business
modules can keep a single interface while the compatibility logic stays in one
place.
"""

from .git_bash import (
    GitBashInstallation,
    MissingGitBashError,
    detect_git_bash,
    ensure_git_bash,
    explain_git_bash_requirement,
)
from .path_helpers import (
    get_local_bash_tool_results_dir,
    get_local_cli_sessions_dir,
    get_local_temp_root,
    get_local_tool_output_dir,
    native_path_to_shell_path,
)
from .process_compat import graceful_kill, reemit_termination_signal, supported_cleanup_signals
from .shell_backend import (
    ShellBackend,
    ShellLaunchConfig,
    WindowsCmdBackend,
    WindowsGitBashBackend,
    WindowsPowerShellBackend,
    WindowsShellInstallation,
    configured_windows_shell_preference,
    create_shell_backend,
    detect_powershell_backend,
    ensure_default_windows_shell,
)

__all__ = [
    "GitBashInstallation",
    "MissingGitBashError",
    "detect_git_bash",
    "ensure_git_bash",
    "explain_git_bash_requirement",
    "get_local_bash_tool_results_dir",
    "get_local_cli_sessions_dir",
    "get_local_temp_root",
    "get_local_tool_output_dir",
    "native_path_to_shell_path",
    "graceful_kill",
    "reemit_termination_signal",
    "supported_cleanup_signals",
    "ShellBackend",
    "ShellLaunchConfig",
    "WindowsCmdBackend",
    "WindowsGitBashBackend",
    "WindowsPowerShellBackend",
    "WindowsShellInstallation",
    "configured_windows_shell_preference",
    "create_shell_backend",
    "detect_powershell_backend",
    "ensure_default_windows_shell",
]
