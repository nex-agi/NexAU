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

"""Cross-platform path helpers.

RFC-0019: Windows support with PowerShell default and optional Git Bash

The helpers in this module keep Python-native paths and shell-consumable paths
explicitly separated. Python file APIs should keep native paths; Windows
PowerShell/cmd backends keep native paths, while optional Git Bash commands use
POSIX-style paths on Windows.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path, PureWindowsPath

_CLI_ROOT_DIR = "nexau"
_BASH_TOOL_RESULTS_DIR = "nexau_bash_tool_results"
_TOOL_OUTPUTS_DIR = "nexau_tool_outputs"
_CLI_SESSIONS_DIR = "cli-sessions"


def is_windows_host() -> bool:
    """Return True when the current Python runtime is on Windows."""
    return sys.platform == "win32"


def get_local_temp_root() -> Path:
    """Return the host temp directory used for local NexAU operations."""
    return Path(tempfile.gettempdir())


def get_local_bash_tool_results_dir() -> Path:
    """Return the base directory used for local shell stdout/stderr artifacts."""
    return get_local_temp_root() / _BASH_TOOL_RESULTS_DIR


def get_local_tool_output_dir() -> Path:
    """Return the base directory used by long-tool-output persistence."""
    return get_local_temp_root() / _TOOL_OUTPUTS_DIR


def get_local_cli_sessions_dir() -> Path:
    """Return the base directory used for CLI session snapshots."""
    return get_local_temp_root() / _CLI_ROOT_DIR / _CLI_SESSIONS_DIR


def native_path_to_shell_path(path: str | Path) -> str:
    """Convert a native local path to a shell-consumable path.

    On Unix hosts this returns a POSIX string representation.
    On Windows hosts this converts ``C:\\foo\\bar`` into ``/c/foo/bar`` for
    Git Bash consumption.
    """
    raw = str(path)
    if raw == "":
        return raw

    if not is_windows_host():
        return Path(raw).as_posix()

    if raw.startswith("/"):
        return raw.replace("\\", "/")

    windows_path = PureWindowsPath(raw)
    if windows_path.drive.startswith("\\\\"):
        return raw.replace("\\", "/")

    drive = windows_path.drive.rstrip(":")
    filtered_parts = [part for part in windows_path.parts if part not in {windows_path.drive, windows_path.root, windows_path.anchor}]
    normalized_tail = "/".join(filtered_parts)

    if drive:
        if normalized_tail:
            return f"/{drive.lower()}/{normalized_tail}"
        return f"/{drive.lower()}"

    return raw.replace("\\", "/")
