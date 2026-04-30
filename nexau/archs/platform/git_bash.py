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

"""Optional Git Bash discovery helpers.

RFC-0019: Windows support with PowerShell default and optional Git Bash

Windows defaults to PowerShell/cmd execution.  Git Bash discovery is still
available for explicitly requested bash-compatible backend paths and should not
block default Windows startup.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitBashInstallation:
    """Resolved Git Bash installation metadata."""

    bash_path: Path
    source: str
    version: str | None


class MissingGitBashError(RuntimeError):
    """Raised when Git Bash is required but unavailable on Windows."""


def _configured_candidates() -> list[Path]:
    """Return explicitly configured Git Bash candidates, highest priority first."""
    candidates: list[Path] = []
    for env_name in ("NEXAU_GIT_BASH_PATH", "NEXAU_BASH_PATH"):
        configured = os.environ.get(env_name)
        if configured:
            candidates.append(Path(configured).expanduser())
    return candidates


def _path_candidate() -> list[Path]:
    """Return the Git Bash candidate resolved from PATH, if available."""
    bash_path = shutil.which("bash")
    if bash_path:
        return [Path(bash_path)]
    return []


def _is_valid_bash_candidate(candidate: Path) -> bool:
    """Return True when *candidate* points to an existing bash executable file."""
    return candidate.exists() and candidate.is_file()


def _common_install_dir_candidates() -> list[Path]:
    """Return common Git for Windows installation locations."""
    candidates: list[Path] = []
    env_candidates = (
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LocalAppData"),
    )
    for root in env_candidates:
        if root is None or root == "":
            continue
        base = Path(root)
        candidates.append(base / "Git" / "bin" / "bash.exe")
        candidates.append(base / "Git" / "usr" / "bin" / "bash.exe")
        candidates.append(base / "Programs" / "Git" / "bin" / "bash.exe")
    return candidates


def _candidate_groups() -> tuple[tuple[str, list[Path]], ...]:
    """Return all Git Bash candidate groups in discovery priority order."""
    return (
        ("configured", _configured_candidates()),
        ("path", _path_candidate()),
        ("common-install-dir", _common_install_dir_candidates()),
    )


def detect_git_bash() -> GitBashInstallation | None:
    """Detect Git Bash on Windows.

    RFC-0019: Git Bash 可用性健康检查

    Discovery order follows RFC-0019: explicit configuration > PATH > common
    installation directories. A candidate is accepted when it exists on disk.

    Do not execute ``bash --version`` here. On some Windows installations the
    Git Bash launcher can hang or leave an orphan helper process; discovery is
    on the agent/tool hot path and must never block a simple shell command.
    """
    if sys.platform != "win32":
        return None

    for source, candidates in _candidate_groups():
        for candidate in candidates:
            if _is_valid_bash_candidate(candidate):
                return GitBashInstallation(
                    bash_path=candidate,
                    source=source,
                    version=None,
                )
    return None


def _find_unusable_bash_paths() -> list[Path]:
    """Return configured bash paths that exist but are not files.

    RFC-0019: unusable Git Bash 诊断辅助

    Called only when ``detect_git_bash()`` returns ``None`` to distinguish
    "completely missing" from "present but not an executable file".
    """
    unusable: list[Path] = []
    for _source, candidates in _candidate_groups():
        for candidate in candidates:
            if candidate.exists() and not candidate.is_file():
                unusable.append(candidate)
    return unusable


def explain_git_bash_requirement() -> str:
    """Return the RFC-0019 guidance text shown when Git Bash is missing."""
    return (
        "Git Bash (Git for Windows) is required for the optional bash-compatible Windows backend. "
        "NexAU will not install it automatically. Install Git for Windows and ensure "
        "bash.exe is discoverable via NEXAU_GIT_BASH_PATH, PATH, or a standard Git install location."
    )


def explain_unusable_git_bash(unusable_paths: list[Path]) -> str:
    """Return guidance when Git Bash paths are not executable files.

    RFC-0019: unusable Git Bash 用户提示
    """
    paths_str = ", ".join(str(p) for p in unusable_paths)
    return (
        f"Git Bash path {paths_str} exists but is not an executable file. "
        f"Set NEXAU_GIT_BASH_PATH to Git for Windows bash.exe or reinstall Git for Windows."
    )


def ensure_git_bash() -> GitBashInstallation:
    """Return the resolved Git Bash installation or raise a fail-fast error.

    RFC-0019: unusable Git Bash 也应 fail-fast

    The hot path verifies that the configured/discovered candidate exists as a
    file. It does not run ``bash --version`` because that probe can hang on
    some Windows installations and block otherwise simple shell commands.
    """
    installation = detect_git_bash()
    if installation is None:
        unusable = _find_unusable_bash_paths()
        if unusable:
            raise MissingGitBashError(explain_unusable_git_bash(unusable))
        raise MissingGitBashError(explain_git_bash_requirement())
    return installation
