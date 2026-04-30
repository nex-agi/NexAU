"""Platform-aware test helpers.

RFC-0020: 测试基建平台化与 shell backend 差异治理

Provides a centralized place for test-only platform detection, optional tool
discovery, newline normalization, and platform-aware temp/output/script path
generation. This keeps platform branching out of individual test cases.
"""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

WhichResolver = Callable[[str], str | None]


@dataclass(frozen=True)
class TestPlatform:
    """Detected host platform details for test gating and helper selection."""

    system: str
    is_windows: bool
    is_posix: bool
    temp_dir: Path
    git_bash_path: str | None
    rg_path: str | None
    ffmpeg_path: str | None
    python_command: str


def default_python_command(system_name: str) -> str:
    """Return the preferred Python launcher name for the given host system."""
    return "python" if system_name == "Windows" else "python3"


def normalize_newlines(text: str) -> str:
    """Normalize CRLF / CR line endings so assertions stay platform-neutral."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def build_test_output_dir_root(temp_dir: str | os.PathLike[str] | Path | None = None) -> Path:
    """Return the platform-aware root directory for local shell output artifacts."""
    base_dir = Path(tempfile.gettempdir()) if temp_dir is None else Path(temp_dir)
    return base_dir / "nexau_bash_tool_results"


def build_test_script_dir_root(temp_dir: str | os.PathLike[str] | Path | None = None) -> Path:
    """Return the platform-aware root directory for generated helper scripts."""
    base_dir = Path(tempfile.gettempdir()) if temp_dir is None else Path(temp_dir)
    return base_dir / "nexau_bash_scripts"


def detect_test_platform(
    system_name: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    which: WhichResolver | None = None,
    temp_dir: str | os.PathLike[str] | Path | None = None,
) -> TestPlatform:
    """Detect host platform and optional external tool availability for tests."""
    resolved_system = platform.system() if system_name is None else system_name
    resolved_env = os.environ if env is None else env
    resolved_which = shutil.which if which is None else which
    resolved_temp_dir = Path(tempfile.gettempdir()) if temp_dir is None else Path(temp_dir)

    git_bash_path = _detect_git_bash_path(resolved_system, resolved_env, resolved_which)
    python_command = default_python_command(resolved_system)
    resolved_python = resolved_which(python_command) or resolved_which("python")

    return TestPlatform(
        system=resolved_system,
        is_windows=resolved_system == "Windows",
        is_posix=resolved_system in {"Linux", "Darwin"},
        temp_dir=resolved_temp_dir,
        git_bash_path=git_bash_path,
        rg_path=resolved_which("rg"),
        ffmpeg_path=resolved_which("ffmpeg"),
        python_command=resolved_python or python_command,
    )


def _detect_git_bash_path(
    system_name: str,
    env: Mapping[str, str],
    which: WhichResolver,
) -> str | None:
    """Resolve the optional Git Bash backend executable for test gates."""
    override = env.get("NEXAU_TEST_GIT_BASH") or env.get("GIT_BASH_PATH")
    if override:
        return override

    if system_name != "Windows":
        return which("bash") or ("/bin/bash" if Path("/bin/bash").exists() else None)

    for name in ("bash.exe", "git-bash.exe", "bash"):
        detected = which(name)
        if detected is not None:
            return detected

    for candidate in _windows_git_bash_candidates(env):
        if candidate.exists():
            return str(candidate)
    return None


def _windows_git_bash_candidates(env: Mapping[str, str]) -> tuple[Path, ...]:
    """Build common Git for Windows bash.exe installation candidates."""
    candidates: list[Path] = []
    for env_key in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        root = env.get(env_key)
        if root:
            candidates.append(Path(root) / "Git" / "bin" / "bash.exe")

    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Git" / "bin" / "bash.exe")

    return tuple(candidates)
