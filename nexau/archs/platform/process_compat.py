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

"""Cross-platform process lifecycle helpers.

RFC-0019: Windows support with PowerShell default and optional Git Bash

This module centralizes subprocess termination and cleanup signal behavior so
sandbox/business code can request lifecycle capabilities without depending on
platform-specific APIs directly.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import NoReturn, Protocol

if sys.platform != "win32":
    from os import getpgid as _getpgid
    from os import killpg as _killpg
else:

    def _getpgid(pid: int) -> int:
        """Raise on Windows because POSIX process groups are unsupported."""
        raise NotImplementedError("POSIX process groups are unavailable on Windows")

    def _killpg(pgid: int, sig: int) -> None:
        """Raise on Windows because POSIX process groups are unsupported."""
        raise NotImplementedError("POSIX process groups are unavailable on Windows")


ProcessHandle = subprocess.Popen[bytes] | subprocess.Popen[str]

# SIGKILL is always numeric 9 on every POSIX system. We keep a fixed numeric
# constant so this module type-checks cleanly on Windows hosts as well.
_SIGKILL: int = 9


class ProcessCompat(Protocol):
    """Platform-specific process lifecycle behavior."""

    def graceful_kill(self, process: ProcessHandle, grace_period: float) -> None:
        """Stop *process* and wait for shutdown within *grace_period*."""
        ...

    def supported_cleanup_signals(self) -> tuple[int, ...]:
        """Return termination signals that should register cleanup handlers."""
        ...

    def reemit_termination_signal(self, signum: int) -> NoReturn:
        """Terminate the current process after cleanup completes."""
        ...


class PosixProcessCompat:
    """POSIX process lifecycle implementation."""

    def graceful_kill(self, process: ProcessHandle, grace_period: float) -> None:
        # 1. SIGTERM — give the process group a chance to clean up.
        try:
            pgid = _getpgid(process.pid)
            _killpg(pgid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.terminate()
            except OSError:
                pass

        # 2. Wait for graceful exit.
        try:
            process.wait(timeout=grace_period)
            return
        except subprocess.TimeoutExpired:
            pass

        # 3. SIGKILL — force-kill the entire process group.
        try:
            pgid = _getpgid(process.pid)
            _killpg(pgid, _SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            process.kill()
        except OSError:
            pass

        # 4. Final wait.
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def supported_cleanup_signals(self) -> tuple[int, ...]:
        return (signal.SIGTERM, signal.SIGINT)

    def reemit_termination_signal(self, signum: int) -> NoReturn:
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
        raise SystemExit(128 + signum)


class WindowsProcessCompat:
    """Windows process lifecycle implementation using stdlib primitives first."""

    def graceful_kill(self, process: ProcessHandle, grace_period: float) -> None:
        try:
            process.terminate()
        except OSError:
            pass

        try:
            process.wait(timeout=grace_period)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            process.kill()
        except OSError:
            pass

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def supported_cleanup_signals(self) -> tuple[int, ...]:
        return (signal.SIGINT,)

    def reemit_termination_signal(self, signum: int) -> NoReturn:
        raise SystemExit(128 + signum)


_PROCESS_COMPAT: ProcessCompat = WindowsProcessCompat() if sys.platform == "win32" else PosixProcessCompat()


def graceful_kill(process: ProcessHandle, grace_period: float = 5.0) -> None:
    """Gracefully terminate a process using the host platform strategy."""
    _PROCESS_COMPAT.graceful_kill(process, grace_period)


def supported_cleanup_signals() -> tuple[int, ...]:
    """Return signals that should register cleanup handlers on this host."""
    return _PROCESS_COMPAT.supported_cleanup_signals()


def reemit_termination_signal(signum: int) -> NoReturn:
    """Terminate the current process after cleanup completes."""
    _PROCESS_COMPAT.reemit_termination_signal(signum)
