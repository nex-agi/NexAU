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

"""Tests for RFC-0019 cross-platform process compatibility helpers."""

from __future__ import annotations

import signal
import subprocess
from typing import cast
from unittest.mock import Mock

import pytest

from nexau.archs.platform import process_compat
from nexau.archs.platform.process_compat import PosixProcessCompat, ProcessHandle, WindowsProcessCompat


def test_windows_graceful_kill_returns_after_graceful_exit() -> None:
    process = Mock()
    process.wait.return_value = 0
    compat = WindowsProcessCompat()

    compat.graceful_kill(cast(ProcessHandle, process), grace_period=0.1)

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=0.1)
    process.kill.assert_not_called()


def test_windows_graceful_kill_forces_kill_after_timeout() -> None:
    process = Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="cmd", timeout=0.1), 0]
    compat = WindowsProcessCompat()

    compat.graceful_kill(cast(ProcessHandle, process), grace_period=0.1)

    process.terminate.assert_called_once_with()
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2


def test_windows_supported_cleanup_signals_only_registers_sigint() -> None:
    assert WindowsProcessCompat().supported_cleanup_signals() == (signal.SIGINT,)


def test_windows_reemit_termination_signal_exits_with_signal_code() -> None:
    with pytest.raises(SystemExit, match=str(128 + signal.SIGINT)):
        WindowsProcessCompat().reemit_termination_signal(signal.SIGINT)


def test_posix_graceful_kill_sends_sigterm_to_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.pid = 1234
    process.wait.return_value = 0
    getpgid_mock = Mock(return_value=5678)
    killpg_mock = Mock()
    monkeypatch.setattr(process_compat, "_getpgid", getpgid_mock)
    monkeypatch.setattr(process_compat, "_killpg", killpg_mock)
    compat = PosixProcessCompat()

    compat.graceful_kill(cast(ProcessHandle, process), grace_period=0.1)

    getpgid_mock.assert_called_once_with(1234)
    killpg_mock.assert_called_once_with(5678, signal.SIGTERM)
    process.wait.assert_called_once_with(timeout=0.1)
    process.kill.assert_not_called()


def test_posix_graceful_kill_sends_sigkill_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.pid = 1234
    process.wait.side_effect = [subprocess.TimeoutExpired(cmd="cmd", timeout=0.1), 0]
    getpgid_mock = Mock(return_value=5678)
    killpg_mock = Mock()
    monkeypatch.setattr(process_compat, "_getpgid", getpgid_mock)
    monkeypatch.setattr(process_compat, "_killpg", killpg_mock)
    compat = PosixProcessCompat()

    compat.graceful_kill(cast(ProcessHandle, process), grace_period=0.1)

    assert getpgid_mock.call_count == 2
    assert killpg_mock.call_args_list[0].args == (5678, signal.SIGTERM)
    assert killpg_mock.call_args_list[1].args == (5678, process_compat._SIGKILL)
    process.kill.assert_called_once_with()
    assert process.wait.call_count == 2


def test_posix_graceful_kill_falls_back_to_process_terminate(monkeypatch: pytest.MonkeyPatch) -> None:
    process = Mock()
    process.pid = 1234
    process.wait.return_value = 0
    monkeypatch.setattr(process_compat, "_getpgid", Mock(side_effect=ProcessLookupError()))
    monkeypatch.setattr(process_compat, "_killpg", Mock())
    compat = PosixProcessCompat()

    compat.graceful_kill(cast(ProcessHandle, process), grace_period=0.1)

    process.terminate.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=0.1)


def test_posix_supported_cleanup_signals_registers_term_and_int() -> None:
    assert PosixProcessCompat().supported_cleanup_signals() == (signal.SIGTERM, signal.SIGINT)
