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

"""Tests for CleanupManager platform signal integration.

RFC-0020: Windows validation keeps cleanup behavior tied to the platform
compatibility layer instead of registering POSIX-only signals directly.
"""

from __future__ import annotations

import importlib
import signal
from unittest.mock import Mock

import pytest

from nexau.archs.main_sub.utils.cleanup_manager import CleanupManager

cleanup_manager_module = importlib.import_module("nexau.archs.main_sub.utils.cleanup_manager")


@pytest.fixture(autouse=True)
def restore_cleanup_manager_state():
    manager = CleanupManager()
    active_agents = list(manager._active_agents)
    sandbox_manager = manager._sandbox_manager
    cleanup_registered = manager._cleanup_registered
    yield
    manager._active_agents.clear()
    for agent in active_agents:
        manager._active_agents.add(agent)
    manager._sandbox_manager = sandbox_manager
    manager._cleanup_registered = cleanup_registered


def _fresh_manager() -> CleanupManager:
    manager = CleanupManager()
    manager._active_agents.clear()
    manager._sandbox_manager = None
    manager._cleanup_registered = False
    return manager


def test_register_cleanup_handlers_uses_platform_supported_signals(monkeypatch) -> None:
    """RFC-0020: Windows cleanup registration must not assume POSIX signals."""
    manager = _fresh_manager()
    signal_mock = Mock()
    monkeypatch.setattr(cleanup_manager_module, "supported_cleanup_signals", lambda: (signal.SIGINT,))
    monkeypatch.setattr(cleanup_manager_module.signal, "signal", signal_mock)

    manager._register_cleanup_handlers()

    signal_mock.assert_called_once_with(signal.SIGINT, manager._signal_handler)
    assert manager._cleanup_registered is True


def test_signal_handler_cleans_resources_before_platform_reemit(monkeypatch) -> None:
    """RFC-0020: signal cleanup delegates final termination to process_compat."""

    class Agent:
        def __init__(self) -> None:
            self.sync_cleanup = Mock()

    manager = _fresh_manager()
    agent = Agent()
    sandbox_manager = Mock()
    manager._active_agents.add(agent)
    manager._sandbox_manager = sandbox_manager
    reemit_mock = Mock(side_effect=SystemExit(128 + signal.SIGINT))
    monkeypatch.setattr(cleanup_manager_module, "reemit_termination_signal", reemit_mock)

    try:
        manager._signal_handler(signal.SIGINT, None)
    except SystemExit as exc:
        assert exc.code == 128 + signal.SIGINT

    sandbox_manager.stop.assert_called_once_with()
    agent.sync_cleanup.assert_called_once_with()
    reemit_mock.assert_called_once_with(signal.SIGINT)
