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

"""Global cleanup management for agents."""

import atexit
import logging
import os
import signal
import threading
import weakref
from types import FrameType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexau.archs.sandbox import BaseSandboxManager

logger = logging.getLogger(__name__)


class CleanupManager:
    """Singleton manager for agent cleanup on process termination."""

    _instance: "CleanupManager | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "CleanupManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Global registry to track all active agents for cleanup
        self._active_agents: weakref.WeakSet[Any] = weakref.WeakSet()
        self._cleanup_registered = False
        self._cleanup_lock = threading.Lock()
        self._initialized = True
        self._sandbox_manager: BaseSandboxManager[Any] | None = None

    def register_agent(self, agent: Any) -> None:
        """Register an agent for cleanup tracking."""
        self._active_agents.add(agent)
        self._register_cleanup_handlers()

    def register_sandbox_manager(self, sandbox_manager: Any) -> None:
        """Register a sandbox manager for cleanup tracking."""
        self._sandbox_manager = sandbox_manager
        self._register_cleanup_handlers()

    def _register_cleanup_handlers(self) -> None:
        """Register cleanup handlers for process termination."""
        with self._cleanup_lock:
            if self._cleanup_registered:
                return

        try:
            # Register signal handlers for graceful shutdown
            # Signal handlers can only be registered in the main thread
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            logger.debug("🔧 Signal handlers registered")
        except ValueError as e:
            # This happens when not in main thread - signal handlers can't be registered
            logger.debug(
                f"Could not register signal handlers (not in main thread): {e}",
            )

            # Register atexit handler as fallback
            atexit.register(self._cleanup_sandbox)
            atexit.register(self._cleanup_all_agents)

            self._cleanup_registered = True
            logger.debug("🔧 Cleanup handlers registered")

    def _cleanup_all_agents(self) -> None:
        """Clean up all active agents and their running sub-agents."""
        try:
            logger.info("🧹 Cleaning up all active agents and sub-agents...")
        except (ValueError, OSError):
            # Logging may fail during interpreter shutdown
            pass
        agents_to_cleanup = list(self._active_agents)

        for agent in agents_to_cleanup:
            try:
                agent.stop()
            except Exception as e:
                try:
                    logger.error(
                        f"❌ Error cleaning up agent {getattr(agent, 'name', 'unknown')}: {e}",
                    )
                except (ValueError, OSError):
                    pass

        try:
            logger.info("✅ Agent cleanup completed")
        except (ValueError, OSError):
            pass

    def _cleanup_sandbox(self) -> None:
        """Clean up active sandbox."""
        try:
            logger.info("🧹 Cleaning up active sandbox...")
        except (ValueError, OSError):
            pass

        if self._sandbox_manager is None:
            return

        try:
            self._sandbox_manager.stop()
        except Exception as e:
            try:
                logger.error(
                    f"❌ Error cleaning up sandbox {getattr(self._sandbox_manager.instance, 'sandbox_id', 'unknown')}: {e}",
                )
            except (ValueError, OSError):
                pass

        try:
            logger.info("✅ Sandbox cleanup completed")
        except (ValueError, OSError):
            pass

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        """Handle termination signals by cleaning up agents."""
        try:
            logger.info(f"🚨 Received signal {signum}, initiating cleanup...")
        except (ValueError, OSError):
            pass
        self._cleanup_sandbox()
        self._cleanup_all_agents()
        # Re-raise the signal with default handler to ensure proper termination
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# Global instance
cleanup_manager = CleanupManager()
