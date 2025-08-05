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

logger = logging.getLogger(__name__)


class CleanupManager:
    """Singleton manager for agent cleanup on process termination."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
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
        self._active_agents: set = weakref.WeakSet()
        self._cleanup_registered = False
        self._cleanup_lock = threading.Lock()
        self._initialized = True

    def register_agent(self, agent) -> None:
        """Register an agent for cleanup tracking."""
        self._active_agents.add(agent)
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
            atexit.register(self._cleanup_all_agents)

            self._cleanup_registered = True
            logger.debug("🔧 Cleanup handlers registered")

    def _cleanup_all_agents(self) -> None:
        """Clean up all active agents and their running sub-agents."""
        logger.info("🧹 Cleaning up all active agents and sub-agents...")
        agents_to_cleanup = list(self._active_agents)

        for agent in agents_to_cleanup:
            try:
                agent.stop()
            except Exception as e:
                logger.error(
                    f"❌ Error cleaning up agent {getattr(agent, 'name', 'unknown')}: {e}",
                )

        logger.info("✅ Agent cleanup completed")

    def _signal_handler(self, signum, frame):
        """Handle termination signals by cleaning up agents."""
        logger.info(f"🚨 Received signal {signum}, initiating cleanup...")
        self._cleanup_all_agents()
        # Re-raise the signal with default handler to ensure proper termination
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)


# Global instance
cleanup_manager = CleanupManager()
