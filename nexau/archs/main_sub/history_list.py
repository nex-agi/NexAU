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

"""HistoryList: A list that automatically persists modifications to SessionManager."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING, Any, SupportsIndex

from nexau.core.messages import Message, Role

if TYPE_CHECKING:
    from nexau.archs.session import AgentRunActionKey, SessionManager

logger = logging.getLogger(__name__)


class HistoryList(list[Message]):
    """A list that intercepts modifications and persists to SessionManager.

    This class provides transparent persistence for agent history with run-level semantics:
    - append/extend operations persist new messages immediately (APPEND action)
    - replace_all() intelligently detects append vs replace operations
    - Index assignment (__setitem__) only updates locally, does NOT persist
      (run-level API doesn't support message-level updates)

    The class maintains backward compatibility with plain list[Message] usage
    while adding automatic persistence when SessionManager is available.

    Note: The run-level API is designed for immutable history. Each agent.run()
    produces new messages that are appended. To modify history, use replace_all()
    which creates a REPLACE action.
    """

    def __init__(
        self,
        messages: list[Message] | None = None,
        *,
        session_manager: SessionManager | None = None,
        history_key: AgentRunActionKey | None = None,
        run_id: str | None = None,
        root_run_id: str | None = None,
        parent_run_id: str | None = None,
        agent_name: str = "",
    ):
        """Initialize HistoryList.

        Args:
            messages: Initial messages (optional)
            session_manager: SessionManager for persistence (optional)
            history_key: Key for history storage (optional)
            run_id: Current run ID (optional)
            root_run_id: Root run ID (optional)
            parent_run_id: Parent run ID (optional)
            agent_name: Agent name for logging (optional)
        """
        super().__init__(messages or [])

        self._session_manager = session_manager
        self._history_key = history_key
        self._run_id = run_id
        self._root_run_id = root_run_id
        self._parent_run_id = parent_run_id
        self._agent_name = agent_name

        # Flag to enable/disable persistence
        self._persistence_enabled = session_manager is not None and history_key is not None

        # Accumulate messages added in current run (for batch persistence)
        self._pending_messages: list[Message] = []
        self._baseline_fingerprints: list[str] = self._compute_fingerprints([m for m in self if m.role != Role.SYSTEM])

    @property
    def has_pending_messages(self) -> bool:
        """Check if there are unflushed pending messages."""
        return bool(self._pending_messages)

    def append(self, item: Message) -> None:
        """Append a message (will be persisted on flush)."""
        super().append(item)

        if self._persistence_enabled and item.role != Role.SYSTEM:
            self._pending_messages.append(item)

    def extend(self, items: list[Message] | tuple[Message, ...]) -> None:  # type: ignore[override]
        """Extend with messages (will be persisted on flush)."""
        super().extend(items)

        if self._persistence_enabled:
            non_system = [msg for msg in items if msg.role != Role.SYSTEM]
            self._pending_messages.extend(non_system)

    def __setitem__(self, key: SupportsIndex | slice, value: Message | list[Message]) -> None:  # type: ignore[override]
        """Intercept item assignment (local update only, no persistence).

        Note: The run-level API doesn't support message-level updates.
        This method only updates the local list. To persist changes,
        use replace_all() which creates a REPLACE action.

        Args:
            key: Index or slice
            value: New message(s)

        Examples:
            >>> history[0] = Message.user("modified")  # Local only
            >>> history[0:2] = [msg1, msg2]  # Local only
            >>> history.replace_all([msg1, msg2])  # Persists
        """
        if isinstance(key, slice):
            if isinstance(value, list):
                super().__setitem__(key, value)
            else:
                assert isinstance(value, Message)
                super().__setitem__(key, [value])
        else:
            if isinstance(value, list):
                super().__setitem__(key, value[0])
            else:
                assert isinstance(value, Message)
                super().__setitem__(key, value)

    def replace_all(self, new_messages: list[Message], *, update_baseline: bool = False) -> None:
        """Replace all messages with smart detection.

        This method intelligently detects whether the operation is:
        1. A simple append (old messages + new messages) -> persist only new
        2. A true replacement (different messages) -> create replace record

        Args:
            new_messages: New message list
            update_baseline: If True, update baseline fingerprints to match new messages.
                           Use this when loading history from storage to set initial state.
                           Default is False to allow flush() to detect changes.
        """
        self.clear()
        super().extend(new_messages)
        if self._persistence_enabled:
            self._pending_messages.clear()
            if update_baseline:
                # Update baseline fingerprints to match the new message list
                # This ensures flush() only persists truly new messages added after this call
                current_non_system = [m for m in self if m.role != Role.SYSTEM]
                self._baseline_fingerprints = self._compute_fingerprints(current_non_system)

    @staticmethod
    def _fingerprint_message(msg: Message) -> str:
        payload = msg.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _compute_fingerprints(cls, messages: list[Message]) -> list[str]:
        return [cls._fingerprint_message(m) for m in messages]

    def _schedule_async(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule an async coroutine to run.

        Args:
            coro: Coroutine to schedule
        """
        try:
            asyncio.get_running_loop()
            # We're in an async context, create a task
            asyncio.create_task(coro)  # type: ignore[arg-type]
        except RuntimeError:
            # No running loop, run synchronously
            asyncio.run(coro)  # type: ignore[arg-type]

    async def _persist_flush_async(
        self,
        *,
        append_messages: list[Message] | None,
        replace_messages: list[Message] | None,
    ) -> None:
        if not self._session_manager or not self._history_key:
            return
        run_id = self._run_id or "unknown"
        root_run_id = self._root_run_id or "unknown"

        try:
            if replace_messages is not None:
                if not replace_messages:
                    return
                await self._session_manager.agent_run_action.persist_replace(
                    key=self._history_key,
                    run_id=run_id,
                    root_run_id=root_run_id,
                    messages=replace_messages,
                    parent_run_id=self._parent_run_id,
                    agent_name=self._agent_name,
                )
            elif append_messages is not None:
                if not append_messages:
                    return
                await self._session_manager.agent_run_action.persist_append(
                    key=self._history_key,
                    run_id=run_id,
                    root_run_id=root_run_id,
                    parent_run_id=self._parent_run_id,
                    agent_name=self._agent_name,
                    messages=append_messages,
                )
        except Exception as e:
            logger.error(f"❌ Failed to flush history: {e}")

    def flush(self) -> None:
        """Flush pending messages to persistence.

        This should be called at the end of each run to persist all accumulated messages.
        """
        if not self._persistence_enabled:
            return

        current_non_system = [m for m in self if m.role != Role.SYSTEM]

        is_append_only = True
        if len(current_non_system) < len(self._baseline_fingerprints):
            is_append_only = False
        else:
            for i in range(len(self._baseline_fingerprints)):
                if self._fingerprint_message(current_non_system[i]) != self._baseline_fingerprints[i]:
                    is_append_only = False
                    break

        append_messages: list[Message] | None = None
        replace_messages: list[Message] | None = None
        if is_append_only:
            if len(current_non_system) > len(self._baseline_fingerprints):
                append_messages = current_non_system[len(self._baseline_fingerprints) :]
        else:
            replace_messages = current_non_system

        if append_messages is not None or replace_messages is not None:
            self._schedule_async(
                self._persist_flush_async(
                    append_messages=append_messages,
                    replace_messages=replace_messages,
                )
            )

        self._pending_messages.clear()
        self._baseline_fingerprints = self._compute_fingerprints(current_non_system)

    def update_context(
        self,
        *,
        run_id: str | None = None,
        root_run_id: str | None = None,
        parent_run_id: str | None = None,
    ) -> None:
        """Update the run context for persistence.

        This is called when starting a new run to update the run IDs.
        Automatically flushes pending messages from previous run before updating context.

        Args:
            run_id: New run ID
            root_run_id: New root run ID
            parent_run_id: New parent run ID
        """
        # Flush pending messages from previous run
        self.flush()

        if run_id is not None:
            self._run_id = run_id
        if root_run_id is not None:
            self._root_run_id = root_run_id
        if parent_run_id is not None:
            self._parent_run_id = parent_run_id

        if self._persistence_enabled:
            current_non_system = [m for m in self if m.role != Role.SYSTEM]
            self._baseline_fingerprints = self._compute_fingerprints(current_non_system)
