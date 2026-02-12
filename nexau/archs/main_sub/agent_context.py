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

"""Agent context manager for context management."""

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from types import TracebackType
from typing import Any, cast

from pydantic_core import core_schema

logger = logging.getLogger(__name__)


class AgentContext:
    """Context manager for agent context."""

    def __init__(self, context: dict[str, Any] | None = None):
        """Initialize agent context with context."""
        self.context = context or {}
        self._original_context: dict[str, Any] | None = None

        # Track if context has been modified (for prompt refresh)
        self._context_modified = False
        self._modification_callbacks: list[Callable[[], None]] = []

    def __enter__(self) -> "AgentContext":
        """Enter the context and set the thread-local context."""
        global _current_context
        self._original_context = _current_context.context.copy() if _current_context else {}

        # Set current context
        _current_context = self
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context and restore previous context."""
        global _current_context
        if self._original_context is not None:
            # Restore previous context if it existed
            if self._original_context:
                _current_context = AgentContext(self._original_context)
            else:
                _current_context = None
        else:
            _current_context = None

    def update_context(self, updates: dict[str, Any]) -> None:
        """Update context with new values."""
        self.context.update(updates)
        self._mark_modified()

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a specific context value."""
        return self.context.get(key, default)

    def set_context_value(self, key: str, value: Any) -> None:
        """Set a specific context value."""
        self.context[key] = value
        self._mark_modified()

    def _mark_modified(self) -> None:
        """Mark the context as modified and trigger callbacks."""
        self._context_modified = True
        for callback in self._modification_callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore callback errors

    def add_modification_callback(self, callback: Callable[[], None]) -> None:
        """Add a callback that gets triggered when context is modified."""
        self._modification_callbacks.append(callback)

    def remove_modification_callback(self, callback: Callable[[], None]) -> None:
        """Remove a modification callback."""
        if callback in self._modification_callbacks:
            self._modification_callbacks.remove(callback)

    def is_modified(self) -> bool:
        """Check if context has been modified."""
        return self._context_modified

    def reset_modification_flag(self) -> None:
        """Reset the modification flag."""
        self._context_modified = False

    def get_context_variables(self) -> dict[str, Any]:
        """Get all context variables for prompt rendering."""
        return self.context.copy()

    def merge_context_variables(
        self,
        existing_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge context variables with existing context, giving priority to context vars."""
        merged = existing_context.copy()
        merged.update(self.get_context_variables())
        return merged

    @classmethod
    def from_sources(
        cls,
        initial_context: dict[str, Any] | None = None,
        legacy_context: dict[str, Any] | None = None,
        template: dict[str, str] | None = None,
    ) -> "AgentContext":
        """Merge multiple context sources into a single AgentContext.

        Priority (highest last): initial_context < legacy_context < template.

        Args:
            initial_context: Base context from agent config.
            legacy_context: Deprecated run()/run_async() context dict.
                If provided, a DeprecationWarning is emitted.
            template: ContextValue.template variables (highest priority).

        Returns:
            A new AgentContext with all sources merged.
        """
        import warnings

        merged: dict[str, Any] = dict(initial_context or {})
        if legacy_context:
            warnings.warn(
                "Passing 'context' dict to run()/run_async() is deprecated. Use 'variables=ContextValue(template={...})' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            merged.update(legacy_context)
        if template:
            merged.update(template)
        return cls(merged)


# Thread-local storage for the current context
_current_context: AgentContext | None = None


def get_context() -> AgentContext | None:
    """Get the current agent context."""
    return _current_context


def get_context_dict() -> dict[str, Any]:
    """Get the current agent context dictionary."""
    if _current_context is None:
        raise RuntimeError(
            "No agent context available. Make sure you're calling this within an agent context.",
        )
    return _current_context.context


def get_context_variables() -> dict[str, Any]:
    """Get all context variables for prompt rendering."""
    if _current_context is None:
        return {}
    return _current_context.get_context_variables()


def merge_context_variables(existing_context: dict[str, Any]) -> dict[str, Any]:
    """Merge context variables with existing context."""
    if _current_context is None:
        return existing_context
    return _current_context.merge_context_variables(existing_context)


class GlobalStorage:
    """Thread-safe storage shared across agents in the same agent hierarchy."""

    def __init__(self):
        self._storage: dict[str, Any] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._storage_lock = threading.RLock()
        self._locks_lock = threading.RLock()

    def to_dict(self) -> dict[str, Any]:
        return dict(self.items())

    @classmethod
    def _validate(cls, v: Any) -> "GlobalStorage":
        if isinstance(v, GlobalStorage):
            return v
        if isinstance(v, dict):
            gs = GlobalStorage()
            gs.update(cast(dict[str, Any], v))
            return gs
        return GlobalStorage()

    @classmethod
    def _serialize(cls, v: Any) -> dict[str, Any]:
        from nexau.archs.session.models.serialization_utils import sanitize_for_serialization

        try:
            if isinstance(v, GlobalStorage):
                return sanitize_for_serialization(v.to_dict())
            if isinstance(v, dict):
                return sanitize_for_serialization(v)
            return {}
        except (RecursionError, ValueError, TypeError) as e:
            # Return empty dict if serialization fails due to circular references
            # or non-serializable objects
            logger.warning("GlobalStorage serialization failed: %s", e)
            return {}

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any) -> core_schema.CoreSchema:
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(cls._validate),
            python_schema=core_schema.no_info_plain_validator_function(cls._validate),
            serialization=core_schema.plain_serializer_function_ser_schema(cls._serialize, info_arg=False),
        )

    def set(self, key: str, value: Any) -> None:
        """Set a value in global storage."""
        # Use key-specific lock if it exists, otherwise use global lock
        key_lock = self._locks.get(key)
        if key_lock:
            with key_lock:
                self._storage[key] = value
        else:
            with self._storage_lock:
                self._storage[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from global storage."""
        # Use key-specific lock if it exists, otherwise use global lock
        key_lock = self._locks.get(key)
        if key_lock:
            with key_lock:
                return self._storage.get(key, default)
        else:
            with self._storage_lock:
                return self._storage.get(key, default)

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple values in global storage."""
        # Check if any keys have specific locks
        keys_with_locks = [k for k in updates.keys() if k in self._locks]

        if keys_with_locks:
            # If some keys have specific locks, we need to handle them individually
            for key, value in updates.items():
                self.set(key, value)
        else:
            # If no keys have specific locks, use global lock
            with self._storage_lock:
                self._storage.update(updates)

    def delete(self, key: str) -> bool:
        """Delete a key from global storage. Returns True if key existed."""
        # Use key-specific lock if it exists, otherwise use global lock
        key_lock = self._locks.get(key)
        if key_lock:
            with key_lock:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False
        else:
            with self._storage_lock:
                if key in self._storage:
                    del self._storage[key]
                    return True
                return False

    def keys(self) -> list[str]:
        """Get all keys in global storage."""
        with self._storage_lock:
            return list(self._storage.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Get all items in global storage."""
        with self._storage_lock:
            return list(self._storage.items())

    def clear(self):
        """Clear all data from global storage."""
        with self._storage_lock:
            self._storage.clear()
        with self._locks_lock:
            self._locks.clear()

    def _get_lock(self, key: str) -> threading.RLock:
        """Get or create a lock for a specific key."""
        with self._locks_lock:
            if key not in self._locks:
                self._locks[key] = threading.RLock()
            return self._locks[key]

    @contextmanager
    def lock_key(self, key: str):
        """Context manager to lock a specific key for exclusive access."""
        lock = self._get_lock(key)
        lock.acquire()
        try:
            yield self
        finally:
            lock.release()

    @contextmanager
    def lock_multiple(self, *keys: str):
        """Context manager to lock multiple keys for exclusive access."""
        locks = [
            self._get_lock(key)
            for key in sorted(
                keys,
            )
        ]  # Sort to prevent deadlock
        for lock in locks:
            lock.acquire()
        try:
            yield self
        finally:
            for lock in reversed(locks):
                lock.release()
