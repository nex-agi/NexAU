"""Agent context manager for context management."""

import threading
from contextlib import contextmanager
from typing import Any


class AgentContext:
    """Context manager for agent context."""

    def __init__(self, context: dict[str, Any] | None = None):
        """Initialize agent context with context."""
        self.context = context or {}
        self._original_context = None

        # Track if context has been modified (for prompt refresh)
        self._context_modified = False
        self._modification_callbacks = []

    def __enter__(self):
        """Enter the context and set the thread-local context."""
        global _current_context
        self._original_context = _current_context.context.copy() if _current_context else {}

        # Set current context
        _current_context = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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

    def update_context(self, updates: dict[str, Any]):
        """Update context with new values."""
        self.context.update(updates)
        self._mark_modified()

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a specific context value."""
        return self.context.get(key, default)

    def set_context_value(self, key: str, value: Any):
        """Set a specific context value."""
        self.context[key] = value
        self._mark_modified()

    def _mark_modified(self):
        """Mark the context as modified and trigger callbacks."""
        self._context_modified = True
        for callback in self._modification_callbacks:
            try:
                callback()
            except Exception:
                pass  # Ignore callback errors

    def add_modification_callback(self, callback):
        """Add a callback that gets triggered when context is modified."""
        self._modification_callbacks.append(callback)

    def remove_modification_callback(self, callback):
        """Remove a modification callback."""
        if callback in self._modification_callbacks:
            self._modification_callbacks.remove(callback)

    def is_modified(self) -> bool:
        """Check if context has been modified."""
        return self._context_modified

    def reset_modification_flag(self):
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
        self._storage = {}
        self._locks = {}
        self._storage_lock = threading.RLock()
        self._locks_lock = threading.RLock()

    def set(self, key: str, value: Any):
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

    def update(self, updates: dict[str, Any]):
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

    def keys(self):
        """Get all keys in global storage."""
        with self._storage_lock:
            return list(self._storage.keys())

    def items(self):
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
