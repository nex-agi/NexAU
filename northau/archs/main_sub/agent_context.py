"""Agent context manager for state and config management."""

import threading
from typing import Dict, Any, Optional
from contextvars import ContextVar


class AgentContext:
    """Context manager for agent state and config."""
    
    def __init__(self, state: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize agent context with state and config."""
        self.state = state or {}
        self.config = config or {}
        self._original_state = None
        self._original_config = None
        
        # Track if context has been modified (for prompt refresh)
        self._context_modified = False
        self._modification_callbacks = []
    
    def __enter__(self):
        """Enter the context and set the thread-local context."""
        global _current_context
        self._original_state = _current_context.state.copy() if _current_context else {}
        self._original_config = _current_context.config.copy() if _current_context else {}
        
        # Set current context
        _current_context = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore previous context."""
        global _current_context
        if self._original_state is not None or self._original_config is not None:
            # Restore previous context if it existed
            if self._original_state or self._original_config:
                _current_context = AgentContext(self._original_state, self._original_config)
            else:
                _current_context = None
        else:
            _current_context = None
    
    def update_state(self, updates: Dict[str, Any]):
        """Update state with new values."""
        self.state.update(updates)
        self._mark_modified()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update config with new values."""
        self.config.update(updates)
        self._mark_modified()
    
    def get_state_value(self, key: str, default: Any = None) -> Any:
        """Get a specific state value."""
        return self.state.get(key, default)
    
    def set_state_value(self, key: str, value: Any):
        """Set a specific state value."""
        self.state[key] = value
        self._mark_modified()
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific config value."""
        return self.config.get(key, default)
    
    def set_config_value(self, key: str, value: Any):
        """Set a specific config value."""
        self.config[key] = value
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
    
    def get_context_variables(self) -> Dict[str, Any]:
        """Get all context variables for prompt rendering."""
        return {
            **self.state,
            **self.config
        }
    
    def merge_context_variables(self, existing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Merge context variables with existing context, giving priority to context vars."""
        merged = existing_context.copy()
        merged.update(self.get_context_variables())
        return merged


# Thread-local storage for the current context
_current_context: Optional[AgentContext] = None


def get_state() -> Dict[str, Any]:
    """Get the current agent state from context."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    return _current_context.state


def get_config() -> Dict[str, Any]:
    """Get the current agent config from context."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    return _current_context.config


def get_context() -> Optional[AgentContext]:
    """Get the current agent context."""
    return _current_context


def update_state(**updates):
    """Update the current agent state."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    _current_context.update_state(updates)


def update_config(**updates):
    """Update the current agent config."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    _current_context.update_config(updates)


def get_state_value(key: str, default: Any = None) -> Any:
    """Get a specific state value from context."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    return _current_context.get_state_value(key, default)


def set_state_value(key: str, value: Any):
    """Set a specific state value in context."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    _current_context.set_state_value(key, value)


def get_config_value(key: str, default: Any = None) -> Any:
    """Get a specific config value from context."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    return _current_context.get_config_value(key, default)


def set_config_value(key: str, value: Any):
    """Set a specific config value in context."""
    if _current_context is None:
        raise RuntimeError("No agent context available. Make sure you're calling this within an agent context.")
    _current_context.set_config_value(key, value)


def get_context_variables() -> Dict[str, Any]:
    """Get all context variables for prompt rendering."""
    if _current_context is None:
        return {}
    return _current_context.get_context_variables()


def merge_context_variables(existing_context: Dict[str, Any]) -> Dict[str, Any]:
    """Merge context variables with existing context."""
    if _current_context is None:
        return existing_context
    return _current_context.merge_context_variables(existing_context)