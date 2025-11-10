"""Public API for the config package with lazy imports to avoid cycles."""

from importlib import import_module

__all__ = ["load_agent_config", "ConfigError"]


def __getattr__(name):
    if name in __all__:
        module = import_module("nexau.archs.config.config_loader")
        return getattr(module, name)
    raise AttributeError(f"module 'nexau.archs.config' has no attribute '{name}'")
