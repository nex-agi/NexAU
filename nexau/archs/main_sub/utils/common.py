import asyncio
import concurrent.futures
import importlib
import os
import re
from collections.abc import Coroutine
from typing import Any, cast

import yaml


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


def run_sync[T](coro: Coroutine[Any, Any, T], timeout: float | None = None) -> T:
    """Run an async coroutine synchronously, handling event loop contexts.

    This function safely runs async code from sync context, whether or not
    an event loop is already running.

    Args:
        coro: The coroutine to run
        timeout: Maximum time to wait in seconds. None means no timeout.

    Returns:
        The result of the coroutine

    Raises:
        TimeoutError: If the operation times out (when timeout is set)
        Exception: Any exception raised by the coroutine
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop - just run directly
        return asyncio.run(coro)

    # Has running loop - use thread pool to avoid deadlock
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)


YamlValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def import_from_string(import_string: str) -> Any:
    """
    Import a function or class from a string specification.

    Args:
        import_string: String in format "module.path:function_name"

    Returns:
        Imported function or class
    """
    try:
        if ":" not in import_string:
            raise ValueError("Import string must contain ':' separator")

        module_path, attr_name = import_string.rsplit(":", 1)

        # Import the module
        module = importlib.import_module(module_path)

        # Get the attribute
        if not hasattr(module, attr_name):
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{attr_name}'",
            )

        return getattr(module, attr_name)

    except ImportError as e:
        raise ConfigError(
            f"Could not import module from '{import_string}': {e}",
        )
    except AttributeError as e:
        raise ConfigError(
            f"Could not import attribute from '{import_string}': {e}",
        )
    except Exception as e:
        raise ConfigError(f"Error importing from '{import_string}': {e}")


def load_yaml_with_vars(path: str | os.PathLike[str]) -> YamlValue:
    with open(path, encoding="utf-8") as f:
        config_text = f.read()

    base_dir = os.path.dirname(os.path.abspath(path))
    config_text = config_text.replace("${this_file_dir}", base_dir)

    # Replace ${env.VAR_NAME} placeholders with environment variables
    env_pattern = re.compile(r"\$\{env\.([A-Za-z_][A-Za-z0-9_]*)\}")

    def _replace_env(match: re.Match[str]) -> str:
        env_name = match.group(1)
        if env_name not in os.environ:
            raise ConfigError(f"Environment variable '{env_name}' is not set")
        return os.environ[env_name]

    config_text = env_pattern.sub(_replace_env, config_text)

    # deal variables in the YAML file
    loaded_config: YamlValue = yaml.safe_load(config_text)

    if not isinstance(loaded_config, dict):
        return loaded_config

    yaml_variables = loaded_config.get("variables")
    if yaml_variables is None:
        return loaded_config

    if not isinstance(yaml_variables, dict):
        raise ConfigError("'variables' must be a mapping if provided in YAML")
    yaml_variables = cast(dict[str, Any], yaml_variables)

    # Replace ${variables.foo.bar} occurrences directly in the raw text
    var_pattern = re.compile(
        r"\$\{variables\.([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\}",
    )

    def _resolve_var(match: re.Match[str]) -> str:
        path = match.group(1).split(".")
        current: YamlValue = yaml_variables
        for part in path:
            if not isinstance(current, dict) or part not in current:
                raise ConfigError(f"Variable '{match.group(1)}' is not defined in 'variables'")
            current = current[part]
        if isinstance(current, (dict, list)):
            raise ConfigError(
                f"Variable '{match.group(1)}' resolves to a non-scalar value and cannot be embedded in a string",
            )
        return str(current)

    config_text = var_pattern.sub(_resolve_var, config_text)
    resolved_config: YamlValue = yaml.safe_load(config_text)
    if isinstance(resolved_config, dict):
        resolved_config.pop("variables", None)
    return resolved_config
