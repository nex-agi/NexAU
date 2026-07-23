import hashlib
import importlib
import importlib.util
import os
import re
import sys
from types import ModuleType
from typing import Any, cast

import yaml


class ConfigError(Exception):
    """Exception raised for configuration errors."""

    pass


YamlValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def import_from_string(import_string: str) -> Any:
    """Import a function or class from a string specification.

    RFC-0197: 解析被同名子模块遮蔽的 lazy package export。

    Args:
        import_string: String in format "module.path:function_name"

    Returns:
        Imported function or class
    """
    try:
        if ":" not in import_string:
            raise ValueError("Import string must contain ':' separator")

        module_path, attr_name = import_string.rsplit(":", 1)

        module_file = os.path.expanduser(module_path)
        if module_file.endswith(".py") or os.path.sep in module_file:
            module_file = os.path.abspath(module_file)
            if not os.path.exists(module_file):
                raise ImportError(f"Python file '{module_file}' does not exist")
            module_name = f"_nexau_dynamic_{hashlib.sha1(module_file.encode('utf-8')).hexdigest()}"
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not load Python file '{module_file}'")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)

        missing = object()
        attribute = module.__dict__.get(attr_name, missing)
        module_getattr = module.__dict__.get("__getattr__")
        if isinstance(attribute, ModuleType) and callable(module_getattr):
            try:
                return module_getattr(attr_name)
            except AttributeError:
                pass
        if attribute is not missing:
            return attribute
        if callable(module_getattr):
            try:
                return module_getattr(attr_name)
            except AttributeError:
                pass
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}'",
        )

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


def load_yaml_text_with_vars(config_text: str, base_dir: str | os.PathLike[str]) -> YamlValue:
    base_dir = os.path.abspath(os.fspath(base_dir))
    yaml_safe_base_dir = base_dir.replace("\\", "/") if os.name == "nt" else base_dir
    config_text = config_text.replace("${this_file_dir}", yaml_safe_base_dir)

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


def load_yaml_with_vars(path: str | os.PathLike[str]) -> YamlValue:
    with open(path, encoding="utf-8") as f:
        config_text = f.read()

    base_dir = os.path.dirname(os.path.abspath(path))
    return load_yaml_text_with_vars(config_text, base_dir)
