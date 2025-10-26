"""
Test utilities and helper functions.

This module contains utility functions and helpers for testing.
"""

import json
import tempfile
from pathlib import Path
from typing import Any


def create_temp_file(content: str, suffix: str = ".txt") -> Path:
    """Create a temporary file with given content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return Path(f.name)


def create_temp_yaml_config(config: dict[str, Any]) -> Path:
    """Create a temporary YAML configuration file."""
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        return Path(f.name)


def create_temp_json_file(data: dict[str, Any]) -> Path:
    """Create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


def assert_file_contains(file_path: Path, expected_content: str) -> bool:
    """Assert that a file contains expected content."""
    with open(file_path) as f:
        content = f.read()
    return expected_content in content


def assert_dict_contains(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Assert that actual dict contains all keys and values from expected dict."""
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, dict):
            if not assert_dict_contains(actual[key], value):
                return False
        elif actual[key] != value:
            return False
    return True


def mock_llm_response(content: str, tool_calls: list = None, finish_reason: str = "stop"):
    """Create a mock LLM response."""
    message = {"content": content, "role": "assistant"}

    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


def mock_tool_call(tool_name: str, arguments: dict[str, Any], call_id: str = "call_123"):
    """Create a mock tool call."""
    return {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(arguments)}}
