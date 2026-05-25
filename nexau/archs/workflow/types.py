"""Shared JSON types for workflow runtime.

RFC-0027: Workflow JSON value boundary

Workflow definitions, node inputs/outputs, event payloads, and durable state are
persisted as JSON-compatible values. Keeping the alias local avoids spreading
untyped dictionaries through the workflow package.
"""

from __future__ import annotations

from typing import cast

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]
type JsonArray = list[JsonValue]


def json_object(value: object, *, label: str = "value") -> JsonObject:
    """Return *value* as a JSON object or raise a clear error."""

    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return cast(JsonObject, value)


def json_array(value: object, *, label: str = "value") -> JsonArray:
    """Return *value* as a JSON array or raise a clear error."""

    if not isinstance(value, list):
        raise TypeError(f"{label} must be a JSON array")
    return cast(JsonArray, value)


def json_value(value: object) -> JsonValue:
    """Narrow an already JSON-compatible object to ``JsonValue``."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        value_list = cast(list[object], value)
        return [json_value(item) for item in value_list]
    if isinstance(value, dict):
        value_dict = cast(dict[object, object], value)
        return {str(key): json_value(item) for key, item in value_dict.items()}
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def merge_json_objects(base: JsonObject, patch: JsonObject) -> JsonObject:
    """Return a shallow JSON object merge used for workflow state patches."""

    merged = dict(base)
    merged.update(patch)
    return merged
