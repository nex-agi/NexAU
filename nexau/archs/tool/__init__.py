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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tool import (
        StructuredToolDefinition,
        Tool,
        build_structured_tool_definition,
        normalize_structured_tool_definition,
        structured_tool_definition_to_anthropic,
        structured_tool_definition_to_openai,
    )
    from .tool_registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolRegistry",
    "StructuredToolDefinition",
    "build_structured_tool_definition",
    "normalize_structured_tool_definition",
    "structured_tool_definition_to_anthropic",
    "structured_tool_definition_to_openai",
]


def _cache_export(name: str, value: object) -> object:
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Lazily resolve tool exports without importing provider SDK type modules."""

    if name == "Tool":
        from .tool import Tool

        return _cache_export(name, Tool)
    if name == "ToolRegistry":
        from .tool_registry import ToolRegistry

        return _cache_export(name, ToolRegistry)
    if name == "StructuredToolDefinition":
        from .tool import StructuredToolDefinition

        return _cache_export(name, StructuredToolDefinition)
    if name == "build_structured_tool_definition":
        from .tool import build_structured_tool_definition

        return _cache_export(name, build_structured_tool_definition)
    if name == "normalize_structured_tool_definition":
        from .tool import normalize_structured_tool_definition

        return _cache_export(name, normalize_structured_tool_definition)
    if name == "structured_tool_definition_to_anthropic":
        from .tool import structured_tool_definition_to_anthropic

        return _cache_export(name, structured_tool_definition_to_anthropic)
    if name == "structured_tool_definition_to_openai":
        from .tool import structured_tool_definition_to_openai

        return _cache_export(name, structured_tool_definition_to_openai)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
