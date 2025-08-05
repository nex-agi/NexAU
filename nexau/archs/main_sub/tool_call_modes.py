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

"""Shared constants and helpers for tool call modes."""

from __future__ import annotations

VALID_TOOL_CALL_MODES: set[str] = {"xml", "openai", "anthorpic"}
STRUCTURED_TOOL_CALL_MODES: set[str] = {"openai", "anthorpic"}
DEFAULT_TOOL_CALL_MODE = "openai"


def normalize_tool_call_mode(mode: str | None) -> str:
    """Normalize a tool_call_mode string and validate it."""
    normalized = (mode or DEFAULT_TOOL_CALL_MODE).lower()
    if normalized not in VALID_TOOL_CALL_MODES:
        raise ValueError(
            "tool_call_mode must be one of 'xml', 'openai', or 'anthorpic'",
        )
    return normalized
