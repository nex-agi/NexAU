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

"""Team collaboration tools.

RFC-0002: Team 协作工具集

Provides YAML-defined tools for team lifecycle management,
messaging, and task coordination.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from nexau.archs.tool.tool import Tool

_TOOL_DIR = Path(__file__).parent

_ALL_TOOL_NAMES: list[str] = [
    "spawn_teammate",
    # "remove_teammate",
    "message",
    "broadcast",
    "list_teammates",
    "list_tasks",
    "create_task",
    "claim_task",
    "update_task_status",
    "release_task",
    "finish_team",
]

_LEADER_ONLY: set[str] = {"spawn_teammate", "remove_teammate", "create_task", "claim_task", "finish_team"}


def _load_tool(name: str) -> Tool:
    """Load a team tool by name.

    RFC-0002: 按名称加载 team tool（YAML + Python binding）
    """
    yaml_path = str(_TOOL_DIR / f"{name}.yaml")
    module = import_module(f"nexau.archs.main_sub.team.tools.{name}")
    binding = getattr(module, name)  # noqa: B009
    return Tool.from_yaml(yaml_path, binding=binding)


_STOP_TOOLS: set[str] = {"finish_team"}


def get_leader_tools() -> list[Tool]:
    """Get all tools for the team leader.

    RFC-0002: Leader 工具集（全部 11 个）
    """
    return [_load_tool(name) for name in _ALL_TOOL_NAMES]


def get_leader_stop_tools() -> set[str]:
    """Get stop tool names for the team leader.

    RFC-0002: Leader stop tool 名称集合
    """
    return set(_STOP_TOOLS)


def get_teammate_tools() -> list[Tool]:
    """Get tools for teammate agents (no spawn/remove/create_task).

    RFC-0002: Teammate 工具集（排除 leader-only 工具）
    """
    return [_load_tool(name) for name in _ALL_TOOL_NAMES if name not in _LEADER_ONLY]
