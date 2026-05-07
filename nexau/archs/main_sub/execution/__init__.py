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

"""Execution components for agent task processing."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .executor import Executor
    from .llm_caller import LLMCaller
    from .stop_reason import AgentStopReason
    from .stop_result import StopResult
    from .subagent_manager import SubAgentManager
    from .tool_executor import ToolExecutor

__all__ = [
    "AgentStopReason",
    "Executor",
    "StopResult",
    "LLMCaller",
    "SubAgentManager",
    "ToolExecutor",
]


def _cache_export(name: str, value: object) -> object:
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Lazily resolve execution exports without importing Executor/LLMCaller."""

    if name == "AgentStopReason":
        from .stop_reason import AgentStopReason

        return _cache_export(name, AgentStopReason)
    if name == "Executor":
        from .executor import Executor

        return _cache_export(name, Executor)
    if name == "StopResult":
        from .stop_result import StopResult

        return _cache_export(name, StopResult)
    if name == "LLMCaller":
        from .llm_caller import LLMCaller

        return _cache_export(name, LLMCaller)
    if name == "SubAgentManager":
        from .subagent_manager import SubAgentManager

        return _cache_export(name, SubAgentManager)
    if name == "ToolExecutor":
        from .tool_executor import ToolExecutor

        return _cache_export(name, ToolExecutor)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
