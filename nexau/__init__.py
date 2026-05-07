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

"""Nexau - A general-purpose agent framework.

To use Nexau core functionality:
    from nexau import Agent, AgentConfig, LLMConfig

To use transports:
    from nexau.transports import SSETransportServer, TransportConfig
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .archs.llm.llm_config import LLMConfig
    from .archs.main_sub.agent import Agent
    from .archs.main_sub.config import AgentConfig
    from .archs.main_sub.skill import Skill
    from .archs.tool.tool import Tool
    from .archs.tracer import BaseTracer, CompositeTracer, Span, SpanType, TraceContext

__all__ = [
    "Agent",
    "Tool",
    "LLMConfig",
    "AgentConfig",
    "Skill",
    # Tracer components
    "BaseTracer",
    "CompositeTracer",
    "Span",
    "SpanType",
    "TraceContext",
]


def _cache_export(name: str, value: object) -> object:
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Lazily resolve public package exports without importing the agent stack."""

    if name == "Agent":
        from .archs.main_sub.agent import Agent

        return _cache_export(name, Agent)
    if name == "Tool":
        from .archs.tool.tool import Tool

        return _cache_export(name, Tool)
    if name == "LLMConfig":
        from .archs.llm.llm_config import LLMConfig

        return _cache_export(name, LLMConfig)
    if name == "AgentConfig":
        from .archs.main_sub.config import AgentConfig

        return _cache_export(name, AgentConfig)
    if name == "Skill":
        from .archs.main_sub.skill import Skill

        return _cache_export(name, Skill)
    if name == "BaseTracer":
        from .archs.tracer import BaseTracer

        return _cache_export(name, BaseTracer)
    if name == "CompositeTracer":
        from .archs.tracer import CompositeTracer

        return _cache_export(name, CompositeTracer)
    if name == "Span":
        from .archs.tracer import Span

        return _cache_export(name, Span)
    if name == "SpanType":
        from .archs.tracer import SpanType

        return _cache_export(name, SpanType)
    if name == "TraceContext":
        from .archs.tracer import TraceContext

        return _cache_export(name, TraceContext)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
