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

"""Agent state management for unified state container."""

from typing import Any, Optional

from langfuse import LangfuseSpan

from .agent_context import AgentContext, GlobalStorage


class AgentState:
    """A unified container for an agent's runtime state.

    This class encapsulates all runtime state for an agent, including:
    - agent_name: The name of the agent
    - agent_id: The unique identifier of the agent
    - context: The AgentContext instance for runtime context management
    - global_storage: The GlobalStorage instance for persistent state
    """

    def __init__(
        self,
        agent_name: str,
        agent_id: str,
        context: AgentContext,
        global_storage: GlobalStorage,
        parent_agent_state: Optional["AgentState"] = None,
        langfuse_span: LangfuseSpan | None = None,
    ):
        """Initialize agent state.

        Args:
            agent_name: The name of the agent
            agent_id: The unique identifier of the agent
            context: The AgentContext instance for runtime context management
            global_storage: The GlobalStorage instance
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.context = context
        self.global_storage = global_storage
        self.parent_agent_state = parent_agent_state
        self.langfuse_span = langfuse_span

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the context.

        Args:
            key: The context key to retrieve
            default: Default value if key not found

        Returns:
            The context value or default
        """
        return self.context.get_context_value(key, default)

    def set_context_value(self, key: str, value: Any) -> None:
        """Set a value in the context.

        Args:
            key: The context key to set
            value: The value to set
        """
        self.context.set_context_value(key, value)

    def get_global_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the global_storage.

        Args:
            key: The global storage key to retrieve
            default: Default value if key not found

        Returns:
            The global storage value or default
        """
        return self.global_storage.get(key, default)

    def set_global_value(self, key: str, value: Any) -> None:
        """Set a value in the global_storage.

        Args:
            key: The global storage key to set
            value: The value to set
        """
        self.global_storage.set(key, value)

    def __repr__(self) -> str:
        """String representation of the agent state."""
        return f"AgentState(agent_name='{self.agent_name}', agent_id='{self.agent_id}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        context_keys = len(self.context.context)
        global_keys = len(self.global_storage.keys())
        return f"AgentState for '{self.agent_name}' with {context_keys} context keys and {global_keys} global keys"
