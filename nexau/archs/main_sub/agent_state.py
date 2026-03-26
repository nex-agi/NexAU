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

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager
    from nexau.archs.main_sub.team.state import AgentTeamState
    from nexau.archs.main_sub.token_trace_session import TokenTraceSession
    from nexau.archs.sandbox.base_sandbox import BaseSandbox, BaseSandboxManager
    from nexau.archs.tool.tool import Tool
    from nexau.archs.tool.tool_registry import ToolRegistry

from .agent_context import AgentContext, GlobalStorage
from .context_value import ContextValue


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
        *,
        agent_name: str,
        agent_id: str,
        run_id: str,
        root_run_id: str,
        context: AgentContext,
        global_storage: GlobalStorage,
        tool_registry: "ToolRegistry",
        parent_agent_state: Optional["AgentState"] = None,
        sandbox: Optional["BaseSandbox"] = None,
        sandbox_manager: Optional["BaseSandboxManager[Any]"] = None,
        variables: ContextValue | None = None,
        team_state: Optional["AgentTeamState"] = None,
        token_trace_session: Optional["TokenTraceSession"] = None,
        subagent_manager: Optional["SubAgentManager"] = None,
    ):
        """Initialize agent state.

        Args:
            agent_name: The name of the agent
            agent_id: The unique identifier of the agent
            run_id: The current run ID
            root_run_id: The root run ID
            context: The AgentContext instance for runtime context management
            global_storage: The GlobalStorage instance
            parent_agent_state: Optional parent state when this is a sub-agent
            tool_registry: ToolRegistry for runtime tool injection
            sandbox: Optional sandbox instance (deprecated, use sandbox_manager)
            sandbox_manager: Optional sandbox manager for lazy sandbox access
            variables: Optional ContextValue with runtime variables
            token_trace_session: Optional token trace session for generate_with_token providers (RFC-0009)
            subagent_manager: Optional SubAgentManager for RecallSubAgent tool access
        """
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.run_id = run_id
        self.root_run_id = root_run_id
        self.context = context
        self.global_storage = global_storage
        self.parent_agent_state = parent_agent_state
        self._tool_registry = tool_registry
        self._sandbox = sandbox
        self._sandbox_manager = sandbox_manager
        self._variables = variables or ContextValue()
        self.team_state = team_state
        self.token_trace_session = token_trace_session
        self._subagent_manager = subagent_manager

    @property
    def subagent_manager(self) -> Optional["SubAgentManager"]:
        """SubAgentManager instance for RecallSubAgent tool access."""
        return self._subagent_manager

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

    def get_variable(self, key: str, default: str | None = None) -> str | None:
        """Get a runtime variable (not in prompt, not exposed to LLM).

        Args:
            key: The variable key to retrieve
            default: Default value if key not found

        Returns:
            The variable value or default
        """
        return self._variables.runtime_vars.get(key, default)

    def get_sandbox_env(self, key: str, default: str | None = None) -> str | None:
        """Get a sandbox environment variable value.

        Args:
            key: The sandbox env key to retrieve
            default: Default value if key not found

        Returns:
            The sandbox env value or default
        """
        return self._variables.sandbox_env.get(key, default)

    @property
    def all_variables(self) -> dict[str, str]:
        """All runtime variables."""
        return dict(self._variables.runtime_vars)

    @property
    def all_sandbox_env(self) -> dict[str, str]:
        """All sandbox environment variables."""
        return dict(self._variables.sandbox_env)

    def get_sandbox(self) -> Optional["BaseSandbox"]:
        """Get the sandbox associated with the agent state.

        功能说明1：使用 start_sync() 在当前事件循环中同步启动 sandbox
        功能说明2：避免在不同事件循环中访问 asyncio 原语导致的问题
        功能说明3：E2B SDK 的 httpx 客户端会在当前事件循环上下文中创建
        功能说明4：如果没有 sandbox_manager，则使用直接设置的 sandbox
        """
        # 使用 start_sync() 在当前线程/事件循环中启动 sandbox
        if self._sandbox_manager is not None:
            return self._sandbox_manager.start_sync()
        return self._sandbox

    def set_sandbox(self, sandbox: "BaseSandbox") -> None:
        """Set the sandbox associated with the agent state."""
        self._sandbox = sandbox

    def add_tool(self, tool: "Tool") -> None:
        """Dynamically add an eager tool into the current execution context.

        RFC-0005: 直接写入 ToolRegistry，不经过 Executor 间接层。
        Deferred runtime additions are intentionally unsupported for now.
        """
        if tool.defer_loading:
            raise ValueError(
                "Runtime-added deferred tools are not supported. Register deferred tools during agent initialization instead.",
            )
        self._tool_registry.add_source("runtime", [tool])

    def __repr__(self) -> str:
        """String representation of the agent state."""
        return f"AgentState(agent_name='{self.agent_name}', agent_id='{self.agent_id}')"

    def __str__(self) -> str:
        """Human-readable string representation."""
        context_keys = len(self.context.context)
        global_keys = len(self.global_storage.keys())
        return f"AgentState for '{self.agent_name}' with {context_keys} context keys and {global_keys} global keys"
