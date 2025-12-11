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

"""Utilities to attach CLI-specific tracing to sub-agent managers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nexau.archs.main_sub.agent_context import get_context
from nexau.archs.main_sub.execution.hooks import FunctionMiddleware
from nexau.archs.main_sub.execution.subagent_manager import SubAgentManager


class CLIEnabledSubAgentManager(SubAgentManager):
    """Sub-agent manager that emits CLI events and injects hooks."""

    def __init__(
        self,
        agent_name: str,
        sub_agent_factories: dict[str, Callable[..., Any]],
        global_storage=None,
        progress_hook=None,
        tool_hook=None,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__(
            agent_name,
            sub_agent_factories,
            global_storage=global_storage,
        )
        self.cli_progress_hook = progress_hook
        self.cli_tool_hook = tool_hook
        self.event_callback = event_callback

    @classmethod
    def from_existing(
        cls,
        manager: SubAgentManager,
        progress_hook,
        tool_hook,
        event_callback,
    ) -> CLIEnabledSubAgentManager:
        new_manager = cls(
            manager.agent_name,
            manager.sub_agent_factories,
            global_storage=manager.global_storage,
            progress_hook=progress_hook,
            tool_hook=tool_hook,
            event_callback=event_callback,
        )
        # Preserve runtime state
        new_manager.running_sub_agents = manager.running_sub_agents
        if manager._shutdown_event.is_set():
            new_manager._shutdown_event.set()
        return new_manager

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.event_callback:
            return
        try:
            self.event_callback(event_type, payload)
        except Exception:
            # CLI logging isn't critical; swallow errors to avoid disrupting execution
            pass

    def _inject_cli_hooks(self, sub_agent) -> None:
        if getattr(sub_agent, "_cli_hooks_injected", False):
            return

        progress_hook = self.cli_progress_hook
        tool_hook = self.cli_tool_hook

        middleware_manager = getattr(sub_agent.executor, "middleware_manager", None)

        if progress_hook and middleware_manager:
            middleware_manager.middlewares.insert(
                0,
                FunctionMiddleware(
                    after_model_hook=progress_hook,
                    name="cli_progress_hook",
                ),
            )

        if tool_hook and middleware_manager:
            middleware_manager.middlewares.insert(
                0,
                FunctionMiddleware(
                    after_tool_hook=tool_hook,
                    name="cli_tool_hook",
                ),
            )

        # Propagate to nested sub-agent managers
        nested_manager = getattr(sub_agent.executor, "subagent_manager", None)
        if nested_manager:
            cli_nested = attach_cli_manager(
                nested_manager,
                self.cli_progress_hook,
                self.cli_tool_hook,
                self.event_callback,
            )
            if cli_nested is not nested_manager:
                sub_agent.executor.subagent_manager = cli_nested
                if hasattr(sub_agent.executor, "batch_processor") and sub_agent.executor.batch_processor:
                    sub_agent.executor.batch_processor.subagent_manager = cli_nested

        sub_agent._cli_hooks_injected = True

    def call_sub_agent(
        self,
        sub_agent_name: str,
        message: str,
        context: dict[str, Any] | None = None,
        parent_agent_state=None,
    ) -> str:
        if self._shutdown_event.is_set():
            raise RuntimeError(f"Agent '{self.agent_name}' is shutting down")

        if sub_agent_name not in self.sub_agent_factories:
            raise ValueError(f"Sub-agent '{sub_agent_name}' not found")

        sub_agent_factory = self.sub_agent_factories[sub_agent_name]

        sub_agent = None

        if self.global_storage is not None:
            try:
                sub_agent = sub_agent_factory(global_storage=self.global_storage)
            except TypeError:
                sub_agent = sub_agent_factory()
                sub_agent.global_storage = self.global_storage
                if hasattr(sub_agent, "executor"):
                    sub_agent.executor.global_storage = self.global_storage
                    if hasattr(sub_agent.executor, "subagent_manager"):
                        sub_agent.executor.subagent_manager.global_storage = self.global_storage
        else:
            sub_agent = sub_agent_factory()
        self.running_sub_agents[sub_agent.config.agent_id] = sub_agent

        self._inject_cli_hooks(sub_agent)

        parent_agent_id = getattr(parent_agent_state, "agent_id", None) if parent_agent_state else None
        self._emit_event(
            "start",
            {
                "agent_name": sub_agent_name,
                "display_name": getattr(sub_agent.config, "name", sub_agent_name),
                "agent_id": sub_agent.config.agent_id,
                "parent_agent_name": self.agent_name,
                "parent_agent_id": parent_agent_id,
                "message": message,
            },
        )

        try:
            effective_context = context
            if effective_context is None:
                current_context = get_context()
                if current_context:
                    effective_context = current_context.context.copy()

            result = sub_agent.run(
                message,
                context=effective_context,
                parent_agent_state=parent_agent_state,
            )

            self.running_sub_agents.pop(sub_agent.config.agent_id)
            self._emit_event(
                "complete",
                {
                    "agent_name": sub_agent_name,
                    "display_name": getattr(sub_agent.config, "name", sub_agent_name),
                    "agent_id": sub_agent.config.agent_id,
                    "parent_agent_name": self.agent_name,
                    "parent_agent_id": parent_agent_id,
                    "result": result,
                },
            )
            return result

        except Exception as exc:
            agent_id = getattr(sub_agent.config, "agent_id", "") if sub_agent else ""
            display_name = getattr(sub_agent.config, "name", sub_agent_name) if sub_agent else sub_agent_name
            self._emit_event(
                "error",
                {
                    "agent_name": sub_agent_name,
                    "display_name": display_name,
                    "agent_id": agent_id,
                    "parent_agent_name": self.agent_name,
                    "parent_agent_id": parent_agent_id,
                    "error": str(exc),
                },
            )
            raise


def attach_cli_manager(
    manager: SubAgentManager | None,
    progress_hook,
    tool_hook,
    event_callback,
) -> SubAgentManager | None:
    """Wrap an existing manager with CLI reporting capabilities."""
    if manager is None or isinstance(manager, CLIEnabledSubAgentManager):
        return manager

    cli_manager = CLIEnabledSubAgentManager.from_existing(
        manager,
        progress_hook,
        tool_hook,
        event_callback,
    )
    return cli_manager


def attach_cli_to_agent(agent, progress_hook, tool_hook, event_callback) -> None:
    """Ensure an agent and its sub-agents emit CLI traces."""
    if getattr(agent, "_cli_hooks_attached", False):
        return

    middleware_manager = getattr(agent.executor, "middleware_manager", None)

    if progress_hook and middleware_manager:
        middleware_manager.middlewares.insert(
            0,
            FunctionMiddleware(
                after_model_hook=progress_hook,
                name="cli_progress_hook",
            ),
        )

    if tool_hook and middleware_manager:
        middleware_manager.middlewares.insert(
            0,
            FunctionMiddleware(
                after_tool_hook=tool_hook,
                name="cli_tool_hook",
            ),
        )

    cli_manager = attach_cli_manager(
        getattr(agent.executor, "subagent_manager", None),
        progress_hook,
        tool_hook,
        event_callback,
    )
    if cli_manager:
        agent.executor.subagent_manager = cli_manager
        if hasattr(agent.executor, "batch_processor") and agent.executor.batch_processor:
            agent.executor.batch_processor.subagent_manager = cli_manager

    agent._cli_hooks_attached = True
