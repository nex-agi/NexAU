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

"""Agent events middleware that bridges llm_aggregators events with agent events."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime

from nexau.archs.llm.llm_aggregators.events import (
    Event,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolCallResultEvent,
    UsageUpdateEvent,
)
from nexau.archs.main_sub.execution.hooks import (
    AfterAgentHookInput,
    AfterModelHookInput,
    AfterToolHookInput,
    BeforeAgentHookInput,
    HookResult,
    Middleware,
)
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason

logger = logging.getLogger(__name__)


def _noop_event_handler(_: Event) -> None:
    """No-op event handler for non-first choices to avoid duplicate UI updates."""
    return None


class AgentEventsMiddleware(Middleware):
    """Middleware that surfaces aggregator events to the user's callback.

    RFC-0023 §阶段 ③ end state — Set A aggregators now live inside
    ``llm_caller`` (one instance per stream call); they pull this
    middleware's ``on_event`` via ``Middleware.get_event_handler`` and
    drive emission directly. This middleware no longer instantiates or
    drives any aggregator: it only owns the ``on_event`` sink and emits
    lifecycle / tool / usage events around the model call.
    """

    def __init__(
        self,
        *,
        session_id: str,
        on_event: Callable[[Event], None] = _noop_event_handler,
    ):
        """Initialize the AgentEventsMiddleware.

        Args:
            on_event: Callback that receives unified Event objects. The
                Set A aggregators owned by ``llm_caller`` resolve this
                callback via ``get_event_handler`` and call it directly
                as they parse stream chunks; lifecycle hooks below also
                emit through it.
        """
        self.session_id = session_id
        self.on_event = on_event

    def before_agent(self, hook_input: BeforeAgentHookInput) -> HookResult:
        """Hook called before agent execution starts.

        Emits RunStartedEvent to signal the beginning of an agent run.
        For sub-agents, includes parent_run_id to establish hierarchy.

        Args:
            hook_input: Input containing agent state and messages

        Returns:
            HookResult with no changes
        """
        agent_state = hook_input.agent_state

        self.on_event(
            RunStartedEvent(
                thread_id=self.session_id,
                root_run_id=agent_state.root_run_id,
                run_id=agent_state.run_id,
                agent_id=agent_state.agent_id,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
        )
        return HookResult.no_changes()

    def after_agent(self, hook_input: AfterAgentHookInput) -> HookResult:
        """Hook called after agent execution finishes.

        Emits RunFinishedEvent to signal the completion of an agent run.
        If the agent stopped due to an error, emits RunErrorEvent instead.

        Args:
            hook_input: Input containing agent state, messages and response

        Returns:
            HookResult with no changes
        """
        agent_state = hook_input.agent_state

        if hook_input.stop_reason in {
            AgentStopReason.ERROR_OCCURRED,
            AgentStopReason.CONTEXT_TOKEN_LIMIT,
        }:
            self.on_event(
                RunErrorEvent(
                    timestamp=int(datetime.now().timestamp() * 1000),
                    run_id=agent_state.run_id,
                    message=hook_input.agent_response,
                )
            )
        else:
            self.on_event(
                RunFinishedEvent(
                    thread_id=self.session_id,
                    run_id=agent_state.run_id,
                    result=hook_input.agent_response,
                    timestamp=int(datetime.now().timestamp() * 1000),
                )
            )
        return HookResult.no_changes()

    def after_tool(self, hook_input: AfterToolHookInput) -> HookResult:
        """Hook called after tool execution.

        Emits ToolCallResultEvent with the tool execution result.

        Args:
            hook_input: Input containing tool execution information and result

        Returns:
            HookResult with no changes (tool output is preserved)
        """

        # Create tool result content as JSON string
        content = json.dumps(hook_input.tool_output, ensure_ascii=False)

        # Emit ToolCallResultEvent
        self.on_event(
            ToolCallResultEvent(
                tool_call_id=hook_input.tool_call_id,
                content=content,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
        )

        return HookResult.no_changes()

    def after_model(self, hook_input: AfterModelHookInput) -> HookResult:
        """Emit a usage update event after each completed LLM call.

        ``llm_caller`` produces ``ModelResponse.usage`` from the Set A
        aggregator's ``build()`` output (RFC-0023 §阶段 ③). The event we
        emit here mirrors that field; downstream consumers of
        ``UsageUpdateEvent`` see the same numbers Set A would surface
        via ``ModelCallFinishedEvent``.
        """
        if hook_input.model_response is None:
            return HookResult.no_changes()

        self.on_event(
            UsageUpdateEvent(
                run_id=hook_input.agent_state.run_id,
                usage=hook_input.model_response.usage,
                timestamp=int(datetime.now().timestamp() * 1000),
            )
        )
        return HookResult.no_changes()
