# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for AgentEventsMiddleware lifecycle hooks.

RFC-0023 §阶段 ③ — the middleware no longer owns Set A aggregators (they
moved into ``llm_caller``); the only behavior left here is emitting
lifecycle / tool / usage events through ``on_event``. Stream-chunk
routing tests were removed with the aggregator factories.
"""

import json
from unittest.mock import Mock

import pytest
from ag_ui.core.events import RunFinishedEvent, RunStartedEvent

from nexau.archs.llm.llm_aggregators.events import RunErrorEvent, ToolCallResultEvent, UsageUpdateEvent
from nexau.archs.main_sub.execution.hooks import (
    AfterAgentHookInput,
    AfterModelHookInput,
    AfterToolHookInput,
    BeforeAgentHookInput,
)
from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.main_sub.execution.stop_reason import AgentStopReason
from nexau.core.usage import TokenUsage


class TestAgentEventsMiddleware:
    """Lifecycle hook coverage for AgentEventsMiddleware."""

    @pytest.fixture
    def mock_agent_state(self):
        state = Mock()
        state.agent_id = "test_agent_123"
        state.run_id = "test_run_123"
        state.root_run_id = "test_root_run_123"
        state.parent_agent_state = None
        return state

    @pytest.fixture
    def events_captured(self):
        return []

    @pytest.fixture
    def middleware(self, events_captured):
        return AgentEventsMiddleware(session_id="test_session", on_event=events_captured.append)

    def test_initialization(self):
        middleware = AgentEventsMiddleware(session_id="session_123")
        assert middleware.session_id == "session_123"

    def test_before_agent_emits_run_started_event(self, middleware, mock_agent_state, events_captured):
        result = middleware.before_agent(BeforeAgentHookInput(agent_state=mock_agent_state, messages=[]))

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, RunStartedEvent)
        assert event.thread_id == "test_session"
        assert event.run_id == "test_run_123"
        assert not result.has_modifications()

    def test_after_agent_emits_run_finished_event(self, middleware, mock_agent_state, events_captured):
        hook_input = AfterAgentHookInput(
            agent_state=mock_agent_state,
            messages=[],
            agent_response="Task completed successfully",
        )

        result = middleware.after_agent(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, RunFinishedEvent)
        assert event.thread_id == "test_session"
        assert event.run_id == "test_run_123"
        assert event.result == "Task completed successfully"
        assert not result.has_modifications()

    def test_after_agent_emits_run_error_event(self, middleware, mock_agent_state, events_captured):
        hook_input = AfterAgentHookInput(
            agent_state=mock_agent_state,
            messages=[],
            agent_response="Task failed",
            stop_reason=AgentStopReason.ERROR_OCCURRED,
        )

        result = middleware.after_agent(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, RunErrorEvent)
        assert event.run_id == "test_run_123"
        assert event.message == "Task failed"
        assert not result.has_modifications()

    def test_after_tool_emits_tool_call_result_event(self, middleware, mock_agent_state, events_captured):
        hook_input = AfterToolHookInput(
            agent_state=mock_agent_state,
            tool_name="test_tool",
            tool_call_id="call_123",
            tool_input={"param": "value"},
            tool_output={"result": "success", "data": [1, 2, 3]},
            sandbox=Mock(),
        )

        result = middleware.after_tool(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, ToolCallResultEvent)
        assert event.tool_call_id == "call_123"
        assert json.loads(event.content) == {"result": "success", "data": [1, 2, 3]}
        assert not result.has_modifications()

    def test_after_model_emits_usage_update_event(self, middleware, mock_agent_state, events_captured):
        hook_input = AfterModelHookInput(
            agent_state=mock_agent_state,
            messages=[],
            max_iterations=5,
            current_iteration=0,
            original_response="ok",
            model_response=ModelResponse(
                content="ok",
                usage=TokenUsage(input_tokens=10, completion_tokens=5, total_tokens=15),
            ),
        )

        result = middleware.after_model(hook_input)

        assert len(events_captured) == 1
        event = events_captured[0]
        assert isinstance(event, UsageUpdateEvent)
        assert event.run_id == "test_run_123"
        assert event.usage.total_tokens == 15
        assert not result.has_modifications()

    def test_after_model_no_response_does_not_emit(self, middleware, mock_agent_state, events_captured):
        hook_input = AfterModelHookInput(
            agent_state=mock_agent_state,
            messages=[],
            max_iterations=5,
            current_iteration=0,
            original_response=None,
            model_response=None,
        )

        result = middleware.after_model(hook_input)

        assert events_captured == []
        assert not result.has_modifications()
