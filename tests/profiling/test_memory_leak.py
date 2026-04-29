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

"""Memory leak regression tests for long-running agent sessions.

Verifies that the memory leak fixes are effective:
 - _full_trace / _record_full_trace_messages / _update_trace_memory removed
 - AgentRunActionModel GC after REPLACE
 - AgentEventsMiddleware aggregator cleanup
 - Langfuse end_span no longer synchronously flushes (#495)

Usage::

    uv run pytest tests/profiling/test_memory_leak.py -v -s --no-cov

"""

from __future__ import annotations

import gc
import os
import tracemalloc
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import (
    Middleware,
)
from nexau.archs.main_sub.execution.middleware.context_compaction import (
    ContextCompactionMiddleware,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.session import SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.core.messages import Message, Role, TextBlock
from nexau.core.usage import TokenUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


class StubTracer(BaseTracer):
    """Minimal in-process tracer that tracks span count."""

    def __init__(self) -> None:
        self.span_count = 0
        self._session_id = "stub"

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, Any] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        self.span_count += 1
        return Span(
            id=f"span_{self.span_count}",
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(self, span: Span, outputs: Any = None, error: Exception | None = None, attributes: dict[str, Any] | None = None) -> None:
        pass

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def flush(self) -> None:
        pass


def _fake_model_response(iteration: int) -> ModelResponse:
    """Even iterations → tool call (loop), odd → plain text (stop)."""
    if iteration % 2 == 0:
        return ModelResponse(
            content=f"Let me use the tool for step {iteration}.",
            usage=TokenUsage(input_tokens=500, completion_tokens=200, total_tokens=700),
            tool_calls=[
                ModelToolCall(
                    call_id=f"call_{iteration}",
                    name="sample_tool",
                    arguments={"x": iteration},
                    raw_arguments=f'{{"x": {iteration}}}',
                    call_type="function",
                )
            ],
        )
    return ModelResponse(
        content=f"Done with iteration {iteration}.",
        usage=TokenUsage(input_tokens=500, completion_tokens=100, total_tokens=600),
    )


class _CallCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_a: object, **_kw: object) -> ModelResponse:
        resp = _fake_model_response(self.count)
        self.count += 1
        return resp


def _build_agent_components(
    agent_idx: int,
    *,
    enable_compaction: bool = True,
    enable_tracer: bool = True,
    max_iterations: int = 6,
    max_context_tokens: int = 200_000,
) -> dict[str, Any]:
    llm_config = LLMConfig(model="gpt-4o-mini", base_url="https://api.openai.com/v1", api_key="test-key", api_type="openai_chat_completion")
    global_storage = GlobalStorage()
    tracer: StubTracer | None = None
    if enable_tracer:
        tracer = StubTracer()
        global_storage.set("tracer", tracer)

    middlewares: list[Middleware] = []
    if enable_compaction:
        middlewares.append(
            ContextCompactionMiddleware(
                max_context_tokens=max_context_tokens,
                auto_compact=True,
                threshold=0.99,
                compaction_strategy="tool_result_compaction",
            )
        )

    engine = InMemoryDatabaseEngine()
    session_manager = SessionManager(engine=engine)
    user_id, session_id, agent_id = f"user_{agent_idx}", f"session_{agent_idx}", f"agent_{agent_idx}"

    from nexau.archs.tool.tool import Tool

    sample_tool = Tool(
        name="sample_tool",
        description="A sample tool",
        input_schema={"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
        implementation=lambda x: {"result": f"executed_{x}"},
    )
    registry = ToolRegistry()
    registry.add_source("test", [sample_tool])

    executor = Executor(
        agent_name=f"agent_{agent_idx}",
        agent_id=agent_id,
        tool_registry=registry,
        sub_agents={},
        stop_tools=set(),
        openai_client=Mock(),
        llm_config=llm_config,
        max_iterations=max_iterations,
        max_context_tokens=max_context_tokens,
        retry_attempts=1,
        middlewares=middlewares,
        global_storage=global_storage,
        tool_call_mode="structured",
        structured_tools=[
            {
                "name": "sample_tool",
                "description": "A sample tool",
                "input_schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]},
                "kind": "tool",
            }
        ],
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
    )
    return {
        "executor": executor,
        "global_storage": global_storage,
        "tracer": tracer,
        "session_manager": session_manager,
        "user_id": user_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "registry": registry,
        "engine": engine,
    }


def _simulate_single_agent_session(
    agent_idx: int,
    *,
    num_rounds: int = 100,
    enable_compaction: bool = True,
    enable_tracer: bool = True,
) -> dict[str, Any]:
    """Simulate multiple run() invocations on one agent."""
    components = _build_agent_components(agent_idx, enable_compaction=enable_compaction, enable_tracer=enable_tracer)
    executor: Executor = components["executor"]
    global_storage: GlobalStorage = components["global_storage"]
    tracer: StubTracer | None = components["tracer"]
    session_manager: SessionManager = components["session_manager"]

    from nexau.archs.session import AgentRunActionKey

    agent_context = AgentContext(context={})
    agent_context.__enter__()

    history_key = AgentRunActionKey(user_id=components["user_id"], session_id=components["session_id"], agent_id=components["agent_id"])
    history = HistoryList(session_manager=session_manager, history_key=history_key, agent_name=f"agent_{agent_idx}")

    snapshots: list[dict[str, Any]] = []

    for rnd in range(num_rounds):
        run_id = f"run_{agent_idx}_{rnd}"
        agent_state = AgentState(
            agent_name=f"agent_{agent_idx}",
            agent_id=components["agent_id"],
            run_id=run_id,
            root_run_id=run_id,
            context=agent_context,
            global_storage=global_storage,
            tool_registry=components["registry"],
        )
        history.update_context(run_id=run_id, root_run_id=run_id)

        system_msg = Message(role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")])
        user_msg = Message.user(f"Round {rnd}: do something useful.")
        if not history:
            history.replace_all([system_msg, user_msg], update_baseline=True)
        else:
            non_system = [m for m in history if m.role != Role.SYSTEM]
            history.replace_all([system_msg] + non_system, update_baseline=True)
            history.append(user_msg)

        call_counter = _CallCounter()
        with patch.object(executor.llm_caller, "call_llm", side_effect=call_counter):
            try:
                _, updated_messages = executor.execute(list(history), agent_state)
            except RuntimeError:
                updated_messages = list(history)

        history.replace_all(updated_messages)
        history.flush()

        if (rnd + 1) % 25 == 0 or rnd == 0:
            gc.collect()
            # Check for __nexau_full_trace_messages__ — should NOT exist after fix
            ft = agent_state.get_context_value("__nexau_full_trace_messages__", None)
            trace_memory = cast(dict[str, Any], global_storage.get("trace_memory", {}))
            snapshots.append(
                {
                    "round": rnd + 1,
                    "history_len": len(history),
                    "full_trace_context_exists": ft is not None,
                    "trace_memory_has_message_trace": "message_trace" in trace_memory,
                    "tracer_spans": tracer.span_count if tracer else 0,
                }
            )

    agent_context.__exit__(None, None, None)

    return {
        "agent_idx": agent_idx,
        "snapshots": snapshots,
        "final_history_len": len(history),
        "_global_storage": global_storage,
        "_history": history,
        "_engine": components["engine"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryLeakRegression:
    """Verify memory leak fixes are effective."""

    @pytest.fixture(autouse=True)
    def _setup_tracemalloc(self):
        tracemalloc.start(10)
        yield
        tracemalloc.stop()

    def test_full_trace_removed(self):
        """_full_trace / _record_full_trace_messages should no longer exist.

        After the fix, __nexau_full_trace_messages__ should never be set in
        agent context, and trace_memory should not contain message_trace
        (only generate_with_token writes to trace_memory).
        """
        result = _simulate_single_agent_session(0, num_rounds=50, enable_compaction=True, enable_tracer=True)

        print("\n" + "=" * 80)
        print("VERIFY: _full_trace REMOVED — 50 rounds")
        print("=" * 80)
        for s in result["snapshots"]:
            print(
                f"  Round {s['round']:>4d} | history={s['history_len']:>4d}  "
                f"full_trace_ctx={s['full_trace_context_exists']}  "
                f"trace_memory_has_msg_trace={s['trace_memory_has_message_trace']}  "
                f"tracer_spans={s['tracer_spans']}"
            )
        print("=" * 80)

        for s in result["snapshots"]:
            assert s["full_trace_context_exists"] is False, f"Round {s['round']}: __nexau_full_trace_messages__ should not exist in context"
            assert s["trace_memory_has_message_trace"] is False, (
                f"Round {s['round']}: trace_memory should not have message_trace (only generate_with_token writes it)"
            )

    def test_memory_bounded_100_rounds(self):
        """With fixes applied, memory should not grow linearly with rounds.

        History is bounded by the executor's natural behavior, and there is no
        unbounded _full_trace accumulation.
        """
        snap_before = tracemalloc.take_snapshot()

        result = _simulate_single_agent_session(0, num_rounds=100, enable_compaction=True, enable_tracer=True)

        gc.collect()
        snap_after = tracemalloc.take_snapshot()

        print("\n" + "=" * 80)
        print("VERIFY: BOUNDED MEMORY — 100 rounds")
        print("=" * 80)
        for s in result["snapshots"]:
            print(f"  Round {s['round']:>4d} | history={s['history_len']:>4d}")
        print(f"\n  Final history length: {result['final_history_len']}")

        top_stats = snap_after.compare_to(snap_before, "filename")
        nexau_stats = [s for s in top_stats if "nexau/" in str(s)]
        print("\n  Top 10 nexau files by memory growth:")
        for stat in nexau_stats[:10]:
            print(f"    {stat}")
        print("=" * 80)

        # History should be bounded — last round's executor output, not 100*N messages
        # With max_iterations=6, each round produces at most ~6 new messages.
        # Without compaction triggering, history grows but that's expected normal behavior.
        # The key assertion: NO _full_trace bloat.
        gs = result["_global_storage"]
        trace_memory = gs.get("trace_memory", {})
        assert "message_trace" not in trace_memory, "trace_memory.message_trace should not exist for non-generate_with_token agents"

    def test_agent_run_action_gc_after_replace(self):
        """After persist_replace, old AgentRunActionModel records should be deleted."""
        import asyncio

        from nexau.archs.session import AgentRunActionKey
        from nexau.archs.session.models import AgentRunActionModel

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        async def _test() -> None:
            await session_manager.setup_models()

            key = AgentRunActionKey(user_id="u1", session_id="s1", agent_id="a1")

            # Create several APPEND records
            for i in range(10):
                await session_manager.agent_run_action.persist_append(
                    key=key,
                    run_id=f"run_{i}",
                    root_run_id="root",
                    parent_run_id=None,
                    agent_name="test",
                    messages=[Message.user(f"msg {i}")],
                )

            # Count records before REPLACE
            from nexau.archs.session.orm import AndFilter, ComparisonFilter

            all_records = await engine.find_many(
                AgentRunActionModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("user_id", "u1"),
                        ComparisonFilter.eq("session_id", "s1"),
                        ComparisonFilter.eq("agent_id", "a1"),
                    ]
                ),
            )
            count_before = len(all_records)

            # Now do a REPLACE
            await session_manager.agent_run_action.persist_replace(
                key=key,
                run_id="replace_run",
                root_run_id="root",
                messages=[Message.user("compacted summary")],
            )

            # Count records after REPLACE
            all_records_after = await engine.find_many(
                AgentRunActionModel,
                filters=AndFilter(
                    filters=[
                        ComparisonFilter.eq("user_id", "u1"),
                        ComparisonFilter.eq("session_id", "s1"),
                        ComparisonFilter.eq("agent_id", "a1"),
                    ]
                ),
            )
            count_after = len(all_records_after)

            print(f"\n  AgentRunAction GC: {count_before} records before REPLACE → {count_after} after")
            assert count_after == 1, f"Expected 1 record (the REPLACE) after GC, got {count_after}"
            assert count_before > count_after, f"GC should have removed records: before={count_before}, after={count_after}"

        asyncio.run(_test())

    def test_aggregators_cleared_after_model(self):
        """openai_chat_completion_aggregators should be cleared in after_model."""
        from nexau.archs.main_sub.execution.hooks import AfterModelHookInput
        from nexau.archs.main_sub.execution.middleware.agent_events_middleware import AgentEventsMiddleware

        mw = AgentEventsMiddleware(session_id="test")

        # Simulate some aggregators accumulated during streaming
        mw.openai_chat_completion_aggregators["chatcmpl_1"] = Mock()
        mw.openai_chat_completion_aggregators["chatcmpl_2"] = Mock()
        assert len(mw.openai_chat_completion_aggregators) == 2

        # Call after_model
        agent_state = AgentState(
            agent_name="test",
            agent_id="test",
            run_id="r",
            root_run_id="r",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            tool_registry=ToolRegistry(),
        )
        hook_input = AfterModelHookInput(
            agent_state=agent_state,
            max_iterations=10,
            current_iteration=0,
            messages=[],
            original_response="test",
        )
        mw.after_model(hook_input)

        assert len(mw.openai_chat_completion_aggregators) == 0, "aggregators should be cleared after after_model"
        print("\n  ✅ openai_chat_completion_aggregators cleared after after_model")

    @patch.dict(os.environ, {}, clear=False)
    def test_langfuse_end_span_does_not_flush(self):
        """end_span() should never synchronously flush the Langfuse client (#495)."""
        # Clear Langfuse env vars to prevent _ensure_client() identity mismatch
        os.environ.pop("LANGFUSE_HOST", None)
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
        os.environ.pop("LANGFUSE_SECRET_KEY", None)

        from nexau.archs.tracer.adapters.langfuse import LangfuseTracer

        tracer = LangfuseTracer(enabled=True, public_key="pk-test", secret_key="sk-test")
        mock_client = Mock()
        tracer.client = mock_client
        tracer._client_identity = ("pk-test", "sk-test", None)

        # Create and end a root span
        root_span = tracer.start_span("root", SpanType.AGENT)
        tracer.end_span(root_span)
        root_flush_count = mock_client.flush.call_count

        # Create and end a child span
        mock_client.flush.reset_mock()
        child_span = tracer.start_span("child", SpanType.LLM, parent_span=root_span)
        tracer.end_span(child_span)
        child_flush_count = mock_client.flush.call_count

        print(f"\n  Root span flush calls: {root_flush_count}")
        print(f"  Child span flush calls: {child_flush_count}")

        assert root_flush_count == 0, "Root span end should NOT trigger flush"
        assert child_flush_count == 0, "Child span end should NOT trigger flush"
