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

"""Production-realistic memory-leak stress tests.

Validates memory-leak fixes with realistic message sizes (~11 KB/round):
  - user prompt ~500 B, assistant ~3-4 KB, tool output ~2.5 KB

Earlier testing discovered that ``ContextCompactionMiddleware`` silently
ignores the ``trigger_threshold`` kwarg (the correct field is ``threshold``
on ``CompactionConfig``).  When the compaction LLM has no valid API key,
the ``_hard_truncation_fallback`` path inside ``SlidingWindowCompaction``
accumulates ~30 KB per compaction call — causing 58 MiB growth over 100
rounds.  The tests below use ``tool_result_compaction`` strategy with
``threshold=0.99`` (the *correct* config field) to avoid this issue and
isolate the _full_trace / aggregator / GC fixes being verified.

Each test is designed to complete well within the CI ``--timeout=120``.
"""

from __future__ import annotations

import gc
import json
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import cast
from unittest.mock import Mock, patch

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.execution.executor import Executor
from nexau.archs.main_sub.execution.hooks import Middleware
from nexau.archs.main_sub.execution.middleware.context_compaction import (
    ContextCompactionMiddleware,
)
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.history_list import HistoryList
from nexau.archs.session import AgentRunActionKey, SessionManager
from nexau.archs.session.orm import InMemoryDatabaseEngine
from nexau.archs.tool.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.core.messages import Message, Role, TextBlock
from nexau.core.usage import TokenUsage

# ---------------------------------------------------------------------------
# Production-realistic message templates (~11 KB / round total)
# ---------------------------------------------------------------------------

_REASONING_BLOCK = (
    "Let me analyze this step by step. First, I need to understand the current "
    "state of the system and identify any potential issues. Based on the context "
    "provided, there are several factors to consider:\n\n"
    "1. The data pipeline processes approximately 10,000 records per batch, with "
    "each record containing nested JSON structures averaging 2KB in size.\n"
    "2. Memory consumption patterns suggest that intermediate buffers are not "
    "being released properly during the transformation phase.\n"
    "3. The connection pool maintains up to 20 active connections, each with its "
    "own query cache that can grow to 50MB under heavy load.\n\n"
    "Given these observations, I recommend examining the buffer management logic "
    "in the ETL pipeline and implementing explicit cleanup hooks after each batch "
    "completes processing. Additionally, we should consider implementing a "
    "sliding window approach for the query cache to prevent unbounded growth.\n\n"
    "```python\n"
    "class BatchProcessor:\n"
    '    """Processes data batches with managed memory lifecycle."""\n\n'
    "    def __init__(self, pool_size: int = 20, cache_max_mb: int = 50):\n"
    "        self.pool = ConnectionPool(max_size=pool_size)\n"
    "        self.cache = LRUCache(max_mb=cache_max_mb)\n"
    "        self._buffer: list[Record] = []\n"
    "        self._metrics = MetricsCollector()\n\n"
    "    def process_batch(self, records: list[Record]) -> BatchResult:\n"
    '        """Process a batch of records with automatic resource cleanup."""\n'
    "        self._metrics.start_batch(len(records))\n"
    "        try:\n"
    "            transformed = self._transform(records)\n"
    "            self._metrics.record_phase('transform', len(transformed))\n"
    "            validated = self._validate(transformed)\n"
    "            self._metrics.record_phase('validate', len(validated))\n"
    "            result = self._load(validated)\n"
    "            self._metrics.record_phase('load', result.count)\n"
    "            return result\n"
    "        except TransformError as e:\n"
    "            self._metrics.record_error('transform', str(e))\n"
    "            raise\n"
    "        finally:\n"
    "            self._buffer.clear()\n"
    "            self.cache.evict_stale()\n"
    "            self._metrics.end_batch()\n\n"
    "    def _transform(self, records: list[Record]) -> list[TransformedRecord]:\n"
    "        results = []\n"
    "        for record in records:\n"
    "            conn = self.pool.acquire()\n"
    "            try:\n"
    "                enriched = conn.enrich(record)\n"
    "                normalized = self._normalize(enriched)\n"
    "                results.append(normalized)\n"
    "            finally:\n"
    "                self.pool.release(conn)\n"
    "        return results\n"
    "```\n\n"
    "The key improvements in this implementation include:\n"
    "- **Explicit buffer cleanup** in the `finally` block prevents memory accumulation\n"
    "- **Cache eviction** after each batch keeps memory usage predictable\n"
    "- **Metrics collection** provides visibility into processing performance\n"
    "- **Connection pool management** with proper acquire/release prevents leaks\n\n"
    "Furthermore, I recommend setting up monitoring alerts for the following thresholds:\n"
    "- RSS memory exceeding 2GB per worker process\n"
    "- Cache hit ratio dropping below 70% (indicates cache thrashing)\n"
    "- Batch processing time exceeding 5 minutes (indicates resource contention)\n"
    "- Connection pool utilization above 90% for more than 2 consecutive minutes\n\n"
)  # ~3.3 KB

_TOOL_OUTPUT_TEMPLATE = (
    '{{"status": "success", "execution_id": "exec_{iter}", '
    '"result": {{"analysis": {{"total_records": 10342, "processed": 10342, '
    '"failed": 0, "skipped": 0, "duplicates_detected": 23}}, '
    '"metrics": {{"avg_latency_ms": 23.4, "p50_latency_ms": 18.2, '
    '"p95_latency_ms": 89.7, "p99_latency_ms": 156.2, '
    '"throughput_rps": 4521.7, "memory_peak_mb": 342.1, '
    '"memory_avg_mb": 218.6, "cpu_utilization_pct": 78.3, '
    '"gc_collections": 12, "gc_collected_objects": 45892, '
    '"cache_hit_ratio": 0.873, "pool_utilization": 0.65}}, '
    '"details": ['
    '{{"record_id": "rec_001", "status": "ok", "transform_ms": 12, "size_bytes": 2048}},'
    '{{"record_id": "rec_002", "status": "ok", "transform_ms": 15, "size_bytes": 1856}},'
    '{{"record_id": "rec_003", "status": "ok", "transform_ms": 9, "size_bytes": 2234}},'
    '{{"record_id": "rec_004", "status": "ok", "transform_ms": 11, "size_bytes": 1920}},'
    '{{"record_id": "rec_005", "status": "ok", "transform_ms": 14, "size_bytes": 2112}}'
    "], "
    '"summary": "Batch processing completed successfully. All 10342 records '
    "were transformed and loaded. Cache hit ratio 87.3%%. GC performed 12 "
    "collections reclaiming 45892 objects. Pool utilization 65%%.\"}}}}'"
)  # ~1.5 KB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _realistic_user_prompt(rnd: int) -> str:
    """~500 byte user prompt."""
    return (
        f"Round {rnd}: I'm working on optimizing our data pipeline that processes "
        f"customer analytics events. The current implementation uses batch sizes "
        f"of 10,000 records but we're seeing memory spikes during peak hours "
        f"(around 2-4 PM UTC). Can you analyze the pipeline configuration and "
        f"suggest improvements? Here's the relevant config snippet:\n"
        f"  batch_size: 10000\n"
        f"  max_retries: 3\n"
        f"  timeout_sec: 300\n"
        f"  buffer_mode: streaming\n"
        f"  compression: gzip\n"
    )


def _fake_model_response(iteration: int) -> ModelResponse:
    """Even → tool call (~4 KB), odd → plain text (~3 KB)."""
    if iteration % 2 == 0:
        content = (
            f"I'll analyze the pipeline metrics for iteration {iteration}. "
            + _REASONING_BLOCK
            + "Let me run the diagnostic tool to get current metrics.\n"
        )
        tool_args = {
            "x": iteration,
            "config": {"batch_size": 10000, "analyze_memory": True},
        }
        return ModelResponse(
            content=content,
            usage=TokenUsage(input_tokens=2000, completion_tokens=800, total_tokens=2800),
            tool_calls=[
                ModelToolCall(
                    call_id=f"call_{iteration}",
                    name="sample_tool",
                    arguments=tool_args,
                    raw_arguments=json.dumps(tool_args),
                    call_type="function",
                )
            ],
        )
    content = (
        f"Based on the diagnostic results from iteration {iteration}, "
        f"here's my analysis:\n\n"
        + _REASONING_BLOCK
        + "In conclusion, the pipeline is operating within acceptable parameters "
        + "but there are opportunities for optimization.\n"
    )
    return ModelResponse(
        content=content,
        usage=TokenUsage(input_tokens=2000, completion_tokens=600, total_tokens=2600),
    )


class _CallCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_a: object, **_kw: object) -> ModelResponse:
        resp = _fake_model_response(self.count)
        self.count += 1
        return resp


class _StubTracer(BaseTracer):
    """Minimal tracer — counts spans, retains no data."""

    def __init__(self) -> None:
        self.span_count = 0
        self._session_id = "stub"
        self._lock = threading.Lock()

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict[str, object] | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, object] | None = None,
    ) -> Span:
        with self._lock:
            self.span_count += 1
            count = self.span_count
        return Span(
            id=f"span_{count}",
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(
        self,
        span: Span,
        outputs: object = None,
        error: Exception | None = None,
        attributes: dict[str, object] | None = None,
    ) -> None:
        pass

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def flush(self) -> None:
        pass


def _build(agent_idx: int) -> dict[str, object]:
    """Build isolated per-agent components."""
    llm_config = LLMConfig(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        api_type="openai_chat_completion",
    )
    gs = GlobalStorage()
    gs.set("tracer", _StubTracer())

    # NOTE: use ``threshold`` (not ``trigger_threshold``!) — CompactionConfig
    # silently ignores unknown kwargs, falling back to the 0.75 default.
    middlewares: list[Middleware] = [
        ContextCompactionMiddleware(
            max_context_tokens=200_000,
            auto_compact=True,
            threshold=0.99,
            compaction_strategy="tool_result_compaction",
        )
    ]

    engine = InMemoryDatabaseEngine()
    sm = SessionManager(engine=engine)
    uid = f"user_{agent_idx}"
    sid = f"session_{agent_idx}"
    aid = f"agent_{agent_idx}"

    tool = Tool(
        name="sample_tool",
        description="diagnostic tool",
        input_schema={
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "config": {"type": "object"},
            },
            "required": ["x"],
        },
        implementation=lambda x, **kw: json.loads(_TOOL_OUTPUT_TEMPLATE.format(iter=x)),
    )
    reg = ToolRegistry()
    reg.add_source("test", [tool])

    executor = Executor(
        agent_name=f"agent_{agent_idx}",
        agent_id=aid,
        tool_registry=reg,
        sub_agents={},
        stop_tools=set(),
        openai_client=Mock(),
        llm_config=llm_config,
        max_iterations=6,
        max_context_tokens=200_000,
        retry_attempts=1,
        middlewares=middlewares,
        global_storage=gs,
        tool_call_mode="structured",
        structured_tools=[
            {
                "name": "sample_tool",
                "description": "diagnostic tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "config": {"type": "object"}},
                    "required": ["x"],
                },
                "kind": "tool",
            }
        ],
        session_manager=sm,
        user_id=uid,
        session_id=sid,
    )
    return {
        "executor": executor,
        "global_storage": gs,
        "session_manager": sm,
        "user_id": uid,
        "session_id": sid,
        "agent_id": aid,
        "registry": reg,
    }


def _run_rounds(
    agent_idx: int,
    num_rounds: int,
    *,
    sample_interval: int = 10,
) -> dict[str, object]:
    """Run *num_rounds* of conversation, returning checkpoint data."""
    comp = _build(agent_idx)
    executor = cast(Executor, comp["executor"])
    gs = cast(GlobalStorage, comp["global_storage"])
    sm = cast(SessionManager, comp["session_manager"])
    uid = cast(str, comp["user_id"])
    sid = cast(str, comp["session_id"])
    aid = cast(str, comp["agent_id"])
    reg = cast(ToolRegistry, comp["registry"])

    ctx = AgentContext(context={})
    ctx.__enter__()

    hk = AgentRunActionKey(user_id=uid, session_id=sid, agent_id=aid)
    history = HistoryList(session_manager=sm, history_key=hk, agent_name=f"agent_{agent_idx}")

    checkpoints: list[dict[str, object]] = []
    leak = False
    t0 = time.monotonic()

    for rnd in range(num_rounds):
        run_id = f"run_{agent_idx}_{rnd}"
        state = AgentState(
            agent_name=f"agent_{agent_idx}",
            agent_id=aid,
            run_id=run_id,
            root_run_id=run_id,
            context=ctx,
            global_storage=gs,
            tool_registry=reg,
        )
        history.update_context(run_id=run_id, root_run_id=run_id)

        sys_msg = Message(role=Role.SYSTEM, content=[TextBlock(text="You are helpful.")])
        usr_msg = Message.user(_realistic_user_prompt(rnd))
        if not history:
            history.replace_all([sys_msg, usr_msg], update_baseline=True)
        else:
            non_sys = [m for m in history if m.role != Role.SYSTEM]
            history.replace_all([sys_msg] + non_sys, update_baseline=True)
            history.append(usr_msg)

        counter = _CallCounter()
        with patch.object(executor.llm_caller, "call_llm", side_effect=counter):
            try:
                _, updated = executor.execute(list(history), state)
            except RuntimeError:
                updated = list(history)

        history.replace_all(updated)
        history.flush()

        if (rnd + 1) % sample_interval == 0 or rnd == 0:
            cur, _ = tracemalloc.get_traced_memory()
            ft = state.get_context_value("__nexau_full_trace_messages__", None)
            tm = cast(dict[str, object], gs.get("trace_memory", {}))
            cp: dict[str, object] = {
                "round": rnd + 1,
                "history_len": len(history),
                "traced_mb": cur / (1024 * 1024),
                "full_trace_exists": ft is not None,
                "trace_memory_has_msg_trace": "message_trace" in tm,
            }
            checkpoints.append(cp)
            if cp["full_trace_exists"] or cp["trace_memory_has_msg_trace"]:
                leak = True

    ctx.__exit__(None, None, None)
    return {
        "agent_idx": agent_idx,
        "num_rounds": num_rounds,
        "final_history_len": len(history),
        "checkpoints": checkpoints,
        "leak_detected": leak,
        "elapsed_sec": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# Tests — each must complete within CI --timeout=120
# ---------------------------------------------------------------------------


class TestRealisticMemoryStress:
    """Memory-leak regression with production-realistic (~11 KB/round) data."""

    @pytest.fixture(autouse=True)
    def _gc_baseline(self) -> None:  # type: ignore[misc]
        gc.collect()
        gc.collect()
        yield  # type: ignore[misc]
        gc.collect()

    def test_50_rounds_memory_bounded(self) -> None:
        """Single agent × 50 rounds (~550 KB message data).

        Verifies:
          1. ``__nexau_full_trace_messages__`` never appears
          2. ``trace_memory.message_trace`` never appears
          3. Traced-memory growth < 30 MB
          4. 2nd-half growth ≤ 2.5× 1st-half (no O(n²))
          5. No single nexau module grows > 20 MB
        """
        tracemalloc.start(5)
        gc.collect()
        snap_before = tracemalloc.take_snapshot()
        mem_before, _ = tracemalloc.get_traced_memory()

        result = _run_rounds(agent_idx=0, num_rounds=50, sample_interval=10)

        gc.collect()
        gc.collect()
        mem_after, _ = tracemalloc.get_traced_memory()
        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        growth_mb = (mem_after - mem_before) / (1024 * 1024)
        cps = cast(list[dict[str, object]], result["checkpoints"])

        # 1-2. No leak vectors
        for cp in cps:
            assert not cp["full_trace_exists"], f"Round {cp['round']}: __nexau_full_trace_messages__!"
            assert not cp["trace_memory_has_msg_trace"], f"Round {cp['round']}: trace_memory.message_trace!"

        # 3. Growth bounded
        assert growth_mb < 30.0, f"Traced memory grew {growth_mb:.1f} MB (limit 30 MB)"

        # 4. Sublinear growth
        if len(cps) >= 4:
            mid = len(cps) // 2
            g1 = cast(float, cps[mid]["traced_mb"]) - cast(float, cps[0]["traced_mb"])
            g2 = cast(float, cps[-1]["traced_mb"]) - cast(float, cps[mid]["traced_mb"])
            if g1 > 0.5:
                assert g2 / g1 < 2.5, f"Super-linear: 2nd half +{g2:.2f} MB vs 1st +{g1:.2f} MB"

        # 5. No single nexau module > 20 MB
        for stat in snap_after.compare_to(snap_before, "filename"):
            if "nexau/" in str(stat) and stat.size_diff > 0:
                assert stat.size_diff / (1024 * 1024) < 20.0, f"nexau module grew too much: {stat}"

    def test_concurrent_isolation(self) -> None:
        """3 concurrent agents × 20 rounds — no cross-session leak."""
        tracemalloc.start(5)
        gc.collect()
        mem_before, _ = tracemalloc.get_traced_memory()

        results: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=3) as pool:
            futs = {pool.submit(_run_rounds, i, 20, sample_interval=10): i for i in range(3)}
            for f in as_completed(futs):
                results.append(f.result())

        gc.collect()
        gc.collect()
        mem_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        growth_mb = (mem_after - mem_before) / (1024 * 1024)

        for r in results:
            assert not cast(bool, r["leak_detected"]), f"Agent {r['agent_idx']} has leak!"

        assert growth_mb < 30.0, f"Aggregate growth {growth_mb:.1f} MB > 30 MB"

    def test_per_round_delta_stable(self) -> None:
        """50 rounds sampled per 5 — per-round cost must not grow."""
        tracemalloc.start(5)
        gc.collect()

        result = _run_rounds(agent_idx=0, num_rounds=50, sample_interval=5)
        tracemalloc.stop()

        cps = cast(list[dict[str, object]], result["checkpoints"])
        assert len(cps) >= 8

        deltas = [cast(float, cps[i]["traced_mb"]) - cast(float, cps[i - 1]["traced_mb"]) for i in range(1, len(cps))]
        q = len(deltas) // 4
        avg_first = sum(deltas[:q]) / q
        avg_last = sum(deltas[-q:]) / q

        if avg_first > 0.001:
            ratio = avg_last / avg_first
            assert ratio < 3.0, (
                f"Per-round delta growing: last-q avg {avg_last:.4f} MB vs first-q avg {avg_first:.4f} MB (ratio {ratio:.1f}×)"
            )
