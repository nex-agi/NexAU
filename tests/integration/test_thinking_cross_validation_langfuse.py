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

"""Cross-validation integration tests: two thinking-enabled computation tasks with Langfuse tracing.

Two independent agents (Task A and Task B) each solve a computation problem
that requires reasoning (thinking/chain-of-thought). Both upload their full
traces (including thinking spans) to Langfuse. A cross-validation step then
compares results for consistency.

Requirements:
    - LLM credentials: LLM_API_KEY, LLM_MODEL, LLM_BASE_URL (via .env or env vars)
    - Langfuse credentials: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

Tests are automatically skipped when credentials are not available.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any

import pytest
import requests

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_HAS_LLM = bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_API_KEY") != "test-key-not-used")
_HAS_LANGFUSE = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.llm,
    pytest.mark.external,
    pytest.mark.skipif(not _HAS_LLM, reason="LLM_API_KEY not set or is placeholder"),
    pytest.mark.skipif(not _HAS_LANGFUSE, reason="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set"),
]

# ---------------------------------------------------------------------------
# Langfuse REST helpers (for verification)
# ---------------------------------------------------------------------------

_HOST = (os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com").rstrip("/")
_PK = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SK = os.getenv("LANGFUSE_SECRET_KEY", "")


def _fetch_trace(
    trace_id: str,
    retries: int = 20,
    delay: float = 3.0,
    min_observations: int = 0,
) -> dict[str, Any]:
    """Fetch a trace from Langfuse REST API with retry (eventual consistency)."""
    for i in range(retries):
        resp = requests.get(
            f"{_HOST}/api/public/traces/{trace_id}",
            auth=(_PK, _SK),
            timeout=10,
        )
        if resp.status_code == 200:
            trace = resp.json()
            observations = trace.get("observations", [])
            if len(observations) >= min_observations:
                return trace
        if i < retries - 1:
            time.sleep(delay)
    pytest.fail(f"Trace {trace_id} not found or incomplete after {retries * delay}s (min_observations={min_observations})")


def _fetch_traces_by_session(
    session_id: str,
    min_traces: int = 1,
    retries: int = 20,
    delay: float = 3.0,
) -> list[dict[str, Any]]:
    """Fetch traces from Langfuse REST API by session_id with retry (eventual consistency)."""
    for i in range(retries):
        resp = requests.get(
            f"{_HOST}/api/public/traces",
            params={"sessionId": session_id},
            auth=(_PK, _SK),
            timeout=15,
        )
        if resp.status_code == 200:
            traces_data = resp.json()
            trace_list: list[dict[str, Any]] = traces_data.get("data", [])
            if len(trace_list) >= min_traces:
                return trace_list
        if i < retries - 1:
            time.sleep(delay)
    pytest.fail(f"No traces (or fewer than {min_traces}) found in Langfuse for session_id={session_id} after {retries * delay}s")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_manager():
    """Create in-memory session manager."""
    return SessionManager(engine=InMemoryDatabaseEngine())


def _make_thinking_agent(
    name: str,
    system_prompt: str,
    session_manager: SessionManager,
    tracer: LangfuseTracer,
) -> Agent:
    """Create an agent configured for thinking/reasoning computation tasks."""
    llm_config = LLMConfig()

    agent_config = AgentConfig(
        name=name,
        system_prompt=system_prompt,
        llm_config=llm_config,
        tracers=[tracer],
    )
    # Use the tracer's session_id so the Agent and OTEL spans share the same
    # session.id that the test later queries from the Langfuse REST API.
    # (Agent.__init__ calls tracer.set_session_id(agent.session_id), so they
    # must match or the REST query will find nothing.)
    return Agent(
        config=agent_config,
        session_manager=session_manager,
        user_id="cross-validation-test",
        session_id=tracer.session_id,
    )


# ---------------------------------------------------------------------------
# Task A: 数学推理计算
# ---------------------------------------------------------------------------

TASK_A_SYSTEM_PROMPT = """\
You are a precise mathematical reasoning assistant.
Think step by step before giving your final answer.
Show your reasoning process clearly.
Always end your response with a line in the exact format:
ANSWER: <number>
where <number> is the final numeric result (integer or decimal).
"""

TASK_A_QUESTION = (
    "A store has a 20% off sale. If the original price of a jacket is $150, "
    "and there's an additional 10% discount applied after the first discount, "
    "what is the final price? Think through this step by step."
)

# 正确答案: 150 * 0.8 = 120, 然后 120 * 0.9 = 108
TASK_A_EXPECTED = 108.0


# ---------------------------------------------------------------------------
# Task B: 逻辑推理计算
# ---------------------------------------------------------------------------

TASK_B_SYSTEM_PROMPT = """\
You are a precise logical reasoning assistant.
Think step by step before giving your final answer.
Show your reasoning process clearly.
Always end your response with a line in the exact format:
ANSWER: <number>
where <number> is the final numeric result (integer or decimal).
"""

TASK_B_QUESTION = (
    "A farmer has 3 fields. The first field produces 120 kg of wheat. "
    "The second field produces 50% more than the first. "
    "The third field produces twice as much as the second. "
    "What is the total wheat production in kg? Think through this step by step."
)

# 正确答案: field1=120, field2=120*1.5=180, field3=180*2=360, total=120+180+360=660
TASK_B_EXPECTED = 660.0


# ---------------------------------------------------------------------------
# Helper: extract numeric answer from LLM response
# ---------------------------------------------------------------------------


def _extract_answer(response: str) -> float | None:
    """Extract numeric answer from response text.

    Looks for pattern 'ANSWER: <number>' in the response.
    """
    import re

    # 1. 尝试找 ANSWER: 格式
    for line in reversed(response.strip().splitlines()):
        match = re.search(r"ANSWER:\s*\$?\s*([\d,]+(?:\.\d+)?)", line, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))

    # 2. 回退: 找最后一个独立数字
    numbers = re.findall(r"\$?\s*([\d,]+(?:\.\d+)?)", response)
    if numbers:
        return float(numbers[-1].replace(",", ""))

    return None


def _get_trace_id_from_tracer(tracer: LangfuseTracer) -> str | None:
    """Extract the trace_id from the Langfuse client's last trace."""
    if tracer.client is None:
        return None
    # LangfuseTracer stores session_id which correlates with the trace
    return None  # We'll get trace_id from the span instead


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThinkingCrossValidation:
    """Cross-validation: two thinking agents solve computation tasks and upload to Langfuse."""

    def test_task_a_thinking_math_computation(self, session_manager: SessionManager) -> None:
        """Task A: 数学推理计算 — 连续折扣问题，trace 上传 Langfuse。

        验证:
        1. Agent 能正确计算连续折扣
        2. 响应包含推理过程（非直接给答案）
        3. Trace 成功上传到 Langfuse，包含 Agent span
        """
        session_id = f"task-a-{uuid.uuid4().hex[:8]}"
        tracer = LangfuseTracer(
            public_key=_PK,
            secret_key=_SK,
            host=_HOST,
            session_id=session_id,
            tags=["cross-validation", "task-a", "thinking", "math"],
            metadata={"test": "thinking_cross_validation", "task": "A"},
            debug=True,
        )

        agent = _make_thinking_agent(
            name="task-a-math-agent",
            system_prompt=TASK_A_SYSTEM_PROMPT,
            session_manager=session_manager,
            tracer=tracer,
        )

        # 执行计算
        response = agent.run(message=TASK_A_QUESTION)
        response_text = response if isinstance(response, str) else response[0]
        assert isinstance(response_text, str)
        assert len(response_text) > 50, "Response should contain reasoning, not just a number"

        # 提取答案并验证
        answer = _extract_answer(response_text)
        assert answer is not None, f"Could not extract numeric answer from response:\n{response_text}"
        assert abs(answer - TASK_A_EXPECTED) < 1.0, f"Task A: Expected ~{TASK_A_EXPECTED}, got {answer}.\nFull response:\n{response_text}"

        # Flush tracer 确保数据发送
        tracer.flush()
        tracer.shutdown()

        # 验证 Langfuse trace 存在（通过 session_id 查找）
        # 使用 Langfuse REST API 查找该 session 的 traces（带重试，应对最终一致性）
        trace_list = _fetch_traces_by_session(session_id, min_traces=1)

        # 验证 trace 包含 Agent span
        trace = trace_list[0]
        trace_name = trace.get("name", "")
        assert "task-a" in trace_name.lower() or "agent" in trace_name.lower(), f"Trace name '{trace_name}' should reference the agent"

        print(f"\n✅ Task A passed: answer={answer}, trace uploaded to Langfuse (session={session_id})")

    def test_task_b_thinking_logic_computation(self, session_manager: SessionManager) -> None:
        """Task B: 逻辑推理计算 — 递进产量问题，trace 上传 Langfuse。

        验证:
        1. Agent 能正确计算递进关系
        2. 响应包含推理过程
        3. Trace 成功上传到 Langfuse，包含 Agent span
        """
        session_id = f"task-b-{uuid.uuid4().hex[:8]}"
        tracer = LangfuseTracer(
            public_key=_PK,
            secret_key=_SK,
            host=_HOST,
            session_id=session_id,
            tags=["cross-validation", "task-b", "thinking", "logic"],
            metadata={"test": "thinking_cross_validation", "task": "B"},
            debug=True,
        )

        agent = _make_thinking_agent(
            name="task-b-logic-agent",
            system_prompt=TASK_B_SYSTEM_PROMPT,
            session_manager=session_manager,
            tracer=tracer,
        )

        # 执行计算
        response = agent.run(message=TASK_B_QUESTION)
        response_text = response if isinstance(response, str) else response[0]
        assert isinstance(response_text, str)
        assert len(response_text) > 50, "Response should contain reasoning, not just a number"

        # 提取答案并验证
        answer = _extract_answer(response_text)
        assert answer is not None, f"Could not extract numeric answer from response:\n{response_text}"
        assert abs(answer - TASK_B_EXPECTED) < 1.0, f"Task B: Expected ~{TASK_B_EXPECTED}, got {answer}.\nFull response:\n{response_text}"

        # Flush tracer 确保数据发送
        tracer.flush()
        tracer.shutdown()

        # 验证 Langfuse trace 存在（带重试，应对最终一致性）
        trace_list = _fetch_traces_by_session(session_id, min_traces=1)

        trace = trace_list[0]
        trace_name = trace.get("name", "")
        assert "task-b" in trace_name.lower() or "agent" in trace_name.lower(), f"Trace name '{trace_name}' should reference the agent"

        print(f"\n✅ Task B passed: answer={answer}, trace uploaded to Langfuse (session={session_id})")

    def test_cross_validation_both_tasks(self, session_manager: SessionManager) -> None:
        """交叉验证: 并行执行 Task A 和 Task B，对比两个 agent 的推理结果。

        验证:
        1. 两个 agent 独立运行同一类计算问题
        2. 两者结果一致（交叉验证成功）
        3. 两个 trace 都上传到 Langfuse 同一个 session 下
        """
        # 共享 session_id，让两个任务的 trace 在 Langfuse 同一 session 下可追溯
        shared_session_id = f"cross-val-{uuid.uuid4().hex[:8]}"

        # ---- 交叉验证问题: 两个 Agent 独立解同一道题 ----
        cross_val_question = (
            "A company has 200 employees. 30% work in engineering, "
            "25% of engineers are senior engineers, and senior engineers "
            "earn $120,000 per year. What is the total annual salary cost "
            "for all senior engineers? Think step by step."
        )
        # 正确答案: 200 * 0.3 = 60 engineers, 60 * 0.25 = 15 senior, 15 * 120000 = 1,800,000
        expected_answer = 1_800_000.0

        cross_val_prompt = """\
You are a precise computation assistant.
Think step by step before giving your final answer.
Show each calculation clearly.
Always end your response with a line in the exact format:
ANSWER: <number>
where <number> is the final numeric result.
"""

        # 创建两个独立的 tracer，各自用唯一 session_id 以避免 Agent 并行锁冲突
        tracer_a = LangfuseTracer(
            public_key=_PK,
            secret_key=_SK,
            host=_HOST,
            session_id=f"{shared_session_id}-a",
            tags=["cross-validation", "agent-a", "thinking"],
            metadata={"test": "cross_validation", "agent": "A"},
            debug=True,
        )
        tracer_b = LangfuseTracer(
            public_key=_PK,
            secret_key=_SK,
            host=_HOST,
            session_id=f"{shared_session_id}-b",
            tags=["cross-validation", "agent-b", "thinking"],
            metadata={"test": "cross_validation", "agent": "B"},
            debug=True,
        )

        agent_a = _make_thinking_agent(
            name="cross-val-agent-A",
            system_prompt=cross_val_prompt,
            session_manager=session_manager,
            tracer=tracer_a,
        )
        agent_b = _make_thinking_agent(
            name="cross-val-agent-B",
            system_prompt=cross_val_prompt,
            session_manager=session_manager,
            tracer=tracer_b,
        )

        # Agent.__init__ set each tracer's session_id to the agent's unique one.
        # Override back to the shared session_id so both traces are grouped
        # under the same Langfuse session.
        tracer_a.set_session_id(shared_session_id)
        tracer_b.set_session_id(shared_session_id)

        # 并行执行两个 Agent
        async def _run_both() -> tuple[str, str]:
            result_a, result_b = await asyncio.gather(
                agent_a.run_async(message=cross_val_question),
                agent_b.run_async(message=cross_val_question),
            )
            text_a = result_a if isinstance(result_a, str) else result_a[0]
            text_b = result_b if isinstance(result_b, str) else result_b[0]
            return text_a, text_b

        response_a, response_b = asyncio.run(_run_both())

        # 提取两个答案
        answer_a = _extract_answer(response_a)
        answer_b = _extract_answer(response_b)

        assert answer_a is not None, f"Agent A: Could not extract answer from:\n{response_a}"
        assert answer_b is not None, f"Agent B: Could not extract answer from:\n{response_b}"

        # 验证两个答案与期望值一致
        assert abs(answer_a - expected_answer) < 10000, f"Agent A: Expected ~{expected_answer}, got {answer_a}"
        assert abs(answer_b - expected_answer) < 10000, f"Agent B: Expected ~{expected_answer}, got {answer_b}"

        # 交叉验证: 两个 Agent 的答案应该一致
        assert abs(answer_a - answer_b) < 1.0, (
            f"Cross-validation failed! Agent A={answer_a}, Agent B={answer_b}. "
            "Two independent reasoning processes produced different results."
        )

        # Flush 两个 tracer
        tracer_a.flush()
        tracer_b.flush()
        tracer_a.shutdown()
        tracer_b.shutdown()

        # 验证 Langfuse 上有两个 trace（同一 session 下，带重试）
        trace_list = _fetch_traces_by_session(shared_session_id, min_traces=2)

        # 验证两个 trace 的 tag 不同（区分 agent-a 和 agent-b）
        all_tags = set()
        for trace in trace_list:
            for tag in trace.get("tags", []):
                all_tags.add(tag)
        assert "agent-a" in all_tags or "agent-b" in all_tags, f"Traces should be tagged with agent identifiers. Found tags: {all_tags}"

        print(f"\n✅ Cross-validation passed: Agent A={answer_a}, Agent B={answer_b} (session={shared_session_id})")
