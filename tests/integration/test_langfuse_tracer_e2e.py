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

"""E2E tests for Langfuse tracer adapter against a real Langfuse service.

These tests verify that the LangfuseTracer can successfully create traces,
spans, and generations on a real Langfuse instance. They require the following
environment variables to be set:

    LANGFUSE_PUBLIC_KEY  - Langfuse project public key
    LANGFUSE_SECRET_KEY  - Langfuse project secret key
    LANGFUSE_HOST        - Langfuse server URL (e.g. https://cloud.langfuse.com)

Tests are automatically skipped when credentials are not available.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest
import requests

from nexau.archs.tracer.adapters.langfuse import _TTFT_ATTRIBUTE_KEY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external,
    pytest.mark.skipif(
        not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"),
        reason="LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set",
    ),
]

_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
_PK = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SK = os.getenv("LANGFUSE_SECRET_KEY", "")


def _make_tracer():
    """Create a real LangfuseTracer connected to the configured Langfuse instance."""
    from nexau.archs.tracer.adapters import LangfuseTracer

    return LangfuseTracer(
        public_key=_PK,
        secret_key=_SK,
        host=_HOST,
        debug=True,
        session_id=f"e2e-test-{uuid.uuid4().hex[:8]}",
    )


def _get_trace_id(tracer, span) -> str:
    """Extract the OTel trace_id from a span's Langfuse vendor object."""
    from langfuse import LangfuseSpan

    vendor = span.vendor_obj
    assert isinstance(vendor, LangfuseSpan), f"Expected LangfuseSpan, got {type(vendor)}"
    return vendor.trace_id


def _fetch_trace(trace_id: str, retries: int = 15, delay: float = 2.0) -> dict[str, Any]:
    """Fetch a trace from Langfuse REST API with retry (eventual consistency)."""
    for i in range(retries):
        resp = requests.get(
            f"{_HOST}/api/public/traces/{trace_id}",
            auth=(_PK, _SK),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
        if i < retries - 1:
            time.sleep(delay)
    pytest.fail(f"Trace {trace_id} not found after {retries * delay}s")


def test_langfuse_e2e_agent_trace_with_tool_and_llm():
    """E2E: 创建完整的 Agent -> Tool + LLM trace 并验证 Langfuse 服务端数据。

    验证:
    1. Trace 创建成功，包含 observations
    2. Tool span 类型为 SPAN
    3. LLM observation 类型为 GENERATION，包含 model 和 completion_start_time
    4. 脏 usage 数据（非 int 值）被清洗后不会导致 generation 数据丢失
    """
    from nexau.archs.tracer.core import SpanType

    tracer = _make_tracer()

    # 1. Agent span（根）
    agent_span = tracer.start_span(
        "test-agent",
        SpanType.AGENT,
        inputs={"user_message": "hello from e2e test"},
    )
    trace_id = _get_trace_id(tracer, agent_span)

    # 2. Tool span（子）
    tool_span = tracer.start_span(
        "search-tool",
        SpanType.TOOL,
        parent_span=agent_span,
        inputs={"query": "test query"},
    )
    tracer.end_span(tool_span, outputs={"result": "tool result"})

    # 3. LLM generation（子），包含脏 usage 数据
    llm_span = tracer.start_span(
        "gpt-4o-mini",
        SpanType.LLM,
        parent_span=agent_span,
    )
    tracer.end_span(
        llm_span,
        outputs={
            "model": "gpt-4o-mini",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                # 脏数据: 非 int 值，应被 _sanitize_usage 过滤
                "prompt_tokens_details": {"modality": "text"},
                "service_tier": "default",
            },
            "content": "hello from LLM",
        },
        attributes={_TTFT_ATTRIBUTE_KEY: 123.45},
    )

    # 4. 结束 Agent span
    tracer.end_span(agent_span, outputs={"response": "agent response"})

    # 5. Flush + shutdown 确保数据发送
    tracer.flush()
    tracer.shutdown()

    # 6. 从 Langfuse REST API 验证
    trace = _fetch_trace(trace_id)

    observations = trace.get("observations", [])
    assert len(observations) >= 3, f"Expected >= 3 observations (agent, tool, llm), got {len(observations)}"

    obs_by_name: dict[str, dict[str, Any]] = {o["name"]: o for o in observations}

    # 验证 tool span
    assert "search-tool" in obs_by_name
    assert obs_by_name["search-tool"]["type"] == "SPAN"

    # 验证 LLM generation - 关键断言
    assert "gpt-4o-mini" in obs_by_name, (
        f"LLM observation 'gpt-4o-mini' missing - generation may have been dropped! Available: {list(obs_by_name.keys())}"
    )
    llm_obs = obs_by_name["gpt-4o-mini"]
    assert llm_obs["type"] == "GENERATION", f"LLM observation should be GENERATION, got {llm_obs['type']}"
    assert llm_obs.get("model") == "gpt-4o-mini"

    # 验证 usage 被正确写入（非 int 字段应被过滤）
    # Langfuse server 将 prompt_tokens -> input, completion_tokens -> output 进行重映射
    usage = llm_obs.get("usageDetails") or {}
    if usage:
        assert usage.get("input") == 100
        assert usage.get("output") == 50
        assert usage.get("total") == 150
        # 非 int 值（被 _sanitize_usage 过滤）不应出现
        assert "prompt_tokens_details" not in usage
        assert "service_tier" not in usage

    # 验证 TTFT / completion_start_time 被设置
    assert llm_obs.get("completionStartTime") is not None, "completionStartTime should be set from time_to_first_token_ms"


def test_langfuse_e2e_generation_survives_empty_usage():
    """E2E: 空 usage dict 不会导致 generation 丢失。"""
    from nexau.archs.tracer.core import SpanType

    tracer = _make_tracer()

    # LLM generation 需要 parent span 才会创建为 generation 类型
    parent = tracer.start_span("agent-wrapper", SpanType.AGENT)
    llm_span = tracer.start_span(
        "llm-empty-usage",
        SpanType.LLM,
        parent_span=parent,
    )
    trace_id = _get_trace_id(tracer, parent)

    tracer.end_span(
        llm_span,
        outputs={"model": "test-model", "usage": {}, "content": "ok"},
    )
    tracer.end_span(parent, outputs={"done": True})
    tracer.flush()
    tracer.shutdown()

    trace = _fetch_trace(trace_id)

    observations = trace.get("observations", [])
    llm_obs = next((o for o in observations if o["name"] == "llm-empty-usage"), None)
    assert llm_obs is not None, "LLM observation missing - generation may have been dropped"
    assert llm_obs["type"] == "GENERATION"
    assert llm_obs.get("model") == "test-model"
