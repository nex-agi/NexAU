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

"""Integration tests for LLM failover middleware.

Tests the failover middleware end-to-end:
- Primary LLM (mock HTTP server returning 500) fails
- Middleware catches the error and switches to fallback
- Fallback LLM (real provider from .env) succeeds
"""

from __future__ import annotations

import http.server
import os
import threading
from typing import ClassVar

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.middleware.llm_failover import (
    LLMFailoverMiddleware,
)
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager

# ---------------------------------------------------------------------------
# Mock HTTP server that always returns 500
# ---------------------------------------------------------------------------


class _FailingLLMHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that returns 500 for all POST requests (simulates a broken LLM)."""

    request_count: ClassVar[int] = 0

    def do_POST(self) -> None:
        _FailingLLMHandler.request_count += 1
        self.send_response(500)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error": {"message": "Internal Server Error", "type": "server_error"}}')

    def log_message(self, format: str, *args: object) -> None:
        # 静默日志，避免测试输出噪音
        pass


@pytest.fixture()
def failing_llm_server():
    """Start a local HTTP server that always returns 500.

    Yields (host, port) tuple. Server is shut down after the test.
    """
    _FailingLLMHandler.request_count = 0
    server = http.server.HTTPServer(("127.0.0.1", 0), _FailingLLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield "127.0.0.1", port
    server.shutdown()


@pytest.fixture()
def session_manager():
    engine = InMemoryDatabaseEngine()
    return SessionManager(engine=engine)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMFailoverIntegration:
    """End-to-end failover: mock failing primary → real fallback from .env."""

    @pytest.mark.llm
    def test_failover_from_broken_primary_to_real_llm(self, failing_llm_server: tuple[str, int], session_manager: SessionManager) -> None:
        """Primary (mock 500 server) fails → fallback (real LLM) succeeds.

        RFC-0003: 集成测试 — 主 provider 500 → 自动降级到真实 LLM
        """
        host, port = failing_llm_server

        # 1. 主 provider: 指向本地 mock server（永远返回 500）
        primary_config = LLMConfig(
            model=os.environ["LLM_MODEL"],
            base_url=f"http://{host}:{port}/v1",
            api_key="sk-fake-will-fail",
            api_type="openai_chat_completion",
            max_retries=0,  # 不让 SDK 自己重试，让 middleware 处理
        )

        # 2. Fallback provider: 使用 .env 中的真实 LLM
        fallback_llm_config: dict[str, str | int | float | bool] = {
            "base_url": os.environ["LLM_BASE_URL"],
            "api_key": os.environ["LLM_API_KEY"],
        }

        # 3. 创建 failover middleware
        failover_middleware = LLMFailoverMiddleware(
            trigger={
                "status_codes": [500, 502, 503],
                "exception_types": ["APIConnectionError"],
            },
            fallback_providers=[
                {"name": "real-llm", "llm_config": fallback_llm_config},
            ],
        )

        # 4. 构建 agent
        config = AgentConfig(
            name="failover_test_agent",
            system_prompt="You are a helpful assistant. Respond with exactly one word.",
            llm_config=primary_config,
            middlewares=[failover_middleware],
            max_iterations=2,
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="failover_test",
        )

        # 5. 运行 — 主 provider 会 500，middleware 应自动切到 fallback
        response = agent.run(message="Say hello")

        # 6. 验证
        assert isinstance(response, str)
        assert len(response) > 0
        # mock server 应该收到过请求（证明确实先尝试了 primary）
        assert _FailingLLMHandler.request_count >= 1

    @pytest.mark.llm
    def test_no_failover_when_primary_works(self, session_manager: SessionManager) -> None:
        """When primary LLM works, no failover should happen.

        RFC-0003: 集成测试 — 主 provider 正常时不触发降级
        """
        # 主 provider: 使用 .env 中的真实 LLM
        primary_config = LLMConfig()

        failover_middleware = LLMFailoverMiddleware(
            trigger={"status_codes": [500], "exception_types": []},
            fallback_providers=[
                {
                    "name": "should-not-be-used",
                    "llm_config": {
                        "base_url": "http://127.0.0.1:1/v1",  # 不可达地址
                        "api_key": "sk-fake",
                    },
                },
            ],
        )

        config = AgentConfig(
            name="no_failover_agent",
            system_prompt="You are a helpful assistant. Respond with exactly one word.",
            llm_config=primary_config,
            middlewares=[failover_middleware],
            max_iterations=2,
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="no_failover_test",
        )

        response = agent.run(message="Say hello")
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.llm
    def test_failover_with_circuit_breaker(self, failing_llm_server: tuple[str, int], session_manager: SessionManager) -> None:
        """Circuit breaker skips primary after repeated failures.

        RFC-0003: 集成测试 — 熔断器在连续失败后跳过主 provider
        """
        host, port = failing_llm_server

        primary_config = LLMConfig(
            model=os.environ["LLM_MODEL"],
            base_url=f"http://{host}:{port}/v1",
            api_key="sk-fake-will-fail",
            api_type="openai_chat_completion",
            max_retries=0,
        )

        failover_middleware = LLMFailoverMiddleware(
            trigger={"status_codes": [500], "exception_types": []},
            fallback_providers=[
                {
                    "name": "real-llm",
                    "llm_config": {
                        "base_url": os.environ["LLM_BASE_URL"],
                        "api_key": os.environ["LLM_API_KEY"],
                    },
                },
            ],
            circuit_breaker={"failure_threshold": 1, "recovery_timeout_seconds": 9999},
        )

        config = AgentConfig(
            name="cb_failover_agent",
            system_prompt="You are a helpful assistant. Respond with exactly one word.",
            llm_config=primary_config,
            middlewares=[failover_middleware],
            max_iterations=2,
        )

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="cb_test_1",
        )

        # 第一次调用：primary 失败 → failover → 成功
        response1 = agent.run(message="Say hello")
        assert isinstance(response1, str)
        assert len(response1) > 0
        first_request_count = _FailingLLMHandler.request_count
        assert first_request_count >= 1

        # 第二次调用：熔断器打开，应直接走 fallback（不再请求 mock server）
        agent2 = Agent(
            config=AgentConfig(
                name="cb_failover_agent_2",
                system_prompt="You are a helpful assistant. Respond with exactly one word.",
                llm_config=primary_config,
                middlewares=[failover_middleware],
                max_iterations=2,
            ),
            session_manager=session_manager,
            user_id="test_user",
            session_id="cb_test_2",
        )
        response2 = agent2.run(message="Say world")
        assert isinstance(response2, str)
        assert len(response2) > 0
        # mock server 不应收到新请求（熔断器跳过了 primary）
        assert _FailingLLMHandler.request_count == first_request_count
