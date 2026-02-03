"""Run HTTP SSE server in a subprocess for integration tests.

Expects LLM_API_KEY, LLM_MODEL, LLM_BASE_URL to be injected by the parent process.
When HTTP_TEST_MOCK_AGENT=1, patches Agent.run_async to skip LLM.

Usage (env vars injected by parent):
    HTTP_TEST_PORT=9999 python -m tests.integration.run_http_server
"""

from __future__ import annotations

import os
import sys

# Apply mock before importing server (so handle_request uses mocked run_async)
if os.environ.get("HTTP_TEST_MOCK_AGENT") == "1":
    from unittest.mock import AsyncMock, patch

    from nexau.archs.main_sub.agent import Agent

    patch.object(
        Agent,
        "run_async",
        new_callable=AsyncMock,
        return_value="Integration test response",
    ).start()


def main() -> None:
    # Env vars (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL) are injected by parent process
    port = int(os.environ.get("HTTP_TEST_PORT", "0"))
    if port <= 0:
        sys.stderr.write("HTTP_TEST_PORT must be set to a positive integer\n")
        sys.exit(1)

    from nexau.archs.llm.llm_config import LLMConfig
    from nexau.archs.main_sub.config import AgentConfig
    from nexau.archs.session import InMemoryDatabaseEngine
    from nexau.archs.transports.http import HTTPConfig, SSETransportServer

    engine = InMemoryDatabaseEngine()
    agent_config = AgentConfig(
        name="test_agent",
        system_prompt="You are a helpful assistant.",
        llm_config=LLMConfig(),  # model from LLM_MODEL in .env
    )
    server = SSETransportServer(
        engine=engine,
        config=HTTPConfig(host="127.0.0.1", port=port),
        default_agent_config=agent_config,
    )

    import uvicorn

    uvicorn.run(
        server.app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
