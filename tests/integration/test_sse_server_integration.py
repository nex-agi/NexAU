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

"""Integration tests for SSE transport server with real Agent and real LLM.

Load LLM_API_KEY (and optionally LLM_BASE_URL, LLM_MODEL) from .env locally, or set in CI.
No Agent mock; all tests use real requests.
"""

import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_http_server_subprocess(port: int) -> subprocess.Popen:
    import os as _os

    # Inherit all parent env vars, explicitly set HTTP_TEST_PORT
    # LLM_* env vars are loaded from .env in conftest before any nexau imports
    env = {**_os.environ, "HTTP_TEST_PORT": str(port)}
    return subprocess.Popen(
        [sys.executable, "-m", "tests.integration.run_http_server"],
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_for_health(base_url: str, timeout_sec: float = 15.0) -> None:
    for _ in range(int(timeout_sec / 0.2) + 1):
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.2)
    pytest.fail("Server did not become ready in time")


def _terminate_subprocess(proc: subprocess.Popen[bytes]) -> None:
    """Terminate subprocess; kill if wait times out."""
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


class TestSSEServerQueryIntegration:
    """Integration tests for /query endpoint with real Agent and real LLM."""

    @pytest.mark.llm
    def test_query_endpoint_success(self):
        """Test /query with real server and real LLM."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)
            response = httpx.post(
                f"{base}/query",
                json={"messages": "Hello", "user_id": "test_user"},
                timeout=90.0,
            )
            if response.status_code != 200:
                raise AssertionError(f"Expected 200, got {response.status_code}. Body: {response.text!r}")
            data = response.json()
            assert data["status"] == "success"
            assert isinstance(data.get("response"), str)
            assert len(data["response"].strip()) > 0
        finally:
            _terminate_subprocess(proc)

    @pytest.mark.llm
    def test_query_endpoint_with_session_id(self):
        """Test /query with session_id using real server and real LLM."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)
            response = httpx.post(
                f"{base}/query",
                json={
                    "messages": "Hello",
                    "user_id": "test_user",
                    "session_id": "sess_123",
                },
                timeout=90.0,
            )
            if response.status_code != 200:
                raise AssertionError(f"Expected 200, got {response.status_code}. Body: {response.text!r}")
            data = response.json()
            assert data["status"] == "success"
            assert len(data["response"].strip()) > 0
        finally:
            _terminate_subprocess(proc)


class TestSSEServerStreamIntegration:
    """Integration tests for /stream endpoint with real Agent and real LLM."""

    @pytest.mark.llm
    def test_stream_endpoint_yields_events(self):
        """Test /stream yields SSE events from real Agent and real LLM."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)
            with httpx.Client(timeout=90.0) as client:
                with client.stream(
                    "POST",
                    f"{base}/stream",
                    json={"messages": "Hello", "user_id": "test_user"},
                ) as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers.get("content-type", "")
                    events = [line for line in response.iter_lines() if line.startswith("data: ")]
            assert len(events) >= 1
            body = "".join(events)
            assert "RUN_STARTED" in body or "TEXT_MESSAGE" in body or "RUN_FINISHED" in body
        finally:
            _terminate_subprocess(proc)


class TestSSETransportHttpIntegration:
    """Integration tests for HTTP transport with to_thread Agent creation.

    Exercises: POST /query -> real handle_request -> asyncio.to_thread(create_agent)
    -> Agent() in worker thread -> real run_async -> real LLM.
    """

    @pytest.mark.llm
    def test_query_http_transport_real_server_and_to_thread_agent_creation(self):
        """Test POST /query against real HTTP server; real Agent via to_thread, real LLM."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)
            response = httpx.post(
                f"{base}/query",
                json={"messages": "Hello", "user_id": "test_user"},
                timeout=90.0,
            )
            if response.status_code != 200:
                raise AssertionError(f"Expected 200, got {response.status_code}. Body: {response.text!r}")
            data = response.json()
            assert data["status"] == "success"
            assert isinstance(data.get("response"), str)
            assert len(data["response"].strip()) > 0
        finally:
            _terminate_subprocess(proc)
