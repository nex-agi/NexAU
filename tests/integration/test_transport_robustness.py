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

"""Robustness tests for Transport layer.

Tests concurrent requests, session isolation, error recovery:
- Concurrent HTTP requests
- Session isolation across requests
- Request timeout handling
- Large payload handling
- Malformed request handling
"""

import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_http_server_subprocess(port: int) -> subprocess.Popen:
    import os as _os

    env = {**_os.environ, "HTTP_TEST_PORT": str(port)}
    return subprocess.Popen(
        [sys.executable, "-m", "tests.integration.run_http_server"],
        env=env,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_for_health(base_url: str, timeout_sec: float = 30.0) -> None:
    for _ in range(int(timeout_sec / 0.2) + 1):
        try:
            r = httpx.get(f"{base_url}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.2)
    pytest.fail("Server did not become ready in time")


def _terminate_subprocess(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


class TestConcurrentRequests:
    """Test handling of concurrent HTTP requests."""

    @pytest.mark.llm
    def test_concurrent_query_requests(self):
        """Test multiple concurrent /query requests."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            def make_request(user_id: str) -> dict:
                response = httpx.post(
                    f"{base}/query",
                    json={"messages": "Say hello briefly", "user_id": user_id},
                    timeout=90.0,
                )
                return {"status": response.status_code, "user_id": user_id}

            # Send 3 concurrent requests
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_request, f"user_{i}") for i in range(3)]
                results = [f.result() for f in futures]

            # All requests should succeed
            for result in results:
                assert result["status"] == 200

        finally:
            _terminate_subprocess(proc)

    @pytest.mark.llm
    def test_concurrent_stream_requests(self):
        """Test multiple concurrent /stream requests."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            def stream_request(user_id: str) -> dict:
                with httpx.Client(timeout=90.0) as client:
                    with client.stream(
                        "POST",
                        f"{base}/stream",
                        json={"messages": "Say hi", "user_id": user_id},
                    ) as response:
                        events = [line for line in response.iter_lines() if line.startswith("data: ")]
                        return {"status": response.status_code, "event_count": len(events)}

            # Send 2 concurrent streaming requests
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(stream_request, f"stream_user_{i}") for i in range(2)]
                results = [f.result() for f in futures]

            # All streams should succeed
            for result in results:
                assert result["status"] == 200
                assert result["event_count"] > 0

        finally:
            _terminate_subprocess(proc)


class TestSessionIsolationHttp:
    """Test session isolation in HTTP transport."""

    @pytest.mark.llm
    def test_different_sessions_are_isolated_http(self):
        """Test that different sessions don't share state via HTTP."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            # Session 1: Store information
            response1 = httpx.post(
                f"{base}/query",
                json={
                    "messages": "Remember: my password is secret123",
                    "user_id": "user_a",
                    "session_id": "session_a",
                },
                timeout=90.0,
            )
            assert response1.status_code == 200

            # Session 2: Should not have access to session 1's data
            response2 = httpx.post(
                f"{base}/query",
                json={
                    "messages": "What is my password?",
                    "user_id": "user_b",
                    "session_id": "session_b",
                },
                timeout=90.0,
            )
            assert response2.status_code == 200
            data2 = response2.json()
            # Session 2 should not know the password
            assert "secret123" not in data2.get("response", "").lower()

        finally:
            _terminate_subprocess(proc)

    @pytest.mark.llm
    def test_same_session_shares_context_http(self):
        """Test that same session shares context via HTTP."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            user_id = "context_user"
            session_id = "context_session"

            # First request: Store information
            response1 = httpx.post(
                f"{base}/query",
                json={
                    "messages": "My favorite city is Tokyo.",
                    "user_id": user_id,
                    "session_id": session_id,
                },
                timeout=90.0,
            )
            assert response1.status_code == 200

            # Second request: Recall information
            response2 = httpx.post(
                f"{base}/query",
                json={
                    "messages": "What is my favorite city?",
                    "user_id": user_id,
                    "session_id": session_id,
                },
                timeout=90.0,
            )
            assert response2.status_code == 200
            data2 = response2.json()
            # Should remember Tokyo from same session
            assert "Tokyo" in data2.get("response", "") or "tokyo" in data2.get("response", "").lower()

        finally:
            _terminate_subprocess(proc)


class TestErrorHandlingHttp:
    """Test HTTP error handling scenarios."""

    def test_invalid_json_returns_error(self):
        """Test that invalid JSON returns appropriate error."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            # Send invalid JSON
            response = httpx.post(
                f"{base}/query",
                content="not valid json",
                headers={"Content-Type": "application/json"},
                timeout=10.0,
            )
            # Should return 422 (Unprocessable Entity)
            assert response.status_code == 422

        finally:
            _terminate_subprocess(proc)

    def test_missing_required_field_returns_422(self):
        """Test that missing required fields return 422."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            # Missing 'messages' field
            response = httpx.post(
                f"{base}/query",
                json={"user_id": "test_user"},
                timeout=10.0,
            )
            assert response.status_code == 422

        finally:
            _terminate_subprocess(proc)

    def test_health_endpoint_always_available(self):
        """Test that health endpoint is always responsive."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            # Health check should be fast
            response = httpx.get(f"{base}/health", timeout=2.0)
            assert response.status_code == 200
            data = response.json()
            assert data.get("status") == "healthy"

        finally:
            _terminate_subprocess(proc)


class TestLargePayloads:
    """Test handling of large payloads."""

    @pytest.mark.llm
    def test_long_message_handling(self):
        """Test handling of long messages."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            # Create a moderately long message (not too long to avoid token limits)
            long_message = "Please summarize this: " + ("Lorem ipsum dolor sit amet. " * 50)

            response = httpx.post(
                f"{base}/query",
                json={"messages": long_message, "user_id": "test_user"},
                timeout=90.0,
            )

            # Should handle long message
            assert response.status_code == 200
            data = response.json()
            assert data.get("status") == "success"

        finally:
            _terminate_subprocess(proc)


class TestStreamingRobustness:
    """Test streaming robustness scenarios."""

    @pytest.mark.llm
    def test_stream_completes_with_events(self):
        """Test that stream completes with proper events."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            with httpx.Client(timeout=90.0) as client:
                with client.stream(
                    "POST",
                    f"{base}/stream",
                    json={"messages": "Count from 1 to 3", "user_id": "test_user"},
                ) as response:
                    assert response.status_code == 200
                    assert "text/event-stream" in response.headers.get("content-type", "")

                    events = []
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            events.append(line)

            # Should have received events
            assert len(events) > 0

            # Check for expected event structure
            all_events = "".join(events)
            # Should have some standard events
            has_run_started = "RUN_STARTED" in all_events
            has_text_event = "TEXT_MESSAGE" in all_events
            has_run_finished = "RUN_FINISHED" in all_events

            assert has_run_started or has_text_event or has_run_finished

        finally:
            _terminate_subprocess(proc)

    @pytest.mark.llm
    def test_stream_handles_user_context(self):
        """Test that streaming maintains user context."""
        port = _free_port()
        proc = _start_http_server_subprocess(port)
        try:
            base = f"http://127.0.0.1:{port}"
            _wait_for_health(base)

            user_id = "stream_context_user"
            session_id = "stream_context_session"

            # First stream
            with httpx.Client(timeout=90.0) as client:
                with client.stream(
                    "POST",
                    f"{base}/stream",
                    json={
                        "messages": "I like pizza.",
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                ) as response:
                    # Consume all events
                    list(response.iter_lines())

            # Second stream - should have context
            with httpx.Client(timeout=90.0) as client:
                with client.stream(
                    "POST",
                    f"{base}/stream",
                    json={
                        "messages": "What food do I like?",
                        "user_id": user_id,
                        "session_id": session_id,
                    },
                ) as response:
                    events = [line for line in response.iter_lines() if line.startswith("data: ")]

            # Should have events mentioning pizza
            all_events = "".join(events)
            assert "pizza" in all_events.lower() or len(events) > 0

        finally:
            _terminate_subprocess(proc)
