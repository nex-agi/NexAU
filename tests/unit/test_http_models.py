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

"""Unit tests for HTTP transport models."""

from nexau.archs.transports.http.models import AgentRequest, AgentResponse
from nexau.core.messages import Message


class TestAgentRequest:
    """Test cases for AgentRequest model."""

    def test_simple_string_message(self):
        """Test request with simple string message."""
        request = AgentRequest(messages="Hello", user_id="user123")
        assert request.messages == "Hello"
        assert request.user_id == "user123"
        assert request.session_id is None
        assert request.context is None

    def test_message_list(self):
        """Test request with message list."""
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
            Message.user("How are you?"),
        ]
        request = AgentRequest(messages=messages, user_id="user123")
        assert len(request.messages) == 3
        assert request.messages[0].role == "user"

    def test_with_session_id(self):
        """Test request with session ID."""
        request = AgentRequest(
            messages="Hello",
            user_id="user123",
            session_id="sess_abc",
        )
        assert request.session_id == "sess_abc"

    def test_with_context(self):
        """Test request with context."""
        request = AgentRequest(
            messages="Hello",
            user_id="user123",
            context={"key": "value"},
        )
        assert request.context == {"key": "value"}

    def test_default_user_id(self):
        """Test default user ID."""
        request = AgentRequest(messages="Hello")
        assert request.user_id == "default-user"


class TestAgentResponse:
    """Test cases for AgentResponse model."""

    def test_success_response(self):
        """Test successful response."""
        response = AgentResponse(status="success", response="Hello!")
        assert response.status == "success"
        assert response.response == "Hello!"
        assert response.error is None

    def test_error_response(self):
        """Test error response."""
        response = AgentResponse(status="error", error="Something went wrong")
        assert response.status == "error"
        assert response.response is None
        assert response.error == "Something went wrong"

    def test_minimal_response(self):
        """Test minimal response with only status."""
        response = AgentResponse(status="success")
        assert response.status == "success"
        assert response.response is None
        assert response.error is None

    # --- RFC-0018: External tool 暂停扩展字段 ---

    def test_external_tool_pause_response(self):
        """RFC-0018: AgentResponse with stop_reason and pending_tool_calls."""
        pending = [
            {"id": "call_abc", "name": "search", "input": {"query": "hello"}},
            {"id": "call_def", "name": "read_file", "input": {"path": "/tmp/a.txt"}},
        ]
        response = AgentResponse(
            status="success",
            response="",
            stop_reason="EXTERNAL_TOOL_CALL",
            pending_tool_calls=pending,
        )
        assert response.status == "success"
        assert response.response == ""
        assert response.stop_reason == "EXTERNAL_TOOL_CALL"
        assert response.pending_tool_calls is not None
        assert len(response.pending_tool_calls) == 2
        assert response.pending_tool_calls[0]["name"] == "search"
        assert response.error is None

    def test_backward_compat_no_external_fields(self):
        """RFC-0018: Existing responses have stop_reason/pending_tool_calls as None."""
        response = AgentResponse(status="success", response="Hello!")
        assert response.stop_reason is None
        assert response.pending_tool_calls is None

    def test_external_tool_pause_json_roundtrip(self):
        """RFC-0018: JSON serialization round-trip for external tool pause."""
        pending = [{"id": "call_1", "name": "ext_tool", "input": {"x": 42}}]
        original = AgentResponse(
            status="success",
            response="",
            stop_reason="EXTERNAL_TOOL_CALL",
            pending_tool_calls=pending,
        )
        json_str = original.model_dump_json()
        restored = AgentResponse.model_validate_json(json_str)
        assert restored.stop_reason == "EXTERNAL_TOOL_CALL"
        assert restored.pending_tool_calls == pending
        assert restored.response == ""
