"""Shared request/response models for HTTP transport.

This module defines the API contract between HTTP clients and servers,
following OpenAI and Anthropic API design patterns.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from nexau.archs.main_sub.context_value import ContextValue
from nexau.core.messages import Message


class AgentRequest(BaseModel):
    """Request model for /query and /stream endpoints.

    Follows OpenAI/Anthropic API style where the main field is 'messages'.
    Supports both simple string queries and full conversation history.

    Examples:
        Simple query:
        >>> AgentRequest(messages="What is Python?", user_id="user123")

        Conversation history:
        >>> from nexau.core.messages import Message
        >>> AgentRequest(messages=[Message.user("Hello"), Message.assistant("Hi!"), Message.user("How are you?")], user_id="user123")
    """

    messages: str | list[Message]
    user_id: str = "default-user"
    session_id: str | None = None
    context: dict[str, Any] | None = None
    variables: ContextValue | None = None


class AgentResponse(BaseModel):
    """Response model for /query endpoint (non-streaming).

    Follows standard REST API success/error pattern.

    RFC-0018: 当 agent 因 external tool 调用暂停时，``stop_reason`` 为
    ``"EXTERNAL_TOOL_CALL"``，``pending_tool_calls`` 携带待执行的工具调用列表
    （每项含 ``id``/``name``/``input``）。调用方处理完毕后通过同一 ``/query``
    端点传入 ``ToolResultBlock`` 消息恢复执行。
    """

    status: str  # "success" or "error"
    response: str | None = None
    error: str | None = None
    # RFC-0018: External tool 暂停时的扩展字段
    stop_reason: str | None = None
    pending_tool_calls: list[dict[str, Any]] | None = None
    # RFC-0018 T7: 只读观测回显 — Agent 生成的 session-level trace_id，
    # 客户端仅用于关联 Langfuse/OTel 等观测后端；resume 时无需回传
    # (凭相同 session_id 由服务端从 SessionModel.current_trace_id 恢复)。
    trace_id: str | None = None


class StopRequest(BaseModel):
    """Request model for /stop endpoint.

    RFC-0001 Phase 4: Transport 层 stop 端点
    """

    user_id: str = "default-user"
    session_id: str
    agent_id: str | None = None
    force: bool = False
    timeout: float = 30.0


class StopResponse(BaseModel):
    """Response model for /stop endpoint.

    RFC-0001 Phase 4: Transport 层 stop 端点
    """

    status: str  # "success" or "error"
    stop_reason: str | None = None
    message_count: int = 0
    error: str | None = None
