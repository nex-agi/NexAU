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
    """

    status: str  # "success" or "error"
    response: str | None = None
    error: str | None = None
