"""SSE client for querying Nexau HTTP servers.

This module provides a programmatic client (SSEClient) for interacting
with Nexau HTTP servers, supporting both synchronous and streaming queries.

Note: CLI functionality has been moved to nexau.cli.commands.http.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from nexau.archs.llm.llm_aggregators.events import (
    Event,
    ImageMessageContentEvent,
    ImageMessageEndEvent,
    ImageMessageStartEvent,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ThinkingTextMessageContentEvent,
    ThinkingTextMessageEndEvent,
    ThinkingTextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from nexau.archs.transports.http.models import AgentRequest, AgentResponse
from nexau.core.messages import Message


def _parse_event_dict(event_data: dict[str, Any]) -> Event:
    """Parse a dict to the appropriate Pydantic Event object.

    Args:
        event_data: Dictionary containing event data from SSE

    Returns:
        Parsed Pydantic Event object

    Raises:
        ValueError: If the event type is unknown or unsupported
    """
    event_type = event_data.get("type")

    if event_type == "TEXT_MESSAGE_START":
        return TextMessageStartEvent(**event_data)
    elif event_type == "TEXT_MESSAGE_CONTENT":
        return TextMessageContentEvent(**event_data)
    elif event_type == "TEXT_MESSAGE_END":
        return TextMessageEndEvent(**event_data)
    elif event_type == "THINKING_TEXT_MESSAGE_START":
        return ThinkingTextMessageStartEvent(**event_data)
    elif event_type == "THINKING_TEXT_MESSAGE_CONTENT":
        return ThinkingTextMessageContentEvent(**event_data)
    elif event_type == "THINKING_TEXT_MESSAGE_END":
        return ThinkingTextMessageEndEvent(**event_data)
    elif event_type == "TOOL_CALL_START":
        return ToolCallStartEvent(**event_data)
    elif event_type == "TOOL_CALL_ARGS":
        return ToolCallArgsEvent(**event_data)
    elif event_type == "TOOL_CALL_END":
        return ToolCallEndEvent(**event_data)
    elif event_type == "TOOL_CALL_RESULT":
        return ToolCallResultEvent(**event_data)
    elif event_type == "RUN_STARTED":
        return RunStartedEvent(**event_data)
    elif event_type == "RUN_FINISHED":
        return RunFinishedEvent(**event_data)
    elif event_type == "RUN_ERROR":
        return RunErrorEvent(**event_data)
    elif event_type == "IMAGE_MESSAGE_START":
        return ImageMessageStartEvent(**event_data)
    elif event_type == "IMAGE_MESSAGE_CONTENT":
        return ImageMessageContentEvent(**event_data)
    elif event_type == "IMAGE_MESSAGE_END":
        return ImageMessageEndEvent(**event_data)
    else:
        raise ValueError(f"Unknown event type: {event_type}")


class SSEClient:
    """Client for querying Nexau SSE servers.

    This client supports both synchronous queries (eagerly waiting for complete response)
    and streaming queries (receiving events as they arrive).

    Example:
        >>> client = SSEClient(base_url="http://localhost:8000")
        >>>
        >>> # Simple query
        >>> response = await client.query(messages="What is Python?")
        >>>
        >>> # With conversation history (using Message objects)
        >>> from nexau.core.messages import Message
        >>> response = await client.query(messages=[Message.user("Hello"), Message.assistant("Hi!"), Message.user("How are you?")])
        >>>
        >>> # Streaming
        >>> async for event in client.stream_events(messages="Tell me a story"):
        ...     if event.type == "TEXT_MESSAGE_CONTENT":
        ...         print(event.delta, end="")
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize the SSE client.

        Args:
            base_url: Base URL of the Nexau SSE server (default: http://127.0.0.1:8000)
            http_client: Optional httpx client (will create one if not provided)
        """
        self.base_url = base_url.rstrip("/")
        self._client = http_client

    async def query(
        self,
        messages: str | list[Message],
        user_id: str = "default-user",
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Send a synchronous query to the agent.

        Args:
            messages: The message(s) to send (str or list[Message])
            user_id: User identifier for session management (default: "default-user")
            session_id: Optional session identifier for continuity
            context: Optional runtime context

        Returns:
            The agent's response as a string

        Raises:
            httpx.HTTPError: If the request fails
            ValueError: If the response indicates an error
        """
        # Build request
        request = AgentRequest(
            messages=messages,
            user_id=user_id,
            session_id=session_id,
            context=context,
        )

        # Use existing client or create a new one
        client = self._client or httpx.AsyncClient(timeout=60.0)

        try:
            response = await client.post(
                f"{self.base_url}/query",
                json=request.model_dump(exclude_none=True, mode="json"),
            )
            response.raise_for_status()

            # Parse response using our Pydantic model
            agent_response = AgentResponse.model_validate(response.json())

            if agent_response.status == "success":
                return agent_response.response or ""
            else:
                raise ValueError(f"Server returned error: {agent_response.error or 'Unknown error'}")

        finally:
            # Only close if we created the client
            if not self._client:
                await client.aclose()

    async def stream_events(
        self,
        messages: str | list[Message],
        user_id: str = "default-user",
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        """Stream events from the agent.

        Args:
            messages: The message(s) to send (str or list[Message])
            user_id: User identifier for session management (default: "default-user")
            session_id: Optional session identifier for continuity
            context: Optional runtime context

        Yields:
            AG UI Core Event objects from nexau.archs.llm.llm_aggregators.events

        Raises:
            httpx.HTTPError: If the request fails
        """
        # Build request
        request = AgentRequest(
            messages=messages,
            user_id=user_id,
            session_id=session_id,
            context=context,
        )

        # Use existing client or create a new one
        client = self._client or httpx.AsyncClient(timeout=60.0)

        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/stream",
                json=request.model_dump(exclude_none=True, mode="json"),
            ) as response:
                response.raise_for_status()

                # Use aiter_lines to read SSE format
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])

                        # Convert dict to Pydantic Event object
                        event = _parse_event_dict(event_data)
                        yield event

        finally:
            # Only close if we created the client
            if not self._client:
                await client.aclose()
