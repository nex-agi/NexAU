"""Normalized representation of model responses and tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelToolCall:
    """Represents a tool call emitted by the model in a model-agnostic format."""

    call_id: str | None
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    call_type: str = "function"
    raw_call: Any = None

    @classmethod
    def from_openai(cls, call: Any) -> ModelToolCall:
        """Create a ModelToolCall from an OpenAI tool call payload."""
        if call is None:
            raise ValueError("call payload cannot be None")

        # Support both dict-style and attribute-style payloads
        get = call.get if isinstance(call, dict) else getattr
        call_id = get("id") if isinstance(call, dict) else getattr(call, "id", None)
        call_type = get("type") if isinstance(call, dict) else getattr(call, "type", "function")

        function = None
        if isinstance(call, dict):
            function = call.get("function")
        else:
            function = getattr(call, "function", None)

        if function is None:
            raise ValueError("OpenAI tool call payload missing function block")

        if isinstance(function, dict):
            name = function.get("name")
            raw_arguments = function.get("arguments")
        else:
            name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", None)

        if not name:
            raise ValueError("OpenAI tool call function block missing name")

        parsed_arguments: dict[str, Any] = {}
        if isinstance(raw_arguments, str):
            raw_arguments_str = raw_arguments.strip()
            if raw_arguments_str:
                try:
                    parsed_arguments = json.loads(raw_arguments_str)
                    if not isinstance(parsed_arguments, dict):
                        parsed_arguments = {"_": parsed_arguments}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw_arguments": raw_arguments}
        elif raw_arguments is not None:
            if isinstance(raw_arguments, dict):
                parsed_arguments = raw_arguments
            else:
                parsed_arguments = {"raw_arguments": json.dumps(raw_arguments)}
        return cls(
            call_id=call_id,
            name=name,
            arguments=parsed_arguments,
            raw_arguments=raw_arguments
            if isinstance(raw_arguments, str)
            else json.dumps(raw_arguments)
            if raw_arguments is not None
            else None,
            call_type=call_type or "function",
            raw_call=call,
        )

    def to_openai_dict(self) -> dict[str, Any]:
        """Convert the tool call back to OpenAI-compatible dict."""
        function_body: dict[str, Any] = {"name": self.name}
        if self.raw_arguments is not None:
            function_body["arguments"] = self.raw_arguments
        else:
            function_body["arguments"] = json.dumps(self.arguments, ensure_ascii=False)
        return {
            "id": self.call_id,
            "type": self.call_type,
            "function": function_body,
        }


@dataclass
class ModelResponse:
    """Normalized model response returned by LLMCaller."""

    content: str | None = None
    tool_calls: list[ModelToolCall] = field(default_factory=list)
    role: str = "assistant"
    raw_message: Any = None

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []

    @classmethod
    def from_openai_message(cls, message: Any) -> ModelResponse:
        """Create a ModelResponse from an OpenAI chat completion message."""
        if message is None:
            raise ValueError("message cannot be None")

        content = getattr(message, "content", None)
        # openai-beta returns list of content parts; join if needed
        if isinstance(content, list):
            # Attempt to join textual content pieces
            content = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)

        raw_tool_calls = getattr(message, "tool_calls", None)
        if raw_tool_calls is None and isinstance(message, dict):
            raw_tool_calls = message.get("tool_calls")

        tool_calls: list[ModelToolCall] = []
        if raw_tool_calls:
            for call in raw_tool_calls:
                try:
                    tool_calls.append(ModelToolCall.from_openai(call))
                except Exception:
                    # If parsing fails, create minimal placeholder
                    tool_calls.append(
                        ModelToolCall(
                            call_id=getattr(call, "id", None) if not isinstance(call, dict) else call.get("id"),
                            name="unknown",
                            arguments={"raw_call": call},
                            raw_arguments=None,
                            call_type=getattr(call, "type", "function") if not isinstance(call, dict) else call.get("type", "function"),
                            raw_call=call,
                        ),
                    )

        role = getattr(message, "role", "assistant") if not isinstance(message, dict) else message.get("role", "assistant")
        return cls(
            content=content,
            tool_calls=tool_calls,
            role=role,
            raw_message=message,
        )

    @classmethod
    def from_anthropic_message(cls, message: Any) -> ModelResponse:
        """Create a ModelResponse from an Anthropic chat completion message."""
        if message is None:
            raise ValueError("message cannot be None")

        content_blocks = getattr(message, "content", None)
        if content_blocks is None and isinstance(message, dict):
            content_blocks = message.get("content")
        if content_blocks is None:
            content_blocks = []

        # Extract text content from content blocks
        text_parts: list[str] = []
        raw_tool_calls: list[Any] = []

        for block in content_blocks:
            block_type = getattr(block, "type", None) if not isinstance(block, dict) else block.get("type")
            if block_type == "text":
                text = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                raw_tool_calls.append(block)

        content = "\n".join(text_parts) if text_parts else ""

        tool_calls: list[ModelToolCall] = []
        if raw_tool_calls:
            for call in raw_tool_calls:
                # Handle both dict and object (ToolUseBlock) cases
                if isinstance(call, dict):
                    call_id = call.get("id")
                    name = call.get("name")
                    arguments = call.get("input", {})
                    call_type = call.get("type", "function")
                else:
                    call_id = getattr(call, "id", None)
                    name = getattr(call, "name", None)
                    arguments = getattr(call, "input", {})
                    call_type = getattr(call, "type", "function")

                tool_calls.append(
                    ModelToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=arguments,
                        raw_arguments=json.dumps(arguments, ensure_ascii=False),
                        call_type=call_type,
                        raw_call=call,
                    ),
                )

        role = getattr(message, "role", "assistant") if not isinstance(message, dict) else message.get("role", "assistant")
        return cls(
            content=content,
            tool_calls=tool_calls,
            role=role,
            raw_message=message,
        )

    def has_content(self) -> bool:
        return bool(self.content and self.content.strip())

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def render_text(self) -> str:
        """Render the response as a human-readable string for logging."""
        parts: list[str] = []
        if self.has_content():
            parts.append(self.content.strip())
        for call in self.tool_calls:
            try:
                arg_preview = json.dumps(call.arguments, ensure_ascii=False)
            except TypeError:
                arg_preview = str(call.arguments)
            parts.append(
                f"[tool_call id={call.call_id or 'unknown'} name={call.name} args={arg_preview}]",
            )
        return "\n".join(parts).strip()

    def to_message_dict(self) -> dict[str, Any]:
        """Convert response into chat completion message dict."""
        message: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [call.to_openai_dict() for call in self.tool_calls]
        return message

    def __str__(self) -> str:  # pragma: no cover - convenience
        rendered = self.render_text()
        return rendered if rendered else ""
