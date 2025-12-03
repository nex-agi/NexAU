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

"""Normalized representation of model responses and tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _item_get(item: Any, key: str, default: Any = None) -> Any:
    """Best-effort attribute/dict access helper for SDK objects."""

    if isinstance(item, dict):
        return item.get(key, default)

    # openai/anthropic SDK objects expose attributes directly
    return getattr(item, key, default)


def _to_serializable_dict(payload: Any) -> dict[str, Any]:
    """Attempt to coerce SDK payloads into simple dicts for parsing."""

    if isinstance(payload, dict):
        return payload

    # Prefer model_dump if available (pydantic models)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump()
        except Exception:  # pragma: no cover - fallback
            pass

    # Fallback to attribute introspection (best-effort, non-recursive)
    result: dict[str, Any] = {}
    for attr in dir(payload):  # pragma: no cover - defensive
        if attr.startswith("_"):
            continue
        try:
            value = getattr(payload, attr)
        except Exception:
            continue
        if callable(value):
            continue
        result[attr] = value
    return result


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize usage information to a standard format.

    Standard format:
    - input_tokens: Number of tokens in the input/prompt
    - reasoning_tokens: Number of tokens used for reasoning (if applicable)
    - completion_tokens: Number of tokens in the completion/output
    - total_tokens: Total number of tokens used

    Supports both OpenAI format (prompt_tokens, completion_tokens, total_tokens)
    and Anthropic format (input_tokens, output_tokens).

    Args:
        usage: Raw usage dict from model response

    Returns:
        Normalized usage dict or None if input is None
    """
    if usage is None:
        return None

    # Ensure usage is a dict
    if not isinstance(usage, dict):
        return None

    normalized: dict[str, Any] = {}

    # Handle input tokens
    if "input_tokens" in usage:
        normalized["input_tokens"] = usage["input_tokens"]
    elif "prompt_tokens" in usage:
        normalized["input_tokens"] = usage["prompt_tokens"]
    else:
        normalized["input_tokens"] = 0

    # Handle reasoning tokens (for models that support it)
    if "reasoning_tokens" in usage:
        normalized["reasoning_tokens"] = usage["reasoning_tokens"]
    else:
        normalized["reasoning_tokens"] = 0

    # Handle completion/output tokens
    if "completion_tokens" in usage:
        normalized["completion_tokens"] = usage["completion_tokens"]
    elif "output_tokens" in usage:
        normalized["completion_tokens"] = usage["output_tokens"]
    else:
        normalized["completion_tokens"] = 0

    # Handle total tokens
    if "total_tokens" in usage:
        normalized["total_tokens"] = usage["total_tokens"]
    else:
        # Calculate total if not provided
        normalized["total_tokens"] = normalized["input_tokens"] + normalized["reasoning_tokens"] + normalized["completion_tokens"]

    return normalized


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
    response_items: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None
    """A description of the chain of thought used by a reasoning model while generating a response.

    Be sure to include these items in your input to the Responses API for subsequent turns of a
    conversation if you are manually managing context.
    """
    usage: dict[str, Any] | None = None
    """Token usage information from the model response (normalized format).

    Standard format:
    - input_tokens: Number of tokens in the input/prompt
    - reasoning_tokens: Number of tokens used for reasoning (if applicable, 0 otherwise)
    - completion_tokens: Number of tokens in the completion/output
    - total_tokens: Total number of tokens used

    All usage data is automatically normalized from provider-specific formats
    (e.g., OpenAI's prompt_tokens/completion_tokens, Anthropic's input_tokens/output_tokens)
    into this standard format.
    """

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []
        if self.response_items is None:
            self.response_items = []
        if self.usage is None:
            self.usage = {}

    @classmethod
    def from_openai_message(cls, message: Any, usage: dict[str, Any] | None = None) -> ModelResponse:
        """Create a ModelResponse from an OpenAI chat completion message.

        Args:
            message: The message object from OpenAI response
            usage: Optional usage statistics from the response object
        """
        if message is None:
            raise ValueError("message cannot be None")

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
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

        # Extract reasoning_content if available (for models like kimi-k2-thinking)
        reasoning_content = None
        if hasattr(message, "reasoning_content"):
            reasoning_content = getattr(message, "reasoning_content")
        elif isinstance(message, dict) and "reasoning_content" in message:
            reasoning_content = message["reasoning_content"]

        # Extract usage information if available in the message itself
        message_usage = None
        if hasattr(message, "usage"):
            message_usage = _to_serializable_dict(getattr(message, "usage"))
        elif isinstance(message, dict) and "usage" in message:
            raw_usage = message.get("usage")
            message_usage = _to_serializable_dict(raw_usage) if raw_usage is not None else None

        # Prefer explicitly passed usage over message-embedded usage
        final_usage = usage if usage is not None else message_usage

        role = getattr(message, "role", "assistant") if not isinstance(message, dict) else message.get("role", "assistant")
        return cls(
            content=content,
            tool_calls=tool_calls,
            role=role,
            raw_message=message,
            reasoning_content=reasoning_content if reasoning_content else None,
            usage=_normalize_usage(final_usage),
        )

    @classmethod
    def from_anthropic_message(cls, message: Any, usage: dict[str, Any] | None = None) -> ModelResponse:
        """Create a ModelResponse from an Anthropic chat completion message.

        Args:
            message: The message object from Anthropic response
            usage: Optional usage statistics from the response object
        """
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

        # Extract usage information if available in the message itself
        message_usage = None
        if hasattr(message, "usage"):
            message_usage = _to_serializable_dict(getattr(message, "usage"))
        elif isinstance(message, dict) and "usage" in message:
            raw_usage = message.get("usage")
            message_usage = _to_serializable_dict(raw_usage) if raw_usage is not None else None

        # Prefer explicitly passed usage over message-embedded usage
        final_usage = usage if usage is not None else message_usage

        role = getattr(message, "role", "assistant") if not isinstance(message, dict) else message.get("role", "assistant")
        return cls(
            content=content,
            tool_calls=tool_calls,
            role=role,
            raw_message=message,
            usage=_normalize_usage(final_usage),
        )

    @classmethod
    def from_openai_response(cls, response: Any) -> ModelResponse:
        """Create a ModelResponse from an OpenAI Responses API payload."""
        if response is None:
            raise ValueError("response cannot be None")

        output_items = _item_get(response, "output", []) or []
        response_items: list[dict[str, Any]] = []
        collected_content: list[str] = []
        tool_calls: list[ModelToolCall] = []
        detected_role = "assistant"

        for raw_item in output_items:
            item_dict = _to_serializable_dict(raw_item)
            response_items.append(item_dict)
            item_type = item_dict.get("type")

            if item_type == "message":
                detected_role = item_dict.get("role", detected_role) or detected_role
                content_blocks = item_dict.get("content", []) or []

                text_parts: list[str] = []
                for block in content_blocks:
                    block_type = _item_get(block, "type")
                    if block_type in {"output_text", "text"}:
                        text = _item_get(block, "text", "")
                        if text:
                            text_parts.append(str(text))
                if text_parts:
                    collected_content.append("\n".join(text_parts))

            elif item_type == "reasoning":
                trace_parts: list[str] = []

                content_blocks = item_dict.get("content", []) or []
                for block in content_blocks:
                    text = _item_get(block, "text")
                    if text:
                        trace_parts.append(str(text))

                summaries = item_dict.get("summary", []) or []
                for summary in summaries:
                    text = _item_get(summary, "text")
                    if text:
                        trace_parts.append(str(text))

            elif item_type in {"function_call", "tool_call"}:
                maybe_call = cls._tool_call_from_response_item(item_dict)
                if maybe_call:
                    tool_calls.append(maybe_call)

        # Fallback to response.output_text helper if no message blocks were found
        if not collected_content:
            output_text = _item_get(response, "output_text", None)
            if output_text:
                collected_content.append(str(output_text))

        combined_content = "\n".join(part for part in collected_content if part).strip() or None

        reasoning_content: str | None = None
        # Collect reasoning content from reasoning items
        reasoning_parts: list[str] = []
        for raw_item in output_items:
            item_dict = _to_serializable_dict(raw_item)
            if item_dict.get("type") == "reasoning":
                content_blocks = item_dict.get("content", []) or []
                for block in content_blocks:
                    text = _item_get(block, "text")
                    if text:
                        reasoning_parts.append(str(text))

                summaries = item_dict.get("summary", []) or []
                for summary in summaries:
                    text = _item_get(summary, "text")
                    if text:
                        reasoning_parts.append(str(text))

        if reasoning_parts:
            reasoning_content = "\n".join(reasoning_parts)

        # Extract usage information from the response
        usage = None
        raw_usage = _item_get(response, "usage")
        if raw_usage is not None:
            usage = _to_serializable_dict(raw_usage)

        return cls(
            content=combined_content,
            tool_calls=tool_calls,
            role=detected_role,
            raw_message=response,
            response_items=response_items,
            reasoning_content=reasoning_content,
            usage=_normalize_usage(usage),
        )

    @staticmethod
    def _tool_call_from_response_item(item: Any) -> ModelToolCall | None:
        """Normalize Responses API function/tool call item into ModelToolCall."""

        call_id = _item_get(item, "call_id") or _item_get(item, "id")
        call_name = _item_get(item, "name")
        call_type = _item_get(item, "type", "function")

        function_payload = _item_get(item, "function")
        if function_payload:
            function_payload = _to_serializable_dict(function_payload)
            call_name = call_name or function_payload.get("name")
            arguments_payload = function_payload.get("arguments")
        else:
            arguments_payload = _item_get(item, "arguments")

        raw_arguments: str | None = None
        parsed_arguments: dict[str, Any] = {}

        if isinstance(arguments_payload, str):
            raw_arguments = arguments_payload
            payload_str = arguments_payload.strip()
            if payload_str:
                try:
                    parsed = json.loads(payload_str)
                    if isinstance(parsed, dict):
                        parsed_arguments = parsed
                    else:
                        parsed_arguments = {"_": parsed}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw_arguments": payload_str}
        elif isinstance(arguments_payload, dict):
            parsed_arguments = arguments_payload
            raw_arguments = json.dumps(arguments_payload, ensure_ascii=False)
        elif isinstance(arguments_payload, list):
            # Responses SDK may return arguments as structured content blocks
            try:
                flattened = "".join(str(_item_get(part, "text", "")) for part in arguments_payload).strip()
            except Exception:
                flattened = ""
            if flattened:
                raw_arguments = flattened
                try:
                    parsed = json.loads(flattened)
                    if isinstance(parsed, dict):
                        parsed_arguments = parsed
                    else:
                        parsed_arguments = {"_": parsed}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw_arguments": flattened}
        elif arguments_payload is not None:
            raw_arguments = json.dumps(arguments_payload, ensure_ascii=False)
            parsed_arguments = {"raw_arguments": raw_arguments}

        if not call_name:
            call_name = "unknown"

        return ModelToolCall(
            call_id=str(call_id) if call_id is not None else None,
            name=str(call_name),
            arguments=parsed_arguments,
            raw_arguments=raw_arguments,
            call_type=str(call_type) if call_type else "function",
            raw_call=item,
        )

    def has_content(self) -> bool:
        return bool(self.content and self.content.strip())

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def render_text(self) -> str:
        """Render the response as a human-readable string for logging."""
        parts: list[str] = []
        if self.reasoning_content:
            parts.append(f"[reasoning]\n{self.reasoning_content}")
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
        if self.response_items:
            message["response_items"] = self.response_items
        if self.reasoning_content:
            message["reasoning_content"] = self.reasoning_content
        return message

    def __str__(self) -> str:  # pragma: no cover - convenience
        rendered = self.render_text()
        return rendered if rendered else ""
