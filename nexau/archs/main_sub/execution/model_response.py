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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from nexau.core.messages import Message

JsonDict = dict[str, Any]


def _empty_json_dict() -> JsonDict:
    return {}


def _empty_tool_call_list() -> list[ModelToolCall]:
    return []


def _empty_json_dict_list() -> list[JsonDict]:
    return []


def _item_get(item: object, key: str, default: Any = None) -> Any:
    """Best-effort attribute/dict access helper for SDK objects."""

    if isinstance(item, Mapping):
        mapping_item = cast(Mapping[str, Any], item)
        return mapping_item.get(key, default)

    if hasattr(item, key):
        return getattr(item, key)

    return default


def _to_serializable_dict(payload: Any) -> dict[str, Any]:
    """Attempt to coerce SDK payloads into simple dicts for parsing."""

    if isinstance(payload, dict):
        return cast(JsonDict, payload)

    # Prefer model_dump if available (pydantic models)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            # See `nexau.archs.main_sub.execution.llm_caller._to_serializable_dict`:
            # some SDK models can trigger Pydantic's serializer warnings during dump.
            try:
                return cast(dict[str, Any], model_dump(mode="json", warnings=False))
            except TypeError:
                try:
                    return cast(dict[str, Any], model_dump(warnings=False))
                except TypeError:
                    return cast(dict[str, Any], model_dump())
        except Exception:  # pragma: no cover - fallback
            pass

    # Fallback to attribute introspection (best-effort, non-recursive)
    result: JsonDict = {}
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


def _coerce_usage(usage: Any) -> dict[str, Any] | None:
    """Best-effort conversion of SDK usage payloads (or mocks) into a dict."""

    if usage is None:
        return None

    if isinstance(usage, Mapping):
        try:
            return dict(cast(Mapping[str, Any], usage))
        except Exception:
            # Fall back to a looser conversion path below
            pass

    if isinstance(usage, dict):  # type: ignore[redundant-expr]
        return cast(dict[str, Any], usage)

    try:
        return _to_serializable_dict(usage)
    except Exception:
        return None


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
    usage = _coerce_usage(usage)
    if usage is None:
        return None

    normalized: dict[str, Any] = {}

    # Handle input tokens
    try:
        if "input_tokens" in usage:
            normalized["input_tokens"] = usage["input_tokens"]
        elif "prompt_tokens" in usage:
            normalized["input_tokens"] = usage["prompt_tokens"]
        else:
            normalized["input_tokens"] = 0
    except Exception:
        normalized["input_tokens"] = 0

    # Handle reasoning tokens (for models that support it)
    try:
        if "reasoning_tokens" in usage:
            normalized["reasoning_tokens"] = usage["reasoning_tokens"]
        else:
            normalized["reasoning_tokens"] = 0
    except Exception:
        normalized["reasoning_tokens"] = 0

    # Handle completion/output tokens
    try:
        if "completion_tokens" in usage:
            normalized["completion_tokens"] = usage["completion_tokens"]
        elif "output_tokens" in usage:
            normalized["completion_tokens"] = usage["output_tokens"]
        else:
            normalized["completion_tokens"] = 0
    except Exception:
        normalized["completion_tokens"] = 0

    # Handle total tokens
    try:
        if "total_tokens" in usage:
            normalized["total_tokens"] = usage["total_tokens"]
        else:
            normalized["total_tokens"] = normalized["input_tokens"] + normalized["reasoning_tokens"] + normalized["completion_tokens"]
    except Exception:
        normalized["total_tokens"] = normalized["input_tokens"] + normalized["reasoning_tokens"] + normalized["completion_tokens"]

    return normalized


@dataclass
class ModelToolCall:
    """Represents a tool call emitted by the model in a model-agnostic format."""

    call_id: str | None
    name: str
    arguments: JsonDict = field(default_factory=_empty_json_dict)
    raw_arguments: str | None = None
    call_type: str = "function"
    raw_call: Any = None

    @classmethod
    def from_openai(cls, call: Any) -> ModelToolCall:
        """Create a ModelToolCall from an OpenAI tool call payload."""
        if call is None:
            raise ValueError("call payload cannot be None")

        # Support both dict-style and attribute-style payloads
        call_id: str | None = None
        if isinstance(call, dict):
            call_dict = cast(JsonDict, call)
            raw_id: Any = call_dict.get("id")
            call_id = str(raw_id) if raw_id is not None else None
            call_type_raw: Any = call_dict.get("type")
            function: Any = call_dict.get("function")
        else:
            call_id_attr = getattr(call, "id", None)
            call_id = str(call_id_attr) if call_id_attr is not None else None
            call_type_raw = getattr(call, "type", "function")
            function = getattr(call, "function", None)

        call_type: str = str(call_type_raw) if call_type_raw else "function"

        if function is None:
            raise ValueError("OpenAI tool call payload missing function block")

        if isinstance(function, dict):
            func_dict = cast(JsonDict, function)
            name = func_dict.get("name")
            raw_arguments = func_dict.get("arguments")
        else:
            name = getattr(function, "name", None)
            raw_arguments = getattr(function, "arguments", None)

        if not name:
            raise ValueError("OpenAI tool call function block missing name")

        parsed_arguments: JsonDict = {}
        if isinstance(raw_arguments, str):
            raw_arguments_str = raw_arguments.strip()
            if raw_arguments_str:
                try:
                    parsed = json.loads(raw_arguments_str)
                    if isinstance(parsed, dict):
                        parsed_arguments = cast(JsonDict, parsed)
                    else:
                        parsed_arguments = {"_": parsed}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw_arguments": raw_arguments}
        elif raw_arguments is not None:
            if isinstance(raw_arguments, dict):
                parsed_arguments = cast(JsonDict, raw_arguments)
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
    tool_calls: list[ModelToolCall] = field(default_factory=_empty_tool_call_list)
    role: str = "assistant"
    raw_message: Any = None
    response_items: list[JsonDict] = field(default_factory=_empty_json_dict_list)
    reasoning_content: str | None = None
    """A description of the chain of thought used by a reasoning model while generating a response.

    Be sure to include these items in your input to the Responses API for subsequent turns of a
    conversation if you are manually managing context.
    """
    usage: JsonDict | None = None
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
        self.tool_calls = list(self.tool_calls)
        self.response_items = list(self.response_items)
        if self.usage is None:
            self.usage = _empty_json_dict()

    @classmethod
    def from_openai_message(cls, message: Any, usage: dict[str, Any] | None = None) -> ModelResponse:
        """Create a ModelResponse from an OpenAI chat completion message.

        Args:
            message: The message object from OpenAI response
            usage: Optional usage statistics from the response object
        """
        if message is None:
            raise ValueError("message cannot be None")

        message_dict: JsonDict | None = cast(JsonDict, message) if isinstance(message, dict) else None
        message_obj: Any = cast(Any, message)

        content = getattr(message_obj, "content", None)
        if content is None and message_dict is not None:
            content = message_dict.get("content")
        # openai-beta returns list of content parts; join if needed
        if isinstance(content, list):
            # Attempt to join textual content pieces
            content_list: list[Any] = cast(list[Any], content)
            text_parts: list[str] = []
            for part in content_list:
                if isinstance(part, dict):
                    part_dict = cast(dict[str, Any], part)
                    text_val = part_dict.get("text", "")
                    text_parts.append(text_val if isinstance(text_val, str) else str(text_val))
                else:
                    text_parts.append(str(part))
            content = "".join(text_parts)

        raw_tool_calls: list[Any] | None = getattr(message_obj, "tool_calls", None)
        if raw_tool_calls is None and message_dict is not None:
            tool_calls_value = message_dict.get("tool_calls")
            raw_tool_calls = cast(list[Any], tool_calls_value) if isinstance(tool_calls_value, list) else None

        tool_calls: list[ModelToolCall] = []
        if raw_tool_calls:
            for call in raw_tool_calls:
                try:
                    tool_calls.append(ModelToolCall.from_openai(call))
                except Exception:
                    # If parsing fails, create minimal placeholder
                    tool_calls.append(
                        ModelToolCall(
                            call_id=getattr(call, "id", None) if not isinstance(call, dict) else cast(dict[str, Any], call).get("id"),
                            name="unknown",
                            arguments={"raw_call": call},
                            raw_arguments=None,
                            call_type=getattr(call, "type", "function")
                            if not isinstance(call, dict)
                            else cast(dict[str, Any], call).get("type", "function"),
                            raw_call=call,
                        ),
                    )

        # Extract reasoning_content if available (for models like kimi-k2-thinking)
        reasoning_content = None
        if hasattr(message_obj, "reasoning_content"):
            reasoning_content = getattr(message_obj, "reasoning_content")
        elif message_dict is not None and "reasoning_content" in message_dict:
            reasoning_content = message_dict["reasoning_content"]

        # Extract usage information if available in the message itself
        message_usage: JsonDict | None = None
        if hasattr(message_obj, "usage"):
            message_usage = _to_serializable_dict(getattr(message_obj, "usage"))
        elif message_dict is not None and "usage" in message_dict:
            raw_usage: Any = message_dict.get("usage")
            message_usage = _to_serializable_dict(raw_usage) if raw_usage is not None else None

        # Prefer explicitly passed usage over message-embedded usage
        final_usage = usage if usage is not None else message_usage

        role = getattr(message_obj, "role", "assistant") if message_dict is None else message_dict.get("role", "assistant")
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

        message_dict: JsonDict | None = cast(JsonDict, message) if isinstance(message, dict) else None
        message_obj: Any = cast(Any, message)

        content_blocks: list[Any] | None = getattr(message_obj, "content", None)
        if content_blocks is None and message_dict is not None:
            raw_blocks = message_dict.get("content")
            content_blocks = cast(list[Any], raw_blocks) if raw_blocks is not None else None
        if content_blocks is None:
            content_blocks = []

        # Extract text content from content blocks
        text_parts: list[str] = []
        raw_tool_calls: list[Any] = []

        for block in content_blocks:
            if isinstance(block, dict):
                block_dict = cast(JsonDict, block)
                block_type_val: str | None = cast(str | None, block_dict.get("type"))
                text_val: Any = block_dict.get("text")
            else:
                block_type_val = cast(str | None, getattr(block, "type", None))
                text_val = getattr(block, "text", None)
            block_type = str(block_type_val) if block_type_val is not None else None
            if block_type == "text":
                if isinstance(text_val, str):
                    text_parts.append(text_val)
            elif block_type == "tool_use":
                raw_tool_calls.append(block)

        content = "\n".join(text_parts) if text_parts else ""

        tool_calls: list[ModelToolCall] = []
        if raw_tool_calls:
            for call in raw_tool_calls:
                # Handle both dict and object (ToolUseBlock) cases
                if isinstance(call, dict):
                    call_dict = cast(JsonDict, call)
                    call_id_val = call_dict.get("id")
                    call_id = str(call_id_val) if call_id_val is not None else None
                    name = call_dict.get("name")
                    arguments_raw: Any = call_dict.get("input", {}) or {}
                    call_type_raw: Any = call_dict.get("type", "function")
                else:
                    call_id_attr = getattr(call, "id", None)
                    call_id = str(call_id_attr) if call_id_attr is not None else None
                    name = getattr(call, "name", None)
                    arguments_raw = getattr(call, "input", {}) or {}
                    call_type_raw = getattr(call, "type", "function")

                name_value = str(name) if name else ""
                call_type = str(call_type_raw) if call_type_raw else "function"
                arguments_dict = cast(JsonDict, arguments_raw) if isinstance(arguments_raw, dict) else {"input": arguments_raw}
                tool_calls.append(
                    ModelToolCall(
                        call_id=call_id,
                        name=name_value,
                        arguments=arguments_dict,
                        raw_arguments=json.dumps(arguments_dict, ensure_ascii=False),
                        call_type=call_type,
                        raw_call=call,
                    ),
                )

        # Extract usage information if available in the message itself
        message_usage: JsonDict | None = None
        if hasattr(message_obj, "usage"):
            message_usage = _to_serializable_dict(getattr(message_obj, "usage"))
        elif message_dict is not None and "usage" in message_dict:
            raw_usage: Any = message_dict.get("usage")
            message_usage = _to_serializable_dict(raw_usage) if raw_usage is not None else None

        # Prefer explicitly passed usage over message-embedded usage
        final_usage = usage if usage is not None else message_usage

        role = getattr(message_obj, "role", "assistant") if message_dict is None else message_dict.get("role", "assistant")
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

        output_items_raw: Any = _item_get(response, "output", []) or []
        output_items: list[Any] = cast(list[Any], output_items_raw) if isinstance(output_items_raw, list) else []
        response_items: list[JsonDict] = []
        collected_content: list[str] = []
        tool_calls: list[ModelToolCall] = []
        detected_role = "assistant"

        for raw_item in output_items:
            item_dict = _to_serializable_dict(raw_item)
            response_items.append(item_dict)
            item_type = item_dict.get("type")

            if item_type == "message":
                detected_role_raw = item_dict.get("role", detected_role)
                detected_role = str(detected_role_raw) if detected_role_raw is not None else detected_role
                msg_content_blocks_raw: Any = item_dict.get("content", []) or []
                msg_content_blocks: list[Any] = cast(list[Any], msg_content_blocks_raw) if isinstance(msg_content_blocks_raw, list) else []

                text_parts: list[str] = []
                for block in msg_content_blocks:
                    block_type = _item_get(block, "type")
                    if block_type in {"output_text", "text"}:
                        text = _item_get(block, "text", "")
                        if text:
                            text_parts.append(str(text))
                if text_parts:
                    collected_content.append("\n".join(text_parts))

            elif item_type == "reasoning":
                trace_parts: list[str] = []

                reasoning_content_blocks_raw: Any = item_dict.get("content", []) or []
                reasoning_content_blocks: list[Any] = (
                    cast(list[Any], reasoning_content_blocks_raw) if isinstance(reasoning_content_blocks_raw, list) else []
                )
                for block in reasoning_content_blocks:
                    text = _item_get(block, "text")
                    if text:
                        trace_parts.append(str(text))

                reasoning_summaries_raw: Any = item_dict.get("summary", []) or []
                reasoning_summaries: list[Any] = (
                    cast(list[Any], reasoning_summaries_raw) if isinstance(reasoning_summaries_raw, list) else []
                )
                for summary in reasoning_summaries:
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
                content_blocks_reasoning_raw: Any = item_dict.get("content", []) or []
                content_blocks_reasoning: list[Any] = (
                    cast(list[Any], content_blocks_reasoning_raw) if isinstance(content_blocks_reasoning_raw, list) else []
                )
                for block in content_blocks_reasoning:
                    text = _item_get(block, "text")
                    if text:
                        reasoning_parts.append(str(text))

                reasoning_summaries_raw = item_dict.get("summary", []) or []
                reasoning_summaries_list: list[Any] = (
                    cast(list[Any], reasoning_summaries_raw) if isinstance(reasoning_summaries_raw, list) else []
                )
                for summary in reasoning_summaries_list:
                    text = _item_get(summary, "text")
                    if text:
                        reasoning_parts.append(str(text))

        if reasoning_parts:
            reasoning_content = "\n".join(reasoning_parts)

        # Extract usage information from the response
        usage: JsonDict | None = None
        raw_usage: Any = _item_get(response, "usage")
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

        call_id_raw: Any = _item_get(item, "call_id") or _item_get(item, "id")
        call_id = str(call_id_raw) if call_id_raw is not None else None
        call_name_raw: Any = _item_get(item, "name")
        call_name = str(call_name_raw) if call_name_raw is not None else None
        call_type_raw: Any = _item_get(item, "type", "function")
        call_type = str(call_type_raw) if call_type_raw is not None else "function"

        function_payload = _item_get(item, "function")
        if function_payload:
            function_payload = _to_serializable_dict(function_payload)
            call_name = call_name or function_payload.get("name")
            arguments_payload = function_payload.get("arguments")
        else:
            arguments_payload = _item_get(item, "arguments")

        raw_arguments: str | None = None
        parsed_arguments: JsonDict = {}

        if isinstance(arguments_payload, str):
            raw_arguments = arguments_payload
            payload_str = arguments_payload.strip()
            if payload_str:
                try:
                    parsed = json.loads(payload_str)
                    if isinstance(parsed, dict):
                        parsed_arguments = cast(JsonDict, parsed)
                    else:
                        parsed_arguments = {"_": parsed}
                except json.JSONDecodeError:
                    parsed_arguments = {"raw_arguments": payload_str}
        elif isinstance(arguments_payload, dict):
            parsed_arguments = cast(JsonDict, arguments_payload)
            raw_arguments = json.dumps(arguments_payload, ensure_ascii=False)
        elif isinstance(arguments_payload, list):
            # Responses SDK may return arguments as structured content blocks
            flattened_parts: list[str] = []
            arguments_payload_list: list[Any] = cast(list[Any], arguments_payload)
            try:
                for part_any in arguments_payload_list:
                    text_part = _item_get(part_any, "text", "")
                    if isinstance(text_part, str) and text_part:
                        flattened_parts.append(text_part)
                    elif text_part:
                        flattened_parts.append(str(text_part))
                flattened = "".join(flattened_parts).strip()
            except Exception:
                flattened = ""
            if flattened:
                raw_arguments = flattened
                try:
                    parsed = json.loads(flattened)
                    if isinstance(parsed, dict):
                        parsed_arguments = cast(JsonDict, parsed)
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
        if self.has_content() and self.content:
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

    def to_ump_message(self) -> Message:
        """Convert response into a vendor-agnostic UMP `Message`."""

        from nexau.core.messages import Message, ReasoningBlock, Role, TextBlock, ToolUseBlock

        try:
            role = Role(self.role)
        except Exception:
            role = Role.ASSISTANT

        blocks: list[Any] = []
        if self.content:
            blocks.append(TextBlock(text=self.content))

        if self.reasoning_content:
            blocks.append(ReasoningBlock(text=self.reasoning_content))

        for call in self.tool_calls:
            blocks.append(
                ToolUseBlock(
                    id=call.call_id or "tool_call",
                    name=call.name,
                    input=call.arguments,
                    raw_input=call.raw_arguments,
                ),
            )

        msg = Message(role=role, content=blocks)
        if self.response_items:
            msg.metadata["response_items"] = self.response_items
        return msg

    def __str__(self) -> str:  # pragma: no cover - convenience
        rendered = self.render_text()
        return rendered if rendered else ""
