"""OpenAI Responses request payload serializers.

RFC-0014: UMP 到 Provider Payload 的统一序列化分层

Provides serializer helpers for converting OpenAI-chat-shaped message payloads
into OpenAI Responses API input items and tool definitions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast


def parse_openai_responses_image_part(part_map: Mapping[str, object]) -> dict[str, object] | None:
    """Convert a legacy ``image_url`` part to Responses ``input_image``.

    RFC-0014: OpenAI Responses 输入图片 part 序列化
    """

    image_url_any = part_map.get("image_url")
    url: str | None = None
    detail: str | None = None

    if isinstance(image_url_any, Mapping):
        image_url_map = cast(Mapping[str, object], image_url_any)
        url_any = image_url_map.get("url")
        if isinstance(url_any, str) and url_any.strip():
            url = url_any.strip()
        detail_any = image_url_map.get("detail")
        if isinstance(detail_any, str) and detail_any in {"low", "high", "auto"}:
            detail = detail_any
    elif isinstance(image_url_any, str) and image_url_any.strip():
        url = image_url_any.strip()

    if not url:
        return None

    payload: dict[str, object] = {"type": "input_image", "image_url": url}
    if detail and detail != "auto":
        payload["detail"] = detail
    return payload


def coerce_openai_responses_tool_output_text(output: Any) -> str:
    """Convert arbitrary tool output into Responses-compatible string.

    RFC-0014: OpenAI Responses tool output 文本降级
    """

    if output is None:
        return ""

    if isinstance(output, list):
        parts: list[str] = []
        output_list: list[Any] = cast(list[Any], output)
        for item in output_list:
            if isinstance(item, dict):
                item_dict = cast(dict[str, Any], item)
                item_type = str(item_dict.get("type") or "")
                if item_type in {"image_url", "input_image", "image"}:
                    parts.append("<image>")
                    continue
                list_text_value: Any = item_dict.get("text") or item_dict.get("content")
                if list_text_value is not None:
                    parts.append(str(list_text_value))
            else:
                parts.append(str(item))
        rendered = "\n".join(parts)
        return rendered if rendered.strip() else "<tool_output>"

    if isinstance(output, dict):
        output_dict = cast(dict[str, Any], output)
        dict_text_value: Any = output_dict.get("text") or output_dict.get("content")
        if dict_text_value is not None:
            return str(dict_text_value)
        return json.dumps(output, ensure_ascii=False)

    return str(output)


def collapse_openai_responses_message_content_to_text(content: Any) -> str:
    """Render structured message content into plain text.

    RFC-0014: OpenAI Responses 指令文本折叠
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in cast(list[Any], content):
            if isinstance(item, Mapping):
                item_map = cast(Mapping[str, Any], item)
                item_type = str(item_map.get("type") or "")
                if item_type in {"text", "input_text", "output_text", "summary_text"}:
                    text_value = item_map.get("text") or item_map.get("content")
                    if text_value is not None:
                        parts.append(str(text_value))
                elif item_type in {"image_url", "input_image", "image"}:
                    parts.append("<image>")
            elif item is not None:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, Mapping):
        content_map = cast(Mapping[str, Any], content)
        content_text = content_map.get("text") or content_map.get("content")
        if content_text is not None:
            return str(content_text)
        return json.dumps(dict(content_map), ensure_ascii=False)

    return str(content)


def ensure_openai_responses_reasoning_summary(reasoning_item: dict[str, Any]) -> list[dict[str, Any]]:
    """Guarantee reasoning items include a Responses-compatible summary list.

    RFC-0014: OpenAI Responses reasoning summary 规范化
    """

    summary_entries = reasoning_item.get("summary")
    sanitized: list[dict[str, Any]] = []

    if isinstance(summary_entries, list):
        summary_entries_list: list[Any] = cast(list[Any], summary_entries)
        for entry in summary_entries_list:
            if isinstance(entry, dict):
                entry_dict = cast(dict[str, Any], entry)
                summary_type: str = cast(str, entry_dict.get("type") or "summary_text")
                text = entry_dict.get("text")
                sanitized.append({"type": summary_type, "text": str(text or "")})
            elif entry is not None:
                sanitized.append({"type": "summary_text", "text": str(entry)})

    if not sanitized:
        fallback_text = collapse_openai_responses_message_content_to_text(reasoning_item.get("content"))
        sanitized.append({"type": "summary_text", "text": fallback_text or ""})

    return sanitized


def sanitize_openai_responses_items_for_input(
    items: list[Any],
    *,
    drop_ephemeral_ids: bool = False,
) -> list[dict[str, Any]]:
    """Strip response-only fields that the Responses API rejects on input.

    RFC-0014: OpenAI Responses 输入 item 清洗
    """

    sanitized: list[dict[str, Any]] = []

    def _is_multimodal_output_list(value: Any) -> bool:
        if not isinstance(value, list):
            return False
        for item in cast(list[Any], value):
            if not isinstance(item, Mapping):
                continue
            item_dict = cast(Mapping[str, Any], item)
            item_type = str(item_dict.get("type") or "")
            if item_type in {"input_image", "image", "image_url", "input_file"}:
                return True
        return False

    for item in items:
        if isinstance(item, Mapping):
            item_mapping = cast(Mapping[str, Any], item)
            item_copy: dict[str, Any] = dict(item_mapping)
            item_copy.pop("status", None)
            if drop_ephemeral_ids:
                item_copy.pop("id", None)
            item_type = item_copy.get("type")
            if item_type == "message":
                item_copy.pop("status", None)
                phase = item_copy.get("phase")
                if not isinstance(phase, str) or not phase:
                    item_copy.pop("phase", None)
            elif item_type == "function_call_output":
                output_value = item_copy.get("output")
                if _is_multimodal_output_list(output_value):
                    cleaned: list[dict[str, Any]] = []
                    for out_item in cast(list[Any], output_value):
                        if not isinstance(out_item, Mapping):
                            continue
                        out_dict = dict(cast(Mapping[str, Any], out_item))
                        out_type = str(out_dict.get("type") or "")
                        if out_type == "output_text":
                            cleaned.append({"type": "input_text", "text": str(out_dict.get("text") or "")})
                            continue
                        if out_type == "image_url":
                            image_url_any = out_dict.get("image_url")
                            url: str | None = None
                            detail: str | None = None
                            if isinstance(image_url_any, Mapping):
                                image_url_map = cast(Mapping[str, Any], image_url_any)
                                url_any = image_url_map.get("url")
                                if isinstance(url_any, str) and url_any.strip():
                                    url = url_any.strip()
                                detail_any = image_url_map.get("detail")
                                if isinstance(detail_any, str) and detail_any in {"low", "high", "auto"}:
                                    detail = detail_any
                            elif isinstance(image_url_any, str) and image_url_any.strip():
                                url = image_url_any.strip()
                            if url:
                                payload: dict[str, Any] = {"type": "input_image", "image_url": url}
                                if detail and detail != "auto":
                                    payload["detail"] = detail
                                cleaned.append(payload)
                            continue
                        cleaned.append(out_dict)
                    item_copy["output"] = cleaned
                else:
                    item_copy["output"] = coerce_openai_responses_tool_output_text(output_value)
            elif item_type == "reasoning":
                item_copy.pop("id", None)
                if not item_copy.get("encrypted_content"):
                    item_copy["summary"] = ensure_openai_responses_reasoning_summary(item_copy)
                else:
                    item_copy.setdefault("summary", [])
                item_copy.pop("content", None)
        else:
            sanitized.append(item)
            continue

        sanitized.append(item_copy)

    return sanitized


def reconstruct_openai_responses_reasoning_items_from_message(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild Responses reasoning items from assistant message fields.

    RFC-0014: 非 Responses 来源 reasoning replay 重建
    """

    reasoning_content = message.get("reasoning_content")
    reasoning_redacted_data = message.get("reasoning_redacted_data")

    has_reasoning_text = isinstance(reasoning_content, str)
    has_redacted_reasoning = isinstance(reasoning_redacted_data, str) and bool(reasoning_redacted_data)
    if not has_reasoning_text and not has_redacted_reasoning:
        return []

    reasoning_item: dict[str, Any] = {"type": "reasoning"}
    if has_reasoning_text:
        reasoning_item["content"] = [{"type": "text", "text": str(reasoning_content)}]
    if has_redacted_reasoning:
        reasoning_item["encrypted_content"] = str(reasoning_redacted_data)
        if has_reasoning_text:
            reasoning_item["summary"] = [{"type": "summary_text", "text": str(reasoning_content)}]

    return sanitize_openai_responses_items_for_input([reasoning_item])


def prepare_openai_responses_api_input(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
    """Convert internal message representation into Responses input items.

    RFC-0014: OpenAI Responses provider payload serializer
    """

    prepared: list[dict[str, Any]] = []
    instructions: list[str] = []

    for message in messages:
        response_items = message.get("response_items")
        if response_items:
            prepared.extend(sanitize_openai_responses_items_for_input(response_items, drop_ephemeral_ids=True))
            continue

        role = message.get("role", "user")
        content = message.get("content", "") or ""

        if role == "tool":
            tool_call_id = message.get("tool_call_id")
            if tool_call_id:
                output_value: Any = None
                if isinstance(content, list):
                    out_items: list[dict[str, Any]] = []
                    for part in cast(list[Any], content):
                        if not isinstance(part, Mapping):
                            continue
                        part_map = cast(Mapping[str, Any], part)
                        part_type = str(part_map.get("type") or "")

                        if part_type in {"text", "output_text", "input_text"}:
                            text_val: Any = part_map.get("text") or part_map.get("content")
                            out_items.append({"type": "input_text", "text": str(text_val or "")})
                            continue

                        if part_type == "image_url":
                            img_part = parse_openai_responses_image_part(part_map)
                            if img_part is not None:
                                out_items.append(cast(dict[str, Any], img_part))
                            continue

                    if any(item.get("type") == "input_image" for item in out_items):
                        output_value = out_items

                if output_value is None:
                    output_value = coerce_openai_responses_tool_output_text(content)

                prepared.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": output_value,
                    },
                )
            continue

        if role == "system":
            instruction_text = collapse_openai_responses_message_content_to_text(content)
            if instruction_text:
                instructions.append(instruction_text)
            continue

        text_type = "input_text" if role == "user" else "output_text"

        content_parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for part in cast(list[Any], content):
                if not isinstance(part, Mapping):
                    continue
                part_map = cast(Mapping[str, Any], part)
                part_type = str(part_map.get("type") or "")

                if part_type in {"text", "input_text", "output_text"}:
                    msg_text_val: object = part_map.get("text") or part_map.get("content")
                    if msg_text_val:
                        content_parts.append({"type": text_type, "text": str(msg_text_val)})
                elif part_type == "image_url":
                    img_part = parse_openai_responses_image_part(part_map)
                    if img_part is not None:
                        content_parts.append(cast(dict[str, Any], img_part))
        elif content:
            content_parts.append({"type": text_type, "text": str(content)})

        message_item: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": content_parts,
        }
        phase = message.get("phase")
        if role == "assistant" and isinstance(phase, str) and phase:
            message_item["phase"] = phase

        # RFC-0014: reasoning 必须在 assistant message 和 function_call 之前，
        # 与 Responses API 原生输出顺序一致：reasoning → message → function_call
        reasoning_items = message.get("reasoning")
        if reasoning_items:
            if isinstance(reasoning_items, list):
                prepared.extend(sanitize_openai_responses_items_for_input(cast(list[Any], reasoning_items)))
        elif role == "assistant":
            prepared.extend(reconstruct_openai_responses_reasoning_items_from_message(message))

        # RFC-0014: 纯 tool_call 的 assistant 不需要空 message item，
        # Responses API 原生输出为 reasoning → function_call（无 message）
        if content_parts or role != "assistant":
            prepared.append(message_item)

        tool_calls_payload_raw = message.get("tool_calls")
        if isinstance(tool_calls_payload_raw, list):
            tool_calls_payload_list: list[Any] = cast(list[Any], tool_calls_payload_raw)
            for tc in tool_calls_payload_list:
                if not isinstance(tc, Mapping):
                    continue
                typed_tool_call: Mapping[str, Any] = cast(Mapping[str, Any], tc)
                tool_call_dict: dict[str, Any] = {}
                for key_any, value_any in typed_tool_call.items():
                    key: str = str(key_any)
                    value: Any = value_any
                    tool_call_dict[key] = value

                function_raw = tool_call_dict.get("function")
                if not isinstance(function_raw, dict):
                    raise ValueError(f"Tool call function must be a dict, got {type(function_raw).__name__}")
                function = cast(dict[str, Any], function_raw)
                if not isinstance(function.get("name"), str) or not function["name"]:
                    raise ValueError("Tool call function must contain a non-empty string name")
                if function.get("arguments") is None:
                    raise ValueError("Tool call function must contain arguments")
                raw_arguments = function.get("arguments")
                if isinstance(raw_arguments, dict):
                    raw_arguments = json.dumps(raw_arguments, ensure_ascii=False)

                prepared.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call_dict.get("id"),
                        "name": function.get("name"),
                        "arguments": raw_arguments,
                    },
                )

    joined_instructions = "\n\n".join(part.strip() for part in instructions if part.strip()) or None
    return prepared, joined_instructions


def normalize_openai_responses_api_tools(tools: list[Any]) -> list[dict[str, Any]]:
    """Ensure tool definitions align with the Responses API schema.

    RFC-0014: OpenAI Responses tools payload 规范化
    """

    normalized: list[dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, Mapping):
            normalized.append(tool)
            continue

        tool_mapping = cast(Mapping[str, Any], tool)
        tool_dict: dict[str, Any] = dict(tool_mapping)
        tool_type = tool_dict.get("type")

        if tool_type == "function":
            function_spec = cast(dict[str, Any] | None, tool_dict.get("function"))

            if isinstance(function_spec, dict):
                name = function_spec.get("name")
                description = function_spec.get("description")
                parameters = function_spec.get("parameters")
                strict = function_spec.get("strict")

                if name and not tool_dict.get("name"):
                    tool_dict["name"] = name
                if description and not tool_dict.get("description"):
                    tool_dict["description"] = description
                if parameters is not None and "parameters" not in tool_dict:
                    tool_dict["parameters"] = parameters
                if strict is not None and "strict" not in tool_dict:
                    tool_dict["strict"] = strict

            if tool_dict.get("name"):
                tool_dict.pop("function", None)

        normalized.append(tool_dict)

    return normalized
