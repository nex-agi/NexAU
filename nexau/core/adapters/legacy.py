"""Legacy message conversions.

These helpers bridge NexAU's historical ``list[dict[str, Any]]`` chat history into
UMP messages, and back into OpenAI Chat-Completions-shaped dicts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, cast

from nexau.core.messages import (
    ImageBlock,
    Message,
    ReasoningBlock,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _maybe_parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_arguments": value}
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
        return {"_": parsed}
    return {"_": value}


def messages_from_legacy_openai_chat(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert legacy OpenAI-style chat dicts into UMP messages.

    Supported shapes:
    - {"role": "...", "content": "..."}
    - assistant tool calls: {"tool_calls": [{"id","type","function":{"name","arguments"}}]}
    - tool results: {"role":"tool","tool_call_id":"...","content":"..."}
    - anthropic-like content blocks in "content": [{"type": "...", ...}] (best-effort)
    - responses-api artifacts: "response_items", "reasoning_content" (stored in metadata / ReasoningBlock)
    """

    result: list[Message] = []
    for idx, raw in enumerate(messages or []):
        role_raw = (raw.get("role") or "user").lower()
        try:
            role = Role(role_raw)
        except Exception:
            logger.warning("Unknown role %r in legacy chat message at index=%s; coercing to %s", role_raw, idx, Role.USER.value)
            role = Role.USER

        content_blocks: list[Any] = []
        content = raw.get("content")

        # Accept structured content lists (Responses API / Anthropic output)
        if isinstance(content, list):
            for part in cast(list[Any], content):
                if not isinstance(part, Mapping):
                    text = _coerce_str(part)
                    if text:
                        content_blocks.append(TextBlock(text=text))
                    continue
                part_dict = cast(Mapping[str, Any], part)
                part_type = str(part_dict.get("type") or "")
                if part_type in {"text", "output_text", "input_text"}:
                    text = _coerce_str(part_dict.get("text") or part_dict.get("content"))
                    if text:
                        content_blocks.append(TextBlock(text=text))
                elif part_type in {"image", "image_url"}:
                    url = part_dict.get("url")
                    if url is None and isinstance(part_dict.get("image_url"), Mapping):
                        url = cast(Mapping[str, Any], part_dict.get("image_url")).get("url")
                    b64 = part_dict.get("base64")
                    mime = _coerce_str(part_dict.get("mime_type") or "image/jpeg")
                    try:
                        content_blocks.append(ImageBlock(url=cast(str | None, url), base64=cast(str | None, b64), mime_type=mime))
                    except Exception:
                        # Best-effort: drop invalid image blocks
                        pass
                elif part_type == "tool_use":
                    content_blocks.append(
                        ToolUseBlock(
                            id=_coerce_str(part_dict.get("id")),
                            name=_coerce_str(part_dict.get("name")),
                            input=_maybe_parse_json(part_dict.get("input")),
                            raw_input=None,
                        ),
                    )
                elif part_type == "tool_result":
                    content_blocks.append(
                        ToolResultBlock(
                            tool_use_id=_coerce_str(part_dict.get("tool_use_id") or part_dict.get("tool_call_id")),
                            content=_coerce_str(part_dict.get("content")),
                            is_error=bool(part_dict.get("is_error") or part_dict.get("error")),
                        ),
                    )
                elif part_type == "reasoning":
                    text = _coerce_str(part_dict.get("text") or part_dict.get("content"))
                    if text:
                        content_blocks.append(ReasoningBlock(text=text))
                else:
                    # Unknown structured part; store as text so we don't drop user-visible info
                    text = _coerce_str(part_dict.get("text") or part_dict.get("content"))
                    if text:
                        content_blocks.append(TextBlock(text=text))
        else:
            text = _coerce_str(content)
            if text:
                content_blocks.append(TextBlock(text=text))

        metadata: dict[str, Any] = {}
        if "response_items" in raw:
            metadata["response_items"] = raw.get("response_items")
        if "reasoning" in raw:
            metadata["reasoning"] = raw.get("reasoning")

        # Prefer explicit reasoning_content into a dedicated block (store-only)
        reasoning_content = raw.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            content_blocks.append(ReasoningBlock(text=reasoning_content))

        # Assistant tool calls (OpenAI Chat Completions style)
        tool_calls_raw = raw.get("tool_calls")
        if isinstance(tool_calls_raw, list):
            for call_any in cast(list[Any], tool_calls_raw):
                call_dict: dict[str, Any]
                if isinstance(call_any, Mapping):
                    call_dict = dict(cast(Mapping[str, Any], call_any))
                else:
                    call_dict = {}

                function_any = call_dict.get("function")
                func_dict: dict[str, Any]
                if isinstance(function_any, Mapping):
                    func_dict = dict(cast(Mapping[str, Any], function_any))
                else:
                    func_dict = {}

                tool_id = _coerce_str(call_dict.get("id"))
                tool_name = _coerce_str(func_dict.get("name"))
                args_raw: Any = func_dict.get("arguments")
                content_blocks.append(
                    ToolUseBlock(
                        id=tool_id or "tool_call",
                        name=tool_name or "unknown",
                        input=_maybe_parse_json(args_raw),
                        raw_input=args_raw if isinstance(args_raw, str) else None,
                    ),
                )

        # Tool result messages (OpenAI tool role)
        if role == Role.TOOL:
            tool_call_id = _coerce_str(raw.get("tool_call_id"))
            # Replace textual content blocks with an explicit tool_result block
            tool_text = "".join(b.text for b in content_blocks if isinstance(b, TextBlock))
            content_blocks = [
                ToolResultBlock(tool_use_id=tool_call_id, content=tool_text),
            ]

        result.append(Message(role=role, content=content_blocks, metadata=metadata))
    return result


def messages_to_legacy_openai_chat(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert UMP messages into OpenAI Chat-Completions-shaped dicts (legacy NexAU format)."""

    output: list[dict[str, Any]] = []
    for msg in messages or []:
        role = msg.role.value

        # Tool results become role=tool messages (OpenAI expected)
        if role == Role.TOOL.value:
            # Prefer the first ToolResultBlock
            tr = next((b for b in msg.content if isinstance(b, ToolResultBlock)), None)
            if tr is None:
                output.append({"role": "tool", "content": msg.get_text_content()})
            else:
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr.tool_use_id,
                        "content": tr.content,
                    },
                )
            continue

        entry: dict[str, Any] = {"role": role, "content": ""}

        # Preserve response-items artifacts if present; they are used by Responses API input reconstruction.
        if "response_items" in msg.metadata:
            entry["response_items"] = msg.metadata["response_items"]
        if "reasoning" in msg.metadata:
            entry["reasoning"] = msg.metadata["reasoning"]

        has_images = any(isinstance(b, ImageBlock) for b in msg.content)
        text_parts: list[str] = []
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []

        for block in msg.content:
            if isinstance(block, TextBlock):
                if has_images:
                    content_parts.append({"type": "text", "text": block.text})
                else:
                    text_parts.append(block.text)
            elif isinstance(block, ReasoningBlock):
                reasoning_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.raw_input if block.raw_input is not None else json.dumps(block.input, ensure_ascii=False),
                        },
                    },
                )
            elif isinstance(block, ToolResultBlock):
                # ToolResultBlocks should be represented as role=tool messages in legacy format.
                output.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": block.content,
                    },
                )
            elif isinstance(block, ImageBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
                # OpenAI Chat Completions supports multi-part content with images:
                # {"content": [{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"..."}}]}
                if block.url:
                    content_parts.append({"type": "image_url", "image_url": {"url": block.url}})
                else:
                    # Use a data URL so downstream OpenAI-compatible clients can send the image.
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.base64}"}})

        entry["content"] = content_parts if has_images else "".join(text_parts)
        if reasoning_parts:
            entry["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        output.append(entry)

    return output
