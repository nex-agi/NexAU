"""Legacy message conversions.

These helpers bridge NexAU's historical ``list[dict[str, Any]]`` chat history into
UMP messages.
"""

from __future__ import annotations

import json
import logging
import re
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
    coerce_tool_result_content,
    parse_base64_data_url,
)

logger = logging.getLogger(__name__)

_INJECTED_TOOL_IMAGES_RE = re.compile(r"^\s*Images returned by tool call\s+(?P<id>.+?)\s*:\s*$")


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
                        content_blocks.append(TextBlock(text=text))  # pyright: ignore[reportCallIssue]
                    continue
                part_dict = cast(Mapping[str, Any], part)
                part_type = str(part_dict.get("type") or "")
                if part_type in {"text", "output_text", "input_text"}:
                    text = _coerce_str(part_dict.get("text") or part_dict.get("content"))
                    if text:
                        content_blocks.append(TextBlock(text=text))  # pyright: ignore[reportCallIssue]
                elif part_type in {"image", "image_url", "input_image"}:
                    url = part_dict.get("url")
                    if url is None and isinstance(part_dict.get("image_url"), Mapping):
                        url = cast(Mapping[str, Any], part_dict.get("image_url")).get("url")
                    if url is None and part_type == "input_image":
                        url = part_dict.get("image_url")
                    b64 = part_dict.get("base64")
                    mime = _coerce_str(part_dict.get("mime_type") or "image/jpeg")
                    detail_any: Any = part_dict.get("detail")
                    detail: str = _coerce_str(detail_any) if detail_any is not None else "auto"
                    if detail not in {"low", "high", "auto"}:
                        detail = "auto"
                    try:
                        # If the image is provided as an OpenAI-style data URL, normalize it into
                        # the UMP ImageBlock(base64=...) form so vendor adapters (e.g. Anthropic)
                        # can send raw base64 without a "data:" prefix.
                        if isinstance(url, str) and url.strip().startswith("data:"):
                            parsed = parse_base64_data_url(url)
                            if parsed:
                                parsed_mime, parsed_b64 = parsed
                                url = None
                                b64 = parsed_b64
                                mime = parsed_mime or mime

                        if isinstance(b64, str) and b64.strip().startswith("data:"):
                            parsed = parse_base64_data_url(b64)
                            if parsed:
                                parsed_mime, parsed_b64 = parsed
                                url = None
                                b64 = parsed_b64
                                mime = parsed_mime or mime

                        content_blocks.append(
                            ImageBlock(
                                url=cast(str | None, url),
                                base64=cast(str | None, b64),
                                mime_type=mime,
                                detail=cast(Any, detail),
                            ),
                        )
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
                    raw_content: Any = part_dict.get("content")
                    # Best-effort: preserve multimodal tool results embedded in legacy structured content.
                    coerced = coerce_tool_result_content(raw_content, fallback_text=_coerce_str(raw_content))
                    content_blocks.append(
                        ToolResultBlock(
                            tool_use_id=_coerce_str(part_dict.get("tool_use_id") or part_dict.get("tool_call_id")),
                            content=coerced,
                            is_error=bool(part_dict.get("is_error") or part_dict.get("error")),
                        ),
                    )
                elif part_type == "reasoning":
                    text = _coerce_str(part_dict.get("text") or part_dict.get("content"))
                    sig = part_dict.get("signature")
                    rd = part_dict.get("redacted_data")
                    if text or rd:
                        content_blocks.append(
                            ReasoningBlock(
                                text=text or "",
                                signature=sig if isinstance(sig, str) else None,
                                redacted_data=rd if isinstance(rd, str) else None,
                            )
                        )
                else:
                    # Unknown structured part; store as text so we don't drop user-visible info
                    text = _coerce_str(part_dict.get("text") or part_dict.get("content"))
                    if text:
                        content_blocks.append(TextBlock(text=text))  # pyright: ignore[reportCallIssue]
        else:
            text = _coerce_str(content)
            if text:
                content_blocks.append(TextBlock(text=text))  # pyright: ignore[reportCallIssue]

        # Undo our historical Chat Completions image workaround:
        #   {"role":"user","content":[{"type":"text","text":"Images returned by tool call X:"},{"type":"image_url",...},...]}
        # When we see that pattern, merge the images back into the preceding tool result block and drop this user message.
        if role == Role.USER and isinstance(content, list):
            first_text = next((b.text for b in content_blocks if isinstance(b, TextBlock) and b.text), None)
            if first_text:
                m = _INJECTED_TOOL_IMAGES_RE.match(first_text)
            else:
                m = None
            if m:
                tool_call_id = m.group("id").strip()
                injected_images = [b for b in content_blocks if isinstance(b, ImageBlock)]
                if tool_call_id and injected_images:
                    # Find the nearest preceding tool message for this call id.
                    target_msg = next(
                        (
                            msg
                            for msg in reversed(result)
                            if msg.role == Role.TOOL
                            and any(isinstance(b, ToolResultBlock) and b.tool_use_id == tool_call_id for b in msg.content)
                        ),
                        None,
                    )
                    if target_msg is not None:
                        tr = next(b for b in target_msg.content if isinstance(b, ToolResultBlock) and b.tool_use_id == tool_call_id)
                        if isinstance(tr.content, str):
                            # Strip placeholders now that we'll carry real image blocks.
                            tool_text = tr.content.replace("<image>", "").strip()
                            merged_parts: list[TextBlock | ImageBlock] = []
                            if tool_text:
                                merged_parts.append(TextBlock(text=tool_text))  # pyright: ignore[reportCallIssue]
                            merged_parts.extend(injected_images)
                            tr.content = merged_parts
                        else:
                            merged_parts_existing: list[TextBlock | ImageBlock] = list(tr.content)
                            merged_parts_existing.extend(injected_images)
                            tr.content = merged_parts_existing
                        # Drop this injected user message.
                        continue

        metadata: dict[str, Any] = {}
        if "response_items" in raw:
            metadata["response_items"] = raw.get("response_items")
        if "reasoning" in raw:
            metadata["reasoning"] = raw.get("reasoning")
        if "reasoning_details" in raw:
            # OpenRouter wire format — preserved verbatim for unmodified echo-back.
            metadata["reasoning_details"] = raw.get("reasoning_details")
        if "thought_signature" in raw:
            metadata["thought_signature"] = raw.get("thought_signature")

        # Prefer explicit reasoning_content into a dedicated block (store-only)
        reasoning_content = raw.get("reasoning_content")
        reasoning_signature = raw.get("reasoning_signature")
        reasoning_redacted_data = raw.get("reasoning_redacted_data")
        if ("reasoning_content" in raw and isinstance(reasoning_content, str)) or reasoning_redacted_data:
            # Thinking blocks MUST come before text/tool blocks for Anthropic compliance.
            content_blocks.insert(
                0,
                ReasoningBlock(
                    text=reasoning_content or "",
                    signature=reasoning_signature,
                    redacted_data=reasoning_redacted_data,
                ),
            )

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
            tool_name = _coerce_str(raw.get("name"))
            if tool_name:
                metadata["tool_name"] = tool_name
            # Replace content blocks with an explicit tool_result block (preserve images if present).
            has_images = any(isinstance(b, ImageBlock) for b in content_blocks)
            if has_images:
                parts: list[TextBlock | ImageBlock] = [b for b in content_blocks if isinstance(b, (TextBlock, ImageBlock))]
                content_blocks = [ToolResultBlock(tool_use_id=tool_call_id, content=parts)]
            else:
                tool_text = "".join(b.text for b in content_blocks if isinstance(b, TextBlock))
                content_blocks = [ToolResultBlock(tool_use_id=tool_call_id, content=tool_text)]

        result.append(Message(role=role, content=content_blocks, metadata=metadata))
    return result
