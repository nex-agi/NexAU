"""Legacy message conversions.

These helpers bridge NexAU's historical ``list[dict[str, Any]]`` chat history into
UMP messages, and back into OpenAI Chat-Completions-shaped dicts.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any, Literal, cast

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

        # Undo our own legacy workaround:
        # messages_to_legacy_openai_chat(tool_image_policy="inject_user_message") emits an extra user message:
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

        # Prefer explicit reasoning_content into a dedicated block (store-only)
        reasoning_content = raw.get("reasoning_content")
        reasoning_signature = raw.get("reasoning_signature")
        reasoning_redacted_data = raw.get("reasoning_redacted_data")
        if (isinstance(reasoning_content, str) and reasoning_content) or reasoning_redacted_data:
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


def messages_to_legacy_openai_chat(
    messages: list[Message],
    *,
    tool_image_policy: Literal["inject_user_message", "embed_in_tool_message"] = "inject_user_message",
) -> list[dict[str, Any]]:
    """Convert UMP messages into OpenAI Chat-Completions-shaped dicts (legacy NexAU format).

    Notes:
    - OpenAI Chat Completions tool-role messages do NOT support image parts (tool content supports text only).
      When tool results contain images, `tool_image_policy="inject_user_message"` will:
        - emit a tool-role message with text (images replaced by "<image>")
        - emit an extra user-role multimodal message containing the images (image_url parts)
    - For OpenAI Responses API input reconstruction, set `tool_image_policy="embed_in_tool_message"` so
      downstream conversion can preserve image_url parts for function_call_output output arrays.
    """

    output: list[dict[str, Any]] = []
    for msg in messages or []:
        # FRAMEWORK is treated as "user" when sent to LLM
        role = "user" if msg.role == Role.FRAMEWORK else msg.role.value

        def _image_part_to_image_url_obj(img: ImageBlock) -> dict[str, Any]:
            url = img.url if img.url else f"data:{img.mime_type};base64,{img.base64}"
            image_url_obj: dict[str, Any] = {"url": url}
            if img.detail != "auto":
                image_url_obj["detail"] = img.detail
            return image_url_obj

        def _emit_tool_result_as_messages(
            *,
            tool_call_id: str,
            tool_content: str | list[TextBlock | ImageBlock],
        ) -> list[dict[str, Any]]:
            if not isinstance(tool_content, list):
                return [{"role": "tool", "tool_call_id": tool_call_id, "content": tool_content}]

            text_parts: list[str] = []
            tool_text_only_parts: list[dict[str, Any]] = []
            image_parts: list[dict[str, Any]] = []
            for part in tool_content:
                if isinstance(part, TextBlock):
                    if part.text:
                        tool_text_only_parts.append({"type": "text", "text": part.text})
                        text_parts.append(part.text)
                else:
                    # image
                    image_parts.append({"type": "image_url", "image_url": _image_part_to_image_url_obj(part)})
                    text_parts.append("<image>")

            if tool_image_policy == "embed_in_tool_message":
                # Used for Responses API input reconstruction (not sent directly to Chat Completions).
                return [{"role": "tool", "tool_call_id": tool_call_id, "content": tool_text_only_parts + image_parts}]

            # Chat Completions spec: tool message content parts can only be text.
            tool_text = "".join(text_parts) or "<tool_output>"
            emitted: list[dict[str, Any]] = [{"role": "tool", "tool_call_id": tool_call_id, "content": tool_text}]
            if image_parts:
                emitted.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"Images returned by tool call {tool_call_id}:"}] + image_parts,
                    },
                )
            return emitted

        # Tool results become role=tool messages (OpenAI expected)
        if role == Role.TOOL.value:
            # Prefer the first ToolResultBlock
            tr = next((b for b in msg.content if isinstance(b, ToolResultBlock)), None)
            if tr is None:
                output.append({"role": "tool", "content": msg.get_text_content()})
            else:
                tool_msgs = _emit_tool_result_as_messages(tool_call_id=tr.tool_use_id, tool_content=tr.content)
                # Propagate tool_name from metadata (used by Gemini REST for functionResponse.name)
                tool_name = msg.metadata.get("tool_name")
                if tool_name:
                    for tm in tool_msgs:
                        tm["name"] = tool_name
                output.extend(tool_msgs)
            continue

        entry: dict[str, Any] = {"role": role, "content": ""}

        # Preserve response-items artifacts if present; they are used by Responses API input reconstruction.
        if "response_items" in msg.metadata:
            entry["response_items"] = msg.metadata["response_items"]
        if "reasoning" in msg.metadata:
            entry["reasoning"] = msg.metadata["reasoning"]
        if "thought_signature" in msg.metadata:
            entry["thought_signature"] = msg.metadata["thought_signature"]

        has_images = any(isinstance(b, ImageBlock) for b in msg.content)
        text_parts: list[str] = []
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        reasoning_signatures: list[str] = []
        reasoning_redacted_data_parts: list[str] = []

        for block in msg.content:
            if isinstance(block, TextBlock):
                if has_images:
                    content_parts.append({"type": "text", "text": block.text})
                else:
                    text_parts.append(block.text)
            elif isinstance(block, ReasoningBlock):
                reasoning_parts.append(block.text)
                if block.signature:
                    reasoning_signatures.append(block.signature)
                if block.redacted_data:
                    reasoning_redacted_data_parts.append(block.redacted_data)
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
                output.extend(_emit_tool_result_as_messages(tool_call_id=block.tool_use_id, tool_content=block.content))
            elif isinstance(block, ImageBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
                # OpenAI Chat Completions supports multi-part content with images:
                # {"content": [{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"..."}}]}
                if block.url:
                    image_url_obj: dict[str, Any] = {"url": block.url}
                    if block.detail != "auto":
                        image_url_obj["detail"] = block.detail
                    content_parts.append({"type": "image_url", "image_url": image_url_obj})
                else:
                    # Use a data URL so downstream OpenAI-compatible clients can send the image.
                    image_url_obj = {"url": f"data:{block.mime_type};base64,{block.base64}"}
                    if block.detail != "auto":
                        image_url_obj["detail"] = block.detail
                    content_parts.append({"type": "image_url", "image_url": image_url_obj})

        entry["content"] = content_parts if has_images else "".join(text_parts)
        if reasoning_parts:
            entry["reasoning_content"] = "".join(reasoning_parts)
        if reasoning_signatures:
            # We take the last signature if multiple thinking blocks exist (rare).
            entry["reasoning_signature"] = reasoning_signatures[-1]
        if reasoning_redacted_data_parts:
            entry["reasoning_redacted_data"] = reasoning_redacted_data_parts[-1]
        if tool_calls:
            entry["tool_calls"] = tool_calls
        output.append(entry)

    return output
