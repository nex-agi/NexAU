"""Anthropic Messages request payload serializers.

RFC-0014: UMP 到 Provider Payload 的统一序列化分层

Provides serializer helpers for converting UMP messages into Anthropic
Messages API ``system`` / ``messages`` payload blocks.
"""

from __future__ import annotations

from typing import Any, cast

from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock


def serialize_ump_to_anthropic_messages_payload(
    messages: list[Message],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert UMP messages into Anthropic Messages API payloads.

    RFC-0014: Anthropic provider payload serializer
    """

    system_blocks: list[dict[str, Any]] = []
    convo: list[dict[str, Any]] = []

    def _image_block_to_anthropic(img: ImageBlock) -> dict[str, Any] | None:
        try:
            if img.base64:
                return {
                    "type": "image",
                    "source": {"type": "base64", "media_type": img.mime_type, "data": img.base64},
                }
            if img.url:
                return {"type": "image", "source": {"type": "url", "url": img.url}}
        except Exception:
            return None
        return None

    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        if pending_tool_results:
            convo.append({"role": Role.USER.value, "content": pending_tool_results.copy()})
            pending_tool_results.clear()

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_text = msg.get_text_content()
            if system_text:
                sys_block: dict[str, Any] = {"type": "text", "text": system_text}
                if "cache" in msg.metadata:
                    sys_block["_cache"] = msg.metadata["cache"]
                system_blocks.append(sys_block)
            continue

        has_companion_assistant_output = any(
            (isinstance(existing_block, TextBlock) and bool(existing_block.text)) or isinstance(existing_block, (ImageBlock, ToolUseBlock))
            for existing_block in msg.content
        )

        content_blocks: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, TextBlock):
                if block.text:
                    content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ReasoningBlock):
                if block.redacted_data:
                    content_blocks.append({"type": "redacted_thinking", "data": block.redacted_data})
                elif block.signature:
                    content_blocks.append({"type": "thinking", "thinking": block.text, "signature": block.signature})
                elif has_companion_assistant_output:
                    content_blocks.append({"type": "thinking", "thinking": block.text})
                elif block.text:
                    content_blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolUseBlock):
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    },
                )
            elif isinstance(block, ImageBlock):
                img_block = _image_block_to_anthropic(block)
                if img_block is not None:
                    content_blocks.append(img_block)
            elif isinstance(block, ToolResultBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
                if isinstance(block.content, list):
                    tool_result_text_parts: list[dict[str, Any]] = []
                    sibling_image_parts: list[dict[str, Any]] = []
                    for part in block.content:
                        if isinstance(part, TextBlock):
                            if part.text:
                                tool_result_text_parts.append({"type": "text", "text": part.text})
                        else:
                            img_block = _image_block_to_anthropic(part)
                            if img_block is not None:
                                sibling_image_parts.append(img_block)
                    content_blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.tool_use_id,
                            "content": tool_result_text_parts,
                            "is_error": block.is_error,
                        },
                    )
                    content_blocks.extend(sibling_image_parts)
                    continue
                content_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_use_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    },
                )

        role = msg.role.value
        if msg.role in (Role.TOOL, Role.FRAMEWORK):
            role = Role.USER.value

        if msg.role == Role.TOOL:
            pending_tool_results.extend(content_blocks)
            continue

        _flush_tool_results()
        convo.append({"role": role, "content": content_blocks})

    _flush_tool_results()

    return system_blocks, convo


def apply_anthropic_last_user_cache_control(
    convo: list[dict[str, Any]],
    *,
    system_cache_control_ttl: str | None = None,
) -> list[dict[str, Any]]:
    """Apply legacy-compatible cache control to the last user text block.

    RFC-0014: Anthropic legacy compatibility helper
    """

    if not system_cache_control_ttl or not convo:
        return convo

    last = convo[-1]
    content = last.get("content")
    if isinstance(content, list) and content:
        content_list = cast(list[object], content)
        first_any = content_list[0]
        if isinstance(first_any, dict):
            first = cast(dict[str, Any], first_any)
            if first.get("type") == "text":
                first["cache_control"] = {"type": "ephemeral", "ttl": system_cache_control_ttl}
    return convo
