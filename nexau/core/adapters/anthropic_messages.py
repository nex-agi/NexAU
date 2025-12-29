from __future__ import annotations

from typing import Any, cast

from nexau.core.adapters.base import LLMAdapter
from nexau.core.messages import ImageBlock, Message, Role, TextBlock, ToolResultBlock, ToolUseBlock


class AnthropicMessagesAdapter(LLMAdapter):
    """Adapter for Anthropic Messages API payloads.

    Returns a tuple of:
    - system: list[{"type": "text", "text": "..."}]
    - messages: list[{"role": "...", "content": [blocks...]}]
    """

    def to_vendor_format(self, messages: list[Message]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        system_blocks: list[dict[str, Any]] = []
        convo: list[dict[str, Any]] = []

        def _image_block_to_anthropic(img: ImageBlock) -> dict[str, Any] | None:
            # Anthropic Messages API supports:
            # {"type":"image","source":{"type":"base64","media_type":"image/png","data":"..."}}
            # {"type":"image","source":{"type":"url","url":"https://..."}}
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

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Anthropic expects "system" as a separate top-level parameter, not message role.
                system_text = msg.get_text_content()
                if system_text:
                    system_blocks.append({"type": "text", "text": system_text})
                continue

            content_blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if block.text:
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
                elif isinstance(block, ToolResultBlock):
                    if isinstance(block.content, list):
                        # Compatibility note:
                        # Some Anthropic-compatible gateways accept tool_result blocks with only text
                        # and reject nested image blocks inside tool_result.content. To maximize
                        # compatibility, we emit images as sibling blocks *outside* the tool_result
                        # (i.e., format: [tool_result{text...}, image, image, ...]).
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
                else:
                    # Ignore unsupported blocks for now (images/reasoning) to keep adapter strict.
                    continue

            # Bedrock Claude Messages only allows roles: "user" | "assistant".
            # Tool results must be sent as a *user* message containing a "tool_result" content block.
            role = msg.role.value
            if msg.role == Role.TOOL:
                role = Role.USER.value

            convo.append({"role": role, "content": content_blocks})

        return system_blocks, convo

    def from_vendor_response(self, response: Any) -> Message:  # pragma: no cover (not wired yet)
        raise NotImplementedError


def anthropic_payload_from_legacy_openai_chat(
    legacy_messages: list[dict[str, Any]],
    *,
    system_cache_control_ttl: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compatibility wrapper for existing NexAU call sites.

    This replaces the old ``openai_to_anthropic_message`` conversion logic with:
      legacy dicts -> UMP -> Anthropic content blocks
    """

    from nexau.core.adapters.legacy import messages_from_legacy_openai_chat

    ump_messages = messages_from_legacy_openai_chat(legacy_messages)
    system, convo = AnthropicMessagesAdapter().to_vendor_format(ump_messages)

    # Preserve old behavior: add cache_control on the first content block of the last user message.
    if system_cache_control_ttl and convo:
        last = convo[-1]
        content = last.get("content")
        if isinstance(content, list) and content:
            content_list = cast(list[object], content)
            first_any = content_list[0]
            if isinstance(first_any, dict):
                first = cast(dict[str, Any], first_any)
                if first.get("type") == "text":
                    first["cache_control"] = {"type": "ephemeral", "ttl": system_cache_control_ttl}

    return system, convo
