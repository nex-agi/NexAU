from __future__ import annotations

import json
from typing import Any, Literal

from nexau.core.messages import ImageBlock, Message, ReasoningBlock, Role, TextBlock, ToolResultBlock, ToolUseBlock

ToolImagePolicy = Literal["inject_user_message", "embed_in_tool_message"]


def serialize_ump_to_openai_chat_payload(
    messages: list[Message],
    *,
    tool_image_policy: ToolImagePolicy = "inject_user_message",
) -> list[dict[str, Any]]:
    """Convert UMP messages into OpenAI Chat-Completions-compatible payload dicts.

    Notes:
    - OpenAI Chat Completions tool-role messages do NOT support image parts.
    - When tool results contain images, ``tool_image_policy='inject_user_message'`` emits
      a tool-role message with text placeholders plus an extra user multimodal message.
    - For downstream Responses input reconstruction, ``tool_image_policy='embed_in_tool_message'``
      preserves tool text/image parts inside the tool message payload.
    """

    output: list[dict[str, Any]] = []
    for msg in messages or []:
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
                    image_parts.append({"type": "image_url", "image_url": _image_part_to_image_url_obj(part)})
                    text_parts.append("<image>")

            if tool_image_policy == "embed_in_tool_message":
                return [{"role": "tool", "tool_call_id": tool_call_id, "content": tool_text_only_parts + image_parts}]

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

        if role == Role.TOOL.value:
            tr = next((b for b in msg.content if isinstance(b, ToolResultBlock)), None)
            if tr is None:
                output.append({"role": "tool", "content": msg.get_text_content()})
            else:
                tool_msgs = _emit_tool_result_as_messages(tool_call_id=tr.tool_use_id, tool_content=tr.content)
                tool_name = msg.metadata.get("tool_name")
                if tool_name:
                    for tm in tool_msgs:
                        tm["name"] = tool_name
                output.extend(tool_msgs)
            continue

        entry: dict[str, Any] = {"role": role, "content": ""}

        if "response_items" in msg.metadata:
            entry["response_items"] = msg.metadata["response_items"]
        if "reasoning" in msg.metadata:
            entry["reasoning"] = msg.metadata["reasoning"]
        if "reasoning_details" in msg.metadata:
            # OpenRouter requires this to be echoed back unmodified for multi-turn reasoning.
            entry["reasoning_details"] = msg.metadata["reasoning_details"]
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
                output.extend(_emit_tool_result_as_messages(tool_call_id=block.tool_use_id, tool_content=block.content))
            elif isinstance(block, ImageBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
                image_url_obj = _image_part_to_image_url_obj(block)
                content_parts.append({"type": "image_url", "image_url": image_url_obj})

        entry["content"] = content_parts if has_images else "".join(text_parts)
        if reasoning_parts:
            entry["reasoning_content"] = "".join(reasoning_parts)
        if reasoning_signatures:
            entry["reasoning_signature"] = reasoning_signatures[-1]
        if reasoning_redacted_data_parts:
            entry["reasoning_redacted_data"] = reasoning_redacted_data_parts[-1]
        if tool_calls:
            entry["tool_calls"] = tool_calls
        output.append(entry)

    return output
