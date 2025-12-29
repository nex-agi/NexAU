"""Unified Message Protocol (UMP).

NexAU historically used ``list[dict[str, Any]]`` as conversation history, mirroring
vendor schemas (OpenAI Chat Completions, Anthropic Messages, OpenAI Responses).

UMP normalizes to:
- Conversation: list[Message]
- Message: role + ordered list of typed blocks
- Block: atomic content unit (text, tool use, tool result, etc.)
"""

from __future__ import annotations

import json
import warnings
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentBlock(BaseModel):
    """Base class for all content blocks."""

    # Intentionally no fields here. Subclasses declare a Literal ``type`` for
    # discriminated unions; keeping ``type: str`` in the base triggers strict
    # type-checker variance complaints when subclasses narrow it to Literals.


class TextBlock(ContentBlock):
    type: Literal["text"] = "text"
    text: str


class ImageBlock(ContentBlock):
    type: Literal["image"] = "image"
    url: str | None = None
    base64: str | None = None
    mime_type: str = "image/jpeg"

    @model_validator(mode="after")
    def _validate_source(self) -> ImageBlock:
        if not self.url and not self.base64:
            raise ValueError("ImageBlock requires either url or base64")
        if self.url and self.base64:
            raise ValueError("ImageBlock cannot have both url and base64")
        return self


class ReasoningBlock(ContentBlock):
    """Optional chain-of-thought style content (store-only; consumers can ignore)."""

    type: Literal["reasoning"] = "reasoning"
    text: str
    signature: str | None = None


class ToolUseBlock(ContentBlock):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]
    # Preserve vendor-provided raw JSON string when available (e.g. OpenAI tool_calls.arguments).
    raw_input: str | None = None


class ToolResultBlock(ContentBlock):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


BlockType = TextBlock | ImageBlock | ReasoningBlock | ToolUseBlock | ToolResultBlock
DiscriminatedBlock = Annotated[BlockType, Field(discriminator="type")]


def _empty_blocks() -> list[DiscriminatedBlock]:
    return []


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    role: Role
    content: list[DiscriminatedBlock] = Field(default_factory=_empty_blocks)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=[TextBlock(text=text)])

    def get_text_content(self) -> str:
        return "".join(block.text for block in self.content if isinstance(block, TextBlock))

    def _as_legacy_openai_chat_dict(self) -> dict[str, Any]:
        """Best-effort legacy OpenAI Chat Completions message dict.

        This mirrors the historical NexAU middleware contract where messages were
        shaped like ``{"role": "...", "content": "..."}``, plus tool-call fields.
        """

        # Tool results become role=tool messages (OpenAI expected)
        if self.role == Role.TOOL:
            tr = next((b for b in self.content if isinstance(b, ToolResultBlock)), None)
            if tr is None:
                return {"role": "tool", "content": self.get_text_content()}
            return {"role": "tool", "tool_call_id": tr.tool_use_id, "content": tr.content}

        entry: dict[str, Any] = {"role": self.role.value, "content": ""}

        # Preserve response-items artifacts if present; they are used by Responses API input reconstruction.
        if "response_items" in self.metadata:
            entry["response_items"] = self.metadata["response_items"]
        if "reasoning" in self.metadata:
            entry["reasoning"] = self.metadata["reasoning"]

        has_images = any(isinstance(b, ImageBlock) for b in self.content)
        text_parts: list[str] = []
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []

        for block in self.content:
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
                # Legacy chat schema expects tool results as a separate role=tool message.
                # For dict-style access compatibility, fold tool results into textual content.
                if has_images:
                    content_parts.append({"type": "text", "text": block.content})
                else:
                    text_parts.append(block.content)
            elif isinstance(block, ImageBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
                if block.url:
                    content_parts.append({"type": "image_url", "image_url": {"url": block.url}})
                else:
                    content_parts.append({"type": "image_url", "image_url": {"url": f"data:{block.mime_type};base64,{block.base64}"}})

        entry["content"] = content_parts if has_images else "".join(text_parts)
        if reasoning_parts:
            entry["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            entry["tool_calls"] = tool_calls
        return entry

    def __getitem__(self, key: str) -> Any:
        """Legacy dict-style access shim for middleware compatibility.

        External middleware previously received ``dict`` messages and frequently used
        patterns like ``msg["content"]``. UMP messages are Pydantic models; we keep
        this shim for a deprecation period so legacy middleware doesn't crash.
        """

        warnings.warn(
            "Dict-style access on Message (e.g. msg['content']) is deprecated; use attributes "
            "(msg.role, msg.content) or msg.get_text_content() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        legacy = self._as_legacy_openai_chat_dict()
        if key in legacy:
            return legacy[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Legacy dict-style get() shim for middleware compatibility."""

        try:
            return self[key]
        except KeyError:
            return default
