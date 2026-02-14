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
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, TypedDict, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_serializer, model_validator


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    FRAMEWORK = "FRAMEWORK"  # Framework-injected user messages (treated as "user" when sent to LLM)
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
    # Optional hint for multimodal-capable models (OpenAI-style):
    # "low" | "high" | "auto" (default).
    detail: Literal["low", "high", "auto"] = "auto"

    @model_validator(mode="after")
    def _validate_source(self) -> ImageBlock:
        if not self.url and not self.base64:
            raise ValueError("ImageBlock requires either url or base64")
        if self.url and self.base64:
            raise ValueError("ImageBlock cannot have both url and base64")
        return self


class ToolOutputImage(BaseModel):
    """Developer-friendly tool output type for returning images.

    Tools may return:
    - `ToolOutputImage(image_url="https://...")`
    - `ToolOutputImage(image_url="data:image/png;base64,....")`
    """

    type: Literal["input_image"] = "input_image"
    image_url: str
    detail: Literal["low", "high", "auto"] = "auto"

    @model_validator(mode="after")
    def _validate(self) -> ToolOutputImage:
        if not self.image_url.strip():
            raise ValueError("ToolOutputImage.image_url must be a non-empty string")
        return self


class ToolOutputImageDict(TypedDict, total=False):
    """Dict form for returning images from tools.

    Example:
      {
        "type": "input_image",
        "image_url": "https://... or data:image/png;base64,...",
        "detail": "high" | "low" | "auto",
      }
    """

    type: Literal["input_image"]
    image_url: str
    detail: Literal["low", "high", "auto"]


def _parse_base64_data_url(url: str) -> tuple[str, str] | None:
    url = url.strip()
    if not url.startswith("data:"):
        return None
    try:
        header, data = url.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None
    mime = header[len("data:") :].split(";", 1)[0] or "image/jpeg"
    # Be liberal in what we accept: downstream vendors require raw base64 without
    # a "data:" prefix and without whitespace/newlines.
    cleaned = "".join(data.split())
    # Guard against common mistakes where callers stringify bytes:
    #   str(base64.b64encode(...)) -> "b'...'"
    if (cleaned.startswith("b'") and cleaned.endswith("'")) or (cleaned.startswith('b"') and cleaned.endswith('"')):
        cleaned = cleaned[2:-1]
    # Also accept accidental quoting.
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (cleaned.startswith('"') and cleaned.endswith('"')):
        cleaned = cleaned[1:-1]
    return mime, cleaned


def parse_base64_data_url(url: str) -> tuple[str, str] | None:
    """Parse a base64 data URL into (mime_type, base64_data).

    This is intentionally tolerant of whitespace and accidental quoting, since
    callers frequently copy/paste base64 strings or stringify bytes.
    """

    return _parse_base64_data_url(url)


def _image_block_from_image_url(image_url: str, *, detail: Literal["low", "high", "auto"] = "auto") -> ImageBlock:
    parsed = _parse_base64_data_url(image_url)
    if parsed:
        mime_type, b64 = parsed
        return ImageBlock(base64=b64, mime_type=mime_type, detail=detail)
    return ImageBlock(url=image_url, detail=detail)


def coerce_tool_result_content(
    value: Any,
    *,
    fallback_text: str | None = None,
) -> str | list[TextBlock | ImageBlock]:
    """Coerce arbitrary tool output into ToolResultBlock.content.

    Supported developer return shapes:
    - `ToolOutputImage(image_url="...")`
    - `{"type": "image", "image_url": "https://..."}` or base64 data URL
    - `{"type": "image", "base64": "...", "media_type": "image/png"}` (common in built-in tools)
    - lists mixing text/image parts

    Backwards-compat: if no image parts are detected and `fallback_text` is provided,
    return the fallback string.
    """

    def from_mapping(obj: dict[str, Any]) -> TextBlock | ImageBlock | None:
        part_type = str(obj.get("type") or "")
        if part_type in {"text", "output_text", "input_text"}:
            text_val: Any = obj.get("text")
            if text_val is None:
                text_val = obj.get("content")
            return TextBlock(text=str(text_val or ""))

        if part_type in {"input_image", "image"}:
            detail_any: Any = obj.get("detail")
            detail: Literal["low", "high", "auto"] = "auto"
            if isinstance(detail_any, str) and detail_any in {"low", "high", "auto"}:
                detail = cast(Literal["low", "high", "auto"], detail_any)

            image_url: Any = obj.get("image_url") or obj.get("url")
            if isinstance(image_url, str) and image_url.strip():
                return _image_block_from_image_url(image_url.strip(), detail=detail)

            b64: Any = obj.get("base64")
            media_type: Any = obj.get("media_type") or obj.get("mime_type") or "image/jpeg"
            if isinstance(b64, str) and b64:
                return ImageBlock(base64=b64, mime_type=str(media_type or "image/jpeg"), detail=detail)
            return None

        # Unknown mapping: try to preserve any user-visible text-ish payload.
        text_val = obj.get("text") or obj.get("content")
        if text_val is None:
            return None
        return TextBlock(text=text_val if isinstance(text_val, str) else str(text_val))

    def coerce_any(item: Any) -> list[TextBlock | ImageBlock]:
        if item is None:
            return []
        if isinstance(item, ToolOutputImage):
            return [_image_block_from_image_url(item.image_url, detail=item.detail)]
        if isinstance(item, str):
            text = item
            if text:
                return [TextBlock(text=text)]
            return []
        if isinstance(item, dict):
            item_dict = cast(dict[str, Any], item)
            # Common wrapper shapes from ToolExecutor: {"result": ...} or {"content": [...]}
            if "type" not in item_dict:
                inner = item_dict.get("content", item_dict.get("result"))
                if inner is not None and isinstance(inner, (list, dict, ToolOutputImage)):
                    return coerce_any(inner)

            block = from_mapping(item_dict)
            if block is None:
                return []
            # Drop empty text blocks
            if isinstance(block, TextBlock) and not block.text:
                return []
            return [block]
        if isinstance(item, list):
            out: list[TextBlock | ImageBlock] = []
            for sub in cast(list[Any], item):
                out.extend(coerce_any(sub))
            return out
        # Last resort stringify
        return [TextBlock(text=str(item))]

    # If tool returns JSON-as-string, try to decode it before coercion.
    if isinstance(value, str):
        raw = value.strip()
        if raw:
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
            if parsed is not None:
                blocks = coerce_any(parsed)
                if any(b.type == "image" for b in blocks):
                    return blocks
        # Fallback: keep historical behavior for normal tool JSON strings
        return fallback_text if fallback_text is not None else value

    blocks = coerce_any(value)
    if any(b.type == "image" for b in blocks):
        return blocks
    if fallback_text is not None:
        return fallback_text
    # No images found: collapse to string to preserve prior behavior (common tool results are JSON dicts)
    if not blocks:
        return ""
    return "".join(b.text for b in blocks if isinstance(b, TextBlock))


class ReasoningBlock(ContentBlock):
    """Optional chain-of-thought style content (store-only; consumers can ignore)."""

    type: Literal["reasoning"] = "reasoning"
    text: str
    signature: str | None = None
    redacted_data: str | None = None


class ToolUseBlock(ContentBlock):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any]
    # Preserve vendor-provided raw JSON string when available (e.g. OpenAI tool_calls.arguments).
    raw_input: str | None = None


ToolResultContentBlock = Annotated[TextBlock | ImageBlock, Field(discriminator="type")]


class ToolResultBlock(ContentBlock):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    # Backwards-compatible: historically tool result content was a plain string.
    # We now also support mixed multimodal content (text + images).
    content: str | list[ToolResultContentBlock]
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
    created_at: datetime | None = Field(default=None)

    @field_serializer("created_at")
    @classmethod
    def serialize_created_at(cls, v: datetime | None) -> str:
        return (v or datetime.now()).isoformat()

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
            if isinstance(tr.content, list):
                # Chat Completions spec: tool-role messages support text only.
                # Collapse images to placeholders so legacy dict access stays usable.
                folded = "".join(p.text if isinstance(p, TextBlock) else "<image>" for p in tr.content) or "<tool_output>"
                return {"role": "tool", "tool_call_id": tr.tool_use_id, "content": folded}
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
                if isinstance(block.content, list):
                    # Fold text portions; represent images as "<image>" placeholders.
                    folded = "".join(p.text if isinstance(p, TextBlock) else "<image>" for p in block.content)
                else:
                    folded = block.content
                if has_images:
                    content_parts.append({"type": "text", "text": folded})
                else:
                    text_parts.append(folded)
            elif isinstance(block, ImageBlock):  # pyright: ignore[reportUnnecessaryIsInstance]
                if block.url:
                    image_url_obj: dict[str, Any] = {"url": block.url}
                    if block.detail != "auto":
                        image_url_obj["detail"] = block.detail
                    content_parts.append({"type": "image_url", "image_url": image_url_obj})
                else:
                    image_url_obj = {"url": f"data:{block.mime_type};base64,{block.base64}"}
                    if block.detail != "auto":
                        image_url_obj["detail"] = block.detail
                    content_parts.append({"type": "image_url", "image_url": image_url_obj})

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
