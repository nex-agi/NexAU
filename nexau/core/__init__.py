# pyright: reportUnusedImport=false

"""Core, vendor-agnostic primitives for NexAU.

This package intentionally contains *no* SDK/vendor client code. It defines the
Unified Message Protocol (UMP) data model and adapters for converting to/from
vendor-specific payloads.
"""

from .messages import (
    BlockType,
    ContentBlock,
    ImageBlock,
    Message,
    ReasoningBlock,
    Role,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
