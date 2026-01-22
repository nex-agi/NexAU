"""STDIO transport configuration for Nexau.

This module provides configuration for STDIO-based transport servers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StdioConfig:
    """Configuration for STDIO transport servers.

    Attributes:
        encoding: Text encoding to use (default: utf-8)
        buffer_size: Input buffer size in bytes (default: 8192)
        delimiter: Message delimiter (default: "\n")
        input_separator: Separator for reading input (optional)
        output_separator: Separator for writing output (optional)
    """

    encoding: str = "utf-8"
    buffer_size: int = 8192
    delimiter: str = "\n"
    input_separator: str | None = None
    output_separator: str | None = "\n"
