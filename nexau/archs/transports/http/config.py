"""HTTP transport configuration for Nexau.

This module provides configuration for HTTP-based transport servers.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HTTPConfig:
    """Configuration for HTTP transport servers.

    Attributes:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 8000)
        cors_origins: List of allowed CORS origins (default: ["*"])
        cors_credentials: Allow credentials (default: True)
        cors_methods: Allowed HTTP methods (default: ["*"])
        cors_headers: Allowed headers (default: ["*"])
        log_level: Logging level (default: "info")
        extra: Additional transport-specific options
    """

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    cors_methods: list[str] = field(default_factory=lambda: ["*"])
    cors_headers: list[str] = field(default_factory=lambda: ["*"])
    log_level: str = "info"
