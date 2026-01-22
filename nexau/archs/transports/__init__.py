"""Nexau Transport Module.

This module provides transport implementations for different communication protocols,
including HTTP (SSE), stdio, WebSocket, etc.

Usage:
    # HTTP (SSE) Transport with ORM DatabaseEngine
    from nexau.archs.transports.http import SSETransportServer, HTTPConfig
    from nexau.archs.session import SQLiteDatabaseEngine
    from nexau import AgentConfig

    # Create engine (uses ~/.nexau/nexau.db by default)
    engine = SQLiteDatabaseEngine()

    # Create server
    agent_config = AgentConfig.from_yaml("agent.yaml")
    server = SSETransportServer(
        engine=engine,
        config=HTTPConfig(host="127.0.0.1", port=8000, cors_origins=["*"]),
        default_agent_config=agent_config,
    )
    server.run()

    # HTTP Client
    from nexau.archs.transports.http import SSEClient

    client = SSEClient("http://localhost:8000")
    response = await client.query("Hello!")
"""

from nexau.archs.transports.base import TransportBase

__all__ = [
    "TransportBase",
]
