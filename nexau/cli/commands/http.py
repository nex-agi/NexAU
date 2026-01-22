"""HTTP transport CLI command - Server only.

This module provides HTTP server command for serving agents over HTTP/SSE.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from nexau import AgentConfig
from nexau.archs.session import SQLDatabaseEngine
from nexau.archs.transports.http import HTTPConfig, SSETransportServer


class ServerArgs(BaseModel):
    """Validated arguments for HTTP server command."""

    agent: str
    host: str
    port: int
    log_level: str
    cors_origins: list[str]


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for HTTP Server command.

    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument(
        "agent",
        type=str,
        help="Path to agent YAML configuration",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        nargs="+",
        default=["*"],
        help="Allowed CORS origins (default: *)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (sets log level to debug)",
    )


def main(args: argparse.Namespace) -> int:
    """Execute HTTP Server command.

    Start FastAPI server providing /query (sync), /stream (SSE),
    /health (health check), /info (server info) endpoints.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 = success, non-0 = error)
    """
    # Load environment variables from .env file
    load_dotenv()

    # If verbose is enabled, use debug log level
    log_level = "debug" if args.verbose else args.log_level

    # Configure Python logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,  # Override any existing configuration
    )

    # Validate and parse arguments with Pydantic
    server_args = ServerArgs(
        agent=args.agent,
        host=args.host,
        port=args.port,
        log_level=log_level,
        cors_origins=args.cors_origins,
    )

    # Load agent configuration
    agent_path = Path(server_args.agent)
    if not agent_path.exists():
        print(f"❌ Agent configuration file not found: {server_args.agent}")
        print("   Please provide a valid path to the agent YAML file.")
        return 1

    print(f"✓ Loading agent from: {server_args.agent}")
    agent_config = AgentConfig.from_yaml(agent_path)

    # Create engine with proper directory handling
    # Expand the home directory and ensure the .nexau directory exists
    db_path = Path.home() / ".nexau" / "nexau.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = SQLDatabaseEngine.from_url(f"sqlite+aiosqlite:///{db_path}")

    # Create HTTP configuration
    config = HTTPConfig(
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level,
        cors_origins=server_args.cors_origins,
    )

    # Create HTTP server
    server = SSETransportServer(
        engine=engine,
        config=config,
        default_agent_config=agent_config,
    )

    # Display server information
    print("=" * 60)
    print("🚀 NexAU HTTP Server Starting")
    print("=" * 60)
    print(f"Agent:        {agent_config.name}")
    print(f"Host:         {server.host}")
    print(f"Port:         {server.port}")
    print(f"Log Level:    {config.log_level}")
    print(f"CORS Origins: {config.cors_origins}")
    print()
    print("📋 Endpoints:")
    print(f"   Health:     {server.health_url}")
    print(f"   Info:       {server.info_url}")
    print(f"   Stream:     http://{server.host}:{server.port}/stream")
    print(f"   Query:      http://{server.host}:{server.port}/query")
    print()
    print("💡 Example Usage:")
    print()
    print("   # 1. Health Check")
    print(f'   curl "{server.health_url}"')
    print()
    print("   # 2. Simple Query (string message)")
    print(f"   curl -X POST '{server.health_url.replace('/health', '/query')}' \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"messages": "What is Python?", "user_id": "user123"}\'')
    print()
    print("   # 3. Query with Message History (list of messages)")
    print(f"   curl -X POST '{server.health_url.replace('/health', '/query')}' \\")
    print('     -H "Content-Type: application/json" \\')
    print("     -d '{")
    print('       "messages": [')
    print('         {"role": "user", "content": [{"type": "text", "text": "Hello"}]},')
    print('         {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},')
    print('         {"role": "user", "content": [{"type": "text", "text": "How are you?"}]}')
    print("       ],")
    print('       "user_id": "user123"')
    print("     }'")
    print()
    print("   # 4. Streaming Query (SSE)")
    print(f"   curl -N -X POST '{server.health_url.replace('/health', '/stream')}' \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"messages": "Tell me a story", "user_id": "user123"}\'')
    print()
    print("📖 API Documentation:")
    print("   Request Body: { messages: str | list[Message], user_id: str, session_id?: str }")
    print("   Response:     { status: 'success' | 'error', response?: str, error?: str }")
    print("=" * 60)
    print()

    # Start server (blocking)
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return 0
    except Exception as e:
        print(f"\n\n❌ Error starting server: {e}", file=sys.stderr)
        return 1

    return 0
