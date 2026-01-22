"""Stdio transport CLI command.

This module provides stdio-based transport command for CLI and IPC communication.
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
from nexau.archs.transports.stdio import StdioConfig, StdioTransport


class ServerArgs(BaseModel):
    """Validated arguments for stdio server command."""

    agent: str
    verbose: bool


def setup_server_parser(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for Stdio Server command.

    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument(
        "agent",
        type=str,
        help="Path to agent YAML configuration file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (debug level logging)",
    )


def server_main(args: argparse.Namespace) -> int:
    """Execute Stdio Server command.

    Starts a stdio transport server that:
    - Reads JSON Lines from stdin
    - Writes JSON Lines to stdout
    - Logs to stderr

    Supports both streaming and synchronous modes via request type field.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 = success, non-0 = error)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Validate and parse arguments with Pydantic
    server_args = ServerArgs(
        agent=args.agent,
        verbose=args.verbose,
    )

    # Configure Python logging to stderr
    log_level = logging.DEBUG if server_args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
        force=True,
    )

    # Load agent configuration
    agent_path = Path(server_args.agent)
    if not agent_path.exists():
        print(f"❌ Agent configuration file not found: {server_args.agent}", file=sys.stderr)
        print("   Please provide a valid path to the agent YAML file.", file=sys.stderr)
        return 1

    print(f"✓ Loading agent from: {server_args.agent}", file=sys.stderr)
    agent_config = AgentConfig.from_yaml(agent_path)

    # Create engine with SQLite persistence
    db_path = Path.home() / ".nexau" / "nexau.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = SQLDatabaseEngine.from_url(f"sqlite+aiosqlite:///{db_path}")

    # Create stdio configuration
    config = StdioConfig()

    # Create stdio transport
    transport = StdioTransport(
        engine=engine,
        config=config,
        default_agent_config=agent_config,
    )

    # Display server information to stderr
    print("=" * 60, file=sys.stderr)
    print("🚀 NexAU Stdio Server Starting", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Agent:        {agent_config.name}", file=sys.stderr)
    print(f"Log Level:    {'DEBUG' if server_args.verbose else 'WARNING'}", file=sys.stderr)
    print("Database:     ~/.nexau/nexau.db", file=sys.stderr)
    print(f"Encoding:     {config.encoding}", file=sys.stderr)
    print("Protocol:     JSON Lines", file=sys.stderr)
    print(file=sys.stderr)
    print("📋 I/O Configuration:", file=sys.stderr)
    print("   Input:      stdin (JSON Lines)", file=sys.stderr)
    print("   Output:     stdout (JSON Lines)", file=sys.stderr)
    print("   Logging:    stderr", file=sys.stderr)
    print(file=sys.stderr)
    print("📖 Message Format:", file=sys.stderr)
    print('   Request:    {"jsonrpc": "2.0-stream", "method": "agent.query"|"agent.stream", "params": {...}, "id": str}', file=sys.stderr)
    print('   Response:   {"jsonrpc": "2.0-stream", "id": str, "result": ...} (sync mode)', file=sys.stderr)
    print('   Events:     {"jsonrpc": "2.0-stream", "id": str, "event": {...}} (stream mode)', file=sys.stderr)
    print('   End:        {"jsonrpc": "2.0-stream", "id": str, "result": null} (stream mode)', file=sys.stderr)
    print(file=sys.stderr)
    print("💡 Example Usage:", file=sys.stderr)
    print(file=sys.stderr)
    print("   # 1. Simple Query (synchronous)", file=sys.stderr)
    print(
        '   echo \'{"jsonrpc":"2.0-stream",'
        '"method":"agent.query",'
        '"params":{"messages":"Hello","user_id":"user123"},'
        '"id":"req_1"}\''
        " | nexau serve stdio agent.yaml",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print("   # 2. Streaming Query", file=sys.stderr)
    print(
        '   echo \'{"jsonrpc":"2.0-stream",'
        '"method":"agent.stream",'
        '"params":{"messages":"Tell me a story","user_id":"user123"},'
        '"id":"req_1"}\''
        " | nexau serve stdio agent.yaml",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print("   # 3. Programmatic Usage (Python)", file=sys.stderr)
    print("   import subprocess, json", file=sys.stderr)
    print('   proc = subprocess.Popen(["nexau", "serve", "stdio", "agent.yaml"],', file=sys.stderr)
    print("                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)", file=sys.stderr)
    print(
        "   request = {"
        '"jsonrpc": "2.0-stream", '
        '"method": "agent.query", '
        '"params": {"messages": "Hello", "user_id": "user"}, '
        '"id": "req_1"'
        "}",
        file=sys.stderr,
    )
    print("   proc.stdin.write(json.dumps(request) + '\\n')", file=sys.stderr)
    print("   response = json.loads(proc.stdout.readline())", file=sys.stderr)
    print(file=sys.stderr)
    print("🔧 Server ready. Reading from stdin...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(file=sys.stderr)

    # Start server (blocks and reads from stdin)
    try:
        transport.start()
        return 0
    except Exception as e:
        print(f"❌ Server error: {e}", file=sys.stderr)
        return 1
