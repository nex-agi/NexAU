"""NexAU Unified CLI - Main dispatcher with chat and serve commands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from nexau.cli.commands import chat, http, stdio


def create_parser() -> argparse.ArgumentParser:
    """Create the main CLI parser with subcommands.

    Returns:
        Configured ArgumentParser with all subcommands
    """
    parser = argparse.ArgumentParser(
        prog="nexau",
        description="NexAU Agent Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,
    )

    # Chat command: Interactive agent chat
    chat_parser = subparsers.add_parser(
        "chat",
        help="Run agent in interactive chat mode",
    )
    chat.setup_parser(chat_parser)
    chat_parser.set_defaults(func=chat.main)

    # Serve command group: Server transport operations
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start agent server with different transports",
    )
    serve_subparsers = serve_parser.add_subparsers(
        dest="transport",
        required=True,
        help="Transport type",
    )

    # Serve HTTP: Start HTTP/SSE server
    serve_http = serve_subparsers.add_parser(
        "http",
        help="Start HTTP server with SSE streaming support",
    )
    http.setup_parser(serve_http)
    serve_http.set_defaults(func=http.main)

    # Serve STDIO: Start STDIO server
    serve_stdio = serve_subparsers.add_parser(
        "stdio",
        help="Start STDIO server (reads stdin, writes stdout)",
    )
    stdio.setup_server_parser(serve_stdio)
    serve_stdio.set_defaults(func=stdio.server_main)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main entry point.

    Args:
        argv: Command line arguments (uses sys.argv if None)

    Returns:
        Exit code (0 = success, non-0 = error)
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # If no command provided, show help
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # Dispatch to appropriate command function
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
