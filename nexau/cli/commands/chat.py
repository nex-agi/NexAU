"""Chat CLI command - Interactive agent chat with persistence."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from nexau import Agent
from nexau.archs.main_sub.config import ConfigError
from nexau.archs.main_sub.config.config import AgentConfigBuilder
from nexau.archs.main_sub.utils import load_yaml_with_vars
from nexau.archs.session import SessionManager, SQLDatabaseEngine
from nexau.cli.agent_runner import create_cli_progress_hook, create_cli_tool_hook
from nexau.cli.agent_runner import send_message as cli_send_message
from nexau.cli.cli_subagent_adapter import attach_cli_to_agent


def setup_parser(parser: argparse.ArgumentParser) -> None:
    """Configure arguments for Chat command.

    Args:
        parser: ArgumentParser instance
    """
    parser.add_argument(
        "agent",
        type=str,
        help="Path to agent YAML configuration file",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Optional query to run (non-interactive mode)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="cli_user",
        help="User identifier for session management (default: cli_user)",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Session identifier for continuity (default: auto-generated)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (debug level logging)",
    )


def build_agent(config_path: Path, session_manager: SessionManager, user_id: str, session_id: str | None) -> Agent:
    """Build agent from YAML configuration with session support.

    Args:
        config_path: Path to YAML configuration file
        session_manager: SessionManager for persistence
        user_id: User identifier
        session_id: Optional session identifier

    Returns:
        Configured Agent instance with session support

    Raises:
        ConfigError: If configuration is invalid
    """

    def build_agent_from_config() -> Agent:
        config_dict = load_yaml_with_vars(config_path)
        if not isinstance(config_dict, dict):
            raise ConfigError(f"Configuration file must be a YAML object/mapping, got {type(config_dict).__name__}")
        agent_config = AgentConfigBuilder(config_dict, base_path=config_path.parent).build_system_prompt_path().get_agent_config()

        # Create agent with session support
        return Agent(
            config=agent_config,
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
        )

    return build_agent_from_config()


def main_interactive(config_path: Path, session_manager: SessionManager, user_id: str, session_id: str | None) -> int:
    """Run agent in interactive mode with persistence.

    Args:
        config_path: Path to agent configuration
        session_manager: SessionManager for persistence
        user_id: User identifier
        session_id: Optional session identifier

    Returns:
        Exit code (0 = success, non-0 = error)
    """
    try:
        agent = build_agent(config_path, session_manager, user_id, session_id)

        # Attach CLI hooks
        cli_progress_hook = create_cli_progress_hook()
        cli_tool_hook = create_cli_tool_hook()

        def handle_subagent_event(event_type: str, payload: dict) -> None:
            event_mapping = {
                "start": ("subagent_start", "message"),
                "complete": ("subagent_complete", "result"),
                "error": ("subagent_error", "error"),
            }
            if event_type not in event_mapping:
                return
            message_type, content_field = event_mapping[event_type]
            content = payload.get(content_field, "")
            cli_send_message(message_type, content, metadata=payload)

        attach_cli_to_agent(agent, cli_progress_hook, cli_tool_hook, handle_subagent_event)

        print("Agent loaded successfully. Type /clear to reset, Ctrl+C to exit.")
        print(f"User ID: {user_id}")
        print(f"Session ID: {agent._session_id}")
        print("Database: ~/.nexau/nexau.db")
        print("=" * 60)

        # Interactive loop
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue

                # Handle /clear command
                if query == "/clear":
                    print("Re-initializing agent with new session...")
                    # Create new agent with new session (session_id=None)
                    agent = build_agent(config_path, session_manager, user_id, None)
                    attach_cli_to_agent(agent, cli_progress_hook, cli_tool_hook, handle_subagent_event)
                    print(f"Agent re-initialized. New session ID: {agent._session_id}")
                    continue

                # Run agent
                response = agent.run(
                    message=query,
                    context={
                        "date": "2025-01-06",  # Simplified
                        "username": user_id,
                        "working_directory": ".",
                    },
                )

                print("\n" + "=" * 60)
                print("Response:")
                print(response)
                print("=" * 60)

            except KeyboardInterrupt:
                print("\n\nExiting...")
                return 0

    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main_non_interactive(config_path: Path, query: str, session_manager: SessionManager, user_id: str, session_id: str | None) -> int:
    """Run agent in non-interactive mode (single query) with persistence.

    Args:
        config_path: Path to agent configuration
        query: Query to run
        session_manager: SessionManager for persistence
        user_id: User identifier
        session_id: Optional session identifier

    Returns:
        Exit code (0 = success, non-0 = error)
    """
    try:
        agent = build_agent(config_path, session_manager, user_id, session_id)

        # Run agent
        response = agent.run(
            message=query,
            context={
                "date": "2025-01-06",
                "username": user_id,
                "working_directory": ".",
            },
        )

        print(response)
        return 0

    except ConfigError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main(args: argparse.Namespace) -> int:
    """Execute Chat command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 = success, non-0 = error)
    """
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    config_path = Path(args.agent)

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    # Create SQLite engine for persistence
    db_path = Path.home() / ".nexau" / "nexau.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = SQLDatabaseEngine.from_url(f"sqlite+aiosqlite:///{db_path}")

    # Create session manager
    session_manager = SessionManager(engine=engine)

    if args.query:
        # Non-interactive mode
        return main_non_interactive(config_path, args.query, session_manager, args.user_id, args.session_id)
    else:
        # Interactive mode
        return main_interactive(config_path, session_manager, args.user_id, args.session_id)
