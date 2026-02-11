"""NexAU Unified CLI package."""

from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run NexAU CLI entrypoint with lazy import."""
    from nexau.cli.main import main as cli_main

    return cli_main(argv)
