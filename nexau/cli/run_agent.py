#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-platform run-agent launcher.

RFC-0019: CLI 入口与默认 PowerShell backend 闭环

Replaces the previous bash-only `run-agent` script with a Python launcher that
works on Windows and Unix alike, verifies the default Windows shell backend,
and builds/runs the bundled Node CLI without depending on shell wrappers.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from nexau.cli.entrypoint_checks import MISSING_WINDOWS_SHELL_EXIT_CODE, ensure_default_windows_shell_for_entrypoint

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CLI_DIR = _PROJECT_ROOT / "cli"
_CLI_DIST = _CLI_DIR / "dist" / "cli.js"
_CLI_SOURCES = (
    _CLI_DIR / "source" / "app.js",
    _CLI_DIR / "source" / "cli.js",
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the run-agent launcher."""
    parser = argparse.ArgumentParser(
        prog="run-agent",
        description="Build and run the NexAU agent CLI",
    )
    parser.add_argument("config", help="Path to the agent YAML configuration file")
    return parser


def cli_needs_build(cli_dist: Path = _CLI_DIST, cli_sources: Sequence[Path] = _CLI_SOURCES) -> bool:
    """Return True when the bundled Node CLI must be rebuilt."""
    if not cli_dist.exists():
        return True

    dist_mtime_ns = cli_dist.stat().st_mtime_ns
    for source_path in cli_sources:
        if not source_path.exists() or source_path.stat().st_mtime_ns > dist_mtime_ns:
            return True
    return False


def _resolve_node_binary() -> str:
    """Return the Node.js binary path or raise a user-facing error."""
    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("Node.js is not installed or not in PATH. Install Node.js (>=16) to run the NexAU CLI.")
    return node_path


def _resolve_npm_binary() -> str:
    """Return the npm binary path or raise a user-facing error."""
    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("npm is not installed or not in PATH. Install Node.js (>=16) to build the NexAU CLI.")
    return npm_path


def ensure_cli_built(cli_dir: Path = _CLI_DIR, cli_dist: Path = _CLI_DIST) -> Path:
    """Build the bundled Node CLI when the dist artifact is missing or stale."""
    if not cli_dir.is_dir():
        raise RuntimeError("CLI directory not found")

    print("Checking if CLI needs to be built...")
    if cli_needs_build(cli_dist=cli_dist):
        npm_binary = _resolve_npm_binary()
        print("Building CLI...")
        subprocess.run([npm_binary, "install"], cwd=cli_dir, check=True)
        subprocess.run([npm_binary, "run", "build"], cwd=cli_dir, check=True)
    else:
        print("CLI already built and up-to-date")

    if not cli_dist.exists():
        raise RuntimeError(f"CLI build did not produce expected artifact: {cli_dist}")
    return cli_dist


def run_node_cli(config_path: Path, cli_path: Path) -> int:
    """Launch the bundled Node CLI with dotenv loaded from the project root."""
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
    node_binary = _resolve_node_binary()
    env = dict(os.environ)
    result = subprocess.run(
        [node_binary, str(cli_path), str(config_path)],
        cwd=_PROJECT_ROOT,
        env=env,
        check=False,
    )
    return result.returncode


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the cross-platform run-agent launcher."""
    parser = create_parser()
    args = parser.parse_args(argv)

    print("NexAU Agent Runner")

    config_path = Path(args.config).expanduser()
    if not config_path.exists() or not config_path.is_file():
        print(f"Error: Config file '{args.config}' not found", file=sys.stderr)
        return 1
    resolved_config_path = config_path.resolve()

    try:
        ensure_default_windows_shell_for_entrypoint()
    except RuntimeError:
        return MISSING_WINDOWS_SHELL_EXIT_CODE

    try:
        cli_path = ensure_cli_built()
    except subprocess.CalledProcessError as exc:
        print(f"Error: CLI build failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Running agent with config: {resolved_config_path}")
    return run_node_cli(resolved_config_path, cli_path)


if __name__ == "__main__":
    raise SystemExit(main())
