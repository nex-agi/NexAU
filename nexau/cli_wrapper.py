# Copyright (c) Nex-AGI. All rights reserved.
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

"""
Wrapper script to launch the Node.js nexau-cli from the Python package installation.

This script locates the installed Node.js CLI files and executes them with node,
allowing users to run 'nexau-cli <yaml-file>' after installing the Python package.
"""

import subprocess
import sys
from pathlib import Path

from nexau.cli.entrypoint_checks import MISSING_WINDOWS_SHELL_EXIT_CODE, ensure_default_windows_shell_for_entrypoint
from nexau.cli.run_agent import ensure_cli_built


def _source_tree_cli_paths() -> tuple[Path, Path] | None:
    """Return ``(cli_dir, cli_dist)`` when running from a source checkout."""
    repo_root = Path(__file__).resolve().parent.parent
    cli_dir = repo_root / "cli"
    cli_source = cli_dir / "source" / "cli.js"
    cli_dist = cli_dir / "dist" / "cli.js"
    if cli_dir.is_dir() and cli_source.is_file():
        return cli_dir, cli_dist
    return None


def find_node_cli():
    """Find the Node.js CLI executable in the package installation."""
    # Try several possible locations where the CLI might be installed

    # Location 1: Original development location (relative to this file)
    script_dir = Path(__file__).parent.parent  # Go up to project root
    dev_cli_path = script_dir / "cli" / "dist" / "cli.js"
    if dev_cli_path.exists():
        return dev_cli_path

    # Location 2: In the same directory as this script (development alternative)
    script_dir = Path(__file__).parent
    dev_cli_path = script_dir / "cli" / "dist" / "cli.js"
    if dev_cli_path.exists():
        return dev_cli_path

    # Location 3: In package data (installed package)
    # When the package is installed, the CLI files should be in package data
    import nexau

    package_dir = Path(nexau.__file__).parent
    installed_cli_path = package_dir / "cli" / "dist" / "cli.js"
    if installed_cli_path.exists():
        return installed_cli_path

    # Location 4: Probe interpreter-managed install roots without hardcoding
    # Unix-only paths or a specific Python minor version.
    import sysconfig

    possible_paths: list[Path] = []
    for scheme_name in ("purelib", "platlib"):
        scheme_path = sysconfig.get_path(scheme_name)
        if scheme_path:
            base_path = Path(scheme_path)
            possible_paths.append(base_path / "nexau-cli" / "cli" / "dist" / "cli.js")
            possible_paths.append(base_path / "nexau" / "cli" / "dist" / "cli.js")

    for path in possible_paths:
        if path.exists():
            return path

    # Location 5: Try to use importlib.resources to access package data
    try:
        import importlib.resources as resources

        # Get the package directory
        package_dir = resources.files("nexau")
        cli_resource = package_dir.joinpath("cli", "dist", "cli.js")
        if cli_resource.is_file():
            with resources.as_file(cli_resource) as cli_path:
                return cli_path
    except Exception:
        # Other error, skip
        pass

    return None


def resolve_node_cli() -> Path | None:
    """Resolve the Node CLI, auto-building the source-tree dist artifact if needed."""
    existing_cli = find_node_cli()
    if existing_cli is not None:
        return existing_cli

    source_tree_paths = _source_tree_cli_paths()
    if source_tree_paths is None:
        return None

    cli_dir, cli_dist = source_tree_paths
    return ensure_cli_built(cli_dir=cli_dir, cli_dist=cli_dist)


def main():
    """Main entry point for the nexau-cli wrapper."""
    try:
        ensure_default_windows_shell_for_entrypoint()
    except RuntimeError:
        sys.exit(MISSING_WINDOWS_SHELL_EXIT_CODE)

    # Find or build the Node.js CLI
    try:
        node_cli_path = resolve_node_cli()
    except subprocess.CalledProcessError as exc:
        print(f"Error: Failed to build nexau Node.js CLI (exit code {exc.returncode}).", file=sys.stderr)
        sys.exit(exc.returncode or 1)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not node_cli_path:
        print("Error: Could not locate nexau Node.js CLI executable.", file=sys.stderr)
        print("Make sure the nexau package is properly installed.", file=sys.stderr)
        sys.exit(1)

    # Check if Node.js is available
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Node.js is not installed or not in PATH.", file=sys.stderr)
        print("Please install Node.js (>=16) to use nexau-cli.", file=sys.stderr)
        sys.exit(1)

    # Build the command to execute
    cmd = ["node", str(node_cli_path)] + sys.argv[1:]

    # Execute the Node.js CLI
    try:
        sys.exit(subprocess.run(cmd).returncode)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error executing nexau-cli: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
