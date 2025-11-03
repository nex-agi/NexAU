#!/usr/bin/env python3
"""
Wrapper script to launch the Node.js northau-cli from the Python package installation.

This script locates the installed Node.js CLI files and executes them with node,
allowing users to run 'northau-cli <yaml-file>' after installing the Python package.
"""

import subprocess
import sys
from pathlib import Path


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
    import northau

    package_dir = Path(northau.__file__).parent
    installed_cli_path = package_dir / "cli" / "dist" / "cli.js"
    if installed_cli_path.exists():
        return installed_cli_path

    # Location 4: Check if node_modules is available and CLI is in expected location
    # This handles the case where the package is installed with pip but CLI is in a known location
    possible_paths = [
        Path(sys.prefix) / "lib" / "python3.12" / "site-packages" / "northau-cli" / "cli" / "dist" / "cli.js",
        Path(sys.prefix) / "local" / "lib" / "python3.12" / "site-packages" / "northau-cli" / "cli" / "dist" / "cli.js",
        Path("/usr/local/lib/python3.12/site-packages/northau-cli/cli/dist/cli.js"),
        Path("/usr/lib/python3.12/site-packages/northau-cli/cli/dist/cli.js"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Location 5: Try to use importlib.resources to access package data
    try:
        import importlib.resources as resources

        # Get the package directory
        package_dir = resources.files("northau")
        # Try to access the CLI file
        cli_path = package_dir / "cli" / "dist" / "cli.js"
        if cli_path.exists():
            return cli_path
    except Exception:
        # Other error, skip
        pass

    return None


def main():
    """Main entry point for the northau-cli wrapper."""
    # Find the Node.js CLI
    node_cli_path = find_node_cli()

    if not node_cli_path:
        print("Error: Could not locate northau Node.js CLI executable.", file=sys.stderr)
        print("Make sure the northau package is properly installed.", file=sys.stderr)
        sys.exit(1)

    # Check if Node.js is available
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Node.js is not installed or not in PATH.", file=sys.stderr)
        print("Please install Node.js (>=16) to use northau-cli.", file=sys.stderr)
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
        print(f"Error executing northau-cli: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
