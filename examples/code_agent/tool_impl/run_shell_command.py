# SPDX-License-Identifier: Apache-2.0
"""
run_shell_command tool (shell) - Executes shell commands.

Based on gemini-cli's shell.ts implementation.
Supports foreground and background execution, timeout handling, and process management.
"""

import shlex
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.sandbox import BaseSandbox, SandboxStatus

from .sandbox_utils import get_sandbox, resolve_path

# Configuration constants
DEFAULT_TIMEOUT_MS = 300000  # 5 minutes default timeout


def _format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}TB"


def run_shell_command(
    command: str,
    description: str | None = None,
    dir_path: str | None = None,
    is_background: bool = False,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    update_output: Callable[[str], None] | None = None,
    agent_state: AgentState | None = None,
) -> dict[str, Any]:
    """
    Executes a shell command.

    On Unix/Linux/macOS: Executes as `bash -c <command>`
    On Windows: Executes as `powershell.exe -NoProfile -Command <command>`

    The following information is returned:
    - Output: Combined stdout/stderr. Can be `(empty)` or partial on error.
    - Exit Code: Only included if non-zero (command failed).
    - Error: Only included if a process-level error occurred.
    - Signal: Only included if process was terminated by a signal.
    - Background PIDs: Only included if background processes were started.
    - Process Group PGID: Only included if available.

    Args:
        command: The exact command to execute
        description: Brief description of the command for the user
        dir_path: Directory to run the command in (optional)
        is_background: Whether to run in background
        timeout_ms: Timeout in milliseconds (0 for no timeout)
        update_output: Callback for streaming output updates

    Returns:
        Dict with llmContent and returnDisplay matching gemini-cli format
    """
    try:
        # Validate command
        if not command or not command.strip():
            return {
                "llmContent": "Command cannot be empty.",
                "returnDisplay": "Error: Empty command.",
                "error": {
                    "message": "Command cannot be empty.",
                    "type": "INVALID_COMMAND",
                },
            }

        sandbox: BaseSandbox = get_sandbox(agent_state)

        # Background execution is intentionally not supported here because
        # BaseSandbox.execute_bash is a foreground/awaited API.
        if is_background:
            msg = "Background execution is not supported by the current sandbox bash API."
            return {
                "llmContent": msg,
                "returnDisplay": msg,
                "error": {"message": msg, "type": "UNSUPPORTED_BACKGROUND"},
            }

        # Determine working directory (string resolution only; checks via sandbox)
        if dir_path:
            cwd = resolve_path(dir_path, sandbox)
            if not sandbox.file_exists(cwd):
                error_msg = f"Directory not found: {dir_path}"
                return {
                    "llmContent": error_msg,
                    "returnDisplay": "Error: Directory not found.",
                    "error": {"message": error_msg, "type": "DIRECTORY_NOT_FOUND"},
                }
            info = sandbox.get_file_info(cwd)
            if not info.is_directory:
                error_msg = f"Path is not a directory: {dir_path}"
                return {
                    "llmContent": error_msg,
                    "returnDisplay": "Error: Path is not a directory.",
                    "error": {"message": error_msg, "type": "NOT_A_DIRECTORY"},
                }
        else:
            cwd = str(sandbox.work_dir)

        # Build description for display
        cmd_description = command
        if dir_path:
            cmd_description += f" [in {dir_path}]"
        else:
            cmd_description += f" [current working directory {cwd}]"
        if description:
            cmd_description += f" ({description.replace(chr(10), ' ')})"
        # Streaming output is not supported by execute_bash; ignore update_output.
        _ = update_output

        # Execute command through sandbox, optionally scoping to directory via `cd`.
        cmd_to_run = command
        if cwd:
            cmd_to_run = f"cd {shlex.quote(cwd)} && {command}"

        start = time.time()
        timeout_arg = timeout_ms if timeout_ms and timeout_ms > 0 else None
        cmd_result = sandbox.execute_bash(cmd_to_run, timeout=timeout_arg)
        duration_ms = int((time.time() - start) * 1000)

        stdout = cmd_result.stdout or ""
        stderr = cmd_result.stderr or ""
        output = stdout
        if stderr:
            output = f"{stdout}\n{stderr}" if stdout else stderr

        exit_code = cmd_result.exit_code
        error_message = cmd_result.error

        # Build result
        llm_parts = []
        if cmd_result.status == SandboxStatus.TIMEOUT:
            timeout_minutes = (timeout_ms / 60000) if timeout_ms else 0
            llm_parts.append(f"Timeout: command timed out after {timeout_minutes:.1f} minutes.")
        else:
            llm_parts.append(f"Output: {output if output else '(empty)'}")

        if error_message:
            llm_parts.append(f"Error: {error_message}")

        if exit_code is not None and exit_code != 0:
            llm_parts.append(f"Exit Code: {exit_code}")

        llm_content = "\n".join(llm_parts)

        # Build return display
        if output and output.strip():
            return_display = output
        elif cmd_result.status == SandboxStatus.TIMEOUT:
            return_display = f"Command timed out after {timeout_ms / 60000:.1f} minutes."
        elif error_message:
            return_display = f"Command failed: {error_message}"
        elif exit_code is not None and exit_code != 0:
            return_display = f"Command exited with code: {exit_code}"
        else:
            return_display = "(empty)"

        result = {
            "llmContent": llm_content,
            "returnDisplay": return_display,
            "duration_ms": duration_ms,
        }

        if error_message or cmd_result.status in (SandboxStatus.ERROR, SandboxStatus.TIMEOUT):
            result["error"] = {
                "message": error_message or "Command failed",
                "type": "SHELL_EXECUTE_ERROR",
            }

        return result

    except Exception as e:
        error_msg = f"Error executing shell command: {str(e)}"
        return {
            "llmContent": error_msg,
            "returnDisplay": error_msg,
            "error": {
                "message": error_msg,
                "type": "SHELL_EXECUTE_ERROR",
            },
        }
