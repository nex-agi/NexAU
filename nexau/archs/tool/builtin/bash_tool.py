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

"""Bash command execution tool implementation."""

import logging
import time
from typing import (
    Literal,
    NotRequired,
    TypedDict,
)

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.sandbox import BaseSandbox, SandboxStatus

logger = logging.getLogger(__name__)

# Maximum output length to prevent overwhelming responses
MAX_OUTPUT_LENGTH = 30000
DEFAULT_TIMEOUT = 120000  # 2 minutes in milliseconds
MAX_TIMEOUT = 6000000  # 100 minutes in milliseconds


class BashResult(TypedDict):
    status: Literal["success", "error", "timeout"]
    duration_ms: int
    command: str
    exit_code: NotRequired[int | None]
    working_directory: NotRequired[str]
    description: NotRequired[str]
    stdout: NotRequired[str]
    stdout_truncated: NotRequired[bool]
    stdout_original_length: NotRequired[int]
    stderr: NotRequired[str]
    stderr_truncated: NotRequired[bool]
    stderr_original_length: NotRequired[int]
    error: NotRequired[str]
    error_type: NotRequired[str]


def bash_tool(
    command: str,
    timeout: int | None = None,
    description: str | None = None,
    agent_state: AgentState | None = None,
) -> BashResult:
    """
    Execute a bash command in a persistent shell session with proper handling and security measures.

    Args:
        command: The bash command to execute (required)
        timeout: Optional timeout in milliseconds (max 6000000ms / 100 minutes)
        description: Clear, concise description of what this command does in 5-10 words
        agent_state: AgentState containing agent context and global storage

    Returns:
        Dict containing execution results
    """
    start_time = time.time()

    # Get sandbox instance
    assert agent_state is not None, "File operation tool invoked, but agent_state is not passed. We need sandbox instance in agent_state."
    sandbox: BaseSandbox | None = agent_state.get_sandbox()
    assert sandbox is not None, "System operation tool invoked, but sandbox is not initialized."

    # Validate timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    elif timeout > MAX_TIMEOUT:
        return {
            "status": "error",
            "error": f"Timeout cannot exceed {MAX_TIMEOUT}ms (100 minutes)",
            "command": command,
            "duration_ms": 0,
        }

    # Validate command
    if not command or not command.strip():
        return {
            "status": "error",
            "error": "Command cannot be empty",
            "command": command,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    # Security warnings for potentially dangerous commands
    dangerous_patterns = [
        "rm -rf /",
        "sudo rm",
        "rm -rf ~",
        "mkfs",
        "fdisk",
        "dd if=",
        "> /dev/",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
    ]

    command_lower = command.lower().strip()
    for pattern in dangerous_patterns:
        if pattern in command_lower:
            logger.warning(
                f"Potentially dangerous command detected: {command}",
            )
            return {
                "status": "error",
                "error": f"Command contains potentially dangerous pattern: {pattern}",
                "command": command,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

    try:
        # Execute command through sandbox
        cmd_result = sandbox.execute_bash(command, timeout)

        duration_ms = cmd_result.duration_ms

        # Determine status
        status: Literal["success", "error", "timeout"]
        if cmd_result.status == SandboxStatus.TIMEOUT:
            status = "timeout"
        elif cmd_result.status == SandboxStatus.SUCCESS:
            status = "success"
        else:
            status = "error"

        # Prepare output
        result: BashResult = {
            "status": status,
            "command": command,
            "exit_code": cmd_result.exit_code,
            "duration_ms": duration_ms,
            "working_directory": str(sandbox.work_dir),
        }

        # Add description if provided
        if description:
            result["description"] = description

        # Handle stdout
        if cmd_result.stdout:
            result["stdout"] = cmd_result.stdout
            result["stdout_truncated"] = cmd_result.truncated and cmd_result.original_stdout_length is not None
            if result["stdout_truncated"] and cmd_result.original_stdout_length is not None:
                result["stdout_original_length"] = cmd_result.original_stdout_length
        else:
            result["stdout"] = ""
            result["stdout_truncated"] = False

        # Handle stderr
        if cmd_result.stderr:
            result["stderr"] = cmd_result.stderr
            result["stderr_truncated"] = cmd_result.truncated and cmd_result.original_stderr_length is not None
            if result["stderr_truncated"] and cmd_result.original_stderr_length is not None:
                result["stderr_original_length"] = cmd_result.original_stderr_length
        else:
            result["stderr"] = ""
            result["stderr_truncated"] = False

        # Add error message if present
        if cmd_result.error:
            result["error"] = cmd_result.error

        # Log execution
        logger.info(
            f"Bash command executed: '{command}' (exit_code={cmd_result.exit_code}, duration={duration_ms}ms)",
        )

        return result

    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "command": command,
            "duration_ms": int((time.time() - start_time) * 1000),
        }
