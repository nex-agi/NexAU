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

import difflib
import importlib
import json
import logging
import os
import time
from typing import Any

from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.sandbox import BaseSandbox, SandboxStatus

from .file_format_validators import validate_csv_format
from .file_state import update_file_timestamp, validate_file_read_state
from .file_type_utils import is_binary_extension

logger = logging.getLogger(__name__)

# Maximum lines to display in preview
MAX_LINES_TO_RENDER = 10
MAX_LINES_TO_RENDER_FOR_ASSISTANT = 16000


def _detect_file_encoding(file_path: str, sandbox: Any) -> str:
    """Detect file encoding."""
    try:
        chardet = importlib.import_module("chardet")
        result = sandbox.read_file(file_path, binary=True)
        if result.status == SandboxStatus.SUCCESS and result.content:
            if isinstance(result.content, bytes):
                raw_data = result.content[:10000]
            elif isinstance(result.content, str):
                raw_data = result.content.encode()[:10000]
            else:
                return "utf-8"
            detection = chardet.detect(raw_data)
            return detection.get("encoding", "utf-8") or "utf-8"
    except ModuleNotFoundError:
        pass
    except Exception:
        pass
    # If chardet is unavailable or detection fails, default to utf-8
    return "utf-8"


def _detect_line_endings(file_path: str, sandbox: Any) -> str:
    """Detect file line endings."""
    try:
        result = sandbox.read_file(file_path, binary=True)
        if result.status == SandboxStatus.SUCCESS and result.content:
            if isinstance(result.content, bytes):
                content = result.content[:1024]
            elif isinstance(result.content, str):
                content = result.content.encode()[:1024]
            else:
                return "\n"
            if b"\r\n" in content:
                return "\r\n"
            elif b"\n" in content:
                return "\n"
            elif b"\r" in content:
                return "\r"
    except Exception:
        pass
    return "\n"  # Default to \n


def _has_write_permission(file_path: str, sandbox: Any) -> bool:
    """Check if we have write permission."""
    try:
        # Check if file exists
        if sandbox.file_exists(file_path):
            file_info = sandbox.get_file_info(file_path)
            return file_info.writable

        # Check if directory exists and is writable
        directory = os.path.dirname(file_path)
        if sandbox.file_exists(directory):
            dir_info = sandbox.get_file_info(directory)
            return dir_info.writable

        # Directory doesn't exist, assume we can create it
        return True
    except Exception:
        return False


def _generate_diff(old_content: str, new_content: str, file_path: str) -> str:
    """Generate diff comparison."""
    try:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{os.path.basename(file_path)}",
            tofile=f"b/{os.path.basename(file_path)}",
            lineterm="",
        )

        return "".join(diff)
    except Exception as e:
        logger.error(f"Failed to generate diff: {e}")
        return ""


def _write_file_content(
    file_path: str,
    content: str,
    sandbox: Any,
    encoding: str = "utf-8",
    line_ending: str = "\n",
) -> None:
    """Write file content."""
    # Normalize line endings
    if line_ending != "\n":
        content = content.replace("\n", line_ending)

    # Write file through sandbox
    result = sandbox.write_file(file_path, content, encoding=encoding, binary=False, create_directories=True)
    if result.status != SandboxStatus.SUCCESS:
        raise Exception(result.error or "Failed to write file")


def file_write_tool(
    file_path: str,
    content: str,
    agent_state: AgentState,
) -> str:
    """
    Write content to a file in the local file system. Overwrites if file exists.

    Before using this tool:
    1. Use ReadFile tool to understand the file content and context
    2. Directory validation (only for new files):
       - Use LS tool to verify parent directory exists and is correct

    Features:
    - Auto-detect and preserve file encoding
    - Auto-detect and preserve line ending format
    - File modification timestamp validation to prevent conflicts
    - Generate detailed diff comparison
    - Support creating directory structure
    - Permission check and security validation

    Returns:
    - JSON string with operation result on success
    - JSON string with error info on failure
    """
    start_time = time.time()

    # Get sandbox instance
    sandbox: BaseSandbox | None = agent_state.get_sandbox()
    assert sandbox is not None, "File operation tool invoked, but sandbox is not initialized."

    try:
        # Validate file path
        if not os.path.isabs(file_path):
            return json.dumps(
                {
                    "error": "File path must be absolute",
                    "file_path": file_path,
                    "success": False,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Check write permission
        if not _has_write_permission(file_path, sandbox):
            return json.dumps(
                {
                    "error": f"No write permission: {file_path}",
                    "file_path": file_path,
                    "success": False,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Check if binary format (cannot write in text mode)
        if is_binary_extension(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            return json.dumps(
                {
                    "error": f"Cannot create binary format file with Write tool ({ext})",
                    "file_path": file_path,
                    "success": False,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # CSV format validation
        if file_path.endswith(".csv"):
            is_valid, error_msg = validate_csv_format(content)
            if not is_valid:
                return json.dumps(
                    {
                        "error": error_msg,
                        "hint": "Use pandas: df.to_csv('file.csv', index=False) or csv module",
                        "file_path": file_path,
                        "success": False,
                        "duration_ms": int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

        # Check if file exists
        file_exists = sandbox.file_exists(file_path)
        operation_type = "update" if file_exists else "create"

        # Read original content and detect encoding
        old_content = ""
        encoding = "utf-8"
        line_ending = "\n"

        if file_exists:
            # Validate file state
            is_valid, error_msg_nullable = validate_file_read_state(file_path)
            if not is_valid:
                error_msg = error_msg_nullable or "File state validation failed"
                return json.dumps(
                    {
                        "error": error_msg,
                        "file_path": file_path,
                        "success": False,
                        "duration_ms": int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            # Detect file encoding and line endings
            encoding = _detect_file_encoding(file_path, sandbox)
            line_ending = _detect_line_endings(file_path, sandbox)

            # Read original content
            try:
                read_result = sandbox.read_file(file_path, encoding=encoding)
                if read_result.status != SandboxStatus.SUCCESS:
                    raise Exception(read_result.error or "Failed to read file")
                if isinstance(read_result.content, str):
                    old_content = read_result.content
                elif isinstance(read_result.content, bytes):
                    old_content = read_result.content.decode(encoding)
                else:
                    old_content = ""
            except Exception as e:
                logger.error(f"Failed to read original file: {e}")
                return json.dumps(
                    {
                        "error": f"Failed to read original file: {str(e)}",
                        "file_path": file_path,
                        "success": False,
                        "duration_ms": int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

        # Write file
        try:
            _write_file_content(file_path, content, sandbox, encoding, line_ending)
        except Exception as e:
            logger.error(f"Failed to write file: {e}")
            return json.dumps(
                {
                    "error": f"Failed to write file: {str(e)}",
                    "file_path": file_path,
                    "success": False,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Update timestamp cache
        update_file_timestamp(file_path, sandbox)

        # Generate diff (for update operations)
        diff_content = ""
        if operation_type == "update" and old_content != content:
            diff_content = _generate_diff(old_content, content, file_path)

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Statistics
        content_lines = content.split("\n")
        num_lines = len(content_lines)

        # Prepare result
        result: dict[str, Any] = {
            "success": True,
            "operation_type": operation_type,
            "file_path": file_path,
            "num_lines": num_lines,
            "encoding": encoding,
            "line_ending": ("CRLF" if line_ending == "\r\n" else "LF" if line_ending == "\n" else "CR"),
            "duration_ms": duration_ms,
        }

        # Add diff info (only for updates)
        if operation_type == "update":
            result["has_changes"] = old_content != content
            if diff_content:
                result["diff"] = diff_content
                # Count changed lines
                diff_lines = diff_content.split("\n")
                added_lines = len(
                    [line for line in diff_lines if line.startswith("+")],
                )
                removed_lines = len(
                    [line for line in diff_lines if line.startswith("-")],
                )
                result["changes"] = {
                    "lines_added": added_lines,
                    "lines_removed": removed_lines,
                }

        # Add content preview (limited lines)
        if num_lines <= MAX_LINES_TO_RENDER:
            result["content_preview"] = content
        else:
            preview_lines = content_lines[:MAX_LINES_TO_RENDER]
            result["content_preview"] = "\n".join(preview_lines)
            result["content_truncated"] = True
            result["truncated_lines"] = num_lines - MAX_LINES_TO_RENDER

        # Success message
        if operation_type == "create":
            result["message"] = f"Successfully created file, {num_lines} lines"
        else:
            if old_content == content:
                result["message"] = "File content unchanged"
            else:
                result["message"] = f"Successfully updated file, {num_lines} lines"

        logger.info(f"File {operation_type} succeeded: {file_path} ({duration_ms}ms)")

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"File write tool execution failed: {e}")
        return json.dumps(
            {
                "error": f"Tool execution failed: {str(e)}",
                "file_path": file_path,
                "success": False,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )
