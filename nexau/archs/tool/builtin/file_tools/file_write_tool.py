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

from .file_format_validators import validate_csv_format
from .file_state import update_file_timestamp, validate_file_read_state
from .file_type_utils import is_binary_extension

logger = logging.getLogger(__name__)

# Maximum lines to display in preview
MAX_LINES_TO_RENDER = 10
MAX_LINES_TO_RENDER_FOR_ASSISTANT = 16000


def _detect_file_encoding(file_path: str) -> str:
    """Detect file encoding."""
    try:
        chardet = importlib.import_module("chardet")
        with open(file_path, "rb") as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result.get("encoding", "utf-8") or "utf-8"
    except ModuleNotFoundError:
        return "utf-8"
    except Exception:
        # If chardet is unavailable or detection fails, default to utf-8
        return "utf-8"


def _detect_line_endings(file_path: str) -> str:
    """Detect file line endings."""
    try:
        with open(file_path, "rb") as f:
            content = f.read(1024)  # Read first 1KB
            if b"\r\n" in content:
                return "\r\n"
            elif b"\n" in content:
                return "\n"
            elif b"\r" in content:
                return "\r"
    except Exception:
        pass
    return "\n"  # Default to \n


def _has_write_permission(file_path: str) -> bool:
    """Check if we have write permission."""
    try:
        # Check if directory exists and is writable
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            # Check if we can create the directory
            parent = os.path.dirname(directory)
            while parent and not os.path.exists(parent):
                parent = os.path.dirname(parent)
            return os.access(parent, os.W_OK) if parent else False

        # Check if file exists
        if os.path.exists(file_path):
            return os.access(file_path, os.W_OK)
        else:
            # Check if directory is writable
            return os.access(directory, os.W_OK)
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
    encoding: str = "utf-8",
    line_ending: str = "\n",
) -> None:
    """Write file content."""
    # Normalize line endings
    if line_ending != "\n":
        content = content.replace("\n", line_ending)

    # Create directory
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Write file
    with open(file_path, "w", encoding=encoding, newline="") as f:
        f.write(content)


def file_write_tool(
    file_path: str,
    content: str,
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
        if not _has_write_permission(file_path):
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
        file_exists = os.path.exists(file_path)
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
            encoding = _detect_file_encoding(file_path)
            line_ending = _detect_line_endings(file_path)

            # Read original content
            try:
                with open(file_path, encoding=encoding) as f:
                    old_content = f.read()
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
            _write_file_content(file_path, content, encoding, line_ending)
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
        update_file_timestamp(file_path)

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


# Class implementation for advanced usage
class FileWriteTool:
    """
    Class implementation of file write tool with advanced features and config options.
    """

    def __init__(
        self,
        max_lines_preview: int = MAX_LINES_TO_RENDER,
        auto_create_dirs: bool = True,
        check_permissions: bool = True,
    ):
        """
        Initialize file write tool.

        Args:
            max_lines_preview: Maximum lines to show in preview.
            auto_create_dirs: Whether to auto-create directories.
            check_permissions: Whether to check permissions.
        """
        self.max_lines_preview = max_lines_preview
        self.auto_create_dirs = auto_create_dirs
        self.check_permissions = check_permissions
        self.logger = logging.getLogger(self.__class__.__name__)

    def write_file(
        self,
        file_path: str,
        content: str,
        encoding: str | None = None,
        line_ending: str | None = None,
    ) -> dict[str, Any]:
        """
        Write file and return structured result.

        Args:
            file_path: File path.
            content: File content.
            encoding: Encoding (optional).
            line_ending: Line ending (optional).

        Returns:
            Dict containing operation result.
        """
        start_time = time.time()

        try:
            # Validate path
            if not os.path.isabs(file_path):
                raise ValueError("File path must be absolute")

            # Check permissions
            if self.check_permissions and not _has_write_permission(file_path):
                raise PermissionError(f"No write permission: {file_path}")

            # Check file state
            file_exists = os.path.exists(file_path)
            operation_type = "update" if file_exists else "create"

            # Handle encoding and line endings
            if file_exists:
                if encoding is None:
                    encoding = _detect_file_encoding(file_path)
                if line_ending is None:
                    line_ending = _detect_line_endings(file_path)
            else:
                encoding = encoding or "utf-8"
                line_ending = line_ending or "\n"

            # Read original content
            old_content = ""
            if file_exists:
                with open(file_path, encoding=encoding) as f:
                    old_content = f.read()

            # Write file
            _write_file_content(file_path, content, encoding, line_ending)

            # Update timestamp
            update_file_timestamp(file_path)

            # Generate result
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "operation_type": operation_type,
                "file_path": file_path,
                "num_lines": len(content.split("\n")),
                "encoding": encoding,
                "line_ending": line_ending,
                "duration_ms": duration_ms,
                "has_changes": old_content != content if file_exists else True,
            }

        except Exception as e:
            self.logger.error(f"File write failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }


def main():
    """Test function."""
    # Test create file
    test_file = "//users/chenlu//src/tools/file_tools/test_file_write.txt"
    test_content = "Hello, World!\nThis is a test file.\nLine 3"

    result = file_write_tool(file_path=test_file, content=test_content)
    print("Create file test:")
    print(result)
    print()

    # Update file timestamp (simulate read)
    update_file_timestamp(test_file)

    # Test update file
    updated_content = "Hello, World!\nThis is an updated test file.\nLine 3\nNew line 4"
    result = file_write_tool(file_path=test_file, content=updated_content)
    print("Update file test:")
    print(result)

    # Cleanup test file
    try:
        os.remove(test_file)
    except Exception:
        pass


if __name__ == "__main__":
    main()
