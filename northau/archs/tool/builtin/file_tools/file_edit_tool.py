# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
File Edit Tool - A tool for safely editing files with comprehensive validation.

This tool provides safe file editing capabilities with features like:
- Create, update, and delete file operations
- String matching and replacement validation
- File encoding and line ending detection
- Timestamp-based conflict detection
- Comprehensive error handling and user feedback

Based on the TypeScript FileEditTool implementation.
"""

import difflib
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Tuple, Union


# Import file state management for read/write coordination
from .file_state import (
    update_file_timestamp,
    get_file_timestamp,
    has_file_timestamp,
    clear_file_timestamps,
)

logger = logging.getLogger(__name__)


def detect_file_encoding(file_path: str) -> str:
    """
    Detect file encoding with fallback to utf-8.

    Args:
        file_path: Path to the file

    Returns:
        Detected encoding name
    """
    try:
        import chardet

        with open(file_path, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result["encoding"]
            if encoding and result["confidence"] > 0.7:
                return encoding
    except ImportError:
        # chardet not available, use fallback
        pass
    except Exception as e:
        logger.warning(f"Error detecting encoding for {file_path}: {e}")

    # Fallback to utf-8
    return "utf-8"


def detect_line_endings(file_path: str) -> str:
    """
    Detect line ending style in a file.

    Args:
        file_path: Path to the file

    Returns:
        Line ending style: 'CRLF', 'LF', or 'CR'
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()

        if b"\r\n" in content:
            return "CRLF"
        elif b"\n" in content:
            return "LF"
        elif b"\r" in content:
            return "CR"
        else:
            return "LF"  # Default for new files
    except Exception as e:
        logger.warning(f"Error detecting line endings for {file_path}: {e}")
        return "LF"


def find_similar_file(file_path: str) -> Optional[str]:
    """
    Find a similar file with different extension if the original doesn't exist.

    Args:
        file_path: Path to the file that doesn't exist

    Returns:
        Path to similar file if found, None otherwise
    """
    try:
        base_path = Path(file_path)
        base_name = base_path.stem
        parent_dir = base_path.parent

        if not parent_dir.exists():
            return None

        # Look for files with same base name but different extensions
        for file in parent_dir.glob(f"{base_name}.*"):
            if file.is_file() and str(file) != file_path:
                return str(file)

        return None
    except Exception as e:
        logger.warning(f"Error finding similar file for {file_path}: {e}")
        return None


def write_text_content(
    file_path: str, content: str, encoding: str = "utf-8", line_ending: str = "LF"
) -> None:
    """
    Write text content to file with specified encoding and line endings.

    Args:
        file_path: Target file path
        content: Content to write
        encoding: File encoding
        line_ending: Line ending style ('LF', 'CRLF', 'CR')
    """
    # Convert line endings if needed
    if line_ending == "CRLF":
        content = content.replace("\n", "\r\n")
    elif line_ending == "CR":
        content = content.replace("\n", "\r")
    # LF is default, no conversion needed

    with open(file_path, "w", encoding=encoding, newline="") as f:
        f.write(content)


def apply_edit(
    file_path: str, old_string: str, new_string: str
) -> Tuple[str, List[Dict]]:
    """
    Apply edit operation and generate diff information.

    Args:
        file_path: Path to the file
        old_string: String to replace
        new_string: Replacement string

    Returns:
        Tuple of (updated_content, diff_info)
    """
    # Read original content
    if os.path.exists(file_path):
        encoding = detect_file_encoding(file_path)
        with open(file_path, "r", encoding=encoding) as f:
            original_content = f.read()
    else:
        original_content = ""

    # Apply replacement
    if old_string == "":
        # Create new file
        updated_content = new_string
    elif new_string == "":
        # Delete content
        updated_content = original_content.replace(old_string, "", 1)
    else:
        # Update content
        updated_content = original_content.replace(old_string, new_string, 1)

    # Generate diff
    diff_info = []
    if original_content != updated_content:
        original_lines = original_content.splitlines(keepends=True)
        updated_lines = updated_content.splitlines(keepends=True)

        # Create unified diff
        diff = list(
            difflib.unified_diff(
                original_lines,
                updated_lines,
                fromfile=f"a/{os.path.basename(file_path)}",
                tofile=f"b/{os.path.basename(file_path)}",
                lineterm="",
            )
        )

        if diff:
            diff_info = [{"type": "unified_diff", "content": "\n".join(diff)}]

    return updated_content, diff_info


def get_snippet_with_context(
    original_content: str, old_string: str, new_string: str, context_lines: int = 4
) -> Tuple[str, int]:
    """
    Get a snippet of the file showing the change with context.

    Args:
        original_content: Original file content
        old_string: String that was replaced
        new_string: Replacement string
        context_lines: Number of context lines to include

    Returns:
        Tuple of (snippet, start_line_number)
    """
    if not original_content and old_string == "":
        # New file creation
        new_lines = new_string.split("\n")
        snippet_lines = new_lines[: context_lines * 2 + 1]
        return "\n".join(snippet_lines), 1

    # Find the replacement position
    before_replacement = original_content.split(old_string)[0] if old_string else ""
    replacement_line = before_replacement.count("\n")

    # Generate updated content
    updated_content = (
        original_content.replace(old_string, new_string, 1)
        if old_string
        else new_string
    )
    updated_lines = updated_content.split("\n")

    # Calculate snippet bounds
    start_line = max(0, replacement_line - context_lines)
    end_line = min(
        len(updated_lines),
        replacement_line + context_lines + new_string.count("\n") + 1,
    )

    # Extract snippet
    snippet_lines = updated_lines[start_line:end_line]
    snippet = "\n".join(snippet_lines)

    return snippet, start_line + 1


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Add line numbers to content.

    Args:
        content: Content to add line numbers to
        start_line: Starting line number

    Returns:
        Content with line numbers
    """
    lines = content.split("\n")
    numbered_lines = []

    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:6d}\t{line}")

    return "\n".join(numbered_lines)


def file_edit_tool(
    file_path: str,
    old_string: str,
    new_string: str,
) -> str:
    """
    A comprehensive file editing tool that safely modifies files with extensive validation.

    This tool supports three main operations:
    1. CREATE: Set old_string to empty string to create a new file
    2. UPDATE: Provide both old_string and new_string to update existing content
    3. REMOVE_CONTENT: Set new_string to empty string to remove the old_string content from the file

    Safety features:
    - Validates that the old_string exists exactly once in the file
    - Checks file timestamps to prevent conflicts
    - Detects file encoding and line endings automatically
    - Provides detailed error messages and suggestions
    - Creates necessary parent directories
    - Preserves file permissions

    Examples:
    - Create file: file_path="new_file.py", old_string="", new_string="print('hello')"
    - Update code: file_path="script.py", old_string="old_function()", new_string="new_function()"
    - Remove content: file_path="config.txt", old_string="debug=true\n", new_string=""

    Note: For safety, this tool only replaces exactly one occurrence of old_string.
    If multiple matches exist, add more context to make the match unique.
    """
    start_time = time.time()

    try:
        # Validate input
        if old_string == new_string:
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "No changes to make: old_string and new_string are exactly the same."
                    ),
                    "file_path": file_path,
                    "operation": "none",
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
            )

        # Resolve absolute path
        abs_file_path = os.path.abspath(file_path)

        # Determine operation type
        if old_string == "":
            operation = "create"
        elif new_string == "":
            operation = "remove_content"  # More accurate than 'delete'
        else:
            operation = "update"

        # Validate file existence based on operation
        file_exists = os.path.exists(abs_file_path)

        if operation == "create" and file_exists:
            return json.dumps(
                {
                    "success": False,
                    "error": "Cannot create new file - file already exists.",
                    "file_path": file_path,
                    "operation": operation,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
            )

        if operation in ["update", "remove_content"] and not file_exists:
            # Try to find a similar file
            similar_file = find_similar_file(abs_file_path)
            error_msg = "File does not exist."
            if similar_file:
                error_msg += f" Did you mean {similar_file}?"

            return json.dumps(
                {
                    "success": False,
                    "error": error_msg,
                    "file_path": file_path,
                    "operation": operation,
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "suggestion": similar_file,
                },
                indent=2,
            )

        # Check for Jupyter notebooks
        if abs_file_path.endswith(".ipynb"):
            return json.dumps(
                {
                    "success": False,
                    "error": (
                        "File is a Jupyter Notebook. Use a specialized notebook editing tool for .ipynb files."
                    ),
                    "file_path": file_path,
                    "operation": operation,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
            )

        # For existing files, perform additional validations
        if file_exists:
            # Check read timestamp (if tracking is enabled)
            read_timestamp = get_file_timestamp(abs_file_path)
            if read_timestamp == 0.0:
                # File hasn't been read yet, this might be okay for some operations
                # but we'll issue a warning
                logger.warning(f"File {abs_file_path} has not been read yet")
            else:
                # Check if file was modified since last read
                try:
                    file_mtime = os.path.getmtime(abs_file_path)
                    if file_mtime > read_timestamp:
                        return json.dumps(
                            {
                                "success": False,
                                "error": (
                                    "File has been modified since it was last read, either by the user or by a linter. Read it again before attempting to edit it."
                                ),
                                "file_path": file_path,
                                "operation": operation,
                                "duration_ms": int((time.time() - start_time) * 1000),
                            },
                            indent=2,
                        )
                except OSError as e:
                    logger.warning(f"Could not check file modification time: {e}")

            # Validate string matching for update/remove_content operations
            if operation in ["update", "remove_content"]:
                encoding = detect_file_encoding(abs_file_path)
                try:
                    with open(abs_file_path, "r", encoding=encoding) as f:
                        file_content = f.read()
                except UnicodeDecodeError as e:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Cannot read file due to encoding issue: {str(e)}"
                            ),
                            "file_path": file_path,
                            "operation": operation,
                            "duration_ms": int((time.time() - start_time) * 1000),
                        },
                        indent=2,
                    )

                # Check if old_string exists in file
                if old_string not in file_content:
                    return json.dumps(
                        {
                            "success": False,
                            "error": "String to replace not found in file.",
                            "file_path": file_path,
                            "operation": operation,
                            "duration_ms": int((time.time() - start_time) * 1000),
                        },
                        indent=2,
                    )

                # Check for multiple matches (safety feature)
                matches = file_content.count(old_string)
                if matches > 1:
                    return json.dumps(
                        {
                            "success": False,
                            "error": (
                                f"Found {matches} matches of the string to replace. For safety, this tool only supports replacing exactly one occurrence at a time. Add more lines of context to your edit and try again."
                            ),
                            "file_path": file_path,
                            "operation": operation,
                            "duration_ms": int((time.time() - start_time) * 1000),
                            "matches_found": matches,
                        },
                        indent=2,
                    )

        # Perform the edit operation
        try:
            # Create parent directories if needed
            parent_dir = os.path.dirname(abs_file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            # Apply the edit
            updated_content, diff_info = apply_edit(
                abs_file_path, old_string, new_string
            )

            # Detect or use default encoding and line endings
            if file_exists:
                encoding = detect_file_encoding(abs_file_path)
                line_ending = detect_line_endings(abs_file_path)
            else:
                encoding = "utf-8"
                line_ending = "LF"

            # Write the updated content
            write_text_content(abs_file_path, updated_content, encoding, line_ending)

            # Update read timestamp
            update_file_timestamp(abs_file_path)

            # Generate result snippet
            original_content = ""
            if file_exists:
                try:
                    with open(abs_file_path, "r", encoding=encoding) as f:
                        # Read the original content before our edit
                        if old_string:
                            original_content = updated_content.replace(
                                new_string, old_string, 1
                            )
                        else:
                            original_content = ""
                except Exception:
                    original_content = ""

            snippet, start_line = get_snippet_with_context(
                original_content, old_string, new_string
            )
            snippet_with_numbers = add_line_numbers(snippet, start_line)

            duration_ms = int((time.time() - start_time) * 1000)

            # Generate appropriate success message
            if operation == "create":
                message = f"The file {file_path} has been created successfully."
            elif operation == "remove_content":
                message = f"Content has been removed from {file_path} successfully."
            else:  # update
                message = f"The file {file_path} has been updated successfully."

            result = {
                "success": True,
                "message": message,
                "file_path": file_path,
                "operation": operation,
                "duration_ms": duration_ms,
                "encoding": encoding,
                "line_ending": line_ending,
                "snippet": {
                    "content": snippet_with_numbers,
                    "start_line": start_line,
                    "description": (
                        f"Here's the result of running `cat -n` on a snippet of the edited file:"
                    ),
                },
            }

            if diff_info:
                result["diff"] = diff_info

            # Log successful operation
            logger.info(
                f"File {operation} operation completed successfully: {abs_file_path}"
            )

            return json.dumps(result, indent=2, ensure_ascii=False)

        except PermissionError:
            return json.dumps(
                {
                    "success": False,
                    "error": f"Permission denied: Cannot write to {file_path}",
                    "file_path": file_path,
                    "operation": operation,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
            )

        except OSError as e:
            return json.dumps(
                {
                    "success": False,
                    "error": f"OS error: {str(e)}",
                    "file_path": file_path,
                    "operation": operation,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
            )

        except Exception as e:
            logger.error(f"Unexpected error during file {operation}: {e}")
            return json.dumps(
                {
                    "success": False,
                    "error": f"Unexpected error: {str(e)}",
                    "file_path": file_path,
                    "operation": operation,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
            )

    except Exception as e:
        logger.error(f"Critical error in file_edit_tool: {e}")
        return json.dumps(
            {
                "success": False,
                "error": f"Critical error: {str(e)}",
                "file_path": file_path,
                "operation": "unknown",
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
        )


# Utility functions for external use
def mark_file_as_read(file_path: str) -> None:
    """
    Mark a file as read by updating its read timestamp.
    This should be called whenever a file is read to enable proper conflict detection.

    Args:
        file_path: Path to the file that was read
    """
    update_file_timestamp(file_path)


def clear_read_timestamps() -> None:
    """Clear all read timestamps. Useful for testing or resetting state."""
    clear_file_timestamps()


def get_file_read_status(file_path: str) -> Dict[str, Union[bool, float, str]]:
    """
    Get the read status of a file.

    Args:
        file_path: Path to check

    Returns:
        Dictionary with read status information
    """
    abs_path = os.path.abspath(file_path)
    read_timestamp = get_file_timestamp(file_path)

    result = {
        "file_path": abs_path,
        "has_been_read": has_file_timestamp(file_path),
        "read_timestamp": read_timestamp if read_timestamp > 0.0 else None,
    }

    if os.path.exists(abs_path):
        try:
            file_mtime = os.path.getmtime(abs_path)
            result["file_mtime"] = file_mtime
            result["modified_since_read"] = (
                read_timestamp > 0.0 and file_mtime > read_timestamp
            )
        except OSError:
            result["file_mtime"] = None
            result["modified_since_read"] = None
    else:
        result["file_exists"] = False

    return result


if __name__ == "__main__":
    # Example usage - demonstrating different operations

    # 创建新文件
    print("=== 创建新文件 ===")
    result = file_edit_tool.invoke(
        {"file_path": "test.py", "old_string": "", "new_string": "print('Hello World')"}
    )
    print(result)
    print()
    # breakpoint()
    # 标记文件为已读（模拟读取操作）
    mark_file_as_read("test.py")

    # 更新现有文件
    print("=== 更新现有文件 ===")
    result = file_edit_tool.invoke(
        {
            "file_path": "test.py",
            "old_string": "print('Hello World')",
            "new_string": "print('Hello Python')",
        }
    )
    print(result)
    print()
    # breakpoint()
    # 再次标记为已读
    mark_file_as_read("test.py")

    # 删除内容
    print("=== 删除内容 ===")
    result = file_edit_tool.invoke(
        {
            "file_path": "test.py",
            "old_string": "print('Hello Python')",
            "new_string": "",
        }
    )
    print(result)
    print()
    # 清理测试文件
    # breakpoint()
    try:
        os.remove("test.py")
        print("清理完成：删除了测试文件 test.py")
    except FileNotFoundError:
        pass