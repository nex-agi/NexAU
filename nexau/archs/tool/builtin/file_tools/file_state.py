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
File State Management Module

This module provides centralized file state management for file tools,
including timestamp tracking for read/write coordination.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Global file timestamp cache for coordinating read/write operations
_file_timestamps: dict[str, float] = {}


def update_file_timestamp(file_path: str, sandbox: Any = None) -> None:
    """
    Update file timestamp cache for read/write coordination.

    This function should be called when a file is successfully read or written
    to maintain consistency between file_read_tool and file_write_tool.

    Args:
        file_path: Absolute path to the file
        sandbox: Optional sandbox instance for file operations
    """
    global _file_timestamps
    try:
        if sandbox:
            if sandbox.file_exists(file_path):
                file_info = sandbox.get_file_info(file_path)
                _file_timestamps[file_path] = file_info.modified_time
                logger.debug(f"Updated timestamp cache for: {file_path}")
            elif file_path in _file_timestamps:
                # Remove from cache if file doesn't exist
                del _file_timestamps[file_path]
        else:
            # Fallback: just mark as accessed without actual timestamp
            _file_timestamps[file_path] = 0.0
    except Exception as e:
        logger.warning(f"Failed to update file timestamp for {file_path}: {e}")


def get_file_timestamp(file_path: str) -> float:
    """
    Get cached file timestamp.

    Args:
        file_path: Absolute path to the file

    Returns:
        Cached timestamp or 0.0 if not found
    """
    return _file_timestamps.get(file_path, 0.0)


def has_file_timestamp(file_path: str) -> bool:
    """
    Check if file has cached timestamp (i.e., was previously read).

    Args:
        file_path: Absolute path to the file

    Returns:
        True if file has cached timestamp, False otherwise
    """
    return file_path in _file_timestamps


def validate_file_read_state(file_path: str, sandbox: Any = None) -> tuple[bool, str | None]:
    """
    Validate if file is in a safe state for writing.

    This function checks if:
    1. File doesn't exist (new file - OK to write)
    2. File exists and has been read (OK to write)
    3. File exists but hasn't been read (NOT OK - need to read first)
    4. File was modified after last read (NOT OK - need to re-read)

    Args:
        file_path: Absolute path to the file
        sandbox: Optional sandbox instance for file operations

    Returns:
        Tuple of (is_valid, error_message)
    """
    if sandbox:
        if not sandbox.file_exists(file_path):
            return True, None
    else:
        # If no sandbox, assume file state is valid
        return True, None

    # Check if file has been read
    cached_timestamp = get_file_timestamp(file_path)
    if cached_timestamp == 0.0:
        return False, "文件尚未被读取。请先读取文件再进行写入操作。"

    # Check if file was modified after last read
    try:
        file_info = sandbox.get_file_info(file_path)
        current_mtime = file_info.modified_time
        if current_mtime > cached_timestamp:
            return (
                False,
                "文件在读取后已被修改（可能是用户手动修改或被其他工具修改）。请重新读取文件后再进行写入。",
            )
    except Exception as e:
        logger.warning(
            f"Failed to check file modification time for {file_path}: {e}",
        )

    return True, None


def clear_file_timestamps():
    """Clear all cached file timestamps. Mainly for testing purposes."""
    global _file_timestamps
    _file_timestamps = {}


def get_timestamp_cache_info() -> dict[str, float]:
    """Get copy of current timestamp cache. Mainly for debugging purposes."""
    return _file_timestamps.copy()
