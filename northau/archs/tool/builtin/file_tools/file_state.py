# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
File State Management Module

This module provides centralized file state management for file tools,
including timestamp tracking for read/write coordination.
"""

import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

# Global file timestamp cache for coordinating read/write operations
_file_timestamps: Dict[str, float] = {}


def update_file_timestamp(file_path: str) -> None:
    """
    Update file timestamp cache for read/write coordination.

    This function should be called when a file is successfully read or written
    to maintain consistency between file_read_tool and file_write_tool.

    Args:
        file_path: Absolute path to the file
    """
    try:
        if os.path.exists(file_path):
            _file_timestamps[file_path] = os.path.getmtime(file_path)
            logger.debug(f"Updated timestamp cache for: {file_path}")
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


def validate_file_read_state(file_path: str) -> tuple[bool, str | None]:
    """
    Validate if file is in a safe state for writing.

    This function checks if:
    1. File doesn't exist (new file - OK to write)
    2. File exists and has been read (OK to write)
    3. File exists but hasn't been read (NOT OK - need to read first)
    4. File was modified after last read (NOT OK - need to re-read)

    Args:
        file_path: Absolute path to the file

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return True, None

    # Check if file has been read
    cached_timestamp = get_file_timestamp(file_path)
    if cached_timestamp == 0.0:
        return False, "文件尚未被读取。请先读取文件再进行写入操作。"

    # Check if file was modified after last read
    try:
        current_mtime = os.path.getmtime(file_path)
        if current_mtime > cached_timestamp:
            return (
                False,
                "文件在读取后已被修改（可能是用户手动修改或被其他工具修改）。请重新读取文件后再进行写入。",
            )
    except Exception as e:
        logger.warning(f"Failed to check file modification time for {file_path}: {e}")

    return True, None


def clear_file_timestamps():
    """Clear all cached file timestamps. Mainly for testing purposes."""
    global _file_timestamps
    _file_timestamps.clear()


def get_timestamp_cache_info() -> Dict[str, float]:
    """Get copy of current timestamp cache. Mainly for debugging purposes."""
    return _file_timestamps.copy()