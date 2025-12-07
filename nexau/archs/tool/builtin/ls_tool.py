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

"""LS tool implementation for listing files and directories."""

import fnmatch
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _to_json(payload: dict[str, Any]) -> str:
    """Serialize payload to JSON for tool response."""
    return json.dumps(payload, ensure_ascii=False)


def _match_ignore_patterns(path: str, ignore_patterns: list[str]) -> bool:
    """
    Check if a path matches any of the ignore patterns.

    Args:
        path: The file/directory path to check
        ignore_patterns: List of glob patterns to ignore

    Returns:
        True if the path should be ignored, False otherwise
    """
    if not ignore_patterns:
        return False

    path_obj = Path(path)

    for pattern in ignore_patterns:
        # Check both the full path and just the filename
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(
            path_obj.name,
            pattern,
        ):
            return True

    return False


def _get_file_info(path: str) -> dict[str, Any]:
    """
    Get detailed information about a file or directory.

    Args:
        path: Path to the file or directory

    Returns:
        Dict containing file information
    """
    try:
        stat_info = os.stat(path)
        path_obj = Path(path)

        info: dict[str, Any] = {
            "name": path_obj.name,
            "path": str(path_obj.absolute()),
            "is_file": path_obj.is_file(),
            "is_dir": path_obj.is_dir(),
            "size": stat_info.st_size if path_obj.is_file() else None,
            "modified_time": stat_info.st_mtime,
            "permissions": oct(stat_info.st_mode)[-3:],
        }

        # Add additional info for files
        if path_obj.is_file():
            info["extension"] = path_obj.suffix.lower()

        # Add item count for directories
        if path_obj.is_dir():
            try:
                items = list(path_obj.iterdir())
                info["item_count"] = len(items)
            except (OSError, PermissionError):
                info["item_count"] = None

        return info

    except (OSError, PermissionError) as e:
        logger.warning(f"Could not get info for {path}: {e}")
        return {
            "name": Path(path).name,
            "path": str(Path(path).absolute()),
            "error": str(e),
            "is_file": False,
            "is_dir": False,
        }


def ls_tool(
    path: str,
    ignore: list[str] | None = None,
) -> str:
    """
    List files and directories in a given path.

    Args:
        path: The absolute path to the directory to list (must be absolute, not relative)
        ignore: Optional list of glob patterns to ignore

    Returns:
        JSON string containing directory listing results
    """
    start_time = time.time()

    # Validate path is absolute
    if not os.path.isabs(path):
        return _to_json(
            {
                "status": "error",
                "error": f"Path must be absolute, got relative path: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    # Check if path exists
    if not os.path.exists(path):
        return _to_json(
            {
                "status": "error",
                "error": f"Path does not exist: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    # Check if it's a directory
    if not os.path.isdir(path):
        return _to_json(
            {
                "status": "error",
                "error": f"Path is not a directory: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    # Check permissions
    if not os.access(path, os.R_OK):
        return _to_json(
            {
                "status": "error",
                "error": f"No read permission for directory: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    try:
        # Get directory contents
        items: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        directories: list[dict[str, Any]] = []
        ignored_count = 0
        error_count = 0

        path_obj = Path(path)

        for item_path in path_obj.iterdir():
            try:
                # Check if item should be ignored
                if ignore and _match_ignore_patterns(str(item_path), ignore):
                    ignored_count += 1
                    continue

                # Get file info
                info = _get_file_info(str(item_path))

                if "error" in info:
                    error_count += 1

                items.append(info)

                # Categorize items
                if info.get("is_file"):
                    files.append(info)
                elif info.get("is_dir"):
                    directories.append(info)

            except (OSError, PermissionError) as e:
                logger.warning(f"Could not access item in {path}: {e}")
                error_count += 1
                continue

        # Sort items: directories first, then files, both alphabetically
        items.sort(
            key=lambda x: (
                not x.get("is_dir", False),
                x.get("name", "").lower(),
            ),
        )

        duration_ms = int((time.time() - start_time) * 1000)

        # Prepare result
        result: dict[str, Any] = {
            "status": "success",
            "path": path,
            "total_items": len(items),
            "directories": len(directories),
            "files": len(files),
            "ignored_items": ignored_count,
            "error_items": error_count,
            "duration_ms": duration_ms,
            "items": items,
            "ignore_patterns": ignore or [],
        }
        result_message = ""

        if len(items) == 0:
            if ignored_count > 0:
                result_message = f"Directory is empty (ignored {ignored_count} items)"
            else:
                result_message = "Directory is empty"
        else:
            result_message = f"Found {len(directories)} directories and {len(files)} files"
            if ignored_count > 0:
                result_message += f" (ignored {ignored_count} items)"
            if error_count > 0:
                result_message += f" (errors accessing {error_count} items)"

        logger.info(
            f"LS completed: listed {len(items)} items in {duration_ms}ms",
        )

        result["message"] = result_message

        # Apply length limit to JSON output
        result_json = json.dumps(result, ensure_ascii=False)
        if len(result_json) > 10000:
            # Calculate how many items to keep to stay under limit
            items_to_keep = len(items)
            while items_to_keep >= 0:
                truncated_result: dict[str, Any] = dict(result)
                truncated_result["items"] = items[:items_to_keep]
                truncated_result["total_items"] = items_to_keep
                truncated_result["truncated_output"] = True
                truncated_result["remaining_items"] = len(items) - items_to_keep
                truncated_result["message"] = f"{result_message} (Output truncated: showing {items_to_keep} of {len(items)} items)"
                serialized = json.dumps(truncated_result, ensure_ascii=False)
                if len(serialized) <= 10000:
                    return serialized
                items_to_keep -= 1

            minimal_result: dict[str, Any] = {
                "status": "success",
                "path": path,
                "total_items": len(items),
                "directories": len(directories),
                "files": len(files),
                "truncated_output": True,
                "remaining_items": len(items),
                "message": (f"Output too long: found {len(directories)} directories and {len(files)} files (details truncated)"),
                "duration_ms": duration_ms,
                "ignore_patterns": ignore or [],
            }
            return _to_json(minimal_result)

        return result_json

    except PermissionError as e:
        return _to_json(
            {
                "status": "error",
                "error": f"Permission denied accessing directory: {str(e)}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error during ls: {e}")
        return _to_json(
            {
                "status": "error",
                "error": f"Unexpected error listing directory: {str(e)}",
                "error_type": type(e).__name__,
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )
