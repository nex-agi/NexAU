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

from nexau.archs.sandbox import BaseSandbox, LocalSandbox

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


def ls_tool(
    path: str,
    ignore: list[str] | None = None,
    sandbox: BaseSandbox | None = None,
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

    # Get sandbox instance
    sandbox = sandbox or LocalSandbox(_work_dir=os.getcwd())

    # Validate path is absolute
    if not Path(path).is_absolute():
        return _to_json(
            {
                "status": "error",
                "error": f"Path must be absolute, got relative path: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    # Check if path exists
    if not sandbox.file_exists(path):
        return _to_json(
            {
                "status": "error",
                "error": f"Path does not exist: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    # Get file info to check if it's a directory
    try:
        file_info = sandbox.get_file_info(path)
    except Exception as e:
        return _to_json(
            {
                "status": "error",
                "error": f"Failed to get file info: {str(e)}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )
    # Check if it's a directory
    if not file_info.is_directory:
        return _to_json(
            {
                "status": "error",
                "error": f"Path is not a directory: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
        )

    try:
        # Get directory contents through sandbox
        file_infos = sandbox.list_files(path, recursive=False)

        items: list[dict[str, Any]] = []
        files: list[dict[str, Any]] = []
        directories: list[dict[str, Any]] = []
        ignored_count = 0
        error_count = 0

        for file_info in file_infos:
            try:
                # Check if item should be ignored
                if ignore and _match_ignore_patterns(file_info.path, ignore):
                    ignored_count += 1
                    continue

                # Convert FileInfo to dict format
                info: dict[str, Any] = {
                    "name": Path(file_info.path).name,
                    "path": file_info.path,
                    "is_file": file_info.is_file,
                    "is_dir": file_info.is_directory,
                    "size": file_info.size if file_info.is_file else None,
                    "modified_time": file_info.modified_time,
                    "permissions": file_info.permissions,
                }

                items.append(info)

                # Categorize items
                if info.get("is_file"):
                    files.append(info)
                elif info.get("is_dir"):
                    directories.append(info)

            except Exception as e:
                logger.warning(f"Could not process item in {path}: {e}")
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
