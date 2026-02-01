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

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from nexau.archs.sandbox import BaseSandbox, LocalSandbox

logger = logging.getLogger(__name__)


def glob_tool(
    pattern: str,
    path: str | None = None,
    limit: int = 100,
    sandbox: BaseSandbox | None = None,
) -> str:
    """
    Search for files using glob patterns. This tool allows you to find files that match
    specific patterns in the filesystem. Supports standard glob syntax including:
    - * matches any sequence of characters (except path separators)
    - ? matches any single character
    - [seq] matches any character in seq
    - ** matches directories recursively

    Examples:
    - '*.py' - all Python files in the current directory
    - '**/*.tsx' - all TypeScript React files recursively
    - 'src/**/*.js' - all JavaScript files in src directory recursively
    - 'test_*.py' - all test files starting with 'test_'
    """
    start_time = time.time()

    # Get sandbox instance
    sandbox = sandbox or LocalSandbox(_work_dir=os.getcwd())

    # Determine the search directory
    search_dir = path if path else str(Path.cwd())

    # Validate that the directory exists
    try:
        if not sandbox.file_exists(search_dir):
            return json.dumps(
                {
                    "error": f"Directory does not exist: {search_dir}",
                    "num_files": 0,
                    "filenames": [],
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "truncated": False,
                },
                indent=2,
            )

        # Get file info to check if it's a directory
        file_info = sandbox.get_file_info(search_dir)
        if not file_info.is_directory:
            return json.dumps(
                {
                    "error": f"Path is not a directory: {search_dir}",
                    "num_files": 0,
                    "filenames": [],
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "truncated": False,
                },
                indent=2,
            )

        if not file_info.readable:
            return json.dumps(
                {
                    "error": f"No read permission for directory: {search_dir}",
                    "num_files": 0,
                    "filenames": [],
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "truncated": False,
                },
                indent=2,
            )
    except Exception as e:
        logger.error(f"Error checking directory permissions: {e}")
        return json.dumps(
            {
                "error": f"Error accessing directory: {str(e)}",
                "num_files": 0,
                "filenames": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "truncated": False,
            },
            indent=2,
        )

    try:
        # Construct full pattern path
        if Path(pattern).is_absolute():
            full_pattern = pattern
        else:
            full_pattern = f"{search_dir}/{pattern}"

        # Perform glob search through sandbox
        matches = sandbox.glob(full_pattern, recursive=True)

        # Filter to only include files (not directories)
        files: list[str] = []
        for match in matches:
            try:
                info = sandbox.get_file_info(match)
                if info.is_file:
                    files.append(match)
            except Exception:
                # Skip files we can't access
                continue

        # Sort files for consistent output
        files.sort()

        # Apply limit and check if truncated
        truncated = len(files) > limit
        if truncated:
            files = files[:limit]

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Prepare result
        result: dict[str, Any] = {
            "num_files": len(files),
            "filenames": files,
            "duration_ms": duration_ms,
            "truncated": truncated,
            "search_directory": search_dir,
            "pattern": pattern,
        }

        # Add truncation message if needed
        if truncated:
            result["message"] = f"Results are truncated to {limit} files. Consider using a more specific pattern."
        elif len(files) == 0:
            result["message"] = "No files found matching the pattern."
        else:
            result["message"] = f"Found {len(files)} file{'s' if len(files) != 1 else ''} matching the pattern."

        logger.info(
            f"Glob search completed: found {len(files)} files in {duration_ms}ms",
        )

        # Apply length limit to JSON output
        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        if len(result_json) > 10000:
            # Calculate how many files to keep to stay under limit
            files_to_keep = len(files)
            while files_to_keep > 0:
                truncated_result: dict[str, Any] = result.copy()
                truncated_result["filenames"] = files[:files_to_keep]
                truncated_result["num_files"] = files_to_keep
                truncated_result["truncated_output"] = True
                truncated_result["remaining_files"] = (
                    len(
                        files,
                    )
                    - files_to_keep
                )
                truncated_result["message"] = f"Found {files_to_keep} files (truncated: {len(files) - files_to_keep} more files not shown)"

                test_json = json.dumps(
                    truncated_result,
                    indent=2,
                    ensure_ascii=False,
                )
                if len(test_json) <= 10000:
                    return test_json
                files_to_keep -= 1

            # If even 0 files is too long, return minimal result
            minimal_result: dict[str, Any] = {
                "truncated_output": True,
                "total_files": len(files),
                "message": f"Output too long: found {len(files)} files (details truncated)",
                "search_directory": search_dir,
                "pattern": pattern,
                "duration_ms": duration_ms,
            }
            return json.dumps(minimal_result, indent=2, ensure_ascii=False)

        return result_json

    except Exception as e:
        logger.error(f"Error during glob search: {e}")
        return json.dumps(
            {
                "error": f"Error during file search: {str(e)}",
                "num_files": 0,
                "filenames": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "truncated": False,
            },
            indent=2,
        )
