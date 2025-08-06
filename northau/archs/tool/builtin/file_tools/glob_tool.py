import glob as python_glob
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


def glob_tool(
    pattern: str,
    path: Optional[str] = None,
    limit: int = 100,
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

    # Determine the search directory
    search_dir = path if path else os.getcwd()

    # Validate that the directory exists and is readable
    try:
        if not os.path.exists(search_dir):
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

        if not os.access(search_dir, os.R_OK):
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
        # Change to the search directory to perform glob search
        original_cwd = os.getcwd()
        os.chdir(search_dir)

        # Perform glob search
        matches = python_glob.glob(pattern, recursive=True)

        # Convert to absolute paths and filter out directories
        files = []
        for match in matches:
            abs_path = os.path.abspath(match)
            if os.path.isfile(abs_path):
                files.append(abs_path)

        # Sort files for consistent output
        files.sort()

        # Apply limit and check if truncated
        truncated = len(files) > limit
        if truncated:
            files = files[:limit]

        # Calculate duration
        duration_ms = int((time.time() - start_time) * 1000)

        # Prepare result
        result = {
            "num_files": len(files),
            "filenames": files,
            "duration_ms": duration_ms,
            "truncated": truncated,
            "search_directory": search_dir,
            "pattern": pattern,
        }

        # Add truncation message if needed
        if truncated:
            result["message"] = (
                f"Results are truncated to {limit} files. Consider using a more specific pattern."
            )
        elif len(files) == 0:
            result["message"] = "No files found matching the pattern."
        else:
            result["message"] = (
                f"Found {len(files)} file{'s' if len(files) != 1 else ''} matching the pattern."
            )

        logger.info(
            f"Glob search completed: found {len(files)} files in {duration_ms}ms"
        )

        return json.dumps(result, indent=2, ensure_ascii=False)

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

    finally:
        # Always restore the original working directory
        try:
            os.chdir(original_cwd)
        except Exception as e:
            logger.warning(f"Failed to restore original working directory: {e}")


# # Alternative class-based implementation for more advanced use cases
# class GlobSearchTool:
#     """
#     A class-based implementation of the glob search tool for more advanced usage.
#     """

#     def __init__(self, default_limit: int = 100):
#         self.default_limit = default_limit
#         self.logger = logging.getLogger(self.__class__.__name__)

#     def search(self, pattern: str, path: Optional[str] = None, limit: Optional[int] = None) -> Dict:
#         """
#         Perform a glob search and return structured results.

#         Args:
#             pattern: The glob pattern to match files against
#             path: The directory to search in (defaults to current working directory)
#             limit: Maximum number of files to return (defaults to instance default)

#         Returns:
#             Dictionary containing search results
#         """
#         start_time = time.time()
#         limit = limit if limit is not None else self.default_limit
#         search_dir = path if path else os.getcwd()

#         try:
#             # Validate directory
#             if not os.path.exists(search_dir):
#                 raise FileNotFoundError(f"Directory does not exist: {search_dir}")

#             if not os.access(search_dir, os.R_OK):
#                 raise PermissionError(f"No read permission for directory: {search_dir}")

#             # Perform search
#             original_cwd = os.getcwd()
#             os.chdir(search_dir)

#             matches = python_glob.glob(pattern, recursive=True)
#             files = [os.path.abspath(match) for match in matches if os.path.isfile(match)]
#             files.sort()

#             truncated = len(files) > limit
#             if truncated:
#                 files = files[:limit]

#             duration_ms = int((time.time() - start_time) * 1000)

#             return {
#                 "success": True,
#                 "num_files": len(files),
#                 "filenames": files,
#                 "duration_ms": duration_ms,
#                 "truncated": truncated,
#                 "search_directory": search_dir,
#                 "pattern": pattern
#             }

#         except Exception as e:
#             self.logger.error(f"Glob search failed: {e}")
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "num_files": 0,
#                 "filenames": [],
#                 "duration_ms": int((time.time() - start_time) * 1000),
#                 "truncated": False
#             }

#         finally:
#             try:
#                 os.chdir(original_cwd)
#             except Exception as e:
#                 self.logger.warning(f"Failed to restore working directory: {e}")


if __name__ == "__main__":
    # Test the tool
    print("Testing glob_tool...")

    # Test with Python files
    result1 = glob_tool.invoke(
        {"pattern": "*.py", "path": "//users/chenlu/", "limit": 5}
    )
    print("Python files:")
    print(result1)
    print()

    # Test with recursive search
    result2 = glob_tool.invoke(
        {
            "pattern": "**/*.py",
            "path": "//users/chenlu/",
            "limit": 10,
        }
    )
    print("Recursive Python files:")
    print(result2)
    print()

    # # Test class-based implementation
    # searcher = GlobSearchTool(default_limit=5)
    # result3 = searcher.search("*.py")
    # print("Class-based search:")
    # print(json.dumps(result3, indent=2, ensure_ascii=False))