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

import glob as python_glob
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
        if python_glob.fnmatch.fnmatch(path, pattern) or python_glob.fnmatch.fnmatch(
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

        info = {
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
        return json.dumps(
            {
                "status": "error",
                "error": f"Path must be absolute, got relative path: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )

    # Check if path exists
    if not os.path.exists(path):
        return json.dumps(
            {
                "status": "error",
                "error": f"Path does not exist: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )

    # Check if it's a directory
    if not os.path.isdir(path):
        return json.dumps(
            {
                "status": "error",
                "error": f"Path is not a directory: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )

    # Check permissions
    if not os.access(path, os.R_OK):
        return json.dumps(
            {
                "status": "error",
                "error": f"No read permission for directory: {path}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )

    try:
        # Get directory contents
        items = []
        files = []
        directories = []
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
        result = {
            "status": "success",
            "path": path,
            "total_items": len(items),
            "directories": len(directories),
            "files": len(files),
            "ignored_items": ignored_count,
            "error_items": error_count,
            "duration_ms": duration_ms,
            "items": items,
        }

        if ignore:
            result["ignore_patterns"] = ignore

        # Add summary message
        if len(items) == 0:
            if ignored_count > 0:
                result["message"] = f"Directory is empty (ignored {ignored_count} items)"
            else:
                result["message"] = "Directory is empty"
        else:
            result["message"] = f"Found {len(directories)} directories and {len(files)} files"
            if ignored_count > 0:
                result["message"] += f" (ignored {ignored_count} items)"
            if error_count > 0:
                result["message"] += f" (errors accessing {error_count} items)"

        logger.info(
            f"LS completed: listed {len(items)} items in {duration_ms}ms",
        )

        # Apply length limit to JSON output
        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        if len(result_json) > 10000:
            # Calculate how many items to keep to stay under limit
            items_to_keep = len(items)
            while items_to_keep > 0:
                truncated_result = result.copy()
                truncated_result["items"] = items[:items_to_keep]
                truncated_result["total_items"] = items_to_keep
                truncated_result["truncated_output"] = True
                truncated_result["remaining_items"] = (
                    len(
                        items,
                    )
                    - items_to_keep
                )
                truncated_result["message"] += f" (Output truncated: showing {items_to_keep} of {len(items)} items)"

                test_json = json.dumps(
                    truncated_result,
                    indent=2,
                    ensure_ascii=False,
                )
                if len(test_json) <= 10000:
                    return json.dumps(truncated_result, indent=2, ensure_ascii=False)
                items_to_keep -= 1

            # If even 0 items is too long, return minimal result
            minimal_result = {
                "status": "success",
                "path": path,
                "total_items": len(items),
                "directories": len(directories),
                "files": len(files),
                "truncated_output": True,
                "remaining_items": len(items),
                "message": f"Output too long: found {len(directories)} directories and {len(files)} files (details truncated)",
                "duration_ms": duration_ms,
            }
            return json.dumps(minimal_result, indent=2, ensure_ascii=False)

        return json.dumps(result, indent=2, ensure_ascii=False)

    except PermissionError as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Permission denied accessing directory: {str(e)}",
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )

    except Exception as e:
        logger.error(f"Unexpected error during ls: {e}")
        return json.dumps(
            {
                "status": "error",
                "error": f"Unexpected error listing directory: {str(e)}",
                "error_type": type(e).__name__,
                "path": path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )


# Alternative class-based implementation for more advanced usage
class LSTool:
    """
    A class-based implementation of the ls tool for more advanced usage.
    Provides additional configuration options and better error handling.
    """

    def __init__(self, show_hidden: bool = True, detailed_info: bool = True):
        self.show_hidden = show_hidden
        self.detailed_info = detailed_info
        self.logger = logging.getLogger(self.__class__.__name__)

    def list_directory(
        self,
        path: str,
        ignore_patterns: list[str] | None = None,
        recursive: bool = False,
        max_depth: int = 1,
    ) -> dict[str, Any]:
        """
        List directory contents with advanced options.

        Args:
            path: Absolute path to the directory to list
            ignore_patterns: List of glob patterns to ignore
            recursive: Whether to list subdirectories recursively
            max_depth: Maximum recursion depth (only used if recursive=True)

        Returns:
            Dict containing listing results
        """
        if recursive:
            return self._list_recursive(path, ignore_patterns, max_depth)
        else:
            return ls_tool(path, ignore_patterns)

    def _list_recursive(
        self,
        path: str,
        ignore_patterns: list[str] | None = None,
        max_depth: int = 1,
        current_depth: int = 0,
    ) -> dict[str, Any]:
        """
        Recursively list directory contents.

        Args:
            path: Path to list
            ignore_patterns: Patterns to ignore
            max_depth: Maximum depth to recurse
            current_depth: Current recursion depth

        Returns:
            Dict containing recursive listing results
        """
        if current_depth >= max_depth:
            return ls_tool(path, ignore_patterns)

        result = ls_tool(path, ignore_patterns)

        if result["status"] != "success":
            return result

        # Add recursive listings for subdirectories
        for item in result["items"]:
            if item.get("is_dir") and not item.get("error"):
                subdir_path = item["path"]
                subdir_result = self._list_recursive(
                    subdir_path,
                    ignore_patterns,
                    max_depth,
                    current_depth + 1,
                )
                item["contents"] = subdir_result

        result["recursive"] = True
        result["max_depth"] = max_depth

        return result

    def find_files_by_extension(
        self,
        path: str,
        extension: str,
        recursive: bool = True,
    ) -> dict[str, Any]:
        """
        Find files with a specific extension.

        Args:
            path: Directory path to search
            extension: File extension to search for (e.g., '.py', '.txt')
            recursive: Whether to search recursively

        Returns:
            Dict containing matching files
        """
        if not extension.startswith("."):
            extension = "." + extension

        if recursive:
            result = self._list_recursive(path, max_depth=10)
        else:
            result = ls_tool(path)

        if result["status"] != "success":
            return result

        matching_files = []
        self._collect_files_by_extension(result, extension, matching_files)

        return {
            "status": "success",
            "path": path,
            "extension": extension,
            "matching_files": matching_files,
            "count": len(matching_files),
            "recursive": recursive,
        }

    def _collect_files_by_extension(
        self,
        result: dict[str, Any],
        extension: str,
        matching_files: list[dict[str, Any]],
    ):
        """Recursively collect files with matching extension."""
        for item in result.get("items", []):
            if item.get("is_file") and item.get("extension") == extension:
                matching_files.append(item)

            if item.get("is_dir") and "contents" in item:
                self._collect_files_by_extension(
                    item["contents"],
                    extension,
                    matching_files,
                )


def main():
    """Test function to demonstrate and validate the ls_tool functionality."""
    print("📁 LSTool 测试开始...")
    print("=" * 50)

    # Test 1: Basic directory listing
    print("\n📋 测试 1: 基本目录列表")
    try:
        current_dir = os.getcwd()
        result = ls_tool(current_dir)

        if result["status"] == "success":
            print("✅ 目录列表成功")
            print(f"📁 路径: {result['path']}")
            print(f"📊 统计: {result['directories']} 个目录, {result['files']} 个文件")
            print(f"⏱️  执行时间: {result['duration_ms']}ms")

            # Show first few items
            items = result["items"][:5]
            print("📄 前5个项目:")
            for item in items:
                item_type = "📁" if item.get("is_dir") else "📄"
                print(f"   {item_type} {item.get('name', 'Unknown')}")

        else:
            print(f"❌ 列表失败: {result['error']}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 2: Directory listing with ignore patterns
    print("\n📋 测试 2: 忽略模式测试")
    try:
        result = ls_tool(current_dir, ignore=[".*", "*.pyc", "__pycache__"])

        if result["status"] == "success":
            print("✅ 带忽略模式的列表成功")
            print(f"🚫 忽略了 {result['ignored_items']} 个项目")
            print(f"📊 显示: {result['directories']} 个目录, {result['files']} 个文件")
        else:
            print(f"❌ 列表失败: {result['error']}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 3: Non-existent directory
    print("\n📋 测试 3: 不存在的目录")
    try:
        result = ls_tool("/nonexistent_directory_12345")

        if result["status"] == "error" and "not exist" in result["error"]:
            print("✅ 正确处理了不存在的目录")
            print(f"⚠️  错误信息: {result['error']}")
        else:
            print("⚠️  错误处理可能有问题")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 4: Relative path (should fail)
    print("\n📋 测试 4: 相对路径测试")
    try:
        result = ls_tool(".")

        if result["status"] == "error" and "absolute" in result["error"]:
            print("✅ 正确拒绝了相对路径")
            print(f"⚠️  错误信息: {result['error']}")
        else:
            print("⚠️  相对路径处理可能有问题")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 5: Class-based implementation
    print("\n📋 测试 5: 类实现测试")
    try:
        ls_tool_instance = LSTool()
        result = ls_tool_instance.list_directory(current_dir)

        if result["status"] == "success":
            print("✅ 类实现工作正常")
            print(f"📊 统计: {result['directories']} 个目录, {result['files']} 个文件")
        else:
            print("❌ 类实现失败")

    except Exception as e:
        print(f"❌ 类实现测试失败: {e}")

    # Test 6: Find files by extension
    print("\n📋 测试 6: 按扩展名查找文件")
    try:
        ls_tool_instance = LSTool()
        result = ls_tool_instance.find_files_by_extension(
            current_dir,
            "py",
            recursive=False,
        )

        if result["status"] == "success":
            print("✅ 扩展名搜索成功")
            print(f"🔍 找到 {result['count']} 个 Python 文件")

            # Show first few files
            files = result["matching_files"][:3]
            for file_info in files:
                print(f"   📄 {file_info.get('name', 'Unknown')}")
        else:
            print("❌ 扩展名搜索失败")

    except Exception as e:
        print(f"❌ 扩展名搜索测试失败: {e}")

    print("\n" + "=" * 50)
    print("🎉 LSTool 测试完成!")

    # Usage tips
    print("\n💡 使用提示:")
    print("  • 总是使用绝对路径，不支持相对路径")
    print("  • 使用忽略模式过滤不需要的文件")
    print("  • 目录按字母顺序排列，目录在文件之前")
    print("  • 检查权限确保可以访问目录")
    print("  • 使用类实现进行递归搜索和高级功能")


if __name__ == "__main__":
    main()
