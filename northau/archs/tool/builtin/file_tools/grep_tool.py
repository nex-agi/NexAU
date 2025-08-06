import json
import logging
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum number of results to return to prevent overwhelming output
MAX_RESULTS = 100


def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available in the system."""
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_ripgrep(
    pattern: str, search_path: str, include_pattern: Optional[str] = None
) -> Tuple[List[str], int]:
    """
    Run ripgrep command and return matching filenames and duration.

    Args:
        pattern: Regular expression pattern to search for
        search_path: Directory to search in
        include_pattern: File pattern to include (e.g., "*.js", "*.{ts,tsx}")

    Returns:
        Tuple of (matching_filenames, duration_ms)
    """
    start_time = time.time()

    # Build ripgrep command
    cmd = ["rg", "-l", "-i", pattern]  # -l: list files, -i: ignore case

    if include_pattern:
        cmd.extend(["--glob", include_pattern])

    cmd.append(search_path)

    try:
        # Run ripgrep command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit
            cwd=search_path,
        )

        # Parse output
        if result.returncode == 0:
            # Found matches
            filenames = [
                line.strip()
                for line in result.stdout.strip().split("\n")
                if line.strip()
            ]
        elif result.returncode == 1:
            # No matches found (this is normal)
            filenames = []
        else:
            # Error occurred
            logger.error(f"Ripgrep error (code {result.returncode}): {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)

        duration_ms = int((time.time() - start_time) * 1000)
        return filenames, duration_ms

    except Exception as e:
        logger.error(f"Error running ripgrep: {e}")
        raise


def _sort_files_by_modification_time(
    filenames: List[str], search_path: str
) -> List[str]:
    """
    Sort files by modification time (newest first), with filename as tiebreaker.

    Args:
        filenames: List of relative filenames
        search_path: Base directory path

    Returns:
        Sorted list of absolute filenames
    """
    files_with_mtime = []

    for filename in filenames:
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(os.path.join(search_path, filename))
            # Get modification time
            mtime = os.path.getmtime(abs_path)
            files_with_mtime.append((abs_path, mtime))
        except (OSError, IOError) as e:
            logger.warning(f"Could not get mtime for {filename}: {e}")
            # Still include the file but with mtime 0
            abs_path = os.path.abspath(os.path.join(search_path, filename))
            files_with_mtime.append((abs_path, 0))

    # Sort by modification time (newest first), then by filename
    files_with_mtime.sort(key=lambda x: (-x[1], x[0]))

    return [filepath for filepath, _ in files_with_mtime]


def grep_tool(
    pattern: str,
    path: Optional[str] = None,
    include: Optional[str] = None,
) -> str:
    """
    Fast content search tool that works with any codebase size using ripgrep.

    Features:
    - Searches file contents using regular expressions
    - Supports full regex syntax (e.g., "log.*Error", "function\\s+\\w+", etc.)
    - Filter files by pattern with the include parameter (e.g., "*.js", "*.{ts,tsx}")
    - Returns matching file paths sorted by modification time (newest first)
    - Uses ripgrep for fast searching across large codebases

    Examples:
    - Search for error patterns: pattern="error|Error|ERROR"
    - Search for function definitions: pattern="function\\s+\\w+"
    - Search in specific file types: pattern="import", include="*.{js,ts,tsx}"
    - Search for TODO comments: pattern="TODO|FIXME|XXX"

    Note: This tool requires ripgrep (rg) to be installed on the system.
    """
    start_time = time.time()

    # Determine search directory
    search_dir = path if path else os.getcwd()

    try:
        # Check if ripgrep is available
        if not _check_ripgrep_available():
            return json.dumps(
                {
                    "error": (
                        "ripgrep (rg) is not installed or not available in PATH. Please install ripgrep to use this tool."
                    ),
                    "num_files": 0,
                    "filenames": [],
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "truncated": False,
                },
                indent=2,
            )

        # Validate search directory
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

        # Run ripgrep search
        try:
            filenames, search_duration_ms = _run_ripgrep(pattern, search_dir, include)
        except subprocess.CalledProcessError as e:
            return json.dumps(
                {
                    "error": f"Ripgrep search failed: {str(e)}",
                    "num_files": 0,
                    "filenames": [],
                    "duration_ms": int((time.time() - start_time) * 1000),
                    "truncated": False,
                },
                indent=2,
            )

        # Sort files by modification time if we have results
        if filenames:
            try:
                sorted_filenames = _sort_files_by_modification_time(
                    filenames, search_dir
                )
            except Exception as e:
                logger.warning(f"Failed to sort by modification time: {e}")
                # Fallback to alphabetical sort with absolute paths
                sorted_filenames = [
                    os.path.abspath(os.path.join(search_dir, f))
                    for f in sorted(filenames)
                ]
        else:
            sorted_filenames = []

        # Apply result limit and check for truncation
        truncated = len(sorted_filenames) > MAX_RESULTS
        if truncated:
            sorted_filenames = sorted_filenames[:MAX_RESULTS]

        # Calculate total duration
        total_duration_ms = int((time.time() - start_time) * 1000)

        # Prepare result
        result = {
            "num_files": len(sorted_filenames),
            "filenames": sorted_filenames,
            "duration_ms": total_duration_ms,
            "truncated": truncated,
            "search_directory": search_dir,
            "pattern": pattern,
        }

        if include:
            result["include_pattern"] = include

        # Add descriptive message
        if len(sorted_filenames) == 0:
            result["message"] = "No files found matching the pattern."
        elif truncated:
            result["message"] = (
                f"Found {len(sorted_filenames)} files (truncated from more results). Consider using a more specific pattern or include filter."
            )
        else:
            result["message"] = (
                f"Found {len(sorted_filenames)} file{'s' if len(sorted_filenames) != 1 else ''} matching the pattern."
            )

        logger.info(
            f"Grep search completed: found {len(sorted_filenames)} files in {total_duration_ms}ms"
        )

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Unexpected error during grep search: {e}")
        return json.dumps(
            {
                "error": f"Unexpected error during search: {str(e)}",
                "num_files": 0,
                "filenames": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "truncated": False,
            },
            indent=2,
        )


# Alternative class-based implementation for advanced usage
class GrepSearchTool:
    """
    A class-based implementation of the grep search tool for more advanced usage.
    Provides additional configuration options and better error handling.
    """

    def __init__(self, max_results: int = MAX_RESULTS, case_sensitive: bool = False):
        self.max_results = max_results
        self.case_sensitive = case_sensitive
        self.logger = logging.getLogger(self.__class__.__name__)

    def search(
        self,
        pattern: str,
        path: Optional[str] = None,
        include_pattern: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> Dict:
        """
        Perform a grep search and return structured results.

        Args:
            pattern: Regular expression pattern to search for
            path: Directory to search in (defaults to current working directory)
            include_pattern: File pattern to include in search
            max_results: Maximum number of files to return (overrides instance default)

        Returns:
            Dictionary containing search results
        """
        start_time = time.time()
        search_dir = path if path else os.getcwd()
        limit = max_results if max_results is not None else self.max_results

        try:
            # Check ripgrep availability
            if not _check_ripgrep_available():
                raise RuntimeError(
                    "ripgrep (rg) is not installed or not available in PATH"
                )

            # Validate directory
            if not os.path.exists(search_dir):
                raise FileNotFoundError(f"Directory does not exist: {search_dir}")

            if not os.access(search_dir, os.R_OK):
                raise PermissionError(f"No read permission for directory: {search_dir}")

            # Perform search
            filenames, search_duration = _run_ripgrep(
                pattern, search_dir, include_pattern
            )

            # Sort results
            if filenames:
                sorted_filenames = _sort_files_by_modification_time(
                    filenames, search_dir
                )
            else:
                sorted_filenames = []

            # Apply limit
            truncated = len(sorted_filenames) > limit
            if truncated:
                sorted_filenames = sorted_filenames[:limit]

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "num_files": len(sorted_filenames),
                "filenames": sorted_filenames,
                "duration_ms": duration_ms,
                "search_duration_ms": search_duration,
                "truncated": truncated,
                "search_directory": search_dir,
                "pattern": pattern,
                "include_pattern": include_pattern,
                "case_sensitive": self.case_sensitive,
            }

        except Exception as e:
            self.logger.error(f"Grep search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "num_files": 0,
                "filenames": [],
                "duration_ms": int((time.time() - start_time) * 1000),
                "truncated": False,
            }


def main():
    """
    Test function to demonstrate and validate the grep_tool functionality.
    """
    import json

    print("🔍 GrepTool 测试开始...")
    print("=" * 50)

    # Test 1: Check ripgrep availability
    print("\n📋 测试 1: 检查 ripgrep 可用性")
    if _check_ripgrep_available():
        print("✅ ripgrep (rg) 已安装并可用")
    else:
        print("❌ ripgrep (rg) 未安装或不可用")
        print("请先安装 ripgrep:")
        print("  conda install -c conda-forge ripgrep")
        print("  或者 sudo apt install ripgrep")
        return

    # Test 2: Basic search test
    print("\n📋 测试 2: 基本搜索测试")
    try:
        result = grep_tool.invoke({"pattern": "import", "path": "."})
        result_dict = json.loads(result)

        if "error" in result_dict:
            print(f"❌ 搜索失败: {result_dict['error']}")
        else:
            print(f"✅ 找到 {result_dict['num_files']} 个包含 'import' 的文件")
            print(f"⏱️  搜索耗时: {result_dict['duration_ms']}ms")
            if result_dict["num_files"] > 0:
                print("📁 前5个匹配文件:")
                for i, filename in enumerate(result_dict["filenames"][:5]):
                    print(f"   {i+1}. {filename}")
                if len(result_dict["filenames"]) > 5:
                    print(f"   ... 还有 {len(result_dict['filenames']) - 5} 个文件")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 3: Search with file type filter
    print("\n📋 测试 3: 带文件类型过滤的搜索")
    try:
        result = grep_tool.invoke({"pattern": "def ", "path": ".", "include": "*.py"})
        result_dict = json.loads(result)

        if "error" in result_dict:
            print(f"❌ 搜索失败: {result_dict['error']}")
        else:
            print(
                f"✅ 在 Python 文件中找到 {result_dict['num_files']} 个包含函数定义的文件"
            )
            print(f"⏱️  搜索耗时: {result_dict['duration_ms']}ms")
            if result_dict["num_files"] > 0:
                print("📁 匹配的 Python 文件:")
                for i, filename in enumerate(result_dict["filenames"][:3]):
                    print(f"   {i+1}. {filename}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 4: Search for non-existent pattern
    print("\n📋 测试 4: 搜索不存在的模式")
    try:
        result = grep_tool.invoke(
            {"pattern": "this_pattern_should_never_exist_12345_xyz", "path": "."}
        )
        result_dict = json.loads(result)

        if result_dict["num_files"] == 0:
            print("✅ 正确处理了空搜索结果")
            print(f"💬 消息: {result_dict['message']}")
        else:
            print(f"⚠️  意外找到了 {result_dict['num_files']} 个文件")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 5: Test class-based implementation
    print("\n📋 测试 5: 类基础实现测试")
    try:
        grep_search = GrepSearchTool(max_results=5)
        result = grep_search.search(pattern="class", path=".", include_pattern="*.py")

        if result["success"]:
            print(f"✅ 类实现工作正常，找到 {result['num_files']} 个文件")
            print(f"⏱️  搜索耗时: {result['duration_ms']}ms")
        else:
            print(f"❌ 类实现失败: {result['error']}")

    except Exception as e:
        print(f"❌ 类实现测试失败: {e}")

    # Test 6: Test invalid directory
    print("\n📋 测试 6: 无效目录测试")
    try:
        result = grep_tool.invoke(
            {"pattern": "test", "path": "/this/directory/does/not/exist"}
        )
        result_dict = json.loads(result)

        if "error" in result_dict and "does not exist" in result_dict["error"]:
            print("✅ 正确处理了无效目录")
        else:
            print("⚠️  无效目录处理可能有问题")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print("\n" + "=" * 50)
    print("🎉 GrepTool 测试完成!")

    # Performance tip
    print("\n💡 使用提示:")
    print("  • 使用正则表达式进行精确搜索: 'function\\s+\\w+'")
    print("  • 使用文件过滤器提高效率: include='*.{js,ts,tsx}'")
    print("  • 搜索错误模式: 'error|Error|ERROR'")
    print("  • 搜索 TODO 注释: 'TODO|FIXME|XXX'")


if __name__ == "__main__":
    main()