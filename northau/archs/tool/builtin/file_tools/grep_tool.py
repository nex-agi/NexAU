import json
import logging
import os
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum number of results to return to prevent overwhelming output
MAX_RESULTS = 100


def _check_ripgrep_available() -> bool:
    """Check if ripgrep (rg) is available in the system."""
    try:
        subprocess.run(['rg', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _run_ripgrep(
    pattern: str,
    search_path: str,
    glob_pattern: Optional[str] = None,
    output_mode: str = 'files_with_matches',
    context_before: Optional[int] = None,
    context_after: Optional[int] = None,
    context_around: Optional[int] = None,
    show_line_numbers: bool = False,
    case_insensitive: bool = True,
    file_type: Optional[str] = None,
    head_limit: Optional[int] = None,
    multiline: bool = False,
) -> tuple[list[str], int]:
    """
    Run ripgrep command and return results based on output mode.

    Args:
        pattern: Regular expression pattern to search for
        search_path: Directory to search in
        glob_pattern: File pattern to include (e.g., "*.js", "*.{ts,tsx}")
        output_mode: Output mode - "content", "files_with_matches", or "count"
        context_before: Lines of context before match (-B)
        context_after: Lines of context after match (-A)
        context_around: Lines of context around match (-C)
        show_line_numbers: Show line numbers in output (-n)
        case_insensitive: Case insensitive search (-i)
        file_type: File type to search (--type)
        head_limit: Limit output lines
        multiline: Enable multiline mode (-U --multiline-dotall)

    Returns:
        Tuple of (results, duration_ms)
    """
    start_time = time.time()

    # Build ripgrep command
    cmd = ['rg']

    # Set output mode
    if output_mode == 'files_with_matches':
        cmd.append('-l')  # list files only
    elif output_mode == 'count':
        cmd.append('-c')  # count matches
    # For "content" mode, no special flag needed

    # Case sensitivity
    if case_insensitive:
        cmd.append('-i')

    # Context options (only for content mode)
    if output_mode == 'content':
        if context_around is not None:
            cmd.extend(['-C', str(context_around)])
        else:
            if context_before is not None:
                cmd.extend(['-B', str(context_before)])
            if context_after is not None:
                cmd.extend(['-A', str(context_after)])

        # Line numbers
        if show_line_numbers:
            cmd.append('-n')

    # Multiline mode
    if multiline:
        cmd.extend(['-U', '--multiline-dotall'])

    # File type filter
    if file_type:
        cmd.extend(['--type', file_type])

    # Glob pattern filter
    if glob_pattern:
        cmd.extend(['--glob', glob_pattern])

    cmd.append(pattern)
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
            output_lines = [
                line.strip()
                for line in result.stdout.strip().split('\n')
                if line.strip()
            ]

            # Apply head_limit if specified
            if head_limit and len(output_lines) > head_limit:
                output_lines = output_lines[:head_limit]

        elif result.returncode == 1:
            # No matches found (this is normal)
            output_lines = []
        else:
            # Error occurred
            logger.error(
                f"Ripgrep error (code {result.returncode}): {result.stderr}",
            )
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                result.stderr,
            )

        duration_ms = int((time.time() - start_time) * 1000)
        return output_lines, duration_ms

    except Exception as e:
        logger.error(f"Error running ripgrep: {e}")
        raise


def _sort_files_by_modification_time(
    filenames: list[str],
    search_path: str,
) -> list[str]:
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
        except OSError as e:
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
    glob: Optional[str] = None,
    output_mode: str = 'files_with_matches',
    **kwargs,
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
                    'error': (
                        'ripgrep (rg) is not installed or not available in PATH. Please install ripgrep to use this tool.'
                    ),
                    'num_files': 0,
                    'filenames': [],
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'truncated': False,
                },
                indent=2,
            )

        # Validate search path (can be file or directory)
        if not os.path.exists(search_dir):
            return json.dumps(
                {
                    'error': f"Path does not exist: {search_dir}",
                    'num_files': 0,
                    'filenames': [],
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'truncated': False,
                },
                indent=2,
            )

        if not os.access(search_dir, os.R_OK):
            return json.dumps(
                {
                    'error': f"No read permission for path: {search_dir}",
                    'num_files': 0,
                    'filenames': [],
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'truncated': False,
                },
                indent=2,
            )

        # Handle file vs directory
        if os.path.isfile(search_dir):
            # If it's a file, search in its parent directory but only in that file
            file_dir = os.path.dirname(search_dir)
            file_name = os.path.basename(search_dir)
            # Use the file as a glob pattern to restrict search to just this file
            if glob:
                # If glob is already specified, we can't search a single file
                return json.dumps(
                    {
                        'error': f"Cannot use glob pattern when searching a specific file: {search_dir}",
                        'num_files': 0,
                        'filenames': [],
                        'duration_ms': int((time.time() - start_time) * 1000),
                        'truncated': False,
                    },
                    indent=2,
                )
            glob = file_name
            search_dir = file_dir

        # Extract additional parameters (support both old and new parameter names)
        context_before = kwargs.get('context_before') or kwargs.get('-B')
        context_after = kwargs.get('context_after') or kwargs.get('-A')
        context_around = kwargs.get('context_around') or kwargs.get('-C')
        show_line_numbers = kwargs.get(
            'show_line_numbers',
            kwargs.get('-n', False),
        )
        case_insensitive = kwargs.get(
            'case_insensitive',
            kwargs.get(
                '-i',
                True,
            ),
        )  # Default to case insensitive

        # Ensure context parameters are integers if provided
        if context_before is not None:
            context_before = int(context_before)
        if context_after is not None:
            context_after = int(context_after)
        if context_around is not None:
            context_around = int(context_around)
        file_type = kwargs.get('type')
        head_limit = kwargs.get('head_limit')
        if head_limit is not None:
            head_limit = int(head_limit)
        multiline = kwargs.get('multiline', False)

        # Run ripgrep search
        try:
            results, search_duration_ms = _run_ripgrep(
                pattern=pattern,
                search_path=search_dir,
                glob_pattern=glob,
                output_mode=output_mode,
                context_before=context_before,
                context_after=context_after,
                context_around=context_around,
                show_line_numbers=show_line_numbers,
                case_insensitive=case_insensitive,
                file_type=file_type,
                head_limit=head_limit,
                multiline=multiline,
            )
        except subprocess.CalledProcessError as e:
            return json.dumps(
                {
                    'error': f"Ripgrep search failed: {str(e)}",
                    'num_files': 0,
                    'filenames': [],
                    'duration_ms': int((time.time() - start_time) * 1000),
                    'truncated': False,
                },
                indent=2,
            )

        # Handle different output modes
        if output_mode == 'content':
            # For content mode, return the matching lines directly
            final_results = results
        elif output_mode == 'count':
            # For count mode, return count results
            final_results = results
        else:  # files_with_matches
            # Sort files by modification time if we have results
            if results:
                try:
                    final_results = _sort_files_by_modification_time(
                        results,
                        search_dir,
                    )
                except Exception as e:
                    logger.warning(f"Failed to sort by modification time: {e}")
                    # Fallback to alphabetical sort with absolute paths
                    final_results = [
                        os.path.abspath(os.path.join(search_dir, f))
                        for f in sorted(results)
                    ]
            else:
                final_results = []

        # Apply result limit and check for truncation (only for files_with_matches mode)
        if output_mode == 'files_with_matches' and not head_limit:
            truncated = len(final_results) > MAX_RESULTS
            if truncated:
                final_results = final_results[:MAX_RESULTS]
        else:
            truncated = False

        # Calculate total duration
        total_duration_ms = int((time.time() - start_time) * 1000)

        # Prepare result based on output mode
        if output_mode == 'content':
            result = {
                'content': final_results,
                'num_lines': len(final_results),
                'duration_ms': total_duration_ms,
                'truncated': truncated,
                'search_directory': search_dir,
                'pattern': pattern,
                'output_mode': output_mode,
            }
        elif output_mode == 'count':
            result = {
                'counts': final_results,
                'num_files': len(final_results),
                'duration_ms': total_duration_ms,
                'truncated': truncated,
                'search_directory': search_dir,
                'pattern': pattern,
                'output_mode': output_mode,
            }
        else:  # files_with_matches
            result = {
                'num_files': len(final_results),
                'filenames': final_results,
                'duration_ms': total_duration_ms,
                'truncated': truncated,
                'search_directory': search_dir,
                'pattern': pattern,
                'output_mode': output_mode,
            }

        if glob:
            result['glob_pattern'] = glob

        # Add descriptive message
        result_count = len(final_results)
        if result_count == 0:
            if output_mode == 'content':
                result['message'] = 'No matching content found.'
            elif output_mode == 'count':
                result['message'] = 'No matches found to count.'
            else:
                result['message'] = 'No files found matching the pattern.'
        elif truncated:
            result['message'] = (
                f"Found {result_count} results (truncated from more). Consider using a more specific pattern or filter."
            )
        else:
            if output_mode == 'content':
                result['message'] = (
                    f"Found {result_count} matching line{'s' if result_count != 1 else ''}."
                )
            elif output_mode == 'count':
                result['message'] = (
                    f"Found match counts for {result_count} file{'s' if result_count != 1 else ''}."
                )
            else:
                result['message'] = (
                    f"Found {result_count} file{'s' if result_count != 1 else ''} matching the pattern."
                )

        logger.info(
            f"Grep search completed: found {result_count} results in {total_duration_ms}ms (mode: {output_mode})",
        )

        # Apply length limit to JSON output
        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        if len(result_json) > 10000:
            # For different output modes, truncate appropriately
            if output_mode == 'files_with_matches':
                # Truncate filenames list
                items_to_keep = len(final_results)
                while items_to_keep > 0:
                    truncated_result = result.copy()
                    truncated_result['filenames'] = final_results[:items_to_keep]
                    truncated_result['num_files'] = items_to_keep
                    truncated_result['truncated_output'] = True
                    truncated_result['remaining_files'] = (
                        len(
                            final_results,
                        )
                        - items_to_keep
                    )
                    truncated_result['message'] = (
                        f"Found {items_to_keep} files (truncated: {len(final_results) - items_to_keep} more files not shown)"
                    )

                    test_json = json.dumps(
                        truncated_result,
                        indent=2,
                        ensure_ascii=False,
                    )
                    if len(test_json) <= 10000:
                        return test_json
                    items_to_keep -= 1
            elif output_mode == 'content':
                # Truncate content lines
                items_to_keep = len(final_results)
                while items_to_keep > 0:
                    truncated_result = result.copy()
                    truncated_result['content'] = final_results[:items_to_keep]
                    truncated_result['num_lines'] = items_to_keep
                    truncated_result['truncated_output'] = True
                    truncated_result['remaining_lines'] = (
                        len(
                            final_results,
                        )
                        - items_to_keep
                    )
                    truncated_result['message'] = (
                        f"Found {items_to_keep} lines (truncated: {len(final_results) - items_to_keep} more lines not shown)"
                    )

                    test_json = json.dumps(
                        truncated_result,
                        indent=2,
                        ensure_ascii=False,
                    )
                    if len(test_json) <= 1000:
                        return test_json
                    items_to_keep -= 1
            elif output_mode == 'count':
                # Truncate count results
                items_to_keep = len(final_results)
                while items_to_keep > 0:
                    truncated_result = result.copy()
                    truncated_result['counts'] = final_results[:items_to_keep]
                    truncated_result['num_files'] = items_to_keep
                    truncated_result['truncated_output'] = True
                    truncated_result['remaining_files'] = (
                        len(
                            final_results,
                        )
                        - items_to_keep
                    )
                    truncated_result['message'] = (
                        f"Found counts for {items_to_keep} files (truncated: {len(final_results) - items_to_keep} more files not shown)"
                    )

                    test_json = json.dumps(
                        truncated_result,
                        indent=2,
                        ensure_ascii=False,
                    )
                    if len(test_json) <= 1000:
                        return test_json
                    items_to_keep -= 1

            # If even minimal result is too long, return basic summary
            minimal_result = {
                'truncated_output': True,
                'total_matches': len(final_results),
                'message': f"Output too long: found {len(final_results)} matches (details truncated)",
                'search_directory': search_dir,
                'pattern': pattern,
                'output_mode': output_mode,
                'duration_ms': total_duration_ms,
            }
            return json.dumps(minimal_result, indent=2, ensure_ascii=False)

        return result_json

    except Exception as e:
        logger.error(f"Unexpected error during grep search: {e}")
        return json.dumps(
            {
                'error': f"Unexpected error during search: {str(e)}",
                'num_files': 0,
                'filenames': [],
                'duration_ms': int((time.time() - start_time) * 1000),
                'truncated': False,
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
    ) -> dict:
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
                    'ripgrep (rg) is not installed or not available in PATH',
                )

            # Validate directory
            if not os.path.exists(search_dir):
                raise FileNotFoundError(
                    f"Directory does not exist: {search_dir}",
                )

            if not os.access(search_dir, os.R_OK):
                raise PermissionError(
                    f"No read permission for directory: {search_dir}",
                )

            # Perform search
            results, search_duration = _run_ripgrep(
                pattern=pattern,
                search_path=search_dir,
                glob_pattern=include_pattern,
                output_mode='files_with_matches',
                case_insensitive=not self.case_sensitive,
            )

            # Sort results
            if results:
                sorted_filenames = _sort_files_by_modification_time(
                    results,
                    search_dir,
                )
            else:
                sorted_filenames = []

            # Apply limit
            truncated = len(sorted_filenames) > limit
            if truncated:
                sorted_filenames = sorted_filenames[:limit]

            duration_ms = int((time.time() - start_time) * 1000)

            return {
                'success': True,
                'num_files': len(sorted_filenames),
                'filenames': sorted_filenames,
                'duration_ms': duration_ms,
                'search_duration_ms': search_duration,
                'truncated': truncated,
                'search_directory': search_dir,
                'pattern': pattern,
                'include_pattern': include_pattern,
                'case_sensitive': self.case_sensitive,
            }

        except Exception as e:
            self.logger.error(f"Grep search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'num_files': 0,
                'filenames': [],
                'duration_ms': int((time.time() - start_time) * 1000),
                'truncated': False,
            }


def main():
    """
    Test function to demonstrate and validate the grep_tool functionality.
    """
    import json

    print('🔍 GrepTool 测试开始...')
    print('=' * 50)

    # Test 1: Check ripgrep availability
    print('\n📋 测试 1: 检查 ripgrep 可用性')
    if _check_ripgrep_available():
        print('✅ ripgrep (rg) 已安装并可用')
    else:
        print('❌ ripgrep (rg) 未安装或不可用')
        print('请先安装 ripgrep:')
        print('  conda install -c conda-forge ripgrep')
        print('  或者 sudo apt install ripgrep')
        return

    # Test 2: Basic search test
    print('\n📋 测试 2: 基本搜索测试')
    try:
        result = grep_tool(pattern='import', path='.')
        result_dict = json.loads(result)

        if 'error' in result_dict:
            print(f"❌ 搜索失败: {result_dict['error']}")
        else:
            print(f"✅ 找到 {result_dict['num_files']} 个包含 'import' 的文件")
            print(f"⏱️  搜索耗时: {result_dict['duration_ms']}ms")
            if result_dict['num_files'] > 0:
                print('📁 前5个匹配文件:')
                for i, filename in enumerate(result_dict['filenames'][:5]):
                    print(f"   {i + 1}. {filename}")
                if len(result_dict['filenames']) > 5:
                    print(f"   ... 还有 {len(result_dict['filenames']) - 5} 个文件")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 3: Search with file type filter
    print('\n📋 测试 3: 带文件类型过滤的搜索')
    try:
        result = grep_tool(pattern='def ', path='.', glob='*.py')
        result_dict = json.loads(result)

        if 'error' in result_dict:
            print(f"❌ 搜索失败: {result_dict['error']}")
        else:
            print(
                f"✅ 在 Python 文件中找到 {result_dict['num_files']} 个包含函数定义的文件",
            )
            print(f"⏱️  搜索耗时: {result_dict['duration_ms']}ms")
            if result_dict['num_files'] > 0:
                print('📁 匹配的 Python 文件:')
                for i, filename in enumerate(result_dict['filenames'][:3]):
                    print(f"   {i + 1}. {filename}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 4: Search for non-existent pattern
    print('\n📋 测试 4: 搜索不存在的模式')
    try:
        result = grep_tool(
            pattern='this_pattern_should_never_exist_12345_xyz',
            path='.',
        )
        result_dict = json.loads(result)

        if result_dict['num_files'] == 0:
            print('✅ 正确处理了空搜索结果')
            print(f"💬 消息: {result_dict['message']}")
        else:
            print(f"⚠️  意外找到了 {result_dict['num_files']} 个文件")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 5: Test class-based implementation
    print('\n📋 测试 5: 类基础实现测试')
    try:
        grep_search = GrepSearchTool(max_results=5)
        result = grep_search.search(
            pattern='class',
            path='.',
            include_pattern='*.py',
        )

        if result['success']:
            print(f"✅ 类实现工作正常，找到 {result['num_files']} 个文件")
            print(f"⏱️  搜索耗时: {result['duration_ms']}ms")
        else:
            print(f"❌ 类实现失败: {result['error']}")

    except Exception as e:
        print(f"❌ 类实现测试失败: {e}")

    # Test 6: Test invalid directory
    print('\n📋 测试 6: 无效目录测试')
    try:
        result = grep_tool(
            pattern='test',
            path='/this/directory/does/not/exist',
        )
        result_dict = json.loads(result)

        if 'error' in result_dict and 'does not exist' in result_dict['error']:
            print('✅ 正确处理了无效目录')
        else:
            print('⚠️  无效目录处理可能有问题')

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print('\n' + '=' * 50)
    print('🎉 GrepTool 测试完成!')

    # Performance tip
    print('\n💡 使用提示:')
    print("  • 使用正则表达式进行精确搜索: 'function\\s+\\w+'")
    print("  • 使用文件过滤器提高效率: include='*.{js,ts,tsx}'")
    print("  • 搜索错误模式: 'error|Error|ERROR'")
    print("  • 搜索 TODO 注释: 'TODO|FIXME|XXX'")


if __name__ == '__main__':
    main()
