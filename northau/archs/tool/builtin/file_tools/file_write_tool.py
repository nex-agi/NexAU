import difflib
import json
import logging
import os
import time
from typing import Optional

from .file_state import update_file_timestamp
from .file_state import validate_file_read_state

# Import file state management for read/write coordination

logger = logging.getLogger(__name__)

# 最大显示行数限制
MAX_LINES_TO_RENDER = 10
MAX_LINES_TO_RENDER_FOR_ASSISTANT = 16000


def _detect_file_encoding(file_path: str) -> str:
    """检测文件编码"""
    try:
        import chardet

        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10KB用于检测
            result = chardet.detect(raw_data)
            return result.get('encoding', 'utf-8') or 'utf-8'
    except (ImportError, Exception):
        # 如果chardet不可用或检测失败，默认使用utf-8
        return 'utf-8'


def _detect_line_endings(file_path: str) -> str:
    """检测文件行结束符"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read(1024)  # 读取前1KB
            if b'\r\n' in content:
                return '\r\n'
            elif b'\n' in content:
                return '\n'
            elif b'\r' in content:
                return '\r'
    except Exception:
        pass
    return '\n'  # 默认使用\n


def _has_write_permission(file_path: str) -> bool:
    """检查是否有写入权限"""
    try:
        # 检查目录是否存在且可写
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            # 检查是否可以创建目录
            parent = os.path.dirname(directory)
            while parent and not os.path.exists(parent):
                parent = os.path.dirname(parent)
            return os.access(parent, os.W_OK) if parent else False

        # 检查文件是否存在
        if os.path.exists(file_path):
            return os.access(file_path, os.W_OK)
        else:
            # 检查目录是否可写
            return os.access(directory, os.W_OK)
    except Exception:
        return False


def _generate_diff(old_content: str, new_content: str, file_path: str) -> str:
    """生成差异对比"""
    try:
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{os.path.basename(file_path)}",
            tofile=f"b/{os.path.basename(file_path)}",
            lineterm='',
        )

        return ''.join(diff)
    except Exception as e:
        logger.error(f"生成差异对比失败: {e}")
        return ''


def _write_file_content(
    file_path: str,
    content: str,
    encoding: str = 'utf-8',
    line_ending: str = '\n',
) -> None:
    """写入文件内容"""
    # 标准化行结束符
    if line_ending != '\n':
        content = content.replace('\n', line_ending)

    # 创建目录
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # 写入文件
    with open(file_path, 'w', encoding=encoding, newline='') as f:
        f.write(content)


def file_write_tool(
    file_path: str,
    content: str,
) -> str:
    """
    将内容写入本地文件系统中的文件。如果文件已存在则覆盖。

    使用此工具前请注意：
    1. 使用ReadFile工具先了解文件的内容和上下文
    2. 目录验证（仅在创建新文件时适用）：
       - 使用LS工具验证父目录是否存在且位置正确

    功能特性：
    - 自动检测和保持文件编码
    - 自动检测和保持行结束符格式
    - 提供文件修改时间戳验证以防止冲突
    - 生成详细的差异对比
    - 支持创建目录结构
    - 权限检查和安全验证

    返回：
    - 成功时返回操作结果的JSON格式字符串
    - 失败时返回错误信息的JSON格式字符串
    """
    start_time = time.time()

    try:
        # 验证文件路径
        if not os.path.isabs(file_path):
            return json.dumps(
                {
                    'error': '文件路径必须是绝对路径',
                    'file_path': file_path,
                    'success': False,
                    'duration_ms': int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # 检查写入权限
        if not _has_write_permission(file_path):
            return json.dumps(
                {
                    'error': f"没有写入权限: {file_path}",
                    'file_path': file_path,
                    'success': False,
                    'duration_ms': int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # 检查文件是否存在
        file_exists = os.path.exists(file_path)
        operation_type = 'update' if file_exists else 'create'

        # 读取原始内容和检测编码
        old_content = ''
        encoding = 'utf-8'
        line_ending = '\n'

        if file_exists:
            # 验证文件状态
            is_valid, error_msg = validate_file_read_state(file_path)
            if not is_valid:
                return json.dumps(
                    {
                        'error': error_msg,
                        'file_path': file_path,
                        'success': False,
                        'duration_ms': int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            # 检测文件编码和行结束符
            encoding = _detect_file_encoding(file_path)
            line_ending = _detect_line_endings(file_path)

            # 读取原始内容
            try:
                with open(file_path, encoding=encoding) as f:
                    old_content = f.read()
            except Exception as e:
                logger.error(f"读取原始文件内容失败: {e}")
                return json.dumps(
                    {
                        'error': f"读取原始文件失败: {str(e)}",
                        'file_path': file_path,
                        'success': False,
                        'duration_ms': int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

        # 写入文件
        try:
            _write_file_content(file_path, content, encoding, line_ending)
        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            return json.dumps(
                {
                    'error': f"写入文件失败: {str(e)}",
                    'file_path': file_path,
                    'success': False,
                    'duration_ms': int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # 更新时间戳缓存
        update_file_timestamp(file_path)

        # 生成差异对比（如果是更新操作）
        diff_content = ''
        if operation_type == 'update' and old_content != content:
            diff_content = _generate_diff(old_content, content, file_path)

        # 计算持续时间
        duration_ms = int((time.time() - start_time) * 1000)

        # 统计信息
        content_lines = content.split('\n')
        num_lines = len(content_lines)

        # 准备结果
        result = {
            'success': True,
            'operation_type': operation_type,
            'file_path': file_path,
            'num_lines': num_lines,
            'encoding': encoding,
            'line_ending': (
                'CRLF'
                if line_ending == '\r\n'
                else 'LF' if line_ending == '\n' else 'CR'
            ),
            'duration_ms': duration_ms,
        }

        # 添加差异信息（仅在更新时）
        if operation_type == 'update':
            result['has_changes'] = old_content != content
            if diff_content:
                result['diff'] = diff_content
                # 统计变更行数
                diff_lines = diff_content.split('\n')
                added_lines = len(
                    [line for line in diff_lines if line.startswith('+')],
                )
                removed_lines = len(
                    [line for line in diff_lines if line.startswith('-')],
                )
                result['changes'] = {
                    'lines_added': added_lines,
                    'lines_removed': removed_lines,
                }

        # 添加内容预览（限制行数）
        if num_lines <= MAX_LINES_TO_RENDER:
            result['content_preview'] = content
        else:
            preview_lines = content_lines[:MAX_LINES_TO_RENDER]
            result['content_preview'] = '\n'.join(preview_lines)
            result['content_truncated'] = True
            result['truncated_lines'] = num_lines - MAX_LINES_TO_RENDER

        # 成功消息
        if operation_type == 'create':
            result['message'] = f"成功创建文件，共 {num_lines} 行"
        else:
            if old_content == content:
                result['message'] = '文件内容未发生变化'
            else:
                result['message'] = f"成功更新文件，共 {num_lines} 行"

        logger.info(f"文件 {operation_type} 操作成功: {file_path} ({duration_ms}ms)")

        return json.dumps(result, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"文件写入工具执行失败: {e}")
        return json.dumps(
            {
                'error': f"工具执行失败: {str(e)}",
                'file_path': file_path,
                'success': False,
                'duration_ms': int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )


# 工具类实现，用于更高级的用法
class FileWriteTool:
    """
    文件写入工具的类实现，提供更高级的功能和配置选项。
    """

    def __init__(
        self,
        max_lines_preview: int = MAX_LINES_TO_RENDER,
        auto_create_dirs: bool = True,
        check_permissions: bool = True,
    ):
        """
        初始化文件写入工具。

        Args:
            max_lines_preview: 预览显示的最大行数
            auto_create_dirs: 是否自动创建目录
            check_permissions: 是否检查权限
        """
        self.max_lines_preview = max_lines_preview
        self.auto_create_dirs = auto_create_dirs
        self.check_permissions = check_permissions
        self.logger = logging.getLogger(self.__class__.__name__)

    def write_file(
        self,
        file_path: str,
        content: str,
        encoding: Optional[str] = None,
        line_ending: Optional[str] = None,
    ) -> dict:
        """
        写入文件并返回结构化结果。

        Args:
            file_path: 文件路径
            content: 文件内容
            encoding: 指定编码（可选）
            line_ending: 指定行结束符（可选）

        Returns:
            包含操作结果的字典
        """
        start_time = time.time()

        try:
            # 验证路径
            if not os.path.isabs(file_path):
                raise ValueError('文件路径必须是绝对路径')

            # 检查权限
            if self.check_permissions and not _has_write_permission(file_path):
                raise PermissionError(f"没有写入权限: {file_path}")

            # 检查文件状态
            file_exists = os.path.exists(file_path)
            operation_type = 'update' if file_exists else 'create'

            # 处理编码和行结束符
            if file_exists:
                if encoding is None:
                    encoding = _detect_file_encoding(file_path)
                if line_ending is None:
                    line_ending = _detect_line_endings(file_path)
            else:
                encoding = encoding or 'utf-8'
                line_ending = line_ending or '\n'

            # 读取原始内容
            old_content = ''
            if file_exists:
                with open(file_path, encoding=encoding) as f:
                    old_content = f.read()

            # 写入文件
            _write_file_content(file_path, content, encoding, line_ending)

            # 更新时间戳
            update_file_timestamp(file_path)

            # 生成结果
            duration_ms = int((time.time() - start_time) * 1000)

            return {
                'success': True,
                'operation_type': operation_type,
                'file_path': file_path,
                'num_lines': len(content.split('\n')),
                'encoding': encoding,
                'line_ending': line_ending,
                'duration_ms': duration_ms,
                'has_changes': old_content != content if file_exists else True,
            }

        except Exception as e:
            self.logger.error(f"文件写入失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path,
                'duration_ms': int((time.time() - start_time) * 1000),
            }


def main():
    """测试函数"""
    # 测试创建文件
    test_file = (
        '/gpfs/users/chenlu/north-deer-flow/src/tools/file_tools/test_file_write.txt'
    )
    test_content = 'Hello, World!\nThis is a test file.\nLine 3'

    result = file_write_tool.invoke(
        {'file_path': test_file, 'content': test_content},
    )
    print('创建文件测试:')
    print(result)
    print()

    # 更新文件时间戳（模拟读取）
    update_file_timestamp(test_file)
    # breakpoint()
    # 测试更新文件
    updated_content = 'Hello, World!\nThis is an updated test file.\nLine 3\nNew line 4'
    result = file_write_tool.invoke(
        {'file_path': test_file, 'content': updated_content},
    )
    print('更新文件测试:')
    print(result)
    # breakpoint()
    # 清理测试文件
    try:
        os.remove(test_file)
    except Exception:
        pass


if __name__ == '__main__':
    main()
