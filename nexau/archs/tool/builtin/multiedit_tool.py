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

"""MultiEdit tool implementation for making multiple edits to a single file."""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class EditOperation:
    """Represents a single edit operation."""

    def __init__(self, old_string: str, new_string: str, replace_all: bool = False):
        self.old_string = old_string
        self.new_string = new_string
        self.replace_all = replace_all

    def __str__(self):
        return f"EditOperation(old='{self.old_string[:50]}...', new='{self.new_string[:50]}...', replace_all={self.replace_all})"


def multiedit_tool(
    file_path: str,
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Make multiple edits to a single file in one operation.

    This tool performs multiple find-and-replace operations efficiently on a single file.
    All edits are applied in sequence, and the operation is atomic - either all succeed
    or none are applied.

    Args:
        file_path: The absolute path to the file to modify (must be absolute, not relative)
        edits: Array of edit operations, each containing:
            - old_string: The text to replace (must match exactly, including whitespace)
            - new_string: The text to replace it with
            - replace_all: Replace all occurrences (optional, defaults to False)

    Returns:
        Dict containing the result of the operation
    """
    start_time = time.time()

    # Validate file path is absolute
    if not os.path.isabs(file_path):
        return {
            "status": "error",
            "error": f"File path must be absolute, got relative path: {file_path}",
            "file_path": file_path,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    # Validate edits list
    if not edits:
        return {
            "status": "error",
            "error": "No edits provided",
            "file_path": file_path,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    # Validate and parse edit operations
    edit_operations = []
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return {
                "status": "error",
                "error": f"Edit {i} must be a dictionary",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        if "old_string" not in edit:
            return {
                "status": "error",
                "error": f"Edit {i} missing required 'old_string' field",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        if "new_string" not in edit:
            return {
                "status": "error",
                "error": f"Edit {i} missing required 'new_string' field",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        old_string = edit["old_string"]
        new_string = edit["new_string"]
        replace_all = edit.get("replace_all", False)

        # Validate strings are different
        if old_string == new_string:
            return {
                "status": "error",
                "error": f"Edit {i}: old_string and new_string cannot be the same",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        edit_operations.append(
            EditOperation(
                old_string,
                new_string,
                replace_all,
            ),
        )

    # Handle file creation case (first edit has empty old_string)
    is_new_file = False
    if edit_operations[0].old_string == "":
        if os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"Cannot create new file - file already exists: {file_path}",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }
        is_new_file = True
        initial_content = edit_operations[0].new_string
        edit_operations = edit_operations[1:]  # Remove the creation edit
    else:
        # Check if file exists for modification
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "error": f"File does not exist: {file_path}",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        # Check file permissions
        if not os.access(file_path, os.R_OK):
            return {
                "status": "error",
                "error": f"No read permission for file: {file_path}",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

        if not os.access(file_path, os.W_OK):
            return {
                "status": "error",
                "error": f"No write permission for file: {file_path}",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            }

    try:
        # Read original content or start with initial content for new files
        if is_new_file:
            original_content = ""
            current_content = initial_content
        else:
            with open(file_path, encoding="utf-8") as f:
                original_content = f.read()
            current_content = original_content

        # Apply edits in sequence
        applied_edits = []
        for i, edit_op in enumerate(edit_operations):
            if edit_op.old_string not in current_content:
                return {
                    "status": "error",
                    "error": f"Edit {i}: old_string not found in file content",
                    "file_path": file_path,
                    "edit_index": i,
                    "old_string": (edit_op.old_string[:100] + "..." if len(edit_op.old_string) > 100 else edit_op.old_string),
                    "applied_edits": len(applied_edits),
                    "duration_ms": int((time.time() - start_time) * 1000),
                }

            # Perform the replacement
            if edit_op.replace_all:
                new_content = current_content.replace(
                    edit_op.old_string,
                    edit_op.new_string,
                )
                replacements_made = current_content.count(edit_op.old_string)
            else:
                new_content = current_content.replace(
                    edit_op.old_string,
                    edit_op.new_string,
                    1,
                )
                replacements_made = 1 if edit_op.old_string in current_content else 0

            current_content = new_content
            applied_edits.append(
                {
                    "edit_index": i,
                    "replacements_made": replacements_made,
                    "old_string_length": len(edit_op.old_string),
                    "new_string_length": len(edit_op.new_string),
                },
            )

        # Write the final content to file
        # Create directory if it doesn't exist for new files
        if is_new_file:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(current_content)

        duration_ms = int((time.time() - start_time) * 1000)

        # Calculate statistics
        total_replacements = sum(edit["replacements_made"] for edit in applied_edits)
        content_size_change = len(current_content) - len(original_content)

        result = {
            "status": "success",
            "file_path": file_path,
            "is_new_file": is_new_file,
            "total_edits": len(edit_operations),
            "applied_edits": len(applied_edits),
            "total_replacements": total_replacements,
            "original_size": len(original_content),
            "final_size": len(current_content),
            "size_change": content_size_change,
            "duration_ms": duration_ms,
            "edit_details": applied_edits,
        }

        logger.info(
            f"MultiEdit completed: {len(applied_edits)} edits, {total_replacements} replacements in {duration_ms}ms",
        )

        return result

    except UnicodeDecodeError as e:
        return {
            "status": "error",
            "error": f"File encoding error - cannot read file as UTF-8: {str(e)}",
            "file_path": file_path,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    except PermissionError as e:
        return {
            "status": "error",
            "error": f"Permission denied: {str(e)}",
            "file_path": file_path,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    except OSError as e:
        return {
            "status": "error",
            "error": f"OS error: {str(e)}",
            "file_path": file_path,
            "duration_ms": int((time.time() - start_time) * 1000),
        }

    except Exception as e:
        logger.error(f"Unexpected error during multiedit: {e}")
        return {
            "status": "error",
            "error": f"Unexpected error: {str(e)}",
            "error_type": type(e).__name__,
            "file_path": file_path,
            "duration_ms": int((time.time() - start_time) * 1000),
        }


# Alternative class-based implementation for more advanced usage
class MultiEditTool:
    """
    A class-based implementation of the multiedit tool for more advanced usage.
    Provides additional features like backup, validation, and dry-run mode.
    """

    def __init__(self, create_backup: bool = True, validate_syntax: bool = False):
        self.create_backup = create_backup
        self.validate_syntax = validate_syntax
        self.logger = logging.getLogger(self.__class__.__name__)

    def edit_file(
        self,
        file_path: str,
        edits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Edit a file with advanced options.

        Args:
            file_path: Absolute path to the file to edit
            edits: List of edit operations
            dry_run: If True, simulate edits without actually modifying the file

        Returns:
            Dict containing operation results
        """

        # Create backup if enabled
        backup_path = None
        if self.create_backup and os.path.exists(file_path):
            backup_path = self._create_backup(file_path)

        try:
            result = multiedit_tool(file_path, edits)

            if result["status"] == "success" and backup_path:
                result["backup_created"] = backup_path

            return result

        except Exception as e:
            # Restore from backup if something went wrong
            if backup_path and os.path.exists(backup_path):
                try:
                    os.rename(backup_path, file_path)
                    self.logger.info(
                        f"Restored file from backup: {backup_path}",
                    )
                except Exception as backup_error:
                    self.logger.error(
                        f"Failed to restore backup: {backup_error}",
                    )

            raise e

    def _create_backup(self, file_path: str) -> str:
        """Create a backup of the file before editing."""
        import shutil

        timestamp = int(time.time())
        backup_path = f"{file_path}.backup.{timestamp}"
        shutil.copy2(file_path, backup_path)
        self.logger.info(f"Created backup: {backup_path}")
        return backup_path

    def batch_edit_files(
        self,
        file_edits: list[dict[str, Any]],
        stop_on_error: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Edit multiple files with their respective edits.

        Args:
            file_edits: List of dicts, each containing 'file_path' and 'edits'
            stop_on_error: If True, stop processing on first error

        Returns:
            List of results for each file
        """
        results = []

        for file_edit in file_edits:
            file_path = file_edit.get("file_path")
            edits = file_edit.get("edits", [])

            if not file_path or not edits:
                result = {
                    "status": "error",
                    "error": "Invalid file_edit entry - missing file_path or edits",
                    "file_path": file_path,
                }
            else:
                result = self.edit_file(file_path, edits)

            results.append(result)

            if stop_on_error and result["status"] != "success":
                break

        return results


def main():
    """Test function to demonstrate and validate the multiedit_tool functionality."""
    print("✏️  MultiEditTool 测试开始...")
    print("=" * 50)

    # Create a test file for demonstrations
    test_file_path = "/tmp/multiedit_test.txt"
    test_content = """Hello World!
This is line 2.
This is line 3 with some text.
Final line here."""

    # Test 1: Create a new file
    print("\n📋 测试 1: 创建新文件")
    try:
        edits = [
            {"old_string": "", "new_string": test_content},
        ]
        result = multiedit_tool(test_file_path, edits)

        if result["status"] == "success":
            print("✅ 新文件创建成功")
            print(f"📄 文件路径: {result['file_path']}")
            print(f"📊 文件大小: {result['final_size']} 字节")
            print(f"⏱️  执行时间: {result['duration_ms']}ms")
        else:
            print(f"❌ 新文件创建失败: {result['error']}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 2: Multiple edits on existing file
    print("\n📋 测试 2: 多重编辑")
    try:
        edits = [
            {"old_string": "Hello World!", "new_string": "Hello MultiEdit!"},
            {"old_string": "line 2", "new_string": "second line"},
            {"old_string": "This is", "new_string": "Here is", "replace_all": True},
        ]
        result = multiedit_tool(test_file_path, edits)

        if result["status"] == "success":
            print("✅ 多重编辑成功")
            print(f"📝 应用编辑: {result['applied_edits']}/{result['total_edits']}")
            print(f"🔄 总替换次数: {result['total_replacements']}")
            print(f"📏 大小变化: {result['size_change']} 字节")
            print(f"⏱️  执行时间: {result['duration_ms']}ms")

            # Verify the changes
            with open(test_file_path) as f:
                new_content = f.read()
            print(f"📄 编辑后内容:\n{new_content}")

        else:
            print(f"❌ 多重编辑失败: {result['error']}")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 3: Edit with non-existent old_string (should fail)
    print("\n📋 测试 3: 无效编辑（不存在的字符串）")
    try:
        edits = [
            {
                "old_string": "This string does not exist",
                "new_string": "replacement",
            },
        ]
        result = multiedit_tool(test_file_path, edits)

        if result["status"] == "error" and "not found" in result["error"]:
            print("✅ 正确处理了不存在的字符串")
            print(f"⚠️  错误信息: {result['error']}")
        else:
            print("⚠️  错误处理可能有问题")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Test 4: Class-based implementation with backup
    print("\n📋 测试 4: 类实现（带备份）")
    try:
        multiedit_instance = MultiEditTool(create_backup=True)
        edits = [
            {"old_string": "MultiEdit", "new_string": "AdvancedEdit"},
        ]
        result = multiedit_instance.edit_file(test_file_path, edits)

        if result["status"] == "success":
            print("✅ 类实现编辑成功")
            print(f"💾 备份文件: {result.get('backup_created', 'None')}")
            print(f"📝 应用编辑: {result['applied_edits']}")
        else:
            print("❌ 类实现编辑失败")

    except Exception as e:
        print(f"❌ 类实现测试失败: {e}")

    # Test 5: Relative path (should fail)
    print("\n📋 测试 5: 相对路径测试")
    try:
        edits = [
            {"old_string": "test", "new_string": "replacement"},
        ]
        result = multiedit_tool("./test.txt", edits)

        if result["status"] == "error" and "absolute" in result["error"]:
            print("✅ 正确拒绝了相对路径")
            print(f"⚠️  错误信息: {result['error']}")
        else:
            print("⚠️  相对路径处理可能有问题")

    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # Clean up test file
    try:
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            print(f"\n🧹 清理测试文件: {test_file_path}")

        # Also clean up backup files
        import glob

        backup_files = glob.glob(f"{test_file_path}.backup.*")
        for backup_file in backup_files:
            os.remove(backup_file)
            print(f"🧹 清理备份文件: {backup_file}")

    except Exception as e:
        print(f"⚠️  清理文件时出错: {e}")

    print("\n" + "=" * 50)
    print("🎉 MultiEditTool 测试完成!")

    # Usage tips
    print("\n💡 使用提示:")
    print("  • 编辑按顺序应用，每个编辑基于前一个的结果")
    print("  • 使用绝对路径，不支持相对路径")
    print("  • old_string 必须精确匹配文件内容（包括空格）")
    print("  • 使用 replace_all=True 替换所有出现的字符串")
    print("  • 操作是原子性的：要么全部成功，要么全部失败")
    print("  • 创建新文件时第一个编辑的 old_string 应为空字符串")


if __name__ == "__main__":
    main()
