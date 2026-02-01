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

"""
E2B sandbox implementation for secure cloud-based code execution.

This implementation uses E2B (https://e2b.dev) to provide isolated cloud sandboxes
for secure code execution and file operations. E2B provides proper security isolation
and is suitable for production use.
"""

# pyright: reportUnknownParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from __future__ import annotations

import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base_sandbox import (
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    SandboxError,
    SandboxFileError,
    SandboxStatus,
    extract_dataclass_init_kwargs,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from e2b import Sandbox as TSandbox  # type: ignore
    from e2b.sandbox.filesystem.filesystem import FileType as TFileType  # type: ignore
else:
    TSandbox = Any
    TFileType = Any

try:
    from e2b import Sandbox  # type: ignore
    from e2b.sandbox.filesystem.filesystem import FileType  # type: ignore

    E2B_AVAILABLE = True
except ImportError:
    logger.warning("E2B SDK not installed. Install it with: pip install e2b")
    Sandbox = None
    FileType = None
    E2B_AVAILABLE = False  # type: ignore


@dataclass(kw_only=True)
class E2BSandbox(BaseSandbox):
    """
    E2B cloud sandbox implementation for secure code execution.

    This implementation uses E2B's cloud infrastructure to provide isolated,
    secure sandboxes for code execution and file operations. It's suitable
    for production use and provides proper security isolation.

    Note: Configuration (template, timeout, api_key, etc.) is managed by E2BSandboxManager.
    This class only contains runtime state.
    """

    _work_dir: str = field(default="/home/user")

    # Unserialized fields
    _sandbox: TSandbox | None = field(default=None, repr=False, init=False)

    @property
    def sandbox(self) -> TSandbox | None:
        return self._sandbox

    @sandbox.setter
    def sandbox(self, sandbox: TSandbox):
        self._sandbox = sandbox
        self._sandbox_id = sandbox.sandbox_id

    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
    ) -> CommandResult:
        """
        Execute a bash command in the E2B sandbox.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in milliseconds (overrides default)

        Returns:
            CommandResult containing execution results
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        start_time = time.time()

        if timeout is None:
            timeout = 120000  # Default: 2 minutes

        timeout_seconds = timeout / 1000.0
        max_output_size = 30000  # Default: 30000 characters

        try:
            # Execute command using E2B's commands.run
            result = self._sandbox.commands.run(
                cmd=command,
                timeout=int(timeout_seconds),
                cwd=str(self.work_dir),
            )

            duration_ms = int((time.time() - start_time) * 1000)

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            stdout_truncated = len(stdout) > max_output_size
            stderr_truncated = len(stderr) > max_output_size

            original_stdout_len = len(stdout) if stdout_truncated else None
            original_stderr_len = len(stderr) if stderr_truncated else None

            # Preserve the actual exit code from the command
            actual_exit_code = result.exit_code

            return CommandResult(
                status=SandboxStatus.SUCCESS if actual_exit_code == 0 else SandboxStatus.ERROR,
                stdout=stdout[:max_output_size],
                stderr=stderr[:max_output_size],
                exit_code=actual_exit_code,
                duration_ms=duration_ms,
                error=None if actual_exit_code == 0 else f"Command failed with exit code {actual_exit_code}",
                truncated=stdout_truncated or stderr_truncated,
                original_stdout_length=original_stdout_len,
                original_stderr_length=original_stderr_len,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to execute bash command: {e}")

            # Check if it's a timeout error
            if "timeout" in str(e).lower():
                return CommandResult(
                    status=SandboxStatus.TIMEOUT,
                    stdout="",
                    stderr="",
                    exit_code=-1,
                    duration_ms=duration_ms,
                    error=f"Command timed out after {timeout}ms",
                    truncated=False,
                )

            # Try to extract exit code from E2B exception message
            # E2B raises exceptions like "Command exited with code 1 and error:\n"
            exit_code = -1
            error_str = str(e)
            if "exited with code" in error_str.lower():
                import re

                match = re.search(r"code (\d+)", error_str)
                if match:
                    exit_code = int(match.group(1))

            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout="",
                stderr="",
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=f"Execution failed: {str(e)}",
                truncated=False,
            )

    def execute_code(
        self,
        code: str,
        language: CodeLanguage | str,
        timeout: int | None = None,
    ) -> CodeExecutionResult:
        """
        Execute Python code in the E2B sandbox.

        Args:
            code: The Python code to execute
            language: Programming language (must be "python" or CodeLanguage.PYTHON)
            timeout: Optional timeout in milliseconds (overrides default)

        Returns:
            CodeExecutionResult containing execution results and outputs
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        start_time = time.time()

        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                return CodeExecutionResult(
                    status=SandboxStatus.ERROR,
                    language=CodeLanguage.PYTHON,
                    error_type="ValueError",
                    error_value=f"Unsupported language: {language}. Only Python is supported.",
                    duration_ms=0,
                )

        if language != CodeLanguage.PYTHON:
            return CodeExecutionResult(
                status=SandboxStatus.ERROR,
                language=language,
                error_type="ValueError",
                error_value=f"Unsupported language: {language}. Only Python is supported.",
                duration_ms=0,
            )

        temp_file_path = None
        try:
            # Create temporary Python file in work_dir
            work_dir = str(self.work_dir) if self.work_dir else "/tmp"
            temp_filename = f"tmp_{uuid.uuid4().hex[:8]}.py"
            temp_file_path = f"{work_dir}/{temp_filename}"

            # Write code to temp file
            self.write_file(temp_file_path, code)

            # Execute the temp file
            result = self.execute_bash(f"python3 {temp_filename}", timeout)

            outputs: list[dict[str, Any]] = []
            if result.stdout:
                outputs.append({"type": "stdout", "text": result.stdout})
            if result.stderr:
                outputs.append({"type": "stderr", "text": result.stderr})

            return CodeExecutionResult(
                status=result.status,
                language=language,
                outputs=outputs,
                error_type=None if result.status == SandboxStatus.SUCCESS else "ExecutionError",
                error_value=result.error,
                traceback=[result.stderr] if result.stderr else None,
                duration_ms=result.duration_ms,
                truncated=result.truncated,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to execute code: {e}")
            return CodeExecutionResult(
                status=SandboxStatus.ERROR,
                language=language,
                error_type=type(e).__name__,
                error_value=str(e),
                duration_ms=duration_ms,
            )
        finally:
            # Clean up temp file
            if temp_file_path:
                try:
                    self.execute_bash(f"rm -f {temp_file_path}", timeout=5000)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file_path}: {e}")

    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file from the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox
            encoding: File encoding (default: utf-8)
            binary: Whether to read file in binary mode

        Returns:
            FileOperationResult containing file content
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Resolve to absolute path if relative
            resolved_path = file_path
            if not file_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_path = f"{work_dir}/{file_path}"

            max_output_size = 30000  # Default: 30000 characters

            content = self._sandbox._filesystem.read(resolved_path, format="bytes")  # type: ignore
            if not binary:
                content = content.decode(encoding)

            # Get file size
            file_info = self.get_file_info(resolved_path)
            file_size = file_info.size

            truncated = len(content) > max_output_size if isinstance(content, str) else len(content) > max_output_size

            if truncated:
                content = content[:max_output_size]

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                content=content,
                size=file_size,
                truncated=truncated,
            )

        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR, file_path=file_path, error=f"Failed to read file: {str(e)}", content=None
            )

    def write_file(
        self,
        file_path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        binary: bool = False,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Write content to a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox
            content: Content to write (string or bytes)
            encoding: File encoding (default: utf-8)
            binary: Whether to write file in binary mode
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Resolve to absolute path if relative
            resolved_path = file_path
            if not file_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_path = f"{work_dir}/{file_path}"

            # Create parent directories if needed
            if create_directories:
                parent_dir = str(Path(resolved_path).parent)
                if parent_dir and parent_dir != ".":
                    self._sandbox.commands.run(cmd=f"mkdir -p {parent_dir}")

            # Write file using E2B filesystem API
            if isinstance(content, bytes):
                # E2B expects string or bytes
                self._sandbox._filesystem.write(resolved_path, content)  # type: ignore
            else:
                self._sandbox._filesystem.write(resolved_path, content)  # type: ignore

            # Get file size - use stat command directly to avoid recursion issues
            size = 0
            try:
                result = self._sandbox.commands.run(cmd=f'stat -c "%s" "{resolved_path}"')
                if result.exit_code == 0:
                    size = int(result.stdout.strip())
            except Exception:
                pass

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=resolved_path,
                size=size,
            )

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to write file: {str(e)}",
            )

    def delete_file(self, file_path: str) -> FileOperationResult:
        """
        Delete a file from the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Resolve to absolute path if relative
            resolved_path = file_path
            if not file_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_path = f"{work_dir}/{file_path}"

            # Check if file exists
            if not self.file_exists(resolved_path):
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=resolved_path,
                    error=f"File does not exist: {resolved_path}",
                )

            # Use E2B filesystem remove
            self._sandbox._filesystem.remove(resolved_path)  # type: ignore

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
            )

        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to delete file: {str(e)}",
            )

    def list_files(
        self,
        directory_path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileInfo]:
        """
        List files in a directory in the E2B sandbox.

        Args:
            directory_path: Path to the directory in the sandbox
            recursive: Whether to list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of FileInfo objects for matching files
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if directory exists
            if not self.file_exists(directory_path):
                raise SandboxFileError(f"Directory does not exist: {directory_path}")

            # Use E2B filesystem list
            entries = self._sandbox._filesystem.list(directory_path)  # type: ignore

            files: list[FileInfo] = []
            for entry in entries:
                # Detect readable/writable from permissions
                readable = True
                writable = True
                if entry.mode:  # type: ignore
                    # Check owner read permission (bit 8)
                    readable = bool(entry.mode & 0o400)  # type: ignore
                    # Check owner write permission (bit 7)
                    writable = bool(entry.mode & 0o200)  # type: ignore

                # Convert E2B EntryInfo to our FileInfo
                file_info = FileInfo(
                    path=entry.path,  # type: ignore
                    exists=True,
                    is_file=entry.type == FileType.FILE,  # type: ignore
                    is_directory=entry.type == FileType.DIR,  # type: ignore
                    size=entry.size,  # type: ignore
                    mode=entry.mode,  # type: ignore
                    permissions=entry.permissions,  # type: ignore
                    modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),  # type: ignore
                    symlink_target=entry.symlink_target,  # type: ignore
                    readable=readable,
                    writable=writable,
                    encoding=None,  # Skip encoding detection in list for performance
                )
                files.append(file_info)

                # Recursively list subdirectories if requested
                if recursive and file_info.is_directory:
                    try:
                        subdir_files = self.list_files(file_info.path, recursive=True, pattern=pattern)
                        files.extend(subdir_files)
                    except Exception as e:
                        logger.warning(f"Failed to list subdirectory {file_info.path}: {e}")

            # Apply pattern filtering if specified
            if pattern:
                import fnmatch

                files = [f for f in files if fnmatch.fnmatch(Path(f.path).name, pattern)]

            return files

        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {e}")
            raise SandboxFileError(f"Failed to list files: {str(e)}")

    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            True if file exists, False otherwise
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            return self._sandbox._filesystem.exists(file_path)  # type: ignore
        except Exception:
            return False

    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the E2B sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileInfo object containing file metadata
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if file exists
            exists = self.file_exists(file_path)

            if not exists:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Use E2B filesystem get_info API
            try:
                entry = self._sandbox._filesystem.get_info(file_path)  # type: ignore
            except Exception:
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            # Detect readable/writable from permissions
            readable = True
            writable = True
            if entry.mode:  # type: ignore
                # Check owner read permission (bit 8)
                readable = bool(entry.mode & 0o400)  # type: ignore
                # Check owner write permission (bit 7)
                writable = bool(entry.mode & 0o200)  # type: ignore

            if entry.type == FileType.FILE:  # type: ignore
                raw_data = self._sandbox._filesystem.read(file_path, format="bytes")  # type: ignore
                encoding = self._detect_file_encoding(bytes(raw_data))
            else:
                encoding = None

            return FileInfo(
                path=file_path,
                exists=True,
                is_file=entry.type == FileType.FILE,  # type: ignore
                is_directory=entry.type == FileType.DIR,  # type: ignore
                size=entry.size,  # type: ignore
                mode=entry.mode,  # type: ignore
                permissions=entry.permissions,  # type: ignore
                modified_time=entry.modified_time.strftime("%Y-%m-%d %H:%M:%S"),  # type: ignore
                symlink_target=entry.symlink_target,  # type: ignore
                readable=readable,
                writable=writable,
                encoding=encoding,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise SandboxFileError(f"Failed to get file info: {str(e)}")

    def create_directory(self, directory_path: str, parents: bool = True) -> bool:
        """
        Create a directory in the E2B sandbox.

        Args:
            directory_path: Path to the directory to create
            parents: Whether to create parent directories if they don't exist

        Returns:
            True if directory created successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Use mkdir command
            cmd = f"mkdir -p {directory_path}" if parents else f"mkdir {directory_path}"
            result = self._sandbox.commands.run(cmd=cmd, cwd=str(self.work_dir))

            if result.exit_code != 0:
                raise SandboxFileError(f"Failed to create directory: {result.stderr}")

            return True

        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise SandboxFileError(f"Failed to create directory: {str(e)}")

    def edit_file(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
    ) -> FileOperationResult:
        """
        Edit a file by replacing old_string with new_string.

        Supports three operations:
        1. CREATE: Set old_string to empty string to create a new file
        2. UPDATE: Provide both old_string and new_string to update existing content
        3. REMOVE_CONTENT: Set new_string to empty string to remove the old_string content

        Args:
            file_path: Path to the file to edit
            old_string: String to replace (empty for file creation)
            new_string: Replacement string (empty for content removal)

        Returns:
            FileOperationResult containing operation status and details
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            file_exists = self.file_exists(file_path)

            # Determine operation type
            if old_string == "" and new_string != "":
                operation = "CREATE"
            elif old_string != "" and new_string == "":
                operation = "REMOVE_CONTENT"
            else:
                operation = "UPDATE"

            # Validate operation
            if operation == "CREATE" and file_exists:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File already exists: {file_path}. Use UPDATE operation instead.",
                )

            if operation != "CREATE" and not file_exists:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}. Use CREATE operation instead.",
                )

            # Read original content
            if file_exists:
                read_result = self.read_file(file_path, binary=False)
                if read_result.status != SandboxStatus.SUCCESS:
                    return read_result
                original_content = read_result.content
            else:
                original_content = ""

            assert isinstance(original_content, str), f"Unexpected content type: {type(original_content)}"

            # Validate string matching for UPDATE/REMOVE operations
            if operation != "CREATE":
                if old_string not in original_content:
                    # Try to normalize common escape sequence issues from LLM
                    normalized_old_string = old_string.replace("\\\\n", "\\n").replace("\\\\t", "\\t").replace("\\\\r", "\\r")
                    normalized_new_string = new_string.replace("\\\\n", "\\n").replace("\\\\t", "\\t").replace("\\\\r", "\\r")

                    if normalized_old_string != old_string and normalized_old_string in original_content:
                        # Use normalized version
                        old_string = normalized_old_string
                        new_string = normalized_new_string
                    else:
                        return FileOperationResult(
                            status=SandboxStatus.ERROR,
                            file_path=file_path,
                            error=f"String to replace not found in file: {file_path}",
                        )

                matches = original_content.count(old_string)
                if matches > 1:
                    return FileOperationResult(
                        status=SandboxStatus.ERROR,
                        file_path=file_path,
                        error=(
                            f"Found {matches} matches of the string to replace. "
                            "For safety, this tool only supports replacing exactly one occurrence at a time. "
                            "Add more lines of context to your edit and try again."
                        ),
                    )

            # Apply the edit
            if operation == "CREATE":
                updated_content = new_string
            elif operation == "REMOVE_CONTENT":
                updated_content = original_content.replace(old_string, "", 1)
            else:
                updated_content = original_content.replace(old_string, new_string, 1)

            # Write the updated content
            write_result = self.write_file(file_path, updated_content)

            return write_result

        except Exception as e:
            logger.error(f"Failed to edit file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to edit file: {str(e)}",
            )

    def glob(
        self,
        pattern: str,
        recursive: bool = True,
    ) -> list[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '*.py', '**/*.txt')
            recursive: Whether to search recursively (default: True)

        Returns:
            List of file paths matching the pattern
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Use find command with pattern matching
            if recursive:
                # Handle ** glob pattern by converting to find command
                if "**" in pattern:
                    # Remove ** and extract the file pattern
                    file_pattern = pattern.replace("**/", "").replace("**", "")
                    if file_pattern.startswith("/"):
                        file_pattern = file_pattern[1:]
                    search_dir = "."
                    cmd = f'find "{search_dir}" -type f -name "{file_pattern}"'
                elif "/" in pattern:
                    search_dir, file_pattern = pattern.rsplit("/", 1)
                    search_dir = search_dir or "."
                    cmd = f'find "{search_dir}" -name "{file_pattern}"'
                else:
                    search_dir = "."
                    file_pattern = pattern
                    cmd = f'find "{search_dir}" -name "{file_pattern}"'
            else:
                cmd = f'ls -1 "{pattern}" 2>/dev/null || true'
            result = self._sandbox.commands.run(cmd=cmd, cwd=str(self.work_dir))

            if result.exit_code != 0 and result.exit_code != 1:
                raise SandboxFileError(f"Glob command failed: {result.stderr}")

            # Parse output
            matches = [line.strip() for line in result.stdout.split("\n") if line.strip()]

            return sorted(matches)

        except Exception as e:
            logger.error(f"Failed to glob pattern {pattern}: {e}")
            raise SandboxFileError(f"Failed to glob: {str(e)}")

    def upload_file(
        self,
        local_path: str,
        sandbox_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Upload a file from the local filesystem to the E2B sandbox.

        Args:
            local_path: Path to the file on the local filesystem
            sandbox_path: Destination path in the sandbox
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            local_file = Path(local_path)

            if not local_file.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=sandbox_path,
                    error=f"Source file does not exist: {local_path}",
                )

            # Read local file
            with open(local_file, "rb") as f:
                content = f.read()

            # Write to sandbox
            return self.write_file(sandbox_path, content, binary=True, create_directories=create_directories)

        except Exception as e:
            logger.error(f"Failed to upload file from {local_path} to {sandbox_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=sandbox_path,
                error=f"Failed to upload file: {str(e)}",
            )

    def download_file(
        self,
        sandbox_path: str,
        local_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Download a file from the E2B sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the file in the sandbox
            local_path: Destination path on the local filesystem
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Read from sandbox
            read_result = self.read_file(sandbox_path, binary=True)

            if read_result.status != SandboxStatus.SUCCESS:
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=local_path,
                    error=f"Failed to read file from sandbox: {read_result.error}",
                )

            # Write to local filesystem
            local_file = Path(local_path)

            if create_directories:
                local_file.parent.mkdir(parents=True, exist_ok=True)

            with open(local_file, "wb") as f:
                content = read_result.content
                if content is None:
                    return FileOperationResult(
                        status=SandboxStatus.ERROR,
                        file_path=local_path,
                        error="Empty content returned from sandbox read",
                    )
                if isinstance(content, str):
                    f.write(content.encode("utf-8"))
                else:
                    f.write(bytes(content))

            file_size = local_file.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=local_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to download file from {sandbox_path} to {local_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=local_path,
                error=f"Failed to download file: {str(e)}",
            )

    def upload_directory(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> bool:
        """
        Upload a directory from the local filesystem to the E2B sandbox.

        Args:
            local_path: Path to the directory on the local filesystem
            sandbox_path: Destination path in the sandbox

        Returns:
            True if directory uploaded successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            local_dir = Path(local_path)

            if not local_dir.exists():
                raise SandboxFileError(f"Source directory does not exist: {local_path}")

            if not local_dir.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {local_path}")

            # Create destination directory
            self.create_directory(sandbox_path, parents=True)

            # Upload all files recursively
            for item in local_dir.rglob("*"):
                if item.is_file():
                    # Calculate relative path
                    rel_path = item.relative_to(local_dir)
                    dest_path = f"{sandbox_path}/{rel_path}"

                    # Upload file
                    result = self.upload_file(str(item), dest_path, create_directories=True)

                    if result.status != SandboxStatus.SUCCESS:
                        logger.warning(f"Failed to upload {item}: {result.error}")

            return True

        except Exception as e:
            logger.error(f"Failed to upload directory from {local_path} to {sandbox_path}: {e}")
            raise SandboxFileError(f"Failed to upload directory: {str(e)}")

    def download_directory(
        self,
        sandbox_path: str,
        local_path: str,
    ) -> bool:
        """
        Download a directory from the E2B sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the directory in the sandbox
            local_path: Destination path on the local filesystem

        Returns:
            True if directory downloaded successfully
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"
        if not self._sandbox:
            raise SandboxError("Sandbox not started. Call start() first.")

        try:
            # Check if source directory exists
            if not self.file_exists(sandbox_path):
                raise SandboxFileError(f"Source directory does not exist: {sandbox_path}")

            # Create local directory
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)

            # List all files recursively
            files = self.list_files(sandbox_path, recursive=True)

            # Resolve sandbox_path to absolute path for comparison
            resolved_sandbox_path = sandbox_path
            if not sandbox_path.startswith("/"):
                work_dir = str(self.work_dir) if self.work_dir else "/home/user"
                resolved_sandbox_path = f"{work_dir}/{sandbox_path}"

            # Download all files
            for file_info in files:
                if file_info.is_file:
                    # Calculate relative path using absolute paths
                    file_abs_path = Path(file_info.path)
                    sandbox_abs_path = Path(resolved_sandbox_path)

                    try:
                        rel_path = file_abs_path.relative_to(sandbox_abs_path)
                    except ValueError:
                        # If relative_to fails, try to extract the relative part manually
                        file_path_str = str(file_abs_path)
                        sandbox_path_str = str(sandbox_abs_path)
                        if file_path_str.startswith(sandbox_path_str):
                            rel_path = Path(file_path_str[len(sandbox_path_str) :].lstrip("/"))
                        else:
                            logger.warning(f"Cannot determine relative path for {file_info.path}")
                            continue

                    dest_path = local_dir / rel_path

                    # Download file
                    result = self.download_file(file_info.path, str(dest_path), create_directories=True)

                    if result.status != SandboxStatus.SUCCESS:
                        logger.warning(f"Failed to download {file_info.path}: {result.error}")

            return True

        except Exception as e:
            logger.error(f"Failed to download directory from {sandbox_path} to {local_path}: {e}")
            raise SandboxFileError(f"Failed to download directory: {str(e)}")


@dataclass(kw_only=True)
class E2BSandboxManager(BaseSandboxManager[E2BSandbox]):
    """
    Manager for E2B sandbox lifecycle and configuration.

    This class handles:
    - Creating and configuring E2B sandboxes
    - Connecting to existing sandboxes
    - Persisting and loading sandbox state
    - Managing sandbox lifecycle (start/stop)
    """

    # E2B configuration fields
    _work_dir: str = field(default="/home/user")
    template: str = field(default_factory=lambda: os.getenv("E2B_TEMPLATE", "base"))
    timeout: int = field(default_factory=lambda: int(os.getenv("E2B_TIMEOUT", "300")))
    api_key: str | None = field(default_factory=lambda: os.getenv("E2B_API_KEY"))
    api_url: str | None = field(default_factory=lambda: os.getenv("E2B_API_URL"))
    force_http: bool = field(default_factory=lambda: os.getenv("E2B_FORCE_HTTP") in ["true", "True", "1"])
    metadata: dict[str, str] = field(default_factory=lambda: {})
    envs: dict[str, str] = field(default_factory=lambda: {})

    def start(self, session_manager: Any, user_id: str, session_id: str, sandbox_config: dict[str, Any]) -> E2BSandbox:
        """
        Start an E2B sandbox for a session.

        Args:
            session_manager: Session manager instance
            user_id: User ID
            session_id: Session ID

        Returns:
            Configured and started E2BSandbox instance
        """
        assert E2B_AVAILABLE and Sandbox is not None, "E2B SDK not installed. Install it with: pip install e2b"

        # Load existing sandbox state if available
        sandbox_state = self.load_sandbox_state(session_manager, user_id, session_id)

        # Try to restore from saved state
        if sandbox_state and sandbox_state.get("sandbox_id"):
            try:
                logger.info(f"Attempting to restore E2B sandbox from state: {sandbox_state.get('sandbox_id')}")

                # Create sandbox instance from saved state
                sandbox_kwargs = extract_dataclass_init_kwargs(E2BSandbox, sandbox_state)
                sandbox = E2BSandbox(**sandbox_kwargs)

                if not sandbox.sandbox_id:
                    raise SandboxError("Sandbox ID not found in state, failed to restore.")

                # Try to reconnect to existing E2B sandbox
                connect_opts: dict[str, Any] = {"api_key": self.api_key}
                if self.api_url:
                    connect_opts["api_url"] = self.api_url

                sandbox.sandbox = Sandbox.connect(sandbox_id=sandbox.sandbox_id, **connect_opts)
                logger.info(f"Successfully reconnected to E2B sandbox: {sandbox.sandbox_id}")

                self._instance = sandbox

                return sandbox

            except Exception as e:
                logger.warning(f"Failed to restore sandbox from state: {e}. Creating new sandbox.")

        # Create new sandbox
        logger.info(f"Creating new E2B sandbox with template: {self.template}")

        # Build create options
        create_opts: dict[str, Any] = {
            "template": self.template,
            "timeout": self.timeout,
            "api_key": self.api_key,
        }

        if self.metadata:
            create_opts["metadata"] = self.metadata
        if self.envs:
            create_opts["envs"] = self.envs
        if self.api_url:
            create_opts["api_url"] = self.api_url
        if self.force_http:
            create_opts["force_http"] = True
        create_opts["auto_pause"] = True

        # Create E2B sandbox via SDK
        e2b_sandbox = Sandbox.beta_create(**create_opts)  # type: ignore

        sandbox_kwargs = extract_dataclass_init_kwargs(E2BSandbox, sandbox_config)
        if "work_dir" not in sandbox_kwargs:
            sandbox_kwargs["_work_dir"] = "/home/user"

        # Create our wrapper instance
        sandbox = E2BSandbox(**sandbox_kwargs)

        sandbox.sandbox = e2b_sandbox
        sandbox.sandbox_id = e2b_sandbox.sandbox_id

        logger.info(f"E2B sandbox created with ID: {sandbox.sandbox_id}")

        # Persist sandbox state
        self.persist_sandbox_state(session_manager, user_id, session_id, sandbox)

        self._instance = sandbox

        return sandbox

    def stop(self) -> bool:
        """
        Stop the E2B sandbox for a session.
        """
        assert E2B_AVAILABLE, "E2B SDK not installed. Install it with: pip install e2b"

        instance: E2BSandbox | None = self._instance

        if not instance:
            logger.info("No sandbox instance found, nothing to stop")
            return False

        try:
            if instance.sandbox:
                # Kill the sandbox to destroy the container
                instance.sandbox.kill()
                logger.info(f"E2B sandbox {instance.sandbox_id} destroyed")
            return True
        except Exception as e:
            logger.error(f"Failed to destroy E2B sandbox: {e}")
            return False

    def pause(self) -> bool:
        """
        Pause the E2B sandbox for a session.
        """
        try:
            if self.instance and self.instance.sandbox:
                self.instance.sandbox.beta_pause()
                return True
        except Exception as e:
            logger.error(f"Failed to pause E2B sandbox: {e}")
            return False
        return True

    def is_running(self) -> bool:
        if self._instance is None or self._instance.sandbox is None:
            return False
        try:
            return self._instance.sandbox.is_running()
        except Exception as e:
            logger.error(f"Failed to check E2B sandbox status: {e}")
            return False
