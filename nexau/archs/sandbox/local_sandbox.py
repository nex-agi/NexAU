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
Local sandbox implementation using native Python commands.

This is a naive implementation that executes commands directly on the local system
without any isolation. It is intended for development and testing purposes only.
For production use, consider using proper sandboxing solutions like E2B or Docker.
"""

import glob as glob_module
import logging
import os
import shutil
import stat as stat_module
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, override

from .base_sandbox import (
    BASH_TOOL_RESULTS_BASE_PATH,
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    SandboxConfig,
    SandboxFileError,
    SandboxStatus,
    extract_dataclass_init_kwargs,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class LocalSandbox(BaseSandbox):
    """
    Local sandbox implementation that executes commands directly on the host system.

    This implementation uses subprocess for command execution and standard file operations.

    Warning: This implementation does NOT provide any security isolation and should
    only be used for development and testing purposes.
    """

    def _ensure_working_directory(self) -> Path:
        """
        Ensure the working directory exists.

        Returns:
            Path to the working directory
        """
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        return Path(self.work_dir)

    def _save_output_to_temp_file(
        self,
        command: str,
        stdout: str,
        stderr: str,
        temp_dir: str | None = None,
    ) -> str:
        """
        Save command output to temporary files in the local filesystem.
        If temp_dir is provided, overwrites existing files; otherwise creates new directory.

        Args:
            command: The command that was executed
            stdout: Standard output content
            stderr: Standard error content
            temp_dir: Optional existing temp directory path (for streaming updates)

        Returns:
            The directory path where files were saved
        """
        # Generate unique ID if temp_dir not provided
        if temp_dir is None:
            temp_dir = f"{BASH_TOOL_RESULTS_BASE_PATH}/{uuid.uuid4().hex[:8]}"

        try:
            # Create the directory if it doesn't exist
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            # Write/overwrite command to command.txt
            Path(f"{temp_dir}/command.txt").write_text(command, encoding="utf-8")

            # Write/overwrite stdout to stdout.log
            Path(f"{temp_dir}/stdout.log").write_text(stdout, encoding="utf-8")

            # Write/overwrite stderr to stderr.log
            Path(f"{temp_dir}/stderr.log").write_text(stderr, encoding="utf-8")

            return temp_dir
        except Exception as e:
            logger.error(f"Failed to save output to temp file: {e}")
            raise Exception(f"Failed to save output to temp file: {e}")

    def _resolve_path(self, path: str) -> Path:
        """
        Resolve a path to an absolute path.

        If the path is relative, it will be resolved against the sandbox working directory.
        If the path is absolute, it will be used as-is.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Absolute Path object
        """
        p = Path(path)
        if p.is_absolute():
            return p
        else:
            return self.work_dir / p

    def _build_local_envs(self, per_call_envs: dict[str, str] | None = None) -> dict[str, str] | None:
        """Build environment variables for local subprocess execution.

        Merges os.environ + self.envs + per_call_envs.
        Returns None if no custom envs are set (subprocess inherits parent env).

        Args:
            per_call_envs: Optional per-call environment variables

        Returns:
            Merged envs dict including os.environ, or None to inherit parent env
        """
        merged = self._merge_envs(per_call_envs)
        if merged is None:
            return None
        # For local subprocess, we must include os.environ as base
        return {**os.environ, **merged}

    @override
    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
        background: bool = False,
        save_output_to_temp_file: bool = False,
    ) -> CommandResult:
        """
        Execute a bash command in the sandbox.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in milliseconds (overrides default)
            cwd: Optional working directory
            user: Optional user to run the command as (not available in LocalSandbox)
            envs: Optional environment variables
            background: Optional flag to run the command in the background
            save_output_to_temp_file: Optional flag to save the output (stdout and stderr) to a temporary file

        Returns:
            CommandResult containing execution results
        """
        if user is not None:
            logger.warning(f"User {user} is not used in local sandbox.")

        start_time = time.time()
        work_dir = self._ensure_working_directory()

        if timeout is None:
            timeout = 120000  # Default: 2 minutes

        timeout_seconds = timeout / 1000.0
        max_output_size = 30000  # Default: 30000 characters

        try:
            if background:
                # Generate temp directory for streaming if requested
                temp_dir = None
                if save_output_to_temp_file:
                    temp_dir = f"{BASH_TOOL_RESULTS_BASE_PATH}/{uuid.uuid4().hex[:8]}"

                # Background mode: start process without waiting
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=cwd or str(work_dir),
                    env=self._build_local_envs(envs),
                )
                bg_pid = process.pid

                task_info: dict[str, Any] = {
                    "process": process,
                    "command": command,
                    "start_time": start_time,
                    "finished": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                    "temp_dir": temp_dir,
                }

                def _consume_output(proc: subprocess.Popen[str], info: dict[str, Any]) -> None:
                    try:
                        if proc.stdout:
                            for line in proc.stdout:
                                info["stdout"] += line
                                # Overwrite log files with complete output if streaming enabled
                                if save_output_to_temp_file and info.get("temp_dir"):
                                    try:
                                        self._save_output_to_temp_file(
                                            info["command"],
                                            info["stdout"],
                                            info["stderr"],
                                            temp_dir=info["temp_dir"],
                                        )
                                    except Exception as e:
                                        logger.error(f"Failed to update streaming log files: {e}")
                    except (ValueError, OSError):
                        pass

                def _consume_stderr(proc: subprocess.Popen[str], info: dict[str, Any]) -> None:
                    try:
                        if proc.stderr:
                            for line in proc.stderr:
                                info["stderr"] += line
                                # Overwrite log files with complete output if streaming enabled
                                if save_output_to_temp_file and info.get("temp_dir"):
                                    try:
                                        self._save_output_to_temp_file(
                                            info["command"],
                                            info["stdout"],
                                            info["stderr"],
                                            temp_dir=info["temp_dir"],
                                        )
                                    except Exception as e:
                                        logger.error(f"Failed to update streaming log files: {e}")
                    except (ValueError, OSError):
                        pass

                def _wait_process(proc: subprocess.Popen[str], info: dict[str, Any]) -> None:
                    try:
                        exit_code = proc.wait()
                        info["exit_code"] = exit_code
                        if exit_code != 0:
                            info["error"] = f"Command failed with exit code {exit_code}"
                    except Exception as exc:
                        info["error"] = str(exc)
                    finally:
                        info["finished"] = True

                stdout_thread = threading.Thread(target=_consume_output, args=(process, task_info), daemon=True)
                stderr_thread = threading.Thread(target=_consume_stderr, args=(process, task_info), daemon=True)
                wait_thread = threading.Thread(target=_wait_process, args=(process, task_info), daemon=True)
                stdout_thread.start()
                stderr_thread.start()
                wait_thread.start()
                task_info["stdout_thread"] = stdout_thread
                task_info["stderr_thread"] = stderr_thread
                task_info["wait_thread"] = wait_thread

                self._background_tasks[bg_pid] = task_info
                duration_ms = int((time.time() - start_time) * 1000)

                stdout_msg = f"Background task started (pid: {bg_pid})"
                if save_output_to_temp_file:
                    stdout_msg += f"\nOutput will be saved to {temp_dir}/ when task completes"

                return CommandResult(
                    status=SandboxStatus.SUCCESS,
                    stdout=stdout_msg,
                    stderr="",
                    exit_code=0,
                    duration_ms=duration_ms,
                    background_pid=bg_pid,
                )

            # Foreground mode
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd or str(work_dir),
                env=self._build_local_envs(envs),
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                duration_ms = int((time.time() - start_time) * 1000)

                return CommandResult(
                    status=SandboxStatus.TIMEOUT,
                    stdout=stdout[:max_output_size] if stdout else "",
                    stderr=stderr[:max_output_size] if stderr else "",
                    exit_code=process.returncode or -1,
                    duration_ms=duration_ms,
                    error=f"Command timed out after {timeout}ms",
                    truncated=len(stdout or "") > max_output_size or len(stderr or "") > max_output_size,
                )

            duration_ms = int((time.time() - start_time) * 1000)

            # Save output to temp file if requested
            temp_file_path = None
            if save_output_to_temp_file:
                try:
                    temp_file_path = self._save_output_to_temp_file(command, stdout or "", stderr or "")
                except Exception as e:
                    logger.error(f"Failed to save output to temp file: {e}")

            stdout_truncated = len(stdout) > max_output_size if stdout else False
            stderr_truncated = len(stderr) > max_output_size if stderr else False

            original_stdout_len = len(stdout) if stdout_truncated else None
            original_stderr_len = len(stderr) if stderr_truncated else None

            # Append temp file path info to stdout if saved
            stdout_output = stdout[:max_output_size] if stdout else ""
            if temp_file_path:
                stdout_output += f"\n\n[Output saved to: {temp_file_path}]"

            return CommandResult(
                status=SandboxStatus.SUCCESS if process.returncode == 0 else SandboxStatus.ERROR,
                stdout=stdout_output,
                stderr=stderr[:max_output_size] if stderr else "",
                exit_code=process.returncode,
                duration_ms=duration_ms,
                error=None if process.returncode == 0 else f"Command failed with exit code {process.returncode}",
                truncated=stdout_truncated or stderr_truncated,
                original_stdout_length=original_stdout_len,
                original_stderr_length=original_stderr_len,
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Failed to execute bash command: {e}")
            return CommandResult(
                status=SandboxStatus.ERROR,
                stdout="",
                stderr="",
                exit_code=-1,
                duration_ms=duration_ms,
                error=f"Execution failed: {str(e)}",
                truncated=False,
            )

    @override
    def get_background_task_status(self, pid: int) -> CommandResult:
        """
        Get the status and output of a background task.

        Args:
            pid: The process ID of the background task

        Returns:
            CommandResult with current status and accumulated output
        """
        if pid not in self._background_tasks:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=f"Background task not found: pid={pid}",
                exit_code=-1,
                error=f"Background task not found: pid={pid}",
            )

        task_info = self._background_tasks[pid]
        max_output_size = 30000
        duration_ms = int((time.time() - task_info["start_time"]) * 1000)

        stdout = task_info.get("stdout", "") or ""
        stderr = task_info.get("stderr", "") or ""
        temp_file_path = task_info.get("temp_dir")

        if task_info["finished"]:
            exit_code = task_info["exit_code"]

            # Append temp file path info to stdout if saved
            stdout_output = stdout[:max_output_size]
            if temp_file_path:
                stdout_output += f"\n\n[Output saved to: {temp_file_path}]"

            return CommandResult(
                status=SandboxStatus.SUCCESS if exit_code == 0 else SandboxStatus.ERROR,
                stdout=stdout_output,
                stderr=stderr[:max_output_size],
                exit_code=exit_code,
                duration_ms=duration_ms,
                error=task_info.get("error"),
                truncated=len(stdout) > max_output_size or len(stderr) > max_output_size,
                background_pid=pid,
            )

        # Task is still running, return accumulated output so far
        return CommandResult(
            status=SandboxStatus.RUNNING,
            stdout=stdout[:max_output_size],
            stderr=stderr[:max_output_size],
            exit_code=-1,
            duration_ms=duration_ms,
            truncated=len(stdout) > max_output_size or len(stderr) > max_output_size,
            background_pid=pid,
        )

    @override
    def kill_background_task(self, pid: int) -> CommandResult:
        """
        Kill a background task.

        Args:
            pid: The process ID of the background task

        Returns:
            CommandResult with the kill operation result
        """
        if pid not in self._background_tasks:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=f"Background task not found: pid={pid}",
                exit_code=-1,
                error=f"Background task not found: pid={pid}",
            )

        task_info = self._background_tasks[pid]
        process: subprocess.Popen[str] = task_info["process"]

        try:
            process.kill()
            process.wait(timeout=5)
            # Wait for consumer threads to finish flushing output
            for thread_key in ("stdout_thread", "stderr_thread", "wait_thread"):
                t = task_info.get(thread_key)
                if t is not None:
                    t.join(timeout=2)
            duration_ms = int((time.time() - task_info["start_time"]) * 1000)
            del self._background_tasks[pid]
            return CommandResult(
                status=SandboxStatus.SUCCESS,
                stdout=f"Background task (pid={pid}) killed successfully",
                exit_code=0,
                duration_ms=duration_ms,
                background_pid=pid,
            )
        except Exception as e:
            return CommandResult(
                status=SandboxStatus.ERROR,
                stderr=str(e),
                exit_code=-1,
                error=f"Failed to kill background task: {str(e)[:200]}",
                background_pid=pid,
            )

    @override
    def execute_code(
        self,
        code: str,
        language: CodeLanguage | str,
        timeout: int | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionResult:
        """
        Execute Python code in the sandbox.

        Args:
            code: The Python code to execute
            language: Programming language (must be "python" or CodeLanguage.PYTHON)
            timeout: Optional timeout in milliseconds (overrides default)
            user: Optional user to run the command as (not available in LocalSandbox)
            envs: Optional environment variables to set

        Returns:
            CodeExecutionResult containing execution results and outputs
        """
        if user is not None:
            logger.warning(f"User {user} is not used in local sandbox.")

        start_time = time.time()

        if isinstance(language, str):
            try:
                language = CodeLanguage(language.lower())
            except ValueError:
                return CodeExecutionResult(
                    status=SandboxStatus.ERROR,
                    language=CodeLanguage.PYTHON,
                    error_type="ValueError",
                    error_value=f"Unsupported language: {language}.",
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

        temp_file = None
        try:
            # Create temporary Python file in work_dir
            import tempfile

            work_path = self.work_dir if self.work_dir else Path.cwd()

            # Create temp file in work_dir
            fd, temp_file = tempfile.mkstemp(suffix=".py", dir=str(work_path), text=True)
            try:
                # Write code to temp file
                os.write(fd, code.encode("utf-8"))
            finally:
                os.close(fd)

            # Execute the temp file
            result = self.execute_bash(f"python3 {Path(temp_file).name}", timeout, envs=envs)

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
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")

    @override
    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file from the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)
            encoding: File encoding (default: utf-8)
            binary: Whether to read file in binary mode

        Returns:
            FileOperationResult containing file content
        """
        try:
            full_path = self._resolve_path(file_path)

            if not full_path.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}",
                )

            file_size = full_path.stat().st_size
            max_output_size = 30000  # Default: 30000 characters

            content: str | bytes
            if binary:
                with open(full_path, "rb") as f:
                    content = f.read()
            else:
                with open(full_path, encoding=encoding) as f:
                    content = f.read()

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
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to read file: {str(e)}",
            )

    @override
    def write_file(
        self,
        file_path: str,
        content: str | bytes,
        encoding: str = "utf-8",
        binary: bool = False,
        create_directories: bool = True,
        user: str | None = None,
    ) -> FileOperationResult:
        """
        Write content to a file in the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)
            content: Content to write (string or bytes)
            encoding: File encoding (default: utf-8)
            binary: Whether to write file in binary mode
            create_directories: Whether to create parent directories if they don't exist
            user: Optional user to run the create_directories command as (not available in LocalSandbox)

        Returns:
            FileOperationResult containing operation status
        """
        if user is not None:
            logger.warning(f"User {user} is not used in local sandbox.")

        try:
            full_path = self._resolve_path(file_path)

            if create_directories:
                full_path.parent.mkdir(parents=True, exist_ok=True)

            if binary:
                with open(full_path, "wb") as f:
                    f.write(content if isinstance(content, bytes) else content.encode(encoding))
            else:
                with open(full_path, "w", encoding=encoding) as f:
                    f.write(content if isinstance(content, str) else content.decode(encoding))

            file_size = full_path.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to write file: {str(e)}",
            )

    @override
    def delete_file(self, file_path: str) -> FileOperationResult:
        """
        Delete a file from the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)

        Returns:
            FileOperationResult containing operation status
        """
        try:
            full_path = self._resolve_path(file_path)

            if not full_path.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=file_path,
                    error=f"File does not exist: {file_path}",
                )

            if full_path.is_dir():
                shutil.rmtree(full_path)
            else:
                full_path.unlink()

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

    @override
    def list_files(
        self,
        directory_path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileInfo]:
        """
        List files in a directory in the sandbox.

        Args:
            directory_path: Path to the directory in the sandbox (relative to working directory or absolute)
            recursive: Whether to list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of FileInfo objects for matching files
        """
        try:
            dir_path = self._resolve_path(directory_path)

            if not dir_path.exists():
                raise SandboxFileError(f"Directory does not exist: {directory_path}")

            if not dir_path.is_dir():
                raise SandboxFileError(f"Path is not a directory: {directory_path}")

            files: list[FileInfo] = []

            if pattern:
                if recursive:
                    glob_pattern = f"**/{pattern}"
                else:
                    glob_pattern = pattern
                paths = dir_path.glob(glob_pattern)
            else:
                if recursive:
                    paths = dir_path.rglob("*")
                else:
                    paths = dir_path.iterdir()

            for path in paths:
                try:
                    stat_result = path.stat()
                    files.append(
                        FileInfo(
                            path=str(path),
                            exists=True,
                            is_file=path.is_file(),
                            is_directory=path.is_dir(),
                            size=stat_result.st_size if path.is_file() else 0,
                            modified_time=datetime.fromtimestamp(stat_result.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                            readable=os.access(path, os.R_OK),
                            writable=os.access(path, os.W_OK),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to get info for {path}: {e}")
                    continue

            return files

        except Exception as e:
            logger.error(f"Failed to list files in {directory_path}: {e}")
            raise SandboxFileError(f"Failed to list files: {str(e)}")

    @override
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)

        Returns:
            True if file exists, False otherwise
        """
        try:
            return self._resolve_path(file_path).exists()
        except Exception:
            return False

    @override
    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the sandbox.

        Args:
            file_path: Path to the file in the sandbox (relative to working directory or absolute)

        Returns:
            FileInfo object containing file metadata
        """
        try:
            path = self._resolve_path(file_path)

            if not path.exists():
                return FileInfo(
                    path=file_path,
                    exists=False,
                )

            stat_result = path.stat()

            if path.is_file():
                with open(path, "rb") as f:
                    raw_data = f.read()
                    encoding = self._detect_file_encoding(raw_data)
            else:
                encoding = None

            return FileInfo(
                path=file_path,
                exists=True,
                is_file=path.is_file(),
                is_directory=path.is_dir(),
                size=stat_result.st_size,
                mode=stat_result.st_mode,
                permissions=stat_module.filemode(stat_result.st_mode)[1:],
                modified_time=datetime.fromtimestamp(stat_result.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                readable=os.access(path, os.R_OK),
                writable=os.access(path, os.W_OK),
                encoding=encoding,
            )

        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise SandboxFileError(f"Failed to get file info: {str(e)}")

    @override
    def create_directory(self, directory_path: str, parents: bool = True, user: str | None = None) -> bool:
        """
        Create a directory in the sandbox.

        Args:
            directory_path: Path to the directory to create (relative to working directory or absolute)
            parents: Whether to create parent directories if they don't exist

        Returns:
            True if directory created successfully
        """
        try:
            path = self._resolve_path(directory_path)

            path.mkdir(parents=parents, exist_ok=True)
            return True

        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            raise SandboxFileError(f"Failed to create directory: {str(e)}")

    @override
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
            file_path: Path to the file to edit (relative to working directory or absolute)
            old_string: String to replace (empty for file creation)
            new_string: Replacement string (empty for content removal)

        Returns:
            FileOperationResult containing operation status and details
        """
        try:
            full_path = self._resolve_path(file_path)
            file_exists = full_path.exists()

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
                with open(full_path, "rb") as f:
                    raw_data = f.read()
                    encoding = self._detect_file_encoding(raw_data)
                    original_content = raw_data.decode(encoding)
            else:
                original_content = ""
                encoding = "utf-8"

            # Validate string matching for UPDATE/REMOVE operations
            if operation != "CREATE":
                if old_string not in original_content:
                    # Try to normalize common escape sequence issues from LLM
                    # LLM might double-escape sequences like \\n when it should be \n
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

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the updated content
            with open(full_path, "w", encoding=encoding, newline="") as f:
                f.write(updated_content)

            file_size = full_path.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=file_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to edit file {file_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=file_path,
                error=f"Failed to edit file: {str(e)}",
            )

    @override
    def glob(
        self,
        pattern: str,
        recursive: bool = True,
        user: str | None = None,
    ) -> list[str]:
        """
        Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '*.py', '**/*.txt')
            recursive: Whether to search recursively (default: True)

        Returns:
            List of file paths matching the pattern (relative to working directory)
        """
        try:
            # If pattern is absolute, use it directly
            # Otherwise, resolve it against working directory
            pattern_path = Path(pattern)
            if pattern_path.is_absolute():
                search_pattern = str(pattern_path)
            else:
                search_pattern = str(Path(self.work_dir) / pattern)

            # Use glob with recursive support
            matches = glob_module.glob(search_pattern, recursive=recursive)

            # Convert absolute paths to relative paths (relative to work_dir)
            result: list[str] = []
            for match in matches:
                match_path = Path(match)
                try:
                    # Try to make it relative to work_dir
                    rel_path = match_path.relative_to(self.work_dir)
                    result.append(str(rel_path))
                except ValueError:
                    # If not under work_dir, return absolute path
                    result.append(str(match_path))

            return sorted(result)

        except Exception as e:
            logger.error(f"Failed to glob pattern {pattern}: {e}")
            raise SandboxFileError(f"Failed to glob: {str(e)}")

    @override
    def upload_file(
        self,
        local_path: str,
        sandbox_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Upload a file from the local filesystem to the sandbox.

        Args:
            local_path: Path to the file on the local filesystem
            sandbox_path: Destination path in the sandbox (relative to working directory or absolute)
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        try:
            src = Path(local_path)
            dst = self._resolve_path(sandbox_path)

            if not src.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=sandbox_path,
                    error=f"Source file does not exist: {local_path}",
                )

            if create_directories:
                dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)

            file_size = dst.stat().st_size

            return FileOperationResult(
                status=SandboxStatus.SUCCESS,
                file_path=sandbox_path,
                size=file_size,
            )

        except Exception as e:
            logger.error(f"Failed to upload file from {local_path} to {sandbox_path}: {e}")
            return FileOperationResult(
                status=SandboxStatus.ERROR,
                file_path=sandbox_path,
                error=f"Failed to upload file: {str(e)}",
            )

    @override
    def download_file(
        self,
        sandbox_path: str,
        local_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Download a file from the sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the file in the sandbox (relative to working directory or absolute)
            local_path: Destination path on the local filesystem
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status
        """
        try:
            src = self._resolve_path(sandbox_path)
            dst = Path(local_path)

            if not src.exists():
                return FileOperationResult(
                    status=SandboxStatus.ERROR,
                    file_path=local_path,
                    error=f"Source file does not exist: {sandbox_path}",
                )

            if create_directories:
                dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)

            file_size = dst.stat().st_size

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

    @override
    def upload_directory(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> bool:
        """
        Upload a directory from the local filesystem to the sandbox.

        Args:
            local_path: Path to the directory on the local filesystem
            sandbox_path: Destination path in the sandbox (relative to working directory or absolute)

        Returns:
            True if directory uploaded successfully
        """
        try:
            src = Path(local_path)
            dst = self._resolve_path(sandbox_path)

            if not src.exists():
                raise SandboxFileError(f"Source directory does not exist: {local_path}")

            if not src.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {local_path}")

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(src, dst)
            return True

        except Exception as e:
            logger.error(f"Failed to upload directory from {local_path} to {sandbox_path}: {e}")
            raise SandboxFileError(f"Failed to upload directory: {str(e)}")

    @override
    def download_directory(
        self,
        sandbox_path: str,
        local_path: str,
    ) -> bool:
        """
        Download a directory from the sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the directory in the sandbox (relative to working directory or absolute)
            local_path: Destination path on the local filesystem

        Returns:
            True if directory downloaded successfully
        """
        try:
            src = self._resolve_path(sandbox_path)
            dst = Path(local_path)

            if not src.exists():
                raise SandboxFileError(f"Source directory does not exist: {sandbox_path}")

            if not src.is_dir():
                raise SandboxFileError(f"Source path is not a directory: {sandbox_path}")

            if dst.exists():
                shutil.rmtree(dst)

            shutil.copytree(src, dst)
            return True

        except Exception as e:
            logger.error(f"Failed to download directory from {sandbox_path} to {local_path}: {e}")
            raise SandboxFileError(f"Failed to download directory: {str(e)}")


@dataclass(kw_only=True)
class LocalSandboxManager(BaseSandboxManager[LocalSandbox]):
    """
    Manager for local sandbox lifecycle and configuration.

    This class handles:
    - Creating and configuring local sandboxes
    - Persisting and loading sandbox state
    - Managing sandbox lifecycle (start/stop)
    """

    @override
    def start(self, session_manager: Any, user_id: str, session_id: str, sandbox_config: SandboxConfig) -> LocalSandbox:
        """
        Start a local sandbox for a session.

        Args:
            session_manager: Session manager instance
            user_id: User ID
            session_id: Session ID
            sandbox_config: Typed sandbox configuration

        Returns:
            Configured and started LocalSandbox instance
        """
        # Load existing sandbox state if available
        sandbox_state = self.load_sandbox_state(session_manager, user_id, session_id)

        # Try to restore from saved state
        if sandbox_state and sandbox_state.get("sandbox_id"):
            try:
                logger.info(f"Attempting to restore local sandbox from state: {sandbox_state.get('sandbox_id')}")

                # Create sandbox instance from saved state
                sandbox_kwargs = extract_dataclass_init_kwargs(LocalSandbox, sandbox_state)
                sandbox = LocalSandbox(**sandbox_kwargs)

                logger.info(f"Successfully restored local sandbox: {sandbox.sandbox_id}")

                return sandbox

            except Exception as e:
                logger.warning(f"Failed to restore sandbox from state: {e}. Creating new sandbox.")

        # Create new sandbox
        logger.info(f"Creating new local sandbox with ID: {session_id}")

        sandbox = LocalSandbox(_work_dir=sandbox_config.work_dir, envs=sandbox_config.envs)

        logger.info(f"Local sandbox created with ID: {sandbox.sandbox_id}, work_dir: {sandbox.work_dir}")

        # Persist sandbox state
        self.persist_sandbox_state(session_manager, user_id, session_id, sandbox)

        return sandbox

    @override
    def stop(
        self,
    ) -> bool:
        """
        Stop the local sandbox for a session.

        Note: For local sandboxes, this just logs the stop event.
        The working directory is preserved for future use.
        """
        return True

    @override
    def pause(self) -> bool:
        """
        Stop the local sandbox for a session.

        Note: For local sandboxes, this just logs the pause event.
        The working directory is preserved for future use.
        """
        return True

    @override
    def is_running(self) -> bool:
        return self._instance is not None
