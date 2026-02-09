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
Base sandbox interface for secure code execution and file operations.

This module provides abstract base classes for implementing sandboxed execution environments.
Sandboxes isolate tool execution from the host system, improving security and enabling
deployment in production environments.
"""

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Any, TypeVar

from nexau.archs.main_sub.skill import Skill
from nexau.archs.session.session_manager import SessionManager
from nexau.core.utils import run_async_function_sync

logger = logging.getLogger(__name__)


class SandboxStatus(Enum):
    """Status of sandbox operations."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    TIMEOUT = "timeout"
    SUCCESS = "success"


class CodeLanguage(Enum):
    """Supported programming languages for code execution."""

    PYTHON = "python"


@dataclass
class CommandResult:
    """
    Result of a command execution in the sandbox.

    Attributes:
        status: Execution status (success, error, timeout, etc.)
        stdout: Standard output from the command
        stderr: Standard error output from the command
        exit_code: Exit code of the command (0 for success)
        duration_ms: Execution duration in milliseconds
        error: Error message if execution failed
        truncated: Whether output was truncated due to size limits
        original_stdout_length: Original length of stdout before truncation (if truncated)
        original_stderr_length: Original length of stderr before truncation (if truncated)
    """

    status: SandboxStatus
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    error: str | None = None
    truncated: bool = False
    original_stdout_length: int | None = None
    original_stderr_length: int | None = None


@dataclass
class CodeExecutionResult:
    """
    Result of code execution in the sandbox.

    Attributes:
        status: Execution status (success, error, timeout, etc.)
        language: Programming language used
        outputs: List of execution outputs (stdout, stderr, return values, etc.)
        error_type: Type of error if execution failed
        error_value: Error message if execution failed
        traceback: Stack trace if execution failed
        duration_ms: Execution duration in milliseconds
        truncated: Whether output was truncated due to size limits
    """

    status: SandboxStatus
    language: CodeLanguage
    outputs: list[dict[str, Any]] | None = None
    error_type: str | None = None
    error_value: str | None = None
    traceback: list[str] | None = None
    duration_ms: int = 0
    truncated: bool = False

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = []


@dataclass
class FileInfo:
    """
    Information about a file in the sandbox.

    Attributes:
        path: File path in the sandbox
        is_file: Whether it's a regular file
        is_directory: Whether it's a directory
        size: File size in bytes
        mode: File mode/permissions as integer
        permissions: File permissions as string (e.g., 'rwxr-xr-x')
        modified_time: File modification timestamp
        symlink_target: Target path if file is a symbolic link
        readable: Whether the file is readable
        writable: Whether the file is writable
        encoding: File encoding (e.g., 'utf-8')
    """

    path: str
    exists: bool = False
    is_file: bool = False
    is_directory: bool = False
    size: int = 0
    mode: int | None = None
    permissions: str | None = None
    modified_time: str | None = None
    symlink_target: str | None = None
    readable: bool = True
    writable: bool = True
    encoding: str | None = None


@dataclass
class FileOperationResult:
    """
    Result of a file operation in the sandbox.

    Attributes:
        status: Operation status (success, error, etc.)
        file_path: Path to the file operated on
        content: File content (for read operations)
        size: File size in bytes
        error: Error message if operation failed
        truncated: Whether content was truncated due to size limits
    """

    status: SandboxStatus
    file_path: str
    content: str | bytes | bytearray | None = None
    size: int = 0
    error: str | None = None
    truncated: bool = False


@dataclass(kw_only=True)
class BaseSandbox(ABC):
    """
    Abstract base class for sandbox implementations.

    This class defines the interface that all sandbox implementations must follow.
    Subclasses should implement the abstract methods to provide actual sandbox functionality.
    """

    sandbox_id: str | None = field(default=None)
    _work_dir: str = field(default_factory=os.getcwd)

    @property
    def work_dir(self):
        return Path(self._work_dir)

    # Bash command execution methods

    @abstractmethod
    def execute_bash(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Execute a bash command in the sandbox.

        Args:
            command: The bash command to execute
            timeout: Optional timeout in milliseconds (overrides default)
            cwd: Optional working directory
            user: Optional user to run the command as (not available in LocalSandbox)
            envs: Optional environment variables

        Returns:
            CommandResult containing execution results

        Raises:
            SandboxError: If command execution fails
        """
        pass

    # Code execution methods

    @abstractmethod
    def execute_code(
        self,
        code: str,
        language: CodeLanguage | str,
        timeout: int | None = None,
        user: str | None = None,
        envs: dict[str, str] | None = None,
    ) -> CodeExecutionResult:
        """
        Execute code in the specified programming language.

        Args:
            code: The code to execute
            language: Programming language (CodeLanguage enum or string)
            timeout: Optional timeout in milliseconds (overrides default)
            user: Optional user to run the code as
            envs: Optional environment variables

        Returns:
            CodeExecutionResult containing execution results and outputs

        Raises:
            SandboxError: If code execution fails
            ValueError: If language is not supported
        """
        pass

    # File operation methods

    @abstractmethod
    def read_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
        binary: bool = False,
    ) -> FileOperationResult:
        """
        Read a file from the sandbox.

        Args:
            file_path: Path to the file in the sandbox
            encoding: File encoding (default: utf-8)
            binary: Whether to read file in binary mode

        Returns:
            FileOperationResult containing file content

        Raises:
            SandboxError: If file read fails
        """
        pass

    @abstractmethod
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
            file_path: Path to the file in the sandbox
            content: Content to write (string or bytes)
            encoding: File encoding (default: utf-8)
            binary: Whether to write file in binary mode
            create_directories: Whether to create parent directories if they don't exist
            user: Optional user to run the create_directories command as

        Returns:
            FileOperationResult containing operation status

        Raises:
            SandboxError: If file write fails
        """
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> FileOperationResult:
        """
        Delete a file from the sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileOperationResult containing operation status

        Raises:
            SandboxError: If file deletion fails
        """
        pass

    @abstractmethod
    def list_files(
        self,
        directory_path: str,
        recursive: bool = False,
        pattern: str | None = None,
    ) -> list[FileInfo]:
        """
        List files in a directory in the sandbox.

        Args:
            directory_path: Path to the directory in the sandbox
            recursive: Whether to list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of FileInfo objects for matching files

        Raises:
            SandboxError: If directory listing fails
        """
        pass

    @abstractmethod
    def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in the sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    def get_file_info(self, file_path: str) -> FileInfo:
        """
        Get information about a file in the sandbox.

        Args:
            file_path: Path to the file in the sandbox

        Returns:
            FileInfo object containing file metadata

        Raises:
            SandboxError: If file info retrieval fails
        """
        pass

    @abstractmethod
    def create_directory(self, directory_path: str, parents: bool = True) -> bool:
        """
        Create a directory in the sandbox.

        Args:
            directory_path: Path to the directory to create
            parents: Whether to create parent directories if they don't exist

        Returns:
            True if directory created successfully, False otherwise

        Raises:
            SandboxFileError: If directory creation fails
        """
        pass

    @abstractmethod
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

        Raises:
            SandboxFileError: If edit operation fails
        """
        pass

    @abstractmethod
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

        Raises:
            SandboxFileError: If glob operation fails
        """
        pass

    # File upload/download methods

    @abstractmethod
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
            sandbox_path: Destination path in the sandbox
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status

        Raises:
            SandboxError: If file upload fails
        """
        pass

    @abstractmethod
    def download_file(
        self,
        sandbox_path: str,
        local_path: str,
        create_directories: bool = True,
    ) -> FileOperationResult:
        """
        Download a file from the sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the file in the sandbox
            local_path: Destination path on the local filesystem
            create_directories: Whether to create parent directories if they don't exist

        Returns:
            FileOperationResult containing operation status

        Raises:
            SandboxError: If file download fails
        """
        pass

    @abstractmethod
    def upload_directory(
        self,
        local_path: str,
        sandbox_path: str,
    ) -> bool:
        """
        Upload a directory from the local filesystem to the sandbox.

        Args:
            local_path: Path to the directory on the local filesystem
            sandbox_path: Destination path in the sandbox

        Returns:
            True if directory uploaded successfully, False otherwise

        Raises:
            SandboxError: If directory upload fails
        """
        pass

    @abstractmethod
    def download_directory(
        self,
        sandbox_path: str,
        local_path: str,
    ) -> bool:
        """
        Download a directory from the sandbox to the local filesystem.

        Args:
            sandbox_path: Path to the directory in the sandbox
            local_path: Destination path on the local filesystem

        Returns:
            True if directory downloaded successfully, False otherwise

        Raises:
            SandboxError: If directory download fails
        """
        pass

    def upload_skill(self, skill: Skill):
        local_folder = skill.folder
        sandbox_folder = self.work_dir / ".skills" / os.path.basename(local_folder)
        self.create_directory(str(sandbox_folder))
        self.upload_directory(str(local_folder), str(sandbox_folder))
        return str(sandbox_folder)

    # Utility methods

    def __str__(self):
        return f"{self.__class__.__name__}: {self.sandbox_id} ({self.work_dir})"

    def __repr__(self):
        return self.__str__()

    def dict(self):
        no_init_names = {f.name for f in fields(self) if f.init}
        return {k: getattr(self, k) for k in no_init_names}

    @staticmethod
    def _detect_file_encoding(raw_data: bytes) -> str:
        """
        Detect file encoding with fallback to utf-8.

        Args:
            raw_data: File binary content

        Returns:
            Detected encoding name
        """
        try:
            import chardet  # type: ignore

            result: dict[str, Any] = chardet.detect(raw_data)  # type: ignore
            encoding: str = result["encoding"]  # type: ignore
            if encoding and result["confidence"] > 0.7:  # type: ignore
                return encoding  # type: ignore
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error detecting encoding: {e}")

        return "utf-8"


TSandbox = TypeVar("TSandbox")


@dataclass(kw_only=True)
class BaseSandboxManager[TSandbox: "BaseSandbox"](ABC):
    """
    Abstract base class for sandbox manager.
    """

    _work_dir: str = field(default_factory=os.getcwd)
    start_future: Future[None] | None = field(default=None, init=False)
    pause_future: Future[None] | None = field(default=None, init=False)
    _session_context: dict[str, Any] = field(default_factory=dict, init=False)  # type: ignore
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=4),
        init=False,
        repr=False,
    )

    @property
    def work_dir(self):
        return Path(self._work_dir)

    # Unserialized fields
    _instance: TSandbox | None = field(default=None, repr=False, init=False)

    @property
    def instance(self) -> TSandbox | None:
        if self.start_future is not None:
            self.start_future.result()
        if not self.is_running():
            if self._session_context:
                logger.warning("Sandbox is not running. Try resuming it...")
                self.start_no_wait(**self._session_context)
                if self.start_future is not None:
                    self.start_future.result()
            else:
                raise SandboxError("Sandbox is not running. Please run `start_no_wait` first.")
        if self._instance is None:
            logger.warning("Sandbox start failed. Tools may not work normaly.")
        return self._instance

    @abstractmethod
    def start(self, session_manager: SessionManager | None, user_id: str, session_id: str, sandbox_config: dict[str, Any]) -> TSandbox:
        """Start a sandbox for a session."""
        ...

    @abstractmethod
    def stop(
        self,
    ) -> bool:
        """Stop a sandbox for a session."""
        ...

    @abstractmethod
    def pause(
        self,
    ) -> bool:
        """Pause a sandbox for a session."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the sandbox is running."""
        ...

    def persist_sandbox_state(
        self,
        session_manager: SessionManager | None,
        user_id: str,
        session_id: str,
        sandbox: BaseSandbox,
    ):
        """Persist sandbox state.

        功能说明1：将 sandbox 状态保存到 session manager
        功能说明2：如果发生事件循环冲突，静默失败（非关键操作）
        功能说明3：这避免了跨事件循环访问 asyncio 原语的问题
        """
        if session_manager is None:
            return None

        async def _persist_sandbox_state():
            sandbox_state = sandbox.dict()
            sandbox_state["sandbox_type"] = sandbox.__class__.__name__
            await session_manager.update_session_sandbox(
                user_id=user_id,
                session_id=session_id,
                sandbox_state=sandbox_state,
            )

        try:
            return run_async_function_sync(_persist_sandbox_state, raise_sync_error=False)
        except RuntimeError as e:
            # 事件循环冲突，静默失败（保存状态是非关键操作）
            if "bound to a different event loop" in str(e) or "Event loop is closed" in str(e):
                logger.warning(f"Event loop conflict persisting sandbox state: {e}. State not saved.")
                return None
            raise

    def load_sandbox_state(
        self,
        session_manager: SessionManager | None,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any] | None:
        """Load sandbox state.

        功能说明1：从 session manager 加载保存的 sandbox 状态
        功能说明2：如果发生事件循环冲突，返回 None 让调用者创建新 sandbox
        功能说明3：这避免了跨事件循环访问 asyncio 原语的问题
        """
        if session_manager is None:
            return None

        async def _load_sandbox_state():
            session = await session_manager.get_session(
                user_id=user_id,
                session_id=session_id,
            )
            if session and session.sandbox_state:
                return session.sandbox_state
            return None

        try:
            return run_async_function_sync(_load_sandbox_state, raise_sync_error=False)
        except RuntimeError as e:
            # 事件循环冲突，返回 None 让调用者创建新 sandbox
            if "bound to a different event loop" in str(e) or "Event loop is closed" in str(e):
                logger.warning(f"Event loop conflict loading sandbox state: {e}. Will create new sandbox.")
                return None
            raise

    def prepare_session_context(
        self,
        session_manager: SessionManager | None,
        user_id: str,
        session_id: str,
        sandbox_config: dict[str, Any],
        upload_assets: list[tuple[str, str]],
    ):
        """Prepare session context for lazy sandbox initialization.

        功能说明1：仅保存会话上下文，不启动 sandbox
        功能说明2：sandbox 会在首次调用 start_sync() 时延迟启动
        功能说明3：确保 sandbox 在正确的事件循环上下文中创建
        """
        self._session_context = {
            "session_manager": session_manager,
            "user_id": user_id,
            "session_id": session_id,
            "sandbox_config": sandbox_config,
            "upload_assets": upload_assets,
        }

    def start_no_wait(
        self,
        session_manager: SessionManager | None,
        user_id: str,
        session_id: str,
        sandbox_config: dict[str, Any],
        upload_assets: list[tuple[str, str]],
    ):
        self._session_context = {
            "session_manager": session_manager,
            "user_id": user_id,
            "session_id": session_id,
            "sandbox_config": sandbox_config,
            "upload_assets": upload_assets,
        }

        def _inner():
            sandbox = self.start(
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                sandbox_config=sandbox_config,
            )
            for src, tgt in upload_assets:
                sandbox.upload_directory(src, tgt)
            self._instance = sandbox

        self.start_future = self._executor.submit(_inner)
        return self.start_future

    def start_sync(self) -> TSandbox | None:
        """Start sandbox synchronously in the current thread/event loop.

        功能说明1：在当前线程同步启动 sandbox，避免 asyncio 事件循环问题
        功能说明2：E2B SDK 的 httpx 客户端会在当前事件循环中创建
        功能说明3：确保 sandbox 和使用它的代码在同一个事件循环上下文中
        功能说明4：用于解决跨线程/事件循环访问 asyncio 原语的问题
        """
        if self._instance is not None:
            return self._instance

        if not self._session_context:
            logger.warning("No session context available for sandbox start")
            return None

        logger.info("Starting sandbox synchronously in current event loop context...")
        sandbox = self.start(
            session_manager=self._session_context.get("session_manager"),
            user_id=self._session_context.get("user_id", ""),
            session_id=self._session_context.get("session_id", ""),
            sandbox_config=self._session_context.get("sandbox_config", {}),
        )

        # Upload assets if any
        upload_assets = self._session_context.get("upload_assets", [])
        for src, tgt in upload_assets:
            sandbox.upload_directory(src, tgt)

        self._instance = sandbox
        return sandbox

    def pause_no_wait(
        self,
    ):
        def _inner():
            self.pause()

        self.pause_thread = Thread(target=_inner)
        self.pause_thread.start()
        return self.pause_thread


class SandboxError(Exception):
    """Base exception for sandbox-related errors."""

    pass


class SandboxTimeoutError(SandboxError):
    """Exception raised when a sandbox operation times out."""

    pass


class SandboxExecutionError(SandboxError):
    """Exception raised when code or command execution fails in the sandbox."""

    pass


class SandboxFileError(SandboxError):
    """Exception raised when file operations fail in the sandbox."""

    pass


def extract_dataclass_init_kwargs[T](
    cls: type[T],
    m: Mapping[str, Any],
    *,
    ignore_extra: bool = True,
) -> dict[str, Any]:
    """
    Extract kwargs for constructing a dataclass `cls` from mapping `m`.

    - Only keeps keys that correspond to dataclass fields with init=True.
    - If ignore_extra is False, raises TypeError on unexpected keys.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")

    init_names = {f.name for f in fields(cls) if f.init}
    kwargs: dict[str, Any] = {k: v for k, v in m.items() if k in init_names}

    if not ignore_extra:
        extra = set(m) - init_names
        if extra:
            raise TypeError(f"Unexpected keys: {sorted(extra)}")

    return kwargs


SandboxAlias = {
    "local": "LocalSandbox",
    "e2b": "E2BSandbox",
}
