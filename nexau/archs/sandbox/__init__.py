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

"""Sandbox module for secure code execution and file operations."""

from typing import TYPE_CHECKING

from .base_sandbox import (
    BaseSandbox,
    BaseSandboxManager,
    CodeExecutionResult,
    CodeLanguage,
    CommandResult,
    FileInfo,
    FileOperationResult,
    SandboxError,
    SandboxExecutionError,
    SandboxFileError,
    SandboxStatus,
    SandboxTimeoutError,
    extract_dataclass_init_kwargs,
)
from .local_sandbox import LocalSandbox, LocalSandboxManager

if TYPE_CHECKING:
    from .e2b_sandbox import E2BSandbox, E2BSandboxManager


def __getattr__(name: str) -> object:
    """Lazily import optional E2B sandbox classes."""
    if name == "E2BSandbox":
        from .e2b_sandbox import E2BSandbox

        return E2BSandbox
    if name == "E2BSandboxManager":
        from .e2b_sandbox import E2BSandboxManager

        return E2BSandboxManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "BaseSandbox",
    "LocalSandbox",
    "E2BSandbox",
    "SandboxStatus",
    "CodeLanguage",
    "CommandResult",
    "CodeExecutionResult",
    "FileInfo",
    "FileOperationResult",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxExecutionError",
    "SandboxFileError",
    "LocalSandboxManager",
    "E2BSandboxManager",
    "BaseSandboxManager",
    "extract_dataclass_init_kwargs",
]
