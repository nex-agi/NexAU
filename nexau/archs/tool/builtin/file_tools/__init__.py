# Copyright (c) Nex-AGI. All rights reserved.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .apply_patch import apply_patch
    from .glob_tool import glob
    from .list_directory import list_directory
    from .read_file import read_file
    from .read_many_files import read_many_files
    from .read_visual_file import read_visual_file
    from .replace import replace
    from .search_file_content import search_file_content
    from .write_file import write_file

__all__ = [
    "read_file",
    "read_visual_file",
    "write_file",
    "replace",
    "apply_patch",
    "glob",
    "list_directory",
    "read_many_files",
    "search_file_content",
]


def _cache_export(name: str, value: object) -> object:
    globals()[name] = value
    return value


def __getattr__(name: str) -> object:
    """Lazily resolve individual file tools for Rust bridge bindings."""

    if name == "read_file":
        from .read_file import read_file

        return _cache_export(name, read_file)
    if name == "read_visual_file":
        from .read_visual_file import read_visual_file

        return _cache_export(name, read_visual_file)
    if name == "write_file":
        from .write_file import write_file

        return _cache_export(name, write_file)
    if name == "replace":
        from .replace import replace

        return _cache_export(name, replace)
    if name == "apply_patch":
        from .apply_patch import apply_patch

        return _cache_export(name, apply_patch)
    if name == "glob":
        from .glob_tool import glob

        return _cache_export(name, glob)
    if name == "list_directory":
        from .list_directory import list_directory

        return _cache_export(name, list_directory)
    if name == "read_many_files":
        from .read_many_files import read_many_files

        return _cache_export(name, read_many_files)
    if name == "search_file_content":
        from .search_file_content import search_file_content

        return _cache_export(name, search_file_content)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
