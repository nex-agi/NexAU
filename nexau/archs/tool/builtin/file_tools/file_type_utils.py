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
File type utilities module.

Provides file format detection to determine if a file is binary or text.
Used to prevent writing binary files (e.g., xlsx, pdf) in text mode.
"""

import os

# ============================================================================
# Binary file extensions (grouped by category for easy maintenance)
# ============================================================================

# Office documents
_OFFICE_BINARY = {".xlsx", ".docx", ".pptx", ".xls", ".doc", ".ppt"}

# Image formats
_IMAGE_BINARY = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tiff", ".tif"}

# Archives
_ARCHIVE_BINARY = {".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z"}

# Audio
_AUDIO_BINARY = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

# Video
_VIDEO_BINARY = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

# Database
_DATABASE_BINARY = {".sqlite", ".sqlite3", ".db"}

# Executables
_EXECUTABLE_BINARY = {".exe", ".dll", ".so", ".dylib", ".bin"}

# Fonts
_FONT_BINARY = {".ttf", ".otf", ".woff", ".woff2", ".eot"}

# Serialized data
_SERIALIZED_BINARY = {".pickle", ".pkl", ".parquet", ".feather", ".npy", ".npz"}

# Other binary formats
_OTHER_BINARY = {".pdf", ".pyc", ".pyo", ".class", ".o", ".a", ".lib"}

# Merge all binary extensions
_BINARY_EXTENSIONS: frozenset[str] = frozenset(
    _OFFICE_BINARY
    | _IMAGE_BINARY
    | _ARCHIVE_BINARY
    | _AUDIO_BINARY
    | _VIDEO_BINARY
    | _DATABASE_BINARY
    | _EXECUTABLE_BINARY
    | _FONT_BINARY
    | _SERIALIZED_BINARY
    | _OTHER_BINARY
)

# ============================================================================
# Text file extensions (grouped by category)
# ============================================================================

# Plain text
_PLAIN_TEXT = {".txt", ".text", ".log"}

# Markup languages
_MARKUP_TEXT = {
    ".md",
    ".markdown",
    ".rst",
    ".adoc",
    ".html",
    ".htm",
    ".xhtml",
    ".xml",
    ".xsl",
    ".xslt",
}

# Data formats
_DATA_TEXT = {".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".ini", ".cfg", ".conf"}

# Programming languages
_CODE_TEXT = {
    # Python
    ".py",
    ".pyi",
    ".pyw",
    # JavaScript/TypeScript
    ".js",
    ".mjs",
    ".cjs",
    ".jsx",
    ".ts",
    ".tsx",
    ".mts",
    ".cts",
    # JVM
    ".java",
    ".kt",
    ".kts",
    ".scala",
    # C/C++
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".cxx",
    # .NET
    ".cs",
    ".fs",
    ".vb",
    # Others
    ".go",
    ".rs",
    ".swift",
    ".rb",
    ".php",
    ".pl",
    ".pm",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".psm1",
    ".bat",
    ".cmd",
    ".sql",
    ".graphql",
    ".gql",
    ".r",
    ".R",
    ".jl",
    ".lua",
    ".vim",
    ".el",
}

# Web related
_WEB_TEXT = {".css", ".scss", ".sass", ".less", ".vue", ".svelte"}

# Config files (no extension or special names)
_CONFIG_TEXT = {".env", ".gitignore", ".dockerignore", ".editorconfig", ".prettierrc"}
_CONFIG_NAMES = {"Makefile", "Dockerfile", "Jenkinsfile"}

# Merge all text extensions
_TEXT_EXTENSIONS: frozenset[str] = frozenset(
    _PLAIN_TEXT | _MARKUP_TEXT | _DATA_TEXT | _CODE_TEXT | _WEB_TEXT | _CONFIG_TEXT | _CONFIG_NAMES
)


# ============================================================================
# Public functions
# ============================================================================


def is_binary_extension(file_path: str) -> bool:
    """
    Check if the file extension indicates a binary format.

    Args:
        file_path: File path or filename.

    Returns:
        True if binary format, False otherwise.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in _BINARY_EXTENSIONS


def is_text_extension(file_path: str) -> bool:
    """
    Check if the file extension indicates a text format.

    Args:
        file_path: File path or filename.

    Returns:
        True if known text format, False otherwise.
    """
    ext = os.path.splitext(file_path)[1].lower()
    name = os.path.basename(file_path)
    return ext in _TEXT_EXTENSIONS or name in _TEXT_EXTENSIONS
