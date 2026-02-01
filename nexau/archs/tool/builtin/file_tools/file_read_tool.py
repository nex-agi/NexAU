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
File Read Tool - A tool for reading text and image files with comprehensive validation.

This tool provides safe file reading capabilities with features like:
- Text file reading with offset and limit support
- Image file reading with base64 encoding
- File size validation and truncation
- Intelligent file suggestion for missing files
- Comprehensive error handling and user feedback

Based on the TypeScript FileReadTool implementation.
"""

import base64
import importlib
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from nexau.archs.sandbox import BaseSandbox, LocalSandbox, SandboxStatus
from nexau.archs.tool.builtin.file_tools.file_state import update_file_timestamp

# Import file state management for read/write coordination

logger = logging.getLogger(__name__)

# Configuration constants
MAX_LINES_TO_READ = 2000
MAX_LINE_LENGTH = 2000
MAX_OUTPUT_SIZE = 0.25 * 1024 * 1024  # 0.25MB in bytes
MAX_IMAGE_SIZE = 3.75 * 1024 * 1024  # 3.75MB in bytes

# Common image extensions
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}

# Common text file extensions for syntax highlighting hints
TEXT_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".scss",
    ".sass",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".csv",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".ps1",
    ".bat",
    ".cmd",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".clj",
    ".hs",
    ".ml",
    ".fs",
    ".r",
    ".m",
    ".mm",
    ".pl",
    ".pm",
    ".lua",
    ".vim",
    ".el",
    ".lisp",
    ".scm",
    ".rkt",
    ".jl",
    ".nim",
    ".cr",
    ".d",
    ".zig",
    ".odin",
    ".v",
    ".dart",
    ".ex",
    ".exs",
    ".erl",
    ".hrl",
}


def detect_file_encoding(file_path: str, sandbox: BaseSandbox) -> str:
    """
    Detect file encoding with fallback to utf-8.

    Args:
        file_path: Path to the file
        sandbox: Sandbox adaptor instance

    Returns:
        Detected encoding name
    """
    try:
        chardet = importlib.import_module("chardet")
        result = sandbox.read_file(file_path, binary=True)
        if result.status == SandboxStatus.SUCCESS and result.content:
            if isinstance(result.content, bytes):
                raw_data = result.content[:10000]
            elif isinstance(result.content, str):
                raw_data = result.content.encode()[:10000]
            else:
                return "utf-8"
            detection = chardet.detect(raw_data)
            encoding = detection["encoding"]
            if encoding and detection["confidence"] > 0.7:
                return encoding
    except ModuleNotFoundError:
        # chardet not available, use fallback
        pass
    except Exception as e:
        logger.warning(f"Error detecting encoding for {file_path}: {e}")

    # Fallback to utf-8
    return "utf-8"


def find_similar_file(file_path: str, sandbox: BaseSandbox) -> str | None:
    """
    Find a similar file with different extension if the original doesn't exist.

    Args:
        file_path: Path to the file that doesn't exist
        sandbox: Sandbox adaptor instance

    Returns:
        Path to similar file if found, None otherwise
    """
    try:
        base_path = Path(file_path)
        base_name = base_path.stem
        parent_dir = str(base_path.parent)

        # Check if parent directory exists
        if not sandbox.file_exists(parent_dir):
            return None

        # Look for files with same base name but different extensions
        try:
            pattern = f"{base_name}.*"
            matches = sandbox.glob(f"{parent_dir}/{pattern}", recursive=False)
            for match in matches:
                if match != file_path:
                    return match
        except Exception:
            pass

        return None
    except Exception as e:
        logger.warning(f"Error finding similar file for {file_path}: {e}")
        return None


def is_image_file(file_path: str) -> bool:
    """Check if file is an image based on extension."""
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


def is_text_file(file_path: str) -> bool:
    """Check if file is a text file based on extension."""
    ext = Path(file_path).suffix.lower()
    return ext in TEXT_EXTENSIONS or ext == "" or ext == ".txt"


def get_file_language(file_path: str) -> str:
    """Get the programming language based on file extension for syntax highlighting."""
    ext = Path(file_path).suffix.lower()

    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sql": "sql",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".fish": "fish",
        ".ps1": "powershell",
        ".bat": "batch",
        ".cmd": "batch",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".java": "java",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".clj": "clojure",
        ".hs": "haskell",
        ".ml": "ocaml",
        ".fs": "fsharp",
        ".r": "r",
        ".m": "objective-c",
        ".mm": "objective-c",
        ".pl": "perl",
        ".pm": "perl",
        ".lua": "lua",
        ".vim": "vim",
        ".el": "elisp",
        ".lisp": "lisp",
        ".scm": "scheme",
        ".rkt": "racket",
        ".jl": "julia",
        ".nim": "nim",
        ".cr": "crystal",
        ".d": "d",
        ".zig": "zig",
        ".v": "v",
        ".dart": "dart",
        ".ex": "elixir",
        ".exs": "elixir",
        ".erl": "erlang",
        ".hrl": "erlang",
    }

    return language_map.get(ext, "text")


def read_text_content(
    file_path: str,
    sandbox: BaseSandbox,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[str, int, int]:
    """
    Read text content from file with offset and limit support.

    Args:
        file_path: Path to the file
        sandbox: Sandbox adaptor instance
        offset: Line number to start reading from (0-based)
        limit: Number of lines to read (None for all remaining lines)

    Returns:
        Tuple of (content, lines_read, total_lines)
    """
    encoding = detect_file_encoding(file_path, sandbox)

    try:
        result = sandbox.read_file(file_path, encoding=encoding, binary=False)
        if result.status != SandboxStatus.SUCCESS:
            raise Exception(result.error or "Failed to read file")

        if result.content is None:
            raise Exception("File content is None")

        if isinstance(result.content, str):
            content = result.content
        elif isinstance(result.content, bytes):
            content = result.content.decode(encoding)
        else:
            raise Exception(f"Unexpected content type: {type(result.content)}")

        lines = content.splitlines(keepends=True)

        total_lines = len(lines)

        # Apply offset
        if offset > 0:
            lines = lines[offset:]

        # Apply limit
        if limit is not None:
            lines = lines[:limit]

        lines_read = len(lines)

        # Truncate long lines
        truncated_lines: list[str] = []
        for line in lines:
            if len(line) > MAX_LINE_LENGTH:
                truncated_lines.append(line[:MAX_LINE_LENGTH] + "...\n")
            else:
                truncated_lines.append(line)

        content = "".join(truncated_lines)

        return content, lines_read, total_lines

    except UnicodeDecodeError as e:
        logger.error(f"Unicode decode error reading {file_path}: {e}")
        # Try with latin-1 as fallback
        try:
            result = sandbox.read_file(file_path, encoding="latin-1", binary=False)
            if result.status != SandboxStatus.SUCCESS:
                raise Exception(result.error or "Failed to read file")

            if result.content is None:
                raise Exception("File content is None")

            if isinstance(result.content, str):
                content = result.content
            elif isinstance(result.content, bytes):
                content = result.content.decode("latin-1")
            else:
                raise Exception(f"Unexpected content type: {type(result.content)}")

            lines = content.splitlines(keepends=True)

            total_lines = len(lines)
            if offset > 0:
                lines = lines[offset:]
            if limit is not None:
                lines = lines[:limit]

            content = "".join(lines[:limit] if limit else lines)
            return content, len(lines), total_lines
        except Exception as fallback_e:
            raise Exception(
                f"Could not read file with any encoding: {e}, {fallback_e}",
            )


def read_image_file(file_path: str, sandbox: BaseSandbox) -> dict[str, Any]:
    """
    Read and encode image file to base64.

    Args:
        file_path: Path to the image file
        sandbox: Sandbox adaptor instance

    Returns:
        Dictionary with image data
    """
    try:
        file_info = sandbox.get_file_info(file_path)
        file_size = file_info.size

        # Check file size limit
        if file_size > MAX_IMAGE_SIZE:
            # Try to compress with PIL if available
            try:
                import io as io_module

                from PIL import Image

                # Read image data from sandbox
                result = sandbox.read_file(file_path, binary=True)
                if result.status != SandboxStatus.SUCCESS:
                    return {"error": result.error or "Failed to read image file"}

                if result.content is None:
                    return {"error": "Image content is None"}

                if isinstance(result.content, bytes):
                    image_bytes = result.content
                elif isinstance(result.content, str):
                    image_bytes = result.content.encode()
                else:
                    return {"error": f"Unexpected content type: {type(result.content)}"}

                img_io = io_module.BytesIO(image_bytes)

                with Image.open(img_io) as img:
                    # Calculate new dimensions maintaining aspect ratio
                    max_dimension = 2000
                    width, height = img.size

                    if width > max_dimension or height > max_dimension:
                        if width > height:
                            new_width = max_dimension
                            new_height = int((height * max_dimension) / width)
                        else:
                            new_height = max_dimension
                            new_width = int((width * max_dimension) / height)

                        img = img.resize(
                            (new_width, new_height),
                            Image.Resampling.LANCZOS,
                        )

                    # Convert to RGB if necessary and save as JPEG
                    if img.mode in ("RGBA", "LA", "P"):
                        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        rgb_img.paste(
                            img,
                            mask=(
                                img.split()[-1]
                                if img.mode
                                in (
                                    "RGBA",
                                    "LA",
                                )
                                else None
                            ),
                        )
                        img = rgb_img

                    # Save to bytes
                    img_bytes = io_module.BytesIO()
                    img.save(
                        img_bytes,
                        format="JPEG",
                        quality=80,
                        optimize=True,
                    )
                    img_data = img_bytes.getvalue()

                    return {
                        "type": "image",
                        "base64": base64.b64encode(img_data).decode("utf-8"),
                        "media_type": "image/jpeg",
                        "original_size": file_size,
                        "compressed_size": len(img_data),
                        "compressed": True,
                    }

            except ImportError:
                return {
                    "error": (f"Image file too large ({file_size} bytes > {MAX_IMAGE_SIZE} bytes) and PIL not available for compression"),
                }
            except Exception as e:
                logger.warning(f"Failed to compress image {file_path}: {e}")
                return {
                    "error": (f"Image file too large ({file_size} bytes > {MAX_IMAGE_SIZE} bytes) and compression failed: {str(e)}"),
                }

        # Read original file from sandbox
        result = sandbox.read_file(file_path, binary=True)
        if result.status != SandboxStatus.SUCCESS:
            return {"error": result.error or "Failed to read image file"}

        if result.content is None:
            return {"error": "Image content is None"}

        if isinstance(result.content, bytes):
            image_data = result.content
        elif isinstance(result.content, str):
            image_data = result.content.encode()
        else:
            return {"error": f"Unexpected content type: {type(result.content)}"}

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type or not mime_type.startswith("image/"):
            ext = Path(file_path).suffix.lower()
            mime_type_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".webp": "image/webp",
                ".tiff": "image/tiff",
                ".tif": "image/tiff",
            }
            mime_type = mime_type_map.get(ext, "image/jpeg")

        return {
            "type": "image",
            "image_url": f"data:{mime_type};base64,{base64.b64encode(image_data).decode('utf-8')}",
            "detail": "auto",
        }

    except Exception as e:
        logger.error(f"Error reading image file {file_path}: {e}")
        return {"error": f"Failed to read image file: {str(e)}"}


def add_line_numbers(content: str, start_line: int = 1) -> str:
    """
    Add line numbers to content.

    Args:
        content: Text content
        start_line: Starting line number

    Returns:
        Content with line numbers
    """
    if not content:
        return content

    lines = content.splitlines()
    max_line_num = start_line + len(lines) - 1
    width = len(str(max_line_num))

    numbered_lines: list[str] = []
    for i, line in enumerate(lines):
        line_num = start_line + i
        numbered_lines.append(f"{line_num:>{width}}: {line}")

    return "\n".join(numbered_lines)


def file_read_tool(
    file_path: str,
    offset: int | float | None = None,
    limit: int | float | None = None,
    sandbox: BaseSandbox | None = None,
) -> str | dict[str, Any]:
    """
    Read a file from the local filesystem. Supports both text and image files.

    Features:
    - Reads text files with optional offset and limit for large files
    - Supports image files (PNG, JPEG, GIF, BMP, WEBP, TIFF) with base64 encoding
    - Automatic file encoding detection for text files
    - File size validation and intelligent error messages
    - Line number support for text files
    - Image compression for large images
    - Smart file suggestions for missing files

    The file_path parameter must be an absolute path, not a relative path.
    By default, it reads up to 2000 lines starting from the beginning of the file.
    You can optionally specify a line offset and limit (especially handy for long files),
    but it's recommended to read the whole file by not providing these parameters.

    Any lines longer than 2000 characters will be truncated.
    For image files, the tool will return base64 encoded image data.
    For Jupyter notebooks (.ipynb files), consider using a specialized notebook tool instead.

    Examples:
    - Read entire file: file_path="/path/to/file.py"
    - Read with offset: file_path="/path/to/file.py", offset=100, limit=50
    - Read image: file_path="/path/to/image.png"
    """
    start_time = time.time()

    try:
        # Get sandbox instance
        sandbox = sandbox or LocalSandbox(_work_dir=os.getcwd())

        # Normalize file path
        file_path = os.path.abspath(file_path)

        # Check if file exists
        if not sandbox.file_exists(file_path):
            similar_file = find_similar_file(file_path, sandbox)
            error_msg = f"File does not exist: {file_path}"
            if similar_file:
                error_msg += f"\nDid you mean: {similar_file}?"

            return json.dumps(
                {
                    "error": error_msg,
                    "file_path": file_path,
                    "exists": False,
                    "similar_file": similar_file,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Get file info
        file_info = sandbox.get_file_info(file_path)

        # Check if it's a directory
        if file_info.is_directory:
            return json.dumps(
                {
                    "error": f"Path is a directory, not a file: {file_path}",
                    "file_path": file_path,
                    "is_directory": True,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Check read permissions
        if not file_info.readable:
            return json.dumps(
                {
                    "error": f"No read permission for file: {file_path}",
                    "file_path": file_path,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Get file size
        file_size = file_info.size

        # Handle image files
        if is_image_file(file_path):
            image_result = read_image_file(file_path, sandbox)

            if "error" in image_result:
                return json.dumps(
                    {
                        "error": image_result["error"],
                        "file_path": file_path,
                        "file_size": file_size,
                        "file_type": "image",
                        "duration_ms": int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            return image_result

        # Handle text files - check size first
        if file_size > MAX_OUTPUT_SIZE and offset is None and limit is None:
            size_kb = round(file_size / 1024)
            max_size_kb = round(MAX_OUTPUT_SIZE / 1024)

            return json.dumps(
                {
                    "error": (
                        f"File content ({size_kb}KB) exceeds maximum allowed size ({max_size_kb}KB). "
                        "Please use offset and limit parameters to read specific portions of the file, "
                        "or use the grep tool to search for specific content."
                    ),
                    "file_path": file_path,
                    "file_size": file_size,
                    "file_type": "text",
                    "max_size": MAX_OUTPUT_SIZE,
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

        # Read text content
        try:
            # Convert 1-based offset to 0-based for internal use, ensuring integers
            internal_offset = (int(offset) - 1) if offset and offset > 0 else 0
            int_limit = int(limit) if limit is not None else None

            content, lines_read, total_lines = read_text_content(
                file_path,
                sandbox,
                internal_offset,
                int_limit,
            )

            # Check content size after reading
            if len(content) > MAX_OUTPUT_SIZE:
                size_kb = round(len(content) / 1024)
                max_size_kb = round(MAX_OUTPUT_SIZE / 1024)

                return json.dumps(
                    {
                        "error": (
                            f"File content ({size_kb}KB) exceeds maximum allowed size ({max_size_kb}KB) after reading. "
                            "Please use smaller offset and limit parameters."
                        ),
                        "file_path": file_path,
                        "file_size": file_size,
                        "content_size": len(content),
                        "file_type": "text",
                        "duration_ms": int((time.time() - start_time) * 1000),
                    },
                    indent=2,
                    ensure_ascii=False,
                )

            # Add line numbers
            start_line_num = int(offset) if offset else 1
            content_with_lines = add_line_numbers(content, start_line_num)

            # Update timestamp cache for file_write_tool compatibility
            update_file_timestamp(file_path, sandbox)

            duration_ms = int((time.time() - start_time) * 1000)

            # Determine if content was truncated
            truncated = (offset is not None and offset > 1) or (limit is not None and lines_read >= limit)

            return json.dumps(
                {
                    "type": "text",
                    "success": True,
                    "file_path": file_path,
                    "file_size": file_size,
                    "content": content_with_lines,
                    "lines_read": lines_read,
                    "total_lines": total_lines,
                    "start_line": start_line_num,
                    "end_line": (start_line_num + lines_read - 1 if lines_read > 0 else start_line_num),
                    "truncated": truncated,
                    "language": get_file_language(file_path),
                    "encoding": detect_file_encoding(file_path, sandbox),
                    "duration_ms": duration_ms,
                },
                indent=2,
                ensure_ascii=False,
            )

        except Exception as e:
            logger.error(f"Error reading text file {file_path}: {e}")
            return json.dumps(
                {
                    "error": f"Failed to read text file: {str(e)}",
                    "file_path": file_path,
                    "file_size": file_size,
                    "file_type": "text",
                    "duration_ms": int((time.time() - start_time) * 1000),
                },
                indent=2,
                ensure_ascii=False,
            )

    except Exception as e:
        logger.error(f"Unexpected error in file_read_tool: {e}")
        return json.dumps(
            {
                "error": f"Unexpected error: {str(e)}",
                "file_path": file_path,
                "duration_ms": int((time.time() - start_time) * 1000),
            },
            indent=2,
            ensure_ascii=False,
        )


# Usage example (for testing)
def main():
    result = file_read_tool(
        file_path="./nexau/archs/tool/builtin/file_tools/file_read_tool.py",
        offset=20,
        limit=50,
    )
    print(result)


if __name__ == "__main__":
    main()
