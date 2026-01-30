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
File Format Validators Module

Provides content-based format validation for structured data files.

Unlike file_type_utils (extension-based detection), this module validates
actual file content to ensure it matches the expected format.

Common use case: Prevent AI from writing incorrect formats
(e.g., Markdown tables instead of CSV).

Architecture:
- file_type_utils.py: Static type detection (extension → binary/text)
- file_format_validators.py: Dynamic format validation (content → valid/invalid)
- file_state.py: State management (timestamps, read/write coordination)
"""

import csv
import io


def is_markdown_table(content: str) -> bool:
    """
    Detect if content is a Markdown table.

    Markdown table characteristics:
    - Has | separators
    - Has --- alignment line (typically second line)

    Args:
        content: File content to check

    Returns:
        True if content appears to be a Markdown table

    Example:
        >>> is_markdown_table("| name | age |\\n|------|-----|\\n| Sum | 25 |")
        True
        >>> is_markdown_table("name,age\\nSum,25")
        False
    """
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return False

    # Check first 3 lines for Markdown table characteristics
    first_lines = lines[: min(3, len(lines))]

    # Characteristic 1: Has | separators
    has_pipe = any("|" in line for line in first_lines)

    # Characteristic 2: Has --- alignment line
    # (line contains only |, -, and spaces)
    has_separator = any(
        line.strip() and set(line.strip().replace("|", "").replace(" ", "").replace("-", "")) == set() for line in first_lines
    )

    return has_pipe and has_separator


def validate_csv_format(content: str) -> tuple[bool, str]:
    """
    Validate if content is valid CSV format.

    This function checks:
    1. Not a Markdown table (common AI mistake)
    2. Can be parsed by Python's csv module
    3. Not empty

    Args:
        content: File content to validate

    Returns:
        (is_valid, error_message)
        - is_valid: True if content is valid CSV
        - error_message: Empty string if valid, error description if invalid

    Example:
        >>> validate_csv_format("name,age\\nSum,25")
        (True, "")
        >>> validate_csv_format("| name | age |\\n|------|-----|")
        (False, "Detected Markdown table format...")
    """
    # Check 1: Detect Markdown table format
    if is_markdown_table(content):
        return (
            False,
            "Detected Markdown table format. Use pandas.to_csv() or csv module instead.",
        )

    # Check 2: Try to parse with csv.Sniffer
    try:
        sniffer = csv.Sniffer()
        sample = content[:1024]  # Use first 1KB for detection

        # Detect CSV dialect
        dialect = sniffer.sniff(sample)

        # Try to parse entire content
        reader = csv.reader(io.StringIO(content), dialect=dialect)
        rows = list(reader)

        # Check 3: Not empty
        if len(rows) == 0:
            return False, "Empty CSV file"

        return True, ""

    except csv.Error as e:
        return False, f"Invalid CSV format: {str(e)}"
    except Exception as e:
        return False, f"Cannot parse as CSV: {str(e)}"
