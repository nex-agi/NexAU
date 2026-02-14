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

"""Unit tests for write_todos builtin tool."""

import pytest

from nexau.archs.tool.builtin.session_tools import write_todos


class TestWriteTodos:
    """Test write_todos tool functionality."""

    def test_update_todo_list_successfully(self):
        """Should update todo list successfully."""
        todos = [
            {"description": "Task 1", "status": "pending"},
            {"description": "Task 2", "status": "in_progress"},
            {"description": "Task 3", "status": "completed"},
        ]
        result = write_todos(todos=todos)
        assert "Successfully updated the todo list" in result["content"]
        assert "1. [pending] Task 1" in result["content"]
        assert "2. [in_progress] Task 2" in result["content"]
        assert "3. [completed] Task 3" in result["content"]

    def test_clear_todo_list(self):
        """Should clear todo list when empty array is provided."""
        result = write_todos(todos=[])
        assert "Successfully cleared the todo list" in result["content"]

    def test_error_when_todos_not_array(self):
        """Should return error when todos is not an array."""
        result = write_todos(todos="not an array")  # type: ignore[arg-type]
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "`todos` parameter must be an array" in result["content"]

    def test_error_when_todo_item_not_object(self):
        """Should return error when todo item is not an object."""
        result = write_todos(todos=["not an object"])  # type: ignore[arg-type]
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_error_when_description_missing(self):
        """Should return error when description is missing."""
        todos = [{"status": "pending"}]
        result = write_todos(todos=todos)  # type: ignore[typeddict-item]
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "non-empty description" in result["content"]

    def test_error_when_description_empty(self):
        """Should return error when description is empty."""
        todos = [{"description": "", "status": "pending"}]
        result = write_todos(todos=todos)
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_error_when_status_invalid(self):
        """Should return error when status is invalid."""
        todos = [{"description": "Task", "status": "invalid_status"}]
        result = write_todos(todos=todos)
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "valid status" in result["content"]

    def test_error_when_multiple_in_progress(self):
        """Should return error when multiple tasks are in_progress."""
        todos = [
            {"description": "Task 1", "status": "in_progress"},
            {"description": "Task 2", "status": "in_progress"},
        ]
        result = write_todos(todos=todos)
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert 'Only one task can be "in_progress"' in result["content"]

    @pytest.mark.parametrize("status", ["pending", "in_progress", "completed", "cancelled"])
    def test_valid_status(self, status):
        """Should accept all valid statuses."""
        todos = [{"description": "Task", "status": status}]
        result = write_todos(todos=todos)
        assert result.get("error") is None
        assert f"[{status}]" in result["content"]
