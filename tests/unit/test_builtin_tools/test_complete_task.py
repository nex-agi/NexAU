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

"""Unit tests for complete_task builtin tool."""

import json

from nexau.archs.tool.builtin.session_tools import complete_task


class TestCompleteTask:
    """Test complete_task tool functionality."""

    def test_complete_with_result(self):
        """Should return success without echoing model-authored result payload."""
        result = complete_task(result="Task completed successfully")
        data = json.loads(result)
        assert data["success"] is True
        assert data["task_completed"] is True
        assert data["status"] == "TASK_COMPLETED"
        assert "result" not in data

    def test_complete_with_empty_result_returns_error(self):
        """Should return error when result is empty string."""
        result = complete_task(result="")
        data = json.loads(result)
        assert data["success"] is False
        assert data["task_completed"] is False
        assert "EMPTY_RESULT" in data.get("type", "")

    def test_complete_with_whitespace_result_returns_error(self):
        """Should return error when result is only whitespace."""
        result = complete_task(result="   \n\t  ")
        data = json.loads(result)
        assert data["success"] is False
        assert data["task_completed"] is False

    def test_complete_with_none_result_returns_error(self):
        """Should return error when result is None and no kwargs."""
        result = complete_task()
        data = json.loads(result)
        assert data["success"] is False
        assert data["task_completed"] is False
        assert "MISSING_RESULT" in data.get("type", "")

    def test_complete_with_result_and_kwargs(self):
        """Should succeed without echoing model-authored output fields."""
        result = complete_task(result="Main result", extra="field")
        data = json.loads(result)
        assert data["success"] is True
        assert data["task_completed"] is True
        assert data["status"] == "TASK_COMPLETED"
        assert "result" not in data
        assert "extra" not in data
