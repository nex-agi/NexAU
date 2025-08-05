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

"""TodoWrite tool implementation for task management in agent context."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ...main_sub.agent_state import AgentState


def todo_write(
    todos: list[dict[str, str]],
    agent_state: Optional["AgentState"] = None,
) -> dict[str, Any]:
    """
    Create and manage a structured task list for the current coding session.

    This tool helps track progress, organize complex tasks, and demonstrate
    thoroughness to the user. It stores the todo list in the agent's context
    so it persists across tool calls and can be rendered in the system prompt.

    Args:
        todos: List of todo items, each containing:
            - content: The task description (required)
            - status: One of "pending", "in_progress", "completed" (required)
            - priority: One of "high", "medium", "low" (optional, defaults to "medium")
            - id: Unique identifier (required)

    Returns:
        Dict containing the result of the operation
    """
    try:
        if not agent_state:
            return {
                "status": "error",
                "error": "Agent state not available",
            }

        # Validate todo items
        validated_todos = []
        for i, todo in enumerate(todos):
            # Validate required fields
            if not isinstance(todo, dict):
                return {
                    "status": "error",
                    "error": f"Todo item {i} must be a dictionary",
                }

            if "content" not in todo or not todo["content"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} missing required 'content' field",
                }

            status = todo.get("status", "pending")

            if status not in ["pending", "in_progress", "completed"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} has invalid status '{status}'. Must be 'pending', 'in_progress', or 'completed'",
                }

            if "id" not in todo or not todo["id"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} missing required 'id' field",
                }

            # Set default priority if not provided
            priority = todo.get("priority", "medium")
            if priority not in ["high", "medium", "low"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} has invalid priority '{priority}'. Must be 'high', 'medium', or 'low'",
                }

            validated_todo = {
                "content": todo["content"],
                "status": status,
                "priority": priority,
                "id": todo["id"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            validated_todos.append(validated_todo)

        # Check for duplicate IDs
        todo_ids = [todo["id"] for todo in validated_todos]
        if len(todo_ids) != len(set(todo_ids)):
            return {
                "status": "error",
                "error": "Duplicate todo IDs found. Each todo must have a unique ID",
            }

        # Count status types for validation
        in_progress_count = sum(1 for todo in validated_todos if todo["status"] == "in_progress")
        # if in_progress_count > 1:
        #     return {
        #         "status": "error",
        #         "error": f"Only one todo can be 'in_progress' at a time. Found {in_progress_count} in_progress todos"
        #     }

        # Store the todo list in agent context
        agent_state.set_global_value("current_todos", validated_todos)
        agent_state.set_global_value(
            "todos_last_updated",
            datetime.now().isoformat(),
        )

        # Generate summary for display
        total_todos = len(validated_todos)
        pending_count = sum(1 for todo in validated_todos if todo["status"] == "pending")
        completed_count = sum(1 for todo in validated_todos if todo["status"] == "completed")

        return {
            "status": "success",
            "message": f"Todo list updated with {total_todos} items",
            "summary": {
                "total": total_todos,
                "pending": pending_count,
                "in_progress": in_progress_count,
                "completed": completed_count,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }
