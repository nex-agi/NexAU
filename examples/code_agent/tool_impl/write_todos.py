# SPDX-License-Identifier: Apache-2.0
"""
write_todos tool - Manages a list of subtasks for complex queries.

Based on gemini-cli's write-todos.ts implementation.
Helps track progress and organize complex multi-step tasks.
"""

from typing import Any

# Valid todo statuses
TODO_STATUSES = ["pending", "in_progress", "completed", "cancelled"]


def write_todos(
    todos: list[dict[str, str]],
) -> dict[str, Any]:
    """
    Lists out subtasks required to complete a user request.

    This tool helps track progress, organize complex queries, and ensure
    no steps are missed. The user can see current progress being made.

    Task state definitions:
    - pending: Work has not begun on a given subtask.
    - in_progress: Marked just prior to beginning work. Only one subtask
      should be in_progress at a time.
    - completed: Subtask was successfully completed with no errors.
    - cancelled: Subtask is no longer needed due to dynamic nature of task.

    Args:
        todos: The complete list of todo items. Each item should have:
               - description: The description of the task
               - status: The current status (pending/in_progress/completed/cancelled)

    Returns:
        Dict with llmContent and returnDisplay matching gemini-cli format
    """
    try:
        # Validate todos parameter
        if not isinstance(todos, list):
            return {
                "llmContent": "`todos` parameter must be an array",
                "returnDisplay": "Error: Invalid todos parameter.",
                "error": {
                    "message": "`todos` parameter must be an array",
                    "type": "INVALID_PARAMETER",
                },
            }

        # Validate each todo item
        for i, todo in enumerate(todos):
            if not isinstance(todo, dict):
                return {
                    "llmContent": "Each todo item must be an object",
                    "returnDisplay": "Error: Invalid todo item.",
                    "error": {
                        "message": "Each todo item must be an object",
                        "type": "INVALID_PARAMETER",
                    },
                }

            description = todo.get("description", "")
            if not isinstance(description, str) or not description.strip():
                return {
                    "llmContent": "Each todo must have a non-empty description string",
                    "returnDisplay": "Error: Missing or empty description.",
                    "error": {
                        "message": "Each todo must have a non-empty description string",
                        "type": "INVALID_PARAMETER",
                    },
                }

            status = todo.get("status", "")
            if status not in TODO_STATUSES:
                return {
                    "llmContent": f"Each todo must have a valid status ({', '.join(TODO_STATUSES)})",
                    "returnDisplay": "Error: Invalid status.",
                    "error": {
                        "message": f"Each todo must have a valid status ({', '.join(TODO_STATUSES)})",
                        "type": "INVALID_PARAMETER",
                    },
                }

        # Check that only one task is in_progress
        in_progress_count = sum(1 for todo in todos if todo.get("status") == "in_progress")

        if in_progress_count > 1:
            return {
                "llmContent": 'Invalid parameters: Only one task can be "in_progress" at a time.',
                "returnDisplay": "Error: Multiple in_progress tasks.",
                "error": {
                    "message": 'Only one task can be "in_progress" at a time.',
                    "type": "INVALID_PARAMETER",
                },
            }

        # Build the todo list string
        if todos:
            todo_list_string = "\n".join(f"{i + 1}. [{todo['status']}] {todo['description']}" for i, todo in enumerate(todos))
            llm_content = f"Successfully updated the todo list. The current list is now:\n{todo_list_string}"
        else:
            llm_content = "Successfully cleared the todo list."

        return {
            "llmContent": llm_content,
            "returnDisplay": {"todos": todos},
        }

    except Exception as e:
        error_msg = f"Error updating todos: {str(e)}"
        return {
            "llmContent": error_msg,
            "returnDisplay": error_msg,
            "error": {
                "message": error_msg,
                "type": "WRITE_TODOS_ERROR",
            },
        }
