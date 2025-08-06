"""TodoWrite tool implementation for task management in agent context."""

from typing import Dict, Any, List
import json
import uuid
from datetime import datetime


def todo_write(todos: List[Dict[str, str]]) -> Dict[str, Any]:
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
        from ...main_sub.agent_context import get_context, set_state_value, get_state_value
        
        context = get_context()
        if not context:
            return {
                "status": "error",
                "error": "No agent context available"
            }
        
        # Validate todo items
        validated_todos = []
        for i, todo in enumerate(todos):
            # Validate required fields
            if not isinstance(todo, dict):
                return {
                    "status": "error",
                    "error": f"Todo item {i} must be a dictionary"
                }
            
            if "content" not in todo or not todo["content"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} missing required 'content' field"
                }
            
            if "status" not in todo:
                return {
                    "status": "error", 
                    "error": f"Todo item {i} missing required 'status' field"
                }
            
            if todo["status"] not in ["pending", "in_progress", "completed"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} has invalid status '{todo['status']}'. Must be 'pending', 'in_progress', or 'completed'"
                }
            
            if "id" not in todo or not todo["id"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} missing required 'id' field"
                }
            
            # Set default priority if not provided
            priority = todo.get("priority", "medium")
            if priority not in ["high", "medium", "low"]:
                return {
                    "status": "error",
                    "error": f"Todo item {i} has invalid priority '{priority}'. Must be 'high', 'medium', or 'low'"
                }
            
            validated_todo = {
                "content": todo["content"],
                "status": todo["status"],
                "priority": priority,
                "id": todo["id"],
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            validated_todos.append(validated_todo)
        
        # Check for duplicate IDs
        todo_ids = [todo["id"] for todo in validated_todos]
        if len(todo_ids) != len(set(todo_ids)):
            return {
                "status": "error",
                "error": "Duplicate todo IDs found. Each todo must have a unique ID"
            }
        
        # Count status types for validation
        in_progress_count = sum(1 for todo in validated_todos if todo["status"] == "in_progress")
        if in_progress_count > 1:
            return {
                "status": "error",
                "error": f"Only one todo can be 'in_progress' at a time. Found {in_progress_count} in_progress todos"
            }
        
        # Store the todo list in agent context
        set_state_value("current_todos", validated_todos)
        set_state_value("todos_last_updated", datetime.now().isoformat())
        
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
                "completed": completed_count
            },
            "todos": validated_todos
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def get_current_todos() -> Dict[str, Any]:
    """
    Get the current todo list from agent context.
    
    Returns:
        Dict containing the current todo list or error
    """
    try:
        from ...main_sub.agent_context import get_context, get_state_value
        
        context = get_context()
        if not context:
            return {
                "status": "error",
                "error": "No agent context available"
            }
        
        todos = get_state_value("current_todos", [])
        last_updated = get_state_value("todos_last_updated", "Never")
        
        if not todos:
            return {
                "status": "success",
                "message": "No todos found",
                "todos": [],
                "summary": {
                    "total": 0,
                    "pending": 0,
                    "in_progress": 0,
                    "completed": 0
                },
                "last_updated": last_updated
            }
        
        # Generate summary
        total_todos = len(todos)
        pending_count = sum(1 for todo in todos if todo["status"] == "pending")
        in_progress_count = sum(1 for todo in todos if todo["status"] == "in_progress")
        completed_count = sum(1 for todo in todos if todo["status"] == "completed")
        
        return {
            "status": "success",
            "todos": todos,
            "summary": {
                "total": total_todos,
                "pending": pending_count,
                "in_progress": in_progress_count,
                "completed": completed_count
            },
            "last_updated": last_updated
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


def clear_todos() -> Dict[str, Any]:
    """
    Clear all todos from the agent context.
    
    Returns:
        Dict containing the result of the operation
    """
    try:
        from ...main_sub.agent_context import get_context, set_state_value
        
        context = get_context()
        if not context:
            return {
                "status": "error",
                "error": "No agent context available"
            }
        
        # Clear the todos
        set_state_value("current_todos", [])
        set_state_value("todos_last_updated", datetime.now().isoformat())
        
        return {
            "status": "success",
            "message": "All todos cleared",
            "todos": []
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }