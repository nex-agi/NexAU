
### Context and Templating

Agents support Jinja2 templating in system prompts:

```python
system_prompt = """
Date: {{date}}
User: {{username}}
Task: {{task_description}}

You are an assistant with access to the following context:
{% for key, value in env_content.items() %}
- {{key}}: {{value}}
{% endfor %}
"""

response = agent.run(
    "Your message here",
    context={
        "date": "2024-01-01",
        "username": "user",
        "task_description": "Complete the project",
        "env_content": {
            "working_dir": "/path/to/project",
            "python_version": "3.12"
        }
    }
)
```

#### Using Full Context Object

```python
from nexau.archs.main_sub.agent_context import get_context

def context_aware_tool(action: str) -> dict:
    """Tool that uses the full context object for advanced operations."""

    # Get the full context object
    ctx = get_context()
    if ctx is None:
        return {"error": "No agent context available"}

    # Check if context was recently modified
    if ctx.is_modified():
        # Context was changed by another tool
        pass

    # Add a callback for when context changes
    def on_context_change():
        print("Context was modified!")

    ctx.add_modification_callback(on_context_change)

    return {"result": f"Executed {action} with full context awareness"}
```
