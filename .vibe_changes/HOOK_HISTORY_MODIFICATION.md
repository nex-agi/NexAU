# After Model Hook History Modification

## Overview

The `after_model_hooks` feature has been enhanced to allow hooks to modify not only the parsed response (tool/agent calls) but also the conversation history messages. This provides much more powerful capabilities for implementing custom logic, safety measures, and conversation management.

## Changes Made

### 1. Updated Hook Protocol

The `AfterModelHook` protocol signature has been changed from:

```python
def __call__(self, original_response: str, parsed_response: ParsedResponse) -> ParsedResponse | None:
```

To:

```python
def __call__(self, hook_input: AfterModelHookInput) -> HookResult:
```

**Key changes:**
- Single `hook_input` parameter of type `AfterModelHookInput` containing all data
- Return type is now a typed `HookResult` class for better type safety
- `HookResult` encapsulates both parsed response and message modifications
- Provides convenient class methods for creating results
- Input is structured and type-safe

### 2. Updated Hook Manager

The `HookManager.execute_hooks()` method now:
- Accepts and passes the conversation history to each hook
- Handles modified messages returned by hooks
- Returns both the final parsed response and final messages

### 3. Updated Execution Flow

The execution flow has been updated throughout the system:
- **ResponseGenerator**: Passes messages to the XML processor and handles returned modified messages
- **Executor**: Passes messages to hooks and returns modified messages along with other results
- **Agent**: The conversation history can now be modified by hooks during execution

## AfterModelHookInput Class

The new `AfterModelHookInput` class encapsulates all input data passed to hooks:

```python
@dataclass
class AfterModelHookInput:
    max_iterations: int                    # The max iterations
    current_iteration: int                 # The current iteration ID
    original_response: str                  # The raw response from the LLM
    parsed_response: ParsedResponse         # The parsed structure containing tool/agent calls
    messages: list[dict[str, str]]         # The current conversation history
```

## HookResult Class

The new `HookResult` class provides a type-safe way to return modifications from hooks. It uses a single `with_modifications` method that accepts optional parameters for maximum flexibility:

```python
@dataclass
class HookResult:
    parsed_response: ParsedResponse | None = None
    messages: list[dict[str, str]] | None = None

    def has_modifications(self) -> bool: ...

    @classmethod
    def no_changes(cls) -> 'HookResult': ...

    @classmethod
    def with_modifications(cls, parsed_response: ParsedResponse | None = None,
                          messages: list[dict[str, str]] | None = None) -> 'HookResult': ...
```

### Convenient Usage Patterns

```python
# No modifications
return HookResult.no_changes()

# Only modify parsed response
return HookResult.with_modifications(parsed_response=modified_parsed)

# Only modify messages
return HookResult.with_modifications(messages=modified_messages)

# Modify both
return HookResult.with_modifications(parsed_response=modified_parsed, messages=modified_messages)

# Direct construction (also valid)
return HookResult(parsed_response=modified_parsed, messages=modified_messages)
```

## Usage Examples

### Basic Hook That Modifies History

```python
def create_context_hook() -> AfterModelHook:
    def context_hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.parsed_response.tool_calls:
            # Add context message before tool execution
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": f"About to execute {len(hook_input.parsed_response.tool_calls)} tool(s)"
            })
            return HookResult.with_modifications(messages=modified_messages)  # Only modify messages

        return HookResult.no_changes()  # No modifications

    return context_hook
```

### Hook That Modifies Both Response and History

```python
def create_combined_hook(allowed_tools: set[str]) -> AfterModelHook:
    def combined_hook(hook_input: AfterModelHookInput) -> HookResult:
        # Filter tool calls
        filtered_calls = [call for call in hook_input.parsed_response.tool_calls if call.tool_name in allowed_tools]

        if len(filtered_calls) != len(hook_input.parsed_response.tool_calls):
            # Create modified parsed response
            modified_parsed = ParsedResponse(
                original_response=hook_input.parsed_response.original_response,
                tool_calls=filtered_calls,
                # ... other fields
            )

            # Add context message about filtering
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": f"Filtered out {len(hook_input.parsed_response.tool_calls) - len(filtered_calls)} disallowed tools"
            })

            return HookResult.with_modifications(parsed_response=modified_parsed, messages=modified_messages)

        return HookResult.no_changes()

    return combined_hook
```

### Using Hooks with Agent

```python
from northau.archs.main_sub.agent import create_agent
from northau.archs.main_sub.execution.hooks import AfterModelHookInput, HookResult

# Create hooks
history_hook = create_context_hook()
filter_hook = create_combined_hook({'web_search', 'file_read'})

# Create agent with hooks
agent = create_agent(
    name="my_agent",
    tools=[...],
    after_model_hooks=[history_hook, filter_hook],
    # ... other parameters
)
```

## Use Cases

### 1. Conversation Management
- Add context messages based on conversation length
- Insert summaries or reminders
- Add timestamps or metadata

### 2. Safety and Compliance
- Filter out disallowed tools and add explanatory messages
- Add warnings before potentially dangerous operations
- Log security-relevant actions

### 3. Enhanced Context
- Add relevant information based on tool calls being made
- Insert domain-specific context
- Provide hints or suggestions

### 4. Debugging and Monitoring
- Log conversation patterns
- Add debug information
- Track tool usage statistics

## Migration Guide

If you have existing hooks, you need to update them to the new signature:

**Before:**
```python
def my_hook(original_response: str, parsed_response: ParsedResponse) -> ParsedResponse | None:
    # ... existing logic
    return modified_parsed_response  # or None
```

**After:**
```python
def my_hook(hook_input: AfterModelHookInput) -> HookResult:
    # Access the data through hook_input
    original_response = hook_input.original_response
    parsed_response = hook_input.parsed_response
    messages = hook_input.messages

    # ... existing logic (unchanged)
    if modified_parsed_response:
        return HookResult.with_modifications(parsed_response=modified_parsed_response)
    else:
        return HookResult.no_changes()
```

## Built-in Hook Factories

The following built-in hook factory functions have been updated:

- `create_logging_hook()`: Now logs message history information
- `create_filter_hook()`: Filters tools/agents (doesn't modify messages)
- `create_rate_limit_hook()`: Limits number of calls (doesn't modify messages)

All existing functionality is preserved while adding the new capabilities.

## Technical Details

### Hook Execution Order
1. Hooks are executed in the order they are provided
2. Each hook receives the current state (after previous hooks have run)
3. If a hook returns modified messages, subsequent hooks will receive those modified messages
4. The final messages are used for the rest of the execution

### Message Format
Messages are standard conversation history format:
```python
{
    "role": "system" | "user" | "assistant",
    "content": "message content"
}
```

### Performance Considerations
- Messages are copied when modified, so large conversation histories may impact performance
- Hooks should be efficient as they run on every LLM response
- Consider the cumulative effect of multiple hooks modifying messages

## Examples

See `example_hook_with_history.py` for complete working examples of hooks that demonstrate:
- Adding context messages based on tool calls
- Conversation length management
- Combined filtering and context addition
- Debug logging without modifications
