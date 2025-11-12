# Feature: Prettier Tool Display

## Overview

Enhanced the CLI to display tool calls in a beautiful, readable format instead of raw XML. Shows execution mode (parallel/sequential), tool names, and parameter previews.

## What Changed

### Before (Raw Display)
```
🔧 Planning to execute 2 tool(s): web_search, web_read
```

### After (Pretty Display)
```
🔧 Planning to execute 2 tool(s) [parallel]:
  1. web_search(query=NexAU framework, max_results=5)
  2. web_read(url=https://example.com)
```

## Features

### 1. **Execution Mode Indicator**
- Shows `[parallel]` when tools run concurrently
- Shows `[sequential]` when tools run one after another
- Helps understand agent's execution strategy

### 2. **Tool Parameter Preview**
- Displays first 3 parameters per tool
- Long values truncated to 100 chars
- Shows count if more parameters exist
- Format: `tool_name(param1=value1, param2=value2)`

### 3. **Structured Layout**
- Bold header line with tool count and execution mode
- Indented numbered list of tools
- Each tool on its own line for readability
- Cyan color for tool details

### 4. **Smart Truncation**
- Long parameter values: `"very long text..." (100 chars max)`
- Many parameters: `param1=val1, param2=val2, ... (+5 more)`
- Prevents UI clutter while showing useful info

## Implementation

### Backend (agent_runner.py)

```python
def create_cli_progress_hook():
    def progress_hook(hook_input: AfterModelHookInput):
        if hook_input.parsed_response.tool_calls:
            tool_count = len(hook_input.parsed_response.tool_calls)
            is_parallel = hook_input.parsed_response.is_parallel_tools
            
            # Format each tool with parameters
            tool_details = []
            for call in hook_input.parsed_response.tool_calls:
                params = format_parameters(call.tool_input)
                tool_details.append(f"{call.tool_name}{params}")
            
            # Send header
            execution_type = "parallel" if is_parallel else "sequential"
            send_message("step", 
                f"Planning to execute {tool_count} tool(s) [{execution_type}]:",
                metadata={"type": "tool_plan_header"})
            
            # Send each tool detail
            for i, detail in enumerate(tool_details, 1):
                send_message("step", 
                    f"  {i}. {detail}",
                    metadata={"type": "tool_detail"})
```

### Frontend (app.js)

```javascript
case 'tool_plan_header':
  icon = '🔧';
  color = 'blue';
  isBold = true;  // Make header bold
  break;

case 'tool_detail':
  icon = '';      // No icon, just indented text
  color = 'cyan';
  break;
```

## Visual Examples

### Single Tool (Sequential)
```
💭 I'll search for information about NexAU...
🔧 Planning to execute 1 tool(s) [sequential]:
  1. web_search(query=NexAU agent framework, max_results=10)
✓ Tool 'web_search' completed
```

### Multiple Tools (Parallel)
```
💭 I'll fetch and analyze the webpage...
🔧 Planning to execute 3 tool(s) [parallel]:
  1. web_read(url=https://docs.example.com)
  2. web_search(query=related documentation)
  3. file_read(file_path=config.yaml)
✓ Tool 'web_read' completed
✓ Tool 'web_search' completed
✓ Tool 'file_read' completed
```

### Long Parameters (Truncated)
```
🔧 Planning to execute 1 tool(s) [sequential]:
  1. file_write(file_path=app.py, contents=def main():
    print("Hello...")..., mode=w)
```

### Many Parameters
```
🔧 Planning to execute 1 tool(s) [sequential]:
  1. api_call(endpoint=/users, method=POST, headers={'Content-Type'...}, ... (+4 more))
```

## Benefits

### 1. **Transparency**
- See exactly what tools will be called
- Understand tool parameters before execution
- Know if tools run in parallel or sequence

### 2. **Debugging**
- Quickly identify incorrect parameters
- Spot tool selection issues
- Understand execution order

### 3. **Learning**
- See how the agent constructs tool calls
- Understand parameter usage patterns
- Learn tool capabilities

### 4. **Readability**
- Structured, indented format
- Color-coded for quick scanning
- No XML/JSON clutter

## Message Types

### New Message Types
- `tool_plan_header`: Bold header with count and execution mode
- `tool_detail`: Individual tool with parameters (indented, no icon)

### Existing Types (Still Used)
- `tool_executed`: When a tool completes
- `agent_thinking`: Agent's reasoning text

## Configuration

No configuration needed - automatically formats all tool calls for any agent.

## Technical Details

### Parameter Formatting
1. Extract `tool_input` dictionary from each tool call
2. Truncate values longer than 100 chars
3. Show first 3 parameters
4. Indicate if more parameters exist
5. Format as `name(key1=val1, key2=val2)`

### Execution Mode Detection
- Read `is_parallel_tools` from `ParsedResponse`
- Display as `[parallel]` or `[sequential]`
- Helps users understand performance implications

### Truncation Rules
- **Value length**: Max 100 chars, then `...`
- **Parameter count**: Show first 3, then `(+N more)`
- **Total line**: No hard limit, terminal handles wrapping

## Performance

- Minimal overhead: Just string formatting
- No additional API calls
- Same information, better presentation
- Truncation prevents memory issues with large params

## Compatibility

- Works with all NexAU tools
- Compatible with both single and parallel execution
- Handles missing parameters gracefully
- Supports all parameter types (str, int, dict, list, etc.)

## Future Enhancements

Potential improvements:
- [ ] Syntax highlighting for parameter values
- [ ] Expandable sections for truncated content
- [ ] Show estimated execution time
- [ ] Display tool descriptions/tooltips
- [ ] Color-code different parameter types
- [ ] Show parameter validation status

## Testing

```bash
cd cli
npm run build
./dist/cli.js ../examples/code_agent/cc_agent.yaml

# Try multi-tool commands
> Search for NexAU and read the first result

# Observe:
# - Tool count and execution mode
# - Each tool listed with parameters
# - Clean, readable format
# - No raw XML/JSON
```

---

**Version**: 0.1.0  
**Date**: November 2, 2025  
**Status**: ✅ Implemented

