# NexAU CLI Architecture

This document explains the technical architecture of the NexAU CLI.

## Overview

The NexAU CLI is a split architecture application with a Node.js frontend (for UI) and a Python backend (for agent execution). The two processes communicate via stdin/stdout using JSON messages.

```
┌─────────────────────────────────────────────────────────┐
│                    User Terminal                         │
└───────────────────────┬─────────────────────────────────┘
                        │
                        │ User Input / Visual Output
                        │
┌───────────────────────▼─────────────────────────────────┐
│              Node.js Process (Ink UI)                    │
│  ┌──────────────────────────────────────────────────┐  │
│  │  App Component (source/app.js)                    │  │
│  │  - Manages conversation state                     │  │
│  │  - Renders UI components                          │  │
│  │  - Handles user input                             │  │
│  │  - Spawns Python subprocess                       │  │
│  └──────────┬────────────────────────┬────────────────┘  │
└─────────────┼────────────────────────┼───────────────────┘
              │                        │
              │ JSON over stdin        │ JSON over stdout
              │                        │
┌─────────────▼────────────────────────▼───────────────────┐
│         Python Process (agent_runner.py)                 │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Agent Runner                                     │  │
│  │  - Loads agent from YAML                          │  │
│  │  - Processes user messages                        │  │
│  │  - Executes agent.run()                           │  │
│  │  - Returns responses as JSON                      │  │
│  └──────────┬───────────────────────────────────────┘  │
└─────────────┼──────────────────────────────────────────┘
              │
              │ Agent execution
              │
┌─────────────▼──────────────────────────────────────────┐
│              NexAU Agent Framework                     │
│  - LLM calls                                             │
│  - Tool execution                                        │
│  - Skill invocation                                      │
│  - Context management                                    │
└──────────────────────────────────────────────────────────┘
```

## Components

### 1. Node.js Frontend (`source/`)

#### `cli.js` - Entry Point
- Parses command-line arguments using `meow`
- Validates YAML file path
- Renders the main App component using Ink

#### `app.js` - Main UI Component
- **State Management**:
  - `messages`: Array of conversation history (each message includes its steps)
  - `currentSteps`: Array of steps for current in-progress task
  - `input`: Current user input text
  - `isProcessing`: Whether agent is currently processing
  - `statusMessage`: Current status message
  - `isReady`: Whether agent is ready for input
  - `error`: Current error message

- **Subcomponents**:
  - `Message`: Renders individual messages with role-based styling
  - `StatusMessage`: Shows loading states and status updates
  - `TextInput`: Handles user input

- **Process Management**:
  - Spawns Python subprocess on mount
  - Handles JSON communication via stdin/stdout
  - Cleans up process on unmount or exit

### 2. Python Backend (`agent_runner.py`)

#### Main Functions

- `main()`: Entry point
  - Loads agent from YAML
  - **Injects progress-tracking hooks**
  - Processes stdin messages in a loop
  - Sends responses to stdout

- `send_message(type, content, metadata)`: Utility for JSON communication
  - Formats messages as `{"type": "...", "content": "...", "metadata": {...}}`
  - Flushes output immediately

- `create_cli_progress_hook()`: Creates an after_model_hook
  - **Captures and displays agent's text responses** (reasoning/thinking)
  - Reports planned tool calls with tool names
  - Reports planned sub-agent calls
  - Filters out pure XML/JSON to show meaningful text

- `create_cli_tool_hook()`: Creates an after_tool_hook
  - Reports tool execution completion
  - Provides preview of tool output

#### Message Types

**From Python to Node.js:**
- `status`: General status updates (loading, etc.)
- `ready`: Agent is ready for input
- `step`: Intermediate progress step (with metadata)
  - `metadata.type`: Step type (thinking, tool_plan, tool_executed, etc.)
  - `metadata.tool_count`: Number of tools (for tool_plan)
  - `metadata.tools`: List of tool names
  - `metadata.iteration`: Current iteration number
- `agent_text`: Agent's reasoning/thinking text (non-tool content)
  - `metadata.type`: "agent_thinking"
  - Shows what the agent is thinking before executing tools
- `response`: Agent's final response to user message
- `error`: Error message

**From Node.js to Python:**
- `message`: User message with content
- `exit`: Shutdown signal

### 3. Communication Protocol

#### Message Format

All messages are JSON objects with this structure:

```json
{
  "type": "message_type",
  "content": "message content"
}
```

#### Flow Example

1. **Agent Startup**:
```
Python → Node: {"type": "status", "content": "Loading agent..."}
Python → Node: {"type": "status", "content": "Agent loaded successfully"}
Python → Node: {"type": "ready", "content": "Agent is ready"}
```

2. **User Sends Message**:
```
Node → Python: {"type": "message", "content": "Write hello.py"}
Python → Node: {"type": "step", "content": "Processing request..."}
Python → Node: {"type": "agent_text", "content": "I'll create a hello world program..."}
Python → Node: {"type": "step", "content": "Planning to execute 1 tool(s): file_write"}
Python → Node: {"type": "step", "content": "Tool 'file_write' completed"}
Python → Node: {"type": "response", "content": "I've created hello.py with..."}
Python → Node: {"type": "ready", "content": ""}
```

3. **Error Handling**:
```
Python → Node: {"type": "error", "content": "Tool execution failed"}
Python → Node: {"type": "ready", "content": ""}
```

## Data Flow

### User Input Flow
```
User types → TextInput → handleSubmit() → 
  Add to messages state →
  Send JSON to Python stdin →
  Set isReady=false
```

### Agent Response Flow
```
Python stdout → JSON parser → 
  Match message type →
  Update appropriate state →
  React re-renders UI
```

## UI Layout

The interface uses Ink's flexbox layout:

```
┌─────────────────────────────────────────┐
│ Header (borderStyle: round)             │ ← Fixed
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│                                          │
│ Conversation History (flexGrow: 1)      │ ← Grows to fill
│                                          │
│ - Message components                     │
│ - Status messages                        │
│ - Error messages                         │
│                                          │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ Input Bar (borderStyle: round)          │ ← Fixed
└─────────────────────────────────────────┘
```

## State Management

### React State Flow

```
Component Mount
  ↓
Spawn Python Process
  ↓
Listen to stdout → Parse JSON → Update State
  ↓
React Re-renders → Ink Updates Terminal
```

### Agent Context

The Python agent receives context with each message:

```python
context = {
    "date": get_date(),
    "username": os.getenv("USER"),
    "working_directory": os.getcwd(),
    "env_content": {
        "date": get_date(),
        "username": os.getenv("USER"),
        "working_directory": os.getcwd(),
    }
}
```

## Error Handling

### Node.js Side
- Validates YAML file exists before starting
- Catches spawn errors
- Handles process exit codes
- Ignores non-JSON output on stdout
- Displays errors in UI

### Python Side
- Try-catch around agent loading
- Try-catch around message processing
- All errors sent as JSON to Node.js
- Maintains ready state after errors

## Build Process

```
source/
  ├── app.js (JSX)
  └── cli.js (JSX)
       ↓
   [Babel with @babel/preset-react]
       ↓
dist/
  ├── app.js (Plain JS)
  └── cli.js (Plain JS + Shebang)
```

## Performance Considerations

1. **Streaming**: Messages are sent line-by-line, allowing real-time updates
2. **Buffering**: stdout is flushed after each message for immediate display
3. **State Updates**: React batches state updates for efficient rendering
4. **Process Cleanup**: Python process is killed on exit to prevent zombie processes

## Security Considerations

1. **YAML Validation**: File existence and extension checked before loading
2. **Path Resolution**: All paths are resolved to absolute paths
3. **Process Isolation**: Python runs as a separate process with limited communication
4. **Input Sanitization**: JSON parsing handles malformed input gracefully

## Dependencies

### Node.js Side
- `ink`: Terminal UI framework
- `ink-text-input`: Text input component
- `ink-spinner`: Loading spinner
- `meow`: CLI argument parser
- `react`: React for Ink components

### Python Side
- `uv`: Python package manager (required)
- `nexau`: The agent framework
- All dependencies managed via `pyproject.toml` and `uv.lock`

The CLI spawns Python using `uv run python` to ensure proper dependency isolation.

## Extension Points

### Adding New Message Types

1. Define in Python:
```python
send_message("custom_type", "content")
```

2. Handle in Node.js:
```javascript
case 'custom_type':
  // Handle new message type
  break;
```

### Adding UI Components

Create new React components in `source/`:

```javascript
const CustomComponent = ({data}) => {
  return <Box>...</Box>;
};
```

### Modifying Agent Context

Edit `agent_runner.py` to add more context:

```python
context = {
    ...existing_context,
    "custom_field": get_custom_data(),
}
```

## Testing Strategy

### Unit Tests
- React components (using ink-testing-library)
- Python message handlers
- JSON parsing/formatting

### Integration Tests
- Full message flow
- Error scenarios
- Agent loading

### Manual Tests
- Different terminal sizes
- Long conversations
- Network interruptions
- Various YAML configurations

## Future Enhancements

1. **Session Persistence**: Save/load conversation history
2. **Multiple Agents**: Switch between agents without restarting
3. **Rich Output**: Support for tables, code blocks, syntax highlighting
4. **Streaming Responses**: Show partial responses as they're generated
5. **Tool Visualization**: Show tool calls as they happen
6. **Debug Mode**: Toggle verbose logging
7. **Configuration**: CLI config file for defaults
8. **Themes**: Customizable color schemes

---

For implementation details, see the source code in `source/` and `agent_runner.py`.

