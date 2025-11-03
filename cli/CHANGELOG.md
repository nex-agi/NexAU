# NorthAU CLI Changelog

## Version 0.1.0 - Interactive CLI with Progress Tracking

### Features

#### Core Functionality
- ✅ **Interactive Terminal UI** built with React + Ink
- ✅ **Two-region layout** (conversation history + input bar)
- ✅ **Real-time agent interaction** via stdin/stdout communication
- ✅ **YAML configuration support** - load any NorthAU agent

#### Progress Tracking (NEW!)
- ✅ **Live intermediate steps** - see what the agent is doing in real-time
- ✅ **Hook-based progress tracking** using NorthAU's hook system
- ✅ **Preserved action history** - all steps remain visible after task completion
- ✅ **Agent reasoning display** - shows agent's text responses and thinking
- ✅ **Color-coded step indicators**:
  - 💭 Agent reasoning/thinking text (magenta)
  - 🔧 Tool planning (blue)
  - ✓ Tool execution completed (green)
  - 🤖 Sub-agent calls (cyan)
  - ▶ Processing start (yellow)
- ✅ **Automatic hook injection** - CLI hooks are automatically added to any agent

#### User Experience
- ✅ **Keyboard controls** - Esc or Ctrl+C to exit
- ✅ **Status indicators** - visual feedback for agent state
- ✅ **Error handling** - friendly error messages
- ✅ **Loading states** - spinners and status messages
- ✅ **Message history** - full conversation displayed

### Technical Implementation

#### Architecture
- **Node.js Frontend**: Ink-based UI with React components
- **Python Backend**: Agent runner with hook injection
- **Communication**: JSON messages over stdin/stdout
- **Dependency Management**: Uses `uv run` for Python isolation

#### Hook System Integration
- `create_cli_progress_hook()`: Intercepts after_model_hooks to report:
  - Agent thinking and reasoning
  - Planned tool calls with tool names
  - Planned sub-agent calls
  - Iteration tracking
  
- `create_cli_tool_hook()`: Intercepts after_tool_hooks to report:
  - Tool execution completion
  - Tool output previews (truncated for display)

#### Message Protocol
- `status`: General loading/status updates
- `ready`: Agent ready for input
- `step`: Intermediate progress step with metadata
- `response`: Final agent response
- `error`: Error messages with stack traces

### Dependencies

#### Node.js
- `ink@^4.1.0` - Terminal UI framework
- `ink-text-input@5.0.1` - Text input component
- `ink-spinner@5.0.0` - Loading animations
- `meow@^11.0.0` - CLI argument parsing
- `react@^18.2.0` - React for Ink

#### Python
- `uv` - Python package manager (required)
- `northau` - Agent framework
- `pyyaml` - YAML parsing

### Files Created

```
cli/
├── source/
│   ├── app.js              # Main React/Ink UI (283 lines)
│   └── cli.js              # CLI entry point
├── dist/                   # Compiled output
│   ├── app.js
│   └── cli.js
├── agent_runner.py         # Python agent wrapper with hooks (219 lines)
├── test-cli.sh             # Quick test script
├── package.json            # Dependencies and scripts
├── readme.md               # Comprehensive documentation (163 lines)
├── QUICKSTART.md           # Quick start guide (106 lines)
├── EXAMPLES.md             # Usage examples (265 lines)
├── ARCHITECTURE.md         # Technical architecture (355 lines)
└── CHANGELOG.md            # This file
```

### Usage

```bash
# From cli directory
npm run build
./dist/cli.js ../examples/fake_claude_code/cc_agent.yaml

# Or use convenience scripts
npm run test-fake-cc
./test-cli.sh

# From project root
cli/dist/cli.js examples/fake_claude_code/cc_agent.yaml
```

### Example Session

```
┌─────────────────────────────────────────┐
│ 🤖 NorthAU Agent CLI (Press Esc to exit)│
└─────────────────────────────────────────┘

❯ You:
  Write a hello world program

⚡ Agent:
  💭 I'll create a simple Python hello world program for you...
  🔧 Planning to execute 1 tool(s) [sequential]:
    1. file_write(file_path=hello.py, contents=print("Hello, World!"))
  ✓ Tool 'file_write' completed
  ─────
  I've created a hello world program in hello.py with the
  following content: print("Hello, World!")

❯ You:
  Run it and then read the output

⚡ Agent:
  💭 I'll execute the hello.py file we just created...
  🔧 Planning to execute 2 tool(s) [parallel]:
    1. bash(command=python hello.py)
    2. file_read(file_path=hello.py)
  ✓ Tool 'bash' completed
  ✓ Tool 'file_read' completed
  ─────
  The program executed successfully and output: Hello, World!

┌─────────────────────────────────────────┐
│ ▶ Type your message and press Enter... │
└─────────────────────────────────────────┘
```

### Requirements

- Node.js >= 16
- Python 3.x
- uv (Python package manager)
- Valid NorthAU agent YAML configuration

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Node.js dependencies
cd cli
npm install

# Build the CLI
npm run build
```

### Known Issues

None currently reported.

### Future Enhancements

Potential improvements for future versions:

- [ ] Session persistence (save/load conversations)
- [ ] Multiple agent switching without restart
- [ ] Rich output formatting (tables, syntax highlighting)
- [ ] Streaming responses (show partial responses)
- [ ] Tool call visualization with parameters
- [ ] Debug mode with verbose logging
- [ ] Configuration file for CLI defaults
- [ ] Customizable color themes
- [ ] Export conversation to file
- [ ] Search conversation history

### Contributing

See main README.md for contribution guidelines.

### License

MIT

