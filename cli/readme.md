# NexAU CLI

An interactive command-line interface for NexAU agents, inspired by Claude Code. Run your NexAU agents in a beautiful terminal UI with conversation history and real-time interaction.

## Features

- 🎨 **Beautiful Terminal UI** - Clean, intuitive interface built with Ink
- 💬 **Interactive Chat** - Real-time conversation with your NexAU agent
- 📜 **Conversation History** - View all messages in the current session
- ⚡ **Live Status Updates** - See when the agent is thinking or processing
- 🔍 **Progress Tracking** - Watch intermediate steps like tool calls and sub-agent execution in real-time
- 🧠 **Thinking Visualization** - See agent reasoning and planning steps
- 🔧 **Tool Execution Feedback** - Get notified when tools are called and completed
- 🎯 **YAML Configuration** - Load any NexAU agent from a YAML file
- ⌨️ **Keyboard Controls** - Easy navigation with Esc or Ctrl+C to exit

## Installation

From the `cli` directory:

```bash
npm install
npm run build
```

## Usage

### Basic Usage

```bash
# Direct execution (if executable)
./dist/cli.js <path-to-agent-yaml>

# Or with node
node dist/cli.js <path-to-agent-yaml>
```

### Examples

Run the fake Claude Code agent:

```bash
./dist/cli.js ../examples/fake_claude_code/cc_agent.yaml
# or
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml
```

Run a custom agent:

```bash
./dist/cli.js ./my_custom_agent.yaml
```

Run from the project root:

```bash
cli/dist/cli.js examples/fake_claude_code/cc_agent.yaml
# or
node cli/dist/cli.js examples/fake_claude_code/cc_agent.yaml
```

## Interface

The CLI provides two main regions:

### 1. Conversation History (Top)
- Displays all messages from the current session
- User messages are shown with a cyan `❯ You:` prefix
- Agent responses are shown with a green `⚡ Agent:` prefix
- **Action history is preserved** - all steps remain visible after completion
- **Intermediate steps** are shown with color-coded icons:
  - 💭 Agent reasoning/thinking text (magenta)
  - 🔧 Tool planning with parameters (blue, bold header)
  - Tool details with parameters (cyan, indented list)
  - Shows [parallel] or [sequential] execution mode
  - ✓ Tool completed (green)
  - 🤖 Sub-agent calls (cyan)
  - ▶ Starting processing (yellow)
- Status messages and errors are displayed in yellow/red

### 2. Input Bar (Bottom)
- Type your message and press Enter to send
- The prompt shows `▶` when ready and `⏸` when processing
- Border turns green when ready for input, gray when waiting
- Input is disabled while the agent is processing

## Keyboard Controls

- **Enter** - Send your message
- **Esc** or **Ctrl+C** - Exit the CLI

## Requirements

- Node.js >= 16
- Python 3.x with NexAU installed
- **uv** (Python package manager) - [Install from here](https://docs.astral.sh/uv/)
- A valid NexAU agent YAML configuration file

The CLI uses `uv run` to execute the Python agent with proper dependency management.

## Development

### Watch Mode

For development with auto-rebuild:

```bash
npm run dev
```

### Project Structure

```
cli/
├── source/
│   ├── app.js      # Main React/Ink application
│   └── cli.js      # CLI entry point
├── agent_runner.py # Python agent wrapper
├── dist/           # Compiled output
├── package.json
└── readme.md
```

## How It Works

The CLI consists of two main components:

1. **Node.js Frontend (Ink)** - Provides the terminal UI and handles user interaction
2. **Python Backend (agent_runner.py)** - Runs the NexAU agent and communicates via stdin/stdout

When you start the CLI:
1. The Node.js process spawns a Python subprocess with your agent YAML file
2. The Python backend injects progress-tracking hooks into the agent
3. The agent loads and sends a "ready" signal
4. You type messages in the input bar
5. Messages are sent to Python as JSON over stdin
6. **As the agent works**, intermediate steps are streamed to the UI:
   - Agent thinking and reasoning
   - Tool calls being planned
   - Tools being executed
   - Sub-agents being called
7. Agent responses come back as JSON over stdout
8. The UI updates in real-time showing all steps

## Troubleshooting

### Agent fails to load
- Verify the YAML file path is correct
- Check that the YAML file is valid
- Ensure all required tools and skills are available
- Check Python logs for more details

### Python/uv not found
- Ensure `uv` is installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Ensure Python 3 is installed
- Make sure you're running the CLI from the project root or a subdirectory

### Terminal rendering issues
- Try resizing your terminal window
- Ensure your terminal supports Unicode characters
- Use a modern terminal emulator (iTerm2, Hyper, Windows Terminal, etc.)

## License

MIT
