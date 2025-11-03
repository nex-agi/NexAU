# Quick Start Guide

Get up and running with NorthAU CLI in under 2 minutes!

## Step 1: Install Dependencies

```bash
cd cli
npm install
```

## Step 2: Build the CLI

```bash
npm run build
```

## Step 3: Run the CLI

### Option A: Use the test script (easiest)

```bash
./test-cli.sh
```

### Option B: Run directly with the fake Claude Code example

```bash
npm run test-fake-cc
```

### Option C: Run with a custom agent

```bash
./dist/cli.js path/to/your/agent.yaml
# or
node dist/cli.js path/to/your/agent.yaml
```

## What You'll See

```
┌─────────────────────────────────────────┐
│ 🤖 NorthAU Agent CLI (Press Esc or Ctrl+C to exit) │
└─────────────────────────────────────────┘

[Conversation history appears here]

┌─────────────────────────────────────────┐
│ ▶ Type your message and press Enter... │
└─────────────────────────────────────────┘
```

## Example Interaction

1. Wait for the agent to load (you'll see a green border when ready)
2. Type your message: `Write a hello world program`
3. Press Enter
4. **Watch the magic happen:**
   - See the agent's reasoning (💭 "I'll create a Python hello world...")
   - Watch it plan tool calls with parameters:
     ```
     🔧 Planning to execute 1 tool(s) [sequential]:
       1. file_write(file_path=hello.py, contents=print(...))
     ```
   - See tools execute in real-time (✓ Tool 'file_write' completed)
   - Get the final response with separator line
5. **Scroll up to see full history** - all actions are preserved!
6. Continue chatting!

## Tips

- **Wait for the green border** before typing - this means the agent is ready
- **Press Esc or Ctrl+C** to exit at any time
- **Scroll up** in your terminal to see older messages if needed
- **Keep your terminal window large** for the best experience

## Troubleshooting

### "uv not found" or "Python not found"
Make sure `uv` is installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then verify:
```bash
uv --version
```

### "Module not found"
Make sure you've installed NorthAU dependencies:
```bash
cd ..
uv sync
```

### "Agent failed to load"
Check that your YAML file path is correct and the agent configuration is valid.

## Next Steps

- Read the full [README.md](./readme.md) for more details
- Check out the [examples](../examples/) directory for more agent configurations
- Build your own agent YAML file and try it out!

---

Happy chatting! 🚀

