# Usage Examples

This document provides various examples of how to use the NorthAU CLI in different scenarios.

## Basic Usage

### Example 1: Run with Fake Claude Code Agent

The simplest way to get started:

```bash
cd cli
npm run test-fake-cc
```

Or directly:

```bash
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml
```

### Example 2: Run with Deep Research Agent

```bash
node dist/cli.js ../examples/deep_research/deep_research_agent.yaml
```

### Example 3: Run with Custom Agent

```bash
node dist/cli.js /path/to/your/custom_agent.yaml
```

## Integration Examples

### Example 4: From a Shell Script

Create a script that launches the CLI with a specific agent:

```bash
#!/bin/bash
# launch-agent.sh

AGENT_PATH="$1"

if [ -z "$AGENT_PATH" ]; then
    echo "Usage: $0 <path-to-agent.yaml>"
    exit 1
fi

cd "$(dirname "$0")/cli"
node dist/cli.js "$AGENT_PATH"
```

Usage:
```bash
./launch-agent.sh examples/fake_claude_code/cc_agent.yaml
```

### Example 5: From Node.js

Spawn the CLI from another Node.js application:

```javascript
import { spawn } from 'child_process';
import path from 'path';

const agentPath = path.resolve('./examples/fake_claude_code/cc_agent.yaml');
const cliPath = path.resolve('./cli/dist/cli.js');

const cli = spawn('node', [cliPath, agentPath], {
    stdio: 'inherit',
    cwd: process.cwd()
});

cli.on('exit', (code) => {
    console.log(`CLI exited with code ${code}`);
});
```

### Example 6: Development Workflow

Set up different npm scripts for different agents:

Add to your `package.json`:

```json
{
  "scripts": {
    "agent:claude": "node cli/dist/cli.js examples/fake_claude_code/cc_agent.yaml",
    "agent:research": "node cli/dist/cli.js examples/deep_research/deep_research_agent.yaml",
    "agent:custom": "node cli/dist/cli.js"
  }
}
```

Then run:
```bash
npm run agent:claude
npm run agent:research
npm run agent:custom path/to/agent.yaml
```

## Advanced Examples

### Example 7: With Environment Variables

Pass environment variables to customize agent behavior:

```bash
OPENAI_API_KEY=sk-xxx \
PYTHONPATH=/custom/path \
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml
```

### Example 8: Running Multiple Agents

Open multiple terminals and run different agents simultaneously:

Terminal 1:
```bash
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml
```

Terminal 2:
```bash
node dist/cli.js ../examples/deep_research/deep_research_agent.yaml
```

### Example 9: With Custom Python Path

If you need to use a specific Python interpreter:

Modify `source/app.js` line where Python is spawned:

```javascript
const python = spawn('/path/to/your/python3', [pythonScript, yamlPath]);
```

Or create a wrapper script:

```bash
#!/bin/bash
# run-with-venv.sh

source /path/to/venv/bin/activate
node cli/dist/cli.js "$@"
```

Usage:
```bash
./run-with-venv.sh examples/fake_claude_code/cc_agent.yaml
```

## Testing Scenarios

### Example 10: Quick Smoke Test

Test that the CLI loads and responds:

```bash
# Start the CLI
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml

# Type a simple message
> Hello, can you help me?

# Wait for response, then exit with Esc
```

### Example 11: Interactive Development

While developing agents, keep the CLI running and modify agent files:

```bash
# Terminal 1: Watch mode for CLI
npm run dev

# Terminal 2: Run the CLI
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml

# Terminal 3: Edit agent files
vim ../examples/fake_claude_code/cc_agent.yaml
# Save changes, restart CLI to test
```

## Automation Examples

### Example 12: CI/CD Testing

Test agent loading in CI/CD:

```bash
#!/bin/bash
# test-agents.sh

for agent in examples/*/*.yaml; do
    echo "Testing $agent..."
    timeout 10s node cli/dist/cli.js "$agent" <<EOF
Hello
EOF
    if [ $? -eq 124 ]; then
        echo "✓ $agent loaded successfully (timed out as expected)"
    else
        echo "✗ $agent failed to load"
        exit 1
    fi
done
```

### Example 13: Batch Processing

Process multiple queries with an agent (requires modification for non-interactive mode):

```python
# batch_process.py
import subprocess
import json

queries = [
    "What is NorthAU?",
    "How do I create an agent?",
    "List available tools"
]

for query in queries:
    print(f"\nProcessing: {query}")
    # This is a conceptual example - actual implementation would need
    # non-interactive mode support in the CLI
```

## Troubleshooting Examples

### Example 14: Debug Mode

Add logging to see what's happening:

Modify `agent_runner.py` to add more verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Example 15: Capture Output

Save conversation to a file:

```bash
node dist/cli.js ../examples/fake_claude_code/cc_agent.yaml | tee conversation.log
```

## Tips

1. **Always build before running**: `npm run build` or use npm scripts that include build
2. **Check Python path**: Make sure `python3` is in your PATH
3. **Verify YAML files**: Ensure agent YAML files are valid before running
4. **Use absolute paths**: When in doubt, use absolute paths for YAML files
5. **Monitor resources**: Large agents may consume significant memory

---

For more information, see the [README.md](./readme.md) and [QUICKSTART.md](./QUICKSTART.md).

