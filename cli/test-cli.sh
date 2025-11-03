#!/bin/bash
# Convenience script to test the NorthAU CLI with the fake Claude Code agent

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if build exists
if [ ! -f "dist/cli.js" ]; then
    echo "Building CLI..."
    npm run build
fi

# Run with the fake Claude Code agent
echo "Starting NorthAU CLI with Fake Claude Code agent..."
./dist/cli.js ../examples/fake_claude_code/cc_agent.yaml

