#!/bin/bash
# Convenience script to test the NexAU CLI with the fake Claude Code agent

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if build exists
if [ ! -f "dist/cli.js" ]; then
    echo "Building CLI..."
    npm run build
fi

# Run with the fake Claude Code agent
echo "Starting NexAU CLI with Fake Claude Code agent..."
./dist/cli.js ../examples/code_agent/cc_agent.yaml

