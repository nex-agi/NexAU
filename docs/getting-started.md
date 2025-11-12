# 🚀 Getting Started

This guide will walk you through installing the NexAU framework and running your first agent.

## Installation

```bash
# Clone the repository
git clone git@github.com:nex-agi/nexau.git
cd nexau

# Install dependencies using uv
pip install uv
uv sync
```

## Environment Setup

First, set up your environment variables. You can export them directly or create a `.env` file in the project root.

```.env
# For LLM Providers
LLM_MODEL="glm-4.5"
LLM_BASE_URL="[https://***REMOVED***/v1/](https://***REMOVED***/v1/)"
LLM_API_KEY="sk-xxxx"

# For Built-in Tools
SERPER_API_KEY="your-serper-api-key"
BP_HTML_PARSER_URL="[http://***REMOVED***/url2md](http://***REMOVED***/url2md)"
BP_HTML_PARSER_API_KEY="xxx"
BP_HTML_PARSER_SECRET="xxx"

# Optional: For Langfuse Tracing
LANGFUSE_SECRET_KEY="sk-lf-xxxx"
LANGFUSE_PUBLIC_KEY="pk-lf-xxxx"
LANGFUSE_HOST="[***REMOVED***](***REMOVED***)"
```

## Running Examples

With your environment configured, you can run the provided examples.

```bash
# Ensure you have python-dotenv installed (`uv pip install python-dotenv`)
# It reads variables from your .env file

# Run the full-featured Claude-code-like example
dotenv run uv run examples/code_agent/start.py

# Run an MCP example
dotenv run uv run python -m examples.mcp.mcp_amap_example
```