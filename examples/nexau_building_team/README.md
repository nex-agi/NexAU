# NexAU Building Team Example

A multi-agent team demo: requirements → RFC → build → test, powered by NexAU.

## Architecture

- **Leader agent** — orchestrates the workflow and delegates to sub-agents
- **RFC Writer agent** — drafts RFC documents from requirements
- **Builder agent** — implements code based on the RFC

The backend exposes SSE endpoints at `http://localhost:8000`. The frontend (Vite + React) connects to it.

## Prerequisites

- Python 3.12+ with [uv](https://github.com/astral-sh/uv)
- Node.js 18+ with npm
- [`dotenv-cli`](https://github.com/entropitor/dotenv-cli) (`npm install -g dotenv-cli`)
- A `.env` file at the repo root with your API keys

## Launch

### 1. Backend (SSE server)

Run from the repo root:

```bash
SANDBOX_WORK_DIR=/path/to/your/workdir dotenv run sh -c 'cd examples/nexau_building_team && uv run start_server.py'
```

`SANDBOX_WORK_DIR` sets the directory the builder agent writes files into. The server starts on `http://0.0.0.0:8000`.

### 2. Frontend (Vite dev server)

In a separate terminal:

```bash
cd examples/nexau_building_team/frontend
npm install   # first time only
npm run dev
```

The UI is available at `http://localhost:5173` by default.
