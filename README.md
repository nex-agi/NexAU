# NexAU Framework

A general-purpose agent framework inspired by Claude Code's architecture for building intelligent agents with tool capabilities.

This framework provides a modular tool system, a flexible agent architecture, and seamless integration with various LLM providers.

**➡️ For the full documentation, please see the [`docs/`](./docs/index.md) directory.**

---

## Installation

### From GitHub Release (Recommended)

**Using pip:**
```bash
# Install from the latest release tag using SSH (you need to use ssh because nexau is a private repo)
pip install git+ssh://git@github.com//nexau.git@v0.1.0

# or visit https://github.com//nexau/releases/ and download whl, then
pip install nexau-0.1.0-py3-none-any.whl
```

**Using uv:**
```bash
# Install from the latest release tag using SSH
uv pip install git+ssh://git@github.com//nexau.git@v0.1.0

# or visit https://github.com//nexau/releases/ and download whl, then
uv pip install nexau-0.1.0-py3-none-any.whl
```

### Install Latest from Main Branch

**Using pip:**
```bash
pip install git+ssh://git@github.com//nexau.git
```

**Using uv:**
```bash
uv pip install git+ssh://git@github.com//nexau.git
```

### From Source

```bash
# Clone the repository
git clone git@github.com:/nexau.git
cd nexau

# Install dependencies using uv
pip install uv
uv sync
```

## Quick Start

1.  **Set up your environment variables** in a `.env` file:
    ```.env
    LLM_MODEL="your-llm-model"
    LLM_BASE_URL="your-llm-api-base-url"
    LLM_API_KEY="your-llm-api-key"
    SERPER_API_KEY="api key from serper.dev" (required if you need to use web search)
    ```

2.  **Run an example:**
    ```bash
    # Ensure you have python-dotenv installed (`uv pip install python-dotenv`)
    dotenv run uv run examples/code_agent/start.py

    Enter your task: Build an algorithm art about 3-body problem
    ```

3. **Use NexAU CLI to run** (Simplified)
    
    **Option 1: Using the run-agent script (Recommended)**
    ```bash
    # One-liner to run any agent config
    ./run-agent examples/code_agent/cc_agent.yaml
    ```
    
    **Option 2: Using npm scripts**
    ```bash
    # One-time setup
    npm run setup-cli
    
    # Run agent (any time after setup)
    npm run agent examples/code_agent/cc_agent.yaml
    
    # Or use the full command name
    npm run run-agent examples/code_agent/cc_agent.yaml
    ```
    
    **Option 3: Manual approach (Original)**
    ```bash
    # Build the cli app
    cd cli
    npm install
    npm run build
    cd ../
    
    # Use cli to run an agent based on a yaml config
    dotenv run cli/dist/cli.js examples/code_agent/cc_agent.yaml
    ```
    

## Development

### Running Tests and Quality Checks

Before submitting a pull request, you can run the same checks that will run in CI:

```bash
# Install dependencies (including dev dependencies)
uv sync

# Run linter
uv run ruff check .

# Run format check
uv run ruff format --check .

# Auto-fix linting issues (optional)
uv run ruff check --fix .

# Auto-format code (optional)
uv run ruff format .

# Run tests with coverage
uv run pytest --cov=nexau --cov-report=html --cov-report=term
```

The coverage report will be generated in the `htmlcov/` directory. Open `htmlcov/index.html` in your browser to view the detailed coverage report.

### Continuous Integration

All pull requests to the `main` branch will automatically run:
- **Linting** - Code quality checks using ruff
- **Format checking** - Code style verification using ruff
- **Tests** - Full test suite with pytest
- **Coverage reporting** - Test coverage analysis

The workflow is defined in `.github/workflows/ci.yml`.

### Continuous Deployment

When a version tag (e.g., `v0.2.0`) is pushed to the repository, the CD workflow automatically:
- Runs the full test suite
- Builds the package
- Creates a GitHub release with changelog

See [RELEASING.md](./RELEASING.md) for detailed release instructions.
