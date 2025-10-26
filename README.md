# Northau Framework

A general-purpose agent framework inspired by Claude Code's architecture for building intelligent agents with tool capabilities.

This framework provides a modular tool system, a flexible agent architecture, and seamless integration with various LLM providers.

**➡️ For the full documentation, please see the [`docs/`](./docs/index.md) directory.**

---

## Installation

### From GitHub Release (Recommended)

```bash
# Install from the latest release tag using SSH (you need to use ssh because northau is a private repo)
pip install git+ssh://git@github.com/china-qijizhifeng/northau.git@v0.1.0

# or visit https://github.com/china-qijizhifeng/northau/releases/ and download whl, then

pip install northau-0.1.0-py3-none-any.whl
```


### Install Latest from Main Branch

```bash
# Using SSH
pip install git+ssh://git@github.com/china-qijizhifeng/northau.git
```

### From Source

```bash
# Clone the repository
git clone git@github.com:china-qijizhifeng/northau.git
cd northau

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
    ```

2.  **Run an example:**
    ```bash
    # Ensure you have python-dotenv installed (`uv pip install python-dotenv`)
    dotenv run uv run examples/deep_research/quickstart.py
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
uv run pytest --cov=northau --cov-report=html --cov-report=term
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

#### Quick Release

```bash
# Using the helper script
./scripts/release.sh 0.2.0

# Or manually
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```