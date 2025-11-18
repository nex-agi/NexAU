<p align="left">
    <a href="README_CN.md">中文</a> &nbsp ｜ &nbsp English
</p>


# NexAU Framework

A general-purpose agent framework for building intelligent agents with tool capabilities.

This framework provides a modular tool system, a flexible agent architecture, and seamless integration with various LLM providers.

**➡️ For the full documentation, please see the [`docs/`](./docs/index.md) directory.**

---

## Installation

### From GitHub Release (Recommended)

**Using pip:**
```bash
# Install from the latest release tag using SSH (you need to use ssh because nexau is a private repo)
pip install git+ssh://git@github.com/nex-agi/nexau.git@v0.3.2

# or visit https://github.com/nex-agi/nexau/releases/ and download whl, then
pip install nexau-0.3.2-py3-none-any.whl
```

**Using uv:**
```bash
# Install from the latest release tag using SSH
uv pip install git+ssh://git@github.com/nex-agi/nexau.git@v0.3.2

# or visit https://github.com/nex-agi/nexau/releases/ and download whl, then
uv pip install nexau-0.3.2-py3-none-any.whl
```

### Install Latest from Main Branch

**Using pip:**
```bash
pip install git+ssh://git@github.com/nex-agi/nexau.git
```

**Using uv:**
```bash
uv pip install git+ssh://git@github.com/nex-agi/nexau.git
```

### From Source

```bash
# Clone the repository
git clone git@github.com:nex-agi/nexau.git
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

    LANGFUSE_SECRET_KEY=sk-lf-xxx
    LANGFUSE_PUBLIC_KEY=pk-lf-xxx
    LANGFUSE_HOST="https://us.cloud.langfuse.com"
    ```
    Optional: NexAU uses Langfuse for tracing, setup your Langfuse keys if you want to enable traces.

2.  **Run an example:**
    ```bash
    # Ensure you have python-dotenv installed (`uv pip install python-dotenv`)
    dotenv run uv run examples/code_agent/start.py

    Enter your task: Build an algorithm art about 3-body problem
    ```

3.  **Prefer Python over YAML?** Create the same agent defined in `examples/code_agent/code_agent.yaml` directly in code, create a new file `code_agent.py`:
    ```python
    import logging
    import os
    from pathlib import Path

    from nexau import Agent, AgentConfig, LLMConfig, Skill, Tool
    from nexau.archs.main_sub.execution.hooks import LoggingMiddleware

    from nexau.archs.tool.builtin import (
        bash_tool,
        file_edit_tool,
        file_read_tool,
        file_write_tool,
        glob_tool,
        grep_tool,
        ls_tool,
        multiedit_tool,
        todo_write,
        web_read,
        web_search,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    base_dir = Path("examples/code_agent")

    # NexAU decouples the definition and implementation (binding) of tools
    tools = [
        Tool.from_yaml(base_dir / "tools/WebSearch.tool.yaml", binding=web_search),
        Tool.from_yaml(base_dir / "tools/WebFetch.tool.yaml", binding=web_read),
        Tool.from_yaml(base_dir / "tools/TodoWrite.tool.yaml", binding=todo_write),
        Tool.from_yaml(base_dir / "tools/Grep.tool.yaml", binding=grep_tool),
        Tool.from_yaml(base_dir / "tools/Glob.tool.yaml", binding=glob_tool),
        Tool.from_yaml(base_dir / "tools/Read.tool.yaml", binding=file_read_tool),
        Tool.from_yaml(base_dir / "tools/Write.tool.yaml", binding=file_write_tool),
        Tool.from_yaml(base_dir / "tools/Edit.tool.yaml", binding=file_edit_tool),
        Tool.from_yaml(base_dir / "tools/Bash.tool.yaml", binding=bash_tool),
        Tool.from_yaml(base_dir / "tools/Ls.tool.yaml", binding=ls_tool),
        Tool.from_yaml(base_dir / "tools/MultiEdit.tool.yaml", binding=multiedit_tool),
    ]

    # NexAU supports Skills (compatible with Claude Skills)
    skills = [
        Skill.from_folder(base_dir / "skills/theme-factory"),
        Skill.from_folder(base_dir / "skills/algorithmic-art"),
    ]

    agent_config = AgentConfig(
        name="nexau_code_agent",
        max_context_tokens=100000,
        system_prompt=str(base_dir / "system-workflow.md"),
        system_prompt_type="jinja",
        tool_call_mode="openai", # xml, openai or anthorpic
        llm_config=LLMConfig(
            temperature=0.7,
            max_tokens=4096,
            model=os.getenv("LLM_MODEL"),
            base_url=os.getenv("LLM_BASE_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            api_type="openai_chat_completion", # support openai_chat_completion (default), openai_responses (especially for gpt-5-codex), anthropic_chat_completion
        ),
        tools=tools,
        skills=skills,
        middlewares=[
            LoggingMiddleware(
                model_logger="nexau_code_agent",
                tool_logger="nexau_code_agent",
                log_model_calls=True,
            ),
        ],
    )

    agent = Agent(config = agent_config)

    print(agent.run("Build an algorithm art about 3-body problem", context={"working_directory": os.getcwd()}))

    ```
    Run it with `dotenv run uv run code_agent.py`

4. **Use NexAU CLI to run**
    
    **Using the run-agent script (Recommended)**
    ```bash
    # One-liner to run any NexAU agent yaml config
    ./run-agent examples/code_agent/code_agent.yaml
    ```
    NexAU CLI supports multi-round human interaction, tool call traces and sub-agent traces, which makes agent debugging easier.
    ![NexAU CLI](assets/nexau_cli.jpeg)

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
