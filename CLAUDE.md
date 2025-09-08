Use `uv run` instead of `python` to run python code

# Folder Structure

```
northau/
├── CLAUDE.md                    # Project instructions
├── README.md                    # Project documentation
├── pyproject.toml              # Python project configuration
├── uv.lock                     # Dependency lock file
├── quickstart.py               # Quick start script
├── tests/                      # Test files
│   ├── test_agent_context.py   # Agent context tests
│   ├── test_context_simple.py  # Simple context tests
│   └── test_context_with_tools.py # Context with tools tests
├── northau/                    # Main package
│   ├── __init__.py
│   ├── archs/                  # Architecture components
│   │   ├── config/             # Configuration management
│   │   │   ├── config_loader.py
│   │   ├── llm/                # LLM configuration
│   │   │   ├── llm_config.py
│   │   ├── main_sub/           # Main agent subsystem
│   │   │   ├── agent.py        # Core agent implementation
│   │   │   ├── agent_context.py # Agent context management
│   │   │   ├── prompt_handler.py # Prompt handling
│   │   │   ├── spec.md         # Specifications
│   │   └── tool/               # Tool system
│   │       ├── tool.py         # Core tool interface
│   │       └── builtin/        # Built-in tools
│   │           ├── bash_tool.py
│   │           ├── file_tool.py
│   │           ├── serper_search.py
│   │           ├── web_reader.py
│   │           └── web_tool.py
└── tools/                      # Tool configurations
    ├── Bash.yaml
    ├── Edit.yaml
    ├── Grep.yaml
    ├── WebRead.yaml
    └── WebSearch.yaml
```
