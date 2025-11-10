# Changelog

All notable changes to the NexAU framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflows for automated testing, linting, and releases
- Automated PyPI publishing on version tags
- Comprehensive release documentation
- Optional OpenAI Responses API support (`api_type: openai_responses`) including preservation of reasoning output for multi-turn conversations

## [0.1.0] - 2024-10-26

### Added
- Initial release of NexAU framework
- General-purpose agent framework inspired by Claude Code's architecture
- Modular tool system with builtin tools
- Flexible agent architecture with main/sub agent support
- Integration with various LLM providers (OpenAI, etc.)
- MCP (Model Context Protocol) integration
- YAML-based agent and tool configuration
- Jinja2 templating for prompts
- Comprehensive test suite with pytest
- Code quality tools (ruff linting and formatting)
- Deep research example with web search and writing capabilities
- Fake Claude Code example demonstrating multi-tool usage
- Documentation for core concepts and advanced guides

### Features
- **Agent System**: Main and sub-agent architecture for complex workflows
- **Tool System**: Extensible tool framework with builtin tools (Bash, File operations, Web, etc.)
- **LLM Support**: Flexible LLM configuration with support for multiple providers
- **Configuration**: YAML-based configuration for agents and tools
- **Tracing**: Built-in tracing and debugging capabilities
- **Hooks**: Lifecycle hooks for customization
- **Global Storage**: Shared state management across agents

[Unreleased]: https://github.com/china-qijizhifeng/nexau/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/china-qijizhifeng/nexau/releases/tag/v0.1.0
