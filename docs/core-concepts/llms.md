# 🧠 Core Concepts: LLM Configuration

NexAU is designed to work with any OpenAI-compatible API, giving you the flexibility to choose from a wide range of providers.

## LLM Configuration

You configure the LLM provider using the `LLMConfig` class.

#### Supported Providers

```python
from nexau.archs.llm import LLMConfig

# OpenAI
llm_config = LLMConfig(
    model="gpt-4",
    base_url="[https://api.openai.com/v1](https://api.openai.com/v1)",
    api_key="your-openai-key",
    temperature=0.7,
    max_tokens=4096
)

# Anthropic Claude (via a compatible proxy)
llm_config = LLMConfig(
    model="claude-3-sonnet-20240229",
    base_url="[https://api.anthropic.com](https://api.anthropic.com)", # or your proxy URL
    api_key="your-anthropic-key",
    temperature=0.7,
    max_tokens=4096
)

# Local/Custom endpoint (e.g., Ollama, vLLM)
llm_config = LLMConfig(
    model="custom-model",
    base_url="http://localhost:8000/v1",
    api_key="not-needed-for-local",
    temperature=0.7,
    max_tokens=4096
)
```

## Custom LLM Generators

For advanced use cases, NexAU supports custom LLM generators. These are functions that intercept the request to the LLM, allowing you to add custom logic like logging, caching, parameter modification, or even routing between different providers.
You can apply a custom generator either programmatically or through YAML configuration. Refer to [Custom LLM Generator](./advanced-guides/custom-llm-generator.md) for details.**
