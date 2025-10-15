
## Custom LLM Generators

Northau supports custom LLM generators, allowing you to customize how LLM requests are processed. This is useful for:
- Adding custom preprocessing/postprocessing
- Integrating with different LLM providers
- Implementing custom caching or logging
- Adjusting parameters based on context

### Creating a Custom LLM Generator

A custom LLM generator is a function that takes the OpenAI client and request parameters, and returns a response:

```python
from typing import Any, Dict

def my_custom_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """
    Custom LLM generator function.

    Args:
        openai_client: The OpenAI client instance
        kwargs: Parameters that would be passed to openai_client.chat.completions.create()

    Returns:
        Response object with same structure as OpenAI's response
    """
    # Add custom logic here
    print(f"🔧 Processing request with {len(kwargs.get('messages', []))} messages")

    # Modify parameters if needed
    modified_kwargs = kwargs.copy()
    if modified_kwargs.get('temperature', 0.7) > 0.5:
        modified_kwargs['temperature'] = 0.3  # Lower temperature for more focused responses
        print("🎯 Adjusted temperature for better focus")

    # Call the LLM (you can use any provider here)
    response = openai_client.chat.completions.create(**modified_kwargs)

    # Add custom post-processing
    if response and hasattr(response, 'choices') and response.choices:
        content = response.choices[0].message.content
        print(f"📊 Generated response: {len(content)} characters")

    return response
```

### Using Custom LLM Generators

#### 1. Programmatic Usage

```python
from northau.archs.main_sub import create_agent
from northau.archs.llm import LLMConfig

# Create agent with custom LLM generator
agent = create_agent(
    name="custom_agent",
    system_prompt="You are a helpful assistant.",
    llm_config=LLMConfig(model="gpt-4", temperature=0.7),
    custom_llm_generator=my_custom_generator,  # Your custom function
    tools=[]
)

# Use the agent normally - custom generator will be used automatically
response = agent.run("Hello, how can you help me?")
```

#### 2. YAML Configuration

You can configure custom LLM generators directly in YAML files:

**Simple Configuration:**
```yaml
name: my_agent
system_prompt: "You are a helpful assistant"
llm_config:
  model: gpt-4
  temperature: 0.7
custom_llm_generator: "my_module.generators:my_custom_generator"
tools: []
```

**Parameterized Configuration:**
```yaml
name: research_agent
system_prompt: "You are a research assistant"
llm_config:
  model: gpt-4
  temperature: 0.7
custom_llm_generator:
  import: "my_module.generators:parameterized_generator"
  params:
    min_temperature: 0.1
    max_temperature: 0.4
    add_context: true
    log_requests: true
tools: []
```

For parameterized generators, create a function that accepts the parameters:

```python
def parameterized_generator(openai_client: Any, kwargs: Dict[str, Any],
                          min_temperature: float = 0.2,
                          max_temperature: float = 0.8,
                          add_context: bool = False,
                          log_requests: bool = False) -> Any:
    """Parameterized custom LLM generator."""
    if log_requests:
        print(f"🔍 LLM Request: {kwargs.get('model', 'unknown')} model")

    # Clamp temperature within specified range
    current_temp = kwargs.get('temperature', 0.7)
    modified_kwargs = kwargs.copy()
    modified_kwargs['temperature'] = max(min_temperature, min(max_temperature, current_temp))

    if add_context:
        # Add custom context to system message
        messages = modified_kwargs.get('messages', [])
        if messages and messages[0].get('role') == 'system':
            enhanced_content = f"{messages[0]['content']}\n\nAdditional Context: Focus on providing detailed, accurate responses."
            modified_kwargs['messages'] = messages.copy()
            modified_kwargs['messages'][0] = {'role': 'system', 'content': enhanced_content}

    return openai_client.chat.completions.create(**modified_kwargs)
```

### Example Use Cases

#### 1. Research-Optimized Generator

```python
def research_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """Generator optimized for research tasks."""
    # Lower temperature for more focused responses
    modified_kwargs = kwargs.copy()
    modified_kwargs['temperature'] = min(kwargs.get('temperature', 0.7), 0.3)

    # Add research context
    messages = kwargs.get('messages', [])
    if messages and messages[0].get('role') == 'system':
        research_prompt = f"{messages[0]['content']}\n\nIMPORTANT: Provide accurate, well-researched responses with citations when possible."
        modified_kwargs['messages'] = messages.copy()
        modified_kwargs['messages'][0] = {'role': 'system', 'content': research_prompt}

    return openai_client.chat.completions.create(**modified_kwargs)
```

#### 2. Multi-Provider Generator

```python
def multi_provider_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """Generator that can switch between different LLM providers."""
    model = kwargs.get('model', 'gpt-4')

    if model.startswith('claude'):
        # Use Anthropic client
        # (You would implement Anthropic client logic here)
        pass
    elif model.startswith('gpt'):
        # Use OpenAI client
        return openai_client.chat.completions.create(**kwargs)
    else:
        # Use custom provider
        # (You would implement custom provider logic here)
        pass
```

#### 3. Caching Generator

```python
import hashlib
import json

cache = {}

def caching_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
    """Generator with response caching."""
    # Create cache key from request
    cache_key = hashlib.md5(json.dumps(kwargs, sort_keys=True).encode()).hexdigest()

    # Check cache first
    if cache_key in cache:
        print("💾 Using cached response")
        return cache[cache_key]

    # Generate new response
    response = openai_client.chat.completions.create(**kwargs)

    # Cache the response
    cache[cache_key] = response
    print("🆕 Generated and cached new response")

    return response
```

### Built-in Custom Generators

Northau includes some built-in custom generators for common use cases:

#### Bypass Generator

A simple pass-through generator that adds logging but doesn't modify the LLM behavior:

```yaml
custom_llm_generator: "northau.archs.main_sub.execution.response_generator:bypass_llm_generator"
```

This generator:
- Logs the number of messages being processed
- Calls the standard OpenAI API without modifications
- Useful for debugging and monitoring LLM calls

### Loading Agents with Custom Generators

```python
from northau.archs.config.config_loader import load_agent_config

# Load agent from YAML with custom LLM generator
agent = load_agent_config('path/to/config.yaml')

# The custom generator will be automatically used for all LLM calls
response = agent.run("Your message here")
```

### Best Practices

1. **Maintain Compatibility**: Ensure your custom generator returns a response object with the same structure as OpenAI's response (with `.choices[0].message.content` attribute).

2. **Error Handling**: Implement proper error handling in your custom generator:
   ```python
   def robust_generator(openai_client: Any, kwargs: Dict[str, Any]) -> Any:
       try:
           return openai_client.chat.completions.create(**kwargs)
       except Exception as e:
           print(f"❌ LLM call failed: {e}")
           # Implement fallback logic or re-raise
           raise
   ```

3. **Parameter Validation**: Validate and sanitize parameters before making LLM calls.

4. **Logging**: Add appropriate logging for debugging and monitoring.

5. **Performance**: Consider the performance impact of your custom logic, especially for high-frequency use cases.

