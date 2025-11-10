
### Hooks

The NexAU framework provides a powerful hook system that allows you to intercept and modify agent behavior after the LLM generates a response but before tool/agent execution. This enables custom logic, safety measures, conversation management, and much more.

#### Hook System Overview

Hooks receive an `AfterModelHookInput` containing:
- `original_response`: The raw LLM response
- `parsed_response`: Parsed tool/agent calls
- `messages`: Current conversation history
- `max_iterations`: Maximum allowed iterations
- `current_iteration`: Current iteration number

Hooks return a `AfterModelHookResult` that can modify:
- `parsed_response`: Filter or modify tool/agent calls
- `messages`: Add context or modify conversation history

#### Basic Hook Usage

```python
from nexau.archs.main_sub.execution.hooks import AfterModelHook, AfterModelHookResult, AfterModelHookInput

def create_context_hook() -> AfterModelHook:
    def context_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        if hook_input.parsed_response and hook_input.parsed_response.tool_calls:
            # Add context message before tool execution
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": f"[HOOK] About to execute {len(hook_input.parsed_response.tool_calls)} tool(s)"
            })
            return AfterModelHookResult.with_modifications(messages=modified_messages)

        return AfterModelHookResult.no_changes()

    return context_hook

# Use with agent
agent = create_agent(
    name="my_agent",
    tools=[...],
    after_model_hooks=[create_context_hook()],
    llm_config=llm_config
)
```

#### Built-in Hook Factories

##### Logging Hook
```python
from nexau.archs.main_sub.execution.hooks import create_logging_hook

# Logs detailed information about each hook execution
logging_hook = create_logging_hook("my_logger")
```

##### Filter Hook
```python
from nexau.archs.main_sub.execution.hooks import create_filter_hook

# Only allow specific tools and agents
filter_hook = create_filter_hook(
    allowed_tools={'web_search', 'file_read'},
    allowed_agents={'research_agent', 'data_agent'}
)
```

##### Remaining Iterations Hook
```python
from nexau.archs.main_sub.execution.hooks import create_remaining_reminder_hook

# Adds iteration count reminders to conversation
reminder_hook = create_remaining_reminder_hook()
```
#### Tool Approve Hook
Each time the tool with `tool_name` is called, the CLI prompts whether to approve (y/n). If no, the agent stops.

```python
from nexau.archs.main_sub.execution.hooks import create_tool_after_approve_hook

tool_after_approve_hook = create_tool_after_approve_hook(
    tool_name='WebSearch'
)
```


#### Advanced Hook Examples

##### Safety and Compliance Hook
```python
def create_safety_hook() -> AfterModelHook:
    def safety_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        if not hook_input.parsed_response:
            return AfterModelHookResult.no_changes()

        # Filter out potentially dangerous tools
        dangerous_tools = {'system_command', 'file_delete', 'network_access'}
        safe_calls = [
            call for call in hook_input.parsed_response.tool_calls
            if call.tool_name not in dangerous_tools
        ]

        if len(safe_calls) != len(hook_input.parsed_response.tool_calls):
            # Create modified parsed response
            from nexau.archs.main_sub.execution.parse_structures import ParsedResponse
            modified_parsed = ParsedResponse(
                original_response=hook_input.parsed_response.original_response,
                tool_calls=safe_calls,
                sub_agent_calls=hook_input.parsed_response.sub_agent_calls,
                batch_agent_calls=hook_input.parsed_response.batch_agent_calls,
                is_parallel_tools=hook_input.parsed_response.is_parallel_tools,
                is_parallel_sub_agents=hook_input.parsed_response.is_parallel_sub_agents
            )

            # Add safety message
            modified_messages = hook_input.messages.copy()
            filtered_count = len(hook_input.parsed_response.tool_calls) - len(safe_calls)
            modified_messages.append({
                "role": "system",
                "content": f"[SAFETY] Blocked {filtered_count} potentially dangerous tool calls"
            })

            return AfterModelHookResult.with_modifications(
                parsed_response=modified_parsed,
                messages=modified_messages
            )

        return AfterModelHookResult.no_changes()

    return safety_hook
```

##### Conversation Management Hook
```python
def create_conversation_manager_hook() -> AfterModelHook:
    def conversation_manager_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        # Add iteration warnings when approaching limit
        remaining = hook_input.max_iterations - hook_input.current_iteration

        if remaining <= 2:
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": f"[WARNING] Only {remaining} iterations remaining. Please provide a conclusive response."
            })
            return AfterModelHookResult.with_modifications(messages=modified_messages)

        # Add conversation length management
        if len(hook_input.messages) > 20:
            modified_messages = hook_input.messages.copy()
            modified_messages.append({
                "role": "system",
                "content": "[INFO] Long conversation detected. Consider summarizing key points."
            })
            return AfterModelHookResult.with_modifications(messages=modified_messages)

        return AfterModelHookResult.no_changes()

    return conversation_manager_hook
```

##### Custom Business Logic Hook
```python
def create_business_logic_hook(user_permissions: set[str]) -> AfterModelHook:
    def business_logic_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        if not hook_input.parsed_response:
            return AfterModelHookResult.no_changes()

        # Apply business rules based on user permissions
        allowed_calls = []
        for call in hook_input.parsed_response.tool_calls:
            if call.tool_name == 'database_query' and 'db_read' not in user_permissions:
                continue  # Skip unauthorized database calls
            elif call.tool_name == 'file_write' and 'file_modify' not in user_permissions:
                continue  # Skip unauthorized file operations
            else:
                allowed_calls.append(call)

        if len(allowed_calls) != len(hook_input.parsed_response.tool_calls):
            # Create modified response with business logic applied
            from nexau.archs.main_sub.execution.parse_structures import ParsedResponse
            modified_parsed = ParsedResponse(
                original_response=hook_input.parsed_response.original_response,
                tool_calls=allowed_calls,
                sub_agent_calls=hook_input.parsed_response.sub_agent_calls,
                batch_agent_calls=hook_input.parsed_response.batch_agent_calls,
                is_parallel_tools=hook_input.parsed_response.is_parallel_tools,
                is_parallel_sub_agents=hook_input.parsed_response.is_parallel_sub_agents
            )

            return AfterModelHookResult.with_modifications(parsed_response=modified_parsed)

        return AfterModelHookResult.no_changes()

    return business_logic_hook
```

#### Using Multiple Hooks

```python
# Combine multiple hooks for comprehensive control
agent = create_agent(
    name="secure_agent",
    tools=[web_search_tool, file_tools, database_tools],
    after_model_hooks=[
        create_logging_hook("agent_logger"),
        create_safety_hook(),
        create_business_logic_hook({'db_read', 'file_modify'}),
        create_conversation_manager_hook(),
        create_filter_hook(allowed_tools={'web_search', 'file_read'})
    ],
    llm_config=llm_config
)
```

#### Hook Execution Order

Hooks are executed in the order they are provided:
1. Each hook receives the current state (after previous hooks)
2. If a hook modifies the parsed response or messages, subsequent hooks see those changes
3. The final result is used for tool/agent execution

#### Hook Result Methods

```python
# No modifications
return AfterModelHookResult.no_changes()

# Only modify parsed response
return AfterModelHookResult.with_modifications(parsed_response=modified_parsed)

# Only modify messages
return AfterModelHookResult.with_modifications(messages=modified_messages)

# Modify both
return AfterModelHookResult.with_modifications(
    parsed_response=modified_parsed,
    messages=modified_messages
)

# Direct construction (also valid)
return AfterModelHookResult(parsed_response=modified_parsed, messages=modified_messages)
```

#### Use Cases

- **Safety & Compliance**: Filter dangerous tools, enforce business rules
- **Debugging & Monitoring**: Log execution patterns, track tool usage
- **Conversation Management**: Add context, manage conversation length
- **Custom Business Logic**: Apply domain-specific rules and permissions
- **Quality Control**: Validate tool parameters, ensure proper usage
- **User Experience**: Add helpful hints, progress indicators, warnings

The hook system provides powerful capabilities for customizing agent behavior while maintaining clean separation of concerns and type safety.

### Tool Hooks

In addition to model hooks, NexAU provides `after_tool_hooks` that intercept and modify tool execution results. These hooks are called after each tool is executed but before the result is processed by the agent.

#### Tool Hook System Overview

Tool hooks receive an `AfterToolHookInput` containing:
- `tool_name`: Name of the executed tool
- `tool_input`: Parameters passed to the tool
- `tool_output`: Result returned by the tool
- `global_storage`: Shared storage for cross-tool state

Tool hooks return an `AfterToolHookResult` that can modify:
- `tool_output`: Transform or filter the tool's output

#### Basic Tool Hook Usage

```python
from nexau.archs.main_sub.execution.hooks import AfterToolHook, AfterToolHookInput, AfterToolHookResult

def create_tool_output_logger() -> AfterToolHook:
    def tool_logger_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        print(f"🔧 Tool '{hook_input.tool_name}' executed")
        print(f"   Input: {hook_input.tool_input}")
        print(f"   Output: {hook_input.tool_output}")

        # Return no modifications - just log
        return AfterToolHookResult.no_changes()

    return tool_logger_hook

# Use with agent
agent = create_agent(
    name="my_agent",
    tools=[web_search_tool, file_tools],
    after_tool_hooks=[create_tool_output_logger()],
    llm_config=llm_config
)
```

#### Built-in Tool Hook Factories

##### Tool Logging Hook
```python
from nexau.archs.main_sub.execution.hooks import create_tool_logging_hook

# Logs detailed tool execution information
tool_logging_hook = create_tool_logging_hook("tool_debug_logger")
```

#### YAML Configuration for Tool Hooks

Tool hooks can be configured directly in YAML files:

```yaml
name: research_agent
system_prompt: "You are a research assistant with web access."
llm_config:
  model: gpt-4
  temperature: 0.7
tools:
  - name: web_search
    yaml_path: ./tools/WebSearch.yaml
    binding: nexau.archs.tool.builtin.web_tool:web_search
  - name: web_read
    yaml_path: ./tools/WebRead.yaml
    binding: nexau.archs.tool.builtin.web_tool:web_read
after_tool_hooks:
  # Simple logging hook
  - import: nexau.archs.main_sub.execution.hooks:create_tool_logging_hook
    params:
      logger_name: "research_tool_debug"

  # Custom transformer
  - import: my_module.hooks:create_custom_tool_transformer
    params:
      add_timestamps: true
      format_output: true
```

#### Advanced Tool Hook Examples

##### Data Validation Hook
```python
def create_tool_validation_hook() -> AfterToolHook:
    def validation_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        tool_output = hook_input.tool_output

        # Validate web search results
        if hook_input.tool_name == "web_search":
            if isinstance(tool_output, dict) and "results" in tool_output:
                # Filter out invalid results
                valid_results = [
                    r for r in tool_output["results"]
                    if r.get("title") and r.get("link")
                ]

                if len(valid_results) != len(tool_output["results"]):
                    modified_output = tool_output.copy()
                    modified_output["results"] = valid_results
                    modified_output["filtered_count"] = len(tool_output["results"]) - len(valid_results)

                    return AfterToolHookResult.with_modifications(tool_output=modified_output)

        return AfterToolHookResult.no_changes()

    return validation_hook
```


#### Using Multiple Tool Hooks

```python
# Combine multiple tool hooks for comprehensive tool management
agent = create_agent(
    name="production_agent",
    tools=[web_search_tool, file_tools, database_tools],
    after_tool_hooks=[
        create_tool_logging_hook("production_tools"),
        create_tool_validation_hook(),
        create_performance_monitor_hook(),
        create_tool_output_filter_hook({"password", "secret", "api_key"}),
        create_error_recovery_hook()
    ],
    llm_config=llm_config
)
```

#### Tool Hook Result Methods

```python
# No modifications
return AfterToolHookResult.no_changes()

# Modify tool output
return AfterToolHookResult.with_modifications(tool_output=modified_output)

# Direct construction (also valid)
return AfterToolHookResult(tool_output=modified_output)
```

#### Tool Hook Use Cases

- **Debugging & Monitoring**: Log tool inputs/outputs, track performance metrics
- **Data Validation**: Validate and sanitize tool outputs before processing
- **Security & Privacy**: Filter sensitive information from tool results
- **Error Handling**: Transform error responses into user-friendly messages
- **Data Transformation**: Format, enhance, or standardize tool outputs
- **Analytics**: Collect usage statistics and performance data
- **Caching**: Implement custom caching strategies for expensive tools
- **Rate Limiting**: Monitor and control tool usage patterns

#### Tool Hook Execution Flow

1. Tool is executed with provided parameters
2. Raw tool output is generated
3. Each tool hook processes the output in sequence
4. Modified output (if any) is passed to the next hook
5. Final output is returned to the agent for processing

Tool hooks provide fine-grained control over tool behavior and enable powerful tool output processing pipelines while maintaining clean separation of concerns.
