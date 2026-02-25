# Built-in Tools Reference

Catalog of all NexAU built-in tools with their binding paths and parameter summaries.

## File Tools

Module: `nexau.archs.tool.builtin.file_tools`

### read_file

Reads file content. Supports text, images, audio, and PDF.

```
Binding: nexau.archs.tool.builtin.file_tools:read_file
Parameters:
  - file_path (string, required): Path to the file
  - offset (number, optional): 0-based line number to start from
  - limit (number, optional): Max lines to read
```

### write_file

Writes content to a file. Creates parent directories if needed.

```
Binding: nexau.archs.tool.builtin.file_tools:write_file
Parameters:
  - file_path (string, required): Path to the file
  - content (string, required): Content to write
```

### replace

Performs string replacement in a file. Use for partial edits.

```
Binding: nexau.archs.tool.builtin.file_tools:replace
Parameters:
  - file_path (string, required): Path to the file
  - old_string (string, required): Text to find
  - new_string (string, required): Replacement text
```

### search_file_content

Searches for patterns in files using regex.

```
Binding: nexau.archs.tool.builtin.file_tools:search_file_content
Parameters:
  - pattern (string, required): Regex pattern to search
  - path (string, optional): Directory to search in
  - file_pattern (string, optional): Glob pattern to filter files
```

### glob

Finds files matching a glob pattern.

```
Binding: nexau.archs.tool.builtin.file_tools:glob
Parameters:
  - pattern (string, required): Glob pattern (e.g., "**/*.py")
  - path (string, optional): Base directory
```

### list_directory

Lists directory contents.

```
Binding: nexau.archs.tool.builtin.file_tools:list_directory
Parameters:
  - path (string, required): Directory path
```

### read_many_files

Reads multiple files at once.

```
Binding: nexau.archs.tool.builtin.file_tools:read_many_files
Parameters:
  - file_paths (array of strings, required): List of file paths
```

## Web Tools

Module: `nexau.archs.tool.builtin.web_tools`

### google_web_search

Searches the web using Google (requires SERPER_API_KEY).

```
Binding: nexau.archs.tool.builtin.web_tools:google_web_search
Parameters:
  - query (string, required): Search query
```

### web_fetch

Fetches and parses content from a URL.

```
Binding: nexau.archs.tool.builtin.web_tools:web_fetch
Parameters:
  - url (string, required): URL to fetch
  - prompt (string, optional): Extraction prompt
```

## Shell Tools

Module: `nexau.archs.tool.builtin.shell_tools`

### run_shell_command

Executes a shell command via `bash -c`.

```
Binding: nexau.archs.tool.builtin.shell_tools:run_shell_command
Parameters:
  - command (string, required): Bash command to execute
  - description (string, optional): Brief description of the command
  - is_background (boolean, optional): Run in background (default: false)
  - dir_path (string, optional): Working directory
```

Note: For synchronous-only execution (no background support), use the same binding but with a tool YAML that omits the `is_background` parameter.

## Session Tools

Module: `nexau.archs.tool.builtin.session_tools`

### write_todos

Writes/updates a todo list for task tracking.

```
Binding: nexau.archs.tool.builtin.session_tools:write_todos
Parameters:
  - todos (array, required): List of todo items with content and status
```

### complete_task

Marks the current task as completed and returns a result.

```
Binding: nexau.archs.tool.builtin.session_tools:complete_task
Parameters:
  - result (string, required): Task completion result/summary
```

### save_memory

Persists a key-value pair to agent memory.

```
Binding: nexau.archs.tool.builtin.session_tools:save_memory
Parameters:
  - key (string, required): Memory key
  - value (string, required): Memory value
```

### ask_user

Asks the user a question and waits for a response.

```
Binding: nexau.archs.tool.builtin.session_tools:ask_user
Parameters:
  - question (string, required): Question to ask the user
```

## Background Task Tools

### background_task_manage_tool

Manages background shell tasks (start, check status, kill).

```
Binding: nexau.archs.tool.builtin:background_task_manage_tool
```

## Common Tool Sets

### Minimal (file read/write only)

```yaml
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
  - name: write_file
    yaml_path: ./tools/write_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:write_file
```

### Standard (file + search + shell)

```yaml
tools:
  - name: read_file
    yaml_path: ./tools/read_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:read_file
  - name: write_file
    yaml_path: ./tools/write_file.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:write_file
  - name: replace
    yaml_path: ./tools/replace.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:replace
  - name: search_file_content
    yaml_path: ./tools/search_file_content.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:search_file_content
  - name: glob
    yaml_path: ./tools/Glob.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:glob
  - name: list_directory
    yaml_path: ./tools/list_directory.tool.yaml
    binding: nexau.archs.tool.builtin.file_tools:list_directory
  - name: run_shell_command
    yaml_path: ./tools/run_shell_command.tool.yaml
    binding: nexau.archs.tool.builtin.shell_tools:run_shell_command
```

### Full (all builtin tools)

Add web tools, session tools, and background task tools to the standard set. See `examples/code_agent/code_agent.yaml` for a complete example.
