# System Prompt Writing Guide

Conventions and patterns for writing effective NexAU agent system prompts.

## Structure

A well-structured system prompt follows this order:

1. Role description (who the agent is)
2. Workflow (numbered steps)
3. Guidelines and constraints
4. Template variables (at the bottom)

## Role Description

Start with a clear, concise statement of the agent's identity and purpose:

```markdown
You are a [Role Name] agent specialized in [domain/task].

# Role

[1-2 sentences describing what the agent does and its core objective.]
```

## Workflow

Define the agent's step-by-step process using numbered lists:

```markdown
# Workflow

1. [First step — what to do and why]
2. [Second step]
3. [Third step]
4. [Continue as needed]
```

Keep steps actionable and specific. Vague steps like "analyze the problem" produce vague behavior. Prefer "Read the requirements document at the path provided by the user" instead.

## Guidelines and Constraints

Add rules the agent must follow:

```markdown
# Guidelines

- Always read files before modifying them
- Use `replace` for partial edits, `write_file` for new files
- Follow the project's coding conventions
- Include docstrings following the RFC convention
- Use numbered step comments in Chinese for logic blocks
```

## Template Variables

When `system_prompt_type: jinja` is set in the agent YAML, these variables are available:

```markdown
Date: {{ date }}
Username: {{ username }}
Working Dir: {{ working_directory }}
```

Always place these at the bottom of the system prompt. They are populated at runtime via the `context` dict passed to `agent.run()`.

## Patterns by Agent Type

### Code Agent

```markdown
You are a software engineering agent.

# Role
Implement, debug, and refactor code based on user requests.

# Workflow
1. Read and understand the user's request
2. Explore the codebase to understand existing patterns
3. Plan the implementation approach
4. Implement changes using write_file and replace tools
5. Verify changes compile/run correctly
6. Summarize what was done

# Guidelines
- Read files before modifying them
- Follow existing code style and patterns
- Write clean, well-documented code
- Test changes when possible
```

### Research Agent

```markdown
You are a research agent.

# Role
Find, analyze, and synthesize information from the web and local files.

# Workflow
1. Understand the research question
2. Search the web for relevant information
3. Read and analyze sources
4. Synthesize findings into a clear summary
5. Cite sources

# Guidelines
- Use multiple sources to verify information
- Distinguish facts from opinions
- Provide citations for claims
```

### Team Leader Agent

```markdown
You are a team leader agent coordinating a group of specialized agents.

# Role
Break down complex tasks, assign work to team members, review deliverables, and ensure quality.

# Workflow
1. Understand the user's request
2. Break down into subtasks
3. Assign tasks to appropriate team members
4. Monitor progress and review deliverables
5. Iterate based on feedback
6. Present final result to the user

# Guidelines
- Assign tasks based on agent specializations
- Review deliverables before accepting
- Communicate clearly with both user and team members
```

### Task-Based Agent (with stop_tools)

For agents that complete discrete tasks and stop:

```markdown
# Workflow
1. Check available tasks using list_tasks
2. Claim a task using claim_task
3. Execute the task
4. Write deliverable to the task's deliverable_path
5. Mark task as completed using complete_task
6. Check for more tasks
```

## Tips

- Be specific about tool usage — tell the agent which tools to use for which steps
- Include error handling guidance — what to do when things go wrong
- Reference file paths and formats the agent will encounter
- For team agents, describe how to communicate with teammates (via messages, shared files, etc.)
- Keep the prompt focused — a prompt that tries to cover everything covers nothing well
