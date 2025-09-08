"""Hook interfaces and utilities for agent execution."""
from dataclasses import dataclass
from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

from .parse_structures import ParsedResponse

if TYPE_CHECKING:
    from ..agent_state import AgentState


@dataclass
class AfterModelHookInput:
    """Input data passed to after_model_hooks.

    This class encapsulates all the information that hooks receive:
    - agent_state: The AgentState containing agent context and global storage
    - original_response: The raw response from the LLM
    - parsed_response: The parsed structure containing tool/agent calls
    - messages: The current conversation history
    """
    agent_state: 'AgentState'
    max_iterations: int
    current_iteration: int
    original_response: str
    parsed_response: ParsedResponse | None
    messages: list[dict[str, str]]


@dataclass
class HookResult:
    """Result returned by after_model_hooks.

    This class encapsulates the modifications that a hook can make:
    - parsed_response: Modified ParsedResponse, or None if no changes
    - messages: Modified conversation history, or None if no changes

    If both fields are None, it indicates the hook made no modifications.
    """
    parsed_response: ParsedResponse | None = None
    messages: list[dict[str, str]] | None = None

    def has_modifications(self) -> bool:
        """Check if this result contains any modifications."""
        return self.parsed_response is not None or self.messages is not None

    @classmethod
    def no_changes(cls) -> 'HookResult':
        """Create a HookResult indicating no modifications."""
        return cls(parsed_response=None, messages=None)

    @classmethod
    def with_modifications(
        cls, parsed_response: ParsedResponse | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> 'HookResult':
        """Create a HookResult with specified modifications.

        Args:
            parsed_response: Modified ParsedResponse, or None if no changes
            messages: Modified message history, or None if no changes

        Returns:
            HookResult with the specified modifications
        """
        return cls(parsed_response=parsed_response, messages=messages)


class AfterModelHook(Protocol):
    """Protocol for after_model_hook implementations.

    This hook is called after the LLM response is parsed but before execution.
    It receives an AfterModelHookInput containing all the relevant data, allowing
    for inspection, modification, or additional processing.
    """

    def __call__(self, hook_input: AfterModelHookInput) -> HookResult:
        """Process the LLM response, parsed calls, and conversation history.

        Args:
            hook_input: AfterModelHookInput containing:
                - agent_state: The AgentState containing agent context and global storage
                - original_response: The original response from the LLM
                - parsed_response: The parsed structure containing all tool/agent calls
                - messages: The current conversation history (list of message dicts)

        Returns:
            HookResult containing any modifications:
            - HookResult.parsed_response: Modified parsed response, or None to use original
            - HookResult.messages: Modified message history, or None to use original

            Use the class methods for convenient creation:
            - HookResult.no_changes(): No modifications
            - HookResult.with_modifications(parsed_response=..., messages=...): Any combination of modifications
        """
        ...


@dataclass
class AfterToolHookInput:
    """Input data passed to after_tool_hooks.

    This class encapsulates all the information that tool hooks receive:
    - agent_state: The AgentState containing agent context and global storage
    - tool_name: The name of the tool that was executed
    - tool_input: The parameters passed to the tool
    - tool_output: The result returned by the tool
    """
    agent_state: 'AgentState'
    tool_name: str
    tool_input: dict[str, Any]
    tool_output: Any


@dataclass
class AfterToolHookResult:
    """Result returned by after_tool_hooks.

    This class encapsulates the modifications that a tool hook can make:
    - tool_output: Modified tool output, or None if no changes

    If tool_output is None, it indicates the hook made no modifications.
    """
    tool_output: Any = None

    def has_modifications(self) -> bool:
        """Check if this result contains any modifications."""
        return self.tool_output is not None

    @classmethod
    def no_changes(cls) -> 'AfterToolHookResult':
        """Create an AfterToolHookResult indicating no modifications."""
        return cls(tool_output=None)

    @classmethod
    def with_modifications(cls, tool_output: Any) -> 'AfterToolHookResult':
        """Create an AfterToolHookResult with modified tool output.

        Args:
            tool_output: Modified tool output

        Returns:
            AfterToolHookResult with the specified modifications
        """
        return cls(tool_output=tool_output)


class AfterToolHook(Protocol):
    """Protocol for after_tool_hook implementations.

    This hook is called after a tool is executed but before its result is processed.
    It receives an AfterToolHookInput containing the tool execution details, allowing
    for inspection, modification, or additional processing of tool results.
    """

    def __call__(self, hook_input: AfterToolHookInput) -> AfterToolHookResult:
        """Process the tool execution result.

        Args:
            hook_input: AfterToolHookInput containing:
                - agent_state: The AgentState containing agent context and global storage
                - tool_name: The name of the executed tool
                - tool_input: The parameters that were passed to the tool
                - tool_output: The result returned by the tool

        Returns:
            AfterToolHookResult containing any modifications:
            - AfterToolHookResult.tool_output: Modified tool output, or None to use original

            Use the class methods for convenient creation:
            - AfterToolHookResult.no_changes(): No modifications
            - AfterToolHookResult.with_modifications(tool_output=...): Modified output
        """
        ...


def create_logging_hook(logger_name: str = 'after_model_hook') -> AfterModelHook:
    """Create a simple logging hook for debugging purposes.

    Args:
        logger_name: Name for the logger

    Returns:
        A hook that logs the parsed response details and message history
    """
    import logging
    logger = logging.getLogger(logger_name)

    def logging_hook(hook_input: AfterModelHookInput) -> HookResult:
        logger.info('🎣 ===== AFTER MODEL HOOK TRIGGERED =====')
        logger.info(f"🎣 Agent name: {hook_input.agent_state.agent_name}")
        logger.info(f"🎣 Agent id: {hook_input.agent_state.agent_id}")
        logger.info(
            f"🎣 Response length: {len(hook_input.original_response)} characters",
        )

        if hook_input.parsed_response is not None:
            logger.info(
                f"🎣 Response summary: {hook_input.parsed_response.get_call_summary()}",
            )
            logger.info(
                f"🎣 Tool calls: {len(hook_input.parsed_response.tool_calls)}",
            )
            logger.info(
                f"🎣 Sub-agent calls: {len(hook_input.parsed_response.sub_agent_calls)}",
            )
            logger.info(
                f"🎣 Batch agent calls: {len(hook_input.parsed_response.batch_agent_calls)}",
            )
            logger.info(
                f"🎣 Is parallel tools: {hook_input.parsed_response.is_parallel_tools}",
            )
            logger.info(
                f"🎣 Is parallel sub-agents: {hook_input.parsed_response.is_parallel_sub_agents}",
            )

            # Log details of each call
            for i, tool_call in enumerate(hook_input.parsed_response.tool_calls):
                logger.info(
                    f"🎣 Tool call {i + 1}: {tool_call.tool_name} with {len(tool_call.parameters)} parameters",
                )

            for i, sub_agent_call in enumerate(hook_input.parsed_response.sub_agent_calls):
                logger.info(
                    f"🎣 Sub-agent call {i + 1}: {sub_agent_call.agent_name}",
                )

            for i, batch_call in enumerate(hook_input.parsed_response.batch_agent_calls):
                logger.info(
                    f"🎣 Batch call {i + 1}: {batch_call.agent_name} on {batch_call.file_path}",
                )
        else:
            logger.info('🎣 No parsed response available')

        logger.info(
            f"🎣 Message history length: {len(hook_input.messages)} messages",
        )

        # Log recent message history for context
        # Show last 3 messages
        for i, msg in enumerate(hook_input.messages[-3:]):
            logger.info(
                f"🎣 Recent message {i + 1}: {msg['role']} -> {msg['content'][:100]}...",
            )

        logger.info('🎣 ===== END AFTER MODEL HOOK =====')

        # Return no changes
        return HookResult.no_changes()

    return logging_hook


def create_remaining_reminder_hook(logger_name: str = 'after_model_hook') -> AfterModelHook:
    """Create a simple logging hook for debugging purposes.

    Args:
        logger_name: Name for the logger

    Returns:
        A hook that logs the parsed response details and message history
    """
    import logging
    logger = logging.getLogger(logger_name)

    def remaining_reminder_hook(hook_input: AfterModelHookInput) -> HookResult:
        logger.info(
            f"🎣 Remaining iterations: {hook_input.max_iterations - hook_input.current_iteration}",
        )
        hook_input.messages.append({
            'role': 'user',
            'content': f"Remaining iterations: {hook_input.max_iterations - hook_input.current_iteration}",
        })

        # Return no changes
        return HookResult.with_modifications(messages=hook_input.messages)

    return remaining_reminder_hook


def create_tool_after_approve_hook(tool_name: str) -> AfterModelHook:
    """Ask for approval before running a tool.

    Args:
        tool_name: The name of the tool to run
    """
    def tool_after_approve_hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.parsed_response and hook_input.parsed_response.tool_calls:
            tool_call_to_remove = []
            for tool_call in hook_input.parsed_response.tool_calls:
                if tool_call.tool_name == tool_name:
                    print(f"🎣 Tool call: {tool_call.tool_name}")
                    print(f"🎣 Tool call parameters: {tool_call.parameters}")
                    # CLI option to approve or reject the tool call
                    while True:
                        approve = input(
                            f"Approve running {tool_name}? (y/n): ",
                        )
                        if approve not in ['y', 'n']:
                            print("🎣 Invalid input. Please enter 'y' or 'n'.")
                            continue
                        else:
                            if approve == 'n':
                                tool_call_to_remove.append(tool_call)
                            break

            if tool_call_to_remove:
                hook_input.parsed_response.tool_calls = [
                    call for call in hook_input.parsed_response.tool_calls
                    if call not in tool_call_to_remove
                ]
            return HookResult.with_modifications(parsed_response=hook_input.parsed_response)
        return HookResult.no_changes()

    return tool_after_approve_hook


def create_filter_hook(
    allowed_tools: set[str] | None = None,
    allowed_agents: set[str] | None = None,
) -> AfterModelHook:
    """Create a hook that filters out disallowed tools or agents.

    Args:
        allowed_tools: Set of allowed tool names (None allows all)
        allowed_agents: Set of allowed agent names (None allows all)

    Returns:
        A hook that filters the parsed response
    """
    import logging
    logger = logging.getLogger('filter_hook')

    def filter_hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.parsed_response is None:
            return HookResult.no_changes()

        parsed_response: ParsedResponse = hook_input.parsed_response
        modified = False

        # Filter tool calls
        if allowed_tools is not None:
            original_count = len(parsed_response.tool_calls)
            parsed_response.tool_calls = [
                call for call in parsed_response.tool_calls
                if call.tool_name in allowed_tools
            ]
            if len(parsed_response.tool_calls) != original_count:
                filtered_count = original_count - \
                    len(parsed_response.tool_calls)
                logger.warning(
                    f"🎣 Filtered out {filtered_count} disallowed tool calls",
                )
                modified = True

        # Filter sub-agent calls
        if allowed_agents is not None:
            original_count = len(parsed_response.sub_agent_calls)
            parsed_response.sub_agent_calls = [
                call for call in parsed_response.sub_agent_calls
                if call.agent_name in allowed_agents
            ]
            if len(parsed_response.sub_agent_calls) != original_count:
                filtered_count = original_count - \
                    len(parsed_response.sub_agent_calls)
                logger.warning(
                    f"🎣 Filtered out {filtered_count} disallowed sub-agent calls",
                )
                modified = True

        # Filter batch agent calls
        if allowed_agents is not None:
            original_count = len(parsed_response.batch_agent_calls)
            parsed_response.batch_agent_calls = [
                call for call in parsed_response.batch_agent_calls
                if call.agent_name in allowed_agents
            ]
            if len(parsed_response.batch_agent_calls) != original_count:
                filtered_count = original_count - \
                    len(parsed_response.batch_agent_calls)
                logger.warning(
                    f"🎣 Filtered out {filtered_count} disallowed batch agent calls",
                )
                modified = True

        # Return modified response if changes were made
        if modified:
            return HookResult.with_modifications(parsed_response=parsed_response)
        else:
            return HookResult.no_changes()

    return filter_hook


def create_tool_logging_hook(logger_name: str = 'after_tool_hook') -> AfterToolHook:
    """Create a simple logging hook for tool execution debugging.

    Args:
        logger_name: Name for the logger

    Returns:
        A hook that logs tool execution details
    """
    import logging
    logger = logging.getLogger(logger_name)

    def tool_logging_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        logger.info('🔧 ===== AFTER TOOL HOOK TRIGGERED =====')
        logger.info(f"🔧 Agent name: {hook_input.agent_state.agent_name}")
        logger.info(f"🔧 Agent id: {hook_input.agent_state.agent_id}")
        logger.info(f"🔧 Tool name: {hook_input.tool_name}")
        logger.info(f"🔧 Tool input: {hook_input.tool_input}")
        logger.info(f"🔧 Tool output type: {type(hook_input.tool_output)}")

        # Log tool output (truncated if too long)
        output_str = str(hook_input.tool_output)
        if len(output_str) > 500:
            logger.info(f"🔧 Tool output (truncated): {output_str[:500]}...")
        else:
            logger.info(f"🔧 Tool output: {output_str}")

        logger.info('🔧 ===== END AFTER TOOL HOOK =====')

        # Return no changes
        return AfterToolHookResult.no_changes()

    return tool_logging_hook


def create_tool_output_filter_hook(filter_keys: set[str]) -> AfterToolHook:
    """Create a hook that filters sensitive keys from tool outputs.

    Args:
        filter_keys: Set of keys to filter from dict outputs

    Returns:
        A hook that removes specified keys from tool outputs
    """
    import logging
    logger = logging.getLogger('tool_filter_hook')

    def tool_filter_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        if isinstance(hook_input.tool_output, dict):
            filtered_output = {
                k: v for k, v in hook_input.tool_output.items()
                if k not in filter_keys
            }

            if len(filtered_output) != len(hook_input.tool_output):
                filtered_count = len(
                    hook_input.tool_output,
                ) - len(filtered_output)
                logger.info(
                    f"🔧 Filtered {filtered_count} sensitive keys from {hook_input.tool_name} output",
                )
                return AfterToolHookResult.with_modifications(tool_output=filtered_output)

        return AfterToolHookResult.no_changes()

    return tool_filter_hook


def create_tool_result_transformer_hook(transform_func) -> AfterToolHook:
    """Create a hook that transforms tool outputs using a custom function.

    Args:
        transform_func: Function that takes (tool_name, tool_input, tool_output) and returns modified output

    Returns:
        A hook that applies the transformation function
    """
    def tool_transformer_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        try:
            transformed_output = transform_func(
                hook_input.agent_state.agent_name,
                hook_input.agent_state.agent_id,
                hook_input.tool_name,
                hook_input.tool_input,
                hook_input.tool_output,
            )

            # Only return modifications if the output actually changed
            if transformed_output != hook_input.tool_output:
                return AfterToolHookResult.with_modifications(tool_output=transformed_output)

        except Exception as e:
            import logging
            logger = logging.getLogger('tool_transformer_hook')
            logger.warning(
                f"⚠️ Tool transformation failed for {hook_input.tool_name}: {e}",
            )

        return AfterToolHookResult.no_changes()

    return tool_transformer_hook


class ToolHookManager:
    """Manages and executes multiple after tool hooks in sequence."""

    def __init__(self, hooks: list[AfterToolHook] | None = None):
        """Initialize tool hook manager.

        Args:
            hooks: List of tool hooks to execute
        """
        self.hooks: list[AfterToolHook] = hooks or []

    def add_hook(self, hook: AfterToolHook):
        """Add a tool hook to the manager.

        Args:
            hook: Tool hook to add
        """
        self.hooks.append(hook)

    def remove_hook(self, hook: AfterToolHook):
        """Remove a tool hook from the manager.

        Args:
            hook: Tool hook to remove
        """
        if hook in self.hooks:
            self.hooks.remove(hook)

    def execute_hooks(self, hook_input: AfterToolHookInput) -> Any:
        """Execute all tool hooks in sequence.

        Args:
            hook_input: AfterToolHookInput containing tool execution details

        Returns:
            Final tool output after all hooks have processed it
        """
        import logging
        logger = logging.getLogger(__name__)

        current_output = hook_input.tool_output

        for i, hook in enumerate(self.hooks):
            try:
                logger.info(f"🔧 Executing tool hook {i + 1}/{len(self.hooks)}")

                # Create input for the hook with current output
                hook_input.tool_output = current_output

                result = hook(hook_input)

                # Handle AfterToolHookResult
                if result.has_modifications():
                    current_output = result.tool_output
                    logger.info(
                        f"🔧 Tool hook {i + 1} modified the tool output",
                    )
                else:
                    logger.info(f"🔧 Tool hook {i + 1} made no modifications")

            except Exception as e:
                logger.warning(f"⚠️ Tool hook {i + 1} failed: {e}")
                # Continue with other hooks even if one fails
                continue

        return current_output

    def __bool__(self) -> bool:
        """Check if there are any tool hooks."""
        return len(self.hooks) > 0

    def __len__(self) -> int:
        """Get the number of tool hooks."""
        return len(self.hooks)


class HookManager:
    """Manages and executes multiple after model hooks in sequence."""

    def __init__(self, hooks: list[AfterModelHook] | None = None):
        """Initialize hook manager.

        Args:
            hooks: List of hooks to execute
        """
        self.hooks: list[AfterModelHook] = hooks or []

    def add_hook(self, hook: AfterModelHook):
        """Add a hook to the manager.

        Args:
            hook: Hook to add
        """
        self.hooks.append(hook)

    def remove_hook(self, hook: AfterModelHook):
        """Remove a hook from the manager.

        Args:
            hook: Hook to remove
        """
        if hook in self.hooks:
            self.hooks.remove(hook)

    def execute_hooks(self, hook_input: AfterModelHookInput) -> tuple[ParsedResponse | None, list[dict[str, str]]]:
        """Execute all hooks in sequence.

        Args:
            hook_input: AfterModelHookInput containing all hook input data

        Returns:
            Tuple of (final_parsed_response, final_messages) after all hooks have processed them
        """
        import logging
        logger = logging.getLogger(__name__)

        current_parsed = hook_input.parsed_response
        current_messages = hook_input.messages

        for i, hook in enumerate(self.hooks):
            try:
                logger.info(f"🎣 Executing hook {i + 1}/{len(self.hooks)}")

                # Create input for the hook
                hook_input.parsed_response = current_parsed
                hook_input.messages = current_messages

                result = hook(hook_input)

                # Handle HookResult
                if result.has_modifications():
                    if result.parsed_response is not None:
                        current_parsed = result.parsed_response
                        logger.info(
                            f"🎣 Hook {i + 1} modified the parsed response",
                        )

                    if result.messages is not None:
                        current_messages = result.messages
                        logger.info(
                            f"🎣 Hook {i + 1} modified the message history",
                        )
                else:
                    logger.info(f"🎣 Hook {i + 1} made no modifications")

            except Exception as e:
                logger.warning(f"⚠️ Hook {i + 1} failed: {e}")
                # Continue with other hooks even if one fails
                continue

        return current_parsed, current_messages

    def __bool__(self) -> bool:
        """Check if there are any hooks."""
        return len(self.hooks) > 0

    def __len__(self) -> int:
        """Get the number of hooks."""
        return len(self.hooks)
