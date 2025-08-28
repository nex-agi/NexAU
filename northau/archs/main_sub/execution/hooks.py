"""Hook interfaces and utilities for agent execution."""

from northau.archs.main_sub.execution.parse_structures import ParsedResponse
from typing import Protocol
from dataclasses import dataclass
from .parse_structures import ParsedResponse


@dataclass
class AfterModelHookInput:
    """Input data passed to after_model_hooks.
    
    This class encapsulates all the information that hooks receive:
    - original_response: The raw response from the LLM
    - parsed_response: The parsed structure containing tool/agent calls
    - messages: The current conversation history
    """
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
    def with_modifications(cls, parsed_response: ParsedResponse | None = None, 
                          messages: list[dict[str, str]] | None = None) -> 'HookResult':
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


def create_logging_hook(logger_name: str = "after_model_hook") -> AfterModelHook:
    """Create a simple logging hook for debugging purposes.
    
    Args:
        logger_name: Name for the logger
        
    Returns:
        A hook that logs the parsed response details and message history
    """
    import logging
    logger = logging.getLogger(logger_name)
    
    def logging_hook(hook_input: AfterModelHookInput) -> HookResult:
        logger.info(f"🎣 ===== AFTER MODEL HOOK TRIGGERED =====")
        logger.info(f"🎣 Response length: {len(hook_input.original_response)} characters")
        logger.info(f"🎣 Response summary: {hook_input.parsed_response.get_call_summary()}")
        logger.info(f"🎣 Tool calls: {len(hook_input.parsed_response.tool_calls)}")
        logger.info(f"🎣 Sub-agent calls: {len(hook_input.parsed_response.sub_agent_calls)}")
        logger.info(f"🎣 Batch agent calls: {len(hook_input.parsed_response.batch_agent_calls)}")
        logger.info(f"🎣 Is parallel tools: {hook_input.parsed_response.is_parallel_tools}")
        logger.info(f"🎣 Is parallel sub-agents: {hook_input.parsed_response.is_parallel_sub_agents}")
        logger.info(f"🎣 Message history length: {len(hook_input.messages)} messages")
        
        # Log details of each call
        for i, tool_call in enumerate(hook_input.parsed_response.tool_calls):
            logger.info(f"🎣 Tool call {i+1}: {tool_call.tool_name} with {len(tool_call.parameters)} parameters")
        
        for i, sub_agent_call in enumerate(hook_input.parsed_response.sub_agent_calls):
            logger.info(f"🎣 Sub-agent call {i+1}: {sub_agent_call.agent_name}")
        
        for i, batch_call in enumerate(hook_input.parsed_response.batch_agent_calls):
            logger.info(f"🎣 Batch call {i+1}: {batch_call.agent_name} on {batch_call.file_path}")
        
        # Log recent message history for context
        for i, msg in enumerate(hook_input.messages[-3:]):  # Show last 3 messages
            logger.info(f"🎣 Recent message {i+1}: {msg['role']} -> {msg['content'][:100]}...")
        
        logger.info(f"🎣 ===== END AFTER MODEL HOOK =====")
        
        # Return no changes
        return HookResult.no_changes()
    
    return logging_hook



def create_remaining_reminder_hook(logger_name: str = "after_model_hook") -> AfterModelHook:
    """Create a simple logging hook for debugging purposes.
    
    Args:
        logger_name: Name for the logger
        
    Returns:
        A hook that logs the parsed response details and message history
    """
    import logging
    logger = logging.getLogger(logger_name)
    
    def remaining_reminder_hook(hook_input: AfterModelHookInput) -> HookResult:
        logger.info(f"🎣 Remaining iterations: {hook_input.max_iterations - hook_input.current_iteration}")
        hook_input.messages.append({
            "role": "user",
            "content": f"Remaining iterations: {hook_input.max_iterations - hook_input.current_iteration}"
        })
        
        # Return no changes
        return HookResult.with_modifications(messages=hook_input.messages)
    
    return remaining_reminder_hook


def create_filter_hook(allowed_tools: set[str] | None = None, 
                      allowed_agents: set[str] | None = None) -> AfterModelHook:
    """Create a hook that filters out disallowed tools or agents.
    
    Args:
        allowed_tools: Set of allowed tool names (None allows all)
        allowed_agents: Set of allowed agent names (None allows all)
        
    Returns:
        A hook that filters the parsed response
    """
    import logging
    logger = logging.getLogger("filter_hook")
    
    def filter_hook(hook_input: AfterModelHookInput) -> HookResult:
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
                filtered_count = original_count - len(parsed_response.tool_calls)
                logger.warning(f"🎣 Filtered out {filtered_count} disallowed tool calls")
                modified = True
        
        # Filter sub-agent calls
        if allowed_agents is not None:
            original_count = len(parsed_response.sub_agent_calls)
            parsed_response.sub_agent_calls = [
                call for call in parsed_response.sub_agent_calls 
                if call.agent_name in allowed_agents
            ]
            if len(parsed_response.sub_agent_calls) != original_count:
                filtered_count = original_count - len(parsed_response.sub_agent_calls)
                logger.warning(f"🎣 Filtered out {filtered_count} disallowed sub-agent calls")
                modified = True
        
        # Filter batch agent calls
        if allowed_agents is not None:
            original_count = len(parsed_response.batch_agent_calls)
            parsed_response.batch_agent_calls = [
                call for call in parsed_response.batch_agent_calls 
                if call.agent_name in allowed_agents
            ]
            if len(parsed_response.batch_agent_calls) != original_count:
                filtered_count = original_count - len(parsed_response.batch_agent_calls)
                logger.warning(f"🎣 Filtered out {filtered_count} disallowed batch agent calls")
                modified = True
        
        # Return modified response if changes were made
        if modified:
            return HookResult.with_modifications(parsed_response=parsed_response)
        else:
            return HookResult.no_changes()
    
    return filter_hook



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
    
    def execute_hooks(self, hook_input: AfterModelHookInput) -> tuple[ParsedResponse, list[dict[str, str]]]:
        """Execute all hooks in sequence.
        
        Args:
            original_response: The original response from the LLM
            parsed_response: The initial parsed structure
            messages: The current conversation history
            
        Returns:
            Tuple of (final_parsed_response, final_messages) after all hooks have processed them
        """
        import logging
        logger = logging.getLogger(__name__)
        
        current_parsed = hook_input.parsed_response
        current_messages = hook_input.messages
        
        for i, hook in enumerate(self.hooks):
            try:
                logger.info(f"🎣 Executing hook {i+1}/{len(self.hooks)}")
                
                # Create input for the hook
                hook_input.parsed_response = current_parsed
                hook_input.messages = current_messages
                
                result = hook(hook_input)
                
                # Handle HookResult
                if result.has_modifications():
                    if result.parsed_response is not None:
                        current_parsed = result.parsed_response
                        logger.info(f"🎣 Hook {i+1} modified the parsed response")
                    
                    if result.messages is not None:
                        current_messages = result.messages
                        logger.info(f"🎣 Hook {i+1} modified the message history")
                else:
                    logger.info(f"🎣 Hook {i+1} made no modifications")
                    
            except Exception as e:
                logger.warning(f"⚠️ Hook {i+1} failed: {e}")
                # Continue with other hooks even if one fails
                continue
        
        return current_parsed, current_messages
    
    def __bool__(self) -> bool:
        """Check if there are any hooks."""
        return len(self.hooks) > 0
    
    def __len__(self) -> int:
        """Get the number of hooks."""
        return len(self.hooks)