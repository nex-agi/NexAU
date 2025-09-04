"""LLM response generation with iteration management."""

import logging
import re
from typing import Any, Callable

from ..utils.xml_utils import XMLUtils
from ..tracing.tracer import Tracer
from .hooks import AfterModelHookInput

logger = logging.getLogger(__name__)




class ResponseGenerator:
    """Handles LLM response generation with iteration control."""
    
    def __init__(self, agent_name: str, openai_client: Any, llm_config: Any, max_iterations: int = 100, 
                 max_context_tokens: int = 128000, retry_attempts: int = 5, global_storage: Any = None,
                 custom_llm_generator: Callable[[Any, dict[str, Any]], Any] | None = None):
        """Initialize response generator.
        
        Args:
            agent_name: Name of the agent
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            max_iterations: Maximum number of iterations
            max_context_tokens: Maximum context token limit
            retry_attempts: Number of retry attempts for API calls
            global_storage: Optional GlobalStorage instance for hooks
            custom_llm_generator: Optional custom LLM generator function that takes (openai_client, kwargs) and returns a response
        """
        self.agent_name = agent_name
        self.openai_client = openai_client
        self.llm_config = llm_config
        self.max_iterations = max_iterations
        self.max_context_tokens = max_context_tokens
        self.retry_attempts = retry_attempts
        self.global_storage = global_storage
        self.custom_llm_generator = custom_llm_generator
        self.queued_messages = []
    
    def enqueue_message(self, message: dict[str, str]) -> None:
        """Enqueue a message to be added to the history."""
        self.queued_messages.append(message)
    
    def generate_response(self, history: list[dict[str, str]], 
                         token_counter: Any, xml_processor: Any, tracer: Tracer | None = None) -> tuple[str, list[dict[str, str]]]:
        """Generate response using OpenAI API or custom LLM generator with XML-based tool and sub-agent calls.
        
        Args:
            history: Complete conversation history including system prompt and user message
            token_counter: Token counting function
            xml_processor: XML call processor
            tracer: Optional tracer for logging
            
        Returns:
            Tuple of (final_response, updated_messages_history)
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI client is not available. Please check your API configuration.")
        
        try:
            # Use history directly as the single source of truth
            # History should already contain system prompt and all messages
            messages = history.copy()
            
            # Loop until no more tool calls or sub-agent calls are made
            iteration = 0
            final_response = ""
            
            logger.info(f"🔄 Starting iterative execution loop for agent '{self.agent_name}'")
            
            while iteration < self.max_iterations:
                logger.info(f"🔄 Iteration {iteration + 1}/{self.max_iterations} for agent '{self.agent_name}'")
                
                # Call OpenAI API with LLM config parameters
                api_params = self.llm_config.to_openai_params()
                api_params['messages'] = messages
                
                logger.info(f"🔄 Queued messages: {self.queued_messages}")
                
                if self.queued_messages:
                    messages.extend(self.queued_messages)
                    self.queued_messages = []
                
                # Count current prompt tokens
                current_prompt_tokens = token_counter(messages)
                
                # Check if prompt exceeds max context tokens - force stop if so
                if current_prompt_tokens > self.max_context_tokens:
                    logger.error(f"❌ Prompt tokens ({current_prompt_tokens}) exceed max_context_tokens ({self.max_context_tokens}). Stopping execution.")
                    final_response += f"\\n\\n[Error: Prompt too long ({current_prompt_tokens} tokens) exceeds maximum context ({self.max_context_tokens} tokens). Execution stopped.]"
                    break
                
                # Calculate max_tokens dynamically based on available budget
                available_tokens = self.max_context_tokens - current_prompt_tokens
                
                # Get desired max_tokens from LLM config or use reasonable default
                desired_max_tokens = api_params.get('max_tokens', 4096)
                calculated_max_tokens = min(desired_max_tokens, available_tokens)
                
                # Ensure we have at least some tokens for response
                if calculated_max_tokens < 50:
                    logger.error(f"❌ Insufficient tokens for response ({calculated_max_tokens}). Stopping execution.")
                    final_response += f"\\n\\n[Error: Insufficient tokens for response ({calculated_max_tokens} tokens). Context too full.]"
                    break
                
                # Set max_tokens based on calculation
                api_params['max_tokens'] = calculated_max_tokens
                
                logger.info(f"🔢 Token usage: prompt={current_prompt_tokens}, max_tokens={api_params['max_tokens']}, available={available_tokens}")
                
                # Add stop sequences for XML closing tags to prevent malformed XML
                xml_stop_sequences = [
                    "</tool_use>",
                    "</sub_agent>", 
                    "</use_parallel_tool_calls>",
                    "</use_parallel_sub_agents>",
                    "</use_batch_agent>"
                ]
                
                # Merge with existing stop sequences if any
                existing_stop = api_params.get('stop', [])
                if isinstance(existing_stop, str):
                    existing_stop = [existing_stop]
                elif existing_stop is None:
                    existing_stop = []
                
                api_params['stop'] = existing_stop + xml_stop_sequences
                
                # Debug logging for LLM messages
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Request Messages for agent '{self.agent_name}':")
                    for i, msg in enumerate(messages):
                        logger.info(f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content']}")
                
                logger.info(f"🧠 Calling LLM for agent '{self.agent_name}' with {api_params['max_tokens']} max tokens...")
                
                # Log LLM request to trace if enabled
                if tracer:
                    tracer.add_llm_request(iteration + 1, api_params)
                
                logger.info(f"🧠 Calling LLM for agent '{self.agent_name}'...")
                response = self._call_openai_with_retry(**api_params)
                if response and hasattr(response, 'choices') and response.choices:
                    assistant_response = response.choices[0].message.content
                else:
                    raise RuntimeError("Invalid response from OpenAI API")
                
                # Log LLM response to trace if enabled
                if tracer:
                    tracer.add_llm_response(iteration + 1, response)
                
                # Add back XML closing tags if they were removed by stop sequences
                assistant_response = XMLUtils.restore_closing_tags(assistant_response)
                
                # Debug logging for LLM response
                if self.llm_config.debug:
                    logger.info(f"🐛 [DEBUG] LLM Response for agent '{self.agent_name}': {assistant_response}")
                
                logger.info(f"💬 LLM Response for agent '{self.agent_name}': {assistant_response}")
                
                # Store this as the latest response (potential final response)
                final_response = assistant_response
                
                # Check if response contains tool calls or sub-agent calls (including parallel formats and batch processing)
                has_tool_calls = bool(re.search(r'<tool_use>.*?</tool_use>', assistant_response, re.DOTALL))
                has_sub_agent_calls = bool(re.search(r'<sub_agent>.*?</sub_agent>', assistant_response, re.DOTALL))
                has_parallel_tool_calls = bool(re.search(r'<use_parallel_tool_calls>.*?</use_parallel_tool_calls>', assistant_response, re.DOTALL))
                has_parallel_sub_agents = bool(re.search(r'<use_parallel_sub_agents>.*?</use_parallel_sub_agents>', assistant_response, re.DOTALL))
                has_batch_agent = bool(re.search(r'<use_batch_agent>.*?</use_batch_agent>', assistant_response, re.DOTALL))
                
                logger.info(f"🔍 Analysis for agent '{self.agent_name}': tool_calls={has_tool_calls}, sub_agent_calls={has_sub_agent_calls}, parallel_tool_calls={has_parallel_tool_calls}, parallel_sub_agents={has_parallel_sub_agents}, batch_agent={has_batch_agent}")
                
                if not has_tool_calls and not has_sub_agent_calls and not has_parallel_tool_calls and not has_parallel_sub_agents and not has_batch_agent:
                    # No more commands to execute, return final response
                    logger.info(f"🏁 No more commands to execute, finishing agent '{self.agent_name}'")
                    break
                
                # Add the assistant's original response to conversation
                messages.append({"role": "assistant", "content": assistant_response})
                
                # Process tool calls and sub-agent calls
                logger.info(f"⚙️ Processing tool/sub-agent calls for agent '{self.agent_name}'...")
                after_model_hook_input = AfterModelHookInput(
                    max_iterations=self.max_iterations,
                    current_iteration=iteration,
                    original_response=assistant_response,
                    parsed_response=None,
                    messages=messages,
                    global_storage=self.global_storage
                )
                processed_response, should_stop, stop_tool_result, updated_messages = xml_processor(after_model_hook_input, tracer=tracer)
                
                # Update messages with any modifications from hooks
                messages = updated_messages
                
                # Check if a stop tool was executed
                if should_stop and len(self.queued_messages) == 0:
                    logger.info(f"🛑 Stop tool detected, returning stop tool result as final response")
                    # Return the stop tool result directly, formatted as JSON if it's not a string
                    if stop_tool_result is not None:
                        if isinstance(stop_tool_result, str):
                            return stop_tool_result, messages
                        else:
                            import json
                            return json.dumps(stop_tool_result, indent=4, ensure_ascii=False), messages
                    else:
                        # Fallback to the processed response if no specific result
                        return processed_response, messages
                
                # Extract just the tool results from processed_response
                tool_results = processed_response.replace(assistant_response, "").strip()
                
                # Add tool results as user feedback with iteration context
                remaining_iterations = self.max_iterations - iteration - 1
                iteration_hint = self._build_iteration_hint(iteration + 1, self.max_iterations, remaining_iterations)
                token_limit_hint = self._build_token_limit_hint(current_prompt_tokens, self.max_context_tokens, available_tokens, desired_max_tokens)
                
                if tool_results:
                    messages.append({
                        "role": "user", 
                        "content": f"Tool execution results:\n{tool_results}\n\n{iteration_hint}\n\n{token_limit_hint}"
                    })
                else:
                    messages.append({
                        "role": "user", 
                        "content": f"{iteration_hint}\n\n{token_limit_hint}"
                    })
                
                iteration += 1
            
            # Add note if max iterations reached
            if iteration >= self.max_iterations:
                final_response += "\\n\\n[Note: Maximum iteration limit reached]"
            
            return final_response, messages
            
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"Error calling OpenAI API: {e}") from e
    
    def _call_openai_with_retry(self, **kwargs: Any) -> Any:
        """Call OpenAI client or custom LLM generator with exponential backoff retry."""
        import time
        
        backoff = 1
        for i in range(self.retry_attempts):
            try:
                # Use custom LLM generator if provided, otherwise use OpenAI client
                if self.custom_llm_generator:
                    response = self.custom_llm_generator(self.openai_client, kwargs)
                else:
                    response = self.openai_client.chat.completions.create(**kwargs)
                
                response_content = response.choices[0].message.content
                stop = kwargs.get('stop', [])
                if stop:
                    for s in stop:
                        response_content = response_content.split(s)[0]
                        response_content = response_content.strip()
                        response.choices[0].message.content = response_content
                if response_content:
                    return response
                else:
                    raise Exception("No response content")
            except Exception as e:
                logger.error(f"❌ LLM call failed (attempt {i+1}/{self.retry_attempts}): {e}")
                if i == self.retry_attempts - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2
    
    def _build_iteration_hint(self, current_iteration: int, max_iterations: int, remaining_iterations: int) -> str:
        """Build a hint message for the LLM about iteration status."""
        if remaining_iterations <= 1:
            return (f"⚠️ WARNING: This is iteration {current_iteration}/{max_iterations}. "
                   f"You have only {remaining_iterations} iteration(s) remaining. "
                   f"Please provide a conclusive response and avoid making additional tool calls or sub-agent calls "
                   f"unless absolutely critical. Focus on summarizing your findings and providing final recommendations.")
        elif remaining_iterations <= 3:
            return (f"🔄 Iteration {current_iteration}/{max_iterations} - {remaining_iterations} iterations remaining. "
                   f"Please be mindful of the remaining steps and work towards a conclusion.")
        else:
            return (f"🔄 Iteration {current_iteration}/{max_iterations} - Continue your response if you have more to say, "
                   f"or if you need to make additional tool calls or sub-agent calls.")
    
    def _build_token_limit_hint(self, current_prompt_tokens: int, max_tokens: int, remaining_tokens: int, desired_max_tokens: int) -> str:
        """Build a hint message for the LLM about token limit."""
        if remaining_tokens < 3 * desired_max_tokens:
            return (f"⚠️ WARNING: Token usage is approaching the limit {current_prompt_tokens}/{max_tokens}."
                   f"You have only {remaining_tokens} tokens left."
                   f"Please be mindful of the token limit and avoid making additional tool calls or sub-agent calls "
                   f"unless absolutely critical. Focus on summarizing your findings and providing final recommendations.")
        else:
            return (f"🔄 Token Usage: {current_prompt_tokens}/{max_tokens} in the current prompt - {remaining_tokens} tokens left."
                    f"Continue your response if you have more to say, or if you need to make additional tool calls or sub-agent calls.")


def bypass_llm_generator(openai_client: Any, kwargs: dict[str, Any]) -> Any:
    """
    Custom LLM generator that does nothing.
    
    Args:
        openai_client: The OpenAI client instance (can be used or ignored)
        kwargs: The parameters that would be passed to openai_client.chat.completions.create()
        
    Returns:
        A response object with the same structure as OpenAI's response
        (must have .choices[0].message.content attribute)
    """
    print(f"🔧 Custom LLM Generator called with {len(kwargs.get('messages', []))} messages")
    
    try:
        # Call the original OpenAI API
        response = openai_client.chat.completions.create(**kwargs)

        return response
        
    except Exception as e:
        print(f"❌ Bypass LLM generator error: {e}")
        # You could implement custom fallback logic here
        raise