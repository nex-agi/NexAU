"""Simple LLM API caller component."""
import logging
import time
from typing import Any
from typing import Callable

logger = logging.getLogger(__name__)


class LLMCaller:
    """Handles LLM API calls with retry logic."""

    def __init__(
        self, openai_client: Any, llm_config: Any, retry_attempts: int = 5,
        custom_llm_generator: Callable[
            [
            Any, dict[str, Any],
            ], Any,
        ] | None = None,
    ):
        """Initialize LLM caller.

        Args:
            openai_client: OpenAI client instance
            llm_config: LLM configuration
            retry_attempts: Number of retry attempts for API calls
            custom_llm_generator: Optional custom LLM generator function
        """
        self.openai_client = openai_client
        self.llm_config = llm_config
        self.retry_attempts = retry_attempts
        self.custom_llm_generator = custom_llm_generator

    def call_llm(self, messages: list[dict[str, str]], max_tokens: int, force_stop_reason: str | None = None) -> str:
        """Call LLM with the given messages and return response content.

        Args:
            messages: List of conversation messages
            max_tokens: Maximum tokens for the response

        Returns:
            The LLM response content as a string

        Raises:
            RuntimeError: If OpenAI client is not available or API call fails
        """
        if not self.openai_client:
            raise RuntimeError(
                'OpenAI client is not available. Please check your API configuration.',
            )

        # Prepare API parameters
        api_params = self.llm_config.to_openai_params()
        api_params['messages'] = messages
        api_params['max_tokens'] = max_tokens

        # Add XML stop sequences to prevent malformed XML
        xml_stop_sequences = [
            '</tool_use>',
            '</use_parallel_tool_calls>',
            '</use_batch_agent>',
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
            logger.info('🐛 [DEBUG] LLM Request Messages:')
            for i, msg in enumerate(messages):
                logger.info(
                    f"🐛 [DEBUG] Message {i}: {msg['role']} -> {msg['content']}",
                )

        logger.info(f"🧠 Calling LLM with {max_tokens} max tokens...")

        # Call LLM with retry
        response = self._call_with_retry(
            force_stop_reason=force_stop_reason, **api_params,
        )

        if response and hasattr(response, 'choices') and response.choices:
            assistant_response = response.choices[0].message.content
        else:
            raise RuntimeError('Invalid response from OpenAI API')

        # Add back XML closing tags if they were removed by stop sequences
        from ..utils.xml_utils import XMLUtils
        assistant_response = XMLUtils.restore_closing_tags(assistant_response)

        # Debug logging for LLM response
        if self.llm_config.debug:
            logger.info(f"🐛 [DEBUG] LLM Response: {assistant_response}")

        logger.info(f"💬 LLM Response: {assistant_response}")

        return assistant_response

    def _call_with_retry(self, force_stop_reason: str | None = None, **kwargs: Any) -> Any:
        """Call OpenAI client or custom LLM generator with exponential backoff retry."""
        if force_stop_reason:
            logger.info(
                f"🛑 LLM call forced to stop due to {force_stop_reason}",
            )

        backoff = 1
        for i in range(self.retry_attempts):
            try:
                # Use custom LLM generator if provided, otherwise use OpenAI client
                if self.custom_llm_generator:
                    response = self.custom_llm_generator(
                        self.openai_client, kwargs,
                    )
                else:
                    if force_stop_reason:
                        return None
                    response = self.openai_client.chat.completions.create(
                        **kwargs,
                    )

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
                    raise Exception('No response content')

            except Exception as e:
                logger.error(
                    f"❌ LLM call failed (attempt {i + 1}/{self.retry_attempts}): {e}",
                )
                if i == self.retry_attempts - 1:
                    raise e
                time.sleep(backoff)
                backoff *= 2


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
    print(
        f"🔧 Custom LLM Generator called with {len(kwargs.get('messages', []))} messages",
    )

    try:
        # Call the original OpenAI API
        response = openai_client.chat.completions.create(**kwargs)

        return response

    except Exception as e:
        print(f"❌ Bypass LLM generator error: {e}")
        # You could implement custom fallback logic here
        raise
