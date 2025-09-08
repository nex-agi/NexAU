"""Token counting utilities for agents."""
import logging
from typing import Callable

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None


class TokenCounter:
    """Handles token counting for LLM messages using various strategies."""

    def __init__(self, strategy: str = 'tiktoken', model: str = 'gpt-4o'):
        """Initialize token counter with specified strategy.

        Args:
            strategy: "tiktoken" or "fallback"
            model: Model name for tiktoken encoding
        """
        self.strategy = strategy
        self.model = model
        self._counter = self._create_counter()

    def _create_counter(self) -> Callable[[list[dict[str, str]]], int]:
        """Create the appropriate token counter based on strategy."""
        if self.strategy == 'tiktoken' and TIKTOKEN_AVAILABLE:
            return self._create_tiktoken_counter()
        else:
            if self.strategy == 'tiktoken':
                logger.warning(
                    'tiktoken not available, using fallback counter',
                )
            return self._create_fallback_counter()

    def _create_tiktoken_counter(self) -> Callable[[list[dict[str, str]]], int]:
        """Create tiktoken-based counter."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)

            def tiktoken_message_counter(messages: list[dict[str, str]]) -> int:
                """Count tokens in messages using tiktoken."""
                total_tokens = 0
                for message in messages:
                    # Add tokens for role and content
                    total_tokens += len(
                        encoding.encode(
                            message.get('content', ''), disallowed_special=(),
                        ),
                    )
                return total_tokens

            return tiktoken_message_counter
        except Exception as e:
            logger.warning(
                f"Failed to create tiktoken encoder: {e}, using fallback",
            )
            return self._create_fallback_counter()

    def _create_fallback_counter(self) -> Callable[[list[dict[str, str]]], int]:
        """Create fallback counter using character approximation."""
        def fallback_message_counter(messages: list[dict[str, str]]) -> int:
            """Fallback token counter using character approximation."""
            total_tokens = 0
            for message in messages:
                # Add tokens for role and content using chars/4 approximation
                total_tokens += len(message.get('role', '')) // 4
                total_tokens += len(message.get('content', '')) // 4
                # Add overhead tokens for message formatting
                total_tokens += 4
            return max(total_tokens, 1)  # Ensure at least 1 token

        return fallback_message_counter

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Count total tokens in a list of messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            Total token count
        """
        return self._counter(messages)
