"""Provider-specific aggregators for OpenAI Responses API only.

This module contains aggregators that are designed specifically for
OpenAI Responses API and should not be used for other LLM providers.

Example usage:
    from nexau.archs.llm.llm_aggregators.openai_responses import (
        OpenAIResponsesAggregator,
    )

    # NOT intended for direct use by end users
    # Used internally by OpenAI LLM providers
"""

from .openai_responses_aggregator import OpenAIResponsesAggregator

__all__ = [
    "OpenAIResponsesAggregator",
]
