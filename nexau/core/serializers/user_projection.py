"""Provider-only projection for user-shaped UMP messages.

Canonical history keeps ``USER`` and ``FRAMEWORK`` as separate messages so
storage and presentation retain their source. Providers receive a temporary
view where a contiguous run containing FRAMEWORK is one ordinary USER turn.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from nexau.core.messages import DiscriminatedBlock, ImageBlock, Message, Role, TextBlock

_USER_SHAPED_ROLES = frozenset({Role.USER, Role.FRAMEWORK})


@dataclass(frozen=True)
class ProviderMessageProjection:
    """One provider-view message and whether FRAMEWORK contributed to it."""

    message: Message
    contains_framework: bool


def _provider_content(message: Message) -> list[DiscriminatedBlock]:
    """Return content that can participate in a provider USER turn."""

    if message.role == Role.FRAMEWORK:
        return [block for block in message.content if isinstance(block, ImageBlock) or (isinstance(block, TextBlock) and bool(block.text))]
    return list(message.content)


def project_user_shaped_messages(messages: Sequence[Message]) -> list[ProviderMessageProjection]:
    """Project USER/FRAMEWORK runs without mutating canonical messages.

    Adjacent ordinary USER messages are left alone. A run is coalesced only
    when at least one meaningful FRAMEWORK message participates. Empty or
    protocol-shaped FRAMEWORK rows remain in canonical history but are omitted
    from the provider view.
    """

    projected: list[ProviderMessageProjection] = []
    index = 0
    while index < len(messages):
        current = messages[index]
        if current.role not in _USER_SHAPED_ROLES:
            projected.append(ProviderMessageProjection(current, False))
            index += 1
            continue

        end = index + 1
        while end < len(messages) and messages[end].role in _USER_SHAPED_ROLES:
            end += 1
        group = messages[index:end]
        if not any(message.role == Role.FRAMEWORK for message in group):
            projected.extend(ProviderMessageProjection(message, False) for message in group)
            index = end
            continue

        content_groups = [(message, content) for message in group if (content := _provider_content(message))]
        meaningful_framework = any(message.role == Role.FRAMEWORK for message, _content in content_groups)
        if not meaningful_framework:
            projected.extend(ProviderMessageProjection(message, False) for message in group if message.role == Role.USER)
            index = end
            continue

        base = next((message for message in group if message.role == Role.USER), content_groups[0][0])
        merged_content: list[DiscriminatedBlock] = []
        for content_index, (_message, content) in enumerate(content_groups):
            if content_index:
                merged_content.append(TextBlock(text="\n\n"))
            merged_content.extend(content)

        provider_message = base.model_copy(
            update={
                "role": Role.USER,
                "content": merged_content,
            }
        )
        projected.append(ProviderMessageProjection(provider_message, True))
        index = end

    return projected


def coalesce_user_shaped_messages(messages: Sequence[Message]) -> list[Message]:
    """Return only the provider-view messages for existing serializers."""

    return [projection.message for projection in project_user_shaped_messages(messages)]
