"""Utilities for consistent sub-agent tool naming."""

SUB_AGENT_TOOL_PREFIX = "sub-agent."
LEGACY_SUB_AGENT_TOOL_PREFIX = "agent:"


def build_sub_agent_tool_name(agent_name: str) -> str:
    """Return the canonical sub-agent tool identifier."""
    return f"{SUB_AGENT_TOOL_PREFIX}{agent_name}"


def is_sub_agent_tool_name(name: str | None) -> bool:
    """Check whether a tool name represents a sub-agent call."""
    if not name:
        return False
    return name.startswith(SUB_AGENT_TOOL_PREFIX) or name.startswith(LEGACY_SUB_AGENT_TOOL_PREFIX)


def extract_sub_agent_name(name: str | None) -> str | None:
    """Extract the logical sub-agent name from a tool identifier."""
    if not name:
        return None
    if name.startswith(SUB_AGENT_TOOL_PREFIX):
        return name[len(SUB_AGENT_TOOL_PREFIX) :]
    if name.startswith(LEGACY_SUB_AGENT_TOOL_PREFIX):
        return name[len(LEGACY_SUB_AGENT_TOOL_PREFIX) :]
    return None
