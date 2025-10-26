from enum import Enum, auto


class AgentStopReason(Enum):
    """Enumerates reasons why agent execution may stop."""

    MAX_ITERATIONS_REACHED = auto()
    STOP_TOOL_TRIGGERED = auto()
    ERROR_OCCURRED = auto()
    CONTEXT_TOKEN_LIMIT = auto()
    SUCCESS = auto()
    NO_MORE_TOOL_CALLS = auto()
