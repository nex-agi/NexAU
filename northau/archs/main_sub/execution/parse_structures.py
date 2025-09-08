"""Data structures for parsed tool calls, sub-agent calls, and batch operations."""
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any
from typing import Union


class CallType(Enum):
    """Types of executable calls."""
    TOOL = 'tool'
    SUB_AGENT = 'sub_agent'
    BATCH_AGENT = 'batch_agent'


@dataclass
class ToolCall:
    """Represents a parsed tool call."""
    tool_name: str
    parameters: dict[str, Any]
    xml_content: str  # Original XML for error reporting
    tool_call_id: str | None = None

    def __post_init__(self):
        if self.tool_call_id is None:
            self.tool_call_id = hashlib.md5((
                self.tool_name + self.xml_content
            ).encode('utf-8')).hexdigest()


@dataclass
class SubAgentCall:
    """Represents a parsed sub-agent call."""
    agent_name: str
    message: str
    xml_content: str  # Original XML for error reporting
    sub_agent_call_id: str | None = None

    def __post_init__(self):
        if self.sub_agent_call_id is None:
            self.sub_agent_call_id = hashlib.md5((
                self.agent_name + self.xml_content
            ).encode('utf-8')).hexdigest()


@dataclass
class BatchAgentCall:
    """Represents a parsed batch agent call."""
    agent_name: str
    file_path: str
    data_format: str
    message_template: str
    xml_content: str  # Original XML for error reporting
    batch_agent_call_id: str | None = None

    def __post_init__(self):
        if self.batch_agent_call_id is None:
            self.batch_agent_call_id = hashlib.md5((
                self.agent_name + self.xml_content
            ).encode('utf-8')).hexdigest()


# Union type for all call types
ExecutableCall = Union[ToolCall, SubAgentCall, BatchAgentCall]


@dataclass
class ParsedResponse:
    """Container for all parsed calls from an LLM response."""
    original_response: str
    tool_calls: list[ToolCall]
    sub_agent_calls: list[SubAgentCall]
    batch_agent_calls: list[BatchAgentCall]
    is_parallel_tools: bool = False
    is_parallel_sub_agents: bool = False

    def get_all_calls(self) -> list[ExecutableCall]:
        """Get all calls in execution order."""
        all_calls = []
        all_calls.extend(self.tool_calls)
        all_calls.extend(self.sub_agent_calls)
        all_calls.extend(self.batch_agent_calls)
        return all_calls

    def has_calls(self) -> bool:
        """Check if there are any calls to execute."""
        return bool(self.tool_calls or self.sub_agent_calls or self.batch_agent_calls)

    def get_call_summary(self) -> str:
        """Get a summary of all calls."""
        summary_parts = []
        if self.tool_calls:
            summary_parts.append(f"{len(self.tool_calls)} tool calls")
        if self.sub_agent_calls:
            summary_parts.append(
                f"{len(self.sub_agent_calls)} sub-agent calls",
            )
        if self.batch_agent_calls:
            summary_parts.append(
                f"{len(self.batch_agent_calls)} batch agent calls",
            )

        if summary_parts:
            return ', '.join(summary_parts)
        return 'no calls'
