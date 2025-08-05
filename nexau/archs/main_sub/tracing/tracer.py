# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trace collection and management for agent execution."""

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracer:
    """Manages trace collection for agent execution."""

    def __init__(self, agent_name: str):
        """Initialize tracer for a specific agent.

        Args:
            agent_name: Name of the agent being traced
        """
        self.agent_name = agent_name
        self._trace_data: list[dict[str, Any]] | None = None
        self._dump_trace_path: str | None = None
        self._trace_lock = threading.Lock()
        self._sub_agent_counter = 0
        self._sub_agent_counter_lock = threading.Lock()

    def start_tracing(self, dump_trace_path: str) -> None:
        """Start trace collection.

        Args:
            dump_trace_path: Path where trace will be dumped
        """
        with self._trace_lock:
            self._trace_data = []
            self._dump_trace_path = dump_trace_path
            logger.info(
                f"📊 Trace logging enabled for agent '{self.agent_name}', will dump to: {dump_trace_path}",
            )

    def stop_tracing(self) -> None:
        """Stop trace collection and clear data."""
        with self._trace_lock:
            self._trace_data = None
            self._dump_trace_path = None

    def is_tracing(self) -> bool:
        """Check if tracing is currently active."""
        with self._trace_lock:
            return self._trace_data is not None

    def add_entry(self, entry: dict[str, Any]) -> None:
        """Add an entry to the trace data.

        Args:
            entry: Trace entry to add
        """
        if not self.is_tracing():
            return

        # Ensure timestamp is present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()

        # Ensure agent_name is present
        if "agent_name" not in entry:
            entry["agent_name"] = self.agent_name

        with self._trace_lock:
            if self._trace_data is not None:
                self._trace_data.append(entry)

    def add_llm_request(self, iteration: int, api_params: dict[str, Any]) -> None:
        """Add LLM request trace entry."""
        entry = {
            "type": "llm_request",
            "iteration": iteration,
            "api_params": {
                # Copy api_params but truncate messages for readability if too long
                **{k: v for k, v in api_params.items() if k != "messages"},
                "messages": [
                    {
                        "role": msg["role"],
                        "content": (msg["content"][:1000] + "..." if len(msg["content"]) > 1000 else msg["content"]),
                    }
                    for msg in api_params.get("messages", [])
                ],
            },
        }
        self.add_entry(entry)

    def add_llm_response(self, iteration: int, response_content: str) -> None:
        """Add LLM response trace entry."""
        entry = {
            "type": "llm_response",
            "iteration": iteration,
            "response": {
                "content": response_content,
            },
        }
        self.add_entry(entry)

    def add_tool_request(self, tool_name: str, parameters: dict[str, Any]) -> None:
        """Add tool request trace entry."""
        entry = {
            "type": "tool_request",
            "tool_name": tool_name,
            "parameters": parameters,
        }
        self.add_entry(entry)

    def add_tool_response(self, tool_name: str, result: Any) -> None:
        """Add tool response trace entry."""
        entry = {
            "type": "tool_response",
            "tool_name": tool_name,
            "result": result,
        }
        self.add_entry(entry)

    def add_subagent_request(self, subagent_name: str, message: str) -> None:
        """Add sub-agent request trace entry."""
        entry = {
            "type": "subagent_request",
            "subagent_name": subagent_name,
            "message": message,
        }
        self.add_entry(entry)

    def add_subagent_response(self, subagent_name: str, result: str) -> None:
        """Add sub-agent response trace entry."""
        entry = {
            "type": "subagent_response",
            "subagent_name": subagent_name,
            "result": result,
        }
        self.add_entry(entry)

    def add_error(self, error: Exception) -> None:
        """Add error trace entry."""
        entry = {
            "type": "error",
            "error": str(error),
            "error_type": type(error).__name__,
        }
        self.add_entry(entry)

    def add_shutdown(self, reason: str = "Signal interrupt (Ctrl+C)") -> None:
        """Add shutdown trace entry."""
        entry = {
            "type": "shutdown",
            "reason": reason,
        }
        self.add_entry(entry)

    def get_trace_data(self) -> list[dict[str, Any]] | None:
        """Get current trace data."""
        with self._trace_lock:
            return self._trace_data.copy() if self._trace_data else None

    def get_dump_path(self) -> str | None:
        """Get current dump path."""
        with self._trace_lock:
            return self._dump_trace_path

    def generate_sub_agent_trace_path(
        self,
        sub_agent_name: str,
        main_trace_path: str,
    ) -> str | None:
        """Generate trace path for sub-agent based on main agent's trace path."""
        try:
            # Get unique sub-agent ID
            with self._sub_agent_counter_lock:
                self._sub_agent_counter += 1
                sub_agent_id = self._sub_agent_counter

            # Parse main trace path
            main_path = Path(main_trace_path)
            main_dir = main_path.parent
            main_stem = main_path.stem  # filename without extension

            # Create subfolder structure: xxx/yyy_sub_agents/
            sub_agents_dir = main_dir / f"{main_stem}_sub_agents"

            # Create sub-agent trace filename: sub_agent_name+sub_agent_id.json
            sub_agent_filename = f"{sub_agent_name}_{sub_agent_id}.json"
            sub_agent_trace_path = sub_agents_dir / sub_agent_filename

            return str(sub_agent_trace_path)

        except Exception as e:
            logger.error(f"❌ Failed to generate sub-agent trace path: {e}")
            return None
