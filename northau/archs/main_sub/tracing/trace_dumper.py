"""Trace file output utilities."""
import json
import logging
import os
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class TraceDumper:
    """Handles dumping trace data to files."""

    @staticmethod
    def dump_trace_to_file(trace_data: list[dict[str, Any]], dump_trace_path: str, agent_name: str) -> None:
        """Dump trace data to a JSON file.

        Args:
            trace_data: List of trace entries
            dump_trace_path: Path to dump the trace file
            agent_name: Name of the agent for metadata
        """
        try:
            # Ensure directory exists
            trace_dir = os.path.dirname(dump_trace_path)
            if trace_dir:  # Only create if there's a directory part
                os.makedirs(trace_dir, exist_ok=True)

            # Prepare trace metadata
            trace_metadata = {
                'agent_name': agent_name,
                'dump_timestamp': datetime.now().isoformat(),
                'total_entries': len(trace_data),
                'entry_types': list({entry.get('type', 'unknown') for entry in trace_data}),
            }

            # Complete trace structure
            complete_trace = {
                'metadata': trace_metadata,
                'trace': trace_data,
            }

            # Write to file
            with open(dump_trace_path, 'w', encoding='utf-8') as f:
                json.dump(complete_trace, f, indent=2, ensure_ascii=False)

            logger.info(
                f"📊 Trace dumped to {dump_trace_path} with {len(trace_data)} entries",
            )

        except Exception as e:
            logger.error(f"❌ Failed to dump trace to {dump_trace_path}: {e}")
            # Don't raise the exception as trace dumping should not break the main execution
