"""Batch processing functionality for agents."""
import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Any

from ..utils.xml_utils import XMLParser

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Handles batch processing of data through sub-agents."""

    def __init__(self, subagent_manager, max_workers: int = 5):
        """Initialize batch processor.

        Args:
            subagent_manager: SubAgentManager instance
            max_workers: Maximum number of parallel workers
        """
        self.subagent_manager = subagent_manager
        self.max_workers = max_workers
        self.xml_parser = XMLParser()

    def execute_batch_agent_from_xml(self, xml_content: str) -> str:
        """Execute batch agent processing from XML content.

        Args:
            xml_content: XML content describing the batch operation

        Returns:
            JSON string with batch processing results
        """
        try:
            # Parse XML using robust parsing
            root = self.xml_parser.parse_xml_content(xml_content)

            # Get agent name
            agent_name_elem = root.find('agent_name')
            if agent_name_elem is None:
                raise ValueError('Missing agent_name in batch agent XML')

            agent_name = (agent_name_elem.text or '').strip()

            # Get input data source
            input_data_elem = root.find('input_data_source')
            if input_data_elem is None:
                raise ValueError(
                    'Missing input_data_source in batch agent XML',
                )

            file_name_elem = input_data_elem.find('file_name')
            if file_name_elem is None:
                raise ValueError('Missing file_name in input_data_source')

            file_path = (file_name_elem.text or '').strip()

            format_elem = input_data_elem.find('format')
            data_format = (
                (format_elem.text or 'jsonl').strip()
                if format_elem is not None
                else 'jsonl'
            )

            # Get message template
            message_elem = root.find('message')
            if message_elem is None:
                raise ValueError('Missing message in batch agent XML')

            message_template = (message_elem.text or '').strip()

            # Validate file exists or create if not exist
            if not os.path.exists(file_path):
                logger.warning(
                    f"File {file_path} does not exist. Creating empty JSONL file.",
                )
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w'):
                    pass  # Create empty file
                return 'Batch processing completed: 0 items processed (file was created but empty)'

            # Execute batch processing
            return self._process_batch_data(
                agent_name, file_path, data_format, message_template,
            )

        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {e}")
        except Exception as e:
            raise ValueError(f"Batch processing error: {e}")

    def _process_batch_data(
        self, agent_name: str, file_path: str, data_format: str, message_template: str,
    ) -> str:
        """Process batch data from file and execute agent calls in parallel.

        Args:
            agent_name: Name of the agent to use for processing
            file_path: Path to the data file
            data_format: Format of the data file
            message_template: Template for messages to send to agent

        Returns:
            JSON string with processing results
        """
        if data_format.lower() != 'jsonl':
            raise ValueError(
                f"Unsupported data format: {data_format}. Only 'jsonl' is supported.",
            )

        # Read and validate JSONL file
        batch_data = []
        try:
            with open(file_path, encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if not isinstance(data, dict):
                            logger.warning(
                                f"Line {line_num}: Expected JSON object, got {type(data).__name__}",
                            )
                            continue
                        batch_data.append((line_num, data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Line {line_num}: Invalid JSON - {e}")
                        continue
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

        if not batch_data:
            return 'Batch processing completed: 0 items processed (no valid JSON objects found)'

        # Validate message template uses valid keys
        template_keys = self._extract_template_keys(message_template)
        sample_data = batch_data[0][1]
        invalid_keys = [key for key in template_keys if key not in sample_data]
        if invalid_keys:
            raise ValueError(
                f"Message template uses invalid keys: {invalid_keys}. Available keys in data: {list(sample_data.keys())}",
            )

        # Execute batch processing in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batch items for parallel execution
            futures = {}
            for line_num, data in batch_data:
                # Render message template with data
                try:
                    rendered_message = self._render_message_template(
                        message_template,
                        data,
                    )
                    # Propagate current tracing context into the worker thread
                    task_ctx = copy_context()
                    future = executor.submit(
                        task_ctx.run,
                        self._execute_batch_item_safe,
                        agent_name,
                        rendered_message,
                        line_num,
                    )
                    futures[future] = (line_num, data)
                except Exception as e:
                    results.append(
                        {
                            'line': line_num,
                            'status': 'error',
                            'error': f"Template rendering failed: {e}",
                            'data': data,
                        },
                    )

            # Collect results as they complete
            for future in as_completed(futures):
                line_num, data = futures[future]
                try:
                    result = future.result()
                    results.append(
                        {
                            'line': line_num,
                            'status': 'success',
                            'result': result,
                            'data': data,
                        },
                    )
                except Exception as e:
                    results.append(
                        {
                            'line': line_num,
                            'status': 'error',
                            'error': str(e),
                            'data': data,
                        },
                    )

        # Sort results by line number
        results.sort(key=lambda x: x['line'])

        # Generate summary
        total_items = len(results)
        successful_items = len(
            [r for r in results if r['status'] == 'success'],
        )
        failed_items = total_items - successful_items

        summary = f"Batch processing completed: {successful_items}/{total_items} items successful, {failed_items} failed"

        # Limit detailed results to first 3 items to prevent overly long responses
        displayed_results = results[:3]
        remaining_count = max(0, len(results) - 3)

        # Include limited detailed results
        detailed_results = {
            'summary': summary,
            'total_items': total_items,
            'successful_items': successful_items,
            'failed_items': failed_items,
            'displayed_results': displayed_results,
            'remaining_items': remaining_count,
        }

        if remaining_count > 0:
            detailed_results['note'] = (
                f"Showing first 3 results. {remaining_count} additional results not displayed to keep response concise."
            )

        return json.dumps(detailed_results, indent=2, ensure_ascii=False)

    def _extract_template_keys(self, template: str) -> list[str]:
        """Extract variable keys from message template.

        Args:
            template: Message template string

        Returns:
            List of variable keys found in template
        """
        # Find all {variable_name} patterns
        keys = re.findall(r'\{([^}]+)\}', template)
        return keys

    def _render_message_template(self, template: str, data: dict[str, Any]) -> str:
        """Render message template with data.

        Args:
            template: Message template string
            data: Data to substitute into template

        Returns:
            Rendered message string
        """
        try:
            return template.format(**data)
        except KeyError as e:
            missing_key = str(e).strip("'\"")
            available_keys = list(data.keys())
            raise ValueError(
                f"Template key '{missing_key}' not found in data. Available keys: {available_keys}",
            )

    def _execute_batch_item_safe(
        self, agent_name: str, message: str, line_num: int,
    ) -> str:
        """Safely execute a single batch item.

        Args:
            agent_name: Name of the agent to use
            message: Message to send to agent
            line_num: Line number for logging

        Returns:
            Result from agent execution
        """
        try:
            logger.info(
                f"🔄 Processing batch item {line_num} with agent '{agent_name}'",
            )
            result = self.subagent_manager.call_sub_agent(agent_name, message)
            logger.info(f"✅ Batch item {line_num} completed successfully")
            return result
        except Exception as e:
            logger.error(f"❌ Batch item {line_num} failed: {e}")
            raise
