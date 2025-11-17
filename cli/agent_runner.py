#!/usr/bin/env python3
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

"""
Agent runner for NexAU CLI with real-time progress tracking.
This script runs the NexAU agent and communicates via stdin/stdout.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import langfuse

# Add parent directory to path to import nexau
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli_subagent_adapter import attach_cli_to_agent

from nexau.archs.config.config_loader import (
    AgentBuilder,
    ConfigError,
    load_yaml_with_vars,
    normalize_agent_config_dict,
)
from nexau.archs.main_sub.execution.hooks import (
    AfterModelHookInput,
    AfterModelHookResult,
    AfterToolHookInput,
    AfterToolHookResult,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from nexau.archs.main_sub.agent import Agent


def get_date():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def send_message(msg_type, content, metadata=None):
    """Send a JSON message to stdout."""
    message = {"type": msg_type, "content": content}
    if metadata:
        message["metadata"] = metadata
    print(json.dumps(message), flush=True)


def create_cli_progress_hook():
    """Create a hook that reports tool calls to the CLI."""

    def progress_hook(hook_input: AfterModelHookInput) -> AfterModelHookResult:
        agent_state = hook_input.agent_state
        agent_name = getattr(agent_state, "agent_name", "agent")
        agent_id = getattr(agent_state, "agent_id", "")
        parent_state = getattr(agent_state, "parent_agent_state", None)
        is_sub_agent = parent_state is not None

        base_metadata = {
            "agent_name": agent_name,
            "agent_id": agent_id,
            "is_sub_agent": is_sub_agent,
            "parent_agent_name": getattr(parent_state, "agent_name", None) if parent_state else None,
            "parent_agent_id": getattr(parent_state, "agent_id", None) if parent_state else None,
            "iteration": hook_input.current_iteration,
        }

        def build_metadata(extra: dict | None = None) -> dict:
            metadata = {k: v for k, v in base_metadata.items() if v is not None}
            if extra:
                metadata.update(extra)
            return metadata

        if hook_input.parsed_response:
            # Extract and display agent's text response (non-tool thinking)
            # This shows what the agent is thinking before executing tools
            if hook_input.original_response:
                # Try to extract text that's not tool calls
                response_text = hook_input.original_response.strip()

                # Send the agent's thinking/reasoning text if it exists
                # This is the text the agent writes before making tool calls
                if response_text and len(response_text) > 10:  # Avoid very short strings
                    # Truncate very long responses for display
                    if len(response_text) > 500:
                        display_text = response_text[:500] + f"... [truncated {len(response_text) - 500} chars]"
                    else:
                        display_text = response_text + f" ({len(response_text)} chars)"

                    # Check if this looks like meaningful text (not just XML/JSON)
                    if not response_text.startswith(("<", "{", "[")):
                        message_type = "subagent_text" if is_sub_agent else "agent_text"
                        send_message(
                            message_type,
                            display_text,
                            metadata=build_metadata({"type": "agent_thinking"}),
                        )

            # Report tool calls with prettier formatting
            if hook_input.parsed_response.tool_calls:
                tool_count = len(hook_input.parsed_response.tool_calls)
                is_parallel = hook_input.parsed_response.is_parallel_tools

                # Format tool calls for display
                tool_details = []
                for i, call in enumerate(hook_input.parsed_response.tool_calls, 1):
                    tool_name = call.tool_name

                    # Format parameters nicely
                    params_preview = ""
                    if hasattr(call, "tool_input") and call.tool_input:
                        # Truncate long parameter values for preview
                        params = {}
                        for key, value in call.tool_input.items():
                            str_value = str(value)
                            if len(str_value) > 100:
                                params[key] = str_value[:100] + f"... [truncated {len(str_value) - 100} chars]"
                            else:
                                params[key] = str_value

                        # Format as readable string
                        params_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:3]])
                        if len(call.tool_input) > 3:
                            params_str += f", ... (+{len(call.tool_input) - 3} more)"
                        params_preview = f"({params_str})"

                    tool_details.append(f"{tool_name}{params_preview}")

                # Send summary message
                execution_type = "parallel" if is_parallel else "sequential"
                message_type = "subagent_step" if is_sub_agent else "step"
                send_message(
                    message_type,
                    f"Planning to execute {tool_count} tool(s) [{execution_type}]:",
                    metadata=build_metadata(
                        {
                            "type": "tool_plan_header",
                            "tool_count": tool_count,
                            "is_parallel": is_parallel,
                        }
                    ),
                )

                # Send each tool as a separate step for better readability
                for i, tool_detail in enumerate(tool_details, 1):
                    send_message(
                        message_type,
                        f"  {i}. {tool_detail}",
                        metadata=build_metadata({"type": "tool_detail"}),
                    )

            # Report sub-agent calls
            if hook_input.parsed_response.sub_agent_calls:
                agent_count = len(hook_input.parsed_response.sub_agent_calls)
                agent_names = [call.agent_name for call in hook_input.parsed_response.sub_agent_calls]

                send_message(
                    message_type,
                    f"Calling {agent_count} sub-agent(s): {', '.join(agent_names)}",
                    metadata=build_metadata(
                        {
                            "type": "subagent_plan",
                            "agent_count": agent_count,
                            "agents": agent_names,
                        }
                    ),
                )

        return AfterModelHookResult.no_changes()

    return progress_hook


def create_cli_tool_hook():
    """Create a hook that reports tool execution to the CLI."""

    def tool_hook(hook_input: AfterToolHookInput) -> AfterToolHookResult:
        # Truncate long outputs for display with clear indicator
        output_preview = str(hook_input.tool_output)
        if len(output_preview) > 200:
            output_preview = output_preview[:200] + f"... [truncated {len(output_preview) - 200} chars]"
        else:
            output_preview = output_preview + f" ({len(output_preview)} chars)"

        agent_state = hook_input.agent_state
        parent_state = getattr(agent_state, "parent_agent_state", None)
        is_sub_agent = parent_state is not None

        metadata = {
            "type": "tool_executed",
            "tool_name": hook_input.tool_name,
            "output_preview": output_preview,
            "agent_name": getattr(agent_state, "agent_name", "agent"),
            "agent_id": getattr(agent_state, "agent_id", ""),
            "is_sub_agent": is_sub_agent,
            "parent_agent_name": getattr(parent_state, "agent_name", None) if parent_state else None,
            "parent_agent_id": getattr(parent_state, "agent_id", None) if parent_state else None,
        }

        # Remove None values for cleanliness
        metadata = {k: v for k, v in metadata.items() if v is not None}

        message_type = "subagent_step" if is_sub_agent else "step"

        send_message(
            message_type,
            f"Tool '{hook_input.tool_name}' completed",
            metadata=metadata,
        )

        return AfterToolHookResult.no_changes()

    return tool_hook


def main():
    if len(sys.argv) < 2:
        send_message("error", "No agent configuration file provided")
        sys.exit(1)

    yaml_path = sys.argv[1]

    try:
        # Load agent from YAML configuration
        send_message("status", "Loading agent configuration...")

        # Create our CLI progress hooks
        cli_progress_hook = create_cli_progress_hook()
        cli_tool_hook = create_cli_tool_hook()

        # Helper to forward sub-agent lifecycle events to the CLI
        def handle_subagent_event(event_type: str, payload: dict):
            event_mapping = {
                "start": ("subagent_start", "message"),
                "complete": ("subagent_complete", "result"),
                "error": ("subagent_error", "error"),
            }

            if event_type not in event_mapping:
                return

            message_type, content_field = event_mapping[event_type]
            content = payload.get(content_field, "")
            send_message(message_type, content, metadata=payload)

        config_path = Path(yaml_path)

        def build_agent_from_config() -> Agent:
            raw_config = load_yaml_with_vars(str(config_path))
            if not raw_config:
                raise ConfigError(
                    f"Empty or invalid configuration file: {config_path}",
                )

            normalized_config = normalize_agent_config_dict(raw_config)

            existing_model_hooks = list(normalized_config.get("after_model_hooks", []))
            existing_tool_hooks = list(normalized_config.get("after_tool_hooks", []))

            normalized_config["after_model_hooks"] = [cli_progress_hook] + existing_model_hooks
            normalized_config["after_tool_hooks"] = [cli_tool_hook] + existing_tool_hooks

            builder = AgentBuilder(normalized_config, config_path.parent)
            return (
                builder.build_core_properties()
                .build_llm_config()
                .build_mcp_servers()
                .build_hooks()
                .build_tools()
                .build_sub_agents()
                .build_skills()
                .build_system_prompt_path()
                .get_agent()
            )

        agent = build_agent_from_config()

        attach_cli_to_agent(agent, cli_progress_hook, cli_tool_hook, handle_subagent_event)

        send_message("status", "Agent loaded successfully")
        send_message("ready", "Agent is ready for input")

        # Process messages from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)

                if data.get("type") == "exit":
                    send_message("status", "Shutting down...")
                    break

                if data.get("type") == "message":
                    user_message = data.get("content", "")
                    if not user_message:
                        continue

                    # Check for /clear command
                    if user_message.strip() == "/clear":
                        send_message("status", "Re-initializing agent...")

                        agent = build_agent_from_config()

                        attach_cli_to_agent(agent, cli_progress_hook, cli_tool_hook, handle_subagent_event)

                        send_message("status", "Agent re-initialized successfully")
                        send_message("response", "Agent has been re-initialized. Conversation history cleared.")
                        send_message("ready", "")
                        continue

                    send_message("step", "Processing request...", metadata={"type": "start"})

                    # Run the agent
                    response = agent.run(
                        user_message,
                        context={
                            "date": get_date(),
                            "username": os.getenv("USER", "user"),
                            "working_directory": os.getcwd(),
                            "env_content": {
                                "date": get_date(),
                                "username": os.getenv("USER", "user"),
                                "working_directory": os.getcwd(),
                            },
                        },
                    )

                    if agent.langfuse_trace_id:
                        response += f"\n\nLangfuse trace URL: {langfuse.get_client().get_trace_url(trace_id=agent.langfuse_trace_id)}"

                    send_message("step", "Request completed", metadata={"type": "complete"})
                    send_message("response", response)
                    send_message("ready", "")

            except ConfigError as e:
                send_message("error", str(e))
                send_message("ready", "")
            except json.JSONDecodeError:
                send_message("error", f"Invalid JSON received: {line}")
            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                send_message("error", f"{str(e)}\n{error_details}")
                send_message("ready", "")

    except ConfigError as e:
        send_message("error", str(e))
        sys.exit(1)
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        send_message("error", f"Failed to load agent: {str(e)}\n{error_details}")
        sys.exit(1)


if __name__ == "__main__":
    main()
