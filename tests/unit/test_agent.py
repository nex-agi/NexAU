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
Unit tests for agent components.
"""

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nexau import Agent, AgentConfig
from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent_context import AgentContext, GlobalStorage
from nexau.archs.main_sub.agent_state import AgentState
from nexau.archs.main_sub.config import ExecutionConfig
from nexau.archs.main_sub.context_value import ContextValue
from nexau.archs.main_sub.execution.middleware.context_compaction import ContextCompactionMiddleware
from nexau.archs.main_sub.execution.model_response import ModelResponse, ModelToolCall
from nexau.archs.main_sub.skill import Skill
from nexau.archs.tool import Tool
from nexau.archs.tool.tool_registry import ToolRegistry
from nexau.archs.tracer.core import BaseTracer, Span, SpanType
from nexau.core.adapters.legacy import messages_from_legacy_openai_chat
from nexau.core.messages import Message, Role, TextBlock, ToolResultBlock
from nexau.core.serializers.anthropic_messages import serialize_ump_to_anthropic_messages_payload
from nexau.core.serializers.openai_chat import serialize_ump_to_openai_chat_payload


class DummyTracer(BaseTracer):
    """Minimal tracer implementation for tests."""

    def start_span(
        self,
        name: str,
        span_type: SpanType,
        inputs: dict | None = None,
        parent_span: Span | None = None,
        attributes: dict | None = None,
    ) -> Span:
        return Span(
            id=name,
            name=name,
            type=span_type,
            parent_id=parent_span.id if parent_span else None,
            inputs=inputs or {},
            attributes=attributes or {},
        )

    def end_span(self, span: Span, outputs=None, error=None, attributes=None) -> None:
        span.outputs = outputs or {}
        span.error = str(error) if error else None
        if attributes:
            span.attributes.update(attributes)


class TestAgent:
    """Test cases for Agent class."""

    def test_agent_initialization(self, agent_config, execution_config, global_storage):
        """Test agent initialization."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config.max_iterations = execution_config.max_iterations
            agent_config.max_context_tokens = execution_config.max_context_tokens
            agent_config.max_running_subagents = execution_config.max_running_subagents
            agent_config.retry_attempts = execution_config.retry_attempts
            agent_config.timeout = execution_config.timeout
            agent_config.tool_call_mode = execution_config.tool_call_mode

            agent = Agent(config=agent_config, global_storage=global_storage)

            expected_exec_config = ExecutionConfig.from_agent_config(agent_config)

            assert agent.config == agent_config
            assert agent.global_storage == global_storage
            assert agent.exec_config == expected_exec_config
            assert agent.openai_client is not None
            assert agent.history == []
            assert agent.queued_messages == []

    def test_agent_initialization_sets_agent_id(self, global_storage):
        """Agent should populate missing agent_id with format 'agent_<8-char-hex>'."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
            )

            agent = Agent(config=agent_config, global_storage=global_storage)

            assert isinstance(agent.agent_id, str)
            # Format: "agent_" prefix + 8 hex chars
            assert agent.agent_id.startswith("agent_")
            hex_part = agent.agent_id[6:]  # Remove "agent_" prefix
            assert len(hex_part) == 8
            # Should be valid hexadecimal
            int(hex_part, 16)

    def test_agent_initialization_no_external_client(self, agent_config, global_storage):
        """Test agent initialization when OpenAI client creation fails."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.side_effect = Exception("API Error")

            agent = Agent(config=agent_config, global_storage=global_storage)

            assert agent.openai_client is None

    def test_generate_with_token_rejects_context_compaction_middleware(self, global_storage):
        """generate_with_token should fail fast when context compaction is configured."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(
                    model="test-model",
                    api_type="generate_with_token",
                    base_url="http://example.com",
                    api_key="test-key",
                    tokenizer_path="test-tokenizer",
                ),
                middlewares=[
                    ContextCompactionMiddleware(
                        compaction_strategy="tool_result_compaction",
                    )
                ],
            )

            with pytest.raises(
                ValueError,
                match="generate_with_token.*does not support ContextCompactionMiddleware",
            ):
                Agent(config=agent_config, global_storage=global_storage)

    def test_add_tool(self, agent_config, global_storage, sample_tool):
        """Test adding tools to agent."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            agent.add_tool(sample_tool)

            assert sample_tool.name in agent.tool_registry
            assert sample_tool in agent.config.tools

    def test_tool_call_payload_includes_sub_agents_openai(self, sample_tool):
        """RFC-0015: Agent is a regular builtin tool registered in ToolRegistry.

        When sub_agents is configured, AgentConfig._finalize() injects an Agent
        tool into self.tools. The tool_call_payload should include 'Agent' as
        a regular tool name, not 'sub-agent-child'.
        """
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            child_config = AgentConfig(name="child", llm_config=LLMConfig(model="gpt-4o-mini"))
            agent_config = AgentConfig(
                name="parent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[sample_tool],
                sub_agents={"child": child_config},
                tool_call_mode="openai",
            )

            agent = Agent(config=agent_config)

            # RFC-0015: Agent is a regular builtin tool, not a virtual sub-agent-{name}
            tool_names = {spec["name"] for spec in agent.tool_call_payload}
            assert sample_tool.name in tool_names
            assert "Agent" in tool_names
            assert "sub-agent-child" not in tool_names

    def test_tool_call_payload_uses_skill_description_for_as_skill_openai(self):
        """Structured payload should expose only the brief skill description for as_skill tools."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            skill_tool = Tool(
                name="web_search",
                description="FULL DESCRIPTION: search the web with examples and workflow guidance.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                implementation=lambda query: {"query": query},
                as_skill=True,
                skill_description="BRIEF SKILL DESCRIPTION",
            )
            agent_config = AgentConfig(
                name="parent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[skill_tool],
                tool_call_mode="openai",
            )

            agent = Agent(config=agent_config)

            payload_by_name = {spec["name"]: spec for spec in agent.tool_call_payload}
            assert payload_by_name["web_search"]["description"] == "BRIEF SKILL DESCRIPTION"
            assert payload_by_name["web_search"]["input_schema"]["properties"]["query"]["type"] == "string"
            assert agent.skill_registry["web_search"].detail is not None
            assert "FULL DESCRIPTION" in (agent.skill_registry["web_search"].detail or "")
            assert [tool.name for tool in agent.config.tools].count("LoadSkill") == 1
            load_skill_tool = next(tool for tool in agent.config.tools if tool.name == "LoadSkill")
            assert "<Skills>" in load_skill_tool.description
            assert "web_search" in load_skill_tool.description

    def test_tool_search_description_is_static(self):
        """Phase 1.5: ToolSearch description should be static (no deferred index suffix)."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            deferred_tool = Tool(
                name="GetWeather",
                description="Get the current weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                implementation=lambda city: {"city": city},
                defer_loading=True,
            )
            agent_config = AgentConfig(
                name="parent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[deferred_tool],
                tool_call_mode="openai",
            )

            agent = Agent(config=agent_config)

            tool_search_tool = next(tool for tool in agent._tool_registry.get_all().values() if tool.name == "ToolSearch")
            # Phase 1.5: description should NOT contain deferred index
            assert "<available-deferred-tools>" not in tool_search_tool.description
            assert "GetWeather" not in tool_search_tool.description

    def test_tool_search_not_registered_without_deferred_tools(self):
        """ToolSearch should NOT appear in tool_call_payload when no deferred tools are configured (#367)."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            eager_tool = Tool(
                name="Calculator",
                description="Do math",
                input_schema={
                    "type": "object",
                    "properties": {"expr": {"type": "string"}},
                    "required": ["expr"],
                },
                implementation=lambda expr: expr,
            )
            agent_config = AgentConfig(
                name="no_deferred",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[eager_tool],
                tool_call_mode="openai",
            )

            agent = Agent(config=agent_config)

            payload_names = {spec["name"] for spec in agent.tool_call_payload}
            assert "ToolSearch" not in payload_names
            assert "Calculator" in payload_names
            assert "ToolSearch" not in agent._tool_registry.get_all()

    def test_tool_search_registered_with_deferred_tools(self):
        """ToolSearch SHOULD appear in tool_call_payload when deferred tools are configured (#367)."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            deferred_tool = Tool(
                name="GetWeather",
                description="Get the current weather",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                implementation=lambda city: {"city": city},
                defer_loading=True,
            )
            agent_config = AgentConfig(
                name="has_deferred",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[deferred_tool],
                tool_call_mode="openai",
            )

            agent = Agent(config=agent_config)

            payload_names = {spec["name"] for spec in agent.tool_call_payload}
            assert "ToolSearch" in payload_names
            # Deferred tool should NOT be in the eager payload
            assert "GetWeather" not in payload_names
            assert "ToolSearch" in agent._tool_registry.get_all()

    def test_tool_call_payload_uses_skill_description_for_as_skill_anthropic(self):
        """Structured payload should expose only the brief skill description for as_skill tools."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            skill_tool = Tool(
                name="web_search",
                description="FULL DESCRIPTION: search the web with examples and workflow guidance.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                implementation=lambda query: {"query": query},
                as_skill=True,
                skill_description="BRIEF SKILL DESCRIPTION",
            )
            agent_config = AgentConfig(
                name="parent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[skill_tool],
                tool_call_mode="anthropic",
            )

            agent = Agent(config=agent_config)

            payload_by_name = {spec["name"]: spec for spec in agent.tool_call_payload}
            assert payload_by_name["web_search"]["description"] == "BRIEF SKILL DESCRIPTION"
            assert payload_by_name["web_search"]["input_schema"]["properties"]["query"]["type"] == "string"
            assert agent.skill_registry["web_search"].detail is not None
            assert "FULL DESCRIPTION" in (agent.skill_registry["web_search"].detail or "")

    def test_reusing_agent_config_keeps_skill_source_folder_and_runtime_mapping(self, tmp_path):
        """Reusing the same AgentConfig should not rewrite skill source folders to sandbox paths."""
        local_skill_folder = tmp_path / "feishu-toolkit"
        local_skill_folder.mkdir()
        folder_skill = Skill(
            name="feishu-skill",
            description="Folder based skill",
            detail="Use files shipped with the skill",
            folder=str(local_skill_folder),
        )
        agent_config = AgentConfig(
            name="folder_skill_agent",
            llm_config=LLMConfig(model="gpt-4o-mini"),
            skills=[folder_skill],
        )

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent1 = Agent(config=agent_config)
            agent2 = Agent(config=agent_config)

        # Local sandbox 共享文件系统，skill 保留原始路径，不上传
        assert agent_config.skills[0].folder == str(local_skill_folder)
        assert agent1.skill_registry["feishu-skill"].folder == str(local_skill_folder)
        assert agent2.skill_registry["feishu-skill"].folder == str(local_skill_folder)
        assert agent1.sandbox_manager._session_context.get("upload_assets", []) == []
        assert agent2.sandbox_manager._session_context.get("upload_assets", []) == []

    def test_tool_call_payload_anthropic_mode(self, sample_tool):
        """RFC-0015: Agent is a regular builtin tool registered in ToolRegistry.

        When sub_agents is configured, AgentConfig._finalize() injects an Agent
        tool. The tool_call_payload should include 'Agent' as a regular tool.
        """
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            child_config = AgentConfig(name="child", llm_config=LLMConfig(model="gpt-4o-mini"))
            agent_config = AgentConfig(
                name="parent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[sample_tool],
                sub_agents={"child": child_config},
                tool_call_mode="anthropic",
            )

            agent = Agent(config=agent_config)

            names = {spec["name"] for spec in agent.tool_call_payload}
            assert sample_tool.name in names
            assert "Agent" in names
            assert "sub-agent-child" not in names

    def test_token_counter_callable_is_wrapped(self, global_storage):
        """Callable token_counter should be wrapped into TokenCounter instance."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            def counter(messages, tools=None):
                return 123

            agent_config = AgentConfig(
                name="token_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                token_counter=counter,
            )

            agent = Agent(config=agent_config, global_storage=global_storage)

            assert agent.executor.token_counter._counter([], None) == 123  # type: ignore[attr-defined]

    def test_custom_llm_provider_failure_falls_back(self, agent_config, global_storage):
        """Errors in custom provider should not replace the default runtime client."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            default_client = Mock(name="default_client")
            mock_openai.OpenAI.return_value = default_client

            agent = Agent(config=agent_config, global_storage=global_storage)

            def bad_provider(_):
                raise RuntimeError("boom")

            captured_client = {}

            async def fake_execute(history, agent_state, *, runtime_client, custom_llm_client_provider=None):
                captured_client["client"] = runtime_client
                return "ok", history or []

            agent.executor.execute_async = fake_execute  # type: ignore[method-assign]

            response = agent.run(message="hello", custom_llm_client_provider=bad_provider)

            assert response == "ok"
            assert captured_client["client"] is default_client

    def test_run_accepts_list_of_messages(self, agent_config, global_storage):
        """Agent.run should accept list[Message] and append them directly to history."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            captured_history: dict[str, list[Message]] = {}

            async def fake_execute(history, agent_state, *, runtime_client, custom_llm_client_provider=None):
                captured_history["history"] = list(history) if history else []
                return "ok", history or []

            agent.executor.execute_async = fake_execute  # type: ignore[method-assign]

            input_messages = [
                Message.user("hello"),
                Message(role=Role.ASSISTANT, content=[TextBlock(text="ack")]),
            ]

            response = agent.run(message=input_messages)

            assert response == "ok"
            assert captured_history["history"][-2:] == input_messages

    def test_add_sub_agent(self, agent_config, global_storage):
        """Test adding sub-agents to agent."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            agent_config = AgentConfig(name="test_sub_agent", llm_config=LLMConfig())

            agent.add_sub_agent("test_sub_agent", agent_config)

            assert "test_sub_agent" in agent.config.sub_agents

    def test_enqueue_message(self, agent_config, global_storage):
        """Test enqueuing messages."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            test_message = {"role": "user", "content": "Test message"}

            agent.enqueue_message(test_message)

            assert any(msg.role == Role.USER and msg.get_text_content() == "Test message" for msg in agent.executor.queued_messages)

    def test_agent_cleanup(self, agent_config, global_storage):
        """Test agent cleanup."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            with patch.object(agent.executor, "cleanup") as mock_cleanup:
                agent.sync_cleanup()

                mock_cleanup.assert_called_once()

    def test_agent_injects_tracer_into_global_storage(self, agent_config, global_storage):
        """Tracer set on config should be injected into global storage."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            tracer = DummyTracer()
            config_with_tracer = agent_config.model_copy(update={"tracers": [tracer]})
            config_with_tracer.resolved_tracer = tracer

            Agent(config=config_with_tracer, global_storage=global_storage)

            assert global_storage.get("tracer") is tracer

    def test_agent_config_tracer_overwrites_existing(self, agent_config, global_storage):
        """Config tracer should overwrite any existing tracer in global_storage."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            parent_tracer = DummyTracer()
            global_storage.set("tracer", parent_tracer)

            agent_payload = agent_config.model_dump()
            child_tracer = DummyTracer()
            agent_payload["tracers"] = [child_tracer]
            config_with_tracer = AgentConfig(**agent_payload)
            config_with_tracer.resolved_tracer = child_tracer

            Agent(config=config_with_tracer, global_storage=global_storage)

            assert global_storage.get("tracer") is child_tracer

    def test_initialize_mcp_tools_success(self, agent_config, global_storage):
        """Test successful MCP tools initialization."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock the sync_initialize_mcp_tools function
            mock_mcp_tool = Mock()
            mock_mcp_tool.name = "test_mcp_tool"

            with patch("nexau.archs.tool.builtin.sync_initialize_mcp_tools") as mock_sync_init:
                mock_sync_init.return_value = [mock_mcp_tool]

                # Add MCP servers to config
                agent_config.mcp_servers = [{"name": "test_server", "type": "stdio", "command": "python", "args": ["server.py"]}]

                agent = Agent(config=agent_config, global_storage=global_storage)

                # Verify MCP tools were registered without mutating config.tools
                assert mock_mcp_tool not in agent.config.tools
                assert "test_mcp_tool" in agent._tool_registry.get_all()
                mock_sync_init.assert_called_once_with(agent_config.mcp_servers)

    def test_initialize_mcp_tools_import_error(self, agent_config, global_storage):
        """Test MCP tools initialization with import error."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock import error
            with patch("nexau.archs.tool.builtin.sync_initialize_mcp_tools", side_effect=ImportError("MCP not available")):
                agent_config.mcp_servers = [{"name": "test_server"}]

                # Should not raise, but log error
                agent = Agent(config=agent_config, global_storage=global_storage)

                # MCP tools should not be added
                assert len(agent.config.tools) == 0

    def test_initialize_mcp_tools_general_error(self, agent_config, global_storage):
        """Test MCP tools initialization with general error."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock general exception
            with patch("nexau.archs.tool.builtin.sync_initialize_mcp_tools", side_effect=Exception("Connection failed")):
                agent_config.mcp_servers = [{"name": "test_server"}]

                # Should not raise, but log error
                agent = Agent(config=agent_config, global_storage=global_storage)

                # MCP tools should not be added
                assert len(agent.config.tools) == 0

    def test_run_basic(self, agent_config, global_storage):
        """Test basic agent run."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            # Mock executor.execute_async
            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "Test response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message(role=Role.USER, content=[TextBlock(text="Test message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="Test response")]),
                    ],
                )

                response = agent.run(message="Test message")

                assert response == "Test response"
                assert len(agent.history) > 0
                assert agent.history[-1].role == Role.ASSISTANT
                assert agent.history[-1].get_text_content() == "Test response"
                mock_execute.assert_called_once()

    def test_stop_tool_history_keeps_tool_result_for_next_round(self, global_storage):
        """Regression: if execution stops on a stop_tool, history must retain a tool-result message.

        This guards against a Bedrock/LiteLLM validation failure when a tool_use is not immediately
        followed by a corresponding tool_result in the next request payload.
        """

        def stop_tool() -> dict:
            # Returning a dict keeps the path aligned with typical tool outputs.
            return {"result": "666"}

        tool = Tool(
            name="stopper",
            description="Stop tool",
            input_schema={"type": "object", "properties": {}, "required": []},
            implementation=stop_tool,
        )

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="stop_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[tool],
                tool_call_mode="openai",
                stop_tools={"stopper"},
            )
            agent = Agent(config=agent_config, global_storage=global_storage)

            first_response = ModelResponse(
                content="I'll call the stop tool.",
                tool_calls=[
                    ModelToolCall(
                        call_id="call_stop",
                        name="stopper",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )

            # Stop tool should terminate after the first tool execution; there should be no second LLM call.
            with patch.object(
                agent.executor.llm_caller, "call_llm_async", new_callable=AsyncMock, side_effect=[first_response]
            ) as mock_call_llm:
                response = agent.run(message="do stop")

            mock_call_llm.assert_called_once()
            assert "666" in response

            # History must contain a tool result block with the original call_id.
            tool_msg = next((m for m in agent.history if m.role == Role.TOOL), None)
            assert tool_msg is not None, "Expected a Role.TOOL message in history for stop tool execution"
            tr = next((b for b in tool_msg.content if isinstance(b, ToolResultBlock) and b.tool_use_id == "call_stop"), None)
            assert tr is not None, "Expected ToolResultBlock with tool_use_id matching the tool_call_id"

            # Chat serializer must preserve a role=tool message (OpenAI chat shape).
            payload = serialize_ump_to_openai_chat_payload(agent.history)
            tool_dict = next((m for m in payload if m.get("role") == "tool" and m.get("tool_call_id") == "call_stop"), None)
            assert tool_dict is not None, "Expected Chat payload role=tool message with matching tool_call_id"

            # And when converting to Anthropic Messages (Bedrock), tool_result must appear immediately after tool_use.
            anthropic_ump = messages_from_legacy_openai_chat(payload + [{"role": "user", "content": "hi"}])
            _, anthropic_msgs = serialize_ump_to_anthropic_messages_payload(anthropic_ump)
            idx = next(
                i
                for i, m in enumerate(anthropic_msgs)
                if m.get("role") == "assistant"
                and any(
                    isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id") == "call_stop" for b in (m.get("content") or [])
                )
            )
            assert idx + 1 < len(anthropic_msgs), "Expected a message after tool_use"
            next_msg = anthropic_msgs[idx + 1]
            assert next_msg.get("role") == "user"
            next_blocks = next_msg.get("content") or []
            assert any(
                isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") == "call_stop" for b in next_blocks
            ), "Expected an immediate tool_result block following the tool_use"

    def test_run_with_history(self, agent_config, global_storage):
        """Test agent run with existing history."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            existing_history = [{"role": "user", "content": "Previous message"}, {"role": "assistant", "content": "Previous response"}]

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "New response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message(role=Role.USER, content=[TextBlock(text="Previous message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="Previous response")]),
                        Message(role=Role.USER, content=[TextBlock(text="New message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="New response")]),
                    ],
                )

                response = agent.run(message="New message", history=existing_history)

                assert response == "New response"
                # Should include system prompt + existing history + new message
                call_args = mock_execute.call_args[0][0]
                assert any(msg.get_text_content() == "Previous message" for msg in call_args)

    def test_run_with_mixed_history_types_raises(self, agent_config, global_storage):
        """Mixed Message and legacy dict history should be rejected explicitly."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)
            mixed_history = [Message.user("Previous question"), {"role": "assistant", "content": "Previous answer"}]

            with pytest.raises(TypeError, match="history must contain only Message objects or only legacy OpenAI-chat dicts"):
                agent.run(message="New question", history=mixed_history)

    def test_run_with_context_state_config(self, agent_config, global_storage):
        """Test agent run with context, state, and config."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Set initial values in agent config
            agent_config.initial_context = {"initial_ctx": "value"}
            agent_config.initial_state = {"initial_state": "value"}
            agent_config.initial_config = {"initial_config": "value"}

            agent = Agent(config=agent_config, global_storage=global_storage)

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "Response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message(role=Role.USER, content=[TextBlock(text="Message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="Response")]),
                    ],
                )

                response = agent.run(
                    message="Message",
                    variables=ContextValue(template={"runtime_ctx": "value"}),
                    state={"runtime_state": "value"},
                    config={"runtime_config": "value"},
                )

                assert response == "Response"
                # Verify merged context, state, and config were used
                mock_execute.assert_called_once()

    def test_run_injects_default_runtime_context_for_all_entrypoints(self, agent_config, global_storage):
        """Agent.run should provide platform facts even when callers omit context."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)
            runtime_context = {
                "date": "2026-04-28 13:40:00",
                "username": "tester",
                "working_directory": r"C:\repo",
                "operating_system": "Windows-11",
                "platform": "win32",
                "shell_tool_backend": "Windows PowerShell backend",
                "shell_tool_guidance": "Use PowerShell command syntax.",
                "env_content": {},
            }

            with (
                patch("nexau.archs.main_sub.agent.build_runtime_prompt_context", return_value=runtime_context),
                patch.object(
                    agent.prompt_builder,
                    "build_system_prompt",
                    wraps=agent.prompt_builder.build_system_prompt,
                ) as mock_build_prompt,
                patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute,
            ):
                mock_execute.return_value = (
                    "Response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message(role=Role.USER, content=[TextBlock(text="Message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="Response")]),
                    ],
                )

                response = agent.run(message="Message")

            assert response == "Response"
            prompt_context = mock_build_prompt.call_args.kwargs["runtime_context"]
            assert prompt_context["operating_system"] == "Windows-11"
            assert prompt_context["shell_tool_backend"] == "Windows PowerShell backend"
            assert prompt_context["working_directory"] == r"C:\repo"
            assert agent._last_context["operating_system"] == "Windows-11"

    def test_run_context_overrides_default_runtime_context(self, agent_config, global_storage):
        """Explicit run context should remain higher priority than default runtime facts."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)
            runtime_context = {
                "date": "2026-04-28 13:40:00",
                "username": "tester",
                "working_directory": r"C:\repo",
                "operating_system": "Windows-11",
                "platform": "win32",
                "shell_tool_backend": "Windows PowerShell backend",
                "shell_tool_guidance": "Use PowerShell command syntax.",
                "env_content": {},
            }

            with (
                patch("nexau.archs.main_sub.agent.build_runtime_prompt_context", return_value=runtime_context),
                patch.object(
                    agent.prompt_builder,
                    "build_system_prompt",
                    wraps=agent.prompt_builder.build_system_prompt,
                ) as mock_build_prompt,
                patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute,
            ):
                mock_execute.return_value = (
                    "Response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message(role=Role.USER, content=[TextBlock(text="Message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="Response")]),
                    ],
                )

                response = agent.run(
                    message="Message",
                    context={"operating_system": "OverrideOS", "shell_tool_backend": "Override shell"},
                )

            assert response == "Response"
            prompt_context = mock_build_prompt.call_args.kwargs["runtime_context"]
            assert prompt_context["operating_system"] == "OverrideOS"
            assert prompt_context["shell_tool_backend"] == "Override shell"
            assert prompt_context["working_directory"] == r"C:\repo"

    def test_run_uses_custom_llm_client_provider_for_main_agent(self, agent_config, global_storage):
        """Custom provider should override the runtime client for the main agent only."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            default_client = Mock(name="default_client")
            mock_openai.OpenAI.return_value = default_client

            agent = Agent(config=agent_config, global_storage=global_storage)

            custom_client = Mock(name="custom_client")
            provider = Mock(side_effect=lambda name: custom_client if name == agent_config.name else None)
            captured_kwargs: dict[str, object] = {}

            def fake_execute(messages, agent_state, *, runtime_client, custom_llm_client_provider):
                captured_kwargs["runtime_client"] = runtime_client
                captured_kwargs["custom_llm_client_provider"] = custom_llm_client_provider
                return "Test response", messages + [Message(role=Role.ASSISTANT, content=[TextBlock(text="Test response")])]

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock, side_effect=fake_execute) as mock_execute:
                response = agent.run(message="Test message", custom_llm_client_provider=provider)

            assert response == "Test response"
            assert captured_kwargs["runtime_client"] is custom_client
            assert captured_kwargs["custom_llm_client_provider"] is provider
            provider.assert_called_once_with(agent_config.name)
            mock_execute.assert_called_once()

    def test_run_custom_llm_client_provider_failure_warns_and_falls_back(
        self,
        agent_config,
        global_storage,
        caplog,
    ):
        """If provider raises, fall back to default client and log warning."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            default_client = Mock(name="default_client")
            mock_openai.OpenAI.return_value = default_client

            agent = Agent(config=agent_config, global_storage=global_storage)

            provider = Mock(side_effect=Exception("boom"))
            captured_kwargs: dict[str, object] = {}

            def fake_execute(messages, agent_state, *, runtime_client, custom_llm_client_provider):
                captured_kwargs["runtime_client"] = runtime_client
                captured_kwargs["custom_llm_client_provider"] = custom_llm_client_provider
                return "Test response", messages + [Message(role=Role.ASSISTANT, content=[TextBlock(text="Test response")])]

            with (
                caplog.at_level("WARNING"),
                patch.object(agent.executor, "execute_async", new_callable=AsyncMock, side_effect=fake_execute),
            ):
                response = agent.run(message="Test message", custom_llm_client_provider=provider)

            assert response == "Test response"
            # Should fall back to default OpenAI client
            assert captured_kwargs["runtime_client"] is default_client
            assert captured_kwargs["custom_llm_client_provider"] is provider
            provider.assert_called_once_with(agent_config.name)
            assert any("custom_llm_client_provider failed" in rec.message for rec in caplog.records)

    def test_run_with_tracing_passes_runtime_client_and_provider(
        self,
        agent_config,
        global_storage,
    ):
        """Tracing path should forward runtime client and provider to _run_inner."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            default_client = Mock(name="default_client")
            mock_openai.OpenAI.return_value = default_client

            tracer = Mock()
            agent_config.tracers = [tracer]
            agent_config.resolved_tracer = tracer
            agent = Agent(config=agent_config, global_storage=global_storage)
            global_storage.set("tracer", tracer)

            custom_client = Mock(name="custom_client")
            provider = Mock(return_value=custom_client)

            called_args: dict[str, object] = {}

            def fake_run_inner(agent_state, merged_context, *, runtime_client, custom_llm_client_provider, on_history_update=None):
                called_args["runtime_client"] = runtime_client
                called_args["custom_llm_client_provider"] = custom_llm_client_provider
                return "Traced response"

            class DummyTraceContext:
                def __init__(self, tracer_arg, span_name, span_type, inputs, attributes):
                    self.tracer_arg = tracer_arg
                    self.span_name = span_name
                    self.span_type = span_type
                    self.inputs = inputs
                    self.attributes = attributes
                    self.outputs = None

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False

                def set_outputs(self, outputs):
                    self.outputs = outputs

            ctx_holder: dict[str, DummyTraceContext] = {}

            def ctx_factory(tracer_arg, span_name, span_type, inputs, attributes):
                ctx = DummyTraceContext(tracer_arg, span_name, span_type, inputs, attributes)
                ctx_holder["ctx"] = ctx
                return ctx

            with (
                patch("nexau.archs.main_sub.agent.TraceContext", side_effect=ctx_factory) as mock_ctx,
                patch.object(
                    agent,
                    "_run_inner",
                    side_effect=fake_run_inner,
                ),
            ):
                # ensure tracer available in global storage for TraceContext
                agent.global_storage.set("tracer", tracer)
                response = agent.run(message="Test message", custom_llm_client_provider=provider)

            assert response == "Traced response"
            assert called_args["runtime_client"] is custom_client
            assert called_args["custom_llm_client_provider"] is provider
            provider.assert_called_once_with(agent_config.name)

            assert mock_ctx.call_count >= 1
            ctx_instance = ctx_holder["ctx"]
            assert ctx_instance.tracer_arg is tracer
            assert ctx_instance.span_name == f"Agent: {agent_config.name}"
            assert ctx_instance.outputs == {"response": "Traced response"}

    def test_run_with_tracing_propagates_exception(self, agent_config, global_storage):
        """Tracing wrapper should re-raise errors from _run_inner."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            tracer = DummyTracer()
            agent_config.tracers = [tracer]
            agent_config.resolved_tracer = tracer
            agent = Agent(config=agent_config, global_storage=global_storage)

            class DummyTraceContext:
                def __init__(self, tracer_arg, span_name, span_type, inputs, attributes):
                    self.tracer_arg = tracer_arg
                    self.span_name = span_name
                    self.span_type = span_type
                    self.inputs = inputs
                    self.attributes = attributes
                    self.outputs = None
                    self.errors: list[Exception] = []

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_val:
                        self.errors.append(exc_val)
                    return False

                def set_outputs(self, outputs):
                    self.outputs = outputs

            ctx_holder: dict[str, DummyTraceContext] = {}

            def ctx_factory(tracer_arg, span_name, span_type, inputs, attributes):
                ctx = DummyTraceContext(tracer_arg, span_name, span_type, inputs, attributes)
                ctx_holder["ctx"] = ctx
                return ctx

            def failing_run_inner(*args, **kwargs):
                raise RuntimeError("inner failure")

            with (
                patch("nexau.archs.main_sub.agent.TraceContext", side_effect=ctx_factory) as mock_ctx,
                patch.object(agent, "_run_inner", side_effect=failing_run_inner),
            ):
                with pytest.raises(RuntimeError, match="inner failure"):
                    agent.run(message="Test message")

            assert mock_ctx.call_count == 1
            ctx_instance = ctx_holder["ctx"]
            assert ctx_instance.tracer_arg is tracer
            assert ctx_instance.outputs is None
            assert ctx_instance.errors and isinstance(ctx_instance.errors[0], RuntimeError)

    def test_run_with_error_handler(self, agent_config, global_storage):
        """Test agent run with error handler."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            error_handler_called = []

            def custom_error_handler(error, agent, context):
                error_handler_called.append(True)
                return f"Error handled: {str(error)}"

            agent_config.error_handler = custom_error_handler
            agent = Agent(config=agent_config, global_storage=global_storage)

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock, side_effect=Exception("Test error")):
                response = agent.run(message="Message")

                assert "Error handled: Test error" in response
                assert len(error_handler_called) == 1
                assert agent.history[-1].role == Role.ASSISTANT

    def test_run_without_error_handler(self, agent_config, global_storage):
        """Test agent run without error handler."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock, side_effect=Exception("Test error")):
                with pytest.raises(Exception, match="Test error"):
                    agent.run(message="Message")

    def test_run_with_parent_agent_state(self, agent_config, global_storage):
        """Test agent run with parent agent state."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            parent_state = AgentState(
                agent_name="parent_agent",
                agent_id="parent_123",
                run_id="run_123",
                root_run_id="run_123",
                context=AgentContext(),
                global_storage=global_storage,
                tool_registry=ToolRegistry(),
            )

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "Response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message(role=Role.USER, content=[TextBlock(text="Message")]),
                        Message(role=Role.ASSISTANT, content=[TextBlock(text="Response")]),
                    ],
                )

                response = agent.run(message="Message", parent_agent_state=parent_state)

                assert response == "Response"
                # Verify parent_agent_state was passed to execute
                call_args = mock_execute.call_args[0]
                agent_state = call_args[1]
                assert agent_state.parent_agent_state == parent_state


class TestCreateAgent:
    """Test cases for agent from yaml function."""

    def test_create_agent_minimal(self):
        """Test creating agent with minimal parameters."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
            )
            agent = Agent(config=agent_config)

            assert agent.config.name == "test_agent"
            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert isinstance(agent.agent_id, str)
            assert len(agent.agent_id) > 0

    def test_create_agent_with_dict_llm_config(self):
        """Test creating agent with dictionary LLM config."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.5),
            )
            agent = Agent(config=agent_config)

            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert agent.config.llm_config.temperature == 0.5

    def test_create_agent_with_llm_kwargs(self):
        """Test creating agent with LLM kwargs."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini", temperature=0.3, max_tokens=2000),
            )
            agent = Agent(config=agent_config)

            assert agent.config.llm_config.model == "gpt-4o-mini"
            assert agent.config.llm_config.temperature == 0.3
            assert agent.config.llm_config.max_tokens == 2000

    def test_create_agent_missing_llm_config(self):
        """Test creating agent without LLM config uses default or env model."""
        agent_config = AgentConfig(name="test_agent")
        # Model can be gpt-4o-mini (default) or from LLM_MODEL env var
        assert agent_config.llm_config.model is not None
        assert len(agent_config.llm_config.model) > 0

    def test_create_agent_with_tools(self, sample_tool):
        """Test creating agent with tools."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tools=[sample_tool],
            )
            agent = Agent(config=agent_config)

            assert len(agent.config.tools) == 1
            assert agent.config.tools[0].name == "sample_tool"

    def test_create_agent_with_mcp_servers(self):
        """Test creating agent with MCP servers."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Mock the MCP initialization to prevent actual server startup
            with patch("nexau.archs.main_sub.agent.Agent._initialize_mcp_tools"):
                mcp_servers = [{"name": "test_server", "type": "stdio", "command": "python", "args": ["server.py"]}]

                agent_config = AgentConfig(
                    name="test_agent",
                    llm_config=LLMConfig(model="gpt-4o-mini"),
                    mcp_servers=mcp_servers,
                )
                agent = Agent(config=agent_config)

                assert len(agent.config.mcp_servers) == 1
                assert agent.config.mcp_servers[0]["name"] == "test_server"

    def test_create_agent_with_hooks(self):
        """Test creating agent with hooks."""

        def mock_hook(*args, **kwargs):
            pass

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                after_model_hooks=[mock_hook],
                before_model_hooks=[mock_hook],
                after_tool_hooks=[mock_hook],
            )
            agent = Agent(config=agent_config)

            assert len(agent.config.after_model_hooks) == 1
            assert len(agent.config.before_model_hooks) == 1
            assert len(agent.config.after_tool_hooks) == 1


class TestAgentState:
    """Test cases for AgentState."""

    def test_agent_state_initialization(self):
        """Test agent state initialization."""
        context = AgentContext({"test": "value"})
        global_storage = GlobalStorage()

        state = AgentState(
            agent_name="test_agent",
            agent_id="test_id_123",
            run_id="run_123",
            root_run_id="run_123",
            context=context,
            global_storage=global_storage,
            tool_registry=ToolRegistry(),
        )

        assert state.agent_name == "test_agent"
        assert state.agent_id == "test_id_123"
        assert state.run_id == "run_123"
        assert state.root_run_id == "run_123"
        assert state.context == context
        assert state.global_storage == global_storage
        assert state.parent_agent_state is None

    def test_agent_state_with_parent(self):
        """Test agent state with parent state."""
        parent_state = AgentState(
            agent_name="parent",
            agent_id="parent_123",
            run_id="run_123",
            root_run_id="run_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            tool_registry=ToolRegistry(),
        )

        child_state = AgentState(
            agent_name="child",
            agent_id="child_456",
            run_id="run_456",
            root_run_id="run_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            parent_agent_state=parent_state,
            tool_registry=ToolRegistry(),
        )

        assert child_state.parent_agent_state == parent_state

    def test_get_set_context_value(self):
        """Test getting and setting context values."""
        state = AgentState(
            agent_name="test",
            agent_id="test_123",
            run_id="run_123",
            root_run_id="run_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            tool_registry=ToolRegistry(),
        )

        state.set_context_value("test_key", "test_value")
        assert state.get_context_value("test_key") == "test_value"
        assert state.get_context_value("missing", "default") == "default"

    def test_get_set_global_value(self):
        """Test getting and setting global values."""
        state = AgentState(
            agent_name="test",
            agent_id="test_123",
            run_id="run_123",
            root_run_id="run_123",
            context=AgentContext(),
            global_storage=GlobalStorage(),
            tool_registry=ToolRegistry(),
        )

        state.set_global_value("global_key", "global_value")
        assert state.get_global_value("global_key") == "global_value"
        assert state.get_global_value("missing", "default") == "default"

    def test_add_tool_writes_to_registry(self, agent_state, sample_tool):
        """add_tool should write directly to ToolRegistry."""
        agent_state.add_tool(sample_tool)

        all_tools = agent_state._tool_registry.get_all()
        assert sample_tool.name in all_tools

    def test_string_representations(self):
        """Test string representations of agent state."""
        state = AgentState(
            agent_name="test_agent",
            agent_id="test_id_123",
            run_id="run_123",
            root_run_id="run_123",
            context=AgentContext({"key": "value"}),
            global_storage=GlobalStorage(),
            tool_registry=ToolRegistry(),
        )

        repr_str = repr(state)
        str_repr = str(state)

        # Test __repr__ contains agent info
        assert "test_agent" in repr_str
        assert "test_id_123" in repr_str

        # Test __str__ contains agent name and key counts
        assert "test_agent" in str_repr
        assert "1 context keys" in str_repr
        assert "0 global keys" in str_repr


class TestAgentContext:
    """Test cases for AgentContext."""

    def test_agent_context_initialization(self):
        """Test agent context initialization."""
        context = AgentContext({"initial": "value"})

        assert context.context == {"initial": "value"}
        assert not context.is_modified()

    def test_context_enter_exit(self):
        """Test context manager functionality."""
        context = AgentContext({"test": "value"})

        # Test entering context
        result = context.__enter__()
        assert result == context

        # Test exiting context (no previous context)
        context.__exit__(None, None, None)
        # Should not raise any exceptions

    def test_update_context(self):
        """Test updating context."""
        context = AgentContext({"initial": "value"})

        context.update_context({"new_key": "new_value", "initial": "updated"})

        assert context.context["initial"] == "updated"
        assert context.context["new_key"] == "new_value"
        assert context.is_modified()

    def test_get_set_context_value(self):
        """Test getting and setting individual context values."""
        context = AgentContext()

        context.set_context_value("test_key", "test_value")
        assert context.get_context_value("test_key") == "test_value"
        assert context.get_context_value("missing", "default") == "default"
        assert context.is_modified()

    def test_merge_context_variables(self):
        """Test merging context variables."""
        context = AgentContext({"context_key": "context_value"})

        existing = {"existing_key": "existing_value"}
        merged = context.merge_context_variables(existing)

        assert merged["existing_key"] == "existing_value"
        assert merged["context_key"] == "context_value"
        # Context should take priority
        merged_conflict = context.merge_context_variables({"context_key": "conflict"})
        assert merged_conflict["context_key"] == "context_value"

    def test_modification_callbacks(self):
        """Test modification callbacks."""
        context = AgentContext()
        callback_called = []

        def test_callback():
            callback_called.append(True)

        context.add_modification_callback(test_callback)
        context.set_context_value("test", "value")

        assert len(callback_called) == 1
        assert context.is_modified()

        # Test removing callback
        context.remove_modification_callback(test_callback)
        context.set_context_value("test2", "value2")

        assert len(callback_called) == 1  # Should not increase

    def test_reset_modification_flag(self):
        """Test resetting modification flag."""
        context = AgentContext()

        context.set_context_value("test", "value")
        assert context.is_modified()

        context.reset_modification_flag()
        assert not context.is_modified()


class TestGlobalStorage:
    """Test cases for GlobalStorage."""

    def test_global_storage_initialization(self):
        """Test global storage initialization."""
        storage = GlobalStorage()

        assert storage._storage == {}
        assert storage._locks == {}

    def test_set_get_values(self):
        """Test setting and getting values."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        assert storage.get("key1") == "value1"
        assert storage.get("missing", "default") == "default"

    def test_update_values(self):
        """Test updating multiple values."""
        storage = GlobalStorage()

        storage.update({"key1": "value1", "key2": "value2"})
        assert storage.get("key1") == "value1"
        assert storage.get("key2") == "value2"

    def test_delete_values(self):
        """Test deleting values."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        assert storage.delete("key1") is True
        assert storage.delete("missing") is False
        assert storage.get("key1") is None

    def test_keys_items_methods(self):
        """Test keys and items methods."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        storage.set("key2", "value2")

        assert set(storage.keys()) == {"key1", "key2"}
        assert set(storage.items()) == {("key1", "value1"), ("key2", "value2")}

    def test_clear_storage(self):
        """Test clearing storage."""
        storage = GlobalStorage()

        storage.set("key1", "value1")
        storage.clear()

        assert storage.keys() == []
        assert storage._storage == {}

    def test_lock_key_context_manager(self):
        """Test key-specific locking."""
        storage = GlobalStorage()

        with storage.lock_key("test_key"):
            storage.set("test_key", "value")

        assert storage.get("test_key") == "value"

    def test_lock_multiple_context_manager(self):
        """Test multiple key locking."""
        storage = GlobalStorage()

        with storage.lock_multiple("key1", "key2"):
            storage.set("key1", "value1")
            storage.set("key2", "value2")

        assert storage.get("key1") == "value1"
        assert storage.get("key2") == "value2"


class TestAgentWithExternalRepos:
    """Test cases for Agent with external repositories."""

    def test_agent_with_external_session_manager(self, agent_config, global_storage):
        """Test agent initialization with external session_manager."""
        from nexau.archs.session import InMemoryDatabaseEngine, SessionManager

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Create external session_manager using InMemoryDatabaseEngine
            engine = InMemoryDatabaseEngine()
            session_manager = SessionManager(engine=engine)

            agent = Agent(
                config=agent_config,
                global_storage=global_storage,
                session_manager=session_manager,
                user_id="test_user",
                session_id="test_session",
            )

            assert agent._session_manager is session_manager
            assert agent._user_id == "test_user"
            assert agent._session_id == "test_session"

    def test_agent_session_manager_provides_engine(self, agent_config, global_storage):
        """Test agent's session_manager is stored correctly."""
        from nexau.archs.session import InMemoryDatabaseEngine, SessionManager

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Create external session_manager using InMemoryDatabaseEngine
            engine = InMemoryDatabaseEngine()
            session_manager = SessionManager(engine=engine)

            agent = Agent(
                config=agent_config,
                global_storage=global_storage,
                session_manager=session_manager,
                user_id="test_user",
                session_id="test_session",
            )

            # Verify session_manager is stored
            assert agent._session_manager is session_manager


class TestAgentFromYaml:
    """Test cases for Agent.from_yaml method."""

    def test_from_yaml_file_not_found(self, tmp_path):
        """Test from_yaml raises ConfigError when file not found."""
        from nexau.archs.main_sub.config import ConfigError

        non_existent_path = tmp_path / "non_existent.yaml"

        with pytest.raises(ConfigError, match="Configuration file not found"):
            Agent.from_yaml(non_existent_path)

    def test_from_yaml_with_overrides_warning(self, sample_yaml_config, global_storage):
        """Test from_yaml emits deprecation warning when overrides provided."""

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            with pytest.warns(DeprecationWarning):
                agent = Agent.from_yaml(
                    Path(sample_yaml_config),
                    overrides={"name": "overridden_name"},
                    global_storage=global_storage,
                )

            assert agent is not None

    def test_from_yaml_with_global_storage_in_config(self, tmp_path, global_storage):
        """Test from_yaml creates new global_storage when not provided."""
        import yaml

        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # Test without providing global_storage - should create new one
            agent = Agent.from_yaml(config_path)

            assert agent.global_storage is not None

    def test_from_yaml_yaml_parse_error(self, tmp_path):
        """Test from_yaml raises ConfigError on YAML parse error."""
        from nexau.archs.main_sub.config import ConfigError

        config_path = tmp_path / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ConfigError, match="YAML parsing error"):
            Agent.from_yaml(config_path)


class TestAgentToolCallPayload:
    """Test cases for tool call payload building."""

    def test_build_tool_call_payload_xml_mode(self, global_storage):
        """Test _build_tool_call_payload returns empty list for xml mode."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                tool_call_mode="xml",
            )

            agent = Agent(config=agent_config, global_storage=global_storage)

            # XML mode should have empty tool_call_payload
            assert agent.tool_call_payload == []

    def test_resolve_token_counter_with_none(self, global_storage):
        """Test _resolve_token_counter returns default TokenCounter when None."""
        from nexau.archs.main_sub.utils.token_counter import TokenCounter

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="test_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
                token_counter=None,
            )

            agent = Agent(config=agent_config, global_storage=global_storage)

            assert isinstance(agent.executor.token_counter, TokenCounter)


class TestAgentHistoryManagement:
    """Test cases for agent history management."""

    def test_run_with_message_list_history(self, agent_config, global_storage):
        """Test agent run with Message list as history."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(config=agent_config, global_storage=global_storage)

            # Create Message list history
            message_history = [
                Message.user("Previous question"),
                Message.assistant("Previous answer"),
            ]

            with patch.object(agent.executor, "execute_async", new_callable=AsyncMock) as mock_execute:
                mock_execute.return_value = (
                    "New response",
                    [
                        Message(role=Role.SYSTEM, content=[TextBlock(text="System prompt")]),
                        Message.user("Previous question"),
                        Message.assistant("Previous answer"),
                        Message.user("New question"),
                        Message.assistant("New response"),
                    ],
                )

                response = agent.run(message="New question", history=message_history)

                assert response == "New response"
                # Verify Message history was passed correctly
                call_args = mock_execute.call_args[0][0]
                assert any(msg.get_text_content() == "Previous question" for msg in call_args)

    def test_agent_name_uses_config_name(self, global_storage):
        """Test agent_name uses config name when provided."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent_config = AgentConfig(
                name="my_custom_agent",
                llm_config=LLMConfig(model="gpt-4o-mini"),
            )

            agent = Agent(config=agent_config, global_storage=global_storage)

            # agent_name should use the config name
            assert agent.agent_name == "my_custom_agent"

    def test_agent_is_root_false(self, agent_config, global_storage):
        """Test agent initialization with is_root=False."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            agent = Agent(
                config=agent_config,
                global_storage=global_storage,
                is_root=False,
            )

            assert agent._is_root is False


class TestAgentAsyncSyncCompatibility:
    """Test Agent initialization works in both sync and async contexts.

    These tests verify that the syncify-based implementation works correctly
    when Agent is created from:
    1. Pure sync context (no event loop)
    2. Async context via asyncify (simulates FastAPI + asyncify pattern)
    """

    def test_agent_init_from_sync_context(self, agent_config, global_storage):
        """Test Agent initialization from pure sync context (no event loop)."""
        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            # This runs without any event loop - syncify should use anyio.run()
            agent = Agent(config=agent_config, global_storage=global_storage)

            assert agent is not None
            assert agent.config == agent_config
            assert agent.global_storage is not None

    def test_agent_init_from_asyncify_context(self, agent_config, global_storage):
        """Test Agent initialization via asyncify (simulates FastAPI pattern)."""
        import asyncio

        from asyncer import asyncify

        with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
            mock_openai.OpenAI.return_value = Mock()

            def create_agent() -> Agent:
                # This runs in anyio worker thread - syncify uses anyio.from_thread.run()
                return Agent(config=agent_config, global_storage=global_storage)

            async def main() -> Agent:
                return await asyncify(create_agent)()

            agent = asyncio.run(main())

            assert agent is not None
            assert agent.config == agent_config
            assert agent.global_storage is not None

    def test_agent_run_from_sync_context(self, agent_config, global_storage):
        """Test Agent.run() from pure sync context."""
        with (
            patch("nexau.archs.main_sub.agent.openai") as mock_openai,
            patch.object(Agent, "run_async") as mock_run_async,
        ):
            mock_openai.OpenAI.return_value = Mock()

            async def mock_run(*args, **kwargs):
                return "sync response"

            mock_run_async.side_effect = mock_run

            agent = Agent(config=agent_config, global_storage=global_storage)
            result = agent.run(message="test")

            assert result == "sync response"

    def test_agent_run_from_asyncify_context(self, agent_config, global_storage):
        """Test Agent.run() via asyncify context."""
        import asyncio

        from asyncer import asyncify

        with (
            patch("nexau.archs.main_sub.agent.openai") as mock_openai,
            patch.object(Agent, "run_async") as mock_run_async,
        ):
            mock_openai.OpenAI.return_value = Mock()

            async def mock_run(*args, **kwargs):
                return "asyncify response"

            mock_run_async.side_effect = mock_run

            def run_agent() -> str | tuple[str, dict[str, Any]]:
                agent = Agent(config=agent_config, global_storage=global_storage)
                return agent.run(message="test")

            async def main() -> str | tuple[str, dict[str, Any]]:
                return await asyncify(run_agent)()

            result = asyncio.run(main())

            assert result == "asyncify response"
