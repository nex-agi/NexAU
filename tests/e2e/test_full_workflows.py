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
End-to-end tests for complete agent workflows with real LLM.

Tests realistic multi-step workflows:
- File operations (write → read → verify)
- Multi-tool orchestration
- Agent with deferred and eager tools mixed
"""

import os
import tempfile
from typing import Any

import pytest

from nexau.archs.llm.llm_config import LLMConfig
from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.archs.tool.tool import Tool
from nexau.archs.tracer.adapters.langfuse import LangfuseTracer
from nexau.core.messages import ToolUseBlock


@pytest.fixture
def langfuse_tracer():
    """Shared LangfuseTracer that flushes after each test."""
    tracer = LangfuseTracer()
    yield tracer
    tracer.flush()


class TestFileWorkflow:
    """E2E tests for file operation workflows with real LLM."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for file operations."""
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.fixture
    def file_tools(self, temp_dir):
        """Create read/write file tools scoped to temp_dir."""

        def write_file(file_name: str, content: str) -> dict[str, str]:
            """Write content to a file in the workspace."""
            path = os.path.join(temp_dir, file_name)
            with open(path, "w") as f:
                f.write(content)
            return {"status": "written", "path": path, "size": str(len(content))}

        def read_file(file_name: str) -> dict[str, str]:
            """Read content from a file in the workspace."""
            path = os.path.join(temp_dir, file_name)
            if not os.path.exists(path):
                return {"error": f"File not found: {file_name}"}
            with open(path) as f:
                return {"content": f.read(), "path": path}

        write_tool = Tool(
            name="write_file",
            description="Write content to a file. Provide file_name and content.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of the file"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["file_name", "content"],
            },
            implementation=write_file,
        )

        read_tool = Tool(
            name="read_file",
            description="Read content from a file. Provide file_name.",
            input_schema={
                "type": "object",
                "properties": {
                    "file_name": {"type": "string", "description": "Name of the file"},
                },
                "required": ["file_name"],
            },
            implementation=read_file,
        )

        return write_tool, read_tool

    @pytest.mark.llm
    def test_write_and_read_file_workflow(self, file_tools, temp_dir, langfuse_tracer):
        """Test agent writes a file, then reads it back to verify."""
        write_tool, read_tool = file_tools

        config = AgentConfig(
            name="file_agent",
            system_prompt="You have write_file and read_file tools. Use them as instructed.",
            llm_config=LLMConfig(),
            tools=[write_tool, read_tool],
            tracers=[langfuse_tracer],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="file_workflow_session",
        )

        response = agent.run(message="Write 'Hello NexAU!' to a file named 'greeting.txt', then read it back and tell me what it says.")
        assert isinstance(response, str)
        assert "Hello NexAU!" in response
        # Verify the file was actually created
        assert os.path.exists(os.path.join(temp_dir, "greeting.txt"))


class TestMultiToolOrchestration:
    """E2E tests for agent orchestrating multiple tools."""

    @pytest.mark.llm
    def test_data_lookup_and_calculation(self, langfuse_tracer):
        """Test agent using a lookup tool and a calculator together."""

        def lookup_price(item: str) -> dict[str, Any]:
            """Look up the price of an item."""
            prices = {"apple": 1.50, "banana": 0.75, "orange": 2.00}
            price = prices.get(item.lower())
            if price is None:
                return {"error": f"Item '{item}' not found"}
            return {"item": item, "price": price}

        def calculate(expression: str) -> dict[str, Any]:
            """Evaluate a math expression."""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

        lookup_tool = Tool(
            name="lookup_price",
            description="Look up the price of an item (apple, banana, orange). Returns the price.",
            input_schema={
                "type": "object",
                "properties": {
                    "item": {"type": "string", "description": "Item name to look up"},
                },
                "required": ["item"],
            },
            implementation=lookup_price,
        )

        calc_tool = Tool(
            name="calculator",
            description="Evaluate a mathematical expression. Returns the result.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"},
                },
                "required": ["expression"],
            },
            implementation=calculate,
        )

        config = AgentConfig(
            name="orchestration_agent",
            system_prompt=("You have lookup_price and calculator tools. Use lookup_price to find prices, then calculator for math."),
            llm_config=LLMConfig(),
            tools=[lookup_tool, calc_tool],
            tracers=[langfuse_tracer],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="orchestration_session",
        )

        response = agent.run(message="How much do 3 apples and 2 bananas cost in total?")
        assert isinstance(response, str)
        # 3 * 1.50 + 2 * 0.75 = 6.0
        assert "6" in response


class TestDeferredToolWorkflow:
    """E2E tests for workflows mixing eager and deferred tools."""

    @pytest.mark.llm
    def test_eager_tool_then_deferred_tool_in_conversation(self, langfuse_tracer):
        """Test agent uses eager tool first, then discovers and uses a deferred tool."""

        def calculate(expression: str) -> dict[str, Any]:
            """Evaluate math expression."""
            try:
                result = eval(expression, {"__builtins__": {}}, {})
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}

        def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict[str, Any]:
            """Convert currency."""
            rates = {"USD_EUR": 0.92, "USD_JPY": 149.5, "EUR_USD": 1.09, "EUR_JPY": 162.4}
            key = f"{from_currency}_{to_currency}"
            rate = rates.get(key)
            if rate is None:
                return {"error": f"Conversion {from_currency} → {to_currency} not supported"}
            return {
                "amount": amount,
                "from": from_currency,
                "to": to_currency,
                "rate": rate,
                "converted": round(amount * rate, 2),
            }

        calc_tool = Tool(
            name="Calculator",
            description="Evaluate mathematical expressions.",
            input_schema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
            implementation=calculate,
            defer_loading=False,
        )

        currency_tool = Tool(
            name="CurrencyConverter",
            description="Convert an amount from one currency to another. Supports USD, EUR, JPY.",
            input_schema={
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {"type": "string", "description": "Source currency code"},
                    "to_currency": {"type": "string", "description": "Target currency code"},
                },
                "required": ["amount", "from_currency", "to_currency"],
            },
            implementation=convert_currency,
            defer_loading=True,
            search_hint="currency exchange rate convert",
        )

        config = AgentConfig(
            name="mixed_workflow_agent",
            system_prompt=(
                "You are a helpful assistant. "
                "For currency conversion tasks, never estimate from your own knowledge. "
                "First call ToolSearch with query '+currency +convert', then call the returned currency converter tool. "
                "Use the tool result as the sole source of truth for the final answer."
            ),
            llm_config=LLMConfig(),
            tools=[calc_tool, currency_tool],
            tracers=[langfuse_tracer],
        )

        engine = InMemoryDatabaseEngine()
        session_manager = SessionManager(engine=engine)

        agent = Agent(
            config=config,
            session_manager=session_manager,
            user_id="test_user",
            session_id="mixed_workflow_session",
        )

        # The model must use ToolSearch → find the deferred currency tool → use it.
        response = agent.run(
            message=(
                "Use tools only. Convert exactly 100 USD to EUR. "
                "If CurrencyConverter is not available yet, first call ToolSearch with query '+currency +convert'. "
                "Then call CurrencyConverter and report its converted value."
            ),
        )
        assert isinstance(response, str)
        assert any(
            isinstance(block, ToolUseBlock) and block.name == "CurrencyConverter" for message in agent.history for block in message.content
        )
        # 100 * 0.92 = 92.0
        assert "92" in response
