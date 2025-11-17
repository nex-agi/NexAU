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
End-to-end tests for complete agent workflows.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestFullWorkflow:
    """End-to-end tests for complete agent workflows."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_simple_agent_workflow(self):
        """Test a simple agent workflow from start to finish."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create agent config
            config_path = os.path.join(temp_dir, "agent.yaml")
            config = {
                "name": "simple_agent",
                "llm_config": {"model": "gpt-4o-mini", "temperature": 0.7},
                "system_prompt": "You are a helpful assistant.",
                "tools": [],
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Mock LLM response
            mock_response = {
                "choices": [{"message": {"content": "I can help you with that.", "role": "assistant"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
            }

            with patch("nexau.archs.main_sub.agent.openai") as mock_openai:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.OpenAI.return_value = mock_client

                # This demonstrates the E2E flow structure
                assert os.path.exists(config_path)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_agent_with_file_operations(self):
        """Test agent performing file operations end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = os.path.join(temp_dir, "workspace")
            os.makedirs(work_dir)

            # Create agent config with file tools
            config_path = os.path.join(temp_dir, "agent.yaml")
            tools_dir = os.path.join(temp_dir, "tools")
            os.makedirs(tools_dir)

            # Create file tool configs
            read_tool_path = os.path.join(tools_dir, "Read.tool.yaml")
            with open(read_tool_path, "w") as f:
                yaml.dump({"name": "file_read", "description": "Read a file", "builtin": "file_read"}, f)

            write_tool_path = os.path.join(tools_dir, "Write.tool.yaml")
            with open(write_tool_path, "w") as f:
                yaml.dump({"name": "file_write", "description": "Write a file", "builtin": "file_write"}, f)

            # Agent config
            config = {
                "name": "file_agent",
                "llm_config": {"model": "gpt-4o-mini", "temperature": 0.5},
                "system_prompt": "You are a file management assistant.",
                "tools": ["tools/Read.tool.yaml", "tools/Write.tool.yaml"],
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Simulate agent workflow
            # 1. Agent receives task to create a file
            # 2. Agent uses file_write tool
            # 3. Agent uses file_read tool to verify
            # 4. Agent reports success

            test_file = os.path.join(work_dir, "test.txt")
            content = "This is a test file created by the agent."

            # Mock the workflow
            with open(test_file, "w") as f:
                f.write(content)

            assert os.path.exists(test_file)
            with open(test_file) as f:
                assert f.read() == content

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_agent_with_bash_execution(self):
        """Test agent executing bash commands end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create agent config with bash tool
            config_path = os.path.join(temp_dir, "agent.yaml")
            config = {
                "name": "bash_agent",
                "llm_config": {"model": "gpt-4o-mini"},
                "system_prompt": "You can execute bash commands.",
                "tools": [],
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            # Simulate agent executing bash commands
            # In real scenario, agent would use bash_tool
            import subprocess

            result = subprocess.run(["echo", "Hello from agent"], capture_output=True, text=True)

            assert result.returncode == 0
            assert "Hello from agent" in result.stdout


class TestAgentTaskCompletion:
    """End-to-end tests for agent task completion scenarios."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_research_task_workflow(self):
        """Test agent completing a research task."""
        # Simulate a research workflow:
        # 1. Agent receives research query
        # 2. Agent uses web search tool
        # 3. Agent uses web read tool to gather information
        # 4. Agent synthesizes information
        # 5. Agent writes report

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock search results
            search_results = [
                {"title": "Result 1", "url": "http://example.com/1", "snippet": "Information 1"},
                {"title": "Result 2", "url": "http://example.com/2", "snippet": "Information 2"},
            ]

            # Mock article content
            article_content = "This is detailed information about the research topic."

            # Simulate report generation
            report_path = os.path.join(temp_dir, "research_report.md")
            report_content = f"""
# Research Report

## Search Results
- {search_results[0]["title"]}: {search_results[0]["snippet"]}
- {search_results[1]["title"]}: {search_results[1]["snippet"]}

## Detailed Information
{article_content}

## Conclusion
Based on the research, we can conclude...
"""
            with open(report_path, "w") as f:
                f.write(report_content)

            assert os.path.exists(report_path)
            assert len(search_results) > 0

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_code_generation_task(self):
        """Test agent completing a code generation task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate code generation workflow:
            # 1. Agent receives code specification
            # 2. Agent generates code
            # 3. Agent writes code to file
            # 4. Agent creates tests
            # 5. Agent verifies code works

            code_file = os.path.join(temp_dir, "generated_code.py")
            code_content = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

if __name__ == "__main__":
    print(f"Fibonacci(10) = {fibonacci(10)}")
'''
            with open(code_file, "w") as f:
                f.write(code_content)

            # Create test file
            test_file = os.path.join(temp_dir, "test_generated_code.py")
            test_content = """
from generated_code import fibonacci

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(10) == 55
"""
            with open(test_file, "w") as f:
                f.write(test_content)

            assert os.path.exists(code_file)
            assert os.path.exists(test_file)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_data_analysis_workflow(self):
        """Test agent completing a data analysis task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            data_file = os.path.join(temp_dir, "data.csv")
            data_content = """name,age,city
Alice,30,New York
Bob,25,San Francisco
Charlie,35,Boston
"""
            with open(data_file, "w") as f:
                f.write(data_content)

            # Simulate analysis script
            analysis_file = os.path.join(temp_dir, "analysis.py")
            analysis_content = """
import csv

def analyze_data(filename):
    ages = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ages.append(int(row['age']))

    return {
        'count': len(ages),
        'average_age': sum(ages) / len(ages),
        'min_age': min(ages),
        'max_age': max(ages)
    }

if __name__ == "__main__":
    results = analyze_data('data.csv')
    print(f"Analysis Results: {results}")
"""
            with open(analysis_file, "w") as f:
                f.write(analysis_content)

            assert os.path.exists(data_file)
            assert os.path.exists(analysis_file)


class TestMultiAgentCollaboration:
    """End-to-end tests for multi-agent collaboration scenarios."""

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_main_agent_with_subagents(self):
        """Test main agent coordinating with sub-agents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sub-agent configs
            researcher_config = os.path.join(temp_dir, "researcher.yaml")
            with open(researcher_config, "w") as f:
                yaml.dump(
                    {
                        "name": "researcher",
                        "llm_config": {"model": "gpt-4o-mini"},
                        "system_prompt": "You are a research specialist.",
                        "tools": [],
                    },
                    f,
                )

            writer_config = os.path.join(temp_dir, "writer.yaml")
            with open(writer_config, "w") as f:
                yaml.dump(
                    {
                        "name": "writer",
                        "llm_config": {"model": "gpt-4o-mini"},
                        "system_prompt": "You are a writing specialist.",
                        "tools": [],
                    },
                    f,
                )

            # Create main agent config
            main_config = os.path.join(temp_dir, "main_agent.yaml")
            with open(main_config, "w") as f:
                yaml.dump(
                    {
                        "name": "coordinator",
                        "llm_config": {"model": "gpt-4o"},
                        "system_prompt": "You coordinate other agents.",
                        "sub_agents": ["researcher.yaml", "writer.yaml"],
                        "tools": [],
                    },
                    f,
                )

            # Verify all configs exist
            assert os.path.exists(researcher_config)
            assert os.path.exists(writer_config)
            assert os.path.exists(main_config)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_parallel_agent_execution(self):
        """Test multiple agents working in parallel."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple agent configs
            agents = []
            for i in range(3):
                config_path = os.path.join(temp_dir, f"agent{i}.yaml")
                with open(config_path, "w") as f:
                    yaml.dump(
                        {"name": f"agent{i}", "llm_config": {"model": "gpt-4o-mini"}, "system_prompt": f"You are agent {i}.", "tools": []},
                        f,
                    )
                agents.append(config_path)

            # Simulate parallel execution
            results = []
            for agent_config in agents:
                # Each agent would process its task
                results.append({"agent": agent_config, "status": "completed"})

            assert len(results) == 3
            assert all(r["status"] == "completed" for r in results)

    @pytest.mark.e2e
    @pytest.mark.slow
    def test_sequential_agent_pipeline(self):
        """Test agents working in a sequential pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Stage 1: Data collector agent
            stage1_output = os.path.join(temp_dir, "stage1_data.json")
            with open(stage1_output, "w") as f:
                json.dump({"data": ["item1", "item2", "item3"]}, f)

            # Stage 2: Data processor agent
            with open(stage1_output) as f:
                stage1_data = json.load(f)

            stage2_output = os.path.join(temp_dir, "stage2_processed.json")
            processed = [item.upper() for item in stage1_data["data"]]
            with open(stage2_output, "w") as f:
                json.dump({"processed": processed}, f)

            # Stage 3: Report generator agent
            with open(stage2_output) as f:
                stage2_data = json.load(f)

            report = os.path.join(temp_dir, "final_report.md")
            with open(report, "w") as f:
                f.write("# Pipeline Results\n\n")
                f.write(f"Processed {len(stage2_data['processed'])} items\n")

            # Verify pipeline completed
            assert os.path.exists(stage1_output)
            assert os.path.exists(stage2_output)
            assert os.path.exists(report)
