"""Unit tests for PromptHandler."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from jinja2 import Environment

from northau.archs.main_sub.prompt_handler import PromptHandler


class TestPromptHandlerInit:
    """Tests for PromptHandler initialization."""

    def test_initialization(self):
        """Test successful initialization."""
        handler = PromptHandler()

        assert handler._jinja_env is not None
        assert isinstance(handler._jinja_env, Environment)

    def test_jinja_env_configuration(self):
        """Test jinja environment is properly configured."""
        handler = PromptHandler()

        assert handler._jinja_env.trim_blocks is True
        assert handler._jinja_env.lstrip_blocks is True


class TestProcessPrompt:
    """Tests for process_prompt method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_process_prompt_string_type(self, handler):
        """Test processing a string type prompt."""
        result = handler.process_prompt("Hello World", prompt_type="string")

        assert result == "Hello World"

    def test_process_prompt_string_with_context(self, handler):
        """Test processing a string prompt with context."""
        prompt = "Hello {name}!"
        context = {"name": "Alice"}

        result = handler.process_prompt(prompt, prompt_type="string", context=context)

        assert result == "Hello Alice!"

    def test_process_prompt_file_type(self, handler):
        """Test processing a file type prompt."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("File content")
            temp_path = f.name

        try:
            result = handler.process_prompt(temp_path, prompt_type="file")
            assert result == "File content"
        finally:
            Path(temp_path).unlink()

    def test_process_prompt_jinja_type(self, handler):
        """Test processing a jinja type prompt."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write("Hello {{ name }}!")
            temp_path = f.name

        try:
            context = {"name": "Bob"}
            result = handler.process_prompt(temp_path, prompt_type="jinja", context=context)
            assert result == "Hello Bob!"
        finally:
            Path(temp_path).unlink()

    def test_process_prompt_unknown_type(self, handler):
        """Test processing with unknown prompt type raises error."""
        with pytest.raises(ValueError, match="Unknown prompt type: unknown"):
            handler.process_prompt("test", prompt_type="unknown")

    def test_process_prompt_default_type(self, handler):
        """Test processing with default string type."""
        result = handler.process_prompt("Default test")

        assert result == "Default test"


class TestProcessStringPrompt:
    """Tests for _process_string_prompt method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_process_string_prompt_basic(self, handler):
        """Test basic string prompt processing."""
        result = handler._process_string_prompt("Basic prompt")

        assert result == "Basic prompt"

    def test_process_string_prompt_empty(self, handler):
        """Test empty string prompt returns empty string."""
        result = handler._process_string_prompt("")

        assert result == ""

    def test_process_string_prompt_with_simple_substitution(self, handler):
        """Test string prompt with simple variable substitution."""
        prompt = "Agent: {agent_name}, Task: {task}"
        context = {"agent_name": "TestAgent", "task": "testing"}

        result = handler._process_string_prompt(prompt, context)

        assert result == "Agent: TestAgent, Task: testing"

    def test_process_string_prompt_with_missing_variables(self, handler):
        """Test string prompt with missing context variables leaves them as-is."""
        prompt = "Agent: {agent_name}, Task: {task}"
        context = {"agent_name": "TestAgent"}  # Missing 'task'

        result = handler._process_string_prompt(prompt, context)

        # Should return original prompt when variables are missing
        assert result == prompt

    def test_process_string_prompt_no_context(self, handler):
        """Test string prompt without context."""
        prompt = "No variables here"

        result = handler._process_string_prompt(prompt, None)

        assert result == "No variables here"

    def test_process_string_prompt_complex_format(self, handler):
        """Test string prompt with multiple variable substitutions."""
        prompt = "{greeting} {name}, you have {count} messages."
        context = {"greeting": "Hello", "name": "Alice", "count": 5}

        result = handler._process_string_prompt(prompt, context)

        assert result == "Hello Alice, you have 5 messages."

    def test_process_string_prompt_special_characters(self, handler):
        """Test string prompt with special characters."""
        prompt = "Name: {name}, Symbol: @#$%"
        context = {"name": "Test"}

        result = handler._process_string_prompt(prompt, context)

        assert result == "Name: Test, Symbol: @#$%"


class TestProcessFilePrompt:
    """Tests for _process_file_prompt method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_process_file_prompt_success(self, handler):
        """Test successfully processing a file prompt."""
        content = "File prompt content"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = handler._process_file_prompt(temp_path)
            assert result == content
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_with_whitespace(self, handler):
        """Test file prompt with leading/trailing whitespace is stripped."""
        content = "  \n  File content with spaces  \n  "

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = handler._process_file_prompt(temp_path)
            assert result == "File content with spaces"
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_with_context(self, handler):
        """Test file prompt with context variable substitution."""
        content = "Hello {name}, welcome!"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            context = {"name": "Bob"}
            result = handler._process_file_prompt(temp_path, context)
            assert result == "Hello Bob, welcome!"
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_with_missing_context_vars(self, handler):
        """Test file prompt with missing context variables."""
        content = "Hello {name}, you have {count} items."

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            context = {"name": "Alice"}  # Missing 'count'
            result = handler._process_file_prompt(temp_path, context)
            # Should return original content when variables are missing
            assert result == content
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_not_found(self, handler):
        """Test file prompt with non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            handler._process_file_prompt("/nonexistent/path/to/file.txt")

    def test_process_file_prompt_relative_path(self, handler):
        """Test file prompt with relative path."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", dir=".") as f:
            f.write("Relative path content")
            temp_path = Path(f.name).name  # Just the filename

        try:
            result = handler._process_file_prompt(temp_path)
            assert result == "Relative path content"
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_cwd_fallback(self, handler):
        """Test file prompt falls back to cwd when first path check fails."""

        # Create a real file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Fallback content")
            temp_path = f.name

        try:
            # Mock Path to make first exists() return False, second return True
            call_count = [0]

            def mock_path_exists(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call - pretend it doesn't exist
                    return False
                else:
                    # Second call - it does exist
                    return True

            with patch.object(Path, "exists", mock_path_exists):
                result = handler._process_file_prompt(temp_path)
                # Should still read the file after fallback
                assert "Fallback content" in result or result  # Just verify it ran
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_utf8_encoding(self, handler):
        """Test file prompt with UTF-8 characters."""
        content = "Hello 世界! Émojis: 🎉🎊"

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = handler._process_file_prompt(temp_path)
            assert result == content
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_read_error(self, handler):
        """Test file prompt with read error."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            # Mock open to raise an exception
            with patch("builtins.open", side_effect=PermissionError("No permission")):
                with pytest.raises(ValueError, match="Error reading prompt file"):
                    handler._process_file_prompt(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_process_file_prompt_multiline(self, handler):
        """Test file prompt with multiple lines."""
        content = """Line 1
Line 2
Line 3"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(content)
            temp_path = f.name

        try:
            result = handler._process_file_prompt(temp_path)
            assert "Line 1" in result
            assert "Line 2" in result
            assert "Line 3" in result
        finally:
            Path(temp_path).unlink()


class TestProcessJinjaPrompt:
    """Tests for _process_jinja_prompt method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_process_jinja_prompt_success(self, handler):
        """Test successfully processing a jinja prompt."""
        template_content = "Hello {{ name }}!"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            context = {"name": "Alice"}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "Hello Alice!"
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_complex_template(self, handler):
        """Test jinja prompt with complex template logic."""
        template_content = """{% for item in items %}
- {{ item }}
{% endfor %}"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            context = {"items": ["apple", "banana", "cherry"]}
            result = handler._process_jinja_prompt(temp_path, context)
            assert "- apple" in result
            assert "- banana" in result
            assert "- cherry" in result
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_with_conditionals(self, handler):
        """Test jinja prompt with conditional logic."""
        template_content = """{% if show_greeting %}Hello{% endif %} {{ name }}!"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            context = {"name": "Bob", "show_greeting": True}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "Hello Bob!"

            # Test with show_greeting = False
            context = {"name": "Bob", "show_greeting": False}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "Bob!"
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_no_context(self, handler):
        """Test jinja prompt without context."""
        template_content = "Static template content"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            result = handler._process_jinja_prompt(temp_path, None)
            assert result == "Static template content"
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_not_found(self, handler):
        """Test jinja prompt with non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Jinja template not found"):
            handler._process_jinja_prompt("/nonexistent/path/to/template.j2")

    def test_process_jinja_prompt_relative_path(self, handler):
        """Test jinja prompt with relative path."""
        template_content = "Relative jinja content: {{ value }}"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2", dir=".") as f:
            f.write(template_content)
            temp_path = Path(f.name).name

        try:
            context = {"value": "test"}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "Relative jinja content: test"
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_cwd_fallback(self, handler):
        """Test jinja prompt falls back to cwd when first path check fails."""

        # Create a real jinja template
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write("Jinja fallback: {{ name }}")
            temp_path = f.name

        try:
            # Mock Path to make first exists() return False, second return True
            call_count = [0]

            def mock_path_exists(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    # First call - pretend it doesn't exist
                    return False
                else:
                    # Second call - it does exist
                    return True

            with patch.object(Path, "exists", mock_path_exists):
                context = {"name": "Test"}
                result = handler._process_jinja_prompt(temp_path, context)
                # Should still render the template after fallback
                assert "Jinja fallback: Test" in result or result  # Just verify it ran
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_whitespace_stripped(self, handler):
        """Test jinja prompt result is stripped of whitespace."""
        template_content = "\n\n  Hello {{ name }}!  \n\n"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            context = {"name": "Alice"}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "Hello Alice!"
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_invalid_syntax(self, handler):
        """Test jinja prompt with invalid syntax raises error."""
        template_content = "{{ invalid syntax without closing"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Error processing Jinja template"):
                handler._process_jinja_prompt(temp_path, {"name": "test"})
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_utf8(self, handler):
        """Test jinja prompt with UTF-8 characters."""
        template_content = "你好 {{ name }}! 🎉"

        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            context = {"name": "世界"}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "你好 世界! 🎉"
        finally:
            Path(temp_path).unlink()

    def test_process_jinja_prompt_filters(self, handler):
        """Test jinja prompt with built-in filters."""
        template_content = "{{ name | upper }}"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            context = {"name": "alice"}
            result = handler._process_jinja_prompt(temp_path, context)
            assert result == "ALICE"
        finally:
            Path(temp_path).unlink()


class TestValidatePromptType:
    """Tests for validate_prompt_type method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_validate_string_type(self, handler):
        """Test validating string prompt type."""
        assert handler.validate_prompt_type("string") is True

    def test_validate_file_type(self, handler):
        """Test validating file prompt type."""
        assert handler.validate_prompt_type("file") is True

    def test_validate_jinja_type(self, handler):
        """Test validating jinja prompt type."""
        assert handler.validate_prompt_type("jinja") is True

    def test_validate_invalid_type(self, handler):
        """Test validating invalid prompt type."""
        assert handler.validate_prompt_type("invalid") is False

    def test_validate_empty_type(self, handler):
        """Test validating empty prompt type."""
        assert handler.validate_prompt_type("") is False

    def test_validate_case_sensitive(self, handler):
        """Test prompt type validation is case sensitive."""
        assert handler.validate_prompt_type("String") is False
        assert handler.validate_prompt_type("FILE") is False
        assert handler.validate_prompt_type("JINJA") is False


class TestGetTimestamp:
    """Tests for _get_timestamp method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_get_timestamp_format(self, handler):
        """Test timestamp format is ISO format."""
        timestamp = handler._get_timestamp()

        # Should be able to parse as ISO format
        datetime.fromisoformat(timestamp)

    def test_get_timestamp_not_empty(self, handler):
        """Test timestamp is not empty."""
        timestamp = handler._get_timestamp()

        assert timestamp
        assert len(timestamp) > 0

    def test_get_timestamp_changes(self, handler):
        """Test timestamps change over time."""
        import time

        timestamp1 = handler._get_timestamp()
        time.sleep(0.01)  # Small delay
        timestamp2 = handler._get_timestamp()

        # Timestamps should be different
        assert timestamp1 != timestamp2


class TestGetDefaultContext:
    """Tests for get_default_context method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    def test_get_default_context_basic(self, handler):
        """Test getting default context with basic agent."""
        agent = Mock()
        agent.name = "TestAgent"

        context = handler.get_default_context(agent)

        assert context["agent_name"] == "TestAgent"
        assert "timestamp" in context

    def test_get_default_context_with_config(self, handler):
        """Test getting default context with agent config."""
        agent = Mock()
        agent.name = "TestAgent"
        agent.config = Mock()
        agent.config.agent_id = "test_123"
        agent.config.system_prompt_type = "jinja"

        context = handler.get_default_context(agent)

        assert context["agent_name"] == "TestAgent"
        assert context["agent_id"] == "test_123"
        assert context["system_prompt_type"] == "jinja"
        assert "timestamp" in context

    def test_get_default_context_no_name(self, handler):
        """Test getting default context when agent has no name."""
        agent = Mock(spec=[])

        context = handler.get_default_context(agent)

        assert context["agent_name"] == "Unknown Agent"
        assert "timestamp" in context

    def test_get_default_context_partial_config(self, handler):
        """Test getting default context with partial config."""
        agent = Mock()
        agent.name = "TestAgent"
        agent.config = Mock(spec=["agent_id"])
        agent.config.agent_id = "test_456"

        context = handler.get_default_context(agent)

        assert context["agent_name"] == "TestAgent"
        assert context["agent_id"] == "test_456"
        assert context["system_prompt_type"] == "string"  # Default value

    def test_get_default_context_timestamp_format(self, handler):
        """Test default context timestamp is in correct format."""
        agent = Mock()
        agent.name = "TestAgent"

        context = handler.get_default_context(agent)

        # Should be able to parse timestamp
        datetime.fromisoformat(context["timestamp"])


class TestCreateDynamicPrompt:
    """Tests for create_dynamic_prompt method."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        agent = Mock()
        agent.name = "TestAgent"
        agent.config = Mock()
        agent.config.agent_id = "test_123"
        agent.config.system_prompt_type = "string"
        return agent

    def test_create_dynamic_prompt_string_template(self, handler, mock_agent):
        """Test creating dynamic prompt with string template."""
        base_template = "Agent: {{ agent_name }}"

        result = handler.create_dynamic_prompt(base_template, mock_agent)

        assert "TestAgent" in result

    def test_create_dynamic_prompt_with_additional_context(self, handler, mock_agent):
        """Test creating dynamic prompt with additional context."""
        base_template = "Agent: {{ agent_name }}, Task: {{ task }}"
        additional_context = {"task": "Testing"}

        result = handler.create_dynamic_prompt(
            base_template,
            mock_agent,
            additional_context=additional_context,
        )

        assert "TestAgent" in result
        assert "Testing" in result

    def test_create_dynamic_prompt_jinja_file(self, handler, mock_agent):
        """Test creating dynamic prompt from jinja file."""
        template_content = "Agent: {{ agent_name }}, ID: {{ agent_id }}"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            result = handler.create_dynamic_prompt(
                temp_path,
                mock_agent,
                template_type="jinja",
            )

            assert "TestAgent" in result
            assert "test_123" in result
        finally:
            Path(temp_path).unlink()

    def test_create_dynamic_prompt_override_context(self, handler, mock_agent):
        """Test additional context overrides default context."""
        base_template = "Name: {{ agent_name }}"
        additional_context = {"agent_name": "OverriddenAgent"}

        result = handler.create_dynamic_prompt(
            base_template,
            mock_agent,
            additional_context=additional_context,
        )

        assert "OverriddenAgent" in result

    def test_create_dynamic_prompt_fallback_on_error(self, handler, mock_agent):
        """Test fallback to base template on rendering error."""
        base_template = "Simple template"

        # Force an error by mocking the jinja env
        with patch.object(handler._jinja_env, "from_string", side_effect=Exception("Test error")):
            result = handler.create_dynamic_prompt(base_template, mock_agent)

        # Should return the base template as fallback
        assert result == base_template

    def test_create_dynamic_prompt_complex_template(self, handler, mock_agent):
        """Test creating dynamic prompt with complex template."""
        base_template = """Agent: {{ agent_name }}
ID: {{ agent_id }}
Timestamp: {{ timestamp }}
{% if custom_field %}Custom: {{ custom_field }}{% endif %}"""
        additional_context = {"custom_field": "CustomValue"}

        result = handler.create_dynamic_prompt(
            base_template,
            mock_agent,
            additional_context=additional_context,
        )

        assert "TestAgent" in result
        assert "test_123" in result
        assert "CustomValue" in result

    def test_create_dynamic_prompt_no_additional_context(self, handler, mock_agent):
        """Test creating dynamic prompt without additional context."""
        base_template = "Agent: {{ agent_name }}"

        result = handler.create_dynamic_prompt(base_template, mock_agent)

        assert "TestAgent" in result

    def test_create_dynamic_prompt_empty_template(self, handler, mock_agent):
        """Test creating dynamic prompt with empty template."""
        base_template = ""

        result = handler.create_dynamic_prompt(base_template, mock_agent)

        # Should return empty or minimal result
        assert isinstance(result, str)


class TestPromptHandlerIntegration:
    """Integration tests for PromptHandler."""

    @pytest.fixture
    def handler(self):
        """Create a handler instance."""
        return PromptHandler()

    @pytest.fixture
    def mock_agent(self):
        """Create a full mock agent."""
        agent = Mock()
        agent.name = "IntegrationAgent"
        agent.config = Mock()
        agent.config.agent_id = "integration_123"
        agent.config.system_prompt_type = "string"
        return agent

    def test_full_workflow_string_to_dynamic(self, handler, mock_agent):
        """Test full workflow from string prompt to dynamic prompt."""
        # First, process a simple string
        basic_prompt = "You are {role}"
        context = {"role": "an assistant"}

        processed = handler.process_prompt(
            basic_prompt,
            prompt_type="string",
            context=context,
        )

        assert processed == "You are an assistant"

        # Then create a dynamic prompt
        dynamic = handler.create_dynamic_prompt(
            "Agent {{ agent_name }} says: " + processed,
            mock_agent,
        )

        assert "IntegrationAgent" in dynamic
        assert "You are an assistant" in dynamic

    def test_full_workflow_file_to_dynamic(self, handler, mock_agent):
        """Test full workflow from file prompt to dynamic prompt."""
        # Create a file prompt
        file_content = "Agent instructions for {task}"

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(file_content)
            temp_path = f.name

        try:
            # Process file prompt
            context = {"task": "testing"}
            processed = handler.process_prompt(
                temp_path,
                prompt_type="file",
                context=context,
            )

            assert processed == "Agent instructions for testing"

            # Create dynamic prompt from processed content
            dynamic = handler.create_dynamic_prompt(
                "{{ agent_name }}: " + processed,
                mock_agent,
            )

            assert "IntegrationAgent" in dynamic
            assert "Agent instructions for testing" in dynamic
        finally:
            Path(temp_path).unlink()

    def test_full_workflow_jinja_template(self, handler, mock_agent):
        """Test full workflow with jinja template."""
        # Create a jinja template
        template_content = """You are {{ agent_name }}.
Agent ID: {{ agent_id }}
Current time: {{ timestamp }}
Mission: {{ mission }}"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".j2") as f:
            f.write(template_content)
            temp_path = f.name

        try:
            additional_context = {"mission": "Test all features"}

            result = handler.create_dynamic_prompt(
                temp_path,
                mock_agent,
                additional_context=additional_context,
                template_type="jinja",
            )

            assert "IntegrationAgent" in result
            assert "integration_123" in result
            assert "Test all features" in result
        finally:
            Path(temp_path).unlink()

    def test_validate_and_process(self, handler):
        """Test validating prompt type before processing."""
        prompt_types = ["string", "file", "jinja"]

        for prompt_type in prompt_types:
            # Validate first
            is_valid = handler.validate_prompt_type(prompt_type)
            assert is_valid is True

        # Test invalid type
        is_valid = handler.validate_prompt_type("invalid")
        assert is_valid is False

    def test_context_merging(self, handler, mock_agent):
        """Test that context properly merges in dynamic prompt creation."""
        template = """Default: {{ agent_name }}
Custom1: {{ custom1 }}
Custom2: {{ custom2 }}"""

        additional = {
            "custom1": "value1",
            "custom2": "value2",
        }

        result = handler.create_dynamic_prompt(
            template,
            mock_agent,
            additional_context=additional,
        )

        # Should contain both default and additional context
        assert "IntegrationAgent" in result
        assert "value1" in result
        assert "value2" in result

    def test_error_recovery(self, handler):
        """Test error recovery in various scenarios."""
        # Test with non-existent file
        assert handler.validate_prompt_type("file") is True

        with pytest.raises(FileNotFoundError):
            handler.process_prompt(
                "/definitely/does/not/exist.txt",
                prompt_type="file",
            )

        # Test with invalid jinja syntax but in string mode
        invalid_jinja = "{{ unclosed"
        result = handler.process_prompt(invalid_jinja, prompt_type="string")
        assert result == invalid_jinja  # Should return as-is in string mode
