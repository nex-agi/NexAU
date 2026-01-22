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

"""Unit tests for Stdio CLI commands."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from nexau.cli.commands.stdio import ServerArgs, server_main, setup_server_parser


class TestServerArgs:
    """Test cases for ServerArgs model."""

    def test_valid_server_args(self):
        """Test valid server arguments."""
        args = ServerArgs(
            agent="/path/to/agent.yaml",
            verbose=False,
        )
        assert args.agent == "/path/to/agent.yaml"
        assert args.verbose is False

    def test_server_args_with_verbose(self):
        """Test server arguments with verbose enabled."""
        args = ServerArgs(
            agent="/path/to/agent.yaml",
            verbose=True,
        )
        assert args.agent == "/path/to/agent.yaml"
        assert args.verbose is True

    def test_server_args_pydantic_validation(self):
        """Test ServerArgs pydantic validation."""
        args = ServerArgs(
            agent="test.yaml",
            verbose=False,
        )
        assert isinstance(args.agent, str)
        assert isinstance(args.verbose, bool)


class TestSetupServerParser:
    """Test cases for setup_server_parser function."""

    def test_setup_server_parser_requires_agent(self):
        """Test server parser requires agent positional argument."""
        parser = argparse.ArgumentParser()
        setup_server_parser(parser)

        args = parser.parse_args(["agent.yaml"])
        assert args.agent == "agent.yaml"
        assert args.verbose is False

    def test_setup_server_parser_with_verbose(self):
        """Test server parser with verbose flag."""
        parser = argparse.ArgumentParser()
        setup_server_parser(parser)

        args = parser.parse_args(["agent.yaml", "--verbose"])
        assert args.agent == "agent.yaml"
        assert args.verbose is True

    def test_setup_server_parser_all_options(self):
        """Test server parser with all options."""
        parser = argparse.ArgumentParser()
        setup_server_parser(parser)

        args = parser.parse_args(["my_agent.yaml", "--verbose"])
        assert args.agent == "my_agent.yaml"
        assert args.verbose is True


class TestServerMain:
    """Test cases for server_main function."""

    def test_server_main_with_valid_agent_config(self):
        """Test server_main with valid agent configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.stdio.load_dotenv"):
                with patch("nexau.cli.commands.stdio.StdioTransport") as mock_transport_class:
                    mock_transport = MagicMock()
                    mock_transport.start.return_value = None
                    mock_transport_class.return_value = mock_transport

                    args = argparse.Namespace(
                        agent=config_path,
                        verbose=False,
                    )

                    result = server_main(args)

                    assert result == 0
                    mock_transport.start.assert_called_once()
        finally:
            Path(config_path).unlink()

    def test_server_main_with_invalid_agent_config(self):
        """Test server_main with non-existent agent configuration file."""
        with patch("nexau.cli.commands.stdio.load_dotenv"):
            args = argparse.Namespace(
                agent="/nonexistent/path/agent.yaml",
                verbose=False,
            )

            result = server_main(args)

            assert result == 1

    def test_server_main_verbose_logging(self):
        """Test server_main with verbose logging enabled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.stdio.load_dotenv"):
                with patch("nexau.cli.commands.stdio.StdioTransport") as mock_transport_class:
                    with patch("logging.basicConfig") as mock_logging:
                        mock_transport = MagicMock()
                        mock_transport.start.return_value = None
                        mock_transport_class.return_value = mock_transport

                        args = argparse.Namespace(
                            agent=config_path,
                            verbose=True,
                        )

                        result = server_main(args)

                        assert result == 0
                        # Check that logging was configured with DEBUG level
                        mock_logging.assert_called_once()
                        call_kwargs = mock_logging.call_args[1]
                        assert call_kwargs["level"] == 10  # DEBUG level
        finally:
            Path(config_path).unlink()

    def test_server_main_transport_exception(self):
        """Test server_main handles transport exceptions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.stdio.load_dotenv"):
                with patch("nexau.cli.commands.stdio.StdioTransport") as mock_transport_class:
                    mock_transport = MagicMock()
                    mock_transport.start.side_effect = Exception("Transport error")
                    mock_transport_class.return_value = mock_transport

                    args = argparse.Namespace(
                        agent=config_path,
                        verbose=False,
                    )

                    result = server_main(args)

                    assert result == 1
        finally:
            Path(config_path).unlink()

    def test_server_main_creates_stdio_config(self):
        """Test server_main creates StdioConfig correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.stdio.load_dotenv"):
                with patch("nexau.cli.commands.stdio.StdioTransport") as mock_transport_class:
                    with patch("nexau.cli.commands.stdio.StdioConfig") as mock_config_class:
                        mock_transport = MagicMock()
                        mock_transport.start.return_value = None
                        mock_transport_class.return_value = mock_transport

                        mock_config = MagicMock()
                        mock_config.encoding = "utf-8"
                        mock_config_class.return_value = mock_config

                        args = argparse.Namespace(
                            agent=config_path,
                            verbose=False,
                        )

                        result = server_main(args)

                        assert result == 0
                        mock_config_class.assert_called_once()
        finally:
            Path(config_path).unlink()

    def test_server_main_creates_sql_database_engine(self):
        """Test server_main creates SQLDatabaseEngine."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.stdio.load_dotenv"):
                with patch("nexau.cli.commands.stdio.StdioTransport") as mock_transport_class:
                    with patch("nexau.cli.commands.stdio.SQLDatabaseEngine") as mock_engine_class:
                        mock_transport = MagicMock()
                        mock_transport.start.return_value = None
                        mock_transport_class.return_value = mock_transport

                        mock_engine = MagicMock()
                        mock_engine_class.from_url.return_value = mock_engine

                        args = argparse.Namespace(
                            agent=config_path,
                            verbose=False,
                        )

                        result = server_main(args)

                        assert result == 0
                        mock_engine_class.from_url.assert_called_once()
        finally:
            Path(config_path).unlink()

    def test_server_main_prints_info_to_stderr(self, capsys):
        """Test server_main prints server info to stderr."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.stdio.load_dotenv"):
                with patch("nexau.cli.commands.stdio.StdioTransport") as mock_transport_class:
                    mock_transport = MagicMock()
                    mock_transport.start.return_value = None
                    mock_transport_class.return_value = mock_transport

                    args = argparse.Namespace(
                        agent=config_path,
                        verbose=False,
                    )

                    server_main(args)

                    captured = capsys.readouterr()
                    assert "NexAU Stdio Server Starting" in captured.err
                    assert "JSON Lines" in captured.err
        finally:
            Path(config_path).unlink()
