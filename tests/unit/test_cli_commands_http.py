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

"""Unit tests for HTTP CLI commands."""

import argparse

from nexau.cli.commands.http import (
    ServerArgs,
    setup_parser,
)


class TestServerArgs:
    """Test cases for ServerArgs model."""

    def test_valid_server_args(self):
        """Test valid server arguments."""
        args = ServerArgs(
            agent="/path/to/agent.yaml",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            cors_origins=["*"],
        )
        assert args.agent == "/path/to/agent.yaml"
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.log_level == "info"
        assert args.cors_origins == ["*"]

    def test_server_args_with_custom_values(self):
        """Test server arguments with custom values."""
        args = ServerArgs(
            agent="/path/to/agent.yaml",
            host="localhost",
            port=9000,
            log_level="debug",
            cors_origins=["http://localhost:3000"],
        )
        assert args.host == "localhost"
        assert args.port == 9000
        assert args.log_level == "debug"


class TestSetupParser:
    """Test cases for setup_parser function."""

    def test_setup_parser_requires_agent(self):
        """Test server parser requires agent argument."""
        parser = argparse.ArgumentParser()
        setup_parser(parser)

        # Agent is required positional argument
        args = parser.parse_args(["agent.yaml"])
        assert args.agent == "agent.yaml"
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.log_level == "info"
        assert args.cors_origins == ["*"]
        assert args.verbose is False

    def test_setup_parser_with_all_args(self):
        """Test server parser with all arguments."""
        parser = argparse.ArgumentParser()
        setup_parser(parser)

        args = parser.parse_args(
            [
                "agent.yaml",
                "--host",
                "localhost",
                "--port",
                "9000",
                "--log-level",
                "debug",
                "--cors-origins",
                "http://localhost:3000",
                "http://localhost:3001",
                "--verbose",
            ]
        )
        assert args.agent == "agent.yaml"
        assert args.host == "localhost"
        assert args.port == 9000
        assert args.log_level == "debug"
        assert args.cors_origins == ["http://localhost:3000", "http://localhost:3001"]
        assert args.verbose is True


class TestMain:
    """Test cases for main function."""

    def test_main_with_invalid_agent_config(self):
        """Test main with non-existent agent configuration."""
        from unittest.mock import patch

        from nexau.cli.commands.http import main

        with patch("nexau.cli.commands.http.load_dotenv"):
            args = argparse.Namespace(
                agent="/nonexistent/path/agent.yaml",
                host="0.0.0.0",
                port=8000,
                log_level="info",
                cors_origins=["*"],
                verbose=False,
            )

            result = main(args)

            assert result == 1

    def test_main_with_valid_agent_config(self):
        """Test main with valid agent configuration."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from nexau.cli.commands.http import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.http.load_dotenv"):
                with patch("nexau.cli.commands.http.SSETransportServer") as mock_server_class:
                    mock_server = MagicMock()
                    mock_server.run.return_value = None
                    mock_server.host = "0.0.0.0"
                    mock_server.port = 8000
                    mock_server.health_url = "http://0.0.0.0:8000/health"
                    mock_server.info_url = "http://0.0.0.0:8000/info"
                    mock_server_class.return_value = mock_server

                    args = argparse.Namespace(
                        agent=config_path,
                        host="0.0.0.0",
                        port=8000,
                        log_level="info",
                        cors_origins=["*"],
                        verbose=False,
                    )

                    result = main(args)

                    assert result == 0
        finally:
            Path(config_path).unlink()

    def test_main_verbose_logging(self):
        """Test main with verbose logging."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from nexau.cli.commands.http import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.http.load_dotenv"):
                with patch("nexau.cli.commands.http.SSETransportServer") as mock_server_class:
                    with patch("logging.basicConfig") as mock_logging:
                        mock_server = MagicMock()
                        mock_server.run.return_value = None
                        mock_server.host = "0.0.0.0"
                        mock_server.port = 8000
                        mock_server.health_url = "http://0.0.0.0:8000/health"
                        mock_server.info_url = "http://0.0.0.0:8000/info"
                        mock_server_class.return_value = mock_server

                        args = argparse.Namespace(
                            agent=config_path,
                            host="0.0.0.0",
                            port=8000,
                            log_level="info",
                            cors_origins=["*"],
                            verbose=True,
                        )

                        result = main(args)

                        assert result == 0
                        mock_logging.assert_called_once()
        finally:
            Path(config_path).unlink()

    def test_main_keyboard_interrupt(self):
        """Test main handles KeyboardInterrupt."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from nexau.cli.commands.http import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.http.load_dotenv"):
                with patch("nexau.cli.commands.http.SSETransportServer") as mock_server_class:
                    mock_server = MagicMock()
                    mock_server.run.side_effect = KeyboardInterrupt()
                    mock_server.host = "0.0.0.0"
                    mock_server.port = 8000
                    mock_server.health_url = "http://0.0.0.0:8000/health"
                    mock_server.info_url = "http://0.0.0.0:8000/info"
                    mock_server_class.return_value = mock_server

                    args = argparse.Namespace(
                        agent=config_path,
                        host="0.0.0.0",
                        port=8000,
                        log_level="info",
                        cors_origins=["*"],
                        verbose=False,
                    )

                    result = main(args)

                    assert result == 0
        finally:
            Path(config_path).unlink()

    def test_main_exception(self):
        """Test main handles exceptions."""
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from nexau.cli.commands.http import main

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test_agent\n")
            f.write("llm_config:\n")
            f.write("  provider: openai\n")
            f.write("  model: gpt-4\n")
            config_path = f.name

        try:
            with patch("nexau.cli.commands.http.load_dotenv"):
                with patch("nexau.cli.commands.http.SSETransportServer") as mock_server_class:
                    mock_server = MagicMock()
                    mock_server.run.side_effect = Exception("Server error")
                    mock_server.host = "0.0.0.0"
                    mock_server.port = 8000
                    mock_server.health_url = "http://0.0.0.0:8000/health"
                    mock_server.info_url = "http://0.0.0.0:8000/info"
                    mock_server_class.return_value = mock_server

                    args = argparse.Namespace(
                        agent=config_path,
                        host="0.0.0.0",
                        port=8000,
                        log_level="info",
                        cors_origins=["*"],
                        verbose=False,
                    )

                    result = main(args)

                    assert result == 1
        finally:
            Path(config_path).unlink()
