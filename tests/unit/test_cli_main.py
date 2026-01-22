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

"""Unit tests for CLI main module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nexau.cli.main import create_parser, main


class TestCreateParser:
    """Test cases for create_parser function."""

    def test_create_parser_structure(self):
        """Test that parser is created with correct structure."""
        parser = create_parser()
        assert parser.prog == "nexau"
        assert "NexAU Agent Framework CLI" in parser.description

    def test_parser_has_chat_command(self):
        """Test that parser has chat command."""
        parser = create_parser()
        args = parser.parse_args(["chat", "test.yaml"])
        assert args.command == "chat"
        assert hasattr(args, "func")

    def test_parser_has_serve_command(self):
        """Test that parser has serve command group."""
        parser = create_parser()

        # Test serve http
        args = parser.parse_args(["serve", "http", "agent.yaml"])
        assert args.command == "serve"
        assert args.transport == "http"
        assert hasattr(args, "func")

    def test_parser_has_serve_http_command(self):
        """Test that parser has serve http command."""
        parser = create_parser()
        args = parser.parse_args(["serve", "http", "agent.yaml"])
        assert args.command == "serve"
        assert args.transport == "http"
        assert args.agent == "agent.yaml"

    def test_parser_has_serve_stdio_command(self):
        """Test that parser has serve stdio command."""
        parser = create_parser()
        args = parser.parse_args(["serve", "stdio", "agent.yaml"])
        assert args.command == "serve"
        assert args.transport == "stdio"
        assert args.agent == "agent.yaml"

    def test_parser_no_command_defaults(self):
        """Test parser behavior with no command."""
        parser = create_parser()
        args = parser.parse_args([])
        assert not hasattr(args, "func")


class TestMain:
    """Test cases for main function."""

    def test_main_no_command_shows_help(self):
        """Test that main shows help when no command is provided."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = MagicMock(spec=[])  # No 'func' attribute
            mock_create_parser.return_value = mock_parser

            result = main([])

            assert result == 1
            mock_parser.print_help.assert_called_once()

    def test_main_with_valid_command(self):
        """Test main with valid command."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(return_value=0)
            mock_args = MagicMock()
            mock_args.func = mock_func

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["test", "command"])

            assert result == 0
            mock_func.assert_called_once_with(mock_args)

    def test_main_keyboard_interrupt(self):
        """Test main handles KeyboardInterrupt."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(side_effect=KeyboardInterrupt())
            mock_args = MagicMock()
            mock_args.func = mock_func

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["test"])

            assert result == 130

    def test_main_exception_handling(self):
        """Test main handles exceptions."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(side_effect=Exception("Test error"))
            mock_args = MagicMock()
            mock_args.func = mock_func

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["test"])

            assert result == 1

    def test_main_uses_sys_argv_by_default(self):
        """Test that main uses sys.argv when argv is None."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = MagicMock(spec=[])
            mock_create_parser.return_value = mock_parser

            main(None)

            # parse_args should be called with None (which uses sys.argv)
            mock_parser.parse_args.assert_called_once_with(None)

    def test_main_command_returns_success(self):
        """Test main returns success code from command."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(return_value=0)
            mock_args = MagicMock()
            mock_args.func = mock_func

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["test"])

            assert result == 0

    def test_main_command_returns_error(self):
        """Test main returns error code from command."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(return_value=1)
            mock_args = MagicMock()
            mock_args.func = mock_func

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["test"])

            assert result == 1

    def test_main_with_chat_command(self):
        """Test main with chat command."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(return_value=0)
            mock_args = MagicMock()
            mock_args.func = mock_func
            mock_args.command = "chat"

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["chat", "test.yaml"])

            assert result == 0
            mock_func.assert_called_once()

    def test_main_with_serve_http_command(self):
        """Test main with serve http command."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(return_value=0)
            mock_args = MagicMock()
            mock_args.func = mock_func
            mock_args.command = "serve"
            mock_args.transport = "http"

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["serve", "http", "agent.yaml"])

            assert result == 0
            mock_func.assert_called_once()

    def test_main_with_serve_stdio_command(self):
        """Test main with serve stdio command."""
        with patch("nexau.cli.main.create_parser") as mock_create_parser:
            mock_func = MagicMock(return_value=0)
            mock_args = MagicMock()
            mock_args.func = mock_func
            mock_args.command = "serve"
            mock_args.transport = "stdio"

            mock_parser = MagicMock()
            mock_parser.parse_args.return_value = mock_args
            mock_create_parser.return_value = mock_parser

            result = main(["serve", "stdio", "agent.yaml"])

            assert result == 0
            mock_func.assert_called_once()
