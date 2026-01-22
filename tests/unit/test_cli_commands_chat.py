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

"""Unit tests for CLI chat command."""

import argparse
from unittest.mock import Mock, patch

import pytest
import yaml

from nexau.cli.commands import chat as chat_cmd


class TestSetupParser:
    """Test cases for setup_parser function."""

    def test_setup_parser_adds_agent_argument(self):
        """Test setup_parser adds agent positional argument."""
        parser = argparse.ArgumentParser()
        chat_cmd.setup_parser(parser)

        # Parse with required argument
        args = parser.parse_args(["test_config.yaml"])
        assert args.agent == "test_config.yaml"
        assert args.query is None
        assert args.user_id == "cli_user"
        assert args.session_id is None
        assert args.verbose is False

    def test_setup_parser_adds_query_argument(self):
        """Test setup_parser adds --query optional argument."""
        parser = argparse.ArgumentParser()
        chat_cmd.setup_parser(parser)

        args = parser.parse_args(["config.yaml", "--query", "test query"])
        assert args.agent == "config.yaml"
        assert args.query == "test query"

    def test_setup_parser_adds_user_id_argument(self):
        """Test setup_parser adds --user-id optional argument."""
        parser = argparse.ArgumentParser()
        chat_cmd.setup_parser(parser)

        args = parser.parse_args(["config.yaml", "--user-id", "alice"])
        assert args.user_id == "alice"

    def test_setup_parser_adds_session_id_argument(self):
        """Test setup_parser adds --session-id optional argument."""
        parser = argparse.ArgumentParser()
        chat_cmd.setup_parser(parser)

        args = parser.parse_args(["config.yaml", "--session-id", "sess_123"])
        assert args.session_id == "sess_123"

    def test_setup_parser_adds_verbose_argument(self):
        """Test setup_parser adds --verbose optional argument."""
        parser = argparse.ArgumentParser()
        chat_cmd.setup_parser(parser)

        args = parser.parse_args(["config.yaml", "--verbose"])
        assert args.verbose is True


class TestBuildAgent:
    """Test cases for build_agent function."""

    def test_build_agent_success(self, tmp_path):
        """Test build_agent creates agent from valid config."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("nexau.cli.commands.chat.Agent") as mock_agent:
            mock_agent.return_value = Mock()
            mock_session_manager = Mock()
            agent = chat_cmd.build_agent(config_path, mock_session_manager, "user1", "sess1")
            assert agent is not None
            mock_agent.assert_called_once()

    def test_build_agent_invalid_config_type(self, tmp_path):
        """Test build_agent raises error for non-dict config."""
        from nexau.archs.main_sub.config import ConfigError

        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            f.write("- item1\n- item2")  # List instead of dict

        mock_session_manager = Mock()
        with pytest.raises(ConfigError, match="must be a YAML object/mapping"):
            chat_cmd.build_agent(config_path, mock_session_manager, "user1", None)


class TestMainNonInteractive:
    """Test cases for main_non_interactive function."""

    def test_main_non_interactive_success(self, tmp_path):
        """Test main_non_interactive returns 0 on success."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("nexau.cli.commands.chat.build_agent") as mock_build:
            mock_agent = Mock()
            mock_agent.run.return_value = "Test response"
            mock_build.return_value = mock_agent
            mock_session_manager = Mock()

            result = chat_cmd.main_non_interactive(config_path, "test query", mock_session_manager, "user1", None)

            assert result == 0
            mock_agent.run.assert_called_once()

    def test_main_non_interactive_config_error(self, tmp_path):
        """Test main_non_interactive returns 1 on ConfigError."""
        from nexau.archs.main_sub.config import ConfigError

        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: [")

        with patch("nexau.cli.commands.chat.build_agent") as mock_build:
            mock_build.side_effect = ConfigError("Invalid config")
            mock_session_manager = Mock()

            result = chat_cmd.main_non_interactive(config_path, "test query", mock_session_manager, "user1", None)

            assert result == 1

    def test_main_non_interactive_general_error(self, tmp_path):
        """Test main_non_interactive returns 1 on general error."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with patch("nexau.cli.commands.chat.build_agent") as mock_build:
            mock_build.side_effect = Exception("Unexpected error")
            mock_session_manager = Mock()

            result = chat_cmd.main_non_interactive(config_path, "test query", mock_session_manager, "user1", None)

            assert result == 1


class TestMain:
    """Test cases for main function."""

    def test_main_file_not_found(self, tmp_path):
        """Test main returns 1 when config file not found."""
        args = argparse.Namespace(
            agent=str(tmp_path / "nonexistent.yaml"),
            query=None,
            user_id="cli_user",
            session_id=None,
            verbose=False,
        )

        result = chat_cmd.main(args)

        assert result == 1

    def test_main_non_interactive_mode(self, tmp_path):
        """Test main calls main_non_interactive when query provided."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        args = argparse.Namespace(
            agent=str(config_path),
            query="test query",
            user_id="cli_user",
            session_id=None,
            verbose=False,
        )

        with (
            patch("nexau.cli.commands.chat.main_non_interactive") as mock_non_interactive,
            patch("nexau.cli.commands.chat.SQLDatabaseEngine"),
            patch("nexau.cli.commands.chat.SessionManager"),
        ):
            mock_non_interactive.return_value = 0

            result = chat_cmd.main(args)

            assert result == 0
            mock_non_interactive.assert_called_once()

    def test_main_interactive_mode(self, tmp_path):
        """Test main calls main_interactive when no query provided."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        args = argparse.Namespace(
            agent=str(config_path),
            query=None,
            user_id="cli_user",
            session_id=None,
            verbose=False,
        )

        with (
            patch("nexau.cli.commands.chat.main_interactive") as mock_interactive,
            patch("nexau.cli.commands.chat.SQLDatabaseEngine"),
            patch("nexau.cli.commands.chat.SessionManager"),
        ):
            mock_interactive.return_value = 0

            result = chat_cmd.main(args)

            assert result == 0
            mock_interactive.assert_called_once()


class TestMainInteractive:
    """Test cases for main_interactive function."""

    def test_main_interactive_config_error(self, tmp_path):
        """Test main_interactive returns 1 on ConfigError."""
        from nexau.archs.main_sub.config import ConfigError

        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            f.write("name: test")

        with patch("nexau.cli.commands.chat.build_agent") as mock_build:
            mock_build.side_effect = ConfigError("Invalid config")
            mock_session_manager = Mock()

            result = chat_cmd.main_interactive(config_path, mock_session_manager, "user1", None)

            assert result == 1

    def test_main_interactive_general_error(self, tmp_path):
        """Test main_interactive returns 1 on general error."""
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            f.write("name: test")

        with patch("nexau.cli.commands.chat.build_agent") as mock_build:
            mock_build.side_effect = Exception("Unexpected error")
            mock_session_manager = Mock()

            result = chat_cmd.main_interactive(config_path, mock_session_manager, "user1", None)

            assert result == 1

    def test_main_interactive_keyboard_interrupt(self, tmp_path):
        """Test main_interactive returns 0 on KeyboardInterrupt."""
        config = {
            "name": "test_agent",
            "system_prompt": "You are a helpful assistant.",
            "llm_config": {"model": "gpt-4o-mini"},
        }
        config_path = tmp_path / "agent.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with (
            patch("nexau.cli.commands.chat.build_agent") as mock_build,
            patch("nexau.cli.commands.chat.attach_cli_to_agent"),
            patch("builtins.input") as mock_input,
        ):
            mock_agent = Mock()
            mock_agent._session_id = "test_session"
            mock_build.return_value = mock_agent
            mock_input.side_effect = KeyboardInterrupt()
            mock_session_manager = Mock()

            result = chat_cmd.main_interactive(config_path, mock_session_manager, "user1", None)

            assert result == 0
