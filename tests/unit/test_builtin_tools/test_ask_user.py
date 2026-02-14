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

"""Unit tests for ask_user builtin tool."""

import pytest

from nexau.archs.tool.builtin.session_tools import ask_user


class TestAskUser:
    """Test ask_user tool functionality."""

    def test_validate_questions_required(self):
        """Should return error when questions list is empty."""
        result = ask_user(questions=[])
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_validate_question_text_required(self):
        """Should return error when question text is missing."""
        result = ask_user(questions=[{"header": "Test"}])
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "question" in result["content"].lower()

    def test_validate_header_required(self):
        """Should return error when header is missing."""
        result = ask_user(questions=[{"question": "What is your name?"}])
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "header" in result["content"].lower()

    def test_validate_header_max_length(self):
        """Should return error when header exceeds 32 characters."""
        result = ask_user(
            questions=[
                {
                    "question": "What is your name?",
                    "header": "This is an extremely long header that exceeds thirty two chars",
                    "type": "text",
                }
            ],
            user_answers={},
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "32" in result["content"]

    def test_validate_max_questions(self):
        """Should return error when more than 4 questions provided."""
        questions = [{"question": f"Q{i}", "header": f"H{i}", "type": "text"} for i in range(5)]
        result = ask_user(questions=questions)
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "4" in result["content"]

    def test_validate_choice_options_required(self):
        """Should return error when choice type has no options."""
        result = ask_user(
            questions=[
                {
                    "question": "Choose one",
                    "header": "Choice",
                    "type": "choice",
                }
            ]
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_validate_choice_options_min_count(self):
        """Should return error when choice type has less than 2 options."""
        result = ask_user(
            questions=[
                {
                    "question": "Choose one",
                    "header": "Choice",
                    "type": "choice",
                    "options": [{"label": "Only one", "description": "desc"}],
                }
            ]
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_validate_choice_options_max_count(self):
        """Should return error when choice type has more than 4 options."""
        result = ask_user(
            questions=[
                {
                    "question": "Choose one",
                    "header": "Choice",
                    "type": "choice",
                    "options": [{"label": f"Option {i}", "description": f"desc {i}"} for i in range(5)],
                }
            ]
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"


class TestAskUserOutputFormat:
    """Test ask_user output format."""

    def test_user_cancelled_format(self):
        """Should format correctly when user cancels."""
        result = ask_user(
            questions=[
                {
                    "question": "What is your name?",
                    "header": "Name",
                    "type": "text",
                }
            ],
            was_cancelled=True,
        )
        assert "dismissed" in result["content"].lower()
        assert result.get("error") is None

    def test_raises_when_user_answers_missing(self):
        """Should raise ValueError when user_answers is None (middleware not configured)."""
        with pytest.raises(ValueError, match="user_answers|AskUserMiddleware"):
            ask_user(
                questions=[
                    {
                        "question": "What is your name?",
                        "header": "Name",
                        "type": "text",
                    }
                ],
                user_answers=None,
                was_cancelled=False,
            )

    def test_user_answers_format(self):
        """Should format user answers correctly."""
        result = ask_user(
            questions=[
                {
                    "question": "What is your name?",
                    "header": "Name",
                    "type": "text",
                }
            ],
            user_answers={"0": "Alice"},
        )
        assert "answers" in result["content"]
        assert "Alice" in result["returnDisplay"]

    def test_yesno_question(self):
        """Should handle yesno type questions."""
        result = ask_user(
            questions=[
                {
                    "question": "Do you agree?",
                    "header": "Agreement",
                    "type": "yesno",
                }
            ],
            user_answers={"0": "Yes"},
        )
        assert result.get("error") is None
        assert "Yes" in result["returnDisplay"]

    def test_choice_question(self):
        """Should handle choice type questions."""
        result = ask_user(
            questions=[
                {
                    "question": "Select your language:",
                    "header": "Language",
                    "type": "choice",
                    "options": [
                        {"label": "Python", "description": "Python programming language"},
                        {"label": "JavaScript", "description": "JavaScript programming language"},
                    ],
                }
            ],
            user_answers={"0": "Python"},
        )
        assert result.get("error") is None
