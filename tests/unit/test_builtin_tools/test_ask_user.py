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

from nexau.archs.tool.builtin.session_tools import ask_user


class TestAskUserValidation:
    """Test ask_user input validation."""

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
        result = ask_user(questions=[{"question": "What?"}])
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
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"
        assert "32" in result["content"]

    def test_validate_max_questions(self):
        """More than 4 questions is allowed (no upper limit enforced)."""
        questions = [{"question": f"Q{i}", "header": f"H{i}", "type": "text"} for i in range(5)]
        result = ask_user(questions=questions)
        assert result.get("error") is None
        assert result["content"] == "Asking user questions, waiting for user answers..."

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
                    "options": [{"label": f"Opt {i}", "description": f"d {i}"} for i in range(5)],
                }
            ]
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_validate_option_label_required(self):
        """Should return error when option label is missing."""
        result = ask_user(
            questions=[
                {
                    "question": "Choose one",
                    "header": "Choice",
                    "type": "choice",
                    "options": [
                        {"label": "A", "description": "desc A"},
                        {"description": "no label"},
                    ],
                }
            ]
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"

    def test_validate_option_description_required(self):
        """Should return error when option description is missing."""
        result = ask_user(
            questions=[
                {
                    "question": "Choose one",
                    "header": "Choice",
                    "type": "choice",
                    "options": [
                        {"label": "A", "description": "desc A"},
                        {"label": "B"},
                    ],
                }
            ]
        )
        assert result.get("error") is not None
        assert result["error"]["type"] == "INVALID_PARAMETER"


class TestAskUserOutput:
    """Test ask_user output format."""

    def test_valid_text_question_returns_content(self):
        """Should return simple content string for valid text question."""
        result = ask_user(
            questions=[
                {
                    "question": "What is your name?",
                    "header": "Name",
                    "type": "text",
                }
            ],
        )
        assert result.get("error") is None
        assert result["content"] == "Asking user questions, waiting for user answers..."

    def test_valid_choice_question_returns_content(self):
        """Should return simple content string for valid choice question."""
        result = ask_user(
            questions=[
                {
                    "question": "Select your language:",
                    "header": "Language",
                    "type": "choice",
                    "options": [
                        {"label": "Python", "description": "Python lang"},
                        {"label": "JavaScript", "description": "JS lang"},
                    ],
                }
            ],
        )
        assert result.get("error") is None
        assert result["content"] == "Asking user questions, waiting for user answers..."

    def test_valid_yesno_question_returns_content(self):
        """Should return simple content string for valid yesno question."""
        result = ask_user(
            questions=[
                {
                    "question": "Do you agree?",
                    "header": "Agreement",
                    "type": "yesno",
                }
            ],
        )
        assert result.get("error") is None
        assert result["content"] == "Asking user questions, waiting for user answers..."

    def test_multiple_valid_questions_returns_content(self):
        """Should return simple content string for multiple valid questions."""
        result = ask_user(
            questions=[
                {"question": "Q1?", "header": "H1", "type": "text"},
                {"question": "Q2?", "header": "H2", "type": "yesno"},
            ],
        )
        assert result.get("error") is None
        assert result["content"] == "Asking user questions, waiting for user answers..."
