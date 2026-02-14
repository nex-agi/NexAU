# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""
ask_user tool - Asks the user questions to gather preferences or clarify requirements.

Based on gemini-cli's ask-user.ts implementation.
Supports multiple question types: choice, text, and yesno.
"""

import json
from typing import Any

# Question types
QUESTION_TYPE_CHOICE = "choice"
QUESTION_TYPE_TEXT = "text"
QUESTION_TYPE_YESNO = "yesno"


def ask_user(
    questions: list[dict[str, Any]],
    user_answers: dict[str, str] | None = None,
    was_cancelled: bool = False,
) -> dict[str, Any]:
    """
    Ask the user one or more questions to gather preferences, clarify requirements,
    or make decisions.

    Question types:
    - choice: Multiple-choice with options (default). Requires 2-4 options.
    - text: Free-form text input.
    - yesno: Yes/No confirmation.

    Args:
        questions: List of question objects, each containing:
            - question: The complete question to ask
            - header: Short label (max 32 chars) displayed as chip/tag
            - type: Question type (choice/text/yesno), defaults to choice
            - options: For choice type, list of {label, description} objects
            - multiSelect: For choice type, allow multiple selections
            - placeholder: For text type, hint text
        user_answers: Dict mapping question index to user's answer
        was_cancelled: Whether user dismissed the dialog

    Returns:
        Dict with result and returnDisplay matching gemini-cli format
    """
    try:
        # Validate questions
        if not questions:
            return {
                "content": "At least one question is required.",
                "returnDisplay": "Error: No questions provided.",
                "error": {
                    "message": "At least one question is required.",
                    "type": "INVALID_PARAMETER",
                },
            }

        if len(questions) > 4:
            return {
                "content": "Maximum 4 questions allowed.",
                "returnDisplay": "Error: Too many questions.",
                "error": {
                    "message": "Maximum 4 questions allowed.",
                    "type": "INVALID_PARAMETER",
                },
            }

        # Validate each question
        for i, q in enumerate(questions):
            question_type = q.get("type", QUESTION_TYPE_CHOICE)

            # Validate required fields
            if not q.get("question"):
                return {
                    "content": f"Question {i + 1}: 'question' is required.",
                    "returnDisplay": "Error: Missing question text.",
                    "error": {
                        "message": f"Question {i + 1}: 'question' is required.",
                        "type": "INVALID_PARAMETER",
                    },
                }

            if not q.get("header"):
                return {
                    "content": f"Question {i + 1}: 'header' is required.",
                    "returnDisplay": "Error: Missing header.",
                    "error": {
                        "message": f"Question {i + 1}: 'header' is required.",
                        "type": "INVALID_PARAMETER",
                    },
                }

            # Validate header length
            if len(q.get("header", "")) > 32:
                return {
                    "content": f"Question {i + 1}: 'header' must be at most 32 characters.",
                    "returnDisplay": "Error: Header too long.",
                    "error": {
                        "message": f"Question {i + 1}: 'header' must be at most 32 characters.",
                        "type": "INVALID_PARAMETER",
                    },
                }

            # Validate choice type has options
            if question_type == QUESTION_TYPE_CHOICE:
                options = q.get("options", [])
                if not options or len(options) < 2:
                    return {
                        "content": f"Question {i + 1}: type='choice' requires 'options' array with 2-4 items.",
                        "returnDisplay": "Error: Insufficient options.",
                        "error": {
                            "message": f"Question {i + 1}: type='choice' requires 'options' array with 2-4 items.",
                            "type": "INVALID_PARAMETER",
                        },
                    }
                if len(options) > 4:
                    return {
                        "content": f"Question {i + 1}: 'options' array must have at most 4 items.",
                        "returnDisplay": "Error: Too many options.",
                        "error": {
                            "message": f"Question {i + 1}: 'options' array must have at most 4 items.",
                            "type": "INVALID_PARAMETER",
                        },
                    }

                # Validate each option
                for j, opt in enumerate(options):
                    if not opt.get("label") or not isinstance(opt.get("label"), str):
                        return {
                            "content": f"Question {i + 1}, option {j + 1}: 'label' is required and must be a non-empty string.",
                            "returnDisplay": "Error: Invalid option label.",
                            "error": {
                                "message": f"Question {i + 1}, option {j + 1}: 'label' is required.",
                                "type": "INVALID_PARAMETER",
                            },
                        }
                    if opt.get("description") is None or not isinstance(opt.get("description"), str):
                        return {
                            "content": f"Question {i + 1}, option {j + 1}: 'description' is required and must be a string.",
                            "returnDisplay": "Error: Invalid option description.",
                            "error": {
                                "message": f"Question {i + 1}, option {j + 1}: 'description' is required.",
                                "type": "INVALID_PARAMETER",
                            },
                        }

        # Handle user response
        if was_cancelled:
            return {
                "content": "User dismissed ask_user dialog without answering.",
                "returnDisplay": "User dismissed dialog",
            }

        # Process answers
        # user_answers must be provided by transport/middleware; None indicates misconfiguration
        if user_answers is None:
            raise ValueError("ask_user requires user_answers from transport. Ensure AskUserMiddleware is configured.")

        answer_entries = list(user_answers.items())
        has_answers = len(answer_entries) > 0

        if has_answers:
            answer_lines: list[str] = []
            for index_str, answer in answer_entries:
                try:
                    idx = int(index_str)
                    question = questions[idx] if idx < len(questions) else None
                    category = question.get("header", f"Q{index_str}") if question else f"Q{index_str}"
                except (ValueError, IndexError):
                    category = f"Q{index_str}"
                answer_lines.append(f"  {category} → {answer}")

            return_display = "**User answered:**\n" + "\n".join(answer_lines)
        else:
            return_display = "User submitted without answering questions."

        return {
            "content": json.dumps({"answers": user_answers}),
            "returnDisplay": return_display,
        }

    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Error in ask_user: {str(e)}"
        return {
            "content": error_msg,
            "returnDisplay": error_msg,
            "error": {
                "message": error_msg,
                "type": "ASK_USER_ERROR",
            },
        }
