"""RFC-0027 workflow expression tests."""

from __future__ import annotations

import pytest

from nexau.archs.workflow.expression import WorkflowExpressionError, evaluate_condition, evaluate_expression, render_template
from nexau.archs.workflow.types import JsonObject


def _context() -> JsonObject:
    return {
        "inputs": {"requirement": "checkout"},
        "vars": {"max_cases": 2},
        "state": {
            "remaining_cases": [{"id": "C1"}, {"id": "C2"}],
            "results": [{"case_id": "C0"}],
        },
        "nodes": {
            "review": {"status": "completed", "output": {"approved": True}},
            "run_one": {"status": "completed", "output": {"case_id": "C1"}},
        },
    }


def test_expression_supports_context_access_slicing_length_and_concat() -> None:
    context = _context()

    assert evaluate_expression("inputs.requirement", context) == "checkout"
    assert evaluate_expression("vars.max_cases", context) == 2
    assert evaluate_expression("state.remaining_cases[0].id", context) == "C1"
    assert evaluate_expression("state.remaining_cases[1:]", context) == [{"id": "C2"}]
    assert evaluate_expression("state.remaining_cases.length", context) == 2
    assert evaluate_expression("state.remaining_cases.size()", context) == 2
    assert evaluate_expression("state.results + [nodes.run_one.output]", context) == [
        {"case_id": "C0"},
        {"case_id": "C1"},
    ]
    assert evaluate_condition("nodes.review.output.approved && vars.max_cases == 2", context)
    assert evaluate_condition("!false", context)


def test_render_template_preserves_full_expression_type_and_interpolates_embedded_text() -> None:
    context = _context()

    assert render_template("{{ state.remaining_cases[0] }}", context) == {"id": "C1"}
    assert render_template("case-{{ state.remaining_cases[0].id }}", context) == "case-C1"
    assert render_template({"case": "{{ state.remaining_cases[0] }}"}, context) == {"case": {"id": "C1"}}


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "open('/tmp/x')",
        "[item for item in state.remaining_cases]",
        "unknown.value",
        "state.remaining_cases[0].missing",
    ],
)
def test_expression_rejects_unsafe_or_invalid_constructs(expression: str) -> None:
    with pytest.raises(WorkflowExpressionError):
        evaluate_expression(expression, _context())
