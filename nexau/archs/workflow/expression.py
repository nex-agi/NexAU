"""Side-effect-free workflow CEL expression evaluator.

RFC-0027: 条件、循环与表达式

Expressions are evaluated by the ``common-expression-language`` package, which
wraps the Rust CEL interpreter. The workflow layer adds only YAML/template
compatibility for RFC-0027 examples: ``.length`` maps to CEL ``.size()`` and
simple slice syntax such as ``items[1:]`` maps to a safe helper function.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Callable
from typing import cast

from nexau.archs.workflow.types import JsonValue, json_value

EvaluationContext = dict[str, JsonValue]
CelEvaluate = Callable[[str, object], object]

_TEMPLATE_PATTERN = re.compile(r"{{\s*(.*?)\s*}}", re.DOTALL)
_LENGTH_PATTERN = re.compile(r"\.length\b")
_SIMPLE_SLICE_PATTERN = re.compile(
    r"(?P<target>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*|\[[^\]:]+\])*)"
    r"\[(?P<start>-?\d*)?:(?P<end>-?\d*)?(?::(?P<step>-?\d*)?)?\]"
)


class WorkflowExpressionError(ValueError):
    """Raised when a workflow expression is invalid or unsafe."""


def _load_cel_evaluate() -> CelEvaluate:
    module = importlib.import_module("cel")
    evaluate_func = module.__dict__.get("evaluate")
    if not callable(evaluate_func):
        raise WorkflowExpressionError("common-expression-language did not expose cel.evaluate")
    return cast(CelEvaluate, evaluate_func)


_CEL_EVALUATE = _load_cel_evaluate()


def evaluate_expression(expression: str, context: EvaluationContext) -> JsonValue:
    """Evaluate a CEL expression against a JSON workflow context."""

    try:
        result = _CEL_EVALUATE(_normalize_expression(expression), _evaluation_context(context))
        return json_value(result)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise WorkflowExpressionError(f"Invalid workflow expression {expression!r}: {exc}") from exc


def evaluate_condition(expression: str, context: EvaluationContext) -> bool:
    """Evaluate an expression and coerce it using workflow truthiness."""

    return _truthy(evaluate_expression(expression, context))


def render_template(value: JsonValue, context: EvaluationContext) -> JsonValue:
    """Render ``{{ ... }}`` expressions recursively.

    If a string is exactly one template expression, the expression value keeps
    its original JSON type. Embedded templates stringify the expression value.
    """

    if isinstance(value, dict):
        return {key: render_template(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [render_template(item, context) for item in value]
    if not isinstance(value, str):
        return value

    template_matches = list(_TEMPLATE_PATTERN.finditer(value))
    if len(template_matches) == 1 and template_matches[0].span() == (0, len(value)):
        return evaluate_expression(template_matches[0].group(1), context)

    def replace(match: re.Match[str]) -> str:
        evaluated = evaluate_expression(match.group(1), context)
        if isinstance(evaluated, dict | list):
            return json.dumps(evaluated, ensure_ascii=False)
        if evaluated is None:
            return ""
        return str(evaluated)

    return _TEMPLATE_PATTERN.sub(replace, value)


def _normalize_expression(expression: str) -> str:
    normalized = _LENGTH_PATTERN.sub(".size()", expression)
    return _SIMPLE_SLICE_PATTERN.sub(_replace_simple_slice, normalized)


def _replace_simple_slice(match: re.Match[str]) -> str:
    start = _slice_part(match.group("start"))
    end = _slice_part(match.group("end"))
    step = _slice_part(match.group("step"))
    return f"workflow_slice({match.group('target')}, {start}, {end}, {step})"


def _slice_part(value: str | None) -> str:
    if value is None or value == "":
        return "null"
    return value


def _evaluation_context(context: EvaluationContext) -> dict[str, object]:
    cel_context: dict[str, object] = dict(context)
    cel_context["workflow_slice"] = _workflow_slice
    return cel_context


def _workflow_slice(value: object, start: object = None, end: object = None, step: object = None) -> JsonValue:
    if not isinstance(value, list | str):
        raise WorkflowExpressionError("Slicing requires a list or string")
    slice_obj = slice(_slice_index(start), _slice_index(end), _slice_index(step))
    if isinstance(value, str):
        return value[slice_obj]
    value_list = cast(list[object], value)
    return json_value(value_list[slice_obj])


def _slice_index(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkflowExpressionError("Slice indexes must be integers")
    return value


def _truthy(value: JsonValue) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return bool(value)
