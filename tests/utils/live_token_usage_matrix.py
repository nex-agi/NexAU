from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from nexau.archs.llm.llm_config import LLMConfig
from nexau.core.usage import TokenUsage

RESULT_STATUS = Literal["PASS", "UPSTREAM_LIMITATION", "FAIL"]

SCENARIO_SINGLE_TURN = "single_turn"
SCENARIO_MULTI_TURN = "multi_turn"
SCENARIO_REASONING = "reasoning"

SINGLE_CODEWORD = "NEXAU-SINGLE-417"
MULTI_CODEWORD = "NEXAU-MULTI-582"
REASONING_EXPECTED_ANSWER = "323"
ANTHROPIC_COMPLEX_REASONING_EXPECTED_ANSWER = "2429"

REPORT_MARKER = "<!-- token-usage-live-report -->"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[2] / ".pytest_cache" / "token_usage_live_matrix_results.json"


@dataclass(frozen=True, slots=True)
class ProviderCaseConfig:
    label: str
    api_type: str
    model: str
    base_url: str
    api_key_envs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveMatrixCase:
    case_id: str
    provider_label: str
    api_type: str
    model: str
    base_url: str
    scenario: str
    api_key_envs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LiveCaseResult:
    case_id: str
    provider_label: str
    api_type: str
    model: str
    scenario: str
    status: RESULT_STATUS
    answer_check: str
    final_answer: str
    raw_usage_excerpt: dict[str, Any]
    expected_usage: dict[str, int]
    actual_usage: dict[str, int]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PROVIDERS: tuple[ProviderCaseConfig, ...] = (
    ProviderCaseConfig(
        label="responses_rayinai_gpt54",
        api_type="openai_responses",
        model="gpt-5.4",
        base_url="https://code.rayinai.com/v1",
        api_key_envs=("TOKEN_MATRIX_RAYINAI_API_KEY",),
    ),
    ProviderCaseConfig(
        label="chat_14_gpt5",
        api_type="openai_chat_completion",
        model="gpt-5",
        base_url="http://14.103.60.158:3001/v1",
        api_key_envs=("TOKEN_MATRIX_GATEWAY_OPENAI_API_KEY",),
    ),
    ProviderCaseConfig(
        label="responses_14_gpt5",
        api_type="openai_responses",
        model="gpt-5",
        base_url="http://14.103.60.158:3001/v1",
        api_key_envs=("TOKEN_MATRIX_GATEWAY_OPENAI_API_KEY",),
    ),
    ProviderCaseConfig(
        label="anthropic_14_claude46",
        api_type="anthropic_chat_completion",
        model="claude-sonnet-4-6",
        base_url="http://14.103.60.158:3001/",
        api_key_envs=("TOKEN_MATRIX_GATEWAY_ANTHROPIC_API_KEY",),
    ),
    ProviderCaseConfig(
        label="gemini_14_gemini31",
        api_type="gemini_rest",
        model="gemini-3.1-pro-preview",
        # This gateway requires /v1beta for Gemini REST requests.
        base_url="http://14.103.60.158:3001/v1beta",
        api_key_envs=("TOKEN_MATRIX_GATEWAY_GEMINI_API_KEY", "TOKEN_MATRIX_GATEWAY_OPENAI_API_KEY"),
    ),
)


def iter_live_matrix_cases() -> list[LiveMatrixCase]:
    cases: list[LiveMatrixCase] = []
    for provider in _PROVIDERS:
        for scenario in (SCENARIO_SINGLE_TURN, SCENARIO_MULTI_TURN, SCENARIO_REASONING):
            cases.append(
                LiveMatrixCase(
                    case_id=f"{provider.label}__{scenario}",
                    provider_label=provider.label,
                    api_type=provider.api_type,
                    model=provider.model,
                    base_url=provider.base_url,
                    scenario=scenario,
                    api_key_envs=provider.api_key_envs,
                )
            )
    return cases


def resolve_case_llm_config(case: LiveMatrixCase) -> LLMConfig:
    api_key = _resolve_api_key(case.api_key_envs)
    extra_params = _scenario_extra_params(case)
    max_tokens = 512 if case.scenario == SCENARIO_REASONING else 256
    return LLMConfig(
        model=case.model,
        base_url=case.base_url,
        api_key=api_key,
        api_type=case.api_type,
        temperature=0.0,
        max_tokens=max_tokens,
        max_retries=1,
        stream=False,
        **extra_params,
    )


def scenario_prompts(case: LiveMatrixCase) -> list[str]:
    if case.scenario == SCENARIO_SINGLE_TURN:
        return [f"Repeat this exact codeword once and nothing else: {SINGLE_CODEWORD}"]
    if case.scenario == SCENARIO_MULTI_TURN:
        return [
            f"Remember this codeword for later: {MULTI_CODEWORD}. Reply ONLY READY.",
            "What codeword did I ask you to remember? Reply with the exact codeword only.",
        ]
    if case.scenario == SCENARIO_REASONING:
        if case.api_type == "anthropic_chat_completion":
            return [
                "Solve this exactly and return ONLY the final integer:\n\n"
                "Compute ((37 * 43) + (91 * 17) - (12 * 12 * 5)) + the number of letters in the word "
                "ABRACADABRA. Think carefully before answering."
            ]
        return ["Think through 17*19 carefully, but return ONLY the final integer."]
    raise ValueError(f"Unsupported scenario: {case.scenario}")


def answer_matches(case: LiveMatrixCase, answer: str) -> tuple[bool, str]:
    normalized = answer.strip()
    if case.scenario == SCENARIO_SINGLE_TURN:
        ok = SINGLE_CODEWORD in normalized
        return ok, "contains codeword" if ok else f"missing codeword {SINGLE_CODEWORD}"
    if case.scenario == SCENARIO_MULTI_TURN:
        ok = MULTI_CODEWORD in normalized
        return ok, "contains recalled codeword" if ok else f"missing recalled codeword {MULTI_CODEWORD}"
    digits = "".join(re.findall(r"\d+", normalized))
    expected_answer = REASONING_EXPECTED_ANSWER
    if case.scenario == SCENARIO_REASONING and case.api_type == "anthropic_chat_completion":
        expected_answer = ANTHROPIC_COMPLEX_REASONING_EXPECTED_ANSWER
    ok = digits == expected_answer
    return ok, f"digits={digits or 'none'}"


def extract_raw_usage(api_type: str, raw_payload: dict[str, Any] | None) -> dict[str, Any]:
    if not raw_payload:
        return {}
    if api_type == "gemini_rest":
        raw = raw_payload.get("usageMetadata")
    else:
        raw = raw_payload.get("usage")
    return dict(raw) if isinstance(raw, dict) else {}


def raw_usage_excerpt(api_type: str, raw_usage: dict[str, Any]) -> dict[str, Any]:
    if not raw_usage:
        return {}
    if api_type == "openai_chat_completion":
        return {
            "prompt_tokens": _int(raw_usage.get("prompt_tokens")),
            "completion_tokens": _int(raw_usage.get("completion_tokens")),
            "total_tokens": _int(raw_usage.get("total_tokens")),
            "prompt_tokens_details": _mapping_copy(raw_usage.get("prompt_tokens_details")),
            "completion_tokens_details": _mapping_copy(raw_usage.get("completion_tokens_details")),
            "cache_write_fields": _cache_write_excerpt(raw_usage),
        }
    if api_type == "openai_responses":
        return {
            "input_tokens": _int(raw_usage.get("input_tokens")),
            "output_tokens": _int(raw_usage.get("output_tokens")),
            "total_tokens": _int(raw_usage.get("total_tokens")),
            "input_tokens_details": _mapping_copy(raw_usage.get("input_tokens_details")),
            "output_tokens_details": _mapping_copy(raw_usage.get("output_tokens_details")),
            "cache_write_fields": _cache_write_excerpt(raw_usage),
        }
    if api_type == "anthropic_chat_completion":
        return {
            "input_tokens": _int(raw_usage.get("input_tokens")),
            "output_tokens": _int(raw_usage.get("output_tokens")),
            "cache_read_input_tokens": _int(raw_usage.get("cache_read_input_tokens")),
            "cache_creation_input_tokens": _int(raw_usage.get("cache_creation_input_tokens")),
            "cache_write_input_tokens": _int(raw_usage.get("cache_write_input_tokens")),
            "cache_write_5m_input_tokens": _int(raw_usage.get("cache_write_5m_input_tokens")),
            "cache_write_1h_input_tokens": _int(raw_usage.get("cache_write_1h_input_tokens")),
            "total_tokens": _int(raw_usage.get("total_tokens")),
        }
    if api_type == "gemini_rest":
        return {
            "promptTokenCount": _int(raw_usage.get("promptTokenCount")),
            "candidatesTokenCount": _int(raw_usage.get("candidatesTokenCount")),
            "thoughtsTokenCount": _int(raw_usage.get("thoughtsTokenCount")),
            "cachedContentTokenCount": _int(raw_usage.get("cachedContentTokenCount")),
            "totalTokenCount": _int(raw_usage.get("totalTokenCount")),
        }
    return dict(raw_usage)


def expected_usage_from_raw(api_type: str, raw_usage: dict[str, Any]) -> dict[str, int]:
    if api_type == "openai_chat_completion":
        prompt_details = _mapping_copy(raw_usage.get("prompt_tokens_details"))
        completion_details = _mapping_copy(raw_usage.get("completion_tokens_details"))
        cache_read = _int(prompt_details.get("cached_tokens"))
        cache_write = _first_non_null_int(raw_usage, _cache_write_fields())
        input_tokens = max(_int(raw_usage.get("prompt_tokens")) - cache_read - cache_write, 0)
        completion_tokens = _int(raw_usage.get("completion_tokens"))
        reasoning_tokens = _int(completion_details.get("reasoning_tokens"))
        total_tokens = _int(raw_usage.get("total_tokens")) or (input_tokens + completion_tokens + cache_read + cache_write)
        return _expected_usage_dict(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

    if api_type == "openai_responses":
        input_details = _mapping_copy(raw_usage.get("input_tokens_details"))
        output_details = _mapping_copy(raw_usage.get("output_tokens_details"))
        cache_read = _int(input_details.get("cached_tokens"))
        cache_write = _first_non_null_int(raw_usage, _cache_write_fields())
        input_tokens = max(_int(raw_usage.get("input_tokens")) - cache_read - cache_write, 0)
        completion_tokens = _int(raw_usage.get("output_tokens"))
        reasoning_tokens = _int(output_details.get("reasoning_tokens"))
        total_tokens = _int(raw_usage.get("total_tokens")) or (input_tokens + completion_tokens + cache_read + cache_write)
        return _expected_usage_dict(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

    if api_type == "anthropic_chat_completion":
        input_tokens = _int(raw_usage.get("input_tokens"))
        completion_tokens = _int(raw_usage.get("output_tokens"))
        cache_read = _int(raw_usage.get("cache_read_input_tokens"))
        cache_write = _first_non_null_int(raw_usage, _cache_write_fields())
        total_tokens = input_tokens + completion_tokens + cache_read + cache_write
        return _expected_usage_dict(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=0,
            total_tokens=total_tokens,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        )

    if api_type == "gemini_rest":
        input_tokens = _int(raw_usage.get("promptTokenCount"))
        completion_tokens = _int(raw_usage.get("candidatesTokenCount"))
        reasoning_tokens = _int(raw_usage.get("thoughtsTokenCount"))
        cache_read_tokens = _int(raw_usage.get("cachedContentTokenCount"))
        total_tokens = _int(raw_usage.get("totalTokenCount")) or (input_tokens + completion_tokens)
        return _expected_usage_dict(
            input_tokens=input_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cache_creation_tokens=0,
            cache_read_tokens=cache_read_tokens,
        )

    raise ValueError(f"Unsupported api_type: {api_type}")


def usage_validation_notes(actual_usage: TokenUsage, expected_usage: dict[str, int]) -> list[str]:
    notes: list[str] = []
    actual_dict = actual_usage.to_dict()

    for key, expected_value in expected_usage.items():
        actual_value = actual_dict.get(key)
        if actual_value != expected_value:
            notes.append(f"{key}: expected {expected_value}, got {actual_value}")

    if any(value < 0 for value in actual_dict.values()):
        notes.append("normalized usage contains negative values")

    if actual_usage.input_tokens_uncached != actual_usage.input_tokens:
        notes.append("input_tokens_uncached does not match input_tokens")

    expected_session_total = (
        expected_usage["input_tokens"]
        + expected_usage["completion_tokens"]
        + expected_usage["reasoning_tokens"]
        + expected_usage["cache_read_tokens"]
        + expected_usage["cache_creation_tokens"]
    )
    if actual_usage.session_total_tokens() != expected_session_total:
        notes.append(f"session_total_tokens mismatch: expected {expected_session_total}, got {actual_usage.session_total_tokens()}")

    return notes


def persist_live_result(result: LiveCaseResult, output_path: Path = DEFAULT_RESULTS_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_live_results(output_path)
    kept = [item for item in existing if item.case_id != result.case_id]
    kept.append(result)
    kept.sort(key=lambda item: item.case_id)
    output_path.write_text(json.dumps([item.to_dict() for item in kept], ensure_ascii=False, indent=2), encoding="utf-8")


def load_live_results(output_path: Path = DEFAULT_RESULTS_PATH) -> list[LiveCaseResult]:
    if not output_path.exists():
        return []
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        return []
    results: list[LiveCaseResult] = []
    for item in loaded:
        if isinstance(item, dict):
            results.append(LiveCaseResult(**item))
    return results


def reset_live_results(output_path: Path = DEFAULT_RESULTS_PATH) -> None:
    if output_path.exists():
        output_path.unlink()


def build_report_markdown(results: list[LiveCaseResult]) -> str:
    ordered = sorted(results, key=lambda item: item.case_id)
    pass_count = sum(1 for item in ordered if item.status == "PASS")
    upstream_count = sum(1 for item in ordered if item.status == "UPSTREAM_LIMITATION")
    fail_count = sum(1 for item in ordered if item.status == "FAIL")

    lines = [
        REPORT_MARKER,
        "## Live Token Usage Matrix Report",
        "",
        "### Summary",
        f"- total cases: {len(ordered)}",
        f"- PASS: {pass_count}",
        f"- UPSTREAM_LIMITATION: {upstream_count}",
        f"- FAIL: {fail_count}",
        "",
        "### Matrix Table",
        "",
        (
            "| case_id | api_type | model | scenario | status | answer_check | input | output | reasoning | "
            "cache_read | cache_write | total | notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for item in ordered:
        usage = item.actual_usage
        notes = "<br>".join(_escape_table_cell(note) for note in item.notes) if item.notes else ""
        lines.append(
            "| {case_id} | {api_type} | {model} | {scenario} | {status} | {answer_check} | {input_tokens} | {completion_tokens} | "
            "{reasoning_tokens} | {cache_read_tokens} | {cache_creation_tokens} | {total_tokens} | {notes} |".format(
                case_id=_escape_table_cell(item.case_id),
                api_type=_escape_table_cell(item.api_type),
                model=_escape_table_cell(item.model),
                scenario=_escape_table_cell(item.scenario),
                status=item.status,
                answer_check=_escape_table_cell(item.answer_check),
                input_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                reasoning_tokens=usage.get("reasoning_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                notes=notes,
            )
        )

    findings = [item for item in ordered if item.status != "PASS"]
    lines.extend(["", "### Detailed Findings", ""])
    if not findings:
        lines.append("- none")
    else:
        for item in findings:
            lines.extend(
                [
                    f"#### {item.case_id}",
                    f"- status: `{item.status}`",
                    f"- answer_check: `{item.answer_check}`",
                    f"- notes: {'; '.join(item.notes) if item.notes else 'none'}",
                    "- raw_usage_excerpt:",
                    "```json",
                    json.dumps(item.raw_usage_excerpt, ensure_ascii=False, indent=2),
                    "```",
                    "- expected_usage:",
                    "```json",
                    json.dumps(item.expected_usage, ensure_ascii=False, indent=2),
                    "```",
                    "- actual_usage:",
                    "```json",
                    json.dumps(item.actual_usage, ensure_ascii=False, indent=2),
                    "```",
                    "",
                ]
            )

    lines.extend(["### Conclusion", ""])
    by_api_type: dict[str, list[LiveCaseResult]] = {}
    for item in ordered:
        by_api_type.setdefault(item.api_type, []).append(item)
    for api_type, items in sorted(by_api_type.items()):
        statuses = {entry.status for entry in items}
        if statuses == {"PASS"}:
            conclusion = "fully aligned in this run"
        elif "FAIL" in statuses:
            conclusion = "has mismatches or execution failures"
        else:
            conclusion = "aligned except upstream reasoning-token exposure gaps"
        lines.append(f"- `{api_type}`: {conclusion}")

    return "\n".join(lines).strip() + "\n"


def serialize_payload(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return dict(payload)
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        try:
            try:
                dumped = model_dump(mode="json", warnings=False)
            except TypeError:
                try:
                    dumped = model_dump(warnings=False)
                except TypeError:
                    dumped = model_dump()
            if isinstance(dumped, dict):
                return dict(dumped)
        except Exception:
            pass
    try:
        dumped = json.loads(json.dumps(payload, ensure_ascii=False))
        if isinstance(dumped, dict):
            return dumped
    except Exception:
        pass
    return None


def _scenario_extra_params(case: LiveMatrixCase) -> dict[str, Any]:
    if case.scenario != SCENARIO_REASONING:
        return {}
    if case.api_type == "openai_chat_completion":
        return {"reasoning": {"effort": "medium"}}
    if case.api_type == "openai_responses":
        return {"reasoning": {"effort": "medium", "summary": "detailed"}}
    if case.api_type == "anthropic_chat_completion":
        return {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    if case.api_type == "gemini_rest":
        return {"thinkingConfig": {"includeThoughts": True, "thinkingBudget": 1024}}
    return {}


def _resolve_api_key(env_names: tuple[str, ...]) -> str:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            return value
    raise RuntimeError(f"Missing API key. Expected one of: {', '.join(env_names)}")


def _expected_usage_dict(
    *,
    input_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    total_tokens: int,
    cache_creation_tokens: int,
    cache_read_tokens: int,
) -> dict[str, int]:
    return {
        "input_tokens": input_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "input_tokens_uncached": input_tokens,
    }


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _cache_write_fields() -> tuple[str, ...]:
    return (
        "cache_creation_input_tokens",
        "cache_write_input_tokens",
        "cache_write_5m_input_tokens",
        "cache_write_1h_input_tokens",
    )


def _cache_write_excerpt(raw_usage: dict[str, Any]) -> dict[str, int]:
    return {field: _int(raw_usage.get(field)) for field in _cache_write_fields() if raw_usage.get(field) is not None}


def _first_non_null_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int:
    for key in keys:
        if mapping.get(key) is not None:
            return _int(mapping.get(key))
    return 0


def _int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")
