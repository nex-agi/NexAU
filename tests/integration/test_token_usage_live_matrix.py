from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from nexau.archs.main_sub.agent import Agent
from nexau.archs.main_sub.config import AgentConfig
from nexau.archs.main_sub.execution.hooks import AfterModelHookInput, HookResult
from nexau.archs.main_sub.execution.model_response import ModelResponse
from nexau.archs.session import InMemoryDatabaseEngine, SessionManager
from nexau.core.usage import TokenUsage
from tests.utils.live_token_usage_matrix import (
    DEFAULT_RESULTS_PATH,
    RESULT_STATUS,
    SCENARIO_MULTI_TURN,
    SCENARIO_REASONING,
    LiveCaseResult,
    LiveMatrixCase,
    answer_matches,
    expected_usage_from_raw,
    extract_raw_usage,
    iter_live_matrix_cases,
    persist_live_result,
    raw_usage_excerpt,
    reset_live_results,
    resolve_case_llm_config,
    scenario_prompts,
    serialize_payload,
    usage_validation_notes,
)

pytestmark = [pytest.mark.integration, pytest.mark.llm]


@dataclass(slots=True)
class _TurnCapture:
    answer: str
    model_response: ModelResponse
    raw_payload: dict[str, Any] | None


class _RecordedPayloads:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any] | None] = []

    def append(self, payload: Any) -> None:
        self.payloads.append(serialize_payload(payload))


class _OpenAIChatCompletionsRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._inner.create(*args, **kwargs)
        self._payloads.append(response)
        return response


class _OpenAIChatRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self.completions = _OpenAIChatCompletionsRecorder(inner.completions, payloads)


class _OpenAIResponsesRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._inner.create(*args, **kwargs)
        self._payloads.append(response)
        return response


class _OpenAIClientRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self.chat = _OpenAIChatRecorder(inner.chat, payloads)
        self.responses = _OpenAIResponsesRecorder(inner.responses, payloads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AnthropicMessagesRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self._payloads = payloads

    def create(self, *args: Any, **kwargs: Any) -> Any:
        response = self._inner.create(*args, **kwargs)
        self._payloads.append(response)
        return response


class _AnthropicClientRecorder:
    def __init__(self, inner: Any, payloads: _RecordedPayloads) -> None:
        self._inner = inner
        self.messages = _AnthropicMessagesRecorder(inner.messages, payloads)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@pytest.fixture(scope="module", autouse=True)
def _reset_live_matrix_results_file():
    reset_live_results(DEFAULT_RESULTS_PATH)


def _capture_after_model(records: list[ModelResponse]):
    def _hook(hook_input: AfterModelHookInput) -> HookResult:
        if hook_input.model_response is not None:
            records.append(hook_input.model_response)
        return HookResult.no_changes()

    return _hook


def _wrap_runtime_client(agent: Agent, case: LiveMatrixCase, payloads: _RecordedPayloads) -> None:
    if agent.openai_client is None:
        return
    if case.api_type in {"openai_chat_completion", "openai_responses"}:
        agent.openai_client = _OpenAIClientRecorder(agent.openai_client, payloads)
        return
    if case.api_type == "anthropic_chat_completion":
        agent.openai_client = _AnthropicClientRecorder(agent.openai_client, payloads)


def _build_agent(
    *,
    case: LiveMatrixCase,
    session_manager: SessionManager,
    user_id: str,
    session_id: str,
    records: list[ModelResponse],
    payloads: _RecordedPayloads,
    llm_config_override: dict[str, Any] | None = None,
) -> Agent:
    llm_config = resolve_case_llm_config(case)
    if llm_config_override:
        for key, value in llm_config_override.items():
            if value is None:
                if hasattr(llm_config, key):
                    setattr(llm_config, key, None)
                else:
                    llm_config.extra_params.pop(key, None)
                continue
            llm_config.update(**{key: value})
    config = AgentConfig(
        name=f"token_usage_{case.case_id}",
        system_prompt="You are a precise assistant. Follow the user's formatting instructions exactly.",
        llm_config=llm_config,
        after_model_hooks=[_capture_after_model(records)],
    )
    agent = Agent(
        config=config,
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
    )
    _wrap_runtime_client(agent, case, payloads)
    return agent


def _normalize_run_result(result: str | tuple[Any, ...]) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, tuple) and result:
        return str(result[0])
    return str(result)


def _turn_payload(index: int, response: ModelResponse, payloads: _RecordedPayloads) -> dict[str, Any] | None:
    if index < len(payloads.payloads) and payloads.payloads[index] is not None:
        return payloads.payloads[index]
    return serialize_payload(response.raw_message)


async def _run_multi_turn_case(case: LiveMatrixCase) -> tuple[list[_TurnCapture], list[str]]:
    notes: list[str] = []
    prompts = scenario_prompts(case)

    for attempt in range(2):
        records: list[ModelResponse] = []
        payloads = _RecordedPayloads()
        session_manager = SessionManager(engine=InMemoryDatabaseEngine())
        user_id = f"user_{case.case_id}"
        session_id = f"session_{case.case_id}"

        try:
            agent1 = _build_agent(
                case=case,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                records=records,
                payloads=payloads,
            )
            turn1_answer = _normalize_run_result(await agent1.run_async(message=prompts[0]))
            await asyncio.sleep(0.1)

            agent2 = _build_agent(
                case=case,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                records=records,
                payloads=payloads,
            )
            turn2_answer = _normalize_run_result(await agent2.run_async(message=prompts[1]))
            break
        except Exception as exc:
            if attempt == 0 and _looks_like_transient_upstream_error(exc):
                notes.append(f"retried after transient upstream error: {exc}")
                continue
            raise

    if len(records) < 2:
        raise AssertionError(f"Expected 2 model responses, got {len(records)}")

    captures = [
        _TurnCapture(answer=turn1_answer, model_response=records[0], raw_payload=_turn_payload(0, records[0], payloads)),
        _TurnCapture(answer=turn2_answer, model_response=records[1], raw_payload=_turn_payload(1, records[1], payloads)),
    ]
    notes.append(f"turn1_total_tokens={captures[0].model_response.usage.total_tokens}")
    return captures, notes


def _run_single_turn_case(case: LiveMatrixCase) -> tuple[list[_TurnCapture], list[str]]:
    notes: list[str] = []
    prompts = scenario_prompts(case)

    llm_config_override: dict[str, Any] | None = None
    retried_with_alt_gemini_thinking = False
    retried_without_chat_reasoning = False
    transient_retry_count = 0

    while True:
        records: list[ModelResponse] = []
        payloads = _RecordedPayloads()
        session_manager = SessionManager(engine=InMemoryDatabaseEngine())
        user_id = f"user_{case.case_id}"
        session_id = f"session_{case.case_id}"
        try:
            agent = _build_agent(
                case=case,
                session_manager=session_manager,
                user_id=user_id,
                session_id=session_id,
                records=records,
                payloads=payloads,
                llm_config_override=llm_config_override,
            )
            answer = _normalize_run_result(agent.run(message=prompts[0]))
            break
        except Exception as exc:
            if (
                case.scenario == SCENARIO_REASONING
                and case.api_type == "openai_chat_completion"
                and not retried_without_chat_reasoning
                and _looks_like_unsupported_openai_chat_reasoning(exc)
            ):
                llm_config_override = {"reasoning": None}
                retried_without_chat_reasoning = True
                notes.append(
                    "upstream limitation: chat completions endpoint rejected reasoning param; retried without explicit reasoning options"
                )
                continue
            if (
                case.scenario == SCENARIO_REASONING
                and case.api_type == "gemini_rest"
                and not retried_with_alt_gemini_thinking
                and _looks_like_unsupported_gemini_thinking_config(exc)
            ):
                llm_config_override = {"thinkingConfig": {"includeThoughts": True, "thoughtBudgetTokens": 1024}}
                retried_with_alt_gemini_thinking = True
                records.clear()
                payloads.payloads.clear()
                notes.append("retried gemini reasoning with thoughtBudgetTokens")
                continue
            if transient_retry_count < 1 and _looks_like_transient_upstream_error(exc):
                transient_retry_count += 1
                notes.append(f"retried after transient upstream error: {exc}")
                continue
            raise

    if not records:
        raise AssertionError("Expected at least one model response")

    capture = _TurnCapture(answer=answer, model_response=records[-1], raw_payload=_turn_payload(len(records) - 1, records[-1], payloads))
    return [capture], notes


def _looks_like_unsupported_gemini_thinking_config(exc: Exception) -> bool:
    message = str(exc).lower()
    return "thinkingbudget" in message or "thoughtbudgettokens" in message or "unknown name" in message or "invalid argument" in message


def _looks_like_unsupported_openai_chat_reasoning(exc: Exception) -> bool:
    message = str(exc).lower()
    return "unexpected keyword argument 'reasoning'" in message or 'unexpected keyword argument "reasoning"' in message


def _looks_like_transient_upstream_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        needle in message
        for needle in (
            "502",
            "503",
            "504",
            "bad gateway",
            "gateway timeout",
            "temporarily unavailable",
            "connection reset",
            "read timed out",
            "timeout error",
        )
    )


def _result_for_failure(case: LiveMatrixCase, exc: Exception) -> LiveCaseResult:
    return LiveCaseResult(
        case_id=case.case_id,
        provider_label=case.provider_label,
        api_type=case.api_type,
        model=case.model,
        scenario=case.scenario,
        status="FAIL",
        answer_check=f"execution error: {type(exc).__name__}",
        final_answer="",
        raw_usage_excerpt={},
        expected_usage=TokenUsage().to_dict(),
        actual_usage=TokenUsage().to_dict(),
        notes=[str(exc)],
    )


def _evaluate_case(case: LiveMatrixCase, captures: list[_TurnCapture], notes: list[str]) -> LiveCaseResult:
    final_capture = captures[-1]
    answer_ok, answer_check = answer_matches(case, final_capture.answer)
    raw_usage = extract_raw_usage(case.api_type, final_capture.raw_payload)
    raw_excerpt = raw_usage_excerpt(case.api_type, raw_usage)
    actual_usage = final_capture.model_response.usage
    expected_usage = expected_usage_from_raw(case.api_type, raw_usage) if raw_usage else TokenUsage().to_dict()

    if not raw_usage:
        notes.append("missing raw usage payload")

    if not answer_ok:
        notes.append(f"answer validation failed: {answer_check}")

    if case.scenario == SCENARIO_MULTI_TURN and actual_usage.context_used_tokens() <= 0:
        notes.append("context_used_tokens must be > 0 for multi-turn second turn")

    if raw_usage:
        notes.extend(usage_validation_notes(actual_usage, expected_usage))

    status: RESULT_STATUS = "PASS"
    if notes and any(
        note.startswith("missing raw usage")
        or note.startswith("answer validation failed")
        or note.startswith("context_used_tokens")
        or ": expected " in note
        or note.startswith("session_total_tokens mismatch")
        or note.startswith("input_tokens_uncached")
        or note.startswith("normalized usage contains negative")
        for note in notes
    ):
        status = "FAIL"
    elif any(note.startswith("upstream limitation:") for note in notes):
        status = "UPSTREAM_LIMITATION"
    elif case.scenario == SCENARIO_REASONING and expected_usage.get("reasoning_tokens", 0) == 0:
        status = "UPSTREAM_LIMITATION"
        notes.append("upstream usage did not expose reasoning_tokens")

    return LiveCaseResult(
        case_id=case.case_id,
        provider_label=case.provider_label,
        api_type=case.api_type,
        model=case.model,
        scenario=case.scenario,
        status=status,
        answer_check=answer_check,
        final_answer=final_capture.answer,
        raw_usage_excerpt=raw_excerpt,
        expected_usage=expected_usage,
        actual_usage=actual_usage.to_dict(),
        notes=notes,
    )


@pytest.mark.parametrize("case", iter_live_matrix_cases(), ids=lambda item: item.case_id)
def test_token_usage_live_matrix(case: LiveMatrixCase):
    try:
        if case.scenario == SCENARIO_MULTI_TURN:
            captures, notes = asyncio.run(_run_multi_turn_case(case))
        else:
            captures, notes = _run_single_turn_case(case)
        result = _evaluate_case(case, captures, notes)
    except RuntimeError as exc:
        if "Missing API key" in str(exc):
            pytest.skip(str(exc))
        result = _result_for_failure(case, exc)
    except Exception as exc:
        result = _result_for_failure(case, exc)

    persist_live_result(result)

    if result.status == "FAIL":
        pytest.fail("; ".join(result.notes) or "live token usage case failed")
