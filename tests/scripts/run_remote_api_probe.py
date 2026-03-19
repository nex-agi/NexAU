#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

PROMPT = (
    "Solve this exactly and return ONLY the final integer:\n\n"
    "Compute ((37 * 43) + (91 * 17) - (12 * 12 * 5)) + the number of letters in the word "
    "ABRACADABRA. Think carefully before answering."
)


@dataclass(frozen=True, slots=True)
class ProbeCase:
    case_id: str
    method: str
    url: str
    headers: dict[str, str]
    body: dict[str, Any]


def _redact_url(url: str) -> str:
    parts = urlsplit(url)
    query = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in {"key", "api_key"}:
            query.append((key, "***"))
        else:
            query.append((key, value))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "x-api-key"}:
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def _extract_answer(case_id: str, payload: dict[str, Any]) -> str:
    if "responses" in case_id:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text
        for item in payload.get("output", []) or []:
            if item.get("type") != "message":
                continue
            text_parts = []
            for block in item.get("content", []) or []:
                if block.get("type") == "output_text":
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts)
        return ""
    if "chat" in case_id:
        choices = payload.get("choices", []) or []
        if choices:
            message = choices[0].get("message", {}) or {}
            content = message.get("content")
            if isinstance(content, str):
                return content
        return ""
    if "anthropic" in case_id:
        text_parts = []
        for block in payload.get("content", []) or []:
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)
    if "gemini" in case_id:
        text_parts = []
        for candidate in payload.get("candidates", []) or []:
            content = candidate.get("content", {}) or {}
            for part in content.get("parts", []) or []:
                if part.get("thought") is True:
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)
    return ""


def _extract_usage(case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    if "gemini" in case_id:
        usage = payload.get("usageMetadata")
    else:
        usage = payload.get("usage")
    return usage if isinstance(usage, dict) else {}


def _build_cases() -> list[ProbeCase]:
    return [
        ProbeCase(
            case_id="responses_rayinai_gpt54",
            method="POST",
            url="https://code.rayinai.com/v1/responses",
            headers={
                "Authorization": "Bearer sk-2ab502329ce03184303e5d6ed2e84496e3fb37ab2f6a4602524727385dec02e3",
                "Content-Type": "application/json",
            },
            body={
                "model": "gpt-5.4",
                "input": PROMPT,
                "max_output_tokens": 512,
                "reasoning": {"effort": "medium", "summary": "detailed"},
                "include": ["reasoning.encrypted_content"],
            },
        ),
        ProbeCase(
            case_id="chat_14_gpt5",
            method="POST",
            url="http://14.103.60.158:3001/v1/chat/completions",
            headers={
                "Authorization": "Bearer sk-8JoEeRVplrADEMiwc7XbEdtrEBxdCbCZkWhboMHEPNWlzgQk",
                "Content-Type": "application/json",
            },
            body={
                "model": "gpt-5",
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 768,
                "temperature": 0,
            },
        ),
        ProbeCase(
            case_id="responses_14_gpt5",
            method="POST",
            url="http://14.103.60.158:3001/v1/responses",
            headers={
                "Authorization": "Bearer sk-8JoEeRVplrADEMiwc7XbEdtrEBxdCbCZkWhboMHEPNWlzgQk",
                "Content-Type": "application/json",
            },
            body={
                "model": "gpt-5",
                "input": PROMPT,
                "max_output_tokens": 512,
                "reasoning": {"effort": "medium", "summary": "detailed"},
                "include": ["reasoning.encrypted_content"],
            },
        ),
        ProbeCase(
            case_id="anthropic_14_claude46",
            method="POST",
            url="http://14.103.60.158:3001/v1/messages",
            headers={
                "x-api-key": "sk-tSWa15zD6TKFjnNDIbKE0h0CvtHJBdrh67OaVt88HJZcYe5z",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            body={
                "model": "claude-sonnet-4-6",
                "max_tokens": 512,
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "messages": [{"role": "user", "content": PROMPT}],
            },
        ),
        ProbeCase(
            case_id="gemini_14_gemini31",
            method="POST",
            url="http://14.103.60.158:3001/v1beta/models/gemini-3.1-pro-preview:generateContent?key=sk-8JoEeRVplrADEMiwc7XbEdtrEBxdCbCZkWhboMHEPNWlzgQk",
            headers={"Content-Type": "application/json"},
            body={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": PROMPT}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 512,
                    "thinkingConfig": {"includeThoughts": True, "thinkingBudget": 1024},
                },
            },
        ),
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Call remote model APIs and persist full JSON responses plus a local report.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".pytest_cache/remote_api_probe"),
        help="Directory where raw responses and the report will be written.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, Any]] = []
    report_lines = [
        "# Remote API Probe Report",
        "",
        f"- output_dir: `{output_dir}`",
        f"- prompt: `{PROMPT}`",
        "",
        "| case_id | status | answer | usage_keys | response_file |",
        "| --- | --- | --- | --- | --- |",
    ]

    for case in _build_cases():
        request_meta = {
            "method": case.method,
            "url": _redact_url(case.url),
            "headers": _redact_headers(case.headers),
            "body": case.body,
        }

        start = time.time()
        status = "SUCCESS"
        response_payload: dict[str, Any] | None = None
        error_payload: dict[str, Any] | None = None
        http_status: int | None = None
        response_text: str | None = None

        try:
            response = requests.request(case.method, case.url, headers=case.headers, json=case.body, timeout=180)
            http_status = response.status_code
            response_text = response.text
            response.raise_for_status()
            response_payload = response.json()
        except Exception as exc:
            if case.case_id == "gemini_14_gemini31":
                fallback_body = json.loads(json.dumps(case.body))
                generation_config = fallback_body.get("generationConfig", {})
                thinking_config = generation_config.get("thinkingConfig", {})
                if thinking_config.get("thinkingBudget") is not None:
                    budget = thinking_config.pop("thinkingBudget")
                    thinking_config["thoughtBudgetTokens"] = budget
                    try:
                        response = requests.request(
                            case.method,
                            case.url,
                            headers=case.headers,
                            json=fallback_body,
                            timeout=180,
                        )
                        http_status = response.status_code
                        response_text = response.text
                        response.raise_for_status()
                        response_payload = response.json()
                        case_dir = output_dir / case.case_id
                        case_dir.mkdir(parents=True, exist_ok=True)
                        _write_json(case_dir / "fallback_request.json", fallback_body)
                    except Exception as fallback_exc:
                        error_payload = {
                            "error_type": type(fallback_exc).__name__,
                            "error": str(fallback_exc),
                            "http_status": http_status,
                            "response_text": response_text,
                        }
            status = "ERROR"
            if response_payload is None:
                error_payload = error_payload or {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "http_status": http_status,
                    "response_text": response_text,
                }
            else:
                status = "SUCCESS"

        duration_ms = round((time.time() - start) * 1000, 2)
        record = {
            "case_id": case.case_id,
            "status": status,
            "duration_ms": duration_ms,
            "request": request_meta,
            "response": response_payload,
            "error": error_payload,
            "answer": _extract_answer(case.case_id, response_payload or {}),
            "usage": _extract_usage(case.case_id, response_payload or {}),
        }
        combined.append(record)

        case_dir = output_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _write_json(case_dir / "request.json", request_meta)
        if response_payload is not None:
            _write_json(case_dir / "response.json", response_payload)
        if error_payload is not None:
            _write_json(case_dir / "error.json", error_payload)

        usage_keys = ", ".join(sorted(record["usage"].keys())) if isinstance(record["usage"], dict) and record["usage"] else "-"
        answer = str(record["answer"]).replace("\n", "\\n")
        response_file = case_dir / ("response.json" if response_payload is not None else "error.json")
        report_lines.append(f"| {case.case_id} | {status} | {answer or '-'} | {usage_keys} | `{response_file}` |")

    _write_json(output_dir / "combined_results.json", combined)
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
