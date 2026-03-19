#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _load_live_report_helpers() -> tuple[
    Path,
    str,
    Callable[[list[Any]], str],
    Callable[[], list[Any]],
    Callable[[Path], list[Any]],
]:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.utils.live_token_usage_matrix import (
        DEFAULT_RESULTS_PATH,
        REPORT_MARKER,
        build_report_markdown,
        iter_live_matrix_cases,
        load_live_results,
    )

    return DEFAULT_RESULTS_PATH, REPORT_MARKER, build_report_markdown, iter_live_matrix_cases, load_live_results


def _run_gh(args: list[str]) -> str:
    completed = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return completed.stdout


def _repo_name_with_owner() -> str:
    payload = json.loads(_run_gh(["repo", "view", "--json", "nameWithOwner"]))
    value = payload.get("nameWithOwner")
    if not isinstance(value, str) or not value:
        raise RuntimeError("Unable to resolve repository nameWithOwner from gh")
    return value


def _find_existing_comment_id(pr_number: int) -> str | None:
    _, report_marker, _, _, _ = _load_live_report_helpers()
    payload = json.loads(_run_gh(["pr", "view", str(pr_number), "--json", "comments"]))
    comments = payload.get("comments", [])
    if not isinstance(comments, list):
        return None
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        body = comment.get("body")
        comment_id = comment.get("id")
        if isinstance(body, str) and report_marker in body and comment_id is not None:
            return str(comment_id)
    return None


def _patch_comment(repo: str, comment_id: str, markdown: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump({"body": markdown}, handle, ensure_ascii=False)
        temp_path = Path(handle.name)
    try:
        subprocess.run(
            ["gh", "api", f"repos/{repo}/issues/comments/{comment_id}", "--method", "PATCH", "--input", str(temp_path)],
            check=True,
        )
    finally:
        temp_path.unlink(missing_ok=True)


def _create_comment(pr_number: int, markdown: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as handle:
        handle.write(markdown)
        temp_path = Path(handle.name)
    try:
        subprocess.run(["gh", "pr", "comment", str(pr_number), "--body-file", str(temp_path)], check=True)
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> int:
    default_results_path, _, build_report_markdown, iter_live_matrix_cases, load_live_results = _load_live_report_helpers()

    parser = argparse.ArgumentParser(description="Post or update the live token-usage matrix report on a PR.")
    parser.add_argument("--pr", type=int, required=True, help="Pull request number to comment on")
    parser.add_argument("--results", type=Path, default=default_results_path, help="Path to the live matrix results JSON file")
    args = parser.parse_args()

    results = load_live_results(args.results)
    expected_count = len(iter_live_matrix_cases())
    if len(results) != expected_count:
        raise RuntimeError(f"Expected {expected_count} results, found {len(results)} in {args.results}")

    markdown = build_report_markdown(results)
    repo = _repo_name_with_owner()
    existing_comment_id = _find_existing_comment_id(args.pr)
    if existing_comment_id:
        _patch_comment(repo, existing_comment_id, markdown)
        print(f"Updated PR comment {existing_comment_id} on #{args.pr}")
    else:
        _create_comment(args.pr, markdown)
        print(f"Created PR comment on #{args.pr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
