"""Import-time regression tests for lightweight Python bridge bindings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEAVY_MODULES = (
    "anthropic",
    "e2b",
    "fastapi",
    "langfuse",
    "mcp",
    "openai",
    "pandas",
    "requests",
    "sqlalchemy",
    "sqlmodel",
    "tiktoken",
    "uvicorn",
)


def _run_child(script: str) -> dict[str, object]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT_ROOT) if not existing_pythonpath else f"{PROJECT_ROOT}{os.pathsep}{existing_pythonpath}"

    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return cast(dict[str, object], payload)


def _assert_no_heavy_modules(payload: dict[str, object]) -> None:
    loaded = payload["loaded"]
    assert isinstance(loaded, dict)
    assert {name for name, is_loaded in loaded.items() if is_loaded} == set()


def test_import_nexau_is_lightweight() -> None:
    payload = _run_child(
        f"""
        import json
        import sys

        import nexau

        heavy_modules = {HEAVY_MODULES!r}
        implementation_modules = (
            "nexau.archs.main_sub.agent",
            "nexau.archs.main_sub.execution.executor",
            "nexau.archs.main_sub.execution.llm_caller",
            "nexau.archs.tool.tool",
            "nexau.archs.tool.builtin.mcp_client",
            "nexau.archs.session.session_manager",
        )
        print(json.dumps({{
            "all": list(nexau.__all__),
            "loaded": {{name: name in sys.modules for name in heavy_modules}},
            "implementation_loaded": {{name: name in sys.modules for name in implementation_modules}},
        }}))
        """
    )

    assert payload["all"] == [
        "Agent",
        "Tool",
        "LLMConfig",
        "AgentConfig",
        "Skill",
        "BaseTracer",
        "CompositeTracer",
        "Span",
        "SpanType",
        "TraceContext",
    ]
    _assert_no_heavy_modules(payload)
    implementation_loaded = payload["implementation_loaded"]
    assert isinstance(implementation_loaded, dict)
    assert {name for name, is_loaded in implementation_loaded.items() if is_loaded} == set()


def test_top_level_public_exports_still_resolve() -> None:
    payload = _run_child(
        """
        import json

        from nexau import Agent, AgentConfig, LLMConfig, Skill, Tool

        print(json.dumps({
            "names": [Agent.__name__, AgentConfig.__name__, LLMConfig.__name__, Skill.__name__, Tool.__name__]
        }))
        """
    )

    assert payload["names"] == ["Agent", "AgentConfig", "LLMConfig", "Skill", "Tool"]


def test_builtin_file_tools_binding_import_is_lightweight() -> None:
    payload = _run_child(
        f"""
        import importlib
        import json
        import sys

        module = importlib.import_module("nexau.archs.tool.builtin.file_tools")
        read_file = module.read_file
        list_directory = module.list_directory

        heavy_modules = {HEAVY_MODULES!r}
        implementation_modules = (
            "nexau.archs.tool.builtin.mcp_client",
            "nexau.archs.tool.builtin.web_tools.google_web_search",
            "nexau.archs.tool.builtin.web_tools.web_fetch",
            "nexau.archs.main_sub.agent",
            "nexau.archs.session.session_manager",
        )
        print(json.dumps({{
            "names": [read_file.__name__, list_directory.__name__],
            "loaded": {{name: name in sys.modules for name in heavy_modules}},
            "implementation_loaded": {{name: name in sys.modules for name in implementation_modules}},
        }}))
        """
    )

    assert payload["names"] == ["read_file", "list_directory"]
    _assert_no_heavy_modules(payload)
    implementation_loaded = payload["implementation_loaded"]
    assert isinstance(implementation_loaded, dict)
    assert {name for name, is_loaded in implementation_loaded.items() if is_loaded} == set()


def test_long_tool_output_middleware_binding_import_is_lightweight() -> None:
    payload = _run_child(
        f"""
        import importlib
        import json
        import sys

        module = importlib.import_module("nexau.archs.main_sub.execution.middleware.long_tool_output")
        middleware_cls = module.LongToolOutputMiddleware

        heavy_modules = {HEAVY_MODULES!r}
        implementation_modules = (
            "nexau.archs.sandbox.local_sandbox",
            "nexau.archs.main_sub.agent",
            "nexau.archs.main_sub.execution.executor",
            "nexau.archs.main_sub.execution.llm_caller",
            "nexau.archs.tool.tool",
            "nexau.archs.session.session_manager",
        )
        print(json.dumps({{
            "name": middleware_cls.__name__,
            "loaded": {{name: name in sys.modules for name in heavy_modules}},
            "implementation_loaded": {{name: name in sys.modules for name in implementation_modules}},
        }}))
        """
    )

    assert payload["name"] == "LongToolOutputMiddleware"
    _assert_no_heavy_modules(payload)
    implementation_loaded = payload["implementation_loaded"]
    assert isinstance(implementation_loaded, dict)
    assert {name for name, is_loaded in implementation_loaded.items() if is_loaded} == set()
