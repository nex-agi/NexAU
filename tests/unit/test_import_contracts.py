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
        "Plugin",
        "Skill",
        "MCPAuthContext",
        "MCPAuthorizationCodeSession",
        "MCPAuthHost",
        "MCPRuntimeFactory",
        "MCPRunScope",
        "MCPServerConfig",
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

        from nexau import Agent, AgentConfig, LLMConfig, Plugin, Skill, Tool

        print(json.dumps({
            "names": [Agent.__name__, AgentConfig.__name__, LLMConfig.__name__, Plugin.__name__, Skill.__name__, Tool.__name__]
        }))
        """
    )

    assert payload["names"] == ["Agent", "AgentConfig", "LLMConfig", "Plugin", "Skill", "Tool"]


def test_top_level_mcp_exports_resolve_from_lazy_public_api() -> None:
    payload = _run_child(
        """
        import json

        from nexau import (
            MCPAuthContext,
            MCPAuthorizationCodeSession,
            MCPAuthHost,
            MCPRuntimeFactory,
            MCPRunScope,
            MCPServerConfig,
        )

        print(json.dumps({
            "names": [
                MCPAuthContext.__name__,
                MCPAuthorizationCodeSession.__name__,
                MCPAuthHost.__name__,
                MCPRuntimeFactory.__name__,
                MCPRunScope.__name__,
                MCPServerConfig.__name__,
            ]
        }))
        """
    )

    assert payload["names"] == [
        "MCPAuthContext",
        "MCPAuthorizationCodeSession",
        "MCPAuthHost",
        "MCPRuntimeFactory",
        "MCPRunScope",
        "MCPServerConfig",
    ]


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


def test_conditional_builtin_tool_schema_resolves_to_a_real_file() -> None:
    """条件注入的工具，其 schema 必须真能被框架自己的解析器打开。

    RFC-0028 把 `web_search` 的 `yaml_path` 拼成 `<schema_root>/<name>.tool.yaml`。
    这条路径解析不到文件时，工具会在**注册阶段**失效 —— 表现是模型说"组件加载失败"，
    而不是任何一条像样的报错。
    """
    from nexau.archs.main_sub.config.config import (
        _BUILTIN_TOOL_SCHEMA_ROOT,
        _CONDITIONAL_BUILTIN_TOOL_BINDINGS,
        _resolve_config_resource,
    )

    # 遍历声明表而不是 `_conditional_builtin_bindings()`：后者按环境变量筛，
    # 测试环境没配密钥时会返回空，断言就成了空转。
    assert _CONDITIONAL_BUILTIN_TOOL_BINDINGS, "条件注入表为空，这个测试就失去意义了"
    for name, _binding, _env_keys in _CONDITIONAL_BUILTIN_TOOL_BINDINGS:
        raw = f"{_BUILTIN_TOOL_SCHEMA_ROOT}/{name}.tool.yaml"
        with _resolve_config_resource(raw, PROJECT_ROOT) as resolved:
            assert Path(resolved).is_file(), f"{name} 的 schema 解析不到文件：{raw} → {resolved}"


def test_builtin_tool_schema_root_targets_the_owning_package() -> None:
    """schema 根必须用持有它的那个子包，不能退回顶层 `nexau`。

    这是一条**钉子**，不是复现。真正的失效条件是「repo 目录与包同名 + repo 的父目录
    在 sys.path + 包只能经 editable 转发器找到」，本仓的测试环境天然不满足（包就在
    PROJECT_ROOT/nexau，PathFinder 直接命中正规包），所以没有任何本仓测试能复现它。
    机制本身另有一条自造假包的测试覆盖。

    2026-08-11 在小北镜像里实测：`files("nexau")` → `/app/nexau`（repo 根，少一层），
    `files("nexau.archs.tool.builtin")` → 正确。改回顶层写法就会让 web_search 再次
    静默失效，所以这里明确禁止。
    """
    from nexau.archs.main_sub.config.config import _BUILTIN_TOOL_SCHEMA_ROOT

    pkg, sep, resource = _BUILTIN_TOOL_SCHEMA_ROOT.partition(":")
    assert sep, "schema 根必须是 `<package>:<resource>` 形式"
    assert pkg != "nexau", (
        "不能用顶层包名定位 schema：宿主把 repo 父目录放进 sys.path、且 repo 目录也叫 "
        "nexau 时，PathFinder 会先把它当命名空间包接受，editable 转发器再没机会作答，"
        "解析结果比真实包目录少一层"
    )
    assert pkg.startswith("nexau."), "schema 根仍应落在 nexau 包内"
    assert resource, "resource 部分不能为空"


def test_top_level_package_name_can_be_shadowed_by_a_same_named_directory() -> None:
    """复现遮蔽机制本身：自造一个假包，不依赖 nexau 怎么安装。

    布局照抄小北镜像：
        root/                    ← 放进 sys.path
          pkgx/                  ← repo 目录，与包同名，**没有** __init__.py
            pkgx/__init__.py     ← 真正的包
            pkgx/sub/__init__.py
            pkgx/sub/data.txt
    再按 setuptools editable 的做法，把一个「知道正确映射」的 finder **append** 到
    sys.meta_path（append 是关键：它因此排在 PathFinder 之后）。

    结论应当是：顶层名解析到 repo 根（错），子包解析到真实目录（对）。
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        repo = root / "pkgx"
        pkg = repo / "pkgx"
        (pkg / "sub").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "sub" / "__init__.py").write_text("")
        (pkg / "sub" / "data.txt").write_text("payload")

        script = f"""
        import json, sys
        from importlib.machinery import PathFinder
        from importlib.resources import files
        from pathlib import Path

        root = {str(root)!r}
        real = {str(pkg)!r}
        sys.path.insert(0, root)

        class _Appended:
            # 照抄 setuptools 生成的 editable finder：顶层名按 MAPPING 找，
            # 子模块按「父包的映射目录」定向找（这正是子包不受遮蔽影响的原因）。
            @classmethod
            def find_spec(cls, fullname, path=None, target=None):
                if fullname == "pkgx":
                    return PathFinder.find_spec(fullname, path=[str(Path(real).parent)])
                if fullname.startswith("pkgx."):
                    return PathFinder.find_spec(fullname, path=[real])
                return None

        sys.meta_path.append(_Appended)   # append：排在 PathFinder 之后

        import pkgx
        top = Path(str(files("pkgx").joinpath("sub/data.txt")))
        deep = Path(str(files("pkgx.sub").joinpath("data.txt")))
        print(json.dumps({{
            "top": [str(top), top.is_file()],
            "deep": [str(deep), deep.is_file()],
        }}))
        """
        completed = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            cwd=tmp,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(completed.stdout)

    top_path, top_ok = payload["top"]
    deep_path, deep_ok = payload["deep"]
    assert not top_ok, f"顶层名本应被同名目录遮蔽而解析错，实际拿到了 {top_path}"
    assert deep_ok, f"子包解析本应正确，实际 {deep_path} 不存在"
