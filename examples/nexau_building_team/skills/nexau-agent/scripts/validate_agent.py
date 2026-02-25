#!/usr/bin/env python3
"""NexAU Agent YAML validator.

Uses AgentConfigSchema.from_yaml() for schema validation, then checks
that all referenced files (tool YAMLs, system prompt, skills, sub-agent
configs) actually exist and are well-formed.

Usage::

    python validate_agent.py <agent.yaml> [--recursive] [--json]
"""

from __future__ import annotations

import argparse
import json as json_mod
import re
import sys
from contextlib import redirect_stderr
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import ValidationError

from nexau.archs.main_sub.config.schema import (
    AgentConfigSchema,
    ConfigError,
)
from nexau.archs.tool.tool import ToolYamlSchema


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------


@dataclass
class Issue:
    severity: str  # "ERROR" | "WARNING"
    category: str
    message: str
    path: str | None = None


@dataclass
class Report:
    yaml_path: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == "ERROR" for i in self.issues)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "ERROR")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "WARNING")


_ENV_VAR_RE = re.compile(r"\$\{env\.")
RESERVED_PARAM_KEYS: set[str] = {"agent_state", "global_storage"}


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _resolve(yaml_dir: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (yaml_dir / p).resolve()


def _has_env_var(value: str) -> bool:
    return bool(_ENV_VAR_RE.search(value))


# -------------------------------------------------------------------
# Phase 1: Schema validation via AgentConfigSchema.from_yaml()
# -------------------------------------------------------------------


def _validate_schema(yaml_path: str, issues: list[Issue]) -> None:
    """Use the framework's own Pydantic validation."""
    try:
        # from_yaml 内部会 traceback.print_exc()，这里抑制 stderr 输出
        with redirect_stderr(StringIO()):
            AgentConfigSchema.from_yaml(yaml_path)
    except ConfigError as exc:
        msg = str(exc)
        # 环境变量未设置时降级为 WARNING（验证场景下通常不会设置）
        if "is not set" in msg and "Environment variable" in msg:
            issues.append(Issue("WARNING", "schema", msg))
        else:
            issues.append(Issue("ERROR", "schema", msg))
    except Exception as exc:  # noqa: BLE001
        issues.append(Issue("ERROR", "schema", f"unexpected error: {exc}"))


# -------------------------------------------------------------------
# Phase 2: File-reference validation (raw YAML, env vars unresolved)
# -------------------------------------------------------------------


def _load_raw(yaml_path: Path) -> dict[str, Any] | None:
    """Load YAML without env-var substitution."""
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    if isinstance(data, dict):
        return cast(dict[str, Any], data)
    return None


def _check_system_prompt(
    config: dict[str, Any],
    yaml_dir: Path,
    issues: list[Issue],
) -> None:
    sp_type: str = str(config.get("system_prompt_type", "string"))
    sp_value = config.get("system_prompt")

    if sp_type in ("file", "jinja"):
        if not sp_value or not isinstance(sp_value, str):
            issues.append(
                Issue(
                    "ERROR",
                    "system_prompt",
                    f"system_prompt is required when system_prompt_type is '{sp_type}'",
                )
            )
            return
        if _has_env_var(sp_value):
            issues.append(
                Issue(
                    "WARNING",
                    "system_prompt",
                    f"system_prompt contains env var, cannot verify: {sp_value}",
                )
            )
            return
        if not _resolve(yaml_dir, sp_value).is_file():
            issues.append(
                Issue(
                    "ERROR",
                    "system_prompt",
                    f"file not found: {sp_value}",
                )
            )


def _check_tool_yaml_content(
    tool_path: Path,
    idx: int,
    issues: list[Issue],
) -> str | None:
    """Validate tool YAML via ToolYamlSchema. Return binding if present."""
    try:
        with open(tool_path, encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        issues.append(
            Issue(
                "ERROR",
                "tool",
                f"tools[{idx}] YAML parse error: {exc}",
                f"tools[{idx}].yaml_path",
            )
        )
        return None

    if not isinstance(raw, dict):
        issues.append(
            Issue(
                "ERROR",
                "tool",
                f"tools[{idx}] YAML is not a mapping",
                f"tools[{idx}].yaml_path",
            )
        )
        return None

    raw_dict = cast(dict[str, Any], raw)
    try:
        schema = ToolYamlSchema.model_validate(raw_dict)
    except ValidationError as exc:
        for err in exc.errors():
            loc = "->".join(str(s) for s in err.get("loc", [])) or "root"
            issues.append(
                Issue(
                    "ERROR",
                    "tool",
                    f"tools[{idx}] YAML validation: {loc}: {err.get('msg')}",
                    f"tools[{idx}].yaml_path",
                )
            )
        return None

    # input_schema 中的保留属性
    props = schema.input_schema.get("properties", {})
    if isinstance(props, dict):
        props_dict = cast(dict[str, Any], props)
        for rk in RESERVED_PARAM_KEYS & set(props_dict.keys()):
            issues.append(
                Issue(
                    "WARNING",
                    "tool",
                    f"tools[{idx}] YAML input_schema contains reserved property '{rk}'",
                    f"tools[{idx}].yaml_path",
                )
            )

    return schema.binding


def _check_tools(
    config: dict[str, Any],
    yaml_dir: Path,
    issues: list[Issue],
) -> None:
    tools_raw = config.get("tools", [])
    if not isinstance(tools_raw, list):
        return

    tools = cast(list[Any], tools_raw)
    for idx, entry in enumerate(tools):
        if not isinstance(entry, dict):
            issues.append(Issue("ERROR", "tool", f"tools[{idx}] is not a mapping"))
            continue

        entry_dict = cast(dict[str, Any], entry)
        name: str | None = entry_dict.get("name") if isinstance(entry_dict.get("name"), str) else None
        yaml_path_val = entry_dict.get("yaml_path")

        if not yaml_path_val or not isinstance(yaml_path_val, str):
            issues.append(
                Issue(
                    "ERROR",
                    "tool",
                    f"tools[{idx}] missing 'yaml_path'",
                    f"tools[{idx}].yaml_path",
                )
            )
            continue

        if _has_env_var(yaml_path_val):
            issues.append(
                Issue(
                    "WARNING",
                    "tool",
                    f"tools[{idx}] yaml_path contains env var, cannot verify: {yaml_path_val}",
                    f"tools[{idx}].yaml_path",
                )
            )
            continue

        resolved = _resolve(yaml_dir, yaml_path_val)
        if not resolved.is_file():
            issues.append(
                Issue(
                    "ERROR",
                    "tool",
                    f"tools[{idx}] yaml_path file not found: {yaml_path_val}",
                    f"tools[{idx}].yaml_path",
                )
            )
            continue

        # 验证 tool YAML 内容
        tool_binding = _check_tool_yaml_content(resolved, idx, issues)

        # binding 检查
        agent_binding = entry_dict.get("binding")
        if not agent_binding and not tool_binding:
            label = f" ({name})" if name else ""
            issues.append(
                Issue(
                    "WARNING",
                    "tool",
                    f"tools[{idx}]{label}: no binding in agent config or tool YAML",
                    f"tools[{idx}].binding",
                )
            )

        # extra_kwargs 保留 key
        extra = entry_dict.get("extra_kwargs", {})
        if isinstance(extra, dict):
            extra_dict = cast(dict[str, Any], extra)
            for rk in RESERVED_PARAM_KEYS & set(extra_dict.keys()):
                issues.append(
                    Issue(
                        "ERROR",
                        "tool",
                        f"tools[{idx}] extra_kwargs contains reserved key '{rk}'",
                        f"tools[{idx}].extra_kwargs",
                    )
                )


def _check_skill(skill_path: Path, idx: int, issues: list[Issue]) -> None:
    if not skill_path.is_dir():
        issues.append(
            Issue(
                "ERROR",
                "skill",
                f"skills[{idx}]: directory not found: {skill_path}",
                f"skills[{idx}]",
            )
        )
        return

    skill_md = skill_path / "SKILL.md"
    if not skill_md.is_file():
        issues.append(
            Issue(
                "ERROR",
                "skill",
                f"skills[{idx}]: SKILL.md not found in {skill_path}",
                f"skills[{idx}]",
            )
        )
        return

    content = skill_md.read_text(encoding="utf-8")
    if not content.startswith("---"):
        issues.append(
            Issue(
                "ERROR",
                "skill",
                f"skills[{idx}]: SKILL.md has no YAML frontmatter",
                f"skills[{idx}]",
            )
        )
        return

    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        issues.append(
            Issue(
                "ERROR",
                "skill",
                f"skills[{idx}]: SKILL.md has invalid frontmatter format",
                f"skills[{idx}]",
            )
        )
        return

    fm = match.group(1)
    if "name:" not in fm:
        issues.append(
            Issue(
                "ERROR",
                "skill",
                f"skills[{idx}]: SKILL.md frontmatter missing 'name'",
                f"skills[{idx}]",
            )
        )
    else:
        name_match = re.search(r"name:\s*(.+)", fm)
        if name_match:
            name_val = name_match.group(1).strip()
            if not re.match(r"^[a-z0-9-]+$", name_val):
                issues.append(
                    Issue(
                        "WARNING",
                        "skill",
                        f"skills[{idx}]: name '{name_val}' does not match hyphen-case",
                        f"skills[{idx}]",
                    )
                )

    if "description:" not in fm:
        issues.append(
            Issue(
                "ERROR",
                "skill",
                f"skills[{idx}]: SKILL.md frontmatter missing 'description'",
                f"skills[{idx}]",
            )
        )


def _check_skills(
    config: dict[str, Any],
    yaml_dir: Path,
    issues: list[Issue],
) -> None:
    skills_raw = config.get("skills", [])
    if not isinstance(skills_raw, list):
        return

    skills = cast(list[Any], skills_raw)
    for idx, entry in enumerate(skills):
        if not isinstance(entry, str):
            issues.append(
                Issue(
                    "ERROR",
                    "skill",
                    f"skills[{idx}] is not a string path",
                    f"skills[{idx}]",
                )
            )
            continue
        if _has_env_var(entry):
            issues.append(
                Issue(
                    "WARNING",
                    "skill",
                    f"skills[{idx}] contains env var, cannot verify: {entry}",
                    f"skills[{idx}]",
                )
            )
            continue
        _check_skill(_resolve(yaml_dir, entry), idx, issues)


def _check_sub_agents(
    config: dict[str, Any],
    yaml_dir: Path,
    issues: list[Issue],
    recursive: bool,
    visited: set[str],
) -> None:
    subs_raw = config.get("sub_agents", [])
    if not isinstance(subs_raw, list):
        return

    subs = cast(list[Any], subs_raw)
    for idx, entry in enumerate(subs):
        if not isinstance(entry, dict):
            issues.append(
                Issue(
                    "ERROR",
                    "sub_agent",
                    f"sub_agents[{idx}] is not a mapping",
                    f"sub_agents[{idx}]",
                )
            )
            continue

        entry_dict = cast(dict[str, Any], entry)
        if "config_path" not in entry_dict:
            issues.append(
                Issue(
                    "ERROR",
                    "sub_agent",
                    f"sub_agents[{idx}] missing 'config_path'",
                    f"sub_agents[{idx}].config_path",
                )
            )
            continue

        cfg_path = str(entry_dict["config_path"])
        if _has_env_var(cfg_path):
            issues.append(
                Issue(
                    "WARNING",
                    "sub_agent",
                    f"sub_agents[{idx}] config_path contains env var: {cfg_path}",
                    f"sub_agents[{idx}].config_path",
                )
            )
            continue

        resolved = _resolve(yaml_dir, cfg_path)
        if not resolved.is_file():
            issues.append(
                Issue(
                    "ERROR",
                    "sub_agent",
                    f"sub_agents[{idx}] config_path file not found: {cfg_path}",
                    f"sub_agents[{idx}].config_path",
                )
            )
            continue

        if recursive:
            real = str(resolved.resolve())
            if real in visited:
                issues.append(
                    Issue(
                        "WARNING",
                        "sub_agent",
                        f"sub_agents[{idx}] circular reference: {cfg_path}",
                        f"sub_agents[{idx}].config_path",
                    )
                )
                continue
            sub_report = validate_agent_yaml(
                str(resolved),
                recursive=True,
                _visited=visited,
            )
            sub_name = entry_dict.get("name", idx)
            for si in sub_report.issues:
                issues.append(
                    Issue(
                        si.severity,
                        si.category,
                        f"[sub_agent '{sub_name}'] {si.message}",
                        si.path,
                    )
                )


def _check_hooks(
    config: dict[str, Any],
    issues: list[Issue],
) -> None:
    hook_fields = (
        "middlewares",
        "after_model_hooks",
        "after_tool_hooks",
        "before_model_hooks",
        "before_tool_hooks",
        "tracers",
    )
    for field_name in hook_fields:
        entries_raw = config.get(field_name)
        if not isinstance(entries_raw, list):
            continue
        entries = cast(list[Any], entries_raw)
        for idx, entry in enumerate(entries):
            if isinstance(entry, str):
                if ":" not in entry:
                    issues.append(
                        Issue(
                            "WARNING",
                            "hook",
                            f"{field_name}[{idx}]: import path missing ':' separator",
                            f"{field_name}[{idx}]",
                        )
                    )
            elif isinstance(entry, dict):
                entry_dict = cast(dict[str, Any], entry)
                if "import" not in entry_dict:
                    issues.append(
                        Issue(
                            "WARNING",
                            "hook",
                            f"{field_name}[{idx}]: dict entry missing 'import' key",
                            f"{field_name}[{idx}]",
                        )
                    )
                else:
                    imp = entry_dict["import"]
                    if isinstance(imp, str) and ":" not in imp:
                        issues.append(
                            Issue(
                                "WARNING",
                                "hook",
                                f"{field_name}[{idx}]: import path missing ':' separator",
                                f"{field_name}[{idx}]",
                            )
                        )


# -------------------------------------------------------------------
# Main validator
# -------------------------------------------------------------------


def validate_agent_yaml(
    yaml_path: str,
    *,
    recursive: bool = False,
    _visited: set[str] | None = None,
) -> Report:
    """Validate an agent YAML and all its references."""
    report = Report(yaml_path=yaml_path)
    path = Path(yaml_path).resolve()

    if _visited is None:
        _visited = set()
    _visited.add(str(path))

    # Phase 1: schema validation via framework
    _validate_schema(str(path), report.issues)

    # Phase 2: file-reference checks (raw YAML, env vars unresolved)
    config = _load_raw(path)
    if config is None:
        if not report.issues:
            report.issues.append(
                Issue(
                    "ERROR",
                    "syntax",
                    f"cannot load YAML: {yaml_path}",
                )
            )
        return report

    yaml_dir = path.parent
    _check_system_prompt(config, yaml_dir, report.issues)
    _check_tools(config, yaml_dir, report.issues)
    _check_skills(config, yaml_dir, report.issues)
    _check_sub_agents(config, yaml_dir, report.issues, recursive, _visited)
    _check_hooks(config, report.issues)

    return report


# -------------------------------------------------------------------
# Output
# -------------------------------------------------------------------


def _print_report(report: Report) -> None:
    print(f"Validating: {report.yaml_path}")
    print("=" * 60)

    if report.issues:
        print()
        for issue in report.issues:
            sev = issue.severity.ljust(7)
            cat = issue.category.ljust(14)
            print(f"[{sev}] {cat} | {issue.message}")
        print()
    else:
        print("\n  No issues found.\n")

    print("=" * 60)
    status = "VALID" if report.is_valid else "INVALID"
    print(f"Result: {status} ({report.error_count} errors, {report.warning_count} warnings)")


def _print_json(report: Report) -> None:
    data = {
        "yaml_path": report.yaml_path,
        "valid": report.is_valid,
        "error_count": report.error_count,
        "warning_count": report.warning_count,
        "issues": [asdict(i) for i in report.issues],
    }
    print(json_mod.dumps(data, indent=2, ensure_ascii=False))


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a NexAU agent YAML and its references.",
    )
    parser.add_argument("agent_yaml", help="Path to the agent YAML file")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively validate sub-agent configs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON",
    )
    args = parser.parse_args()

    report = validate_agent_yaml(args.agent_yaml, recursive=args.recursive)

    if args.json_output:
        _print_json(report)
    else:
        _print_report(report)

    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
