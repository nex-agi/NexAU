"""Regression tests for the built-in tool schema single source of truth.

RFC-0197: `.tool.yaml` schema 单一事实源收口。
"""

import subprocess
from collections import defaultdict
from difflib import unified_diff
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_ROOT = REPO_ROOT / "nexau" / "archs" / "tool" / "builtin"
BUILTIN_SCHEMA_DIR = BUILTIN_ROOT / "schemas"


def _tracked_tool_schema_paths() -> list[Path]:
    """Return tracked tool schemas from the current repository."""
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.tool.yaml"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [REPO_ROOT / raw_path for raw_path in result.stdout.split("\0") if raw_path]


def _duplicate_tool_schema_groups() -> list[tuple[str, tuple[Path, ...]]]:
    """Group tracked schemas by basename and keep duplicated names."""
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in _tracked_tool_schema_paths():
        grouped_paths[path.name].append(path)
    return [(name, tuple(sorted(paths))) for name, paths in sorted(grouped_paths.items()) if len(paths) > 1]


DUPLICATE_TOOL_SCHEMA_GROUPS = _duplicate_tool_schema_groups()


@pytest.mark.parametrize(
    ("schema_name", "schema_paths"),
    DUPLICATE_TOOL_SCHEMA_GROUPS,
    ids=[name for name, _ in DUPLICATE_TOOL_SCHEMA_GROUPS],
)
def test_tool_schema_single_source_of_truth(
    schema_name: str,
    schema_paths: tuple[Path, ...],
) -> None:
    """Require byte-identical content for every duplicated schema basename.

    RFC-0197: 防止仓内 `.tool.yaml` 副本再次漂移。
    """
    reference_path = schema_paths[0]
    reference_content = reference_path.read_bytes()
    drift_diffs: list[str] = []

    for candidate_path in schema_paths[1:]:
        candidate_content = candidate_path.read_bytes()
        if candidate_content == reference_content:
            continue
        drift_diffs.append(
            "".join(
                unified_diff(
                    reference_content.decode("utf-8", errors="replace").splitlines(keepends=True),
                    candidate_content.decode("utf-8", errors="replace").splitlines(keepends=True),
                    fromfile=reference_path.relative_to(REPO_ROOT).as_posix(),
                    tofile=candidate_path.relative_to(REPO_ROOT).as_posix(),
                )
            )
        )

    if drift_diffs:
        pytest.fail(
            f"Schema drift detected for {schema_name}:\n" + "\n".join(drift_diffs),
            pytrace=False,
        )


def _string_key_mapping(value: object) -> dict[str, object]:
    """Narrow a parsed YAML mapping to string keys."""
    assert isinstance(value, dict)
    assert all(isinstance(key, str) for key in value)
    return {key: item for key, item in value.items() if isinstance(key, str)}


def test_ask_user_schema_keeps_both_hardenings() -> None:
    """Keep both structural constraints and defensive prose for ask_user.

    RFC-0197: 同时钉死 type 必填、options 硬约束与防呆/推荐项提示。
    """
    schema = _string_key_mapping(yaml.safe_load((BUILTIN_SCHEMA_DIR / "ask_user.tool.yaml").read_text(encoding="utf-8")))
    input_schema = _string_key_mapping(schema["input_schema"])
    properties = _string_key_mapping(input_schema["properties"])
    questions = _string_key_mapping(properties["questions"])
    question_items = _string_key_mapping(questions["items"])
    question_properties = _string_key_mapping(question_items["properties"])
    options = _string_key_mapping(question_properties["options"])

    question_required = question_items["required"]
    assert isinstance(question_required, list)
    assert "type" in question_required
    assert questions["maxItems"] == 4
    assert options["minItems"] == 2
    assert options["maxItems"] == 4

    description = schema["description"]
    assert isinstance(description, str)
    assert "empty `options` list" in description
    assert "it is rejected" in description
    assert "prefer 'text'" in description
    assert "Use `type: text`" in description

    options_description = options["description"]
    assert isinstance(options_description, str)
    assert "Order options by recommendation" in options_description
    assert '" (Recommended)"' in options_description
    assert '"（推荐）"' in options_description
    assert "Mark only one option as recommended" in options_description


def test_builtin_schema_dir_is_unique() -> None:
    """Keep one built-in schema directory after the Phase 1c consolidation.

    RFC-0197: `schemas/` 是 builtin 下唯一存放工具 schema 的目录。
    """
    schema_directories = sorted(path for path in BUILTIN_ROOT.iterdir() if path.is_dir() and any(path.glob("*.yaml")))

    assert schema_directories == [BUILTIN_SCHEMA_DIR]
    assert not (BUILTIN_ROOT / "description").exists()
