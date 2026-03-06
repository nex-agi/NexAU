# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Official Codex scenario fixture compatibility tests.

Runs the NexAU apply_patch builtin tool against all 22 official Codex
scenario fixtures and verifies that the resulting filesystem state matches.

Official refs:
  - codex-rs/apply-patch/tests/fixtures/scenarios/001..022
  - codex-rs/apply-patch/tests/suite/scenarios.rs

Requires the official codex-rs checkout to be present. Set the environment
variable CODEX_RS_PATH to point to the clone root, or the tests will attempt
the default location ~/codex/codex-rs. If the fixtures are not found, the
entire module is skipped.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from nexau.archs.sandbox.local_sandbox import LocalSandbox
from nexau.archs.tool.builtin.file_tools.apply_patch import apply_patch

# ---------------------------------------------------------------------------
# Locate official scenario fixtures
# ---------------------------------------------------------------------------
_CODEX_RS_DEFAULT = os.path.expanduser("~/codex/codex-rs")
_CODEX_RS_PATH = os.environ.get("CODEX_RS_PATH", _CODEX_RS_DEFAULT)
_SCENARIOS_ROOT = os.path.join(_CODEX_RS_PATH, "apply-patch", "tests", "fixtures", "scenarios")

_HAS_FIXTURES = os.path.isdir(_SCENARIOS_ROOT)


def _discover_scenarios():
    """Return sorted list of scenario directory Paths."""
    if not _HAS_FIXTURES:
        return []
    root = Path(_SCENARIOS_ROOT)
    return sorted(p for p in root.iterdir() if p.is_dir())


def _snapshot_dir(root: Path) -> dict[str, tuple[str, bytes | None]]:
    """Snapshot a directory tree as {relative_posix_path: ('file'|'dir', content)}."""
    entries: dict[str, tuple[str, bytes | None]] = {}
    if not root.is_dir():
        return entries
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_dir():
            entries[rel] = ("dir", None)
        elif path.is_file():
            entries[rel] = ("file", path.read_bytes())
    return entries


def _make_agent_state(work_dir: str):
    sandbox = LocalSandbox(sandbox_id="test", work_dir=work_dir)
    agent_state = Mock()
    agent_state.get_sandbox.return_value = sandbox
    return agent_state


# ---------------------------------------------------------------------------
# Parametrised test – one case per scenario directory
# ---------------------------------------------------------------------------
_SCENARIO_DIRS: list[Path] = _discover_scenarios()
_SCENARIO_IDS: list[str] = [p.name for p in _SCENARIO_DIRS]


@pytest.mark.skipif(not _HAS_FIXTURES, reason="Official Codex fixtures not found")
@pytest.mark.parametrize("scenario", _SCENARIO_DIRS, ids=_SCENARIO_IDS)
def test_official_scenario(scenario: Path):
    """Execute a single official scenario and compare the filesystem snapshot."""
    tmp = Path(tempfile.mkdtemp(prefix="compat-scenario-"))
    try:
        # 1. Seed the input directory
        input_dir = scenario / "input"
        if input_dir.is_dir():
            for src in input_dir.rglob("*"):
                rel = src.relative_to(input_dir)
                dst = tmp / rel
                if src.is_dir():
                    dst.mkdir(parents=True, exist_ok=True)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

        # 2. Read patch text
        patch_text = (scenario / "patch.txt").read_text(encoding="utf-8")

        # 3. Run apply_patch
        agent_state = _make_agent_state(str(tmp))
        apply_patch(input=patch_text, agent_state=agent_state)

        # 4. Compare filesystem state against expected/
        actual = _snapshot_dir(tmp)
        expected = _snapshot_dir(scenario / "expected")
        assert actual == expected, _diff_summary(actual, expected, scenario.name)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _diff_summary(
    actual: dict[str, tuple[str, bytes | None]],
    expected: dict[str, tuple[str, bytes | None]],
    name: str,
) -> str:
    """Build a human-readable diff summary for assertion failures."""
    lines = [f"Scenario {name} filesystem mismatch:"]
    ak, ek = set(actual), set(expected)
    only_actual = sorted(ak - ek)
    only_expected = sorted(ek - ak)
    if only_actual:
        lines.append(f"  only in actual: {only_actual}")
    if only_expected:
        lines.append(f"  only in expected: {only_expected}")
    for k in sorted(ak & ek):
        if actual[k] != expected[k]:
            atype, adata = actual[k]
            etype, edata = expected[k]
            lines.append(f"  mismatch {k}: actual=({atype}, {len(adata or b'')} bytes) expected=({etype}, {len(edata or b'')} bytes)")
    return "\n".join(lines)
