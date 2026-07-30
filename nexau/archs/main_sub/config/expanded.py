"""Internal typed carriers produced after config expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

YamlValue = dict[str, "YamlValue"] | list["YamlValue"] | str | int | float | bool | None


@dataclass(frozen=True)
class ExpandedSubAgentConfig:
    """A plugin-rendered sub-agent config ready for in-memory loading."""

    name: str
    config_path: str
    source_id: str
    inline_config: dict[str, YamlValue]
    inline_base_path: Path
    timeout_seconds: float | None = None
    max_calls_per_run: int | None = None
