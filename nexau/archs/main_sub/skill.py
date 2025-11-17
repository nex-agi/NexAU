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

from pathlib import Path
from typing import Any

import yaml

from nexau.archs.main_sub.agent_state import AgentState


class Skill:
    def __init__(self, name: str, description: str, detail: str, folder: str):
        self.name: str = name
        self.description: str | None = description
        self.detail: str | None = detail
        self.folder: str = folder

    @classmethod
    def from_folder(cls, folder: Path) -> "Skill":
        """Load a skill from a YAML file."""
        folder = Path(folder).absolute()
        # Try to find SKILL.md or SKILL.yaml
        skill_md = folder / "SKILL.md"

        if skill_md.exists():
            # Load from SKILL.md with YAML frontmatter
            skill_data, detail_content = cls._load_yaml_formatted(skill_md)
            return cls(name=skill_data["name"], description=skill_data["description"], detail=detail_content, folder=str(folder))  # type: ignore
        else:
            raise FileNotFoundError(f"SKILL.md not found in {folder}")

    @classmethod
    def _load_yaml_formatted(cls, skill_path: Path) -> tuple[dict[str, Any], str]:  # type: ignore
        """Parse YAML frontmatter from a file.

        Expected format:
        ---
        name: skill-name
        description: skill description
        ---

        (rest of file content for detail)

        Returns:
            tuple: (metadata dict, content after frontmatter)
        """
        with open(skill_path) as f:
            content = f.read()

        # Check if file starts with YAML frontmatter
        if not content.startswith("---"):
            raise ValueError(f"File {skill_path} does not start with YAML frontmatter (---)")

        # Find the closing --- marker
        lines = content.split("\n")
        if lines[0].strip() != "---":
            raise ValueError(f"File {skill_path} does not start with --- marker")

        # Find the second --- marker
        end_idx = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break

        if end_idx is None:
            raise ValueError(f"File {skill_path} does not have closing --- marker for YAML frontmatter")

        # Extract YAML content between the --- markers
        yaml_content = "\n".join(lines[1:end_idx])

        # Extract content after frontmatter
        detail_content = "\n".join(lines[end_idx + 1 :]).strip()

        # Parse YAML
        try:
            metadata: dict[str, Any] = yaml.safe_load(yaml_content)
            return metadata, detail_content
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML frontmatter in {skill_path}: {e}")


def load_skill(skill_name: str, agent_state: "AgentState") -> str:
    """Load a skill from skill folders."""
    skills: dict[str, Skill] = agent_state.get_global_value("skill_registry", {})
    if skill_name not in skills:
        raise ValueError(f"Skill {skill_name} not found")
    skill = skills[skill_name]
    response = f"Found the skill details of `{skill.name}`.\n"
    response += "Note that the paths mentioned in skill description are relative to the skill folder.\n"
    response += f"""<SkillDetails>
<SkillName>{skill.name}</SkillName>
<SkillFolder>{skill.folder}</SkillFolder>
<SkillDescription>{skill.description}</SkillDescription>
<SkillDetail>{skill.detail}</SkillDetail>
</SkillDetails>"""
    return response
