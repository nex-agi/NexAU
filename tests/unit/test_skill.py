"""
Unit tests for Skill loading and management.
"""

from pathlib import Path

import pytest

from nexau.archs.main_sub.skill import Skill, load_skill


class TestSkillInit:
    """Test cases for Skill initialization."""

    def test_init_with_all_parameters(self):
        """Test Skill initialization with all parameters."""
        skill = Skill(name="test_skill", description="Test description", detail="Test detail content", folder="/path/to/folder")

        assert skill.name == "test_skill"
        assert skill.description == "Test description"
        assert skill.detail == "Test detail content"
        assert skill.folder == "/path/to/folder"

    def test_init_with_none_values(self):
        """Test Skill initialization with None values."""
        skill = Skill(name="test_skill", description=None, detail=None, folder="")

        assert skill.name == "test_skill"
        assert skill.description is None
        assert skill.detail is None
        assert skill.folder == ""


class TestSkillFromFolder:
    """Test cases for Skill.from_folder classmethod."""

    def test_from_folder_with_valid_skill_md(self, temp_dir):
        """Test loading skill from a valid SKILL.md file."""
        skill_folder = Path(temp_dir) / "test_skill"
        skill_folder.mkdir()

        skill_content = """---
name: test-skill
description: A test skill for testing
---

This is the detailed content of the skill.
It can span multiple lines.

## Section 1
Some content here.
"""
        skill_file = skill_folder / "SKILL.md"
        skill_file.write_text(skill_content)

        skill = Skill.from_folder(skill_folder)

        assert skill.name == "test-skill"
        assert skill.description == "A test skill for testing"
        assert "This is the detailed content" in skill.detail
        assert "## Section 1" in skill.detail
        assert str(skill_folder.absolute()) == skill.folder

    def test_from_folder_missing_skill_md(self, temp_dir):
        """Test loading skill from folder without SKILL.md."""
        skill_folder = Path(temp_dir) / "empty_skill"
        skill_folder.mkdir()

        with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
            Skill.from_folder(skill_folder)

    def test_from_folder_with_absolute_path(self, temp_dir):
        """Test that skill folder is stored as absolute path."""
        skill_folder = Path(temp_dir) / "abs_skill"
        skill_folder.mkdir()

        skill_content = """---
name: absolute-skill
description: Testing absolute paths
---

Detail content.
"""
        (skill_folder / "SKILL.md").write_text(skill_content)

        # Pass absolute path
        skill = Skill.from_folder(skill_folder)

        # Folder should be stored as absolute path string
        assert Path(skill.folder).is_absolute()
        assert skill.folder == str(skill_folder.absolute())


class TestLoadYamlFormatted:
    """Test cases for Skill._load_yaml_formatted classmethod."""

    def test_load_yaml_formatted_valid(self, temp_dir):
        """Test loading valid YAML frontmatter."""
        skill_file = Path(temp_dir) / "test.md"
        content = """---
name: yaml-test
description: Testing YAML parsing
---

Content after frontmatter.
More content here.
"""
        skill_file.write_text(content)

        metadata, detail = Skill._load_yaml_formatted(skill_file)

        assert metadata["name"] == "yaml-test"
        assert metadata["description"] == "Testing YAML parsing"
        assert "Content after frontmatter" in detail
        assert "More content here" in detail

    def test_load_yaml_formatted_no_frontmatter(self, temp_dir):
        """Test loading file without YAML frontmatter."""
        skill_file = Path(temp_dir) / "no_frontmatter.md"
        skill_file.write_text("Just regular content\nNo frontmatter here")

        with pytest.raises(ValueError, match="does not start with YAML frontmatter"):
            Skill._load_yaml_formatted(skill_file)

    def test_load_yaml_formatted_no_closing_marker(self, temp_dir):
        """Test loading file with missing closing --- marker."""
        skill_file = Path(temp_dir) / "incomplete.md"
        content = """---
name: incomplete
description: Missing closing marker

This looks like content but no closing marker
"""
        skill_file.write_text(content)

        with pytest.raises(ValueError, match="does not have closing --- marker"):
            Skill._load_yaml_formatted(skill_file)

    def test_load_yaml_formatted_invalid_yaml(self, temp_dir):
        """Test loading file with invalid YAML syntax."""
        skill_file = Path(temp_dir) / "invalid_yaml.md"
        content = """---
name: invalid
description: [unclosed bracket
---

Content here
"""
        skill_file.write_text(content)

        with pytest.raises(ValueError, match="Failed to parse YAML frontmatter"):
            Skill._load_yaml_formatted(skill_file)

    def test_load_yaml_formatted_first_line_not_marker(self, temp_dir):
        """Test loading file where first line is not exactly ---."""
        skill_file = Path(temp_dir) / "bad_start.md"
        content = """--- extra text
name: bad
---

Content
"""
        skill_file.write_text(content)

        with pytest.raises(ValueError, match="does not start with --- marker"):
            Skill._load_yaml_formatted(skill_file)

    def test_load_yaml_formatted_empty_frontmatter(self, temp_dir):
        """Test loading file with empty frontmatter."""
        skill_file = Path(temp_dir) / "empty.md"
        content = """---
---

Content here
"""
        skill_file.write_text(content)

        metadata, detail = Skill._load_yaml_formatted(skill_file)

        # Empty YAML should return None or empty dict
        assert metadata is None or metadata == {}
        assert "Content here" in detail

    def test_load_yaml_formatted_whitespace_handling(self, temp_dir):
        """Test that whitespace in frontmatter is handled correctly."""
        skill_file = Path(temp_dir) / "whitespace.md"
        content = """---
name:   whitespace-test
description:   Testing whitespace
extra_field: value
---

Detail content with leading/trailing spaces.
"""
        skill_file.write_text(content)

        metadata, detail = Skill._load_yaml_formatted(skill_file)

        # YAML should handle whitespace automatically
        assert metadata["name"] == "whitespace-test"
        assert metadata["description"] == "Testing whitespace"
        assert metadata["extra_field"] == "value"
        assert detail.startswith("Detail content")

    def test_load_yaml_formatted_multiline_values(self, temp_dir):
        """Test loading YAML with multiline values."""
        skill_file = Path(temp_dir) / "multiline.md"
        content = """---
name: multiline-test
description: |
  This is a multiline
  description that spans
  multiple lines
---

Detail content.
"""
        skill_file.write_text(content)

        metadata, detail = Skill._load_yaml_formatted(skill_file)

        assert "multiline" in metadata["description"].lower()
        assert "multiple lines" in metadata["description"]


class TestLoadSkill:
    """Test cases for load_skill function."""

    def test_load_skill_success(self, agent_state):
        """Test successfully loading a skill from agent state."""
        # Set up skill in agent state
        test_skill = Skill(
            name="test-skill", description="Test skill description", detail="Detailed skill information", folder="/path/to/skill"
        )

        skill_registry = {"test-skill": test_skill}
        agent_state.set_global_value("skill_registry", skill_registry)

        result = load_skill("test-skill", agent_state)

        assert "Found the skill details of `test-skill`" in result
        assert "<SkillName>test-skill</SkillName>" in result
        assert "<SkillFolder>/path/to/skill</SkillFolder>" in result
        assert "<SkillDescription>Test skill description</SkillDescription>" in result
        assert "<SkillDetail>Detailed skill information</SkillDetail>" in result
        assert "Note that the paths mentioned in skill description" in result

    def test_load_skill_not_found(self, agent_state):
        """Test loading a skill that doesn't exist."""
        # Empty skill registry
        agent_state.set_global_value("skill_registry", {})

        with pytest.raises(ValueError, match="Skill nonexistent not found"):
            load_skill("nonexistent", agent_state)

    def test_load_skill_empty_registry(self, agent_state):
        """Test loading skill when registry doesn't exist."""
        # No skill registry set

        with pytest.raises(ValueError, match="Skill test-skill not found"):
            load_skill("test-skill", agent_state)

    def test_load_skill_with_special_characters(self, agent_state):
        """Test loading skill with special characters in content."""
        test_skill = Skill(
            name="special-skill",
            description="Description with <special> & characters",
            detail="Detail with ```code``` and **markdown**",
            folder="/path/with/special chars",
        )

        skill_registry = {"special-skill": test_skill}
        agent_state.set_global_value("skill_registry", skill_registry)

        result = load_skill("special-skill", agent_state)

        # Special characters should be preserved
        assert "<special>" in result
        assert "&" in result
        assert "```code```" in result
        assert "**markdown**" in result

    def test_load_skill_with_none_values(self, agent_state):
        """Test loading skill with None description and detail."""
        test_skill = Skill(name="minimal-skill", description=None, detail=None, folder="")

        skill_registry = {"minimal-skill": test_skill}
        agent_state.set_global_value("skill_registry", skill_registry)

        result = load_skill("minimal-skill", agent_state)

        assert "<SkillName>minimal-skill</SkillName>" in result
        assert "<SkillDescription>None</SkillDescription>" in result
        assert "<SkillDetail>None</SkillDetail>" in result


class TestSkillIntegration:
    """Integration tests for Skill functionality."""

    def test_full_workflow_from_folder_to_load(self, temp_dir, agent_state):
        """Test complete workflow: create SKILL.md, load it, and retrieve it."""
        # Create a skill folder with SKILL.md
        skill_folder = Path(temp_dir) / "integration_skill"
        skill_folder.mkdir()

        skill_content = """---
name: integration-skill
description: Full integration test skill
---

# Integration Skill Details

This skill demonstrates the full workflow from folder to retrieval.

## Features
- Feature 1
- Feature 2
"""
        (skill_folder / "SKILL.md").write_text(skill_content)

        # Load the skill from folder
        skill = Skill.from_folder(skill_folder)

        # Register it in agent state
        skill_registry = {skill.name: skill}
        agent_state.set_global_value("skill_registry", skill_registry)

        # Load and verify
        result = load_skill("integration-skill", agent_state)

        assert "integration-skill" in result
        assert "Full integration test skill" in result
        assert "Feature 1" in result
        assert "Feature 2" in result

    def test_multiple_skills_in_registry(self, temp_dir, agent_state):
        """Test managing multiple skills in the registry."""
        skills = []

        for i in range(3):
            folder = Path(temp_dir) / f"skill_{i}"
            folder.mkdir()

            content = f"""---
name: skill-{i}
description: Skill number {i}
---

Details for skill {i}
"""
            (folder / "SKILL.md").write_text(content)
            skill = Skill.from_folder(folder)
            skills.append(skill)

        # Register all skills
        skill_registry = {skill.name: skill for skill in skills}
        agent_state.set_global_value("skill_registry", skill_registry)

        # Load each skill and verify
        for i in range(3):
            result = load_skill(f"skill-{i}", agent_state)
            assert f"skill-{i}" in result
            assert f"Skill number {i}" in result
            assert f"Details for skill {i}" in result
