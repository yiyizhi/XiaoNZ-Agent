"""Skill loader.

Skill directory layout (compatible with the AgentSkills standard):

    data/skills/
        <skill-name>/
            SKILL.md            # YAML frontmatter + markdown body
            <any other files>   # referenced from SKILL.md if needed

SKILL.md frontmatter (minimum):

    ---
    name: web-search
    description: Search the web for up-to-date information.
    ---

    # Body is plain markdown, loaded verbatim into the model context
    # when the model calls `load_skill(name="web-search")`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (meta, body). Tolerates missing frontmatter."""
    if not text.startswith("---"):
        return {}, text
    # Find the closing '---' on its own line
    lines = text.splitlines(keepends=True)
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx == -1:
        return {}, text
    fm_text = "".join(lines[1:close_idx])
    body = "".join(lines[close_idx + 1 :]).lstrip("\n")
    try:
        meta = yaml.safe_load(fm_text) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError as e:
        logger.warning("skill.frontmatter_parse_failed err=%s", e)
        meta = {}
    return meta, body


class SkillStore:
    def __init__(self, skills_dir: Path):
        self.dir = skills_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _load_one(self, skill_dir: Path) -> Skill | None:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            return None
        text = skill_file.read_text(encoding="utf-8")
        meta, body = _parse_frontmatter(text)
        name = str(meta.get("name") or skill_dir.name).strip()
        description = str(meta.get("description") or "").strip()
        if not description:
            logger.warning("skill.missing_description name=%s", name)
        return Skill(
            name=name,
            description=description,
            body=body.strip(),
            path=skill_file,
        )

    def list_all(self) -> list[Skill]:
        """Scan skills_dir for every subdirectory containing SKILL.md."""
        skills: list[Skill] = []
        for child in sorted(self.dir.iterdir()) if self.dir.exists() else []:
            if not child.is_dir():
                continue
            skill = self._load_one(child)
            if skill is not None:
                skills.append(skill)
        return skills

    def get(self, name: str) -> Skill | None:
        """Look up a skill by its frontmatter `name` (falls back to dir name)."""
        for skill in self.list_all():
            if skill.name == name:
                return skill
        return None
