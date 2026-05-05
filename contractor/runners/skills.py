from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from contractor.tools.memory import MemoryNote, MemoryTools

SKILLS_BASE_DIR = Path(__file__).parent.parent / "skills"

_INDEX_FILENAME = "index.md"
_MD_SUFFIX = ".md"


@dataclass(slots=True, frozen=True)
class SkillFile:
    skill: str
    name: str
    description: str
    content: str
    is_index: bool


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text

    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}, text

    if not isinstance(meta, dict):
        return {}, text

    return meta, parts[2].lstrip("\n")


def _memory_name(skill: str, rel_path: Path) -> tuple[str, bool]:
    if rel_path.name == _INDEX_FILENAME:
        return skill, True
    rel_no_ext = rel_path.with_suffix("").as_posix()
    return f"{skill}/{rel_no_ext}", False


def _default_description(skill: str, rel_path: Path, is_index: bool) -> str:
    if is_index:
        return f"{skill} skill"
    return f"{skill} skill / {rel_path.with_suffix('').as_posix()}"


def load_skill(skill: str) -> list[SkillFile]:
    skill_dir = SKILLS_BASE_DIR / skill
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"skill {skill!r} not found at {skill_dir}")

    files: list[SkillFile] = []
    for path in sorted(skill_dir.rglob(f"*{_MD_SUFFIX}")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(skill_dir)
        text = path.read_text(encoding="utf-8")
        meta, content = _parse_frontmatter(text)
        name, is_index = _memory_name(skill, rel_path)
        description = (
            meta.get("description")
            or _default_description(skill, rel_path, is_index)
        )
        files.append(
            SkillFile(
                skill=skill,
                name=name,
                description=str(description),
                content=content,
                is_index=is_index,
            )
        )

    return files


def load_skills(skills: Iterable[str]) -> list[SkillFile]:
    out: list[SkillFile] = []
    for s in skills:
        out.extend(load_skill(s))
    return out


def _skill_files_to_memories(files: Iterable[SkillFile]) -> list[MemoryNote]:
    return [
        MemoryNote(
            name=f.name,
            memory=f.content,
            description=f.description,
            tags=["skill", f.skill],
        )
        for f in files
    ]


async def inject_skills(
    skills: Iterable[str],
    *,
    namespace: str,
    artifact_service: Any,
    app_name: str,
    user_id: str,
) -> None:
    """Load `skills` from disk and inject them as memories under `namespace`.

    Every skill file (index and references) is tagged with both "skill" and
    the owning skill name so `skills_read(name)` can resolve any reference,
    not just the index.
    """
    skill_list = list(skills)
    if not skill_list:
        return

    files = load_skills(skill_list)
    if not files:
        return

    mem_tools = MemoryTools(name=namespace)
    await mem_tools.inject(
        memories=_skill_files_to_memories(files),
        artifact_service=artifact_service,
        app_name=app_name,
        user_id=user_id,
    )
