"""Live, read-only loaders for the explorer UI.

Walks the four on-disk surfaces the UI renders — agent prompts, task
templates, skills, and workflow configs — straight from the package tree
on every call, so edits show up on refresh without a restart. Nothing here
imports the agent runtime; it only parses YAML/Markdown, keeping the server
dependency-light and side-effect free.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# analytics_ui/reader.py -> parents[1] == repo root; the data lives in the
# sibling ``contractor`` package tree.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONTRACTOR = _REPO_ROOT / "contractor"
AGENTS_DIR = _CONTRACTOR / "agents"
TASKS_DIR = _CONTRACTOR / "tasks"
SKILLS_DIR = _CONTRACTOR / "skills"
WORKFLOWS_DIR = _CONTRACTOR / "workflows"


# ───────────────────────── helpers ─────────────────────────


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _safe_yaml(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def _first_meaningful_line(markdown: str, limit: int = 200) -> str:
    """A one-line gloss for nav cards: first heading or first prose line."""
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line or line.startswith(("```", "---", "<!--")):
            continue
        line = line.lstrip("#").strip()
        if line:
            return line[:limit]
    return ""


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (frontmatter_dict, body) for a `---`-fenced markdown file."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_raw = text[3:end].strip()
    body = text[end + 4 :].lstrip("\n")
    data = _safe_yaml(fm_raw)
    return (data if isinstance(data, dict) else {}), body


def _version_sort_key(version: str) -> tuple[int, Any]:
    """Sort `v7` above `v6` … `v0`; non-numeric versions fall back to name."""
    if version.startswith("v") and version[1:].isdigit():
        return (0, -int(version[1:]))
    return (1, version)


# ───────────────────────── agents ─────────────────────────


@dataclass
class AgentVersion:
    id: str
    file: str


@dataclass
class AgentSummary:
    name: str
    active: str
    versions: list[str]
    summary: str


def list_agents() -> list[AgentSummary]:
    if not AGENTS_DIR.is_dir():
        return []
    out: list[AgentSummary] = []
    for d in sorted(AGENTS_DIR.iterdir()):
        manifest = d / "prompt.yml"
        if not (d.is_dir() and manifest.is_file()):
            continue
        data = _safe_yaml(_read_text(manifest)) or {}
        versions = sorted((data.get("versions") or {}).keys(), key=_version_sort_key)
        active = data.get("active") or (versions[0] if versions else "")
        summary = ""
        if active:
            summary = _first_meaningful_line(_agent_version_text(d.name, active))
        out.append(
            AgentSummary(name=d.name, active=active, versions=versions, summary=summary)
        )
    return out


def _agent_manifest(name: str) -> dict[str, Any] | None:
    manifest = AGENTS_DIR / name / "prompt.yml"
    if not manifest.is_file():
        return None
    data = _safe_yaml(_read_text(manifest))
    return data if isinstance(data, dict) else {}


def _agent_version_text(name: str, version: str) -> str:
    data = _agent_manifest(name) or {}
    spec = (data.get("versions") or {}).get(version)
    if not isinstance(spec, dict):
        return ""
    rel = spec.get("file")
    if not rel:
        return ""
    return _read_text(AGENTS_DIR / name / rel)


def get_agent(name: str) -> dict[str, Any] | None:
    data = _agent_manifest(name)
    if data is None:
        return None
    versions = sorted((data.get("versions") or {}).keys(), key=_version_sort_key)
    active = data.get("active") or (versions[0] if versions else "")
    return {
        "name": name,
        "active": active,
        "versions": versions,
        "summary": _first_meaningful_line(_agent_version_text(name, active))
        if active
        else "",
    }


def get_agent_version(name: str, version: str) -> dict[str, Any] | None:
    data = _agent_manifest(name)
    if data is None:
        return None
    if version not in (data.get("versions") or {}):
        return None
    return {"name": name, "version": version, "content": _agent_version_text(name, version)}


# ───────────────────────── tasks ─────────────────────────

_TASK_FIELDS = (
    "name",
    "objective",
    "instructions",
    "output_format",
    "context",
    "artifacts",
    "skills",
    "iterations",
    "format",
    "max_steps",
)


@dataclass
class TaskSummary:
    name: str
    active: str
    versions: list[str]
    summary: str
    skills: list[str]


def _task_manifest(name: str) -> dict[str, Any] | None:
    manifest = TASKS_DIR / f"{name}.yml"
    if not manifest.is_file():
        return None
    data = _safe_yaml(_read_text(manifest))
    return data if isinstance(data, dict) else {}


def _task_version_body(name: str, version: str) -> dict[str, Any]:
    data = _task_manifest(name) or {}
    spec = (data.get("versions") or {}).get(version)
    if not isinstance(spec, dict) or not spec.get("file"):
        return {}
    raw = _safe_yaml(_read_text(TASKS_DIR / spec["file"]))
    if not isinstance(raw, dict):
        return {}
    body = raw.get("task")
    return body if isinstance(body, dict) else {}


def list_tasks() -> list[TaskSummary]:
    if not TASKS_DIR.is_dir():
        return []
    out: list[TaskSummary] = []
    for manifest in sorted(TASKS_DIR.glob("*.yml")):
        name = manifest.stem
        data = _safe_yaml(_read_text(manifest)) or {}
        versions = sorted((data.get("versions") or {}).keys(), key=_version_sort_key)
        active = data.get("active") or (versions[0] if versions else "")
        body = _task_version_body(name, active) if active else {}
        skills = body.get("skills") or []
        out.append(
            TaskSummary(
                name=name,
                active=active,
                versions=versions,
                summary=_first_meaningful_line(str(body.get("objective") or "")),
                skills=[str(s) for s in skills] if isinstance(skills, list) else [],
            )
        )
    return out


def get_task(name: str, version: str | None = None) -> dict[str, Any] | None:
    data = _task_manifest(name)
    if data is None:
        return None
    versions = sorted((data.get("versions") or {}).keys(), key=_version_sort_key)
    active = data.get("active") or (versions[0] if versions else "")
    use = version or active
    if use not in (data.get("versions") or {}):
        return None
    body = _task_version_body(name, use)
    fields = {k: body.get(k) for k in _TASK_FIELDS if k in body}
    # Anything the template carries beyond the known set, surfaced verbatim.
    extra = {k: v for k, v in body.items() if k not in _TASK_FIELDS}
    skills = body.get("skills") or []
    spec = (data.get("versions") or {}).get(use) or {}
    raw = _read_text(TASKS_DIR / spec["file"]) if spec.get("file") else ""
    return {
        "name": name,
        "active": active,
        "version": use,
        "versions": versions,
        "fields": fields,
        "extra": extra,
        "skills": [str(s) for s in skills] if isinstance(skills, list) else [],
        "raw": raw,
    }


# ───────────────────────── skills ─────────────────────────


@dataclass
class SkillSummary:
    name: str
    description: str
    references: list[str]


def list_skills() -> list[SkillSummary]:
    if not SKILLS_DIR.is_dir():
        return []
    out: list[SkillSummary] = []
    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        index = d / "index.md"
        fm, body = _split_frontmatter(_read_text(index)) if index.is_file() else ({}, "")
        desc = str(fm.get("description") or _first_meaningful_line(body))
        refs = []
        ref_dir = d / "references"
        if ref_dir.is_dir():
            refs = sorted(p.stem for p in ref_dir.glob("*.md"))
        out.append(SkillSummary(name=d.name, description=desc, references=refs))
    return out


def get_skill(name: str) -> dict[str, Any] | None:
    d = SKILLS_DIR / name
    if not d.is_dir():
        return None
    index = d / "index.md"
    full = _read_text(index) if index.is_file() else ""
    fm, body = _split_frontmatter(full)
    refs = []
    ref_dir = d / "references"
    if ref_dir.is_dir():
        refs = sorted(p.stem for p in ref_dir.glob("*.md"))
    return {
        "name": name,
        "description": str(fm.get("description") or _first_meaningful_line(body)),
        "frontmatter": fm,
        "content": body,
        "raw": full,
        "references": refs,
    }


def get_skill_reference(name: str, ref: str) -> dict[str, Any] | None:
    # ref is a bare stem; reject path-escape attempts.
    if "/" in ref or "\\" in ref or ".." in ref:
        return None
    path = SKILLS_DIR / name / "references" / f"{ref}.md"
    if not path.is_file():
        return None
    return {"name": name, "ref": ref, "content": _read_text(path)}


# ───────────────────────── workflow config ─────────────────────────


def workflow_config(module_file: str) -> dict[str, Any]:
    """Load the sibling config.yaml of a workflow module (best effort)."""
    cfg = Path(module_file).parent / "config.yaml"
    if not cfg.is_file():
        return {}
    data = _safe_yaml(_read_text(cfg))
    return data if isinstance(data, dict) else {}
