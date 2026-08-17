"""Security regressions for the local analytics explorer."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from analytics_ui import reader, server
from analytics_ui.server import _route_api


def test_decoded_api_identifier_cannot_escape_skills_root(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    safe = skills / "safe-skill"
    safe.mkdir(parents=True)
    (safe / "index.md").write_text("# Safe skill", encoding="utf-8")

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "index.md").write_text("# Private document", encoding="utf-8")
    monkeypatch.setattr(reader, "SKILLS_DIR", skills)

    assert _route_api(["skills", "safe-skill"])["raw"] == "# Safe skill"

    # _handle_api decodes each segment after splitting, so an encoded absolute
    # path arrives at _route_api as one value beginning with '/'.
    encoded = str(outside).replace("/", "%2F")
    parts = [unquote(part) for part in f"skills/{encoded}".split("/") if part]
    assert parts == ["skills", str(outside)]
    assert _route_api(parts) is None
    assert _route_api(["skills", ".."]) is None

    # Keep the reader safe even when called directly, without the HTTP router.
    assert reader.get_skill(str(outside)) is None


def test_manifest_referenced_files_remain_inside_content_root(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    agent = agents / "safe_agent"
    agent.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    (agent / "prompt.yml").write_text(
        "active: v1\nversions:\n  v1:\n    file: ../../outside.md\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(reader, "AGENTS_DIR", agents)

    result = reader.get_agent_version("safe_agent", "v1")
    assert result is not None
    assert result["content"] == ""


def test_list_readers_ignore_symlink_escapes(tmp_path, monkeypatch):
    agents = tmp_path / "agents"
    tasks = tmp_path / "tasks"
    skills = tmp_path / "skills"
    agents.mkdir()
    tasks.mkdir()
    skills.mkdir()

    outside_agent = tmp_path / "outside-agent"
    outside_agent.mkdir()
    (outside_agent / "prompt.yml").write_text(
        "active: v1\nversions:\n  v1:\n    file: prompt.md\n",
        encoding="utf-8",
    )
    (agents / "escaped-agent").symlink_to(outside_agent, target_is_directory=True)

    outside_task = tmp_path / "outside-task.yml"
    outside_task.write_text("active: v1\nversions: {v1: {}}\n", encoding="utf-8")
    (tasks / "escaped-task.yml").symlink_to(outside_task)

    outside_skill = tmp_path / "outside-skill"
    outside_skill.mkdir()
    (outside_skill / "index.md").write_text("private skill", encoding="utf-8")
    (skills / "escaped-skill").symlink_to(outside_skill, target_is_directory=True)

    monkeypatch.setattr(reader, "AGENTS_DIR", agents)
    monkeypatch.setattr(reader, "TASKS_DIR", tasks)
    monkeypatch.setattr(reader, "SKILLS_DIR", skills)

    assert reader.list_agents() == []
    assert reader.list_tasks() == []
    assert reader.list_skills() == []


def test_skill_children_cannot_follow_symlinks_outside_root(tmp_path, monkeypatch):
    skills = tmp_path / "skills"
    skill = skills / "safe-skill"
    skill.mkdir(parents=True)

    outside_index = tmp_path / "private.md"
    outside_index.write_text("private skill content", encoding="utf-8")
    (skill / "index.md").symlink_to(outside_index)

    outside_refs = tmp_path / "private-references"
    outside_refs.mkdir()
    (outside_refs / "secret.md").write_text("private reference", encoding="utf-8")
    (skill / "references").symlink_to(outside_refs, target_is_directory=True)
    monkeypatch.setattr(reader, "SKILLS_DIR", skills)

    result = reader.get_skill("safe-skill")
    assert result is not None
    assert result["raw"] == ""
    assert result["references"] == []
    assert reader.list_skills()[0].description == ""
    assert reader.list_skills()[0].references == []


def test_static_path_rejects_sibling_prefix_escape(tmp_path, monkeypatch):
    static = tmp_path / "static"
    static.mkdir()
    index = static / "index.html"
    index.write_text("safe shell", encoding="utf-8")
    sibling = tmp_path / "static_evil"
    sibling.mkdir()
    secret = sibling / "secret.txt"
    secret.write_text("private", encoding="utf-8")
    monkeypatch.setattr(server, "STATIC_DIR", static)

    assert server._resolve_static_path("/app.js") == index.resolve()
    assert server._resolve_static_path("/../static_evil/secret.txt") == index.resolve()
    assert server._resolve_static_path("/../static_evil/secret.txt") != secret.resolve()


def test_eval_verdict_renderer_uses_text_nodes_and_filters_link_schemes():
    app_js = (
        Path(__file__).resolve().parents[3] / "analytics_ui" / "static" / "app.js"
    ).read_text(encoding="utf-8")

    case_row = app_js.split("function caseRow", 1)[1].split(
        "function toolUsageCard", 1
    )[0]
    assert "html: `${c.expected_verdict}" not in case_row
    assert "text: c.actual_verdict" in case_row
    assert "safeLinkHref(href)" in app_js
    assert "['http:', 'https:', 'mailto:'].includes(protocol)" in app_js
