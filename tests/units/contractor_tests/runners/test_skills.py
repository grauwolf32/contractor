from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from contractor.runners import skills as m
from contractor.runners.skills import (
    SkillFile,
    _default_description,
    _memory_name,
    _parse_frontmatter,
    _skill_files_to_memories,
    inject_skills,
    load_skill,
    load_skills,
)


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("Just content.\n")
        assert meta == {}
        assert body == "Just content.\n"

    def test_well_formed_frontmatter(self):
        text = "---\nname: foo\ndescription: bar\n---\nThe body.\n"
        meta, body = _parse_frontmatter(text)
        assert meta == {"name": "foo", "description": "bar"}
        assert body == "The body.\n"

    def test_missing_closing_delimiter(self):
        text = "---\nname: foo\n(no close)\n"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_malformed_yaml_falls_back_to_raw(self):
        text = "---\n: : :::\n---\nbody\n"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_non_dict_yaml_falls_back(self):
        # YAML list at top of frontmatter is valid YAML but not a metadata dict.
        text = "---\n- a\n- b\n---\nbody\n"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter_yields_empty_dict(self):
        text = "---\n\n---\nbody\n"
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == "body\n"

    def test_strips_leading_newline_after_close(self):
        text = "---\nk: v\n---\n\n\nbody\n"
        _, body = _parse_frontmatter(text)
        # Only leading newlines are stripped by lstrip("\n"), not other whitespace.
        assert body == "body\n"


class TestMemoryName:
    def test_index_returns_skill_and_is_index_true(self):
        name, is_index = _memory_name("trace", Path("index.md"))
        assert name == "trace"
        assert is_index is True

    def test_top_level_reference(self):
        name, is_index = _memory_name("trace", Path("controls.md"))
        assert name == "trace/controls"
        assert is_index is False

    def test_nested_reference(self):
        name, is_index = _memory_name("trace", Path("references/sinks.md"))
        assert name == "trace/references/sinks"
        assert is_index is False


class TestDefaultDescription:
    def test_index_description(self):
        assert _default_description("trace", Path("index.md"), True) == "trace skill"

    def test_reference_description(self):
        assert (
            _default_description("trace", Path("references/sinks.md"), False)
            == "trace skill / references/sinks"
        )


class TestLoadSkill:
    def test_missing_skill_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            load_skill("nonexistent")

    def test_loads_index_and_references(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        skill_dir = tmp_path / "demo"
        (skill_dir / "references").mkdir(parents=True)
        (skill_dir / "index.md").write_text(
            "---\ndescription: demo overview\n---\nIndex body.\n",
            encoding="utf-8",
        )
        (skill_dir / "references" / "ref.md").write_text(
            "Just a reference, no frontmatter.\n",
            encoding="utf-8",
        )

        files = load_skill("demo")
        assert {f.name for f in files} == {"demo", "demo/references/ref"}

        index = next(f for f in files if f.is_index)
        assert index.name == "demo"
        assert index.description == "demo overview"
        assert index.content == "Index body.\n"

        ref = next(f for f in files if not f.is_index)
        assert ref.name == "demo/references/ref"
        assert ref.description == "demo skill / references/ref"
        assert ref.content == "Just a reference, no frontmatter.\n"

    def test_uses_default_description_when_frontmatter_missing_description(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        skill_dir = tmp_path / "demo"
        skill_dir.mkdir()
        (skill_dir / "index.md").write_text(
            "---\nname: demo\n---\nbody\n", encoding="utf-8"
        )
        [index] = load_skill("demo")
        assert index.description == "demo skill"

    def test_ignores_non_markdown_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        skill_dir = tmp_path / "demo"
        skill_dir.mkdir()
        (skill_dir / "index.md").write_text("body\n", encoding="utf-8")
        (skill_dir / "notes.txt").write_text("ignored\n", encoding="utf-8")
        files = load_skill("demo")
        assert len(files) == 1
        assert files[0].name == "demo"

    def test_empty_skill_dir_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        (tmp_path / "demo").mkdir()
        assert load_skill("demo") == []


class TestLoadSkills:
    def test_aggregates_multiple(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            (d / "index.md").write_text("body\n", encoding="utf-8")

        files = load_skills(["a", "b"])
        assert {f.skill for f in files} == {"a", "b"}
        assert len(files) == 2


class TestSkillFilesToMemories:
    def test_tags_include_skill_and_marker(self):
        f = SkillFile(
            skill="trace",
            name="trace/controls",
            description="d",
            content="c",
            is_index=False,
        )
        [mem] = _skill_files_to_memories([f])
        assert mem.name == "trace/controls"
        assert mem.memory == "c"
        assert mem.description == "d"
        assert set(mem.tags) == {"skill", "trace"}


class TestInjectSkills:
    @pytest.mark.asyncio
    async def test_empty_skills_short_circuits(self, monkeypatch):
        inject_mock = AsyncMock()
        monkeypatch.setattr(
            "contractor.tools.memory.MemoryTools.inject", inject_mock
        )
        await inject_skills(
            [],
            namespace="ns",
            artifact_service=object(),
            app_name="app",
            user_id="u",
        )
        inject_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_files_short_circuits(self, tmp_path, monkeypatch):
        # Skill directory exists but has no markdown — load_skills returns [].
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        (tmp_path / "demo").mkdir()
        inject_mock = AsyncMock()
        monkeypatch.setattr(
            "contractor.tools.memory.MemoryTools.inject", inject_mock
        )
        await inject_skills(
            ["demo"],
            namespace="ns",
            artifact_service=object(),
            app_name="app",
            user_id="u",
        )
        inject_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_injects_loaded_skills(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "SKILLS_BASE_DIR", tmp_path)
        skill_dir = tmp_path / "demo"
        skill_dir.mkdir()
        (skill_dir / "index.md").write_text("body\n", encoding="utf-8")

        inject_mock = AsyncMock()
        monkeypatch.setattr(
            "contractor.tools.memory.MemoryTools.inject", inject_mock
        )
        artifact_service = object()
        await inject_skills(
            ["demo"],
            namespace="ns",
            artifact_service=artifact_service,
            app_name="app",
            user_id="u",
        )
        inject_mock.assert_awaited_once()
        kwargs = inject_mock.await_args.kwargs
        memories = kwargs["memories"]
        assert len(memories) == 1
        assert memories[0].name == "demo"
        assert kwargs["app_name"] == "app"
        assert kwargs["user_id"] == "u"
        assert kwargs["artifact_service"] is artifact_service
