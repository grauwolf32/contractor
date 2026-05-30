from pathlib import Path

import pytest

from contractor.runners import models as m
from contractor.runners.models import (RenderedTask, TaskTemplate,
                                       _artifact_var_name, _normalize_name,
                                       _resolve_task_version)


class TestNormalizeName:
    def test_lowercases_and_collapses_specials(self):
        assert _normalize_name("Hello World!") == "hello_world"

    def test_strips_leading_trailing_underscores(self):
        # Inner runs of underscores are kept verbatim; only the surrounding
        # underscores are stripped. (The regex replaces non-alphanumeric runs
        # with a single `_`, but pre-existing underscore runs are not collapsed.)
        assert _normalize_name("__foo__bar__") == "foo__bar"

    def test_empty_falls_back_to_task(self):
        assert _normalize_name("") == "task"
        assert _normalize_name("///") == "task"


class TestArtifactVarName:
    def test_path_segments_joined_with_double_underscore(self):
        # `_artifact_var_name` mirrors the planner's `_safe_identifier` shape so
        # downstream task instructions can address upstream artifacts by name.
        assert (
            _artifact_var_name("plan/v1/result")
            == "artifact__plan__v1__result"
        )

    def test_strips_empty_segments(self):
        assert (
            _artifact_var_name("/plan//result/")
            == "artifact__plan__result"
        )

    def test_normalizes_special_chars_per_segment(self):
        assert (
            _artifact_var_name("Plan A/v.1/Result!")
            == "artifact__plan_a__v_1__result"
        )


# ─── TaskTemplate.load / _resolve_task_version ────────────────────────────────


def _write_task_manifest(
    tasks_dir: Path,
    *,
    name: str,
    active: str,
    versions: dict[str, str],
):
    manifest = "active: {active}\nversions:\n".format(active=active)
    for v, body in versions.items():
        manifest += f"  {v}:\n    file: {body}\n"
    (tasks_dir / f"{name}.yml").write_text(manifest, encoding="utf-8")


def _write_task_body(tasks_dir: Path, rel_path: str, body: dict):
    import yaml as _yaml

    path = tasks_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_yaml.safe_dump({"task": body}), encoding="utf-8")


@pytest.fixture()
def tasks_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "TASKS_BASE_DIR", tmp_path)
    return tmp_path


class TestResolveTaskVersion:
    def test_missing_manifest_raises(self, tasks_dir):
        with pytest.raises(ValueError, match="not found"):
            _resolve_task_version("nope", None)

    def test_active_version_used_when_unspecified(self, tasks_dir):
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v2",
            versions={"v1": "demo/v1.yml", "v2": "demo/v2.yml"},
        )
        _write_task_body(tasks_dir, "demo/v1.yml", {})
        _write_task_body(tasks_dir, "demo/v2.yml", {})

        _, resolved, body_path = _resolve_task_version("demo", None)
        assert resolved == "v2"
        assert body_path.name == "v2.yml"

    def test_explicit_version_overrides_active(self, tasks_dir):
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v2",
            versions={"v1": "demo/v1.yml", "v2": "demo/v2.yml"},
        )
        _write_task_body(tasks_dir, "demo/v1.yml", {})
        _write_task_body(tasks_dir, "demo/v2.yml", {})

        _, resolved, _ = _resolve_task_version("demo", "v1")
        assert resolved == "v1"

    def test_unknown_version_raises_with_available_list(self, tasks_dir):
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v1",
            versions={"v1": "demo/v1.yml"},
        )
        _write_task_body(tasks_dir, "demo/v1.yml", {})

        with pytest.raises(ValueError, match=r"v9.*Available versions: v1"):
            _resolve_task_version("demo", "v9")

    def test_missing_body_raises(self, tasks_dir):
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v1",
            versions={"v1": "demo/v1.yml"},
        )
        # Don't write the body file.
        with pytest.raises(ValueError, match="body for demo@v1 not found"):
            _resolve_task_version("demo", None)

    def test_missing_active_or_versions_raises(self, tasks_dir):
        (tasks_dir / "demo.yml").write_text("just: something\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must declare 'active:' and 'versions:'"):
            _resolve_task_version("demo", None)


class TestTaskTemplateLoad:
    def test_loads_with_defaults(self, tasks_dir):
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v1",
            versions={"v1": "demo/v1.yml"},
        )
        _write_task_body(
            tasks_dir,
            "demo/v1.yml",
            {
                "name": "Demo task",
                "objective": "do {project_path}",
                "instructions": "step 1",
                "output_format": "yaml",
            },
        )

        tpl = TaskTemplate.load("demo")
        assert tpl.key == "demo"
        assert tpl.version == "v1"
        assert tpl.title == "Demo task"
        assert tpl.objective == "do {project_path}"
        assert tpl.default_iterations == 1
        assert tpl.format == "json"
        assert tpl.default_artifacts == []
        assert tpl.default_skills == []

    def test_load_missing_task_key_raises(self, tasks_dir):
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v1",
            versions={"v1": "demo/v1.yml"},
        )
        (tasks_dir / "demo" / "v1.yml").parent.mkdir(parents=True, exist_ok=True)
        (tasks_dir / "demo" / "v1.yml").write_text(
            "not_a_task: oops\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="missing top-level 'task:'"):
            TaskTemplate.load("demo")


# ─── RenderedTask.from_template (brace-interpolation guards) ──────────────────


def _make_template(*, objective="", instructions="", output_format=""):
    return TaskTemplate(
        key="t",
        version="v1",
        title="T",
        objective=objective,
        instructions=instructions,
        output_format=output_format,
    )


class TestRenderedTaskFromTemplate:
    def test_variables_and_params_substitute(self):
        tpl = _make_template(
            objective="for {project_path}",
            instructions="model={model}",
            output_format="ok",
        )
        r = RenderedTask.from_template(
            tpl,
            variables={"project_path": "/p"},
            params={"model": "qwen"},
            artifacts={},
        )
        assert r.objective == "for /p"
        assert r.instructions == "model=qwen"

    def test_artifacts_surfaced_as_var(self):
        tpl = _make_template(
            instructions="prior: {artifact__plan__v1__result}",
        )
        r = RenderedTask.from_template(
            tpl,
            variables={},
            params={},
            artifacts={"plan/v1/result": "PRIOR-CONTENT"},
        )
        assert r.instructions == "prior: PRIOR-CONTENT"

    def test_artifacts_yaml_dump_available(self):
        tpl = _make_template(instructions="{artifacts}")
        r = RenderedTask.from_template(
            tpl,
            variables={},
            params={},
            artifacts={"plan/v1/result": "X"},
        )
        # YAML dump preserves order and quotes strings; just assert the key is present.
        assert "plan/v1/result" in r.instructions

    def test_params_override_variables(self):
        tpl = _make_template(instructions="{x}")
        r = RenderedTask.from_template(
            tpl,
            variables={"x": "from-var"},
            params={"x": "from-param"},
            artifacts={},
        )
        assert r.instructions == "from-param"

    def test_missing_variable_raises_key_error(self):
        # This is the load-bearing guard from CLAUDE.md: bare `{id}` in
        # template strings is interpreted as a session-state lookup by ADK and
        # crashes if unset. str.format reproduces the same trap, so any
        # template author who forgets `?` or doesn't pass the variable gets a
        # clear KeyError at render time rather than at run time.
        tpl = _make_template(instructions="hello {id}")
        with pytest.raises(KeyError, match="id"):
            RenderedTask.from_template(
                tpl, variables={}, params={}, artifacts={}
            )

    def test_unused_extra_variables_are_ignored(self):
        tpl = _make_template(instructions="static text")
        r = RenderedTask.from_template(
            tpl,
            variables={"unused": "X"},
            params={"also_unused": "Y"},
            artifacts={},
        )
        assert r.instructions == "static text"

    def test_format_task_includes_inbox_section_when_artifacts_present(self):
        tpl = _make_template()
        r = RenderedTask.from_template(
            tpl,
            variables={},
            params={},
            artifacts={"plan/result": "x"},
        )
        text = r._format_task()
        assert "INBOX:" in text
        assert "plan/result" in text

    def test_format_task_omits_inbox_when_no_artifacts(self):
        tpl = _make_template()
        r = RenderedTask.from_template(
            tpl, variables={}, params={}, artifacts={}
        )
        assert "INBOX:" not in r._format_task()


class TestSafeIdentifier:
    def test_planner_safe_identifier_matches_normalize_name_shape(self):
        # The planner's `_safe_identifier` and models.py `_normalize_name`
        # implement the same transformation. Asserting equivalence guards
        # against drift between the two sites (`_artifact_var_name` uses
        # `_normalize_name`; the planner builds agent names from
        # `_safe_identifier`, and they must agree).
        from contractor.agents.planning_agent.agent import _safe_identifier

        for value in ["plain", "Mixed Case", "weird!chars", "", "/", "__a__"]:
            assert _safe_identifier(value) == _normalize_name(value)


# ─── Checkpoint ─────────────────────────────────────────────────────────────

from contractor.runners.models import Checkpoint, CheckpointEntry


class TestCheckpoint:
    def _entry(self, ref: str = "task:0", task_id: int = 0) -> CheckpointEntry:
        return CheckpointEntry(
            task_id=task_id,
            ref=ref,
            template_key="t",
            template_version="v1",
            published_artifacts={"result": "t/result", "summary": "t/summary"},
        )

    def test_get_returns_matching_entry(self):
        cp = Checkpoint(workflow="test", entries=[self._entry("a:0")])
        assert cp.get("a:0") is not None
        assert cp.get("b:0") is None

    def test_mark_done_adds_entry(self):
        cp = Checkpoint(workflow="test")
        cp.mark_done(self._entry("a:0"))
        assert len(cp.entries) == 1
        assert cp.get("a:0") is not None

    def test_mark_done_replaces_existing(self):
        cp = Checkpoint(workflow="test", entries=[self._entry("a:0", task_id=0)])
        cp.mark_done(self._entry("a:0", task_id=5))
        assert len(cp.entries) == 1
        assert cp.get("a:0").task_id == 5

    def test_save_and_load_roundtrip(self, tmp_path):
        cp = Checkpoint(workflow="my_pipe", entries=[self._entry("a:0")])
        path = tmp_path / "checkpoint.json"
        cp.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert loaded.workflow == "my_pipe"
        assert len(loaded.entries) == 1
        assert loaded.get("a:0").published_artifacts == {
            "result": "t/result",
            "summary": "t/summary",
        }

    def test_load_returns_none_for_missing_file(self, tmp_path):
        assert Checkpoint.load(tmp_path / "nope.json") is None

    def test_load_returns_none_for_corrupt_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert Checkpoint.load(path) is None

    def test_load_returns_none_for_wrong_version(self, tmp_path):
        import json
        path = tmp_path / "old.json"
        path.write_text(json.dumps({"version": 999, "tasks": []}), encoding="utf-8")
        assert Checkpoint.load(path) is None

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        cp = Checkpoint(workflow="test", entries=[self._entry()])
        cp.save(path)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
