from pathlib import Path

import pytest

from contractor.runners import models as m
from contractor.runners.models import (
    RenderedTask,
    TaskTemplate,
    _artifact_var_name,
    _normalize_name,
    _resolve_task_version,
)


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
    manifest = f"active: {active}\nversions:\n"
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

    @pytest.mark.parametrize(
        "missing_field", ["objective", "instructions", "output_format"],
    )
    def test_load_missing_required_field_raises_value_error(
        self, tasks_dir, missing_field,
    ):
        # A body without objective/instructions/output_format used to raise a
        # bare KeyError; it must follow the same descriptive ValueError
        # pattern (naming the body path) as the neighboring validation.
        _write_task_manifest(
            tasks_dir,
            name="demo",
            active="v1",
            versions={"v1": "demo/v1.yml"},
        )
        body = {
            "objective": "o",
            "instructions": "i",
            "output_format": "yaml",
        }
        del body[missing_field]
        _write_task_body(tasks_dir, "demo/v1.yml", body)

        with pytest.raises(
            ValueError, match=f"missing required '{missing_field}:'",
        ) as exc_info:
            TaskTemplate.load("demo")
        # The body path is part of the message, like the neighboring errors.
        assert "v1.yml" in str(exc_info.value)


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

    def test_colliding_artifact_refs_raise_naming_both(self):
        # "oas-build/result" and "oas_build/result" both normalize to
        # "artifact__oas_build__result" — the later artifact used to silently
        # win. Now the ambiguity is rejected, naming both refs.
        tpl = _make_template(instructions="{artifact__oas_build__result}")
        with pytest.raises(ValueError, match="normalize to") as exc_info:
            RenderedTask.from_template(
                tpl,
                variables={},
                params={},
                artifacts={
                    "oas-build/result": "A",
                    "oas_build/result": "B",
                },
            )
        message = str(exc_info.value)
        assert "oas-build/result" in message
        assert "oas_build/result" in message
        assert "artifact__oas_build__result" in message

    def test_distinct_artifact_refs_do_not_collide(self):
        tpl = _make_template(
            instructions="{artifact__a__result} {artifact__b__result}",
        )
        r = RenderedTask.from_template(
            tpl,
            variables={},
            params={},
            artifacts={"a/result": "A", "b/result": "B"},
        )
        assert r.instructions == "A B"

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

    def test_stale_snapshots_merge_instead_of_losing_parallel_entries(
        self, tmp_path
    ):
        path = tmp_path / "checkpoint.json"
        first_snapshot = Checkpoint(
            workflow="parallel",
            entries=[self._entry("class:a", task_id=0)],
        )
        second_snapshot = Checkpoint(
            workflow="parallel",
            entries=[self._entry("class:b", task_id=1)],
        )

        first_snapshot.save(path)
        second_snapshot.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert {entry.ref for entry in loaded.entries} == {"class:a", "class:b"}

    def test_stale_snapshot_does_not_overwrite_newer_existing_entry(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        initial = Checkpoint(
            workflow="parallel",
            entries=[self._entry("shared", task_id=0)],
        )
        initial.save(path)

        first_snapshot = Checkpoint.load(path)
        stale_snapshot = Checkpoint.load(path)
        assert first_snapshot is not None
        assert stale_snapshot is not None

        first_snapshot.mark_done(self._entry("shared", task_id=10))
        first_snapshot.save(path)
        stale_snapshot.mark_done(self._entry("sibling", task_id=20))
        stale_snapshot.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert {entry.ref: entry.task_id for entry in loaded.entries} == {
            "shared": 10,
            "sibling": 20,
        }

    def test_directly_appended_entry_is_merged(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        initial = Checkpoint(
            workflow="parallel",
            entries=[self._entry("existing", task_id=0)],
        )
        initial.save(path)

        snapshot = Checkpoint.load(path)
        assert snapshot is not None
        snapshot.entries.append(self._entry("appended", task_id=1))
        snapshot.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert {entry.ref for entry in loaded.entries} == {"existing", "appended"}

    def test_directly_mutated_entry_is_merged_without_replaying_stale_siblings(
        self, tmp_path
    ):
        path = tmp_path / "checkpoint.json"
        initial = Checkpoint(
            workflow="parallel",
            entries=[
                self._entry("shared", task_id=0),
                self._entry("updated-by-sibling", task_id=1),
            ],
        )
        initial.save(path)

        snapshot = Checkpoint.load(path)
        sibling = Checkpoint.load(path)
        assert snapshot is not None
        assert sibling is not None
        sibling.mark_done(self._entry("updated-by-sibling", task_id=20))
        sibling.save(path)

        shared = snapshot.get("shared")
        assert shared is not None
        shared.published_artifacts["records"] = "t/records"
        snapshot.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert loaded.get("shared").published_artifacts["records"] == "t/records"
        assert loaded.get("updated-by-sibling").task_id == 20

    def test_directly_removed_entry_is_deleted_without_losing_sibling_addition(
        self, tmp_path
    ):
        path = tmp_path / "checkpoint.json"
        Checkpoint(
            workflow="parallel",
            entries=[self._entry("remove", task_id=0)],
        ).save(path)

        snapshot = Checkpoint.load(path)
        sibling = Checkpoint.load(path)
        assert snapshot is not None
        assert sibling is not None
        sibling.mark_done(self._entry("keep", task_id=1))
        sibling.save(path)

        snapshot.entries.clear()
        snapshot.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert [entry.ref for entry in loaded.entries] == ["keep"]

    def test_failed_merge_save_does_not_make_imported_siblings_dirty_on_retry(
        self, tmp_path, monkeypatch
    ):
        path = tmp_path / "checkpoint.json"
        Checkpoint(
            workflow="parallel",
            entries=[self._entry("local", task_id=0)],
        ).save(path)

        stale = Checkpoint.load(path)
        sibling = Checkpoint.load(path)
        assert stale is not None
        assert sibling is not None
        sibling.mark_done(self._entry("sibling", task_id=1))
        sibling.save(path)
        stale.mark_done(self._entry("local", task_id=2))

        real_replace = Path.replace
        failed = False

        def fail_first_replace(source: Path, target: Path) -> Path:
            nonlocal failed
            if not failed and target == path:
                failed = True
                raise OSError("simulated replace failure")
            return real_replace(source, target)

        monkeypatch.setattr(Path, "replace", fail_first_replace)
        with pytest.raises(OSError, match="simulated replace failure"):
            stale.save(path)
        assert [entry.ref for entry in stale.entries] == ["local"]

        newer = Checkpoint.load(path)
        assert newer is not None
        newer.mark_done(self._entry("sibling", task_id=3))
        newer.save(path)
        stale.save(path)

        loaded = Checkpoint.load(path)
        assert loaded is not None
        assert {entry.ref: entry.task_id for entry in loaded.entries} == {
            "local": 2,
            "sibling": 3,
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

    def test_load_returns_none_for_entry_missing_required_field(
        self, tmp_path, caplog,
    ):
        import json
        import logging
        path = tmp_path / "partial.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "workflow": "test",
                # Valid JSON, wrong shape: entry lacks ref/template_key/….
                "tasks": [{"task_id": 0}],
            }),
            encoding="utf-8",
        )
        with caplog.at_level(
            logging.WARNING, logger="contractor.runners.models.checkpoint",
        ):
            assert Checkpoint.load(path) is None
        assert any(
            "ignoring corrupt checkpoint" in r.getMessage() for r in caplog.records
        )

    def test_load_returns_none_for_non_dict_entries(self, tmp_path):
        import json
        path = tmp_path / "shape.json"
        path.write_text(
            json.dumps({"version": 1, "workflow": "test", "tasks": ["oops"]}),
            encoding="utf-8",
        )
        assert Checkpoint.load(path) is None

    def test_load_returns_none_for_duplicate_refs(self, tmp_path):
        import json

        path = tmp_path / "duplicate.json"
        task = {
            "task_id": 0,
            "ref": "same",
            "template_key": "t",
            "template_version": "v1",
            "published_artifacts": {},
        }
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "workflow": "test",
                    "tasks": [task, {**task, "task_id": 1}],
                }
            ),
            encoding="utf-8",
        )

        assert Checkpoint.load(path) is None

    def test_save_is_atomic(self, tmp_path):
        path = tmp_path / "checkpoint.json"
        cp = Checkpoint(workflow="test", entries=[self._entry()])
        cp.save(path)
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()


class TestDefaultParams:
    """Param defaults make a {placeholder} optional: omitted -> default,
    supplied -> override; param-less templates are unaffected."""

    def _template(self) -> TaskTemplate:
        return TaskTemplate(
            key="t",
            version="v1",
            title="T",
            objective="Do work on {project_path}.",
            instructions="Focus: {focus}. Mode: {mode}.",
            output_format="report",
            default_params={"focus": "GENERIC_DEFAULT", "mode": "fast"},
        )

    def test_omitted_param_uses_default(self):
        r = RenderedTask.from_template(
            self._template(),
            variables={"project_path": "."},
            params={},
            artifacts={},
        )
        assert "GENERIC_DEFAULT" in r.instructions
        assert "Mode: fast" in r.instructions
        assert "{focus}" not in r.instructions  # no KeyError, fully rendered

    def test_param_overrides_default(self):
        r = RenderedTask.from_template(
            self._template(),
            variables={"project_path": "."},
            params={"focus": "INJECTED"},
            artifacts={},
        )
        assert "Focus: INJECTED" in r.instructions
        assert "Mode: fast" in r.instructions  # untouched default still applies

    def test_variable_beats_default_param_below_params(self):
        # Precedence: default_params < variables < params.
        tmpl = TaskTemplate(
            key="t", version="v1", title="T",
            objective="{x}", instructions="{x}", output_format="o",
            default_params={"x": "from_default"},
        )
        r = RenderedTask.from_template(
            tmpl, variables={"x": "from_var"}, params={}, artifacts={}
        )
        assert r.objective == "from_var"
        r2 = RenderedTask.from_template(
            tmpl, variables={"x": "from_var"}, params={"x": "from_param"}, artifacts={}
        )
        assert r2.objective == "from_param"

    def test_real_knowledge_tasks_declare_focus_default(self):
        for name in ("knowledge_discovery", "knowledge_consolidation"):
            t = TaskTemplate.load(name)
            assert "focus" in t.default_params
            r = RenderedTask.from_template(
                t, variables={"project_path": "."}, params={}, artifacts={}
            )
            assert "{focus}" not in (r.objective + r.instructions)
