"""Tests for scripts/visualize_subtasks.py.

The script lives outside the Python package, so we load it via importlib
to avoid forcing ``scripts/`` into the source tree.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "visualize_subtasks.py"
)


@pytest.fixture(scope="module")
def vs():
    spec = importlib.util.spec_from_file_location("visualize_subtasks", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves type annotations via sys.modules[cls.__module__];
    # register the module before exec so dataclass introspection succeeds.
    sys.modules["visualize_subtasks"] = module
    spec.loader.exec_module(module)
    return module


def test_extract_runs_prefers_task_finished_over_iteration(vs):
    events = [
        {
            "type": "iteration_finished",
            "task_name": "build",
            "task_id": 0,
            "result": {"records": [{"task_id": "1", "title": "old", "status": "new"}]},
        },
        {
            "type": "task_finished",
            "task_name": "build",
            "task_id": 0,
            "records": [{"task_id": "1", "title": "final", "status": "done"}],
        },
    ]
    runs = vs.extract_runs(events)
    assert len(runs) == 1
    assert runs[0].event_type == "task_finished"
    assert runs[0].records[0]["title"] == "final"


def test_extract_runs_unwraps_task_failed_last_result(vs):
    events = [
        {
            "type": "task_failed",
            "task_name": "trace",
            "task_id": 2,
            "last_result": {
                "records": [{"task_id": "1", "title": "boom", "status": "incomplete"}],
            },
        },
    ]
    runs = vs.extract_runs(events)
    assert len(runs) == 1
    assert runs[0].records[0]["status"] == "incomplete"


def test_extract_runs_skips_events_without_records(vs):
    events = [
        {"type": "task_started", "task_name": "x", "task_id": 0},
        {"type": "task_finished", "task_name": "x", "task_id": 0, "records": []},
    ]
    assert vs.extract_runs(events) == []


def test_build_tree_dotted_ids_form_parent_child(vs):
    records = [
        {"task_id": "1", "title": "root", "status": "decomposed"},
        {"task_id": "1.1", "title": "child a", "status": "done"},
        {"task_id": "1.2", "title": "child b", "status": "incomplete"},
        {"task_id": "1.2.1", "title": "grandchild", "status": "skipped"},
        {"task_id": "2", "title": "second root", "status": "done"},
    ]
    nodes, roots = vs.build_tree(records)

    assert roots == ["1", "2"]
    assert nodes["1"].children == ["1.1", "1.2"]
    assert nodes["1.2"].children == ["1.2.1"]
    assert nodes["1.2.1"].children == []
    assert nodes["1.1"].status == "done"
    assert nodes["1.2"].status == "incomplete"


def test_build_tree_synthesizes_missing_ancestors(vs):
    records = [{"task_id": "3.2.7", "title": "orphan", "status": "done"}]
    nodes, roots = vs.build_tree(records)

    assert roots == ["3"]
    assert nodes["3"].status == "unknown"
    assert nodes["3.2"].status == "unknown"
    assert nodes["3"].children == ["3.2"]
    assert nodes["3.2"].children == ["3.2.7"]


def test_build_tree_sorts_children_numerically(vs):
    records = [
        {"task_id": "1.10", "title": "ten", "status": "done"},
        {"task_id": "1.2", "title": "two", "status": "done"},
        {"task_id": "1.1", "title": "one", "status": "done"},
        {"task_id": "1", "title": "root", "status": "decomposed"},
    ]
    nodes, _ = vs.build_tree(records)
    assert nodes["1"].children == ["1.1", "1.2", "1.10"]


def test_render_tree_writes_png(vs, tmp_path: Path):
    records = [
        {"task_id": "1", "title": "do thing", "status": "done"},
        {"task_id": "2", "title": "fail thing", "status": "incomplete"},
    ]
    nodes, roots = vs.build_tree(records)
    out = tmp_path / "graph.png"
    vs.render_tree(nodes, roots, "test", out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_dot_writes_expected_content(vs, tmp_path: Path):
    records = [
        {"task_id": "1", "title": "root", "status": "decomposed"},
        {"task_id": "1.1", "title": "child", "status": "done"},
    ]
    nodes, _ = vs.build_tree(records)
    out = tmp_path / "graph.dot"
    vs.render_dot(nodes, "test", out)
    text = out.read_text(encoding="utf-8")
    assert 'digraph "test"' in text
    assert '"1" -> "1.1";' in text
    assert vs.STATUS_COLOURS["done"] in text


def test_main_end_to_end(vs, tmp_path: Path, monkeypatch, capsys):
    metrics = tmp_path / "metrics.jsonl"
    events = [
        {
            "type": "task_finished",
            "task_name": "build",
            "task_id": 0,
            "template_key": "build",
            "records": [
                {"task_id": "1", "title": "root", "status": "decomposed"},
                {"task_id": "1.1", "title": "ok", "status": "done"},
                {"task_id": "1.2", "title": "bad", "status": "incomplete"},
            ],
        },
    ]
    with metrics.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")

    monkeypatch.setattr("sys.argv", ["visualize_subtasks", str(metrics)])
    vs.main()

    out_dir = metrics.parent / "subtask_graphs"
    pngs = list(out_dir.glob("*.png"))
    dots = list(out_dir.glob("*.dot"))
    assert len(pngs) == 1
    assert len(dots) == 1
    assert "Done." in capsys.readouterr().out
