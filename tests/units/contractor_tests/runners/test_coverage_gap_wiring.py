"""Wiring tests: ``track_coverage_gap`` -> worker fs tools ``capture_in_scope``.

These assert the QW8 "go live" wiring: when the resolved ``ObservationConfig``
enables ``track_coverage_gap``, the worker's fs tools are built with the
in-scope walk on (``capture_in_scope=True``); when it is off (the default),
nothing is passed and the walk stays off — byte-identical to pre-feature.
"""

from __future__ import annotations

import inspect
from functools import partial
from pathlib import Path

import pytest

from cli.fs import RootedLocalFileSystem
from contractor.agents.codereview_agent.agent import build_codereview_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.runners.task_runner import TaskRunner
from contractor.tools.observations import FILE_PATHS_STATE_KEY, ObservationConfig
from tests.units.contractor_tests.helpers import mk_tool_context


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.txt").write_text("hello")
    (root / "dir").mkdir()
    (root / "dir" / "inner.txt").write_text("inner")
    return root


def _in_scope_after_read(agent) -> list[str]:
    """Invoke ``read_file`` on a built agent's fs tool and return ``in_scope``.

    ``in_scope`` is non-empty only when the tools were built with
    ``capture_in_scope=True`` (the walk ran); otherwise it is ``[]``.
    """
    read_file = next(t for t in agent.tools if getattr(t, "__name__", "") == "read_file")
    ctx = mk_tool_context(invocation_id="inv-1")
    read_file("/file.txt", tool_context=ctx)
    return ctx.state[FILE_PATHS_STATE_KEY]["in_scope"]


# ── factory-level: the flag threads into the fs tools ──────────────────────────


def test_codereview_capture_in_scope_enabled_walks_tree(project_root: Path):
    fs = RootedLocalFileSystem(project_root)
    agent = build_codereview_agent(
        "codereview_agent", fs, namespace="ns", capture_in_scope=True
    )
    assert set(_in_scope_after_read(agent)) == {"/file.txt", "/dir/inner.txt"}


def test_codereview_capture_in_scope_disabled_no_walk(project_root: Path):
    fs = RootedLocalFileSystem(project_root)
    agent = build_codereview_agent("codereview_agent", fs, namespace="ns")
    assert _in_scope_after_read(agent) == []


def test_swe_capture_in_scope_enabled_walks_tree(project_root: Path):
    fs = RootedLocalFileSystem(project_root)
    agent = build_swe_agent("swe_agent", fs, namespace="ns", capture_in_scope=True)
    assert set(_in_scope_after_read(agent)) == {"/file.txt", "/dir/inner.txt"}


def test_swe_capture_in_scope_disabled_no_walk(project_root: Path):
    fs = RootedLocalFileSystem(project_root)
    agent = build_swe_agent("swe_agent", fs, namespace="ns")
    assert _in_scope_after_read(agent) == []


# ── runner-level: the single wiring point picks the right kwargs ───────────────


def test_coverage_gap_kwargs_off_by_default():
    builder = partial(build_codereview_agent, name="codereview_agent", fs=None)
    # Feature off -> nothing passed -> byte-identical default call.
    assert TaskRunner._coverage_gap_kwargs(builder, ObservationConfig()) == {}
    assert (
        TaskRunner._coverage_gap_kwargs(
            builder, ObservationConfig(enabled=True, track_coverage_gap=False)
        )
        == {}
    )


def test_coverage_gap_kwargs_on_when_enabled_and_supported():
    builder = partial(build_codereview_agent, name="codereview_agent", fs=None)
    assert TaskRunner._coverage_gap_kwargs(
        builder, ObservationConfig(enabled=True, track_coverage_gap=True)
    ) == {"capture_in_scope": True}


def test_coverage_gap_kwargs_skipped_for_unsupported_builder():
    # A builder without ``capture_in_scope`` (no fs tools) is never handed the
    # kwarg, even when the feature is on — so it can't crash on an unknown arg.
    builder = partial(build_oas_linter_agent, name="oas_linter_agent", fs=None)
    assert "capture_in_scope" not in inspect.signature(builder).parameters
    assert (
        TaskRunner._coverage_gap_kwargs(
            builder, ObservationConfig(enabled=True, track_coverage_gap=True)
        )
        == {}
    )
