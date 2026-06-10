"""Regression tests for cli.main._build_event_handler UI lifecycle (bug 1).

Pre-fix, ``task_failed`` (and ``run_finished``) were in ``_UI_STOP_EVENTS``.
vuln-scan workflows catch per-finding ``task_failed`` and keep going, but once
the handler called ``ui.stop()`` it still took the ``if ui is not None`` branch
and returned, so every later event vanished (no live render, no print
fallback). The UI must now stop only on the single terminal
``workflow_finished`` event.
"""
from __future__ import annotations

import pytest

import cli.main as cli_main


class _FakeUI:
    instances: list[_FakeUI] = []

    def __init__(self, *, workflow_name: str) -> None:
        self.workflow_name = workflow_name
        self.events: list[object] = []
        self.started = False
        self.stopped = False
        _FakeUI.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def on_event(self, event: object) -> None:
        self.events.append(event)


class _FakeMetrics:
    def __init__(self, _output_dir) -> None:  # noqa: ANN001 - matches MetricsSink(output_dir)
        pass

    def matches(self, _event) -> bool:  # noqa: ANN001
        return False

    async def write(self, _event) -> None:  # noqa: ANN001
        pass


class _Ev:
    def __init__(self, type_: str) -> None:
        self.type = type_


@pytest.fixture
def patched(monkeypatch, tmp_path):
    _FakeUI.instances.clear()
    monkeypatch.setattr(cli_main, "LiveRenderer", _FakeUI)
    monkeypatch.setattr(cli_main, "MetricsSink", _FakeMetrics)
    handler = cli_main._build_event_handler(tmp_path, "oas_build", enable_ui=True)
    return handler, _FakeUI.instances[-1]


@pytest.mark.asyncio
async def test_task_failed_does_not_stop_ui(patched):
    handler, ui = patched
    await handler(_Ev("task_failed"))
    assert ui.stopped is False
    assert [getattr(e, "type", None) for e in ui.events] == ["task_failed"]


@pytest.mark.asyncio
async def test_run_finished_does_not_stop_ui(patched):
    handler, ui = patched
    await handler(_Ev("run_finished"))
    assert ui.stopped is False


@pytest.mark.asyncio
async def test_events_after_task_failed_still_render(patched):
    handler, ui = patched
    await handler(_Ev("task_failed"))
    await handler(_Ev("tool_call"))
    await handler(_Ev("task_started"))
    # Pre-fix these two would have been suppressed after stop().
    assert [getattr(e, "type", None) for e in ui.events] == [
        "task_failed",
        "tool_call",
        "task_started",
    ]


@pytest.mark.asyncio
async def test_workflow_finished_stops_ui(patched):
    handler, ui = patched
    await handler(_Ev("workflow_finished"))
    assert ui.stopped is True
    assert [getattr(e, "type", None) for e in ui.events] == ["workflow_finished"]


@pytest.mark.asyncio
async def test_skip_event_types_not_forwarded(patched):
    handler, ui = patched
    await handler(_Ev("agent_run_start"))
    assert ui.events == []
    assert ui.stopped is False


def test_workflow_finished_is_the_only_stop_event():
    assert frozenset({"workflow_finished"}) == cli_main._UI_STOP_EVENTS
