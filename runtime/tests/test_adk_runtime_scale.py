"""Real custom-Gateway long-session work and immutable State acceptance."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fakes.adk_scale import run_workload

import contractor_runtime.telemetry.metrics as metrics_module
from contractor_runtime.contracts import ToolCallRecord

WORKLOAD = json.loads((Path(__file__).parent / "testdata/adk213-scale-workloads.json").read_text())


@pytest.mark.parametrize("turns", WORKLOAD["main_turns"])
@pytest.mark.parametrize("mode", WORKLOAD["session_modes"])
def test_long_real_gateway_session_preserves_results_and_bounds_work(
    tmp_path, monkeypatch, turns, mode
):
    projections = 0
    original_projection = metrics_module._record_projection

    def project(record):
        nonlocal projections
        projections += isinstance(record, ToolCallRecord)
        return original_projection(record)

    monkeypatch.setattr(metrics_module, "_record_projection", project)
    result = asyncio.run(
        run_workload(
            tmp_path,
            turns=turns,
            width=WORKLOAD["calls_per_tool_turn"],
            mode=mode,
            subtasks=WORKLOAD["sequential_subtasks"],
        )
    )
    expected_tools = (turns - 1) * WORKLOAD["calls_per_tool_turn"]
    subtasks = WORKLOAD["sequential_subtasks"]
    assert len(set(result["logical_ids"])) == subtasks
    assert result["work"]["model_calls"] == turns * subtasks
    assert result["work"]["finalizer_calls"] == subtasks
    for index, logical_id in enumerate(result["logical_ids"]):
        effects = result["effects"][result["starts"][index] : result["finishes"][index]]
        assert sorted(value for owner, value, _ in effects if owner == index) == list(
            range(expected_tools)
        )
        assert {owner_id for _, _, owner_id in effects} == {logical_id}
    assert len(result["effects"]) == expected_tools * subtasks
    assert projections == expected_tools * subtasks
    assert result["reported_tool_calls"] == expected_tools * subtasks
    if mode == "shared":
        assert len(set(result["work"]["session_ids"])) == 1
    else:
        assert result["work"]["session_ids"] == []
    assert max(result["snapshot_bytes"]) <= WORKLOAD["ceilings"]["state_bytes"]
    assert result["report_bytes"] <= WORKLOAD["ceilings"]["report_bytes"]
    assert result["retained_tool_records"] == min(
        expected_tools * subtasks, WORKLOAD["ceilings"]["tool_records"]
    )
    # A main response contributes at most call+response (or terminal) events;
    # each logical invocation adds a user message and terminal State event.
    session_turns = turns * (subtasks if mode == "shared" else 1)
    max_events = 2 * session_turns + 2 * subtasks
    work = result["work"]
    if mode == "isolated":
        # The successful isolated path reads State for synchronization and carry.
        assert work["max_read_events"] == 0
    else:
        assert work["max_read_events"] <= max_events
    assert work["max_event_scan"] <= max_events
    # ADK 2.15 visits the metered list five times per main request: auth,
    # confirmation, two content/rewind passes and resume selection. Routing,
    # scope and user-event setup add seven per logical invocation. These are
    # SDK reads, not a per-tool allowance; the width comparison below guards
    # that parallel tool callbacks add no whole-history passes.
    assert work["event_scans"] <= 5 * turns * subtasks + 7 * subtasks
    assert work["event_visits"] <= work["event_scans"] * max_events
    assert work["session_reads"] <= 4 * subtasks
    assert work["read_event_copies"] <= 4 * subtasks * max_events
    assert work["callback_calls"]["before_model_callback"] == turns * subtasks
    assert work["callback_calls"]["before_tool_callback"] == expected_tools * subtasks
    assert work["callback_calls"]["after_tool_callback"] == expected_tools * subtasks
    assert len(set(work["created_session_ids"])) == (1 if mode == "shared" else subtasks)


def test_more_parallel_tools_do_not_multiply_whole_history_passes(tmp_path):
    narrow = asyncio.run(
        run_workload(tmp_path / "two", turns=50, width=2, mode="shared", subtasks=3)
    )
    wide = asyncio.run(
        run_workload(tmp_path / "eight", turns=50, width=8, mode="shared", subtasks=3)
    )
    assert len(wide["effects"]) == 4 * len(narrow["effects"])
    for field in ("event_scans", "event_visits", "event_index_visits", "read_event_copies"):
        assert wide["work"][field] == narrow["work"][field]
    assert wide["retained_tool_records"] == WORKLOAD["ceilings"]["tool_records"]
    assert wide["reported_tool_calls"] == 1176
    assert wide["retained_first_call_id"] == "tool-00000177"
    assert wide["retained_last_call_id"] == "tool-00001176"
    assert wide["detail_truncated"]
    assert wide["report_bytes"] <= WORKLOAD["ceilings"]["report_bytes"]
