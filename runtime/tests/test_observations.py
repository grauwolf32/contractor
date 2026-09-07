from __future__ import annotations

import asyncio
import json

import pytest

import contractor_runtime.worker.observations as observations_module
from contractor_runtime.projectfs import WorkspaceObservationMetadata
from contractor_runtime.telemetry.invocations import InvocationMetricsReducer
from contractor_runtime.worker.observations import (
    WorkspaceObservationReducer,
    WorkspaceToolObservation,
    lean_workspace_summary,
    validate_workspace_observation,
)
from contractor_runtime.worker.state import MAX_AGENT_STATE_SNAPSHOT_BYTES, WorkerStateStore


def metadata(*paths: str, digest_character: str = "0") -> WorkspaceObservationMetadata:
    return WorkspaceObservationMetadata(
        digest="sha256:" + digest_character * 64,
        managed_text_paths=tuple(paths),
    )


def test_reducer_tracks_unique_coverage_and_deterministic_call_order() -> None:
    reducer = WorkspaceObservationReducer.from_metadata(metadata("a.py", "b.py", "c.py"))
    reducer.record(
        WorkspaceToolObservation(discovered=("b.py", "a.py")),
        ordinal=1,
    )
    reducer.record(WorkspaceToolObservation(read=("b.py",)), ordinal=2)
    reducer.record(WorkspaceToolObservation(read=("c.py", "b.py")), ordinal=3)
    reducer.record(WorkspaceToolObservation(matched=("a.py", "a.py")), ordinal=4)
    reducer.record(WorkspaceToolObservation(modified=("created.py",)), ordinal=5)

    state = reducer.snapshot()
    assert [item["path"] for item in state["interactions"]] == [
        "a.py",
        "b.py",
        "c.py",
        "created.py",
    ]
    by_path = {item["path"]: item for item in state["interactions"]}
    assert by_path["a.py"]["discoveryCalls"] == 1
    assert by_path["a.py"]["matchCalls"] == 1
    assert by_path["b.py"]["readCalls"] == 2
    assert by_path["b.py"]["firstOrdinal"] == 1
    assert by_path["b.py"]["lastOrdinal"] == 3

    summary, truncated = lean_workspace_summary(state)
    assert summary is not None
    assert summary.model_dump(by_alias=True) == {
        "scopedFiles": 3,
        "scopeComplete": True,
        "discoveredFiles": 2,
        "readFiles": 2,
        "matchedFiles": 1,
        "modifiedFiles": 1,
        "detailComplete": True,
        "unreadFiles": 1,
        "filesRead": ["b.py", "c.py"],
        "filesReadTruncated": False,
    }
    assert truncated is False


def test_reducer_bounds_scope_interactions_and_lean_path_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observations_module, "MAX_WORKSPACE_SCOPE_PATHS", 2)
    monkeypatch.setattr(observations_module, "MAX_WORKSPACE_INTERACTIONS", 3)
    monkeypatch.setattr(observations_module, "MAX_LEAN_FILES_READ", 2)
    reducer = WorkspaceObservationReducer.from_metadata(metadata("a.py", "b.py", "c.py", "d.py"))
    assert reducer.scope_paths == ("a.py", "b.py")
    assert reducer.scope_complete is False
    for ordinal, path in enumerate(("a.py", "b.py", "c.py", "d.py"), start=1):
        reducer.record(WorkspaceToolObservation(read=(path,)), ordinal=ordinal)

    state = reducer.snapshot()
    assert len(state["interactions"]) == 3
    assert state["detailComplete"] is False
    summary, truncated = lean_workspace_summary(state)
    assert summary is not None
    assert summary.scoped_files == 2
    assert summary.read_files == 3
    assert summary.files_read == ["a.py", "b.py"]
    assert summary.files_read_truncated is True
    assert summary.unread_files is None
    assert truncated is True


def test_scope_byte_limit_stops_at_a_deterministic_lexical_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "a-" + "x" * 40
    second = "b-" + "y" * 40
    monkeypatch.setattr(
        observations_module,
        "MAX_WORKSPACE_SCOPE_PATH_BYTES",
        len(json.dumps([first], separators=(",", ":")).encode("utf-8")),
    )
    reducer = WorkspaceObservationReducer.from_metadata(metadata(first, second))
    assert reducer.scope_paths == (first,)
    assert reducer.scope_complete is False


def test_reducer_canonicalizes_defective_workspace_metadata() -> None:
    reducer = WorkspaceObservationReducer.from_metadata(
        metadata("b.py", "a.py", "b.py", "../invalid")
    )

    assert reducer.scope_paths == ("a.py", "b.py")
    assert reducer.scope_complete is False
    assert validate_workspace_observation(reducer.snapshot()) == reducer.snapshot()


def test_state_validation_rejects_unknown_content_and_non_normalized_paths() -> None:
    reducer = WorkspaceObservationReducer.from_metadata(metadata("safe/file.py"))
    reducer.record(WorkspaceToolObservation(read=("safe/file.py",)), ordinal=1)
    valid = reducer.snapshot()
    assert validate_workspace_observation(valid) == valid

    unknown = {**valid, "source": "recognizable-source-secret"}
    with pytest.raises(ValueError, match="fields"):
        validate_workspace_observation(unknown)
    traversal = {**valid, "scopePaths": ["../secret"]}
    with pytest.raises(ValueError, match="path"):
        validate_workspace_observation(traversal)
    inconsistent = {
        **valid,
        "interactions": [
            {
                **valid["interactions"][0],
                "readCalls": 0,
            }
        ],
    }
    with pytest.raises(ValueError, match="inconsistent"):
        validate_workspace_observation(inconsistent)


def test_worker_state_shared_envelope_trims_optional_workspace_detail() -> None:
    async def scenario() -> None:
        paths = tuple(f"{index:05d}-{'x' * 180}.py" for index in range(9_000))
        reducer = WorkspaceObservationReducer.from_metadata(metadata(*paths))
        reducer.record(WorkspaceToolObservation(read=paths), ordinal=1)
        full = reducer.snapshot()
        assert full["scopeComplete"] is True
        assert full["detailComplete"] is True

        state = WorkerStateStore()
        metrics = InvocationMetricsReducer().snapshot()
        first = await state.begin_invocation(
            invocation_id="worker-large-first",
            subtask_id="1",
            metrics=metrics,
            workspace=full,
        )
        assert len(encoded_state(first)) <= MAX_AGENT_STATE_SNAPSHOT_BYTES
        fitted = first["currentInvocation"]["workspace"]
        assert not (fitted["scopeComplete"] and fitted["detailComplete"])
        completed = await state.complete_invocation(
            invocation_id="worker-large-first",
            phase="succeeded",
            metrics=metrics,
            workspace=full,
        )
        second = await state.begin_invocation(
            invocation_id="worker-large-second",
            subtask_id="2",
            metrics=metrics,
            workspace=full,
        )
        assert completed["lastCompletedInvocation"] is not None
        assert len(encoded_state(second)) <= MAX_AGENT_STATE_SNAPSHOT_BYTES
        for field_name in ("currentInvocation", "lastCompletedInvocation"):
            invocation = second[field_name]
            assert invocation is not None
            workspace = invocation["workspace"]
            assert not (workspace["scopeComplete"] and workspace["detailComplete"])

    asyncio.run(scenario())


def encoded_state(state: dict[str, object]) -> bytes:
    return json.dumps(
        {"apiVersion": "contractor/v1alpha1", "state": state},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
