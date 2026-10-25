"""Real-Runner contracts that dependency preprocessing must not weaken."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fakes.model import scripted_model, text_result, thought_result, tool_call
from test_adk_runtime import (
    FakeArtifactClient,
    create_runtime,
    selected_tools,
    stage_request,
)

from contractor_runtime.allocation import WorkerState
from contractor_runtime.core.input_errors import ToolInputError


@pytest.mark.parametrize("field", ["offset", "length"])
@pytest.mark.parametrize("value", [1.0, True, "1", None])
def test_runner_preserves_raw_artifact_integer_rejection(
    tmp_path: Path, field: str, value: object
) -> None:
    async def scenario() -> None:
        state = WorkerState()
        client = FakeArtifactClient()
        tools = await selected_tools(tmp_path, client, state)
        model = scripted_model(
            [
                tool_call(
                    "read_artifact",
                    {"namespace": "inputs", "name": "source", field: value},
                    call_id="invalid-integer",
                ),
                text_result("Rejected invalid read"),
            ]
        )
        runtime = await create_runtime(tmp_path, state, tools, model)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            assert client.calls == []
            assert (
                state.metrics.build_report(report_id="raw-read", duration_ms=1)
                .metrics.tools["read_artifact"]
                .failed
                == 1
            )
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize("values", [[1.0], [True], ["1"], [None]])
def test_runner_preserves_raw_integer_list_rejection(tmp_path: Path, values: list[object]) -> None:
    async def scenario() -> None:
        effects: list[list[int]] = []

        async def strict_list(values: list[int]) -> dict[str, bool]:
            """Record a list only when each original element is an integer."""
            if any(type(value) is not int for value in values):
                raise ToolInputError("values must contain only integers")
            effects.append(values)
            return {"ok": True}

        model = scripted_model(
            [
                tool_call("strict_list", {"values": values}, call_id="invalid-list"),
                text_result("Rejected invalid list"),
            ]
        )
        runtime = await create_runtime(tmp_path, WorkerState(), {"strict_list": strict_list}, model)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is not None
            assert effects == []
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (text_result(""), "worker_result_invalid"),
        (text_result(" \n"), "worker_result_invalid"),
        (thought_result(), "worker_result_missing"),
    ],
)
def test_runner_empty_visible_output_keeps_missing_result_outcome(
    tmp_path: Path, response: object, code: str
) -> None:
    async def scenario() -> None:
        model = scripted_model([response])
        runtime = await create_runtime(tmp_path, WorkerState(), {}, model)
        try:
            result = await runtime.invoke(stage_request())
            assert result.result is None
            assert result.failure is not None
            assert result.failure.code == code
            assert len(model.requests) == 1
        finally:
            await runtime.finalize(datetime.now(UTC) + timedelta(seconds=1))

    asyncio.run(scenario())
