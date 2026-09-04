from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from fakes.model import json_result, scripted_model

from contractor_runtime.contracts import ModelPolicyRef, ResolvedModelPolicy
from contractor_runtime.result_finalizer import (
    MAX_RESULT_FINALIZER_INPUT_BYTES,
    ResultFinalizerFailure,
    WorkerResultFinalizer,
    build_result_finalizer_prompt,
)


@dataclass
class RecordingObserver:
    before: int = 0
    after: int = 0
    errors: int = 0
    usage: Any | None = None

    async def before_result_finalizer_call(self, *, invocation_id: str) -> None:
        assert invocation_id == "worker-invocation"
        self.before += 1

    async def after_result_finalizer_call(self, *, invocation_id: str, usage: Any | None) -> None:
        assert invocation_id == "worker-invocation"
        self.after += 1
        self.usage = usage

    async def on_result_finalizer_error(self, *, invocation_id: str, error: BaseException) -> None:
        assert invocation_id == "worker-invocation"
        assert error is not None
        self.errors += 1


def test_result_finalizer_is_one_tool_free_structured_adk_call() -> None:
    async def scenario() -> None:
        terminal_text = "Exact terminal text\nwith formatting."
        model = scripted_model(
            [json_result({"subtaskId": "1.2", "result": terminal_text})],
            auto_result_finalizer=False,
        )
        observer = RecordingObserver()
        finalizer = WorkerResultFinalizer(
            model=model,
            policy=policy(),
            observer=observer,
        )

        candidate = await finalizer.run(
            subtask_id="1.2",
            result_text=terminal_text,
            invocation_id="worker-invocation",
        )

        assert json.loads(candidate or "") == {
            "subtaskId": "1.2",
            "result": terminal_text,
        }
        assert observer.before == 1
        assert observer.after == 1
        assert observer.errors == 0
        assert observer.usage is not None
        assert len(model.requests) == 1
        request = model.requests[0]
        assert request["toolNames"] == []
        assert request["hasResponseSchema"] is True
        assert request["responseMimeType"] == "application/json"
        assert request["maxOutputTokens"] == 2048
        assert request["temperature"] == 0.0
        request_payload = json.loads(request["contentText"].split("\n", 1)[1])
        assert request_payload == {"resultText": terminal_text, "subtaskId": "1.2"}
        assert "Objective" not in request["contentText"]
        assert "ArtifactRef" not in request["contentText"]

    asyncio.run(scenario())


def test_result_finalizer_prompt_is_deterministic_and_bounded() -> None:
    first = build_result_finalizer_prompt(subtask_id="2", result_text="line\ntext")
    second = build_result_finalizer_prompt(subtask_id="2", result_text="line\ntext")

    assert first == second
    assert len(first.encode("utf-8")) <= MAX_RESULT_FINALIZER_INPUT_BYTES
    payload = json.loads(first.split("\n", 1)[1])
    assert payload == {"resultText": "line\ntext", "subtaskId": "2"}

    with pytest.raises(ResultFinalizerFailure, match="input_too_large"):
        build_result_finalizer_prompt(
            subtask_id="2",
            result_text="\x00" * (64 * 1024),
        )


def policy() -> ResolvedModelPolicy:
    return ResolvedModelPolicy(
        ref=ModelPolicyRef(
            policyId="worker",
            version="1",
            digest="sha256:" + "1" * 64,
        ),
        model="worker-model",
        maxOutputTokens=2048,
        maxModelCalls=4,
        maxToolCalls=8,
        maxTotalTokens=32768,
        temperature=0.0,
    )
