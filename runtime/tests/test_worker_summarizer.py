from __future__ import annotations

import json

from google.adk.events import Event
from google.genai import types

from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    StageContentRequest,
    WorkerObservations,
)
from contractor_runtime.summarizer import (
    MAX_SUMMARIZER_INPUT_BYTES,
    TranscriptRecorder,
    build_summarizer_prompt,
)

SECRET = "summary-projection-secret"
HOST_PATH = "/srv/private/contractor/run-1"


def test_transcript_keeps_only_complete_groups_and_redacts_forbidden_values() -> None:
    recorder = TranscriptRecorder(secrets=(SECRET, HOST_PATH))
    recorder.record(
        Event(
            invocationId="worker-1",
            author="contractor_worker",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="private chain", thought=True),
                    types.Part(
                        function_call=types.FunctionCall(
                            id="call-1",
                            name="probe",
                            args={
                                "token": SECRET,
                                "relative": "src/main.py",
                                "physical": f"{HOST_PATH}/src/main.py",
                            },
                        )
                    ),
                ],
            ),
        )
    )
    recorder.record(
        Event(
            invocationId="worker-1",
            author="probe",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="call-1",
                            name="probe",
                            response={"authorization": SECRET, "ok": True},
                        )
                    )
                ],
            ),
        )
    )

    groups = recorder.finish()

    assert len(groups) == 1
    assert len(groups[0]) == 2
    encoded = json.dumps(groups, sort_keys=True)
    assert "private chain" not in encoded
    assert SECRET not in encoded
    assert HOST_PATH not in encoded
    assert encoded.count("[REDACTED]") >= 3
    assert "src/main.py" in encoded


def test_transcript_discards_unpaired_calls_and_responses() -> None:
    call_only = TranscriptRecorder()
    call_only.record(_tool_call_event("orphan-call"))
    assert call_only.finish() == ()
    assert call_only.truncated is True

    response_only = TranscriptRecorder()
    response_only.record(_tool_response_event("orphan-response"))
    assert response_only.finish() == ()
    assert response_only.truncated is True

    mismatched = TranscriptRecorder()
    mismatched.record(_tool_call_event("expected-call"))
    mismatched.record(_tool_response_event("different-call"))
    assert mismatched.finish() == ()
    assert mismatched.truncated is True


def test_summarizer_document_is_bounded_and_prefers_newest_complete_groups() -> None:
    recorder = TranscriptRecorder()
    for index in range(12):
        recorder.record(
            Event(
                invocationId="worker-1",
                author="contractor_worker",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=f"event-{index:02d}:" + "x" * 70_000)],
                ),
            )
        )
    groups = recorder.finish()
    request = _request(
        objective=f"Inspect {HOST_PATH} with {SECRET}",
        instructions=f"Never disclose {SECRET}",
    )

    prompt = build_summarizer_prompt(
        request,
        WorkerObservations(
            profile="lean@1",
            tools={},
            workspace=None,
            truncated=False,
        ),
        groups,
        transcript_truncated=recorder.truncated,
        secrets=(SECRET, HOST_PATH),
    )

    assert len(prompt.encode("utf-8")) <= MAX_SUMMARIZER_INPUT_BYTES
    assert SECRET not in prompt
    assert HOST_PATH not in prompt
    payload = json.loads(prompt.split("\n", 1)[1])
    assert payload["task"]["objective"] == "Inspect [REDACTED] with [REDACTED]"
    assert payload["transcriptTruncated"] is True
    retained = [
        part["text"].split(":", 1)[0]
        for group in payload["transcript"]
        for event in group
        for part in event["parts"]
    ]
    assert retained
    assert retained == sorted(retained)
    assert retained[-1] == "event-11"
    assert "event-00" not in retained


def _request(
    *, objective: str = "Inspect source", instructions: str = "Use evidence"
) -> StageContentRequest:
    return StageContentRequest(
        apiVersion=API_VERSION,
        subtaskId="1.2",
        objective=objective,
        instructions=instructions,
        parameters={"language": "python"},
        artifacts={"source": ArtifactRef(namespace="inputs", name="source", revision="revision-1")},
        resultArtifacts={},
    )


def _tool_call_event(call_id: str) -> Event:
    return Event(
        invocationId="worker-1",
        author="contractor_worker",
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=types.FunctionCall(id=call_id, name="probe", args={}))],
        ),
    )


def _tool_response_event(call_id: str) -> Event:
    return Event(
        invocationId="worker-1",
        author="probe",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=call_id,
                        name="probe",
                        response={"ok": True},
                    )
                )
            ],
        ),
    )


def test_summary_sends_large_projection_without_local_context_trimming() -> None:
    import asyncio

    from fakes.model import json_result, scripted_model
    from fakes.spec import allocation_spec

    from contractor_runtime.summarizer import TerminalSummarizer

    async def scenario() -> None:
        model = scripted_model([json_result({"subtaskId": "1", "result": "summary"})])
        config = allocation_spec(summarizer=True).agent_template.summarizer
        assert config is not None
        # Below 512 KiB, but above the removed byte-based admission allowance.
        prompt = "история " * 20_000
        summarizer = TerminalSummarizer(model=model, policy=config.model_policy)
        await summarizer.run(prompt=prompt, invocation_id="large-summary")
        assert len(model.requests) == 1
        assert model.requests[0]["contentText"] == prompt

    asyncio.run(scenario())
