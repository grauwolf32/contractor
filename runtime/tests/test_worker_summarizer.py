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


def _summary_policy(context: int = 118_000, output: int = 8192):
    from contractor_runtime.contracts import ResolvedModelPolicy

    return ResolvedModelPolicy.model_validate(
        {
            "ref": {"policyId": "test-summary", "version": "1", "digest": "sha256:" + "a" * 64},
            "model": "worker-model",
            "contextWindowTokens": context,
            "maxOutputTokens": output,
            "maxModelCalls": 1,
        }
    )


def _summary_request(prompt: str, *, system: str = "System instructions"):
    from google.adk.models.llm_request import LlmRequest

    from contractor_runtime.contracts import WorkerModelResult

    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
        config=types.GenerateContentConfig(
            system_instruction=system, response_schema=WorkerModelResult, max_output_tokens=8192
        ),
    )


def test_summary_admission_counts_full_request_and_keeps_complete_newest_groups() -> None:
    from contractor_runtime.openai_gateway_llm import completion_request_byte_bound
    from contractor_runtime.summarizer import (
        SUMMARIZER_FRAMING_TOKEN_RESERVE,
        fit_summarizer_request,
    )

    groups = [
        [
            {"role": "model", "parts": [{"text": f"group-{n}:" + "界" * 8000}]},
            {"role": "user", "parts": [{"text": f"response-{n}"}]},
        ]
        for n in range(12)
    ]
    prompt = build_summarizer_prompt(
        _request(),
        WorkerObservations(profile="lean@1", tools={}, workspace=None, truncated=False),
        groups,
        transcript_truncated=False,
    )
    request = _summary_request(prompt, system="Mandatory instructions " + "s" * 20_000)
    original = json.loads(prompt.split("\n", 1)[1])
    policy = _summary_policy()
    capacity = (
        policy.context_window_tokens - policy.max_output_tokens - SUMMARIZER_FRAMING_TOKEN_RESERVE
    )
    assert completion_request_byte_bound(policy.model, request) > capacity
    fit_summarizer_request(request, policy)
    assert completion_request_byte_bound(policy.model, request) <= capacity
    retained = json.loads(request.contents[0].parts[0].text.split("\n", 1)[1])
    assert retained["task"] == original["task"]
    assert retained["observations"] == original["observations"]
    assert retained["transcriptTruncated"] is True
    assert retained["transcript"]
    assert retained["transcript"] == groups[-len(retained["transcript"]) :]
    assert all(len(group) == 2 for group in retained["transcript"])
    assert request.config.system_instruction.endswith("s" * 20_000)
    assert request.config.response_schema is not None


def test_summary_exact_admission_boundary_does_not_truncate() -> None:
    from contractor_runtime.openai_gateway_llm import completion_request_byte_bound
    from contractor_runtime.summarizer import (
        SUMMARIZER_FRAMING_TOKEN_RESERVE,
        fit_summarizer_request,
    )

    prompt = build_summarizer_prompt(
        _request(),
        WorkerObservations(profile="lean@1", tools={}, workspace=None, truncated=False),
        [],
        transcript_truncated=False,
    )
    request = _summary_request(prompt)
    size = completion_request_byte_bound("worker-model", request)
    policy = _summary_policy(context=size + 8192 + SUMMARIZER_FRAMING_TOKEN_RESERVE)
    fit_summarizer_request(request, policy)
    assert request.contents[0].parts[0].text == prompt


def test_summary_oversized_task_fails_before_calling_the_model() -> None:
    import asyncio

    import pytest
    from fakes.model import scripted_model

    from contractor_runtime.summarizer import SummarizerFailure, TerminalSummarizer

    async def scenario() -> None:
        model = scripted_model([])
        summarizer = TerminalSummarizer(model=model, policy=_summary_policy(context=16_384))
        prompt = build_summarizer_prompt(
            _request(instructions="immutable " * 12_000),
            WorkerObservations(profile="lean@1", tools={}, workspace=None, truncated=False),
            [],
            transcript_truncated=False,
        )
        with pytest.raises(SummarizerFailure) as captured:
            await summarizer.run(prompt=prompt, invocation_id="test-invocation")
        assert captured.value.code == "input_context_exceeded"
        assert captured.value.retryable is False
        assert "immutable" not in str(captured.value)
        assert not model.requests
        assert summarizer.usage.model_calls == 0

    asyncio.run(scenario())
