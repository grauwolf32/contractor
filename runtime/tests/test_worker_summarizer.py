from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from google.adk.events import Event
from google.genai import types

import contractor_runtime.worker.summarizer as summarizer_module
from contractor_runtime.contracts import (
    API_VERSION,
    ArtifactRef,
    StageContentRequest,
    WorkerObservations,
)
from contractor_runtime.reasoning import (
    MAX_REASONING_PART_BYTES,
    MAX_REASONING_TOTAL_BYTES,
    REASONING_INPUT_INSTRUCTION,
    REASONING_TAG,
    REASONING_TRUNCATED_MEMBER,
    TRUNCATION_MARKER,
    tagged_size,
)
from contractor_runtime.worker.summarizer import (
    MAX_SUMMARIZER_INPUT_BYTES,
    TranscriptRecorder,
    build_summarizer_prompt,
)

SECRET = "summary-projection-secret"
HOST_PATH = "/srv/private/contractor/run-1"


def test_summary_checks_semantic_admission_before_model_or_attempt_charge() -> None:
    import asyncio

    from fakes.model import json_result, scripted_model
    from fakes.spec import summarizer_config

    from contractor_runtime.worker.summarizer import TerminalSummarizer

    def expired() -> None:
        raise TimeoutError("Invocation deadline exceeded")

    async def scenario() -> None:
        model = scripted_model([json_result({"subtaskId": "1", "result": "late"})])
        config = summarizer_config()
        assert config is not None
        summarizer = TerminalSummarizer(
            model=model,
            policy=config.model_policy,
            include_reasoning=False,
            check_admission=expired,
        )
        with pytest.raises(TimeoutError):
            await summarizer.run(prompt="Never admitted", invocation_id="expired-summary")
        assert model.requests == []
        assert summarizer.usage.model_calls == 0

    asyncio.run(scenario())


def test_transcript_keeps_only_complete_groups_and_redacts_forbidden_values() -> None:
    recorder = TranscriptRecorder(include_reasoning=False, secrets=(SECRET, HOST_PATH))
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
    call_only = TranscriptRecorder(include_reasoning=False)
    call_only.record(_tool_call_event("orphan-call"))
    assert call_only.finish() == ()
    assert call_only.truncated is True

    response_only = TranscriptRecorder(include_reasoning=False)
    response_only.record(_tool_response_event("orphan-response"))
    assert response_only.finish() == ()
    assert response_only.truncated is True

    mismatched = TranscriptRecorder(include_reasoning=False)
    mismatched.record(_tool_call_event("expected-call"))
    mismatched.record(_tool_response_event("different-call"))
    assert mismatched.finish() == ()
    assert mismatched.truncated is True


def test_summarizer_document_is_bounded_and_prefers_newest_complete_groups() -> None:
    recorder = TranscriptRecorder(include_reasoning=False)
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
            truncated=False,
        ),
        groups,
        transcript_truncated=recorder.truncated,
        include_reasoning=False,
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
        deadline=datetime.now(UTC) + timedelta(hours=1),
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


def test_summary_uses_literal_configured_instructions() -> None:
    import asyncio

    from fakes.model import json_result, scripted_model
    from fakes.spec import summarizer_config

    from contractor_runtime.worker.summarizer import TerminalSummarizer

    async def scenario() -> None:
        model = scripted_model([json_result({"subtaskId": "1", "result": "summary"})])
        config = summarizer_config()
        assert config is not None
        instructions = 'Составь итог {subtaskId}. JSON: {"result": "text"}.'
        prompt = "Worker transcript"
        summarizer = TerminalSummarizer(
            model=model,
            policy=config.model_policy,
            instructions=instructions,
            include_reasoning=False,
        )
        await summarizer.run(prompt=prompt, invocation_id="literal-summary")
        assert len(model.requests) == 1
        assert instructions in model.requests[0]["systemInstruction"]
        assert model.requests[0]["contentText"] == prompt

    asyncio.run(scenario())


def test_summary_sends_large_projection_without_local_context_trimming() -> None:
    import asyncio

    from fakes.model import json_result, scripted_model
    from fakes.spec import summarizer_config

    from contractor_runtime.worker.summarizer import TerminalSummarizer

    async def scenario() -> None:
        model = scripted_model([json_result({"subtaskId": "1", "result": "summary"})])
        config = summarizer_config()
        assert config is not None
        # Below 512 KiB, but above the removed byte-based admission allowance.
        prompt = "история " * 20_000
        summarizer = TerminalSummarizer(
            model=model, policy=config.model_policy, include_reasoning=False
        )
        await summarizer.run(prompt=prompt, invocation_id="large-summary")
        assert len(model.requests) == 1
        assert model.requests[0]["contentText"] == prompt

    asyncio.run(scenario())


LONE_SURROGATE = "\ud800"


def test_summary_json_measure_rejects_values_the_document_cannot_encode() -> None:
    # Deliberately unlike the tolerant telemetry measure (telemetry/metrics.py).
    payload = {"b": "é", "a": [1, None, True]}
    compact = len(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    preamble = len(summarizer_module._DOCUMENT_PREAMBLE.encode("utf-8"))
    assert summarizer_module._json_size(payload) == compact
    assert summarizer_module._document_size(payload) == preamble + compact
    for measure in (summarizer_module._json_size, summarizer_module._document_size):
        with pytest.raises(ValueError):
            measure({"value": math.nan})
        with pytest.raises(TypeError):
            measure({"value": object()})
        with pytest.raises(TypeError):
            measure({1: "integer key", "a": "text key"})
        with pytest.raises(UnicodeEncodeError):
            measure({"value": LONE_SURROGATE})


def test_summary_text_bound_keeps_whole_characters_and_appends_its_marker() -> None:
    limit = summarizer_module._MAX_PROJECTED_TEXT_BYTES
    bound = summarizer_module._bound_text
    assert bound("short", ()) == ("short", False)
    assert bound(f"a {SECRET} b", (SECRET,)) == ("a [REDACTED] b", True)
    assert bound("a" * limit, ()) == ("a" * limit, False)
    # A character crossing the limit is dropped whole; the marker follows it.
    assert bound("a" * (limit - 1) + "é", ()) == ("a" * (limit - 1) + "[TRUNCATED]", True)
    with pytest.raises(UnicodeEncodeError):
        bound(LONE_SURROGATE, ())


REASONING_CANARY = "reasoning-hypothesis-canary"


def _reasoning_call_event(call_id: str, *reasoning: str, text: str | None = None) -> Event:
    parts = [types.Part(text=value, thought=True) for value in reasoning]
    if text is not None:
        parts.append(types.Part(text=text))
    parts.append(
        types.Part(function_call=types.FunctionCall(id=call_id, name="probe", args={"n": call_id}))
    )
    return Event(
        invocationId="worker-1",
        author="contractor_worker",
        content=types.Content(role="model", parts=parts),
    )


def _model_event(*parts: types.Part) -> Event:
    return Event(
        invocationId="worker-1",
        author="contractor_worker",
        content=types.Content(role="model", parts=list(parts)),
    )


def _payload(prompt: str) -> dict[str, Any]:
    return json.loads(prompt.split("\n", 1)[1])


def _parts(groups: Any) -> list[dict[str, Any]]:
    return [part for group in groups for event in group for part in event["parts"]]


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_transcript_reasoning_is_tagged_on_its_model_turn_only_when_enabled(
    include_reasoning: bool,
) -> None:
    recorder = TranscriptRecorder(include_reasoning=include_reasoning, secrets=(SECRET, HOST_PATH))
    recorder.record(
        _reasoning_call_event(
            "call-1",
            f"{REASONING_CANARY}: read {HOST_PATH}/a.py with {SECRET}",
            text="Reading a.py",
        )
    )
    recorder.record(_tool_response_event("call-1"))
    # Opaque or absent reasoning: a thought part without text, an empty one.
    recorder.record(
        _model_event(
            types.Part(thought=True, thought_signature=b"opaque"),
            types.Part(text="", thought=True),
            types.Part(text="Next step"),
        )
    )
    groups = recorder.finish()
    encoded = json.dumps(groups, ensure_ascii=False)

    assert SECRET not in encoded and HOST_PATH not in encoded
    first_model_parts = groups[0][0]["parts"]
    if include_reasoning:
        assert first_model_parts[0] == {
            REASONING_TAG: f"{REASONING_CANARY}: read [REDACTED]/a.py with [REDACTED]"
        }
        assert first_model_parts[1] == {"text": "Reading a.py"}
        assert sum(1 for part in _parts(groups) if REASONING_TAG in part) == 1
    else:
        assert REASONING_CANARY not in encoded
        assert all(REASONING_TAG not in part for part in _parts(groups))
    assert groups[1] == ({"role": "model", "parts": [{"text": "Next step"}]},)
    assert recorder.reasoning_truncated is False
    assert recorder.truncated is False


def test_reasoning_only_turn_is_a_group_only_when_enabled() -> None:
    event = _model_event(types.Part(text=REASONING_CANARY, thought=True))
    disabled = TranscriptRecorder(include_reasoning=False)
    disabled.record(event)
    enabled = TranscriptRecorder(include_reasoning=True)
    enabled.record(event)

    assert disabled.finish() == ()
    assert enabled.finish() == (({"role": "model", "parts": [{REASONING_TAG: REASONING_CANARY}]},),)


def test_oversized_multibyte_reasoning_is_cut_and_marked() -> None:
    recorder = TranscriptRecorder(include_reasoning=True, secrets=(SECRET,))
    recorder.record(_reasoning_call_event("call-1", "界" * 5000 + SECRET))
    recorder.record(_tool_response_event("call-1"))
    groups = recorder.finish()
    part = groups[0][0]["parts"][0]

    assert part[REASONING_TAG].endswith(TRUNCATION_MARKER)
    assert tagged_size(part[REASONING_TAG]) <= MAX_REASONING_PART_BYTES
    assert recorder.reasoning_truncated is True
    assert SECRET not in json.dumps(groups)


def test_retained_reasoning_is_bounded_newest_first_and_never_evicts_evidence() -> None:
    # A realistic ceiling: one group per normal model call (LIM-TPL-3).
    calls = 1000
    with_reasoning = TranscriptRecorder(include_reasoning=True)
    without_reasoning = TranscriptRecorder(include_reasoning=False)
    for index in range(calls):
        call_id = f"call-{index:04d}"
        event = _reasoning_call_event(call_id, f"{index:04d}" + "r" * 9000)
        for recorder in (with_reasoning, without_reasoning):
            recorder.record(event)
            recorder.record(_tool_response_event(call_id))
    enabled = with_reasoning.finish()
    disabled = without_reasoning.finish()

    assert len(enabled) == len(disabled) == calls
    assert [
        [
            {"role": event["role"], "parts": [p for p in event["parts"] if REASONING_TAG not in p]}
            for event in group
        ]
        for group in enabled
    ] == [list(group) for group in disabled]
    kept = [part[REASONING_TAG] for part in _parts(enabled) if REASONING_TAG in part]
    assert sum(tagged_size(text) for text in kept) <= MAX_REASONING_TOTAL_BYTES
    assert len(kept) == MAX_REASONING_TOTAL_BYTES // MAX_REASONING_PART_BYTES
    assert [text[:4] for text in kept] == [f"{index:04d}" for index in range(calls - 8, calls)]
    assert with_reasoning.reasoning_truncated is True


def _observations() -> WorkerObservations:
    return WorkerObservations(profile="lean@1", tools={}, truncated=False)


def test_summary_document_without_reasoning_is_unchanged_by_the_option() -> None:
    recorder = TranscriptRecorder(include_reasoning=False, secrets=(SECRET,))
    recorder.record(_reasoning_call_event("call-1", REASONING_CANARY, text="Reading"))
    recorder.record(_tool_response_event("call-1"))
    prompt = build_summarizer_prompt(
        _request(),
        _observations(),
        recorder.finish(),
        transcript_truncated=recorder.truncated,
        include_reasoning=False,
        secrets=(SECRET,),
    )
    payload = _payload(prompt)
    assert REASONING_CANARY not in prompt
    assert set(payload) == {"task", "observations", "transcript", "transcriptTruncated"}


def test_summary_document_admits_reasoning_after_evidence_within_both_bounds() -> None:
    recorder = TranscriptRecorder(include_reasoning=True, secrets=(SECRET,))
    for index in range(4):
        call_id = f"call-{index}"
        recorder.record(_reasoning_call_event(call_id, f"{REASONING_CANARY}-{index} {SECRET}"))
        recorder.record(_tool_response_event(call_id))
    prompt = build_summarizer_prompt(
        _request(),
        _observations(),
        recorder.finish(),
        transcript_truncated=recorder.truncated,
        include_reasoning=True,
        reasoning_truncated=recorder.reasoning_truncated,
        secrets=(SECRET,),
    )
    payload = _payload(prompt)

    assert payload[REASONING_TRUNCATED_MEMBER] is False
    assert payload["transcriptTruncated"] is False
    reasoning = [
        part[REASONING_TAG] for part in _parts(payload["transcript"]) if REASONING_TAG in part
    ]
    assert reasoning == [f"{REASONING_CANARY}-{index} [REDACTED]" for index in range(4)]
    assert SECRET not in prompt


@pytest.mark.parametrize("spare", [0, 1, 40])
def test_reasoning_never_displaces_evidence_at_the_document_bound(spare: int) -> None:
    # Evidence is sized so that the evidence-only document leaves ``spare``
    # bytes; the 36-byte reasoning part of the newest group then fits only
    # when spare covers its serialized bytes and separator.
    def groups_for(filler: int) -> tuple[tuple[dict[str, Any], ...], ...]:
        recorder = TranscriptRecorder(include_reasoning=True)
        for index in range(8):
            call_id = f"call-{index}"
            recorder.record(
                _reasoning_call_event(call_id, f"hypothesis-{index}", text="e" * 60_000)
            )
            recorder.record(_tool_response_event(call_id))
        recorder.record(_model_event(types.Part(text="f" * filler)))
        return recorder.finish()

    def evidence_prompt(groups: Any) -> str:
        return build_summarizer_prompt(
            _request(),
            _observations(),
            groups,
            transcript_truncated=False,
            include_reasoning=True,
        )

    probe = _payload(evidence_prompt(groups_for(1)))
    assert probe["transcriptTruncated"] is False
    # Grow the final evidence group until the evidence-only document leaves
    # exactly ``spare`` bytes (reasoning is admitted only afterwards).
    without = [
        [
            {"role": e["role"], "parts": [p for p in e["parts"] if REASONING_TAG not in p]}
            for e in group
        ]
        for group in groups_for(1)
    ]
    evidence_only = summarizer_module._DOCUMENT_PREAMBLE + json.dumps(
        {**probe, "transcript": without, REASONING_TRUNCATED_MEMBER: False},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    filler = 1 + MAX_SUMMARIZER_INPUT_BYTES - len(evidence_only.encode("utf-8")) - spare
    groups = groups_for(filler)
    payload = _payload(evidence_prompt(groups))

    assert payload["transcriptTruncated"] is False
    assert len(payload["transcript"]) == 9
    assert payload["transcript"][-1][0]["parts"] == [{"text": "f" * filler}]
    reasoning = [
        part[REASONING_TAG] for part in _parts(payload["transcript"]) if REASONING_TAG in part
    ]
    # {"reasoning":"hypothesis-7"} is 28 bytes plus a separator.
    assert reasoning == (["hypothesis-7"] if spare >= 29 else [])
    assert payload[REASONING_TRUNCATED_MEMBER] is True


def test_summary_document_charges_reasoning_against_its_total_bound() -> None:
    groups = tuple(
        (
            {
                "role": "model",
                "parts": [{REASONING_TAG: "r" * (MAX_REASONING_PART_BYTES - 16)}, {"text": str(i)}],
            },
        )
        for i in range(9)
    )
    payload = _payload(
        build_summarizer_prompt(
            _request(),
            _observations(),
            groups,
            transcript_truncated=False,
            include_reasoning=True,
        )
    )
    kept = [part for part in _parts(payload["transcript"]) if REASONING_TAG in part]
    assert len(payload["transcript"]) == 9
    assert len(kept) == 8
    # The newest groups keep their reasoning; the oldest one loses only it.
    assert payload["transcript"][0] == [{"role": "model", "parts": [{"text": "0"}]}]
    assert payload[REASONING_TRUNCATED_MEMBER] is True


@pytest.mark.parametrize("include_reasoning", [False, True])
def test_reasoning_instruction_is_added_only_when_enabled(include_reasoning: bool) -> None:
    import asyncio

    from fakes.model import json_result, scripted_model
    from fakes.spec import summarizer_config

    from contractor_runtime.worker.summarizer import TerminalSummarizer

    async def scenario() -> None:
        model = scripted_model([json_result({"subtaskId": "1", "result": "summary"})])
        config = summarizer_config()
        assert config is not None
        configured = "Составь итог {subtaskId}."
        summarizer = TerminalSummarizer(
            model=model,
            policy=config.model_policy,
            instructions=configured,
            include_reasoning=include_reasoning,
        )
        await summarizer.run(prompt="Worker transcript", invocation_id="reasoning-summary")
        instruction = str(model.requests[0]["systemInstruction"])
        assert instruction.startswith(configured)
        assert (REASONING_INPUT_INSTRUCTION in instruction) is include_reasoning

    asyncio.run(scenario())
