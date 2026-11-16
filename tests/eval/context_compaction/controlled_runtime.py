"""Test-only composition of the ordinary Runtime for frozen continuation probes.

There is no model/tool loop here. Native ADK callbacks, admission, accounting,
failure handling and cleanup remain in the production Worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import harness as h
from google.genai import types

from contractor_runtime.adapters.content import model_request_content
from contractor_runtime.contracts import WorkerSessionMode
from contractor_runtime.redaction import runtime_secrets
from contractor_runtime.telemetry.redaction import TelemetryRedactor
from contractor_runtime.threads import to_thread_until_done
from contractor_runtime.worker.compaction import ContextArchive
from contractor_runtime.worker.runtime import AdkWorkerRuntime

CASES = frozenset({"source-review", "api-dependency", "negative-memory"})
TEMPLATES = {
    "context-eval-controlled-compact": "compact@1",
    "context-eval-masking": "masking",
}
MASK = "[Observation masked; re-read the exact artifact if needed.]"
MAX_QUESTIONS_BYTES = 64 * 1024
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 4 * 1024 * 1024
CASE_PREFIX = re.compile(r'^\s*\{\s*"id"\s*:\s*"([a-z-]+)"')


def load_questions(path, expected):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_QUESTIONS_BYTES + 1)
    h.require(len(raw) <= MAX_QUESTIONS_BYTES, "private question document exceeds its bound")
    h.require(h.digest(raw) == expected, "private question pin mismatch")
    document = h.decode(raw)
    h.require(
        isinstance(document, dict)
        and set(document) == {"schemaVersion", "questions"}
        and document["schemaVersion"] == "context-compaction-probes/v1",
        "invalid private question document",
    )
    questions = document["questions"]
    h.require(
        isinstance(questions, dict)
        and set(questions) == CASES
        and all(isinstance(q, str) and 0 < len(q) <= 2048 for q in questions.values()),
        "invalid private question matrix",
    )
    return questions


def request_digest(request):
    # Explicit SDK projection, not raw provider transport bytes. The native
    # content owner excludes callable tools_dict and credentials in settings.
    value = TelemetryRedactor(()).value(model_request_content(request), redact=False)
    fingerprint, size = hashlib.sha256(), 0
    for chunk in json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), allow_nan=False
    ).iterencode(value):
        raw = chunk.encode()
        size += len(raw)
        h.require(size <= MAX_REQUEST_BYTES, "SDK request projection exceeds its record bound")
        fingerprint.update(raw)
    return "sha256:" + fingerprint.hexdigest()


class PrivateRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def __call__(self, record):
        h.require(h.ID.fullmatch(record["runId"]) is not None, "invalid private record identity")
        raw = h.encoded(record) + b"\n"
        path = self.directory / (record["runId"] + ".jsonl")
        size = path.stat().st_size if path.exists() else 0
        h.require(size + len(raw) <= MAX_RECORD_BYTES, "private records exceed their bound")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(raw)


class RequestProbe:
    def __init__(self, *, run_id, mode, questions, record, secrets=()):
        h.require(mode in {"compact@1", "masking"}, "invalid controlled request mode")
        self.run_id, self.mode = run_id, mode
        self.questions, self.record = questions, record
        self.archive = (
            ContextArchive(secrets=secrets, skill_tools=frozenset()) if mode == "masking" else None
        )
        self.cursor = 0
        self.session_id = None
        self.case = None

    async def project(self, callback_context, request, *, generation=0):
        current_session = callback_context.session.id
        h.require(
            bool(current_session) and self.session_id in {None, current_session},
            "controlled probe belongs to another isolated session",
        )
        self.session_id = current_session
        events = callback_context.session.events
        h.require(len(events) >= self.cursor, "controlled session history changed")
        for event in events[self.cursor :]:
            if self.archive is not None:
                await self.archive.observe(event)
            for response in event.get_function_responses():
                data = response.response or {}
                artifact = data.get("artifact", {})
                if (
                    response.name == "read_text_artifact"
                    and artifact.get("namespace") == "inputs"
                    and artifact.get("name") == "source"
                    and isinstance(data.get("text"), str)
                ):
                    match = CASE_PREFIX.match(data["text"][:512])
                    if match and match[1] in CASES:
                        h.require(
                            self.case in {None, match[1]}, "controlled source identity changed"
                        )
                        self.case = match[1]
        self.cursor = len(events)
        masked = 0
        if self.archive is not None:
            h.require(
                self.archive.complete and self.archive.unsafe is None,
                "controlled history is unsafe or incomplete",
            )
            older = {identity for group in self.archive.groups[:-1] for identity in group.event_ids}
            responses = {
                response.id
                for event in events
                if event.id in older
                for response in event.get_function_responses()
            }
            contents = [content.model_copy(deep=True) for content in request.contents]
            for content in contents:
                for part in content.parts or []:
                    response = part.function_response
                    if (
                        response is not None
                        and response.id in responses
                        and response.name == "read_text_artifact"
                        and isinstance(response.response.get("text"), str)
                        and bool(response.response["text"])
                    ):
                        original = response.response
                        response.response = {
                            **original,
                            "text": MASK,
                            "maskedSha256": h.digest(h.encoded(original)),
                        }
                        masked += 1
            if masked:
                request.contents = contents
        active = (self.mode == "compact@1" and generation > 0) or masked > 0
        question = None
        if active:
            h.require(self.case in self.questions, "controlled source identity is unavailable")
            question = self.questions[self.case]
            # The request projection is not a persisted user event. Restate the
            # question on later tool turns, including after another checkpoint.
            request.contents = [
                *request.contents,
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text="Later question; include futureAnswer in the report: " + question
                        )
                    ],
                ),
            ]
        try:
            fingerprint = await to_thread_until_done(
                request_digest, request, name="context-eval-request-digest"
            )
            await to_thread_until_done(
                self.record,
                {
                    "runId": self.run_id,
                    "caseId": self.case,
                    "mode": self.mode,
                    "nativeGeneration": generation,
                    "maskedResponses": masked,
                    "questionSha256": h.digest(question.encode()) if question else None,
                    "sdkRequestSha256": fingerprint,
                    "projection": "model_request_content/after-agent-callback",
                    "recordedAtNs": time.time_ns(),
                },
                name="context-eval-private-record",
            )
        except Exception:
            # Optional evidence does not replace native failure/publication or
            # cleanup. The evaluator must treat missing records as unavailable.
            return


def controlled_worker(questions, record):
    class ControlledWorker(AdkWorkerRuntime):
        def __init__(self, context, model, **options):
            template = context.template_ref.template_id if context.template_ref else None
            mode = TEMPLATES.get(template)
            if mode is not None:
                h.require(
                    context.worker_session_mode == WorkerSessionMode.ISOLATED
                    and context.summarizer is None
                    and (context.compaction is not None) == (mode == "compact@1"),
                    "unsupported controlled Worker configuration",
                )
                h.require(
                    "read_text_artifact" in context.tools
                    and set(context.tools) <= {"read_text_artifact", "write_text_artifact"}
                    and not context.resolved_skills,
                    "controlled probes require the frozen text-only capability set",
                )
            super().__init__(context, model, **options)
            if mode is None:
                return
            probe = RequestProbe(
                run_id=context.run_id,
                mode=mode,
                questions=questions,
                record=record,
                secrets=runtime_secrets(context.runtime_settings),
            )
            prior = self._agent.before_model_callback

            async def project(*, callback_context, llm_request):
                if prior is not None:
                    result = await prior(callback_context=callback_context, llm_request=llm_request)
                    if result is not None:
                        return result
                generation = self._compaction.generation if self._compaction is not None else 0
                await probe.project(callback_context, llm_request, generation=generation)
                return None

            self._agent.before_model_callback = project

    return ControlledWorker


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--questions-sha256", required=True)
    parser.add_argument("--private-records", required=True)
    settings, runtime_args = parser.parse_known_args()
    questions = load_questions(settings.questions, settings.questions_sha256)
    from contractor_runtime import cli
    from contractor_runtime.worker import factory

    factory.AdkWorkerRuntime = controlled_worker(
        questions, PrivateRecorder(settings.private_records)
    )
    cli.main(runtime_args)


if __name__ == "__main__":
    try:
        main()
    except (h.Rejected, OSError):
        print("Controlled context-evaluation setup rejected", file=sys.stderr)
        sys.exit(2)
