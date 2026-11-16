"""Offline real-ADK regressions for the private controlled Runtime composition."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import controlled_runtime as c
import harness as h
import test_harness  # noqa: F401 -- installs the existing network-denying audit hook
from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import PrivateAttr

sys.path.insert(0, str(h.REPO / "runtime/tests"))
from fakes.model import scripted_model, text_result, tool_call
from fakes.worker_state import WorkerState
from test_adk_runtime import (
    CapturingScriptedLlm,
    assert_balanced_tool_calls,
    build_context,
    stage_request,
)
from test_worker_compaction import compact_context, summary, usage

from contractor_runtime.contracts import (
    AgentTemplateRef,
    SharedConfig,
    WorkerSessionMode,
    decode_private,
)
from contractor_runtime.llm.client import new_gateway_client
from contractor_runtime.llm.openai import OpenAICompatibleGatewayLlm
from contractor_runtime.shared_config import install_shared_config
from contractor_runtime.worker import factory

QUESTIONS = {name: "Which earlier receipt supports this conclusion?" for name in c.CASES}
QUESTION = QUESTIONS["source-review"]
SOURCE = '{"id":"source-review","inputs":{"fact":"EARLIER_FACT"}}'


def pair(number, text):
    return [
        Event(
            id=f"call-event-{number}",
            author="worker",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            id=f"call-{number}",
                            name="read_text_artifact",
                            args={"start_line": number},
                        )
                    )
                ],
            ),
        ),
        Event(
            id=f"reply-event-{number}",
            author="worker",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id=f"call-{number}",
                            name="read_text_artifact",
                            response={
                                "artifact": {
                                    "namespace": "inputs",
                                    "name": "source",
                                    "revision": "r1",
                                },
                                "text": text,
                            },
                        )
                    )
                ],
            ),
        ),
    ]


def request(events):
    return LlmRequest(
        model="offline",
        contents=[event.content for event in events],
        config=types.GenerateContentConfig(system_instruction="Keep exact evidence."),
    )


class ClosedModel(OpenAICompatibleGatewayLlm):
    """A real owned Gateway handle with local scripted responses and no I/O."""

    _script: Any = PrivateAttr()

    def __init__(self, *, model, responses):
        super().__init__(
            model=model,
            client_handle=new_gateway_client(
                base_url="https://offline.invalid/v1", api_key="offline-only-key", timeout_seconds=5
            ),
        )
        self._script = CapturingScriptedLlm(model=model, responses=responses)

    @property
    def payloads(self):
        return self._script.payloads

    async def generate_content_async(self, llm_request, stream=False):
        async for response in self._script.generate_content_async(llm_request, stream=stream):
            yield response


class ControlledRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = decode_private(
            SharedConfig, (h.REPO / "runtime/tests/testdata/shared-config.json").read_bytes()
        )
        install_shared_config(config.snapshot())

    def test_private_document_is_closed_bounded_and_hash_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            raw = h.encoded(
                {"schemaVersion": "context-compaction-probes/v1", "questions": QUESTIONS}
            )
            path.write_bytes(raw)
            self.assertEqual(c.load_questions(path, h.digest(raw)), QUESTIONS)
            with self.assertRaises(h.Rejected):
                c.load_questions(path, h.digest(b"other"))
            for bad in [
                {
                    "schemaVersion": "context-compaction-probes/v1",
                    "questions": QUESTIONS,
                    "answers": {},
                },
                {"schemaVersion": "context-compaction-probes/v1", "questions": {}},
                [],
            ]:
                data = h.encoded(bad)
                path.write_bytes(data)
                with self.assertRaises(h.Rejected):
                    c.load_questions(path, h.digest(data))
            path.write_bytes(b" " * (c.MAX_QUESTIONS_BYTES + 1))
            with self.assertRaises(h.Rejected):
                c.load_questions(path, h.digest(path.read_bytes()))

    def test_masks_complete_older_groups_without_mutating_session_or_receipts(self):
        async def scenario():
            records = []
            events = [*pair(1, SOURCE), *pair(2, "LATEST_FACT")]
            original = [event.model_dump(mode="json") for event in events]
            probe = c.RequestProbe(
                run_id="run-1", mode="masking", questions=QUESTIONS, record=records.append
            )
            value = request(events)
            await probe.project(
                SimpleNamespace(session=SimpleNamespace(id="session-1", events=events)), value
            )
            responses = [
                part.function_response
                for content in value.contents
                for part in content.parts or []
                if part.function_response
            ]
            self.assertEqual([r.id for r in responses], ["call-1", "call-2"])
            self.assertEqual(responses[0].response["text"], c.MASK)
            self.assertEqual(responses[1].response["text"], "LATEST_FACT")
            self.assertEqual(responses[0].response["artifact"]["revision"], "r1")
            self.assertTrue(h.DIGEST.fullmatch(responses[0].response["maskedSha256"]))
            self.assertEqual([event.model_dump(mode="json") for event in events], original)
            self.assertIn(QUESTION, value.contents[-1].parts[0].text)
            self.assertEqual(records[0]["maskedResponses"], 1)
            self.assertNotIn("EARLIER_FACT", json.dumps(records))
            self.assertNotIn(QUESTION, json.dumps(records))

        asyncio.run(scenario())

    def test_incomplete_history_does_not_mask_or_inject(self):
        async def scenario():
            events = pair(1, SOURCE)[:1]
            probe = c.RequestProbe(
                run_id="run-1", mode="masking", questions=QUESTIONS, record=lambda _: None
            )
            value = request(events)
            with self.assertRaises(h.Rejected):
                await probe.project(
                    SimpleNamespace(session=SimpleNamespace(id="session-1", events=events)), value
                )
            self.assertEqual(len(value.contents), 1)

        asyncio.run(scenario())

    def test_empty_old_observation_does_not_activate_a_later_question(self):
        async def scenario():
            events = [*pair(1, ""), *pair(2, SOURCE)]
            records = []
            probe = c.RequestProbe(
                run_id="run-1", mode="masking", questions=QUESTIONS, record=records.append
            )
            value = request(events)
            await probe.project(
                SimpleNamespace(session=SimpleNamespace(id="session-1", events=events)), value
            )
            self.assertNotIn(QUESTION, str(value))
            self.assertEqual(records[0]["maskedResponses"], 0)
            self.assertEqual(value.model_dump(), request(events).model_dump())

        asyncio.run(scenario())

    def test_checkpoint_question_is_late_restated_and_scoped_to_one_session(self):
        async def scenario():
            events = pair(1, SOURCE)
            context = SimpleNamespace(session=SimpleNamespace(id="session-1", events=events))
            probe = c.RequestProbe(
                run_id="run-1", mode="compact@1", questions=QUESTIONS, record=lambda _: None
            )
            first = request(events)
            await probe.project(context, first, generation=0)
            self.assertNotIn(QUESTION, str(first))
            for _ in range(2):
                later = request(events)
                await probe.project(context, later, generation=1)
                self.assertIn(QUESTION, str(later))
            context.session.id = "other-session"
            with self.assertRaises(h.Rejected):
                await probe.project(context, request(events), generation=1)

        asyncio.run(scenario())

    def test_native_worker_observes_masking_and_retains_safety_and_cleanup(self):
        for outcome in ["success", "gateway-error"]:
            for broken_recorder in [False, True]:
                with self.subTest(outcome=outcome, broken_recorder=broken_recorder):
                    asyncio.run(self.native_worker(outcome, broken_recorder))

    def test_ordinary_worker_retains_its_native_callbacks_and_session_policy(self):
        async def scenario():
            model = ClosedModel(model="offline", responses=[text_result("done")])
            with tempfile.TemporaryDirectory() as directory:
                context = build_context(Path(directory), WorkerState(), {})
                with patch.object(
                    factory,
                    "AdkWorkerRuntime",
                    c.controlled_worker(QUESTIONS, lambda _: self.fail("ordinary Worker recorded")),
                ):
                    runtime = await factory.AdkWorkerRuntimeFactory(lambda _: model).create(context)
                try:
                    self.assertIsNone(runtime._agent.before_model_callback)
                    result = await runtime.invoke(stage_request())
                    self.assertIsNone(result.failure)
                    self.assertEqual(result.result.result, "done")
                    self.assertNotIn(QUESTION, json.dumps(model.payloads))
                finally:
                    await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
                self.assertTrue(model.closed)

        asyncio.run(scenario())

    def test_unsupported_shared_control_joins_native_factory_rollback(self):
        async def scenario():
            async def read_text_artifact() -> dict:
                """Read the frozen source."""
                return {}

            model = ClosedModel(model="offline", responses=[])
            with tempfile.TemporaryDirectory() as directory:
                context = replace(
                    build_context(
                        Path(directory), WorkerState(), {"read_text_artifact": read_text_artifact}
                    ),
                    template_ref=AgentTemplateRef(
                        templateId="context-eval-masking", version="1", digest="sha256:" + "e" * 64
                    ),
                )
                with (
                    patch.object(
                        factory, "AdkWorkerRuntime", c.controlled_worker(QUESTIONS, lambda _: None)
                    ),
                    self.assertRaises(h.Rejected),
                ):
                    await factory.AdkWorkerRuntimeFactory(lambda _: model).create(context)
                self.assertTrue(model.closed)
                self.assertEqual(model.payloads, [])

        asyncio.run(scenario())

    def test_optional_projection_and_record_bounds_do_not_change_execution_input(self):
        async def scenario():
            events = pair(1, SOURCE)
            original = request(events)
            records = []
            probe = c.RequestProbe(
                run_id="run-1",
                mode="compact@1",
                questions=QUESTIONS,
                record=records.append,
            )
            with patch.object(c, "MAX_REQUEST_BYTES", 1):
                await probe.project(
                    SimpleNamespace(session=SimpleNamespace(id="session-1", events=events)),
                    original,
                )
            self.assertEqual(original.model_dump(), request(events).model_dump())
            self.assertEqual(records, [])

        asyncio.run(scenario())
        with tempfile.TemporaryDirectory() as directory:
            recorder = c.PrivateRecorder(directory)
            with patch.object(c, "MAX_RECORD_BYTES", 1), self.assertRaises(h.Rejected):
                recorder({"runId": "run-1", "sdkRequestSha256": "sha256:" + "a" * 64})
            self.assertEqual(list(Path(directory).iterdir()), [])

    async def native_worker(self, outcome, broken_recorder):
        records, effects = [], []

        def record(row):
            if broken_recorder:
                raise OSError("private recorder unavailable")
            records.append(row)

        async def read_text_artifact(start_line: int = 1) -> dict:
            """Read a frozen evidence page."""
            effects.append(start_line)
            return {
                "artifact": {"namespace": "inputs", "name": "source", "revision": "r1"},
                "text": SOURCE if start_line == 1 else f"PAGE_{start_line}",
            }

        responses = [
            tool_call("read_text_artifact", {"start_line": n}, call_id=f"call-{n}")
            for n in [1, 2, 3]
        ]
        if outcome == "gateway-error":
            responses[2].error_code = "SAFETY"
            responses[2].error_message = "Model rejected this response"
        else:
            responses.append(text_result("done"))
        model = ClosedModel(model="offline", responses=responses)
        with tempfile.TemporaryDirectory() as directory:
            state = WorkerState()
            context = replace(
                build_context(Path(directory), state, {"read_text_artifact": read_text_artifact}),
                worker_session_mode=WorkerSessionMode.ISOLATED,
                template_ref=AgentTemplateRef(
                    templateId="context-eval-masking", version="1", digest="sha256:" + "e" * 64
                ),
            )
            with patch.object(factory, "AdkWorkerRuntime", c.controlled_worker(QUESTIONS, record)):
                runtime = await factory.AdkWorkerRuntimeFactory(lambda _: model).create(context)
            try:
                result = await runtime.invoke(stage_request())
                if outcome == "success":
                    self.assertIsNone(result.failure)
                    self.assertEqual(result.result.result, "done")
                    self.assertEqual(effects, [1, 2, 3])
                else:
                    self.assertIsNone(result.result)
                    self.assertEqual(result.failure.code, "worker_execution_failed")
                    self.assertEqual(effects, [1, 2])
                for payload in model.payloads[:3]:
                    assert_balanced_tool_calls(payload)
                self.assertNotIn(QUESTION, json.dumps(model.payloads[:2]))
                self.assertIn(QUESTION, json.dumps(model.payloads[2]))
                self.assertIn(c.MASK, json.dumps(model.payloads[2]))
                self.assertEqual(len(records) == 0, broken_recorder)
                self.assertNotIn(QUESTION, json.dumps(records))
            finally:
                await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
            self.assertTrue(model.closed)

    def test_native_compactor_finishes_before_the_later_question_is_disclosed(self):
        async def scenario():
            async def read_text_artifact(start_line: int = 1) -> dict:
                """Read a frozen evidence page."""
                return {
                    "artifact": {"namespace": "inputs", "name": "source", "revision": "r1"},
                    "text": (SOURCE if start_line == 1 else "SECOND_FACT") + "x" * 2048,
                }

            model = ClosedModel(
                model="worker-model",
                responses=[
                    usage(tool_call("read_text_artifact", {"start_line": 1}, call_id="first"), 100),
                    usage(
                        tool_call("read_text_artifact", {"start_line": 2}, call_id="second"), 500
                    ),
                    usage(text_result("done"), 100),
                ],
            )
            compactor = scripted_model(
                [summary("First source page was read.")],
                model="worker-compactor-model",
                auto_result_finalizer=False,
            )
            with tempfile.TemporaryDirectory() as directory:
                context = replace(
                    compact_context(
                        Path(directory),
                        WorkerState(),
                        {"read_text_artifact": read_text_artifact},
                        max_tool_calls=2,
                    ),
                    template_ref=AgentTemplateRef(
                        templateId="context-eval-controlled-compact",
                        version="1",
                        digest="sha256:" + "e" * 64,
                    ),
                )

                def select(build):
                    return (
                        compactor if build.model_policy.model == "worker-compactor-model" else model
                    )

                with patch.object(
                    factory, "AdkWorkerRuntime", c.controlled_worker(QUESTIONS, lambda _: None)
                ):
                    runtime = await factory.AdkWorkerRuntimeFactory(select).create(context)
                try:
                    result = await runtime.invoke(stage_request())
                    self.assertIsNone(result.failure)
                    self.assertEqual(result.result.result, "done")
                    self.assertEqual(len(compactor.requests), 1)
                    self.assertNotIn(QUESTION, compactor.requests[0]["contentText"])
                    self.assertNotIn(QUESTION, json.dumps(model.payloads[:2]))
                    self.assertIn(QUESTION, json.dumps(model.payloads[2]))
                finally:
                    await runtime.finalize(datetime.now(UTC) + timedelta(seconds=10))
                self.assertTrue(model.closed)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
