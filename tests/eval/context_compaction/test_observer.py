"""Evidence gaps and before-checkpoint activity must not become successful probes."""

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import observer
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Status

from contractor_runtime.adapters.content import MAX_CONTENT_BYTES, encode_content


class ObserverTests(unittest.TestCase):
    def test_actual_isolated_consumer_entry_uses_the_pinned_sibling_client(self):
        diagnostics = {
            "status": "available",
            "complete": True,
            "adapters": [
                {
                    "ref": "otlp-http@1",
                    "flushSucceeded": True,
                    "failedOperations": "0",
                    "droppedSpans": {"queueOverflow": "0"},
                }
            ],
        }
        responses = {
            "/v1/runs/run-test": {
                "state": "succeeded",
                "deletable": True,
                "attempts": [{"stageExecutionId": "stage-test"}],
            },
            "/v1/runs/run-test/attempts/stage-test/runtime-diagnostics?limit=10": {
                "items": [diagnostics],
                "page": {"hasMore": False},
            },
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                raw = json.dumps(responses[self.path]).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder, tool_start=31)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                with patch.dict(
                    os.environ,
                    {
                        "CONTRACTOR_CONTEXT_EVAL_API": f"http://127.0.0.1:{server.server_port}",
                        "CONTRACTOR_CONTEXT_EVAL_TOKEN": "test-only-session",
                        "CONTRACTOR_CONTEXT_EVAL_EVIDENCE": folder,
                    },
                ):
                    result = observer.h.run_observer(
                        str(Path(observer.__file__).resolve()),
                        "run-test",
                        observer.h.fixtures()[1]["source-review"],
                        timeout=10,
                    )
                self.assertEqual(result["retrieved"], ["obs-P17"])
                self.assertEqual(result["usefulAfterCheckpoint"], 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def fixture(
        self,
        folder,
        *,
        tool_start=10,
        failed=False,
        truncated=False,
        paginated=False,
        phase=None,
        alias="configured/summarizer-model",
    ):
        request = ExportTraceServiceRequest()
        resource = request.resource_spans.add()
        label = resource.resource.attributes.add(key="contractor.run.id")
        label.value.string_value = "run-test"
        scope = resource.scope_spans.add()
        for index, (name, start, end, attributes) in enumerate(
            [
                (
                    "contractor.worker.model",
                    20,
                    30,
                    {"model.alias": alias, **({"model.phase": phase} if phase else {})},
                ),
                (
                    "contractor.worker.tool",
                    tool_start,
                    tool_start + 1,
                    {
                        "tool.name": "read_artifact",
                        "contractor.input": "{}",
                        "contractor.output": encode_content(
                            {"text": "x" * (MAX_CONTENT_BYTES + 100)}
                            if truncated
                            else {"text": '"receipt": "obs-P17"', "truncated": paginated}
                        ),
                    },
                ),
            ]
        ):
            span = scope.spans.add(
                trace_id=b"t" * 16,
                span_id=bytes([index + 1]) * 8,
                name=name,
                start_time_unix_nano=start,
                end_time_unix_nano=end,
                status=Status(
                    code=Status.STATUS_CODE_ERROR if failed and index else Status.STATUS_CODE_OK
                ),
            )
            for key, value in attributes.items():
                span.attributes.add(key=key).value.string_value = value
        path = Path(folder) / "traces/run-test/1.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(MessageToDict(request)))
        return path

    def test_useful_continuation_requires_a_successful_later_tool(self):
        for start, failed, expected in [(10, False, 0), (31, False, 1), (31, True, 0)]:
            with self.subTest(start=start, failed=failed), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder, tool_start=start, failed=failed)
                result = observer.observe(folder, "run-test")
                self.assertEqual(result["usefulAfterCheckpoint"], expected)
                self.assertEqual(result["retrieved"], [] if failed else ["obs-P17"])

    def test_actual_mask_activation_can_supply_the_checkpoint_time(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder, tool_start=10)
            path = Path(folder) / "projections/run-test.jsonl"
            path.parent.mkdir()
            path.write_text(
                json.dumps(
                    {
                        "runId": "run-test",
                        "questionSha256": "sha256:" + "1" * 64,
                        "nativeGeneration": 0,
                        "maskedResponses": 1,
                        "recordedAtNs": 5,
                    }
                )
                + "\n"
            )
            self.assertEqual(observer.observe(folder, "run-test")["usefulAfterCheckpoint"], 1)

    def test_native_compactor_phase_is_independent_of_the_configured_model_alias(self):
        for start, failed, expected in [(10, False, 0), (31, False, 1), (31, True, 0)]:
            with self.subTest(start=start, failed=failed), tempfile.TemporaryDirectory() as folder:
                self.fixture(
                    folder,
                    tool_start=start,
                    failed=failed,
                    phase="compaction",
                    alias="configured/another-model",
                )
                self.assertEqual(
                    observer.observe(folder, "run-test")["usefulAfterCheckpoint"], expected
                )

    def test_explicit_other_model_phases_are_not_checkpoints_through_an_alias(self):
        for phase in ["main", "finalization"]:
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                self.fixture(folder, tool_start=31, phase=phase)
                self.assertEqual(observer.observe(folder, "run-test")["usefulAfterCheckpoint"], 0)

    def test_paginated_artifact_read_is_complete_telemetry_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder, tool_start=31, paginated=True)
            result = observer.observe(folder, "run-test")
            self.assertEqual(result["usefulAfterCheckpoint"], 1)
            self.assertEqual(result["retrieved"], ["obs-P17"])

    def test_truncated_write_arguments_are_unavailable_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder, tool_start=31)
            document = json.loads(path.read_text())
            tool = document["resourceSpans"][0]["scopeSpans"][0]["spans"][1]
            values = {item["key"]: item["value"] for item in tool["attributes"]}
            values["tool.name"]["stringValue"] = "write_text_artifact"
            values["contractor.input"]["stringValue"] = encode_content(
                {"name": "progress", "text": "x" * (MAX_CONTENT_BYTES + 100)}
            )
            values["contractor.output"]["stringValue"] = json.dumps(
                {"artifact": {"namespace": "builder", "name": "progress"}}
            )
            path.write_text(json.dumps(document))
            with self.assertRaises(ValueError):
                observer.observe(folder, "run-test")

    def test_native_export_gaps_cannot_be_reported_as_no_activity(self):
        import copy

        complete = [
            {
                "status": "available",
                "complete": True,
                "adapters": [
                    {
                        "ref": "otlp-http@1",
                        "flushSucceeded": True,
                        "failedOperations": "0",
                        "droppedSpans": {"queueOverflow": "0", "encodingFailed": "0"},
                    }
                ],
            }
        ]
        observer.verify_export(complete)
        for mutate in [
            lambda d: d[0].update(complete=False),
            lambda d: d[0]["adapters"][0].update(flushSucceeded=False),
            lambda d: d[0]["adapters"][0]["droppedSpans"].update(queueOverflow="1"),
            lambda d: d[0]["adapters"][0].update(failedOperations="1"),
        ]:
            changed = copy.deepcopy(complete)
            mutate(changed)
            with self.assertRaises(ValueError):
                observer.verify_export(changed)
        with self.assertRaises(ValueError):
            observer.verify_export([])

    def test_missing_foreign_and_truncated_evidence_is_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                observer.observe(folder, "run-test")
            path = self.fixture(folder, truncated=True)
            with self.assertRaises(ValueError):
                observer.observe(folder, "run-test")
            path.write_text(path.read_text().replace('"run-test"', '"run-foreign"'))
            with self.assertRaises(ValueError):
                observer.observe(folder, "run-test")
