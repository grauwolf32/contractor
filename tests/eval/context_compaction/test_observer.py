"""Evidence gaps and before-checkpoint activity must not become successful probes."""

import json
import tempfile
import unittest
from pathlib import Path

import observer
from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.proto.trace.v1.trace_pb2 import Status


class ObserverTests(unittest.TestCase):
    def fixture(self, folder, *, tool_start=10, failed=False, truncated=False):
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
                    {"model.alias": "configured/summarizer-model"},
                ),
                (
                    "contractor.worker.tool",
                    tool_start,
                    tool_start + 1,
                    {
                        "tool.name": "read_text_artifact",
                        "contractor.input": "{}",
                        "contractor.output": json.dumps(
                            {"text": '"receipt": "obs-P17"', "truncated": truncated}
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
