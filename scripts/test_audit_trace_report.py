"""Exercise the CLI over a real HTTP socket using Phoenix's native span format."""

import contextlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import audit_trace_report as report

AUDIT = "audit_test"
TRACE = "a" * 32
PLANNER_TRACE = "b" * 32
FOREIGN_TRACE = "c" * 32
CANARY = "CAPTURED-CONTENT-MUST-NOT-REACH-STDOUT"
PROMPT_CANARY = "PROMPT-CONTENT-MUST-NOT-REACH-AGGREGATES"


def span(index, name=report.TOOL, trace_id=TRACE, outcome="succeeded", **attrs):
    return {
        "id": "global-" + str(index),
        "name": name,
        "context": {"trace_id": trace_id, "span_id": f"{index:016x}"},
        "parent_id": None,
        "span_kind": "UNKNOWN",
        "start_time": "2026-11-20T10:00:00+03:00",
        "end_time": "2026-11-20T07:00:01Z",
        "status_code": "OK" if outcome == "succeeded" else "ERROR",
        "status_message": "",
        "attributes": {"outcome": outcome, "duration.ms": index * 10, **attrs},
        "events": [],
    }


def fixture():
    values = [
        span(
            10,
            report.LIFECYCLE,
            **{
                report.AUDIT_LABEL: AUDIT,
                report.TASK_LABEL: "task_test",
                "counts.tool_calls": 5,
                "counts.model_calls": 2,
            },
        ),
        span(
            11,
            "contractor.planner.invocation",
            PLANNER_TRACE,
            **{report.AUDIT_LABEL: AUDIT},
        ),
        span(
            1,
            **{
                "tool.name": "read_file",
                "contractor.output": json.dumps({"data": CANARY}),
            },
        ),
        span(
            2,
            outcome="failed",
            **{
                "tool.name": "read_file",
                "contractor.output": json.dumps(
                    {
                        "error": {
                            "code": "not_found",
                            "message": "=" + CANARY,
                            "retryable": False,
                        },
                    }
                ),
            },
        ),
        span(
            3,
            outcome="cancelled",
            **{"tool.name": "http_request", "error.type": "CancelledError"},
        ),
        span(
            4,
            **{
                "tool.name": "http_request",
                "contractor.output": '{"truncated":true,"preview":"x"}',
            },
        ),
        span(
            5,
            outcome="unrecognized",
            **{"tool.name": "http_request", "contractor.output": "bad-json"},
        ),
        span(
            6,
            "contractor.worker.model",
            **{
                "gen_ai.request.model": "configured-model",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 2,
                "gen_ai.usage.cache_read.input_tokens": 4,
                "gen_ai.input.messages": json.dumps(
                    {"tool_calls": [{"name": "not_a_real_call"}], "input": PROMPT_CANARY}
                ),
            },
        ),
        span(7, "contractor.worker.model", **{"gen_ai.request.model": "configured-model"}),
        span(
            8,
            "contractor.planner.model",
            PLANNER_TRACE,
            **{"gen_ai.request.model": "configured-model"},
        ),
        span(12, report.LIFECYCLE, FOREIGN_TRACE, **{report.AUDIT_LABEL: "another_audit"}),
    ]
    values[6]["status_code"] = "UNSET"
    return values


class PhoenixFixture:
    def __init__(self, spans=None, mode="normal"):
        self.spans = fixture() if spans is None else spans
        self.mode, self.requests = mode, []

    def __enter__(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                owner.requests.append((self.path, self.headers.get("Authorization")))
                query = parse_qs(urlsplit(self.path).query)
                if owner.mode == "broken_http":
                    self.wfile.write((CANARY + "\r\n\r\n").encode())
                    self.close_connection = True
                    return
                if owner.mode == "http_error":
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(CANARY.encode())
                    return
                if owner.mode == "redirect":
                    self.send_response(302)
                    self.send_header("Location", "/leaked")
                    self.end_headers()
                    return
                if "attribute" in query:
                    key, value = query["attribute"][0].split(":", 1)
                    value = json.loads(value)
                    rows = [s for s in owner.spans if s["attributes"].get(key) == value]
                    if owner.mode == "foreign_audit":
                        rows = owner.spans[-1:]
                else:
                    rows = [
                        s for s in owner.spans if s["context"]["trace_id"] == query["trace_id"][0]
                    ]
                    # Simulate overlapping pages while keeping the same immutable span identity.
                    rows += rows[:1]
                    if owner.mode == "foreign_trace":
                        rows = owner.spans[-1:]
                offset, limit = int(query.get("cursor", [0])[0]), int(query["limit"][0])
                page = rows[offset : offset + limit]
                next_cursor = str(offset + limit) if offset + limit < len(rows) else None
                if owner.mode == "repeated_cursor":
                    next_cursor = "0"
                if owner.mode == "empty_cursor":
                    page, next_cursor = [], "1"
                body = json.dumps({"data": page, "next_cursor": next_cursor}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = "http://127.0.0.1:" + str(self.server.server_port)
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class AuditTraceReportTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.out = self.directory / "export"

    def run_cli(self, extra, *, output=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = report.main([AUDIT, "--output", str(output or self.out), *extra])
        self.assertNotIn(CANARY, stdout.getvalue() + stderr.getvalue())
        return code, stdout.getvalue(), stderr.getvalue()

    def summary(self, directory=None):
        return json.loads(((directory or self.out) / "summary.json").read_text())

    def assert_incomplete(self):
        self.assertEqual(
            json.loads((self.out / "manifest.json").read_text())["state"], "incomplete"
        )
        self.assertFalse((self.out / "report.md").exists())

    def test_real_cli_paginates_whole_traces_and_offline_replay_matches(self):
        with PhoenixFixture() as server:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(report.__file__).resolve()),
                    AUDIT,
                    "--phoenix-url",
                    server.url,
                    "--page-size",
                    "2",
                    "--output",
                    str(self.out),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertNotIn(CANARY, process.stdout + process.stderr)
        self.assertGreater(len(server.requests), 5)
        queries = [parse_qs(urlsplit(path).query) for path, _ in server.requests]
        self.assertTrue(any(q.get("trace_id") == [TRACE] and "attribute" not in q for q in queries))
        result = self.summary()
        self.assertEqual(result["trace_count"], 2)
        self.assertEqual(result["span_count"], 10)
        counts = result["tool_calls"]
        self.assertEqual(
            (counts["calls"], counts["errors"], counts["cancelled"], counts["unknown"]),
            (5, 1, 1, 1),
        )
        self.assertEqual(
            (counts["total_ms"], counts["mean_ms"], counts["p50_ms"], counts["p95_ms"]),
            (150, 30, 30, 50),
        )
        self.assertEqual(result["traces_with_counter_mismatch"], 0)
        self.assertEqual(result["models"][0]["input_tokens"], 10)
        self.assertEqual(result["models"][0]["input_tokens_measured_calls"], 1)
        self.assertEqual(result["models"][0]["cached_input_tokens"], 4)
        self.assertEqual(
            result["tool_output_coverage"],
            {"available": 2, "malformed": 1, "missing": 1, "truncated": 1},
        )
        self.assertEqual(result["observed_start_time"], "2026-11-20T07:00:00+00:00")
        self.assertIn(CANARY, (self.out / "spans.jsonl").read_text())
        self.assertIn("'=" + CANARY, (self.out / "errors.csv").read_text())
        self.assertNotIn(PROMPT_CANARY, (self.out / "report.md").read_text())
        self.assertNotIn(PROMPT_CANARY, (self.out / "summary.json").read_text())
        self.assertIn(CANARY, (self.out / "report.md").read_text())
        self.assertEqual(stat.S_IMODE(self.out.stat().st_mode), 0o700)
        for path in self.out.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        replay = self.directory / "offline"
        self.assertEqual(
            self.run_cli(["--input", str(self.out / "spans.jsonl")], output=replay)[0],
            0,
        )
        self.assertEqual(self.summary(replay), result)

    def test_no_traces_returns_unavailable_counts_and_exit_three(self):
        with PhoenixFixture([]) as server:
            self.assertEqual(self.run_cli(["--phoenix-url", server.url])[0], 3)
        self.assertIsNone(self.summary()["tool_calls"])
        self.assertFalse(self.summary()["traces_found"])
        self.assertIn("No labelled traces", (self.out / "report.md").read_text())

    def test_error_outcomes_cursor_and_filter_violations_fail_safely(self):
        for mode in (
            "http_error",
            "broken_http",
            "redirect",
            "repeated_cursor",
            "empty_cursor",
            "foreign_audit",
            "foreign_trace",
        ):
            with self.subTest(mode=mode), PhoenixFixture(mode=mode) as server:
                self.out = self.directory / mode
                with mock.patch.dict(os.environ, {"TEST_PHOENIX_KEY": CANARY}):
                    code, _, _ = self.run_cli(
                        [
                            "--phoenix-url",
                            server.url,
                            "--page-size",
                            "2",
                            "--api-key-env",
                            "TEST_PHOENIX_KEY",
                        ]
                    )
                self.assertEqual(code, 1)
                self.assert_incomplete()
                self.assertTrue(all(auth == "Bearer " + CANARY for _, auth in server.requests))
                if mode in ("http_error", "broken_http", "redirect"):
                    self.assertEqual(len(server.requests), 1)

    def test_limits_bound_rows_traces_bytes_and_http_response(self):
        for flag, value in (
            ("--max-spans", "2"),
            ("--max-traces", "1"),
            ("--max-bytes", "20"),
        ):
            with self.subTest(flag=flag), PhoenixFixture() as server:
                self.out = self.directory / flag[2:]
                self.assertEqual(self.run_cli(["--phoenix-url", server.url, flag, value])[0], 1)
                self.assert_incomplete()
        with PhoenixFixture() as server, mock.patch.object(report, "BYTE_LIMIT", 100):
            self.out = self.directory / "response-limit"
            self.assertEqual(self.run_cli(["--phoenix-url", server.url])[0], 1)
            self.assert_incomplete()

    def test_counter_mismatch_and_absent_counters_are_explicit(self):
        for expected, mismatch, missing in ((6, 1, 0), (None, 0, 1)):
            values = fixture()
            values[0]["attributes"]["counts.tool_calls"] = expected
            with self.subTest(expected=expected), PhoenixFixture(values) as server:
                self.out = self.directory / str(expected)
                self.assertEqual(self.run_cli(["--phoenix-url", server.url])[0], 0)
            self.assertEqual(self.summary()["traces_with_counter_mismatch"], mismatch)
            self.assertEqual(self.summary()["worker_traces_without_counter_comparison"], missing)
            self.assertEqual(self.summary()["tool_calls"]["calls"], 5)

    def test_large_audit_counts_every_page(self):
        values = [
            span(
                6000,
                report.LIFECYCLE,
                **{
                    report.AUDIT_LABEL: AUDIT,
                    "counts.tool_calls": 5000,
                    "counts.model_calls": 0,
                },
            )
        ] + [span(i, **{"tool.name": "read_file"}) for i in range(1, 5001)]
        with PhoenixFixture(values) as server:
            self.assertEqual(
                self.run_cli(["--phoenix-url", server.url, "--page-size", "200"])[0], 0
            )
        self.assertEqual(self.summary()["tool_calls"]["calls"], 5000)
        self.assertEqual(self.summary()["span_count"], 5001)
        self.assertEqual(self.summary()["traces_with_counter_mismatch"], 0)
        self.assertGreater(len(server.requests), 25)

    def test_existing_destination_is_never_modified(self):
        self.out.mkdir()
        marker = self.out / "manifest.json"
        marker.write_text("keep")
        self.assertEqual(self.run_cli([])[0], 1)
        self.assertEqual(marker.read_text(), "keep")

    def test_bad_offline_data_and_conflicting_duplicates_fail(self):
        root = fixture()[0]
        bad = span(1, outcome=[])
        for name, raw in (
            ("malformed", '{"name":'),
            ("duplicate_json", '{"x":1,"x":2}'),
            ("bad_span", json.dumps(bad)),
            ("foreign", json.dumps(span(1))),
            (
                "duplicate_id",
                json.dumps(root) + "\n" + json.dumps({**root, "name": "changed"}),
            ),
        ):
            with self.subTest(name=name):
                source = self.directory / (name + ".jsonl")
                source.write_text(raw + "\n")
                self.out = self.directory / name
                self.assertEqual(self.run_cli(["--input", str(source)])[0], 1)
                self.assert_incomplete()

    def test_returned_error_shapes_and_exception_type_are_distinct(self):
        for payload, code, message in (
            ({"error_code": "bad", "message": "reason"}, "bad", "reason"),
            ({"error": "plain reason"}, "", "plain reason"),
        ):
            row = report.compact_span(
                span(
                    1,
                    outcome="failed",
                    **{
                        "contractor.output": json.dumps(payload),
                        "error.type": "ValueError",
                    },
                )
            )
            self.assertEqual(row["error_code"], code)
            self.assertEqual(row["error_message"], message)
            self.assertEqual(row["error_type"], "ValueError")

    def test_deadline_is_injected_and_interrupt_leaves_incomplete_manifest(self):
        args = report.parse_args([AUDIT, "--deadline", "1"])
        clock = mock.Mock(side_effect=[0, 0.5, 1])
        budget = report.Budget(args, clock=clock)
        self.assertEqual(budget.remaining(), 0.5)
        with self.assertRaisesRegex(report.ReportError, "deadline"):
            budget.remaining()
        with mock.patch.object(report, "export_phoenix", side_effect=KeyboardInterrupt):
            self.assertEqual(self.run_cli([])[0], 130)
        self.assert_incomplete()

    def test_urls_reject_embedded_secrets_and_redirects(self):
        args = report.parse_args([AUDIT, "--phoenix-url", "http://user:secret@localhost:6006"])
        with self.assertRaises(report.ReportError):
            report.Phoenix(args, report.Budget(args))


if __name__ == "__main__":
    unittest.main()
