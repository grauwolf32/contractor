"""The informational sampler must reject false or unsafe measurement evidence."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import measure_adk_runtime as measure


class MeasurementEvidenceTest(unittest.TestCase):
    def selected(self):
        return argparse.Namespace(
            turns=3,
            width=2,
            subtasks=2,
            mode="shared",
            ceilings={
                "state_bytes": 4194304,
                "tool_records": 1000,
                "report_bytes": 1048576,
            },
        )

    def raw(self):
        callbacks = {
            "before_run_callback": 2,
            "before_model_callback": 6,
            "after_model_callback": 6,
            "before_tool_callback": 8,
            "after_tool_callback": 8,
        }
        return {
            "turns": 3,
            "width": 2,
            "mode": "shared",
            "subtasks": 2,
            "logical_ids": ["private-logical-a", "private-logical-b"],
            "effects": [
                (owner, index, "private-logical-" + ("a" if owner == 0 else "b"))
                for owner in range(2)
                for index in range(4)
            ],
            "starts": [0, 4],
            "finishes": [4, 8],
            "work": {
                "created_session_ids": ["private-session"],
                "session_ids": ["private-session"] * 2,
                "model_calls": 6,
                "finalizer_calls": 2,
                "callback_calls": callbacks,
                "callback_seconds": {key: 0.01 for key in callbacks},
                "request_bytes": 80,
                "max_request_bytes": 10,
                "provider_seconds": 0.01,
                "construction_seconds": 0.01,
            },
            "request_sizes": [10] * 8,
            "snapshot_bytes": [100, 200],
            "retained_tool_records": 8,
            "reported_tool_calls": 8,
            "retained_first_call_id": "tool-00000001",
            "retained_last_call_id": "tool-00000008",
            "detail_truncated": False,
            "report_bytes": 300,
            "elapsed_seconds": 0.2,
        }

    def test_export_discards_logical_session_and_retained_call_ids(self):
        safe = measure.counters(self.raw(), self.selected())
        text = json.dumps(safe)
        self.assertNotIn("private-", text)
        self.assertNotIn("tool-000000", text)
        self.assertEqual(safe["effects"], 8)
        self.assertEqual(safe["retained_call_ordinals"], [1, 8])

    def test_lost_or_foreign_effect_identity_cannot_count_as_measured(self):
        for change in ("missing", "foreign"):
            with self.subTest(change=change):
                raw = self.raw()
                if change == "missing":
                    raw["effects"].pop()
                else:
                    raw["effects"][0] = (0, 0, "different-logical-owner")
                with self.assertRaises(ValueError):
                    measure.counters(raw, self.selected())

    def test_invented_callback_charge_and_payload_counter_fail_closed(self):
        raw = self.raw()
        raw["work"]["callback_calls"]["before_model_callback"] += 1
        with self.assertRaises(ValueError):
            measure.counters(raw, self.selected())
        raw = self.raw()
        raw["work"]["unapproved_payload"] = "private-payload"
        with self.assertRaises(ValueError):
            measure.counters(raw, self.selected())

    def test_source_root_guard_rejects_an_import_from_outside_repository(self):
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            measure.source("json", Path(folder))

    def test_parallel_callback_aggregate_is_not_subtracted_as_wall_time(self):
        raw = self.raw()
        raw["work"]["callback_seconds"] = {
            name: 100.0 for name in raw["work"]["callback_seconds"]
        }
        result = measure.counters(raw, self.selected())
        self.assertGreater(
            result["callback_aggregate_seconds_non_disjoint"], result["elapsed_seconds"]
        )
        self.assertAlmostEqual(result["framework_seconds_excluding_mock_handler"], 0.19)

    def test_network_guard_denies_internet_before_connecting(self):
        with (
            mock.patch.object(measure.socket.socket, "connect") as connect,
            mock.patch.object(measure.socket.socket, "connect_ex"),
            mock.patch.object(measure.socket, "getaddrinfo"),
            mock.patch.object(measure.socket, "gethostbyname"),
            mock.patch.object(measure.socket, "gethostbyname_ex"),
        ):
            measure.disable_network()
            with measure.socket.socket() as stream, self.assertRaises(ValueError):
                stream.connect(("127.0.0.1", 1))
            with self.assertRaises(RuntimeError):
                measure.socket.getaddrinfo("example.test", 443)
            connect.assert_not_called()

    def test_cli_rejects_zero_samples_before_importing_framework(self):
        quiet = io.StringIO()
        with contextlib.redirect_stderr(quiet), self.assertRaises(SystemExit) as error:
            measure.arguments(
                [
                    "--repository",
                    "/unused",
                    "--fixture",
                    "/unused",
                    "--workloads",
                    "/unused",
                    "--turns",
                    "50",
                    "--mode",
                    "shared",
                    "--samples",
                    "0",
                ]
            )
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
