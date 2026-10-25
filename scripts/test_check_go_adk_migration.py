"""A migration cannot pass through absent cases, waived failures or unsafe adaptation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_go_adk_migration as migration


class GoADKReportValidationTest(unittest.TestCase):
    def manifest(self):
        return {
            "schema": ["module/package:TestSchema/profile"],
            "lifecycle": ["module/package:TestJoined"],
        }

    def events(self):
        rows = []
        for cases in self.manifest().values():
            for value in cases:
                package, name = value.split(":", 1)
                rows += [
                    {"Action": "run", "Package": package, "Test": name},
                    {"Action": "pass", "Package": package, "Test": name},
                ]
        rows.append({"Action": "pass", "Package": "module/package"})
        return rows

    def test_each_group_requires_started_and_passed_cases_and_package(self):
        result = migration.validate_cases(self.events(), self.manifest())
        self.assertEqual(result["groups"]["schema"]["passed"], 1)
        self.assertEqual(result["executed_leaf_cases"], 2)
        for removed in range(len(self.events())):
            with (
                self.subTest(removed=removed),
                self.assertRaises(migration.MigrationError),
            ):
                migration.validate_cases(
                    self.events()[:removed] + self.events()[removed + 1 :],
                    self.manifest(),
                )

    def test_empty_group_and_zero_total_cannot_pass(self):
        for manifest in ({}, {"schema": []}, {"schema": ["p:t"], "lifecycle": []}):
            with (
                self.subTest(manifest=manifest),
                self.assertRaises(migration.MigrationError),
            ):
                migration.validate_cases([], manifest)

    def test_skipped_failed_or_unknown_skip_fails_even_with_a_forged_pass(self):
        for action, name in (
            ("skip", "TestSchema/profile"),
            ("fail", "TestJoined"),
            ("skip", "TestUnexpected"),
        ):
            rows = self.events() + [
                {"Action": action, "Package": "module/package", "Test": name}
            ]
            with (
                self.subTest(action=action, name=name),
                self.assertRaises(migration.MigrationError),
            ):
                migration.validate_cases(rows, self.manifest())

    def test_optional_live_skip_is_named_and_recorded(self):
        name = next(iter(migration.OPTIONAL_SKIPS))
        package, test = name.split(":", 1)
        rows = self.events() + [{"Action": "skip", "Package": package, "Test": test}]
        result = migration.validate_cases(rows, self.manifest())
        self.assertEqual(result["optional_skips"][0]["case"], name)

    def test_expected_compile_failure_requires_all_api_breaks_and_nonzero_exit(self):
        diagnostic = "undefined: agent.ToolContext\nundefined: agent.CallbackContext\nnot enough arguments in call to adksession.NewEvent"
        self.assertTrue(all(migration.expected_compile_failure(1, diagnostic).values()))
        for code, text in (
            (0, diagnostic),
            (124, diagnostic),
            (1, "network download unavailable"),
            (1, "undefined: agent.ToolContext"),
        ):
            with (
                self.subTest(code=code, text=text),
                self.assertRaises(migration.MigrationError),
            ):
                migration.expected_compile_failure(code, text)

    def test_invalid_report_identity_is_not_a_case(self):
        for value in (
            {"Action": 1},
            {"Action": "pass", "Package": "module/package", "Test": 4},
        ):
            with self.subTest(value=value), self.assertRaises(migration.MigrationError):
                migration.validate_cases([value], self.manifest())

    def test_import_inventory_is_dynamic_and_only_rewrites_import_literals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "internal/example/test.go"
            path.parent.mkdir(parents=True)
            path.write_text(
                'package example\nimport (\n "google.golang.org/adk/agent"\n)\nconst prompt = "google.golang.org/adk/model"\n'
            )
            sibling = path.with_name("payload.go")
            sibling.write_text(
                'package example\nconst payload = "google.golang.org/adk/model"\n'
            )
            self.assertEqual(
                migration.import_inventory(root), ["internal/example/test.go"]
            )
            migration.mechanical_imports(root)
            self.assertIn('"google.golang.org/adk/v2/agent"', path.read_text())
            self.assertIn(
                'const prompt = "google.golang.org/adk/model"', path.read_text()
            )

    def test_mechanical_contexts_use_the_actual_history_helper_context(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "internal/planner/session/service_test.go"
            history = (
                root
                / "internal/planner/streamline/go_adk_history_characterization_test.go"
            )
            old.parent.mkdir(parents=True)
            history.parent.mkdir(parents=True)
            old.write_text(
                'package session\nvar _ agent.ToolContext\nvar _ agent.CallbackContext\nfunc test() { session.NewEvent("seed") }\n'
            )
            history.write_text(
                'package streamline\nfunc test(ctx context.Context) { adksession.NewEvent("seed") }\n'
            )
            with mock.patch.object(migration, "checked", return_value=""):
                counts = migration.mechanical_contexts(
                    root,
                    [
                        old.relative_to(root).as_posix(),
                        history.relative_to(root).as_posix(),
                    ],
                )
            self.assertEqual(
                counts, {"tool_context": 1, "callback_context": 1, "event_context": 2}
            )
            self.assertIn('session.NewEvent(t.Context(), "seed")', old.read_text())
            self.assertIn('adksession.NewEvent(ctx, "seed")', history.read_text())

    def test_sdk_output_and_source_paths_are_not_module_graph_evidence(self):
        raw = json.dumps(
            {
                "Path": "module",
                "Version": "v1.2.3",
                "Sum": "h1:fixture",
                "Dir": "/private/path",
                "Origin": {"URL": "credential-bearing"},
            }
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            mock.patch.object(migration, "checked", return_value=raw),
        ):
            root = Path(folder)
            (root / "go.mod").write_text("module example\n")
            (root / "go.sum").write_text("")
            evidence = migration.module_graph(root, {})
            self.assertNotIn("private", json.dumps(evidence))
            self.assertNotIn("credential-bearing", json.dumps(evidence))

    def test_failure_evidence_keeps_completed_phase_statuses(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evidence.json"

            def fail(_root, _version, evidence):
                evidence["source_revision"] = "fixture-revision"
                evidence["phases"].append({"phase": "baseline", "exit_code": 1})
                raise migration.MigrationError("required baseline failed")

            with (
                mock.patch.object(migration, "run", side_effect=fail),
                mock.patch(
                    "sys.argv",
                    ["probe", "--candidate", "v2.6.0", "--write-evidence", str(output)],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(migration.main(), 1)
            evidence = json.loads(output.read_text())
            self.assertFalse(evidence["accepted"])
            self.assertEqual(
                evidence["phases"], [{"phase": "baseline", "exit_code": 1}]
            )
            self.assertEqual(evidence["source_revision"], "fixture-revision")

    def test_late_failure_or_interrupt_revokes_provisional_acceptance(self):
        for error in (
            migration.MigrationError("Source checkout changed during characterization"),
            OSError("fixture late failure"),
            ValueError("fixture late failure"),
            KeyboardInterrupt(),
        ):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as folder,
            ):
                output = Path(folder) / "evidence.json"

                def fail_after_acceptance(_root, _version, evidence, caught=error):
                    evidence["accepted"] = True
                    evidence["phases"].append(
                        {"phase": "all-required-cases", "exit_code": 0}
                    )
                    raise caught

                with (
                    mock.patch.object(
                        migration, "run", side_effect=fail_after_acceptance
                    ),
                    mock.patch(
                        "sys.argv",
                        [
                            "probe",
                            "--candidate",
                            "v2.6.0",
                            "--write-evidence",
                            str(output),
                        ],
                    ),
                ):
                    self.assertEqual(
                        migration.main(),
                        130 if isinstance(error, KeyboardInterrupt) else 1,
                    )
                evidence = json.loads(output.read_text())
                self.assertFalse(evidence["accepted"])
                self.assertEqual(evidence["phases"][0]["exit_code"], 0)
                self.assertTrue(
                    evidence.get("interrupted") or evidence.get("setup_failure")
                )

    def test_live_optins_are_removed_without_losing_test_database(self):
        with mock.patch.dict(
            "os.environ",
            {
                "CONTRACTOR_RUN_LIVE_GATE": "1",
                "CONTRACTOR_TEST_DATABASE_URL": "fixture-database",
                "CONTRACTOR_TEST_OTLP_ENDPOINT": "external-collector",
            },
            clear=True,
        ):
            result = migration.environment()
        self.assertNotIn("CONTRACTOR_RUN_LIVE_GATE", result)
        self.assertNotIn("CONTRACTOR_TEST_OTLP_ENDPOINT", result)
        self.assertEqual(result["CONTRACTOR_TEST_DATABASE_URL"], "fixture-database")

    def test_projection_dispositions_require_all_sixteen_cases(self):
        for candidate in (False, True):
            records = list(migration.expected_observations(candidate).values())
            result = migration.validate_observations(records, candidate=candidate)
            self.assertEqual(result["case_count"], 16)
            with self.assertRaises(migration.MigrationError):
                migration.validate_observations(records[:-1], candidate=candidate)
        baseline = list(migration.expected_observations(False).values())
        with self.assertRaises(migration.MigrationError):
            migration.validate_observations(baseline, candidate=True)

    def test_observations_retain_only_bounded_fixed_tokens(self):
        record = migration.expected_observations(False)["history:complete_pair"].copy()
        record["calls"] = ["secret-provider-id"]
        record["values"] = ["secret-tool-result"]
        record["model_calls"] = "secret-counter"
        record["raw_preserved"] = "secret-value"
        safe = migration.safe_observation(record)
        self.assertNotIn("secret", json.dumps(safe))
        self.assertEqual(safe["calls"], ["unexpected"])
        self.assertEqual(safe["model_calls"], "unexpected")
        with self.assertRaises(migration.MigrationError):
            migration.validate_observations([safe], candidate=False)
        for payload in (
            [],
            {**record, "extra": "secret"},
            {**record, "case": "secret"},
        ):
            with (
                self.subTest(payload=payload),
                self.assertRaises(migration.MigrationError),
            ):
                migration.safe_observation(payload)

    def test_malformed_observation_keeps_real_exit_without_raw_output(self):
        import subprocess

        for payload in ('{"extra":"secret"}', "invalid-secret-json"):
            event = {
                "Action": "output",
                "Package": "fixture",
                "Output": migration.OBSERVATION_MARKER + payload,
            }
            phases = []
            with (
                mock.patch.object(
                    migration,
                    "command",
                    return_value=subprocess.CompletedProcess(
                        [], 7, json.dumps(event), ""
                    ),
                ),
                self.assertRaises(migration.MigrationError),
            ):
                migration.execute_tests(
                    "baseline", Path("."), ["go", "test"], {}, phases
                )
            self.assertEqual(phases[0]["exit_code"], 7)
            self.assertNotIn("secret", json.dumps(phases))

    def test_phase_case_inventory_has_no_alias_to_aggregate(self):
        import subprocess

        output = json.dumps({"Action": "run", "Package": "fixture", "Test": "TestOne"})
        phases = []
        with (
            mock.patch.object(
                migration,
                "command",
                return_value=subprocess.CompletedProcess([], 0, output, ""),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            aggregate = migration.execute_tests(
                "baseline", Path("."), ["go", "test"], {}, phases
            )
        aggregate.append({"Action": "fail", "Package": "different"})
        self.assertEqual(len(phases[0]["case_events"]), 1)
        self.assertEqual(phases[0]["case_counts"]["run"], 1)

    def test_original_review_cannot_be_overwritten(self):
        original = migration.ROOT / "tasks/evidence/v419-go-adk-26-review.json"
        with (
            mock.patch(
                "sys.argv",
                ["probe", "--candidate", "v2.6.0", "--write-evidence", str(original)],
            ),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as failure,
        ):
            migration.main()
        self.assertEqual(failure.exception.code, 2)

    def test_repeated_observations_cannot_hide_a_mismatch(self):
        records = list(migration.expected_observations(False).values())
        self.assertEqual(
            migration.validate_observations(records + records, candidate=False)[
                "case_count"
            ],
            16,
        )
        changed = records[0].copy()
        changed["raw_preserved"] = False
        with self.assertRaises(migration.MigrationError):
            migration.validate_observations(records + [changed], candidate=False)

    def test_compact_report_keeps_all_case_actions_and_package_failure(self):
        evidence = {
            "phases": [
                {
                    "phase": "baseline",
                    "exit_code": 1,
                    "case_events": [
                        {"Package": "fixture", "Test": "TestOne", "Action": "run"},
                        {"Package": "fixture", "Test": "TestOne", "Action": "pass"},
                        {"Package": "fixture", "Test": "TestOne", "Action": "fail"},
                        {"Package": "fixture", "Action": "fail"},
                    ],
                }
            ]
        }
        migration.compact_evidence(evidence)
        phase = evidence["phases"][0]
        self.assertEqual(phase["case_actions"], {"fixture:TestOne": "run/pass/fail"})
        self.assertEqual(phase["package_actions"], {"fixture": "fail"})
        self.assertEqual(phase["exit_code"], 1)

    def test_new_evidence_preserves_previous_failed_run(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evidence.json"
            output.write_text(
                json.dumps(
                    {
                        "accepted": False,
                        "phases": [{"phase": "old-baseline", "exit_code": 1}],
                    }
                )
            )
            with (
                mock.patch.object(
                    migration, "run", side_effect=lambda _r, _v, evidence: evidence
                ),
                mock.patch(
                    "sys.argv",
                    ["probe", "--candidate", "v2.6.0", "--write-evidence", str(output)],
                ),
            ):
                self.assertEqual(migration.main(), 1)
            evidence = json.loads(output.read_text())
            self.assertFalse(evidence["previous_runs"][0]["accepted"])
            self.assertEqual(evidence["previous_runs"][0]["phases"][0]["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
