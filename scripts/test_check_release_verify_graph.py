"""Negative tests for the release-graph guard; run by make lint."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import check_release_verify_graph as guard
import select_ci_stages as selector

STAGES = list(guard.FAST_STAGES) + ["release-verify-race"]
CI_STAGES = list(selector.RELEASE_STAGE_TIMEOUTS)
WORKFLOW = """name: CI
on:
  pull_request:
  push:
    tags: ["v*"]
  workflow_dispatch:
concurrency:
  group: ci-${{ github.event_name == 'pull_request' && github.ref || github.run_id }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
jobs:
  select:
    steps:
      - name: Select checks
        run: python3 scripts/select_ci_stages.py
  stage:
    needs: select
    runs-on: ubuntu-24.04
    strategy:
      fail-fast: false
      matrix: ${{ fromJSON(needs.select.outputs.matrix) }}
    steps:
      - run: |
          make -k ${{ matrix.stage }} 2>&1 | tee log
      - name: Run Linux Git container proof
        if: matrix.stage == 'release-verify-integration'
        shell: bash
        run: make test-git-read-only
  pr-verify:
    needs: stage
  release-verify:
    needs: stage
"""
MAKEFILE = """verify: lint test build ui-verify
release-verify: {stages}
release-verify-lint: lint build
release-verify-unit: test
release-verify-ui: ui-verify
release-verify-race: ; go test -race ./...
lint: ; gofmt -l .
test: ; {unit}
build: ; go build ./...
ui-verify: ; pnpm build
"""


def stage_order(stages: str, unit: str = "go test ./...") -> list[str]:
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "Makefile").write_text(MAKEFILE.format(stages=stages, unit=unit))
        with mock.patch.object(guard, "ROOT", Path(directory)):
            return guard.check_stage_order()


class StageOrderTest(unittest.TestCase):
    def test_fast_stages_first(self) -> None:
        self.assertEqual(stage_order(" ".join(STAGES)), STAGES)

    def test_order_holds_when_the_guard_runs_inside_make(self) -> None:
        # make lint runs the guard as a recipe of a parent make -k.
        with mock.patch.dict(os.environ, {"MAKELEVEL": "1", "MAKEFLAGS": "k"}):
            self.assertEqual(stage_order(" ".join(STAGES)), STAGES)

    def test_heavy_stage_before_lint_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must start with"):
            stage_order("release-verify-race " + " ".join(guard.FAST_STAGES))

    def test_race_suite_in_a_fast_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "heavy suites"):
            stage_order(" ".join(STAGES), unit="go test -race ./...")


class WorkflowTest(unittest.TestCase):
    def test_macos_stage_runner_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("needs: select\n    runs-on: ubuntu-24.04", "needs: select\n    runs-on: macos-15")
        with self.assertRaisesRegex(SystemExit, "Linux Ubuntu runner"):
            guard.check_ci_workflow(CI_STAGES, workflow)

    def test_removed_linux_container_pipeline_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("make test-git-read-only", "true")
        with self.assertRaisesRegex(SystemExit, "Linux Git container pipeline"):
            guard.check_ci_workflow(CI_STAGES, workflow)

    def test_complete_workflow_passes(self) -> None:
        guard.check_ci_workflow(CI_STAGES, WORKFLOW)

    def test_cancelling_pushes_is_rejected(self) -> None:
        workflow = WORKFLOW.replace(
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
            "cancel-in-progress: true",
        )
        with self.assertRaisesRegex(SystemExit, "only for pull requests"):
            guard.check_ci_workflow(CI_STAGES, workflow)

    def test_shared_push_group_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("github.run_id", "github.ref")
        with self.assertRaisesRegex(SystemExit, "only for pull requests"):
            guard.check_ci_workflow(CI_STAGES, workflow)

    def test_missing_selector_matrix_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("matrix: ${{ fromJSON(needs.select.outputs.matrix) }}", "matrix: {}")
        with self.assertRaisesRegex(SystemExit, "selector matrix"):
            guard.check_ci_workflow(CI_STAGES, workflow)

    def test_fail_fast_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("fail-fast: false", "fail-fast: true")
        with self.assertRaisesRegex(SystemExit, "must not cancel"):
            guard.check_ci_workflow(CI_STAGES, workflow)

    def test_missing_manual_full_gate_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("  workflow_dispatch:\n", "")
        with self.assertRaisesRegex(SystemExit, "full gate only"):
            guard.check_ci_workflow(CI_STAGES, workflow)


ADVISORIES = ["go run golang.org/x/vuln/cmd/govulncheck@v1.8.0 ./cmd/...", "python3 scripts/audit_runtime_dependencies.py"]
ADVISORY_JOB = """  advisories:
    runs-on: ubuntu-24.04
    steps:
      - run: make -k advisories
"""


class AdvisoryScanTest(unittest.TestCase):
    def test_advisory_job_outside_release_passes(self) -> None:
        guard.check_advisories_outside_release(["go test ./..."], ADVISORIES, WORKFLOW + ADVISORY_JOB)

    def test_scan_in_release_verify_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "release-verify runs live advisory scans"):
            guard.check_advisories_outside_release(ADVISORIES[1:], ADVISORIES, WORKFLOW + ADVISORY_JOB)

    def test_missing_advisory_job_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "its own job"):
            guard.check_advisories_outside_release([], ADVISORIES, WORKFLOW)

    def test_floating_runner_image_is_rejected(self) -> None:
        workflow = WORKFLOW + ADVISORY_JOB.replace("ubuntu-24.04", "ubuntu-latest")
        with self.assertRaisesRegex(SystemExit, "pin runner images"):
            guard.check_advisories_outside_release([], ADVISORIES, workflow)


class DocumentedStagesTest(unittest.TestCase):
    def table(self, stages: list[str]) -> str:
        return "| Stage | Runs |\n| --- | --- |\n" + "".join(f"| `{stage}` | x |\n" for stage in stages)

    def test_matching_table_passes(self) -> None:
        guard.check_documented_stages(STAGES, self.table(STAGES))

    def test_reordered_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "documents stages"):
            guard.check_documented_stages(STAGES, self.table(list(reversed(STAGES))))


E2E = frozenset({"e2e"})
PACKAGE = "example/tests/e2e"


class FakeInventory:
    """Packages and test lists without invoking go."""

    def __init__(self, tagged: set[str], plain: frozenset[str] = frozenset()) -> None:
        self.tagged = frozenset(tagged)
        self.plain = plain

    def packages(self, tags, patterns):
        return (PACKAGE,) if "./tests/e2e" in patterns else ()

    def tests(self, tags, packages):
        names = self.tagged | self.plain if "e2e" in tags else self.plain
        return {package: names for package in packages}


def e2e_test(run: str | None, source: str = "release-verify") -> guard.GoTest:
    return guard.GoTest(source, E2E, ("./tests/e2e",), run)


class RunSelectionTest(unittest.TestCase):
    def test_alternatives(self) -> None:
        self.assertEqual(guard.run_alternatives("^(TestA|TestB)$"), ["^(?:TestA)$", "^(?:TestB)$"])
        self.assertEqual(guard.run_alternatives("Lease|Reconcile"), ["Lease", "Reconcile"])
        self.assertEqual(guard.run_alternatives("^TestA/sub|case"), ["^TestA"])
        self.assertEqual(guard.run_alternatives("^(TestA)|(TestB)$"), ["^(TestA)", "(TestB)$"])
        self.assertEqual(guard.run_alternatives("^(Test(A|B))$"), ["^(?:Test(A|B))$"])

    def test_existing_names_pass(self) -> None:
        inventory = FakeInventory({"TestA", "TestB"})
        self.assertEqual(guard.check_selected_tests_exist([e2e_test("^(TestA|TestB)$")], inventory), 2)

    def test_renamed_selected_test_is_rejected(self) -> None:
        inventory = FakeInventory({"TestA", "TestBRenamed"})
        with self.assertRaisesRegex(SystemExit, "TestB"):
            guard.check_selected_tests_exist([e2e_test("^(TestA|TestB)$")], inventory)

    def test_empty_selection_is_not_a_name(self) -> None:
        self.assertEqual(guard.check_selected_tests_exist([e2e_test("^$")], FakeInventory(set())), 0)


class E2EReachabilityTest(unittest.TestCase):
    def reach(self, tagged: set[str], release: list[guard.GoTest], opt_in: dict | None = None, allowlist: dict | None = None) -> int:
        inventory = FakeInventory(tagged)
        with mock.patch.object(guard, "OPT_IN_E2E_TESTS", allowlist or {}):
            return guard.check_e2e_reachable({PACKAGE: frozenset(tagged)}, release, opt_in or {}, inventory)

    def test_selected_tests_pass(self) -> None:
        self.assertEqual(self.reach({"TestA", "TestB"}, [e2e_test("^(TestA|TestB)$")]), 2)

    def test_unselected_tagged_test_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "TestB .* is not selected by release-verify"):
            self.reach({"TestA", "TestB"}, [e2e_test("^TestA$")])

    def test_allowlisted_test_needs_its_opt_in_gate(self) -> None:
        allowlist = {"TestB": ("test-opt-in", "needs a scanner")}
        self.assertEqual(
            self.reach({"TestA", "TestB"}, [e2e_test("^TestA$")], {"test-opt-in": [e2e_test("^TestB$", "test-opt-in")]}, allowlist),
            2,
        )
        with self.assertRaisesRegex(SystemExit, "not selected by its opt-in target"):
            self.reach({"TestA", "TestB"}, [e2e_test("^TestA$")], {"test-opt-in": [e2e_test("^TestA$", "test-opt-in")]}, allowlist)

    def test_allowlist_must_stay_exact(self) -> None:
        with self.assertRaisesRegex(SystemExit, "remove it from the opt-in allowlist"):
            self.reach({"TestA"}, [e2e_test(None)], {"test-opt-in": [e2e_test(None)]}, {"TestA": ("test-opt-in", "x")})
        with self.assertRaisesRegex(SystemExit, "no longer exists"):
            self.reach({"TestA"}, [e2e_test(None)], {}, {"TestGone": ("test-opt-in", "x")})
        with self.assertRaisesRegex(SystemExit, "without a reason"):
            self.reach({"TestA"}, [], {"test-opt-in": [e2e_test(None)]}, {"TestA": ("test-opt-in", "")})


class IdentityInventory:
    def packages(self, tags, patterns):
        return tuple(patterns)


class NonRacePassTest(unittest.TestCase):
    CONSTRAINED = {"./internal/findingintake": frozenset({"integration"})}

    def check(self, *tests: guard.GoTest) -> None:
        guard.check_non_race_passes(self.CONSTRAINED, list(tests), IdentityInventory())

    def test_complete_non_race_pass_passes(self) -> None:
        self.check(guard.GoTest("x", frozenset({"integration"}), ("./internal/findingintake",), None))

    def test_race_only_pass_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "without -race"):
            self.check(guard.GoTest("x", frozenset({"integration"}), ("./internal/findingintake",), None, race=True))

    def test_pass_without_the_needed_tag_or_with_a_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "without -race"):
            self.check(guard.GoTest("x", frozenset(), ("./internal/findingintake",), None))
        with self.assertRaisesRegex(SystemExit, "without -race"):
            self.check(guard.GoTest("x", frozenset({"integration"}), ("./internal/findingintake",), "Budget"))


class ArtifactBackendMatrixTest(unittest.TestCase):
    SWITCHING = {"./internal/artifacts": frozenset(), "./internal/findingintake": frozenset({"integration"})}
    PACKAGES = ("./internal/artifacts", "./internal/findingintake")

    def check(self, *tests: guard.GoTest, switching=None) -> int:
        return guard.check_artifact_backend_matrix(
            self.SWITCHING if switching is None else switching, list(tests), IdentityInventory()
        )

    def filesystem_pass(self, **changes) -> guard.GoTest:
        fields = {
            "source": "x", "tags": frozenset({"integration"}), "packages": self.PACKAGES, "run": None,
            "environment": frozenset({guard.FILESYSTEM_BACKEND}),
        }
        return guard.GoTest(**{**fields, **changes})

    def test_parsed_command_carries_its_environment(self) -> None:
        test = guard.parse_go_test(
            "CONTRACTOR_TEST_ARTIFACT_BACKEND=filesystem go test -tags=integration -count=1 ./internal/artifacts", "x"
        )
        self.assertEqual(test.environment, frozenset({guard.FILESYSTEM_BACKEND}))
        self.assertEqual(guard.parse_go_test("go test ./internal/artifacts", "x").environment, frozenset())

    def test_complete_filesystem_pass_passes(self) -> None:
        self.assertEqual(self.check(self.filesystem_pass()), 2)
        self.assertEqual(self.check(self.filesystem_pass(race=True)), 2)

    def test_pass_on_the_default_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no complete CONTRACTOR_TEST_ARTIFACT_BACKEND=filesystem pass"):
            self.check(self.filesystem_pass(environment=frozenset()))
        with self.assertRaisesRegex(SystemExit, "no complete"):
            self.check(self.filesystem_pass(environment=frozenset({"CONTRACTOR_TEST_ARTIFACT_BACKEND=postgres"})))

    def test_partial_pass_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "internal/artifacts"):
            self.check(self.filesystem_pass(run="TestPostgresIntegration64MiB"))
        with self.assertRaisesRegex(SystemExit, r"\['./internal/findingintake'\]"):
            self.check(self.filesystem_pass(tags=frozenset()))
        with self.assertRaisesRegex(SystemExit, r"\['./internal/findingintake'\]"):
            self.check(self.filesystem_pass(packages=("./internal/artifacts",)))

    def test_lost_backend_switch_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no Go test reads"):
            self.check(self.filesystem_pass(), switching={})


class LintStageTest(unittest.TestCase):
    COMMANDS = [
        "test -z \"$(gofmt -l cmd internal tests tools)\"",
        "python3 scripts/check_database_erd.py",
        "go run ./cmd/contractor-server config validate --root ./configs",
        "go -C tools/testonlyexports run . -tags=integration,e2e -baseline tools/testonlyexports/baseline.txt /repo",
        "CONTRACTOR_ERD_USAGE_OUT= go test -count=1 -run '^TestERDUsageMatchesTheSQLScan$' ./internal/persistence",
    ]

    def test_complete_lint_stage_passes(self) -> None:
        guard.check_lint_stage(self.COMMANDS)

    def test_lost_catalog_or_erd_check_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "shipped configuration catalog"):
            guard.check_lint_stage(self.COMMANDS[:2] + self.COMMANDS[3:])
        with self.assertRaisesRegex(SystemExit, "database ERD against the migrations"):
            guard.check_lint_stage([self.COMMANDS[0], *self.COMMANDS[2:]])
        with self.assertRaisesRegex(SystemExit, "database ERD's usage lists"):
            guard.check_lint_stage(self.COMMANDS[:4])

    def test_lost_test_only_export_check_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "exported identifiers that only tests use"):
            guard.check_lint_stage(self.COMMANDS[:3] + self.COMMANDS[4:])


M = guard.MODULE


class RaceCoverageTest(unittest.TestCase):
    WITH_TESTS = {M + "internal/a", M + "internal/b", M + "tests/faults"}
    TAGGED_ONLY = {M + "tests/ui-stack", M + "internal/tagged"}

    def check(self, raced, exceptions=None, integration=frozenset({M + "internal/tagged"})) -> int:
        return guard.check_race_coverage(
            raced, self.WITH_TESTS, self.TAGGED_ONLY, set(integration),
            {"tests/ui-stack": "browser process tests"} if exceptions is None else exceptions,
        )

    def test_complete_coverage_passes(self) -> None:
        self.assertEqual(self.check([{M + "internal/a"}, {M + "internal/b", M + "tests/faults"}]), 3)

    def test_unraced_package_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "tests/faults has tests but runs under -race in no release stage"):
            self.check([{M + "internal/a"}, {M + "internal/b"}])

    def test_tagged_only_package_needs_a_race_pass_or_exception(self) -> None:
        with self.assertRaisesRegex(SystemExit, "ui-stack has only tagged tests"):
            self.check([{M + "internal/a", M + "internal/b", M + "tests/faults"}], exceptions={})

    def test_package_raced_twice_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "raced by 2 release passes"):
            self.check([{M + "internal/a", M + "internal/b"}, {M + "internal/b", M + "tests/faults"}])

    def test_exceptions_need_a_reason_and_must_stay_unraced(self) -> None:
        raced = [{M + "internal/a", M + "internal/b", M + "tests/faults"}]
        with self.assertRaisesRegex(SystemExit, "has no reason"):
            self.check(raced, exceptions={"tests/ui-stack": ""})
        with self.assertRaisesRegex(SystemExit, "remove the exception"):
            self.check(raced, exceptions={"tests/ui-stack": "x", "internal/a": "slow"})
        with self.assertRaisesRegex(SystemExit, "has no tests"):
            self.check(raced, exceptions={"tests/ui-stack": "x", "internal/gone": "removed"})

    def test_substitution_is_evaluated_like_make(self) -> None:
        command = "go test -p 1 -race -count=1 $(printf '%s\\n' b a | sort -u)"
        self.assertEqual(guard.substitution(command), "printf '%s\\n' b a | sort -u")
        self.assertEqual(guard.substituted_packages(command), {"a", "b"})


class IntegrationParallelismTest(unittest.TestCase):
    command = "go test -race -tags=integration -v -count=1 -p {parallelism} ./fixture -run '^TestTagged$'"

    def check(self, command: str) -> int:
        commands = [command]
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(guard, "ROOT", Path(directory)),
            mock.patch.object(guard, "release_selection", return_value={"./fixture": {"TestTagged"}}),
            mock.patch.object(guard, "dry_run", return_value=commands),
            mock.patch.object(guard, "EXCEPTIONS", {}),
        ):
            return guard.check_integration_graph()

    def test_serial_and_parallel_integration_commands_keep_coverage(self) -> None:
        for parallelism in ("1", "2", "4"):
            with self.subTest(parallelism=parallelism):
                self.assertEqual(self.check(self.command.format(parallelism=parallelism)), 1)

    def test_invalid_or_missing_parallelism_is_rejected(self) -> None:
        for parallelism in ("0", "-1", "auto", ""):
            with self.subTest(parallelism=parallelism):
                with self.assertRaisesRegex(SystemExit, "positive integer"):
                    self.check(self.command.format(parallelism=parallelism))

    def test_parallel_command_still_requires_uncached_execution(self) -> None:
        with self.assertRaisesRegex(SystemExit, "lost race, verbose evidence, count"):
            self.check(self.command.format(parallelism="2").replace("-count=1 ", ""))


class IterativeGateSelectionTest(unittest.TestCase):
    def gate(self):
        return SimpleNamespace(
            matrix=lambda: {
                "postgres": {guard.MODULE + "internal/fixture": ["TestStorage/subcase"]},
                "process": {guard.MODULE + "tests/e2e": ["TestJourney/second_round"]},
            },
            pattern=lambda cases: "^(" + "|".join(case.split("/")[0] for case in cases) + ")$",
        )

    def test_direct_target_declares_only_its_real_boundary_and_race_mode(self):
        for boundary, tag, package, case, race in (
            ("postgres", "integration", "./internal/fixture", "TestStorage", True),
            ("process", "e2e", "./tests/e2e", "TestJourney", False),
        ):
            with self.subTest(boundary=boundary), mock.patch.object(guard, "load_script", return_value=self.gate()):
                selected = guard.script_go_tests(
                    "scripts/test-audit-iterative.py",
                    f"python3 scripts/test-audit-iterative.py --boundary {boundary}",
                )
                self.assertEqual(len(selected), 1)
                self.assertEqual(selected[0].tags, frozenset({tag}))
                self.assertEqual(selected[0].packages, (package,))
                self.assertEqual(selected[0].run, f"^({case})$")
                self.assertEqual(selected[0].race, race)

    def test_missing_or_invalid_boundary_cannot_declare_phantom_tests(self):
        for argument in ("", "--boundary", "--boundary unknown"):
            with self.subTest(argument=argument), mock.patch.object(guard, "load_script", return_value=self.gate()):
                with self.assertRaisesRegex(SystemExit, "boundary"):
                    guard.script_go_tests("scripts/test-audit-iterative.py", "python3 gate.py " + argument)


class ConsolidatedFamilyTest(unittest.TestCase):
    def test_transitive_unwrapped_test_commands_are_rejected(self) -> None:
        recipes = (
            "go test ./...", "cd runtime && uv run pytest tests/test_allocation.py",
            "python3 scripts/test-findings-e2e.py", "python3 scripts/shared_fixture_gate.py",
            "cd runtime && uv run python -m unittest discover -s ../scripts",
        )
        for recipe in recipes:
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                Path(directory, "Makefile").write_text("release-verify-families: nested\nnested: ; " + recipe + "\n")
                with mock.patch.object(guard, "ROOT", Path(directory)):
                    with self.assertRaisesRegex(SystemExit, "repeat test execution"):
                        guard.check_family_commands(guard.dry_run("release-verify-families", "RELEASE_CONSOLIDATED=1"))
        guard.check_family_commands([":", "python3 scripts/shared_fixture_gate.py --sources-only"])

    def test_full_runtime_and_report_gates_are_required(self) -> None:
        commands = ["cd runtime && uv run pytest -W error tests --junitxml=report.xml"] + [
            "python3 scripts/" + name for name in (
                "release_test_reports.py verify-runtime", "release_test_reports.py verify-audit",
                "release_test_reports.py verify-findings", "shared_fixture_gate.py --go-report report",
                "release_test_reports.py verify-iterative-postgres",
                "release_test_reports.py verify-iterative-process",
            )
        ]
        guard.check_runtime_commands(commands)
        for changed in (
            commands + commands[:1], [commands[0] + " -k one", *commands[1:]], [commands[0].replace("-W error ", ""), *commands[1:]],
            [commands[0].replace("error tests ", "error tests/test_one.py "), *commands[1:]],
            *[commands[:index] + commands[index + 1:] for index in range(1, len(commands))],
        ):
            with self.subTest(commands=changed), self.assertRaises(SystemExit):
                guard.check_runtime_commands(changed)

    def test_lost_test_tag_race_and_backend_are_rejected(self) -> None:
        inventory = FakeInventory({"TestA", "TestB"})
        family = [guard.GoTest("family", E2E, ("./tests/e2e",), "^(TestA|TestB)$", True)]
        complete = guard.GoTest("canonical", E2E, ("./tests/e2e",), None, True)
        guard.check_family_coverage(family, [complete], inventory)
        for replacement in (
            guard.GoTest("canonical", E2E, ("./tests/e2e",), "^TestA$", True),
            guard.GoTest("canonical", E2E, ("./tests/e2e",), None, False),
            guard.GoTest("canonical", frozenset(), ("./tests/e2e",), None, True),
        ):
            with self.subTest(replacement=replacement), self.assertRaisesRegex(SystemExit, "family coverage lost"):
                guard.check_family_coverage(family, [replacement], inventory)
        filesystem = guard.GoTest("family", E2E, ("./tests/e2e",), None, True, frozenset({guard.FILESYSTEM_BACKEND}))
        with self.assertRaisesRegex(SystemExit, "family coverage lost"):
            guard.check_family_coverage([filesystem], [complete], inventory)


class DatabaseSkipGuardTest(unittest.TestCase):
    def test_skip_after_failed_ping_is_rejected(self) -> None:
        source = """
func pool(t *testing.T) {
	if err := admin.Ping(ctx); err != nil {
		admin.Close()
		t.Skipf("cannot reach: %v", err)
	}
}
"""
        self.assertEqual(guard.database_skip_violations(source), [5])

    def test_skip_after_failed_pool_creation_is_rejected(self) -> None:
        source = """
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Skip("no pool")
	}
"""
        self.assertEqual(guard.database_skip_violations(source), [4])

    def test_reworded_unreachable_skip_is_rejected(self) -> None:
        source = '\tt.Skipf("PostgreSQL is unavailable: %v", err)\n'
        self.assertEqual(guard.database_skip_violations(source), [1])

    def test_fatal_and_unset_url_skip_are_allowed(self) -> None:
        source = """
	if databaseURL == "" {
		t.Skip("CONTRACTOR_TEST_DATABASE_URL is not set")
	}
	if err := admin.Ping(ctx); err != nil {
		t.Fatalf("CONTRACTOR_TEST_DATABASE_URL is set but PostgreSQL is unreachable: %v", err)
	}
	if err := run(); err != nil {
		t.Skip("unrelated precondition")
	}
"""
        self.assertEqual(guard.database_skip_violations(source), [])

    def test_skip_after_the_error_branch_is_allowed(self) -> None:
        source = """
	if err := admin.Ping(ctx); err != nil {
		t.Fatal(err)
	}
	if !superuser {
		t.Skip("requires a superuser")
	}
"""
        self.assertEqual(guard.database_skip_violations(source), [])




class CompactionReportReuseTests(unittest.TestCase):
    def test_each_canonical_report_has_one_verifier_and_the_family_stays_focused(self):
        commands = [
            "verify-compaction-go --report go-unit.jsonl",
            "verify-compaction-runtime --report runtime.xml",
            "verify-compaction-process --report go-process-a.jsonl --scope A",
            "verify-compaction-process --report go-process-c.jsonl --scope C",
        ]
        focused = ["verify-compaction-" + kind for kind in ("go", "runtime", "process")]
        guard.check_compaction_report_reuse(commands, focused)
        for changed in (
            commands[1:],
            commands + [commands[0]],
            commands + ["worker-compaction-process.jsonl"],
        ):
            with self.assertRaises(SystemExit):
                guard.check_compaction_report_reuse(changed, focused)
        with self.assertRaises(SystemExit):
            guard.check_compaction_report_reuse(commands, focused[:-1])


if __name__ == "__main__":
    unittest.main()


class ContextCompactionEvalBoundaryTest(unittest.TestCase):
    def test_offline_gate_and_explicit_live_entry_are_separate(self):
        guard.check_context_compaction_eval_boundary(["python tests/eval/context_compaction/gate.py"], ["python tests/eval/context_compaction/harness.py --manifest frozen --observer private"], "ordinary CI")

    def test_missing_duplicate_or_live_release_paths_fail(self):
        live = ["python tests/eval/context_compaction/harness.py --manifest frozen --observer private"]
        gate = "python tests/eval/context_compaction/gate.py"
        for commands in ([], [gate, gate], [gate, live[0]]):
            with self.assertRaises(SystemExit):
                guard.check_context_compaction_eval_boundary(commands, live, "ordinary CI")
        with self.assertRaises(SystemExit):
            guard.check_context_compaction_eval_boundary([gate], live, "make eval-context-compaction-live")
