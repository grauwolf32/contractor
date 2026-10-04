"""Negative tests for the release-graph guard; run by make lint."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_release_verify_graph as guard

STAGES = list(guard.FAST_STAGES) + ["release-verify-race"]
WORKFLOW = """name: CI
concurrency:
  group: ci-${{ github.event_name == 'pull_request' && github.ref || github.run_id }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}
jobs:
  stage:
    strategy:
      fail-fast: false
      matrix:
        include:
          - stage: release-verify-lint
          - stage: release-verify-unit
          - stage: release-verify-ui
          - stage: release-verify-race
    steps:
      - run: |
          make -k ${{ matrix.stage }} 2>&1 | tee log
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
    def test_complete_workflow_passes(self) -> None:
        guard.check_ci_workflow(STAGES, WORKFLOW)

    def test_cancelling_pushes_is_rejected(self) -> None:
        workflow = WORKFLOW.replace(
            "cancel-in-progress: ${{ github.event_name == 'pull_request' }}",
            "cancel-in-progress: true",
        )
        with self.assertRaisesRegex(SystemExit, "only for pull requests"):
            guard.check_ci_workflow(STAGES, workflow)

    def test_shared_push_group_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("github.run_id", "github.ref")
        with self.assertRaisesRegex(SystemExit, "only for pull requests"):
            guard.check_ci_workflow(STAGES, workflow)

    def test_missing_stage_job_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("          - stage: release-verify-race\n", "")
        with self.assertRaisesRegex(SystemExit, "runs stages"):
            guard.check_ci_workflow(STAGES, workflow)

    def test_fail_fast_is_rejected(self) -> None:
        workflow = WORKFLOW.replace("fail-fast: false", "fail-fast: true")
        with self.assertRaisesRegex(SystemExit, "must not cancel"):
            guard.check_ci_workflow(STAGES, workflow)


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


if __name__ == "__main__":
    unittest.main()
