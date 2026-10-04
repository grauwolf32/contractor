"""Negative tests for the release-graph guard; run by make lint."""

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


class DocumentedStagesTest(unittest.TestCase):
    def table(self, stages: list[str]) -> str:
        return "| Stage | Runs |\n| --- | --- |\n" + "".join(f"| `{stage}` | x |\n" for stage in stages)

    def test_matching_table_passes(self) -> None:
        guard.check_documented_stages(STAGES, self.table(STAGES))

    def test_reordered_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "documents stages"):
            guard.check_documented_stages(STAGES, self.table(list(reversed(STAGES))))


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
