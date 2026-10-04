import unittest

from select_ci_stages import (
    BROWSER_A,
    BROWSER_B,
    CAPABILITY_E2E,
    INTEGRATION,
    LINT,
    PROCESS_A,
    PROCESS_B,
    PROCESS_C,
    UI,
    UI_STACK_OPERATIONS,
    UNIT,
    RELEASE_STAGE_TIMEOUTS,
    matrix_for_event,
    stages_for_paths,
)


class SelectCIStagesTest(unittest.TestCase):
    def test_ui_stack_spec_runs_only_its_browser_shard(self):
        self.assertEqual(stages_for_paths(["ui/e2e/stack.spec.ts"]), [LINT, UI, UI_STACK_OPERATIONS])

    def test_ui_source_runs_both_browser_shards(self):
        self.assertEqual(stages_for_paths(["ui/src/routes/workflows/run-form.tsx"]), [LINT, UI, BROWSER_A, BROWSER_B])

    def test_full_browser_shard_deduplicates_focused_operations(self):
        self.assertEqual(
            stages_for_paths(["ui/e2e/stack.spec.ts", "ui/src/routes/workflows/run-form.tsx"]),
            [LINT, UI, BROWSER_A, BROWSER_B],
        )

    def test_runtime_and_server_run_unit_and_integration(self):
        self.assertEqual(
            stages_for_paths(["runtime/contractor/worker.py", "internal/httpapi/runs.go"]),
            [LINT, UNIT, INTEGRATION, CAPABILITY_E2E],
        )

    def test_e2e_tests_select_process_shards(self):
        self.assertEqual(stages_for_paths(["tests/e2e/stack_test.go"]), [LINT, UNIT, PROCESS_A, PROCESS_B, PROCESS_C])

    def test_gate_and_docs_only_stay_fast(self):
        self.assertEqual(stages_for_paths([".github/workflows/ci.yml", "make/release.mk", "docs/testing/README.md"]), [LINT])

    def test_unknown_tree_gets_broad_checks(self):
        self.assertEqual(stages_for_paths(["newtree/feature.go"]), [LINT, UNIT, UI, INTEGRATION])

    def test_manual_and_tag_runs_select_every_release_stage(self):
        for event in ("workflow_dispatch", "push"):
            self.assertEqual(
                [row["stage"] for row in matrix_for_event(event, [])["include"]],
                list(RELEASE_STAGE_TIMEOUTS),
            )


if __name__ == "__main__":
    unittest.main()
