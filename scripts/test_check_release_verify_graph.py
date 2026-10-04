"""Negative tests for the release-graph guard; run by make lint."""

import unittest

import check_release_verify_graph as guard


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
