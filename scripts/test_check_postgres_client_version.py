"""Tests for the pg_dump and pg_restore version check; run by make lint."""

import unittest

import check_postgres_client_version as check


def versions(dump: str, restore: str) -> dict[str, str]:
    return {
        "pg_dump": f"pg_dump (PostgreSQL) {dump}\n",
        "pg_restore": f"pg_restore (PostgreSQL) {restore}\n",
    }


class ProblemsTest(unittest.TestCase):
    def test_older_pg_dump_names_both_versions(self) -> None:
        self.assertEqual(
            check.problems("170006", versions("16.4 (Ubuntu 16.4-1.pgdg22.04+1)", "17.6")),
            ["pg_dump 16.4 is older than the PostgreSQL server 17.6; use pg_dump 17 or newer"],
        )

    def test_older_pg_restore_names_both_versions(self) -> None:
        self.assertEqual(
            check.problems("170006", versions("17.6 (Homebrew)", "16.10 (Homebrew)")),
            ["pg_restore 16.10 is older than the PostgreSQL server 17.6; use pg_restore 17 or newer"],
        )

    def test_both_older_tools_are_reported(self) -> None:
        self.assertEqual(len(check.problems("170000", versions("15.8", "16.4"))), 2)

    def test_same_or_newer_major_passes_whatever_the_minor(self) -> None:
        self.assertEqual(check.problems("170009", versions("17.0", "17.0")), [])
        self.assertEqual(check.problems("170006", versions("18beta1", "18.1")), [])

    def test_unreadable_versions_are_rejected(self) -> None:
        for version_num in ("", "17.6", "90624"):
            with self.subTest(version_num=version_num), self.assertRaises(ValueError):
                check.problems(version_num, versions("17.6", "17.6"))
        for output in ("pg_dump 17.6", "pg_restore (PostgreSQL) 17.6", ""):
            with self.subTest(output=output), self.assertRaises(ValueError):
                check.problems("170006", {"pg_dump": output, "pg_restore": "pg_restore (PostgreSQL) 17.6"})


if __name__ == "__main__":
    unittest.main()
