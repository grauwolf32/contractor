"""Negative tests for the ERD freshness check; run by make lint."""

import json
import unittest

import check_database_erd as check


def page(version: str, tables: list[str]) -> str:
    """An ERD page as build_erd.py writes it, with "</" escaped in the data."""
    data = json.dumps({"schemaVersion": version, "tables": [{"name": name} for name in tables]})
    data = data.replace("</", "<\\/")
    return '<html><script type="application/json" id="data">' + data + "</script></html>"


BASELINE = """
-- CREATE TABLE commented_out (id int);
/* CREATE TABLE block_comment (id int); /* nested */ still a comment */
CREATE TABLE audits (audit_id text PRIMARY KEY, note text DEFAULT 'CREATE TABLE literal (id int)');
CREATE TABLE IF NOT EXISTS public.runs (run_id text);
CREATE UNLOGGED TABLE scratch_cache (id int);
CREATE TEMP TABLE session_only (id int);
CREATE TABLE "Quoted""Name" (id int);
CREATE FUNCTION make_table() RETURNS void LANGUAGE plpgsql AS $body$
BEGIN
    CREATE TABLE routine_table (id int);
END;
$body$;
DO $$
BEGIN
    CREATE TABLE do_block_table (id int);
END;
$$;
CREATE TABLE other_schema.hidden (id int);
"""
LATER = """
DROP TABLE IF EXISTS scratch_cache, "Quoted""Name" CASCADE;
ALTER TABLE ONLY runs RENAME TO workflow_runs;
ALTER TABLE workflow_runs RENAME COLUMN run_id TO id;
"""
MIGRATIONS = [("000001_baseline.sql", BASELINE), ("000002_later.sql", LATER)]
TABLES = ["audits", "do_block_table", "workflow_runs"]


class MigratedTablesTest(unittest.TestCase):
    def test_statements_apply_in_order_and_skip_what_is_not_executed(self) -> None:
        self.assertEqual(check.migrated_tables(MIGRATIONS), set(TABLES))

    def test_created_quoted_names_keep_their_case(self) -> None:
        self.assertEqual(check.migrated_tables(MIGRATIONS[:1]) - set(TABLES), {"runs", "scratch_cache", 'Quoted"Name'})

    def test_renaming_an_unknown_table_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "renames table missing"):
            check.migrated_tables([("000001_x.sql", "ALTER TABLE missing RENAME TO present;")])


class ErdProblemsTest(unittest.TestCase):
    def test_fresh_page_passes(self) -> None:
        self.assertEqual(check.erd_problems(page("000002", TABLES), MIGRATIONS), [])

    def test_older_schema_version_is_reported(self) -> None:
        self.assertEqual(
            check.erd_problems(page("000001", TABLES), MIGRATIONS),
            ["the page shows schemaVersion 000001, the newest migration is 000002"],
        )

    def test_missing_and_extra_tables_are_reported(self) -> None:
        problems = check.erd_problems(page("000002", ["audits", "runs", "workflow_runs"]), MIGRATIONS)
        self.assertEqual(
            problems,
            [
                "the page draws tables the migrations no longer create: runs",
                "the page lacks tables the migrations create: do_block_table",
            ],
        )

    def test_a_new_migration_makes_the_page_stale(self) -> None:
        added = [*MIGRATIONS, ("000003_reports.sql", "CREATE TABLE audit_report_pages (audit_id text);")]
        self.assertEqual(
            check.erd_problems(page("000002", TABLES), added),
            [
                "the page shows schemaVersion 000002, the newest migration is 000003",
                "the page lacks tables the migrations create: audit_report_pages",
            ],
        )

    def test_page_without_schema_data_is_rejected(self) -> None:
        self.assertEqual(
            check.erd_problems("<html></html>", MIGRATIONS),
            ["the page has no schema data; regenerate it with make docs-erd"],
        )
        self.assertRegex(
            check.erd_problems('<script type="application/json" id="data">{"tables": []}</script>', MIGRATIONS)[0],
            "unreadable",
        )

    def test_misnamed_or_missing_migrations_are_rejected(self) -> None:
        self.assertEqual(
            check.erd_problems(page("000001", []), [("1_baseline.sql", "")]),
            ["migration 1_baseline.sql is not named NNNNNN_name.sql"],
        )
        self.assertEqual(check.erd_problems(page("000001", []), []), ["there are no migrations"])


if __name__ == "__main__":
    unittest.main()
