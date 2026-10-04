#!/usr/bin/env python3
"""Regenerate docs/spec/database-erd.html from the current migrations.

Creates a scratch database on the PostgreSQL server named by ERD_DATABASE_URL
(or CONTRACTOR_TEST_DATABASE_URL), migrates it with `contractor-server
migrate`, reads the catalog, lays the diagram out with ELK and fills
template.html. The scratch database is dropped afterwards. Needs psql, Go and
Node.js; run `npm ci --prefix scripts/erd` first (`make docs-erd` does both).
"""

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "docs" / "spec" / "database-erd.html"
MIGRATIONS = ROOT / "internal" / "persistence" / "migrations"
# The migration runner's own bookkeeping table is not part of the model.
EXCLUDED_TABLES = ("contractor_schema_migrations",)

TABLES_SQL = """
SELECT json_agg(t ORDER BY t.name) FROM (
  SELECT c.relname AS name,
    (SELECT json_agg(json_build_object(
              'n', a.attname, 't', format_type(a.atttypid, a.atttypmod),
              'nn', a.attnotnull, 'gen', a.attgenerated <> '') ORDER BY a.attnum)
       FROM pg_attribute a
      WHERE a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped) AS cols,
    (SELECT json_agg(a.attname ORDER BY array_position(k.conkey, a.attnum))
       FROM pg_constraint k
       JOIN pg_attribute a ON a.attrelid = k.conrelid AND a.attnum = ANY(k.conkey)
      WHERE k.conrelid = c.oid AND k.contype = 'p') AS pk,
    (SELECT json_agg(u.cols) FROM (
        SELECT (SELECT json_agg(a.attname ORDER BY array_position(k.conkey, a.attnum))
                  FROM pg_attribute a
                 WHERE a.attrelid = k.conrelid AND a.attnum = ANY(k.conkey)) AS cols
          FROM pg_constraint k
         WHERE k.conrelid = c.oid AND k.contype = 'u' ORDER BY k.conname) u) AS uniq,
    (SELECT json_agg(tg.tgname ORDER BY tg.tgname) FROM pg_trigger tg
      WHERE tg.tgrelid = c.oid AND NOT tg.tgisinternal) AS triggers
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
  WHERE n.nspname = 'public' AND c.relkind = 'r' AND c.relname <> ALL(%s)
) t
"""

FOREIGN_KEYS_SQL = """
SELECT json_agg(f ORDER BY f."from", f.name) FROM (
  SELECT k.conrelid::regclass::text AS "from", k.confrelid::regclass::text AS "to",
    k.conname AS name,
    (SELECT json_agg(a.attname ORDER BY array_position(k.conkey, a.attnum))
       FROM pg_attribute a
      WHERE a.attrelid = k.conrelid AND a.attnum = ANY(k.conkey)) AS fc,
    (SELECT json_agg(a.attname ORDER BY array_position(k.confkey, a.attnum))
       FROM pg_attribute a
      WHERE a.attrelid = k.confrelid AND a.attnum = ANY(k.confkey)) AS tc,
    k.confdeltype AS del
  FROM pg_constraint k JOIN pg_namespace n ON n.oid = k.connamespace
  WHERE k.contype = 'f' AND n.nspname = 'public'
    AND k.conrelid::regclass::text <> ALL(%s) AND k.confrelid::regclass::text <> ALL(%s)
) f
"""


def server_url() -> str:
    url = os.environ.get("ERD_DATABASE_URL") or os.environ.get("CONTRACTOR_TEST_DATABASE_URL")
    if not url:
        sys.exit("set ERD_DATABASE_URL or CONTRACTOR_TEST_DATABASE_URL to a PostgreSQL server")
    return url


def with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path="/" + database))


def psql(url: str, sql: str) -> str:
    return subprocess.run(
        ["psql", url, "-X", "-v", "ON_ERROR_STOP=1", "-Atc", sql],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sql_text_array(values: tuple[str, ...]) -> str:
    return "ARRAY[" + ", ".join("'" + v.replace("'", "''") + "'" for v in values) + "]::text[]"


def read_schema(url: str) -> dict:
    excluded = sql_text_array(EXCLUDED_TABLES)
    return {
        "tables": json.loads(psql(url, TABLES_SQL % excluded)),
        "fks": json.loads(psql(url, FOREIGN_KEYS_SQL % (excluded, excluded))),
    }


def table_usage(tables: list[str]) -> dict[str, list[str]]:
    """Packages whose production Go code names each table in SQL."""
    sources: dict[str, list[str]] = {}
    for path in sorted([*ROOT.glob("internal/**/*.go"), *ROOT.glob("cmd/**/*.go")]):
        if path.name.endswith("_test.go") or "migrations" in path.parts:
            continue
        relative = path.relative_to(ROOT).parts
        package = relative[1] if relative[0] == "internal" else "cmd"
        sources.setdefault(package, []).append(path.read_text())
    usage = {}
    for table in tables:
        pattern = re.compile(rf"(?i)\b(from|join|into|update|table)\s+{table}\b")
        usage[table] = [
            package
            for package, texts in sorted(sources.items())
            if any(pattern.search(text) for text in texts)
        ]
    return usage


def latest_migration() -> str:
    versions = sorted(p.name.split("_", 1)[0] for p in MIGRATIONS.glob("[0-9]*_*.sql"))
    return versions[-1]


def main() -> None:
    server = server_url()
    scratch = "contractor_erd_" + secrets.token_hex(4)
    psql(server, f"CREATE DATABASE {scratch}")
    try:
        scratch_url = with_database(server, scratch)
        subprocess.run(
            ["go", "run", "./cmd/contractor-server", "migrate"],
            cwd=ROOT,
            check=True,
            env={**os.environ, "CONTRACTOR_DATABASE_URL": scratch_url},
        )
        schema = read_schema(scratch_url)
    finally:
        psql(server, f"DROP DATABASE IF EXISTS {scratch}")

    with tempfile.TemporaryDirectory() as work:
        work_dir = Path(work)
        schema_path = work_dir / "schema.json"
        usage_path = work_dir / "usage.json"
        data_path = work_dir / "data.json"
        schema_path.write_text(json.dumps(schema))
        usage_path.write_text(json.dumps(table_usage([t["name"] for t in schema["tables"]])))
        subprocess.run(
            [
                "node",
                str(HERE / "layout.cjs"),
                str(schema_path),
                str(HERE / "meta.json"),
                str(usage_path),
                str(data_path),
                latest_migration(),
            ],
            check=True,
        )
        data = data_path.read_text().replace("</", "<\\/")

    page = (HERE / "template.html").read_text().replace("__DATA__", data)
    OUTPUT.write_text(page)
    tables, keys = len(schema["tables"]), len(schema["fks"])
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {tables} tables, {keys} foreign keys")


if __name__ == "__main__":
    main()
