#!/usr/bin/env python3
"""Fail when docs/spec/database-erd.html no longer matches the migrations.

The page records the newest migration it was built from (schemaVersion) and
the tables of that schema. Without a database, this check requires that
schemaVersion to be the newest migration and the page's tables to be exactly
the tables the migrations leave behind after every CREATE TABLE, DROP TABLE
and ALTER TABLE ... RENAME TO. Regenerate the page with make docs-erd (see
scripts/erd/README.md); make lint runs this check.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = re.compile(r'<script type="application/json" id="data">(.*?)</script>', re.S)
MIGRATION = re.compile(r"^([0-9]{6})_[a-z0-9_]+\.sql$")
# A table of the public schema, the only schema the page draws.
NAME = r'(?:public\.)?("(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*)(?![\w$."])'
CREATE = re.compile(rf"\bCREATE\s+(TEMP\s+|TEMPORARY\s+|UNLOGGED\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{NAME}", re.I)
DROP = re.compile(r"\bDROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([^;]*)", re.I)
RENAME = re.compile(rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?{NAME}\s+RENAME\s+TO\s+{NAME}", re.I)
DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
# A dollar-quoted body runs during the migration only as a DO block; a
# routine body runs later, when the routine is called.
DO_BLOCK = re.compile(r"\bDO(?:\s+LANGUAGE\s+[A-Za-z_][A-Za-z0-9_]*)?\s*$", re.I)


def identifier(token: str) -> str:
    """The table name PostgreSQL stores for an SQL identifier token."""
    if token.startswith('"'):
        return token[1:-1].replace('""', '"')
    return token.lower()


def executed(sql: str) -> str:
    """The migration text PostgreSQL executes, same length: comments, string
    literals and routine bodies are blanked, DO blocks are kept."""
    out: list[str] = []
    index, end = 0, len(sql)
    while index < end:
        if sql.startswith("--", index):
            stop = sql.find("\n", index)
            stop = end if stop < 0 else stop
        elif sql.startswith("/*", index):
            depth, stop = 1, index + 2
            while stop < end and depth:
                if sql.startswith("/*", stop):
                    depth, stop = depth + 1, stop + 2
                elif sql.startswith("*/", stop):
                    depth, stop = depth - 1, stop + 2
                else:
                    stop += 1
        elif sql[index] == "'":
            escapes = index > 0 and sql[index - 1] in "Ee" and (index < 2 or not re.match(r"\w", sql[index - 2]))
            stop = index + 1
            while stop < end:
                if escapes and sql[stop] == "\\":
                    stop += 2
                elif sql[stop] == "'" and sql.startswith("''", stop):
                    stop += 2
                elif sql[stop] == "'":
                    break
                else:
                    stop += 1
            stop = min(stop + 1, end)
        elif sql[index] == '"':
            stop = index + 1
            while stop < end and (sql[stop] != '"' or sql.startswith('""', stop)):
                stop += 2 if sql[stop] == '"' else 1
            stop = min(stop + 1, end)
            out.append(sql[index:stop])
            index = stop
            continue
        elif (quote := DOLLAR_QUOTE.match(sql, index)) and not (index and re.match(r"\w", sql[index - 1])):
            close = sql.find(quote.group(), quote.end())
            close = end if close < 0 else close
            body = sql[quote.end() : close]
            # Entries are single characters or whole blanked spans, so the
            # last 64 hold at least the last 64 characters of code.
            kept = executed(body) if DO_BLOCK.search("".join(out[-64:])) else " " * len(body)
            tag = " " * len(quote.group())
            out.append(tag + kept + (tag if close < end else ""))
            index = min(close + len(quote.group()), end)
            continue
        else:
            out.append(sql[index])
            index += 1
            continue
        out.append(" " * (stop - index))
        index = stop
    return "".join(out)


def migrated_tables(migrations: list[tuple[str, str]]) -> set[str]:
    """The tables left by applying (name, SQL) migrations in order."""
    tables: set[str] = set()
    for name, sql in migrations:
        text = executed(sql)
        statements = sorted(
            [(match.start(), "create", match) for match in CREATE.finditer(text)]
            + [(match.start(), "drop", match) for match in DROP.finditer(text)]
            + [(match.start(), "rename", match) for match in RENAME.finditer(text)],
            key=lambda statement: statement[0],
        )
        for _, kind, match in statements:
            if kind == "create":
                if match.group(1) is None or match.group(1).strip().upper() == "UNLOGGED":
                    tables.add(identifier(match.group(2)))
            elif kind == "rename":
                old, new = identifier(match.group(1)), identifier(match.group(2))
                if old not in tables:
                    raise SystemExit(f"{name}: renames table {old}, which no earlier migration creates")
                tables.discard(old)
                tables.add(new)
            else:
                listed = re.sub(r"\b(?:CASCADE|RESTRICT)\b", "", match.group(1), flags=re.I)
                for token in filter(None, (part.strip() for part in listed.split(","))):
                    found = re.fullmatch(NAME, token)
                    if found is None:
                        raise SystemExit(f"{name}: cannot read the dropped table {token!r}")
                    tables.discard(identifier(found.group(1)))
    return tables


def erd_problems(page: str, migrations: list[tuple[str, str]]) -> list[str]:
    """Differences between the ERD page and (name, SQL) migrations."""
    data = DATA.search(page)
    if data is None:
        return ["the page has no schema data; regenerate it with make docs-erd"]
    try:
        schema = json.loads(data.group(1))
        recorded_version = schema["schemaVersion"]
        drawn = {table["name"] for table in schema["tables"]}
    except (ValueError, KeyError, TypeError) as error:
        return [f"the page's schema data is unreadable ({error}); regenerate it with make docs-erd"]
    versions = []
    for name, _ in migrations:
        version = MIGRATION.match(name)
        if version is None:
            return [f"migration {name} is not named NNNNNN_name.sql"]
        versions.append(version.group(1))
    if not versions:
        return ["there are no migrations"]
    problems = []
    if recorded_version != max(versions):
        problems.append(f"the page shows schemaVersion {recorded_version}, the newest migration is {max(versions)}")
    tables = migrated_tables(sorted(migrations))
    if drawn - tables:
        problems.append(f"the page draws tables the migrations no longer create: {', '.join(sorted(drawn - tables))}")
    if tables - drawn:
        problems.append(f"the page lacks tables the migrations create: {', '.join(sorted(tables - drawn))}")
    return problems


def load_builder():
    """scripts/erd/build_erd.py, which owns the page and migration paths."""
    spec = importlib.util.spec_from_file_location("build_erd", ROOT / "scripts" / "erd" / "build_erd.py")
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load scripts/erd/build_erd.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    builder = load_builder()
    migrations = [(path.name, path.read_text()) for path in sorted(builder.MIGRATIONS.glob("*.sql"))]
    problems = erd_problems(builder.OUTPUT.read_text(), migrations)
    page = builder.OUTPUT.relative_to(ROOT)
    if problems:
        sys.exit(f"{page} is stale; regenerate it with make docs-erd:\n  " + "\n  ".join(problems))
    print(f"{page} matches the {len(migrations)} migrations")


if __name__ == "__main__":
    main()
