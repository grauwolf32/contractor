# Database ERD generator

Builds [`docs/spec/database-erd.html`](../../docs/spec/database-erd.html), an
interactive diagram of the PostgreSQL schema: tables grouped by subsystem,
keys, what each table is for, which processes use it, and audit verdicts.

```sh
CONTRACTOR_TEST_DATABASE_URL=postgres://… make docs-erd
```

`ERD_DATABASE_URL` overrides the server. The script creates a scratch
database there, applies the migrations with `contractor-server migrate`, reads
the catalog, lays the diagram out with [ELK](https://eclipse.dev/elk/) and
drops the database. It needs `psql`, Go and Node.js.

| File | Contents |
| --- | --- |
| `meta.json` | Subsystems, and per table: subsystem, short description, why it exists and the processes that use it; verdict flags; tables removed by the schema reductions and where their data went |
| `template.html` | The page: styles, renderer and the `__DATA__` slot |
| `layout.cjs` | ELK layout of both views (keys and all columns) |
| `build_erd.py` | Orchestration and catalog queries; takes per-package table usage from the table ownership scan |

After a migration adds, drops or renames a table, update `meta.json`; the
build fails and names every table it cannot describe, and every flag or
description that no longer matches the schema. Give the table its owner
package and the same group in the table ownership registry
[`internal/persistence/table_ownership.txt`](../../internal/persistence/table_ownership.txt)
too: `TestTableOwnershipMatchesTheSchemaDocuments` fails while `meta.json`
and the registry name different tables or groups (PER-R23 in
[persistence.md](../../docs/spec/platform/persistence.md)).

The "where in code" list of each table names the packages, by directory as
the registry names them, whose SQL the ownership scan of
`TestSQLAccessFollowsTableOwnership` sees reading or writing the table:
`build_erd.py` runs `TestERDUsageMatchesTheSQLScan` in
`internal/persistence` with `CONTRACTOR_ERD_USAGE_OUT` set to get them. A
package that reaches a table only through a migration function or trigger is
not listed.

Regenerate the page with every migration and whenever a package starts or
stops using a table. `make lint` runs two checks that need no database:
[`scripts/check_database_erd.py`](../check_database_erd.py) fails when the
page's `schemaVersion` is not the newest migration or when its tables differ
from those the migrations leave after their `CREATE TABLE`, `DROP TABLE` and
`ALTER TABLE ... RENAME TO` statements, and `TestERDUsageMatchesTheSQLScan`
fails when a usage list differs from the scan.
