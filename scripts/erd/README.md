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
| `meta.json` | Subsystems, and per table: subsystem, short description, why it exists and the processes that use it; verdict flags; tables removed by V365 |
| `template.html` | The page: styles, renderer and the `__DATA__` slot |
| `layout.cjs` | ELK layout of both views (keys and all columns) |
| `build_erd.py` | Orchestration, catalog queries and per-package table usage |

After a migration adds, drops or renames a table, update `meta.json`; the
build fails and names every table it cannot describe, and every flag or
description that no longer matches the schema.

Regenerate the page with every migration. `make lint` runs
[`scripts/check_database_erd.py`](../check_database_erd.py), which needs no
database: it fails when the page's `schemaVersion` is not the newest migration
or when its tables differ from those the migrations leave after their
`CREATE TABLE`, `DROP TABLE` and `ALTER TABLE ... RENAME TO` statements.
