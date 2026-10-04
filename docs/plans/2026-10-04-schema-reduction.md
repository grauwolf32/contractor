# Schema reduction and SQL placement — 2026-10-04

Tasks V365-001–V365-022. Source analysis: the
[database schema review](../research/2026-10-03-database-schema-review.md).

## Goal

Reduce the number of tables without losing behaviour, and keep large SQL
statements out of ordinary Go code.

The schema has 91 tables. This plan removes 6 tables that have no production
reader and folds 14 strict 1:1 extensions into their parents, leaving 71.
V365-016 is deferred: Planner reports turned out not to be a 1:1 extension.
Every removal and merge keeps the observable API, CLI and UI behaviour; tests
that asserted the removed tables are rewritten against the new storage.

## SQL placement rule

A raw-string SQL literal of 10 or more lines in production Go code lives in a
`*_sql.go` file next to its caller: `store.go` keeps its statements in
`store_sql.go`. Each statement is a package-level variable named after what it
does, ending in `SQL`, with a doc comment that says what the statement does and
which function runs it. Moving a statement never changes its text. Statements
shorter than 10 lines may stay inline.

V365-004 adds a test that fails when a production literal of 10 or more lines
appears outside a `*_sql.go` file.

## Stages

Stage 1 moves SQL without changing behaviour. Later stages edit statements in
the new files instead of moving them twice.

| Task | Change | Tables |
| --- | --- | ---: |
| V365-001 | Move Audit and finding SQL (`auditstore`, `auditservice`, `findingintake`) | 91 |
| V365-002 | Move run execution SQL (`runstore`, `scheduler`, `telemetry`, `performance`) | 91 |
| V365-003 | Move managed-eval SQL (`evalstore`) | 91 |
| V365-004 | Move remaining SQL (`artifacts`, `credentials`, `projectstore`, `projectlifecycle`, `settingsstore`); add the placement test | 91 |
| V365-005 | Drop `stage_execution_reports` | 90 |
| V365-006 | Drop `planner_events` | 89 |
| V365-007 | Drop `configuration_publications` | 88 |
| V365-008 | Drop `eval_selection_history` | 87 |
| V365-009 | Drop `artifact_pins` | 86 |
| V365-010 | Drop `eval_view_members`; member pages read `eval_view_pairs` | 85 |
| V365-011 | Fold `runtime_credential_creations` and `runtime_credential_tombstones` into `runtime_credentials` | 83 |
| V365-012 | Fold `runtime_config_publications` into `runtime_config_versions` | 82 |
| V365-013 | Fold `llm_credential_identities` and `llm_credential_tombstones` into `credential_operations` | 80 |
| V365-014 | Fold `artifact_git_sources` into `artifact_versions` | 79 |
| V365-015 | Fold `workflow_run_metadata_labels` into `workflow_runs` | 78 |
| V365-016 | Deferred: `planner_execution_reports` holds several immutable reports per stage; `stage_metrics` is derived | 78 |
| V365-017 | Fold `audit_collection_receipts` into `audit_executions` | 77 |
| V365-018 | Fold `audit_coverage_rows` and `audit_proposal_items` into `audit_items` | 75 |
| V365-019 | Fold `audit_report_candidates` into `audit_artifact_links` and the review request | 74 |
| V365-020 | Fold `finding_proposal_retention` into `finding_proposal_receipts` | 73 |
| V365-021 | Fold `eval_project_dependencies` into `eval_submissions` | 72 |
| V365-022 | Fold `eval_view_generations` into `eval_projection_queue` | 71 |

Each schema task adds one forward migration that moves existing rows before it
drops a table, so an existing database upgrades in place. Where a whole-row
immutability trigger is the only reason for a side table, the merge replaces it
with a trigger that protects the immutable columns and allows the formerly
separate fields to be set once.

## Out of scope

- `audit_finding_contributions`: specifications 19 and 28 define contributing
  proposals, which are not implemented. Removing the table is a product decision.
- The five questionable tables: `artifact_scopes`, `artifact_blob_settings`,
  `audit_events`, `eval_collections` and `eval_plan_resources`.
- Defects DB-03 and DB-04 from the review; they do not change the table count.
- Squashing migrations into one baseline. That follows once this plan lands.
