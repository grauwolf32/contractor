# Managed Evals release verification

This gate covers the managed experiment protocol and the browser journey from
[spec 30](../spec/30-managed-evals.md). It uses disposable PostgreSQL schemas,
the real Server, Python Runtime and production Node UI, and a deterministic
loopback model Gateway. It does not call a paid model or a live target.

## Reproduce

Use Go 1.25, Python 3.13/uv, Node 24.20.x, Corepack/pnpm 11.24.0,
PostgreSQL 17 and Chromium's host libraries. The database principal must be
able to create/drop schemas. Keep this database separate from application data.

```sh
export CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable'
export CONTRACTOR_EVAL_EVIDENCE_DIR="$PWD/.local/evidence/managed-evals"
make test-evals
make ui-typecheck ui-lint ui-test ui-build
make verify-public-api
```

`make test-evals` installs locked UI/Runtime dependencies and Chromium, runs
the full domain/store/service/coordinator race suites, Eval public-API tests,
and both process-stack browser journeys. Missing database configuration,
failed tests and Go test skips fail the command. The process suite allows
26 minutes: its serial managed matrix executes 48 ordinary Workers with
production heartbeat/release timing, including two children per Audit.

The companion browser fixtures use an independently running UI service:

```sh
cd ui
corepack pnpm dev --host 127.0.0.1 --port 4173
# In another terminal, from ui/:
CONTRACTOR_UI_E2E_BASE_URL=http://127.0.0.1:4173 \
  corepack pnpm exec playwright test e2e/evals-setup.spec.ts \
  e2e/evals-comparison.spec.ts e2e/evals-skills.spec.ts
```

Restart that service after changing the UI package version; build and runtime
versions are deliberately checked at startup. These browser fixtures replace
API responses and are separate from the real process evidence below.

## Evidence boundaries

| Layer | Assertions |
| --- | --- |
| Real native Workflow and Audit | Author/import two cases through UI, retain exact A/B variants, two repetitions, saved draft after reload, Prepare before Start, eight members per experiment, lost Start response, Server restart, original plan/start/deadline, ordinary executions, owner review and safe report |
| Real external Workflow and Audit | Independent Python standard-library producer, eight-member registration, full one-item inventory pages, exact outputs, immutable attributed results/assessments, explicit selection/finalization, lost response replays and the same inspection UI |
| Real ownership and privacy | A second owner cannot read experiment/member/pair/report/review/inventory routes; private canary is absent from Worker requests and exported reports |
| PostgreSQL faults | Contending claims and stale epochs, durable suboperation replay, Audit create-before-start cancellation, uncertain Run/deletion fences, frozen deadlines, stale selections and unavailable evidence |
| PostgreSQL accounting and scale | 10,000 expected members, complete snapshot counts across pages, bounded queries/responses, exact shared bins/percentiles/deltas, progress gaps; Audit discovery/check/assessment/retry inventory and deduplicated stages, overlapping child durations and missing reports |
| UI component and mocked browser | Stale CAS/readiness and review, recovery receipts, private import/storage separation, 390px/1280px layout, keyboard dialog dismissal/focus, chart tables and navigation, external controls and legacy links |

`tests/ui-stack/evals_producer.py` runs with `python3 -I -S`. The Runtime
environment also asserts that `playground_evals` is unavailable. Shared portable
schema names and copied conformance fixtures are protocol compatibility, not
runtime imports. No Playground checkout, daemon, URL or database is required.

The real Audit fixture uses two check children. Discovery/assessment roles,
rounds, retries, missing children and overlapping lifetimes use deterministic
PostgreSQL fixtures, not claims about extra real process campaigns. Likewise,
the 10,000-member test measures selected persistence/query behavior, not a
10,000-member browser authoring session or model throughput.

## Artifacts

The selected evidence directory contains:

- `gate.json`: implementation commit, worktree state, exact Go commands and
  passed/skipped test counts;
- `domain-store-service.jsonl`, `public-api.jsonl`, `process-browser.jsonl`:
  complete structured Go results;
- `native-evidence.json` and `external-evidence.json`: experiment IDs, frozen
  identity/timing and safe public reports;
- `browser/native` and `browser/external`: Playwright traces and comparison
  screenshots at both viewports.

Artifacts are local test output, not committed secrets or a hosted report.
The [implementation record](../plans/2026-09-20-managed-evals-ui.md) records
the completed run and optional Playground compatibility evidence.

## Optional Playground client

In the independent `playground-v2` repository:

```sh
cd evals
uv run --extra dev pytest tests/test_managed_evals.py
uv run --extra dev pytest
```

The managed suite covers Workflow/Audit in both modes, private local scoring,
full multi-role inventory paging, response-loss recovery, source/CAS/deletion
fences, frozen execution order and the original journal's preservation. The
full suite includes existing portable and direct CLI compatibility. Contractor's
required gate does not invoke or install this client.

These checks establish deterministic protocol, lifecycle and UI behavior.
They do not measure model quality, participant usability, V40 live readiness,
six-program dispatch, or justify publishing new default configurations.
