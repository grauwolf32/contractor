# V32 integration and demo update

Historical integration/deployment record from 2026-09-06. Subsequent commits and
fresh checks are recorded in the [main review](2026-09-07-main-uncommitted-review.md);
the uncommitted status and test counts below describe the original snapshot.

Integrated on 2026-09-06 from `v32-performance-ui` at
`1519b7ad` onto main baseline `5a6ebc3d`. V32-005, V32-006 and V32-008
were already implemented and marked completed in that branch; main still had
the earlier statuses and planned API contracts. Their original implementation
commit references are preserved. Integration changes remain uncommitted.

## Integration adjustments

- Preserve existing migration `000052_run_stage_resumptions.sql`; introduce
  performance policy as `000053_allocation_performance_policy.sql`.
- Give the resumed-Run allocation test fixture an explicit unsupported
  performance policy, as required for newly recorded allocations.
- Preserve current Run-detail wording and all other main features, including
  resume controls, content-capture settings and trusted private-network origins.
- Disambiguate the browser stack's Worker telemetry checkbox now that the
  same group contains the capture-content checkbox.
- Refresh the specification/index status while retaining unrelated in-progress
  configuration, documentation and task changes in the shared working tree.

## Fresh verification

- Race-enabled V32-005 API/lifecycle tests passed for Control Plane, telemetry,
  public HTTP and Scheduler.
- Real PostgreSQL tests passed for performance, telemetry, RunStore and public
  HTTP; the resume fixture was corrected and RunStore rerun successfully.
  Tests used isolated schemas, not application data.
- `make verify-public-api ui-typecheck` passed.
- UI lint, all 255 tests in 43 files, six Node-server tests and production
  build passed. Node 26.7.0 emits the existing declared-24.20.x engine warning.
- The full production browser stack passed all 21 scenarios after the selector
  adjustment, including Performance and completed-allocation history.

The general `go test ./...` also exposed an unrelated stale matrix reference
already present in main after V42: `tests/e2e/worker_summarizer_matrix.yml`
still references `test_gateway_model_disables_hidden_provider_retries`, removed
by the Gateway-client change. Its replacement tests bounded SDK retries under
a different contract. This integration does not rewrite that policy or claim
that the entire current-main `make verify` gate is green.

## Demo deployment

- Release: `.local/demo/releases/performance-integrated-20260906`.
- Applied exactly one migration; the database ledger is now version 53.
- Restarted only `contractor-demo-server.service` and
  `contractor-demo-ui.service`; there were no active Runs/allocations.
- Runtime processes, credentials and performance/profiling startup choices were
  left unchanged. Existing loopback/LAN UI and API origins remain available.
- Operator configuration is pinned to the release's clean baseline snapshot
  through `performance-release.conf` under the user service's runtime drop-ins,
  avoiding deployment of unrelated dirty configuration edits. Managed config
  storage remains the existing demo directory.
- UI deep links `/operations/performance` and
  `/operations/allocations/completed` serve the verified build.
- Authenticated current performance, history and allocation-history endpoints
  returned HTTP 200; collection is enabled and collector/writer loss counters
  were zero. Existing terminal allocations are visible.

Older allocations without a pinned collection policy or resource report remain
explicitly unavailable/legacy; no historical CPU/RSS is fabricated. The old
Server binary and UI release are retained. A rollback must keep migration-ledger
compatibility in mind; do not remove migration 53 or blindly run an older
migration command. The deployment performs no user Run or Audit replay.
