# V38-006 implementation

Scope: finish assessment, selection and complete comparison APIs from spec 30,
starting from the preserved V38-006 work in `feat/v38-managed-evals`.
Implementation branch: `feat/v38-006-comparison`, based on main `15974cba`.
Implementation commit: `9fc021501ce34389ecc7f35ca65bdf7959a25ce9`.

- [x] Preserve the original work and move a copy onto current main.
- [x] Refactor comparison, collection and persistence; verify evidence invalidation.
- [x] Implement attributed result/assessment, owner review and CAS selection APIs.
- [x] Implement snapshot pages, pair detail, charts, reports and execution inventory.
- [x] Synchronize OpenAPI and generated clients.
- [x] Verify deterministic fixtures and disposable PostgreSQL integration, including
      owner boundaries, concurrency, large matrices and Audit inventory gaps.

## Implementation

The collector now separates native record construction, selected-record
validation, exact output observation and nullable usage accounting. Registered
checks have separate policy validation and bounded execution. Comparison groups
members and suite totals in linear passes; chart bins use direct assignment.
Collection, snapshot, chart and diagnostic limits have named constants.

Public routes cover immutable results/assessments, owner review, CAS selection,
pairs and selected record detail, charts, reports and paginated execution
inventories. External results cannot override observed execution/usage. Record
replay preserves the receipt and producer-activity clock. Losing a selection race
preserves the review. Later/native/external records never silently replace a
selection; only the initial native record pair uses an audited system CAS.

Migration 59 adds selected pair/chart projections, immutable view/selection
history, evidence invalidation and bounded observation reads. A bulk matrix
insert invalidates the experiment queue once per statement. Publication includes
all expected members and summaries in one transaction; readers retain a stale
complete generation during re-projection. Evidence recovery creates a new
snapshot, including when the selected documents match an older generation.

JSON/Markdown reports retain source attribution and omit private rubrics and
free-text assessment reasons. Owner review alone returns pinned private criteria.
Execution inventories retain all Audit roles, rounds, retries and deleted refs.
Usage counts unique Run/stage observations and uses parent Audit wall duration.

## Verification

Disposable PostgreSQL 17, isolated schemas, no paid models, live targets or
shared demo changes:

- Required `go test -race -count=1 ./internal/evaldomain ./internal/evalstore
  ./internal/evalservice ./internal/httpapi/public -run 'Test.*Eval'`: passed,
  183 tests/subtests, no skips. Includes native/external revisions, CAS retention,
  missing evidence, interrupted publication, authentication/CSRF and owner limits.
- Additional race suites for `evalcoordinator`, `projectlifecycle`, `auditstore`,
  `artifacts` and `persistence/postgres`: passed. The migration inventory test was
  updated for the new pair/chart tables and rerun successfully.
- 10,000-member persistence fixture: 5,000 pairs across 50 suites, 50 pages,
  two SQL queries per selected page, maximum response 632,876 bytes. Complete
  terminal/quality/token pair counts are 3,750/2,500/2,500, preserving trace-small's
  3/2/2 ratio. Exact percentiles and at most 20 bins agree with the selected cohort.
- Real stage-metric SQL fixture: check/discovery/assessment and retries total
  100 tokens; a missing child's metrics leave 60 known tokens with partial
  coverage. Replayed stage snapshots/inventory do not double totals. Parent
  duration remains 2,000 ms. Private metric fields and display aggregates are
  excluded. Full authoritative inventory pagination is covered separately.
- 500 recorded progress observations return at most 200 real buckets; missing
  history and unavailable suite counts stay gaps, and older snapshots exclude
  later observations.
- `make verify-public-api`, both UI TypeScript configurations and `go vet ./...`:
  passed. UI uses the existing dependency directory through a local symlink and
  `pnpm_config_verify_deps_before_run=warn make ui-typecheck` to prevent pnpm from
  reinstalling shared dependencies. Node 26.7.0 emitted the package's Node 24.20.x
  engine warning; TypeScript 5.9.3 completed both configurations.
- OpenAPI and Go/TypeScript client regeneration is byte-for-byte stable. The
  Python generator passes Ruff formatting/lint. SQL formatting was checked for
  unchanged tokens; Go formatting and `git diff --check` pass.
- All 29 original uncommitted V38-006 files still match their saved SHA-256 values.

The large fixture tests persistence and selected reads, not a 10,000-member
portable authoring journey. Browser setup/comparison (V38-007/008), the optional
Playground client (V38-009) and process/browser release gates (V38-010) remain
separate tasks. The earlier main merge included only V38-002–005.
