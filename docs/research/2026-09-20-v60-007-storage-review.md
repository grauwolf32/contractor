# V60-007 — Artifact authority and persistence recovery

Review baseline: `8edeabf2`, isolated worktree `review/v60-deep-review`,
2026-09-20. This report covers the Artifact and storage slice of V60; it does
not claim whole-project correctness. Implementation of V38, V40, V55 and V61
is outside this review. No production data, deployment or live model was used.

## Contracts and source history

Reviewed specifications 03, 08, 09, 10, 17, 18, 23 and 24 against the public
Artifact handlers, allocation-bound private handler, transaction-bound
repositories, physical blob adapters, lifecycle purge and PostgreSQL migrator.

- Public Project/Run access first resolves ownership using the authenticated
  principal. Private access derives RunScope from the live allocation's
  principal/instance grant; request fields cannot choose another scope.
  Memory notes and Skill packages retain these Artifact authority boundaries.
- Project write admission is enforced again in SQL by
  `contractor_require_active_project`, not only by the HTTP precheck. Its
  `FOR SHARE` lock serializes create/update/publication with Project deletion.
- CAS updates the binding and creates its immutable version/revision together.
  Exact reads select a retained revision; they do not fall forward to current.
  Purge locks scope, versions and blobs in a stable order, then checks surviving
  references using fresh READ COMMITTED statement snapshots.
- Filesystem publication stages complete bytes before metadata publication.
  Failed/ambiguous commits may leave orphans; known committed references must
  never be eagerly unlinked. Physical generation keys prevent old cleanup from
  deleting a recreated same-digest blob. Offline cleanup closes query rows
  before subsequent database operations.
- `ApplyMigrations` holds one transaction advisory lock and applies pending DDL
  and ledger rows in one transaction. Recorded names/checksums and unknown
  versions are checked before applying new migrations.

Relevant decisions include `76f72511` (filesystem transaction integration),
`69fb7f32` (offline cleanup), `4edd8d52` (backend release parity), `ab63123a`
(verify/repair missing deduplicated objects without overwriting corruption),
`e6bf1c1f` (immutable Git-source provenance), `4cfd75c8` (bounded database waits),
`5d286dea` (concurrent reference-safe collection), `381a86f7` (Project deletion)
and `40ac02fd` (bounded Memory namespace enumeration). Their contracts explain
why physical orphans, ephemeral filesystem loss and offline cleanup are
accepted behavior rather than newly discovered defects.

## Controlled verification added by this review

`internal/artifacts/storage_review_postgres_test.go` adds actual PostgreSQL
interleavings. It waits for `pg_stat_activity` to confirm that the competing
transaction is blocked before committing the winner; a simultaneous goroutine
start alone is not counted as evidence of the intended ordering.

| Scenario | Assertion |
| --- | --- |
| Stale CAS, PostgreSQL and filesystem | First writer holds its transaction; second writer waits, then receives ArtifactConflict. Exactly two revisions remain; original and winning exact refs both read their own bytes. |
| Project create/update/publication versus deletion, both backends | Twelve cases cover both winning orders for all three mutations. Deletion first rejects the mutation without changing its target; mutation first commits before deletion and remains readable for cleanup. |
| Cleanup with one database connection | Filesystem write, same-digest CAS update, both exact reads, terminal Run purge and offline cleanup finish with MaxConns=1, no remaining references/orphans and no acquired connection. |

`internal/persistence/postgres/storage_recovery_review_test.go` installs the
real embedded migration prefix through version 47 with a historical inline
blob. A held relation lock stops migration 48 at its actual blob ALTER TABLE,
after its earlier statements have executed. Both context cancellation and
`pg_terminate_backend` interrupt that database session. The test verifies that
the ledger remains at 47 and the partially created settings table is absent,
then runs the normal migrator to the latest embedded schema. Historical
payload bytes, digest, length and selected PostgreSQL backend survive, and a
second migration run is a no-op. Separate cases reject a newer ledger version
and checksum drift without applying further ledger rows.

Existing focused suites additionally cover exact input/Skill forks, frozen
Project outputs, Git provenance after source movement, shared references during
purge, retained publication racing purge, corruption/missing content, lost
acknowledgements, concurrent dedup repair, generation-safe unlink and offline
cleanup. Existing migration tests cover concurrent fresh install, constraints,
historical manual-resumption upgrade, model-free allocation upgrade and
migration-lock cancellation with a one-connection pool.

## Results and evidence levels

All three required gates passed on 2026-09-20. Machine-readable command times,
counts, source/log hashes and toolchain details are in
[`tasks/evidence/v60-007.json`](../../tasks/evidence/v60-007.json). Raw local logs
are under `.local/v60-review/v60-007-*`. The environment used Go 1.25.6,
PostgreSQL 17.11, Python 3.13.14, Node 24.20.0, pnpm 11.24.0 and Podman 6.1.0.

| Command | Exit / wall time | Executed evidence |
| --- | --- | --- |
| `make test-artifact-integration` | 0 / 2.640 s | One Go/Python mTLS lifecycle test; no skips. |
| `make test-artifact-blob-backends` | 0 / 295.989 s | 784 Go PASS records including subtests; 23 Python cases; both production container backends; no skips. |
| `make test-git-artifacts` | 0 / 426.286 s | 939 Go PASS records including subtests; 24 Python cases; 486 UI unit tests in 69 files; five Chromium cases; both production Git container backends. |

The Git gate has one explained host-side `TestReadOnlyGitProbe` skip: that
helper runs inside `TestRealGitReadOnlyContainer`, where its actual child test
passed. No mandatory case was skipped. Counts include subtests and overlapping
suites; they are not a claim of that many distinct independent scenarios.
The initial focused review runs also passed: 17 Artifact and six migration
PASS records, with no skips. The same added tests then ran in the required
gates. Upgrade evidence here reaches schema 61; the subsequent additive
V60-027/schema-62 correction has its own upgrade regression and evidence.

`make test-artifact-integration` is a real Go/Python mTLS protocol exercise with
an in-memory repository. Database evidence comes from the PostgreSQL suites;
it must not be inferred from this command alone. The blob and Git release
gates additionally build and run the production Server in scratch Podman
containers, using a real Python Runtime, isolated PostgreSQL and fake Gateways.
Their PostgreSQL/filesystem cases exercise public/private boundaries and
read-only roots. Git's UI gate runs Chromium against the production static UI
with Playwright-routed API fixtures. It proves the browser interactions; real
Git import and persistence are verified separately by the production container
tests, not by those mocked browser responses.

## Findings and remaining boundaries

No confirmed production defect has been found in this slice. The hypotheses
that stale CAS could publish a losing revision, Project deletion could be
bypassed by a racing mutation, filesystem cleanup could deadlock on its own
single connection, and interrupted blob DDL could leave a partially upgraded
ledger were rejected by the new controlled tests.

Physical orphan retention after an ambiguous transaction or externally owned
transaction is intentional; offline cleanup is the recovery path. Filesystem
power-loss durability, populated-store backend switching and S3 remain outside
the supported contract. Retained metadata alone does not promise that an
ephemeral filesystem volume still contains bytes. The backup/restore
walkthrough is recorded separately by V60-009; the migration tests here are
not represented as a backup restoration.
