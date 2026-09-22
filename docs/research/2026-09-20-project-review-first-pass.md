# Project review: first pass — 2026-09-20

Based on the [V60 plan](../plans/2026-09-20-project-review.md), baseline commit
`3ebe07cb` after V59 completion. The investigation used combined V59/V58 code;
completion metadata did not change the Runtime paths under review.

The first pass covered three areas: execution/runtime, data/Audit/product and
auth/quality/operations. Four defects were initially reproduced; the integration
run and independent verification added two more, for a total of **six confirmed
findings**. This is not a completed review of all subsystems. Deeper checks
V60-005–010 remain a separate queue.

**Implementation outcome: PR-01–06 are fixed and verified.** PR-06 was closed
in [V60-012 with report-retry verification](2026-09-20-run-deletion-audit-revisions.md).
The [original combined integration verification](../../tasks/evidence/v60-integration.json)
records original implementation hashes, actual commands and remaining limits.
The descriptions below preserve the defects as observed when discovered.

[First-pass evidence](../../tasks/evidence/v60-001.json) contains exact commands
for executed checks, original probe results, a snapshot of concurrent branches
and task-graph checks. Source line numbers below refer to the state before
PR-01–04 fixes, not subsequent line shifts.

## Confirmed defects

### PR-01 — P2: a valid password cannot be submitted through HTTP login

`internal/httpapi/public/auth_handlers.go:17,40` limited raw JSON to 2048 bytes;
`internal/auth/password.go:28` permits passwords of 12–1024 UTF-8 bytes. Both
rules appear in spec 06's authentication section, but are inconsistent after
JSON escaping.

Reproduction used the real auth service and HTTP handler. HashPassword and
direct `Service.Login` succeed for all variants; password length is 1024 UTF-8
bytes. HTTP with compact JSON serialization produces:

| Password | JSON size with test username | Before the fix |
| --- | --- | --- |
| ASCII without escaping | 1058 bytes | 200 |
| Backslash or quote | 2082 bytes | 400 |
| U+0001 serialized as `\u0001` | 6178 bytes | 400 |

The largest standard compact serialization for a permitted 64-character ASCII
username and 1024 single-byte control characters occupies
`29 + 64 + 6 × 1024 = 6237` bytes. [V60-002](../../tasks/v60/v60-002-login-json-body-bound.yml)
therefore raises only the raw-body cap to a justified 8 KiB while preserving the
decoded password bound. This does not promise acceptance of unlimited JSON
whitespace. Regression coverage must also check overflow, malformed/unknown
fields, origin and rate-limit safeguards.

### PR-02 — P2: provenance returns new data under old revisions

`internal/httpapi/public/audit_review_handlers.go:311–370` reads Audit/finding
revisions, then calls `ListFindingProvenance` and returns the original envelope.
The service rereads the finding in `internal/auditservice/finding_review.go:403`.
A concurrent mutation between reads can change both the list and `SupportsCurrent`.
This path had no final revision check.

Spec 19's Finding review/provenance pagination contract and V25-011 R8 require a
conflict to prevent mixed history. A deterministic test with the real handler
and a fake interleaving returned `200`, revisions `9/4` and a new current assessment
after mutation to `10/5`. The expected `409` was absent.

This proves HTTP orchestration behavior for the specified interleaving; it is
not a measurement of PostgreSQL race probability. The fix belongs to
[V60-003](../../tasks/v60/v60-003-provenance-consistent-reads.yml): consistency checking
covers reading and hydration while preserving the existing cursor contract.

### PR-03 — P2: provenance waits for a second connection while holding the first

`finding_review.go:500–532` holds `pool.Query` rows and calls receipt hydration
inside `rows.Next()`. For that hydration, `findingintake/postgres.go:209` requests
another connection from the same pool. pgxpool releases the first connection
only when rows are closed or exhausted.

Real PostgreSQL, a separate disposable test schema and the same retained finding
produced: `MaxConns=2` — PASS in about 0.03 s; `MaxConns=1` — deadline after about
1.99 s with `authorize Audit finding receipt: context deadline exceeded`.
The `GetFinding` control passes with both pools. All connections are released
on return: this is nested waiting, not a leak.

V60-003 must close rows before hydrating receipts. Regression coverage checks
a real minimal pool, owner/exact identity, retention and preserved V59 page=200.
The same effect is logically possible under concurrent saturation of a larger
pool, but the first probe did not measure that separately.

### PR-04 — P1: redaction breaks the mandatory Audit release gate

`scripts/test-audit-completion-e2e.py:118` applies `redact` to the entire Go JSON
event line before storage and validation. `redact:104` replaces every occurrence
of the DB password. In CI that password is `contractor`, so every Package
`github.com/grauwolf32/contractor/...` changes. Password `pass` similarly changes Action.

Offline reproduction builds a full report from the actual mandatory matrix:
unchanged events pass all 54 required cases; after current redaction with the
CI password, the verifier reports mandatory cases missing. With a long password
that does not match metadata, the same matrix passes. This is a deterministic
evidence-processing defect; the investigation did not execute a full GitHub Actions run.

[V60-004](../../tasks/v60/v60-004-audit-gate-event-redaction.yml) must preserve framework
identity/status and redact diagnostic data. Weakening required cases or changing
the CI password is not a fix. Tests must compose subprocess capture with the
verifier, including secret-free logs, skip/fail/missing/malformed and nonzero exit.

## Additional confirmed findings

### Addendum PR-05 — P1: the mandatory matrix lagged behind V57-004

The real `make test-audit-completion-e2e` successfully ran 331 Python tests, then
rejected the absence of required
`test_artifact_observations_survive_a_reminder_and_join_verified_publication`.
History at `7f553b22` shows that the test was renamed to
`test_audit_typed_assembly_preserves_reminder_publication_and_refs_without_model_decode`;
its previous assertions were preserved and strengthened, but the matrix was not updated.

This is a V57-004 integration omission. [V60-011](../../tasks/v60/v60-011-audit-gate-runtime-matrix.yml)
preserves the required minimum of `1` for the current strengthened test. A new
offline AST check compares the matrix with real Python test declarations: it
fails on the old entry and passes after correction. Synthetic JUnit probes copied
names from the matrix itself, so they could not detect this drift. AST checks
do not prove collection/execution; that remains the full gate's responsibility.

### Addendum PR-06 — P2: Run deletion does not invalidate Audit revision

`internal/runstore/delete_store.go:98,120–133` changes execution tombstones and
receipt retention without changing Audit revision. Spec 19:1571–1576 requires
revision advancement for projection-visible execution/receipt/provenance changes.

A real PostgreSQL probe used production `ImportIntoAudit`, followed by
`DeleteReleasedTerminalRun` between provenance reads. Old caller pins were
accepted even though `Origin.RunDeleted` changed from `false → true`; Audit
revision remained `2` and finding revision `1`. This is a separate upstream
invalidation defect: V60-003's comparison of correctly captured revisions cannot
detect a mutation that does not itself advance the revision.

[V60-012](../../tasks/v60/v60-012-run-deletion-audit-revisions.yml) is complete:
managed executions, native receipts and all destination Audit holds participate
in atomic invalidation. Lock ordering is coordinated with import/purge; rollback,
HTTP pins/cursor and report retry were checked on PostgreSQL. `Attempt.RunDeleted`
is not equated with `Origin.RunDeleted`: the fields may refer to different Runs.

The separate task has a specific reason: the report importer writes immutable
`report.json`/`report.md` before revision CAS, and the bytes include `Audit.UpdatedAt`.
The code paths in `auditimport/report.go:223,262–325` and
`auditimport/artifacts.go:166–167` show that a naive revision/timestamp bump
between write and CAS can leave an existing immutable name with incompatible
bytes on retry. At the first pass, this was a code-derived conclusion not yet
checked by a fault probe. V60-012 reproduced the immutable collision. The adopted
and verified rule is that deletion during `finalizing` advances revision while
preserving report timestamp and bytes for retry. Pending report review compares
its subject with the retained candidate, so no new equality check against the
current Audit revision should be introduced there.

## Accepted decisions and rejected suspicions

| Path | Verified basis | Classification / next step |
| --- | --- | --- |
| Claim ID absent from terminal SQL | Historical decision `7ca8daa6`, then `docs/reviews/architecture-orchestration.md:43–45`, bounds protection at ErrClaimLost detection. Spec 20 assumes one active Server/Scheduler. | **accepted**. The pre-detection window can be tested as an accepted limitation; do not introduce multi-server fencing disguised as a local fix. |
| Cleanup `_stop` has await without its own timeout | The enclosing allocation owner bounds the full operation by deadline; failed cleanup leaves Runtime fenced and triggers force-exit. | Suspicion of a locally missing timeout **rejected**. An additional process check should verify absence of child resources after real exit. |
| Project delete might bypass Artifact write admission | Current CAS is in SQL; migration 39 admission triggers take FOR SHARE, and the cleanup exception is restricted to the exact Audit namespace. | Superficial handler TOCTOU suspicion **rejected**. Next: controlled DB races between write/update/publication and delete, plus rollback. |
| A repeated Run does not overwrite Project output | Spec 17 requires create-only publication; Scheduler preserves a successful Run on nonfatal Project publication failure. | **accepted**. Check concurrent publication and repeat with a deleted/changed exact source. |
| Additional tool-free finalizer | V21-001 confirms the ADK/tools/structured-output limitation; V57-004 preserved ordinary completion. | **accepted** as the current contract. V57-003 was archived after reassessing its research value; whether an exact-copy LLM call is needed remains in the architecture review. No change to this contract was implemented. |
| No A2A connection reuse | V57-005 measured the TLS effect and retained transport because of retry/expiry/certificate differences and unproven production benefit. | **accepted** pending new measurements and agreed semantics. |
| Different Go/Python JSON escaping under a shared wire cap | Different serializers may count bytes differently; a cross-language negative case has not yet run. | **hypothesis**; check with a shared fixture matrix in V60-006. Do not change the shared cap now. |
| Missing SARIF/structured finding analysis | Spec 28 is explicitly draft/not implemented. | **existing backlog**, not a regression in a delivered contract. |

## What was checked and what remains

The execution/runtime pass reviewed durable finalizing recovery, cancel/claim
cleanup, shielded session cleanup, two-phase allocation release, trusted result
assembly and Go A2A correlation. Selected offline checks passed: seven Scheduler
tests, four A2A groups, ten cancellation/session tests and 29 decoder/assembler
tests. This is not a full fault/process suite.

The data pass reviewed Artifact CAS/deletion fences, provenance/retention,
Project output publication and UI repeat. New negative probes for PR-02/03 fail
on the original code, as expected when reproducing defects. The PostgreSQL probe
used only a dedicated test schema, removed afterward.

The quality pass reproduced PR-01 through real HTTP and PR-04 through a local
report/redactor/verifier composition, and reviewed CI/gate composition and
migration fences. This first pass did not run the full release gate, a fresh
install, restore, browser journeys or live models. V59's 435 UI tests and full
Go tests are not relabeled as new V60 results.

Final fixes, original implementation hashes and regressions actually executed
are recorded in V60-002–004 task files and first-pass evidence. The six tasks
V60-005–010 remain pending until separate, deeper execution.

## Implemented fixes and combined verification

| Task | Status / result |
| --- | --- |
| V60-001 | Plan and first pass complete; accepted decisions separated from proven defects and verification gaps. |
| V60-002 | Complete: 8 KiB login body, 19 new HTTP subcases; maximum username/escaping, Content-Length/chunked, 8192/8193, decoded bounds and safeguards. |
| V60-003 | Complete: optional internal pins passed from HTTP; before/after revisions read with an owner-scoped JOIN; rows closed before batch hydration. PostgreSQL pool=1 checks 201 unique receipts, repeated records, missing receipts and concurrent revision changes. |
| V60-004 | Complete: Go event identity/status preserved; diagnostics redacted after JSON parsing; arbitrary extra fields not retained. 14 subprocess composition regressions supplement existing required tests. |
| V60-011 | Complete: mandatory Runtime test updated without weakening assertions/minimum; offline declaration checks detect stale names. Full gate passed with a PostgreSQL password matching CI. |
| V60-012 | Complete: Run→Audit lock order and atomic revision bump; finalizing timestamp preserves immutable retry. Concurrent import/purge, rollback, terminal Audit, pending review and real HTTP stale pins/cursors verified. |

On combined code, `go test -count=1 ./...`, `go vet ./...`, affected Go packages
with real PostgreSQL and `-race`, the public API gate and mandatory pagination
gate passed. `make test-audit-completion-e2e` passed **158 Go cases and 331 Runtime
tests with no selected skips**. The initial failure caused by the old test name
is retained in [V60-011 evidence](../../tasks/evidence/v60-011.json), not hidden
by the successful repeat. This is the full Audit completion gate, not the full
`release-verify`.

V60 changes neither UI nor generated OpenAPI clients. Results for 435 UI tests,
typecheck/lint/build and byte-reproducible generation belong to the separately
recorded V59 verification. Local PostgreSQL runs use their own disposable
container and unique schemas; live services and models are untouched.

Additional V60-012 run: **173 Go cases and 331 Runtime tests with no selected
skips** in the Audit completion gate, five affected packages with PostgreSQL and
`-race`, and the public pagination gate. [Evidence](../../tasks/evidence/v60-012.json)
distinguishes this run from the original 158 Go cases above.

## Separate observation for V60-008

During V60-012, a temporary PostgreSQL probe tested repeated import of one receipt:
replay into the same Audit succeeds; import into a second compatible Audit fails
with `audit_findings_pkey`, SQLSTATE `23505`. The transaction rolls back, leaving
holds A=1/B=0. Migration 041 derives the global `finding_id` solely from receipt
ID, while the holds table is keyed by `(receipt_id, audit_id)`.

This is a confirmed implementation limitation. No explicit accepted requirement
for one receipt in multiple Audits was found: existing support for multiple
Audits and compatible import does not fully define this edge case.
[V60-008](../../tasks/v60/v60-008-audit-product-journeys-review.yml) must clarify the
contract, expected HTTP outcome and safe identity/migration strategy, then create
a correction task. V60-012 checks multiple destination Audits for different
receipts from one Run and does not claim to fix this collision.
