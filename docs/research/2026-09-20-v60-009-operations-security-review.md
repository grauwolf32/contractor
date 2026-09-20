# V60-009 — Authentication, gates and operational recovery

Reviewed 2026-09-20 from `8edeabf2` in isolated branch
`review/v60-deep-review`. Source corrections from V60-026 through V60-040 are
part of the frozen full-release snapshot, with committed main through `bc32a76b` merged
before that invocation. No production deployment or live model calls.

## Source, decisions and representative verification

| Area | Trace and result |
| --- | --- |
| Public credentials | `auth_handlers.go` requires exactly one supported Authorization header and never falls back to a cookie after a supplied bearer fails. Cookie authentication rejects duplicate session cookies; unsafe requests require one allowed Origin and one session-bound CSRF token before the domain handler. New ten-case regression verifies actual Run cancellation stays untouched on rejection and succeeds for the valid control. |
| Login and sessions | V60-002/`80ac450d` separates escaped JSON body size from supported decoded password size. Existing tests cover password limits, owner-only nonsymlink bootstrap, rate limits, cookie flags, expiry, revocation and process-local restart. `5a6ebc3d` deliberately permits configured trusted private-network demo origins; it does not authorize arbitrary origins. |
| Private authority | `3e359fb9` binds Runtime registration/allocation to the mTLS certificate principal. `make test-mtls` passed 10 Go events and five Python cases; public bearer/cookie credentials do not replace private certificate authority. Certificate helpers, private Artifact integration and full process gates cover separate levels. |
| Secret projections | Login clears the password field and wipes its temporary byte slice. Errors expose bounded codes. New negative requests assert no bearer, password, cookie or CSRF value in the response. Existing credential tests check redacted projections and required owner-only master-key material. This is not a claim that every allocation of a Go string can be erased from memory. |
| Fresh install and upgrade | V60-007 runs real migrations, cancellation/backend termination during historical blob DDL, retry, checksum drift and unknown-newer-version rejection. V60-027 additionally checks schema 61 to 62 without rewriting legacy finding history. Process gates use the production binary, isolated PostgreSQL and fake Gateways. |
| Backup recovery | New `tests/integration/restore` uses actual `pg_dump`/`pg_restore`, fresh databases and both payload backends. It removes the original database and filesystem bytes before restore; exact old/latest revisions, binary payloads, digests, idempotency and CAS survive. Filesystem metadata alone cannot recover missing bytes. The operational guide records the required consistent offline snapshot and separately retained keys/configs. |
| Gate credibility | Make prerequisites and CI were inspected before execution. CI runs `release-verify` with PostgreSQL 17, locked uv/pnpm dependencies and Node 24.20. The first combined invocation shares common prerequisites once. V60-031 found that browser aliases omitted promised fixture files; the corrected full UI-stack gate is rerun separately with seven explicit files and validated Playwright execution evidence. Runtime opt-ins and host-only Git helpers are recorded separately from mandatory executed cases. |
| Performance | Existing metrics use fixed dimensions, bounded retention and separate diagnostic connections. Three isolated child samples per scenario measured enabled/disabled instrumentation, collection and profiling; the real PostgreSQL and browser cases belong to the release gate. No unmeasured optimization was introduced. |

History also includes `5ea78cc5` (production configuration gate) and the V60
redaction, transaction-reader and session-generation corrections. They retain
safe diagnostics and pinned runtime behavior, not broad authorization based on
request-supplied owner or allocation IDs.

## Measurements

Environment: Linux amd64, eight logical CPUs; Go 1.25.6; Python 3.13.14;
Node 24.20.0/pnpm 11.24.0; disposable PostgreSQL 17; PostgreSQL client tools
18.6; locked Runtime/UI dependencies. `GOMAXPROCS=2`, `GOFLAGS=-p=2 -v`.
Other review suites shared this host. These are reproducible local observations,
not universal performance thresholds.

Command: `go run ./tools/performancebench -repetitions=3 -http-requests=10000
-collection-cycles=240 -profile-seconds=1` (10.30 s).

| Scenario | Mean throughput/s | Mean p95 | Mean RSS |
| --- | ---: | ---: | ---: |
| HTTP instrumentation disabled | 4,891,350 | 204 ns | 10.9 MiB |
| HTTP instrumentation enabled | 939,680 | 1,800 ns | 14.9 MiB |
| Collection enabled | 2,215 | 1.092 ms | 15.2 MiB |
| Profiling off / idle / active | 5,186,466 / 4,821,477 / 4,636,492 | 304 / 314 / 322 ns | 10.3 / 12.4 / 13.3 MiB |

HTTP samples had substantial dispersion (throughput CV 0.28/0.23, enabled p95
CV 0.45), so they do not justify a regression verdict. Collection retained
240 frames / 625,120 bytes with zero pending minutes and performed 60 logical
database reads, 12 size reads and 60 history writes. Its database is the
measurement harness; real database behavior is verified separately. Raw sample
values and command timings are retained in task evidence.

## Verification and boundaries

- Focused authentication/secret/mTLS suite: PASS, 53 Go tests/subtests,
  21.27 s, no skips.
- `make test-mtls`: PASS, 10 Go events and five Python tests, 2.12 s.
- Actual backup/restore rehearsal: PASS, both backends, 17.59 s; repeated on final schema 62 in 13.17 s, no skips.
- Full release plus explicit public API, Audit hardening, Findings and browser
  aliases: PASS on `e49c6523`, 5,072.16 s (84 min 32 s), exit 0. Exact command,
  source, hashes and executed-case evidence are in `tasks/evidence/v60-009.json`.

Go event counts include parents and subtests and may overlap across commands.
The standard Go verification prerequisite can replay cached package results;
these log events are not claimed as new physical executions. Required database
and process targets use explicit `-count=1`.
The backup test restores domain persistence and then resumes repository writes;
the production startup/process evidence comes from separate release fixtures.
It does not simulate power loss, HA failover, production volume permissions or
restore encrypted credentials without the separately retained master key.
Live models remain outside scope. Optional sqlmap absence is reported, not
turned into successful scanner evidence.

The initial release invocation failed lint before execution (V60-030). The
next invocation at `9cfd09cc` passed lint, broad Go/Python/UI checks and the
original UI selection, then failed after 2,272.32 s in production Memory
configuration staging (V60-033). Targets after that failure were not executed.
Both failures and their logs are retained. V60-031/032 correct browser selection
and stale fixture selectors; the full affected UI prerequisite passed at
`45b31abf`. The original one-file UI success does not establish the restored
fixture journeys. The subsequent release attempts and final result are recorded
separately below and in task evidence.

Confirmed review findings are bounded corrections: V60-026 fixes result-size
parity; V60-027 fixes cross-Audit receipt identities; V60-028 repairs stale Audit
revision use in a mandatory test; V60-029 repairs the fault matrix's moved test
reference. V60-030 fixes existing test formatting, V60-031 restores omitted
browser coverage, and V60-032 updates stale selectors without weakening their
assertions. V60-033 stages the configured summarizer instruction omitted by a
production Memory test fixture. No additional authentication bypass or storage recovery defect was
confirmed by this slice.

The subsequent `make -k` invocation at `7100eb41` preserves independent target
results after failure. It exposed stale Code Analysis, Agent Skills, Summarizer
and Project Workspace fixtures (V60-036–039). External Evals also returned an
unexpected deadline error after two earlier full external journeys passed.
Concurrent unrelated PostgreSQL queries timed out, and host memory/IO pressure
was observed; neither a code defect nor an infrastructure cause is established.
The diagnostic incident record retains the timestamps and hashes without
turning that correlation into a conclusion. No retry or timeout was changed
to hide the failure. That failed invocation does not establish release success.

The next affected-first invocation at `747839c2` passed Code Analysis, Agent
Skills and Summarizer. It then exposed a second stale Project Workspace
assertion: its exact binding set omitted the normative retained repeat request.
V60-040 adds that one expected system record using the existing constants.
The complete Project target subsequently passed at `e49c6523` (164.995 s,
plus four Runtime cases), as did all three Audit process cases (474.894 s).
The same frozen invocation passed the complete release and explicit public
API/Findings/browser gates. It recorded 13,507 Go PASS events, including
parents/subtests, overlapping commands and 24 cached package output lines.
The base Runtime suite passed 2,468 tests with 39 explicit optional skips; the
UI base suite passed 492 tests in 70 files. All 20 required Chromium cases in
seven files passed with zero skipped, unexpected or flaky results. Native Evals
passed in 509.31 s and external Evals in 480.63 s. Audit completion verified
173 Go and 331 Runtime cases without selected skips; Findings verified all
13 required process cases plus 48 Runtime cases (171.237 s Go package time).

The final monitor stopped normally after 5,073 samples and zero collection
errors. Its raw records and summary hashes are retained in evidence. The
earlier external Evals deadline did not recur, but its cause remains unknown.
The final Runtime optional count adds five absent-Katana-binary cases to the
34 skips classified in V60-006. Earlier separate Podman/Artifact-bridge results
retain their original source attribution. No optional live-model or missing
scanner-binary case is represented as an executed success.

## Integration follow-up

The integration source `43cea5b89ac8c9bf9d43101f399079c31aabd439` includes
committed main through `bacb2817`: explicit Planner model access, required
batch readers, the current Memory catalog, strict persisted session/Audit role
validation and typed Evals receipts. Five overlapping process fixtures and
three catalog-dependent regressions were reconciled. They retain exact pinned
budgets, repeat bindings, Memory tools, validated snapshot time, instruction
staging and HTTP retry classification. A real Server restart checks preserved
Project pins and history after removing the current Audit catalog entries.

The first integration invocation passed six focused regression groups, catalog
validation, affected PostgreSQL race tests, Streamline/Gateway recovery,
build/package compilation, and the Memory, Project, Audit and Agent Skills
process targets. It then failed in the Findings boundary fixture because its
expected tool set omitted six newly configured Memory tools. UI execution was
not reached. V60-041 corrects that single expected list with the existing helper;
the failed invocation and its earlier passing stages retain source attribution.
Compile-only commands are not represented as process execution.

At `c25b4b28ed0e865dd54fb0bd74b917a1306e99f0`, the complete Findings and UI stack
invocation passed in 1,285.97 s, exit zero: 13 required Findings process cases,
48 Runtime cases, all 20 Chromium cases in seven required files, native Evals
(510.85 s) and external Evals (481.27 s). The browser selection had zero skipped,
unexpected or flaky cases. Its 19 route-fixture journeys and one actual
Operations process case are distinguished from the separate actual Evals
journeys. Optional screenshot OCR was unavailable. The stopped database
monitor recorded 1,286 samples with no collection errors. Evidence retains
commands, source commits and SHA-256 hashes for both integration invocations.
The complete release result above remains attributed to `e49c6523`; later
integration used the affected checks listed here.

All 41 V60 tasks are completed. Original failed runs and the unexplained
historical Evals deadline remain in the evidence; successful follow-up results
do not rewrite them.
