# V60-009 — Authentication, gates and operational recovery

Reviewed 2026-09-20 from `8edeabf2` in isolated branch
`review/v60-deep-review`. Source corrections from V60-026 through V60-029 are
part of the final gate snapshot. No production deployment or live model calls.

## Source, decisions and representative verification

| Area | Trace and result |
| --- | --- |
| Public credentials | `auth_handlers.go` requires exactly one supported Authorization header and never falls back to a cookie after a supplied bearer fails. Cookie authentication rejects duplicate session cookies; unsafe requests require one allowed Origin and one session-bound CSRF token before the domain handler. New ten-case regression verifies actual Run cancellation stays untouched on rejection and succeeds for the valid control. |
| Login and sessions | V60-002/`80ac450d` separates escaped JSON body size from supported decoded password size. Existing tests cover password limits, owner-only nonsymlink bootstrap, rate limits, cookie flags, expiry, revocation and process-local restart. `5a6ebc3d` deliberately permits configured trusted private-network demo origins; it does not authorize arbitrary origins. |
| Private authority | `3e359fb9` binds Runtime registration/allocation to the mTLS certificate principal. `make test-mtls` passed 10 Go events and five Python cases; public bearer/cookie credentials do not replace private certificate authority. Certificate helpers, private Artifact integration and full process gates cover separate levels. |
| Secret projections | Login clears the password field and wipes its temporary byte slice. Errors expose bounded codes. New negative requests assert no bearer, password, cookie or CSRF value in the response. Existing credential tests check redacted projections and required owner-only master-key material. This is not a claim that every allocation of a Go string can be erased from memory. |
| Fresh install and upgrade | V60-007 runs real migrations, cancellation/backend termination during historical blob DDL, retry, checksum drift and unknown-newer-version rejection. V60-027 additionally checks schema 61 to 62 without rewriting legacy finding history. Process gates use the production binary, isolated PostgreSQL and fake Gateways. |
| Backup recovery | New `tests/integration/restore` uses actual `pg_dump`/`pg_restore`, fresh databases and both payload backends. It removes the original database and filesystem bytes before restore; exact old/latest revisions, binary payloads, digests, idempotency and CAS survive. Filesystem metadata alone cannot recover missing bytes. The operational guide records the required consistent offline snapshot and separately retained keys/configs. |
| Gate credibility | Make prerequisites and CI were inspected before execution. CI runs `release-verify` with PostgreSQL 17, locked uv/pnpm dependencies and Node 24.20. The combined final invocation shares common prerequisites once; it does not remove tests. Runtime opt-ins and host-only Git helpers are recorded separately from mandatory executed cases. |
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
- Actual backup/restore rehearsal: PASS, both backends, 17.59 s, no skips.
- Required public API and combined release/Audit gates: final results are
  recorded after the frozen source run in `tasks/evidence/v60-009.json`.

Go event counts include parents and subtests and may overlap across commands.
The backup test restores domain persistence and then resumes repository writes;
the production startup/process evidence comes from separate release fixtures.
It does not simulate power loss, HA failover, production volume permissions or
restore encrypted credentials without the separately retained master key.
Live models remain outside scope. Optional sqlmap absence is reported, not
turned into successful scanner evidence.

Confirmed review findings are bounded corrections: V60-026 fixes result-size
parity; V60-027 fixes cross-Audit receipt identities; V60-028 repairs stale Audit
revision use in a mandatory test; V60-029 repairs the fault matrix's moved test
reference. No additional authentication bypass or storage recovery defect was
confirmed by this slice. Task completion requires the final mandatory gates.
