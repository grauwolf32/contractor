# V60-034: Keep the Audit replacement catalog loadable

The production-process fixture removes authoring entries and current standard
bindings, restarts Server, and verifies that pinned Audit history remains
readable while the current catalog is unavailable. Two stale assumptions in
its removal list prevented that restart. Both corrections are confined to
the disposable test fixture; production configuration and retention behavior
are unchanged.

## Initial instruction failure

An isolated copy of `configs` passed the real
`contractor-server config validate --root` command: 33 workflows, 41 templates,
11 Audit profiles and 52 instructions. Applying the original filesystem
removal list made validation exit 1: the surviving
`audit_asvs_source_verification_v3_memory.yaml` referenced the deleted
`instructions/audit-asvs-source-verifier-worker.md`. The Top 10 version-3
workflow shared the analogous deleted instruction.

Implementation `020156fab8043a2dfe5c43188cd96dca547020ae` retained these two
shared files and extracted `removeAuditProgramAuthoringEntries`. The initial
focused regression passed in 2.990 s (three Go PASS events, zero skips), and
the real CLI passed in 0.253 s. Those checks proved configuration parsing,
not database-backed Server startup. The initial focused regression's added
expectation that version-2 Audit profiles should remain available was
incorrect and is removed by the followup.

## Actual process failure and followup

The required `test-audits-process` target at source
`7100eb4171f8009cc7fe0591bb4fff8504313c14` failed. After 363.78 s, the Audit
journey reached its restart and Server rejected the remaining ASVS version-2
profile because `owasp-asvs@5.0.0` was no longer in the current standard
catalog. The other two process tests passed; the Go package exited 1 after
481.807 s, with two PASS events, one FAIL event and zero skips. The immutable
target segment is `.local/v60-review/v60-034-release-process-before.log`
(SHA256 `e1d9f70930a49b6c96702d03aee3804fa05e2eb4a94450beab5a7541add1f2ec`).
The overall release invocation also failed; no release success is claimed.

`validateCatalogStandards` in `internal/app/composition_catalogs.go` resolves
every remaining profile's standards against the actual Artifact store.
`config.Load` does not perform that check. The removal list deleted both
original standard directories and their current database bindings but left
both version-2 profiles referring to them.

The followup removes those two dependent profiles, extends profile
list absence checks to both versions, and preserves version-3 workflows with
their exact shared instructions. It restores no current standard bindings or
packages. Current-binding SQL deletion and every original historical Audit,
coverage, report and finding-backtrace assertion remain unchanged.

A new focused test builds and starts the real Server on loopback using an
isolated PostgreSQL database and the canonical operator configuration root.
It creates a Project, pins both standards through the actual standards
catalog, invokes the exact filesystem/SQL replacement helper, and restarts
Server. It verifies readiness, absence of both profile versions from the
profile list, HTTP 404 for both current standards, and unchanged readable
retained Project standard pins and digests. It uses no Runtime or model.

With only this test overlaid onto the frozen sources, the same Server
startup failure reproduced in 7.108 s (exit 1). Overlaying the followup
passed both focused tests in 6.482 s: four Go PASS events including the two
configuration subtests, zero failures and zero skips. The followup patch is
`.local/v60-review/v60-034-followup.patch`, SHA256
`e3dccbc9b10b47fc0dc3e65846f7c55ddf23216eb630d240e1cd12efcd4e3ff2`;
`git apply --check` passed. These were overlay executions while tracked
sources remained frozen, not a completed implementation commit.

## Applied verification

Followup implementation `747839c255fc3eef800281d837b1c57ab87f892d` applies the
correction. The canonical focused command passed on actual tracked source,
without an overlay, in 9.17 s: four Go PASS events and zero skips.

The first integrated invocation at the followup source commit stopped at
the earlier Project Workspaces target before Audit execution (V60-040). After
that separate fixture correction, the full `make test-audits-process` target
passed at source `e49c65230b185c0daf575a202a22e4508bb7ed4b`. The Audit journey passed
in 363.73 s, heterogeneous Runtime placement in 46.02 s, and scheduler
concurrency in 65.13 s. The package completed in 474.894 s: three Go PASS events,
zero failures and zero skips.

The unchanged full Audit journey reached every retained-history assertion
after restart: pinned baselines, coverage, reports and finding backtraces stayed
readable while both current profile versions and original standards remained
absent. This is the required process evidence in addition to the small
startup/pin regression.

The immutable full-target log is `.local/v60-review/v60-034-full-target.log`
(SHA256 `6d0d07e92789dafa9e6b55919244e0be565e33c5b4b7e3038261a6490817652b`), copied from
lines 14–27 before the subsequent release lint target. Both required V60-034
commands passed; completion metadata remains separate from original
implementation `020156fab8043a2dfe5c43188cd96dca547020ae` and followup
`747839c255fc3eef800281d837b1c57ab87f892d`. The overall release invocation is
still running and is not claimed passed.

Exact commands, exit codes, timings, original and followup source hashes,
and immutable local log hashes are in
[V60-034 evidence](../../tasks/evidence/v60-034.json).
