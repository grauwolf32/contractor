# V60-040: Expect the exact retained Run repeat request

The integrated Project Workspaces gate at source
`747839c255fc3eef800281d837b1c57ab87f892d` passed the corrected Worker budget
checks, then failed after 56.39 s. Its raw PostgreSQL binding query returned
`contractor-system/repeat-request`, which was missing from the exact expected
binding map. The failure stopped the target before its Python process tests.

[Artifact spec 03](../spec/03-artifact-plane.md) explicitly permits that exact
hidden physical record while excluding it from public and Runtime Artifact
APIs. Commit `c008980672cdf8ac15a6f9204a9be11382e3c145` introduced it in public
Run creation. The test inspects physical storage, so the extra binding is
required persisted state, not a visibility defect.

Implementation `e49c65230b185c0daf575a202a22e4508bb7ed4b` adds five lines: an
`artifactpolicy` import and the exact named constant entry with its media type
and explanatory comment in both existing expected maps. The assertion helper
is untouched. It still rejects any missing or additional binding and checks
every media type, nonempty revision, and frozen state. The repeat request uses
the ordinary trusted write and is nonfrozen; only output bindings are frozen.
No namespace is filtered from the database query.

The remaining Project process path was reviewed without changing expectations.
The retained request adds no fork/bind lineage; exact lineage and per-analysis
revision checks remain valid. Project output publication is create-only, so
the second Run retains the expected `already_present` workspace publications.
The eight scripted stages, three OpenAPI and five LikeC4 validator invocations,
two Runs after idempotent replay, source/output bytes, Server restart, Runtime
release, target revisions and credential cleanup remain consistent with the
current code. This Project scenario does not delete Runs; Audit has a separate
Run-deletion process journey. No additional confirmed mismatch was found.

The actual existing process failure is the before regression. No new validator
or mirror unit tests were added for this bounded expectation correction.
`git apply --check` passed, and the patch was committed before the next source
freeze. Full `make test-project-workspaces-e2e` then passed at implementation
source `e49c65230b185c0daf575a202a22e4508bb7ed4b`: the real process completed in
164.99 s (Go package 164.995 s), followed by four Python workspace-process
passes in 0.95 s, with zero failures or skips. Every original later assertion
passed, including the second Run after Server restart.

Commands, immutable failure log hash, source/patch hashes and acceptance status
are recorded in [V60-040 evidence](../../tasks/evidence/v60-040.json).

The immutable full-target log is `.local/v60-review/v60-039-040-full-target.log`
(SHA256 `fa758c8c0b838243c3d319894728d86f62b6f07d542acd3ec949f68fee2661a0`), captured
from lines 1–13 before the Audit target. Required task verification is complete;
completion metadata is recorded separately. The overall release is still in
progress and is not claimed passed.
