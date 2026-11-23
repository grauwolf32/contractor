# Maintenance backlog — 2026-10-09

[Research index](README.md)

- **Date:** 2026-10-09; status rechecked 2026-10-23 at `af9fd522`.
- **Reviewed:** `main` at `63496210830a4c12816dfb9fd22e132645fb60cb` against
  the task records and the [2026-10-04 code review](2026-10-04-code-review.md).
  The implementation ended at `00778946ae0e25b2e5afcad74c4dc9672a502158`.
- **Scope:** V261-001 and V341-001..V363-001. V348-001 was already completed.
- **Outcome:** 20 tasks completed (V342–V346, V349–V363); V261, V341 and
  V347 left open for remote acceptance.

## Status on 2026-10-23

The three open tasks are unchanged; no later CI result is recorded in their
task files. The local evidence below is for `00778946`, so the remote checks
must run on the `main` of the day.

| Task | Status | What is left |
| --- | --- | --- |
| V261-001 | in progress | A4: a successful full GitHub Actions `release-verify` on current `main` |
| V341-001 | in progress | CI factory-probe durations (`Runtime probe:` records) that justify or reduce the 12 s factory and 10 s Trailmark probe timeouts |
| V347-001 | pending | Apply branch protection to `main` with an administrator credential and read it back |

## Remaining remote acceptance

CI runs on pull requests, `v*` tags and manual dispatch; a push to `main` does
not start it. Pull requests report the `pr-verify` verdict, tags and manual
runs report `release-verify`.

**V261 and V341.** An authenticated maintainer dispatches the full workflow on
`main`:

```sh
gh workflow run ci.yml --repo grauwolf32/contractor --ref main
```

Acceptance needs the actual `release-verify` result for that revision and the
`Runtime probe:` timing records in the process-stage reports. Local probe
timings do not satisfy the CI timeout criterion.

**V347.** The prepared request,
[2026-10-09-main-protection.json](2026-10-09-main-protection.json), requires
`pr-verify` on an up-to-date branch, enforces it for administrators and
disallows force pushes and deletion. It does not require `release-verify`:
that job is skipped on pull requests, so it cannot gate a merge. The endpoint
needs repository Administration write access
([GitHub branch protection API](https://docs.github.com/en/rest/branches/branch-protection#update-branch-protection)):

```sh
gh api --method PUT repos/grauwolf32/contractor/branches/main/protection \
  --input docs/research/2026-10-09-main-protection.json
gh api repos/grauwolf32/contractor/branches/main/protection \
  --jq '{checks: .required_status_checks.contexts, strict: .required_status_checks.strict, admins: .enforce_admins.enabled}'
```

The expected readback is `checks: ["pr-verify"]`, `strict: true`,
`admins: true`. The last readback (2026-10-09) was `protected: false`.

## Completed in this pass

| Area | Tasks | Result |
| --- | --- | --- |
| Runtime and workspace | V342, V345, V346, V349, V353, V357 | Explicit invariants instead of asserts; fsspec and test-only APIs removed; bounded parse/diff stalls and OpenAPI nesting; one telemetry redaction policy |
| Audit and findings | V343, V355, V358–V361 | Store-owned review writes and shared item authority; missing FK indexes; matching association limits; unreadable-document isolation; byte-bounded receipt batches; visible stop reasons |
| Eval | V350, V351, V362 | Indexed bounded claims, statement-level collection invalidation, light plan metadata, shared selector and media contracts; real 20,000-experiment and 1,000-member fixtures |
| UI, source import and CLI | V344, V352, V354, V356, V363 | Shared PKI CLI tables; openable Git bundles and relative config roots; REST retry backoff; shared host syntax; exact config route names |

V341 also landed partially (`c28f8cb1`): the deterministic Audit fixture
requires exact submitted Run counts (1, 2, 10 and 4) and one accepted attempt
per executed item, and the Code Analysis fixture requires idle slots with
confirmed leases at the release and reuse boundaries.

## Local release verification on 2026-10-09

Disposable PostgreSQL 17 databases; the long stages ran in separate databases
with `GOMAXPROCS=2` and `GOFLAGS=-p=1`, keeping the exact release selections
and assertions. This is a composition of stage runs and successful reruns of
corrected substeps, not one successful `make release-verify` and not a GitHub
Actions verdict. The first aggregate run exposed obsolete matrix and test
references, a stale Findings tool declaration, UI timeouts under competing
local load and the query-plan regression measured below.

| Release stage | Result |
| --- | --- |
| `release-verify-lint` | Go formatting, vet, staticcheck and build; 52 script tests; release graph; Ruff |
| `release-verify-unit` | All Go packages; Runtime 3,621 passed, 39 optional skips |
| `release-verify-ui` | Generation, lint, typecheck; 1,757 tests in 163 files, 13 static-server tests, production build |
| `release-verify-families` | Hardening and configuration selections; Audit completion 197 Go and 402 Runtime cases; Findings 14 process cases and Runtime contracts |
| `release-verify-browser-a` | 50 mocked browser cases, real Operations stack, native Eval stack, fixture helpers |
| `release-verify-browser-b` | Real external Eval stack |
| `release-verify-race`, `-race-discovered` | Exact and discovered package selections under the race detector |
| `release-verify-integration` | All 59 tagged integration selections; non-race Finding Intake suite in 68.8 s |
| `release-verify-process-a/b/c` | All 19 production-process fixtures, including Agent Skills, Audit Programs and Code Analysis |

## Measurements

**Review authority query (V343).** The integration stage exposed a query-plan
regression in two controller fixtures with 205 pending item reviews.
PostgreSQL expanded the shared authority view into a nested anti join before
checking expiry, repeatedly scanning unrelated reviews: 1,327.852 ms and
424,331 shared buffer hits (`EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)`,
PostgreSQL 17), above the fixture's one-second operation budget. Keeping the
live-authority subquery correlated to the exact item with `OFFSET 0` brought
it to 7.260 ms and 6,131 buffer hits. Claiming and renewal both use this
boundary; the view and review semantics are unchanged.

**Tree-sitter parse stalls (V346).** A 4 MiB Python file with 80,660 top-level
definitions blocked the event loop for 408 ms when parsed from one byte
string. Feeding the parser through the read callback in 8 KiB chunks with a
GIL yield cut the largest gap to about 41 ms; the regression bound is 100 ms,
including native finalization and symbol extraction. Small files keep the
direct byte-string path. The progress callback of the pinned tree-sitter
0.25.2 is not used: it segfaulted on Python 3.13. Revisit when tree-sitter is
upgraded.
