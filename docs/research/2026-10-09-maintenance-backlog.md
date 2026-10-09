# Maintenance backlog — 2026-10-09

Reviewed main `63496210830a4c12816dfb9fd22e132645fb60cb` against the task records and the 2026-10-04 code review. The selected scope is V261-001 and V341-001–V363-001; V348-001 is already completed and is excluded from the remaining count.

## Current acceptance

- **V261-001 remains in progress.** Its linked issue #304 is closed and describes race/process deduplication, not the three failing process fixtures. The latest available main CI run is [37194642464](https://github.com/grauwolf32/contractor/actions/runs/37194642464) at `e015298c`: both process shards passed, but the browser stage and aggregate release verdict failed. This is historical CI evidence, not a result for current main.
- **20 of the 22 open maintenance follow-ups are completed.** V342–V346 and V349–V363 have implementation commits and verification recorded in their task files. V341 and V347 remain open for remote acceptance. Together with V261, the selected scope has three remaining tasks.
- **V347-001 is still necessary.** The GitHub branches API returned `protected: false` for main. The current workflow emits `pr-verify` for selected PR stages and `release-verify` for full tag/manual runs. Protect PR merges with the current PR verdict rather than requiring a skipped full-release verdict. No API write credential is available locally; settings acceptance must stay open until applied and read back.
- **V348-001 stays completed.** No old task or unrelated feature is reopened by this review.

## Work order

1. Reproduce V261's original Agent Skills, Audit Programs and Code Analysis process cases on disposable PostgreSQL, then run the current release stages. Preserve exact assertions and record any real failures. CI acceptance requires an actual successful remote full-gate run.
2. Address bounded, independent correctness fixes: V349, V354, V356, V357, V358, V360 and V363; follow with source import/config roots (V352) and assessment indexes (V355).
3. Fix finding document isolation and byte-bounded hydration (V359/V361), then Eval claim/collection/plan performance (V350/V351).
4. Complete measured Runtime work and refactors (V341/V342/V343/V344/V345/V346/V353), with one implementation commit per task and the required acceptance checks.
5. Apply V347 protection when GitHub settings write access is available. This does not prevent the independent repository work.

Use an isolated PostgreSQL container for checks. The demo database and existing user work are outside the test fixtures. Pending tasks become in progress before implementation and completed only after their acceptance and required checks pass.

## Implemented and verified

| Area | Tasks | Result |
|---|---|---|
| Runtime and workspace | V342, V345, V346, V349, V353, V357 | Explicit invariants; removed fsspec and test-only APIs; bounded parse/diff stalls and OpenAPI nesting; consistent telemetry redaction. Full Runtime suite: 3,621 passed, 39 optional skips; Ruff passed. |
| Audit and findings | V343, V355, V358–V361 | Store-owned review writes and shared item authority; missing FK indexes; matching association limits; unreadable document isolation; byte-bounded receipt batches. PostgreSQL suites and failure regressions passed. |
| Eval | V350, V351, V362 | Indexed bounded claims, statement-level collection invalidation, lightweight immutable plan metadata, shared published selector/media contracts. Real 20,000-experiment and 1,000-member fixtures passed. |
| UI, source import and CLI | V344, V352, V354, V356, V363 | Shared PKI CLI; readable Git bundles and relative config roots; REST retry backoff; shared host syntax and exact config route names. UI tests, generation, typecheck and lint passed. |

V360 belongs to the Audit row above and adds the missing user-visible stop reasons. Migration additions are 99–102; previously applied migration files are unchanged.

## Remaining remote acceptance

- **V261:** the three originally failing process fixtures pass locally. Audit tool declarations and obsolete release matrix/test selections were corrected. The full local release stages are being checked. A successful full GitHub Actions run for current main remains required.
- **V341:** the deterministic Audit fixture now requires exact submitted Run counts and one accepted attempt per executed item, allowing items in one Workflow batch to share a Run. All scenarios passed with exactly 1, 2, 10 and 4 Runs. Safe factory probe timings are retained in successful process test output. Code Analysis requires idle slots with confirmed leases at the release/reuse boundaries; the initial registration observation permits the already queued Run to occupy its slot. CI probe measurements, required before justifying or reducing the timeouts, remain outstanding.
- **V347:** public GitHub API readback still reports `protected: false` for main. [The prepared request](2026-10-09-main-protection.json) requires `pr-verify`, an up-to-date branch and administrator enforcement, and disallows force pushes and deletion. The endpoint needs repository Administration write access, which is unavailable here. See [GitHub's branch protection API](https://docs.github.com/en/rest/branches/branch-protection#update-branch-protection).

With an authenticated repository administrator, apply the prepared settings and read them back:

```sh
gh api --method PUT repos/grauwolf32/contractor/branches/main/protection \
  --input docs/research/2026-10-09-main-protection.json
gh api repos/grauwolf32/contractor/branches/main/protection \
  --jq '{checks: .required_status_checks.contexts, strict: .required_status_checks.strict, admins: .enforce_admins.enabled}'
```

The expected readback is `checks: ["pr-verify"]`, `strict: true`, `admins: true`. This is a prepared change, not a claim that settings have been applied.


V346 measurement on the disposable local runner: a 4 MiB Python file with
80,660 top-level definitions blocked the event loop for 408 ms when passed as
one byte string. Callback input with 8 KiB chunks and a GIL yield reduced the
largest observed gap to about 41 ms. The regression bound is 100 ms on this
real parse, including native finalization and symbol extraction. Small files
retain the direct byte-string path. The pinned tree-sitter 0.25.2 progress
callback is not used: it segfaulted on this Python 3.13 runner; chunked input
uses the supported read callback and leaves parse results unchanged.
