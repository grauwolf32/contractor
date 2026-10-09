# Maintenance backlog — 2026-10-09

Reviewed main `63496210830a4c12816dfb9fd22e132645fb60cb` against the task records and the 2026-10-04 code review. The selected scope is V261-001 and V341-001–V363-001; V348-001 is already completed and is excluded from the remaining count.

## Current acceptance

- **V261-001 remains in progress.** Its linked issue #304 is closed and describes race/process deduplication, not the three failing process fixtures. The latest available main CI run is [37194642464](https://github.com/grauwolf32/contractor/actions/runs/37194642464) at `e015298c`: both process shards passed, but the browser stage and aggregate release verdict failed. This is historical CI evidence, not a result for current main.
- **22 follow-ups remain pending.** Each record now carries a dated reassessment. Source inspection is not sufficient to close acceptance; required tests remain outstanding.
- **V347-001 is still necessary.** The GitHub branches API returned `protected: false` for main. The current workflow emits `pr-verify` for selected PR stages and `release-verify` for full tag/manual runs. Protect PR merges with the current PR verdict rather than requiring a skipped full-release verdict. No API write credential is available locally; settings acceptance must stay open until applied and read back.
- **V348-001 stays completed.** No old task or unrelated feature is reopened by this review.

## Work order

1. Reproduce V261's original Agent Skills, Audit Programs and Code Analysis process cases on disposable PostgreSQL, then run the current release stages. Preserve exact assertions and record any real failures. CI acceptance requires an actual successful remote full-gate run.
2. Address bounded, independent correctness fixes: V349, V354, V356, V357, V358, V360 and V363; follow with source import/config roots (V352) and assessment indexes (V355).
3. Fix finding document isolation and byte-bounded hydration (V359/V361), then Eval claim/collection/plan performance (V350/V351).
4. Complete measured Runtime work and refactors (V341/V342/V343/V344/V345/V346/V353), with one implementation commit per task and the required acceptance checks.
5. Apply V347 protection when GitHub settings write access is available. This does not prevent the independent repository work.

Use an isolated PostgreSQL container for checks. The demo database and existing user work are outside the test fixtures. Pending tasks become in progress before implementation and completed only after their acceptance and required checks pass.


V346 measurement on the disposable local runner: a 4 MiB Python file with
80,660 top-level definitions blocked the event loop for 408 ms when passed as
one byte string. Callback input with 8 KiB chunks and a GIL yield reduced the
largest observed gap to about 41 ms. The regression bound is 100 ms on this
real parse, including native finalization and symbol extraction. Small files
retain the direct byte-string path. The pinned tree-sitter 0.25.2 progress
callback is not used: it segfaulted on this Python 3.13 runner; chunked input
uses the supported read callback and leaves parse results unchanged.
