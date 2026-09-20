# V60 — completion of the remaining review slices

Review date: 2026-09-20. Work was isolated on `review/v60-deep-review` from
`8edeabf21194f5bba8dd258d53a072af713c630e`; unrelated working-tree changes were
preserved. Each task records its original implementation commit separately
from completion metadata. Committed main through `bc32a76b` was merged before
the final invocation, currently at `e49c6523`. Per-task evidence distinguishes actual PostgreSQL,
process, container, browser-fixture and unit-test observations.

## Review records

| Task | Scope | Report |
| --- | --- | --- |
| V60-005 | Execution, lease/recovery, pause/cancel/retry | [Execution review](2026-09-20-v60-005-execution-review.md) |
| V60-006 | Runtime, wire parity, tools, session/summarizer | [Runtime review](2026-09-20-v60-006-runtime-review.md) |
| V60-007 | Artifact authority, concurrency, migrations, blob/Git backends | [Storage review](2026-09-20-v60-007-storage-review.md) |
| V60-008 | Audit, finding retention, review/report and UI journeys | [Audit/product review](2026-09-20-v60-008-audit-product-review.md) |
| V60-009 | Credentials, release gates, real backup/restore, measurements | [Operations review](2026-09-20-v60-009-operations-security-review.md) |
| V60-010 | Immutable configuration, portable evaluations and backlog truth | [Configuration/evaluation review](2026-09-20-v60-010-configuration-evaluation-review.md) |

## Confirmed corrections

| Task | Problem and resulting behavior |
| --- | --- |
| V60-026 | Go's optional HTML/JavaScript escaping rejected results accepted by Python. Both now count compact UTF-8 JSON for the typed Worker/Planner result contracts while preserving numeric limits, mandatory escaping and exact integer digits. Shared boundary fixtures and real SDK/Planner paths verify it. |
| V60-027 | Importing the same receipt into a second compatible Audit collided with a global finding/assessment ID. Additive schema 62 creates independent Audit identities and preserves legacy IDs/history, same-Audit replay, review decisions and retained evidence. |
| V60-028 | An Audit gate reused a revision invalidated by source Run deletion. It now explicitly checks stale-revision rejection and retries with the refreshed revision before the existing purge checks. |
| V60-029 | Fault matrix referenced a regression's old file after extraction. Its reference now resolves to the unchanged partial-preparation cleanup test. |
| V60-030 | Two unformatted composite-literal fields stopped release lint. Only gofmt whitespace changed. |
| V60-031 | Browser aliases omitted their promised fixture suites. The shared gate explicitly selects seven files and validates actual Playwright results, rejecting missing/skipped/flaky cases. |
| V60-032 | Restored browser coverage exposed stale selectors after delivered UI changes. Exact badges, accessible Project actions and expanded task/performance details preserve the existing assertions. |
| V60-033 | Production Memory test staging omitted the configured summarizer instruction. The staged template now preserves both configured and legacy instruction behavior; the actual process gate passes. |
| V60-034 | Audit catalog replacement deleted instructions still used by surviving workflows. The fixture retains shared instruction files while removing both generations of profiles that depend on the removed standards. A real Server restart checks current catalog absence and preserved exact Project pins. |
| V60-035 | Findings E2E compared per-read snapshot time as durable review state. Both timestamps remain validated, and every other response field remains subject to exact equality; the full Findings gate passes. |
| V60-036 | Generated Code Analysis templates referenced a retired model policy and prevented process startup. They select the current production Worker policy, preserving the complete scripted tool surface. |
| V60-037 | A scripted Agent Skills failure used nonretryable HTTP 400 while expecting a new Stage attempt. HTTP 503 with transport retries disabled preserves the intended Scheduler retry and pinned-Skills checks. |
| V60-038 | A scripted Summarizer timeout triggered a transport retry that the one-request fixture rejected. The HTTP 504 response now suppresses physical retries while retaining its retryable Stage classification. |
| V60-039 | Project workflow tests hardcoded obsolete Worker budget limits. Expectations use the immutable execution configuration while retaining exact limits and observed usage checks. |
| V60-040 | Exact Project Run binding checks omitted the required retained repeat request. Both expected sets now name that one system record through existing constants; all exact-count, media, revision and frozen-state checks remain unchanged. |

The review also clarifies the established Queue Pause release exception and
updates stale V37/V38 delivery statements. It does not introduce new lifecycle
semantics, change result limits, take over V40/V55/V61 work or claim live model
quality.

## Final verification

One complete fail-fast invocation passed at
`e49c65230b185c0daf575a202a22e4508bb7ed4b`, with committed main through
`bc32a76b` included: full `release-verify`, explicit public API, Audit hardening,
Findings and both browser aliases. It took 5,072.16 s and exited zero. The
source tree stayed unchanged during execution. The exact command and log hash
are in [V60-009 evidence](../../tasks/evidence/v60-009.json).

The base Runtime suite passed 2,468 tests and recorded 39 explicit optional
skips; the base UI suite passed 492 tests. All 20 required Chromium cases in
seven files passed without skips or flaky results. Real native and external
Evals passed in 509.31 s and 480.63 s. Audit completion verified 173 Go/331
Runtime cases, and Findings verified 13 required process/48 Runtime cases,
without selected skips. Go log counts include parent tests, overlapping
commands and cached package output; they are not unique physical executions.

Original failed attempts remain recorded: formatting, Memory staging,
omitted/stale browser fixtures, Audit catalog replacement, changing Findings
read timestamps, and the later tool/Project fixtures. Each confirmed defect
has its own correction and verification record. One previous external Evals
run failed on a deadline. The full final run passed without changing retries
or timeouts; the earlier cause remains unestablished. Database/host monitoring
collected 5,073 samples without collection errors, and its records are retained.

Storage recovery additionally performed actual dump/restore into fresh
databases for both payload backends. Schema upgrade tests preserved legacy
finding history and recovered from interrupted historical blob DDL. Real
Podman opt-in tests used an installed supervisor image digest, with their
original source attribution retained. Optional live models, absent sqlmap and
Katana binaries remain explicitly outside executed proof. Raw logs stay local;
summaries and hashes are committed in task evidence.

## Integration with concurrent work

Later committed main changes retire superseded catalog entries and tighten
persisted configuration readers. Their integration and focused verification
are recorded separately from the frozen full-release result above.
