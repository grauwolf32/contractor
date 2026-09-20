# V60-032: Browser fixture expectations after restored gate coverage

V60-031 restored seven required browser files to `test-ui-stack`. A separate
fixture preflight reproduced three failures in previously omitted cases.
The corrections affect only assertions in three `ui/e2e` files; no product
code, fixture response, timeout or selected case was changed.

## Confirmed stale expectations

| File | Existing UI contract and correction |
| --- | --- |
| `audits.spec.ts` | `640b4890` introduced a conclusion excerpt alongside the assessment badge and a collapsed **Read task and result** section. The old `Inconclusive` substring locator matched two elements. Selecting the exact badge exposed a second stale expectation: task, result and gap text were checked before opening their disclosure. The test now opens it and retains all three assertions within that section. |
| `lifecycle-controls.spec.ts` | `90a1ab39` and spec 06 place confirmed deletion in the accessible **Project actions** menu. The removed **Additional actions** text caused a timeout. The test now selects the existing accessible menu name; confirmation, waiting for release, eventual deletion, CSRF and revision checks are unchanged. |
| `performance.spec.ts` | `f5ae4a4f` / V58-010 intentionally show compact GPU summaries before secondary counters. The test now opens **Detailed counters and collection diagnostics**, asserts the same GPU's exact **44 °C** reading and absent unsupported **Power draw**, then closes it. Existing history, identity/color stability, mobile layout and unavailable-device assertions remain intact. |

The initial Audit selector correction alone was insufficient: the second run
revealed the collapsed task/result expectations. That intermediate failure is
retained in the evidence rather than reported as a passing attempt.

## Actual fixture verification

Each invocation copied the existing `ui/dist`, static server and package
metadata into an owned temporary directory. A separate Node server bound an
OS-selected loopback port and served that copy. No build or mutation of the
original distribution occurred. The copied asset manifest is identical across
all three runs. Each temporary server and directory was cleaned up afterward.

Playwright executed six files against that static UI with routed API fixtures.
These observations establish browser behavior, not real backend wiring.
`stack.spec.ts` was deliberately reserved for the full affected gate.

| Invocation | Wall time | Actual Playwright result |
| --- | ---: | --- |
| Before correction | 53.764 s | 16 passed, 3 failed, 0 skipped, 0 flaky |
| Initial correction | 41.471 s | 18 passed, 1 failed, 0 skipped, 0 flaky |
| Final correction | 20.631 s | 19 passed, 0 failed, 0 skipped, 0 flaky |

The final 19 cases comprise four Audit, one lifecycle, two performance, five
Project workspace, six repeat and one Scheduler settings case. Playwright's
own execution duration was 19.643 s; the table includes setup and teardown.
Targeted Prettier and ESLint checks and the complete E2E TypeScript compilation
also passed. No test assertions were removed.

[V60-032 evidence](../../tasks/evidence/v60-032.json) records commands, per-file
counts, local report/log hashes, copied-asset identity and final source hashes.
The local orchestration script is retained as review evidence; the reproducible
required repository command remains `make test-ui-stack`, which includes these
19 fixtures, the real stack flow and strict JSON execution validation.

## Full affected gate

The subsequent actual combined invocation of `make --trace test-ui-stack
test-audits-browser test-lifecycle-controls-browser
test-scheduler-concurrency-browser test-performance-browser` passed with exit 0
in 1107.48 s. Common aliases reused the single completed prerequisite.

The Go suite recorded 21 PASS events including subtests and zero skips. The
strict Playwright JSON report recorded **20 passed, zero skipped, unexpected
or flaky** across all seven required files: the same 19 fixture cases plus the
real stack flow. Their report SHA-256 is
`99ae4c066f6919d73361aab515ca8a547b09a0558d23ad18f2ed0c940f02ddf5`.
The native and external managed Evals process cases also passed in 511.67 s
and 481.69 s respectively. Counts are execution events, not independent proofs
per Make alias.

Optional screenshot OCR was unavailable; it is explicitly logged and is not a
skipped required Go or Playwright case. Required network, browser storage and
trace secret checks still ran. The full gate results and final log hash are
recorded in the linked evidence. All V60-032 acceptance criteria are satisfied;
completion metadata is recorded separately from implementation `45b31abf`.
