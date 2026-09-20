# V60-038: Summarizer provider-timeout fixture

The full release on `7100eb41` failed
`TestWorkerSummarizerProductionBoundaries` after 136.01 s in the
`provider-timeout` scenario. The Stage correctly failed with
`worker_summarization_failed` and cause `gateway_unavailable`, but its final
failure was non-retryable, contrary to the fixture's expectation.

The fixture sends HTTP 504 on its first summarizer request. The allocation-owned
HTTPX client permits a physical transport retry. The fixture rejects that second
request with HTTP 400 because its scenario requires exactly one request. Runtime
therefore preserves the final `BadRequestError` and `retryable=false`. This is a
reproduced fixture-contract mismatch, independent of host pressure or deadlines.

[Spec02](../spec/02-runtime-and-a2a.md) separates bounded transport attempts from
final failure classification: `X-Should-Retry: false` suppresses transport retries,
while HTTP 504 still classifies as retryable. [Spec15](../spec/15-worker-summarization.md)
requires one logical summarizer call, permits that transport attempt series and
preserves Gateway retryability. No production correction is needed.

Implementation `371babb9aeba5d540913cc391c34b2476d2428b4` adds that response
header only to the fixture's scripted 504. Its exact request-count assertion and all Stage, token-accounting,
redaction, cancellation and allocation-reuse assertions remain intact. No timeout
or retry budget changes.

## Verification

A local Go overlay adds a probe which starts the actual `summarizerGateway` and
calls it through the current Python `new_gateway_client`. It keeps default
transport retries enabled and checks both the classified error and actual
fixture call counts.

| Check | Result |
| --- | --- |
| Original release target | Failed: one process test, 136.01 s; immutable excerpt preserved. |
| Actual transport, original fixture | Failed as expected: `BadRequestError`, `retryable=false`, two requests and a duplicate-call fixture error; 3.668 s. |
| Actual transport, corrected fixture overlay | Passed: `InternalServerError`, `retryable=true`, one request, no fixture errors or skips; 3.123 s. |
| Full overlay `make test-worker-summarizer-e2e` | Interrupted on coordinator request before the process/DB target, after matrix checks and 92 Python cases passed; not a completed gate. |
| Full target on corrected tracked source | Passed without overlay: real process case 205.52 s; Go package 205.525 s. Target segment contains 987 Go PASS events and 92 Python cases (7.13 s), zero skips. |
| Shared hardening matrix prerequisite | Passed earlier in the same make invocation: 139 Go PASS events, zero skips. |
| `git diff --check` | Passed. |

The initial probes ran while source `7100eb41` was frozen; the relevant fixture
and Runtime files were checked byte-for-byte against that commit. After the
minimal correction was committed, the complete target ran on frozen source
`747839c255fc3eef800281d837b1c57ab87f892d` as part of the integrated make
invocation. No overlay was set for this execution. The existing process journey
proved the provider failure, exact logical and physical call counts,
cancellation, token accounting, redaction and allocation reuse.

Immutable local target and matrix excerpts retain their source line positions
and SHA256 hashes. The target completed successfully and make advanced to Project
workspaces; the aggregate release was still running when this report was
finalized, so this report does not claim its eventual outcome. Exact commands,
overlay files, outcomes and log hashes are in
[V60-038 evidence](../../tasks/evidence/v60-038.json).
