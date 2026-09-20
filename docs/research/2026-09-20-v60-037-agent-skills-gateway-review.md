# V60-037: Agent Skills retry fixture

The complete release on `7100eb41` failed
`TestAgentSkillsMVPProcesses` after 36.36 s. Its initial artifact-copy Worker
succeeded, then the first Skill-bearing builder terminated with
`worker_gateway_unavailable`, `BadRequestError` and `retryable=false`.

The Gateway intentionally failed that call after its Skill disclosure checks.
`failedAgentSkillGatewayStage` sets `modelFail`; the shared fake Gateway returned
HTTP 400 without adding a fixture-validation failure. Thus the empty
`gatewayFailures` list does not indicate a transport or environment problem.
The existing assertion expects a failed builder followed by a new successful
builder StageExecution with the same exact package A.

Accepted Gateway classification from `b76f1de4`, retained by the HTTPX change in
`be33cebb`, makes HTTP 400 non-retryable. Spec05 requires non-retryable failures
to execute the retry policy's terminal action. The fixture, rather than
production behavior, must express a transient failure for the intended retry.

The bounded correction returns HTTP 503 with `X-Should-Retry: false` only for
the intentional scripted failure. This suppresses another physical HTTP attempt
inside the same model call while preserving `retryable=true` for the Scheduler.
Ordinary malformed requests and fixture-validation failures still return 400.
All `modelFail` consumers were inspected: only the Agent Skills first-attempt
fixture and its dedicated retry-probe unit enable it.

The existing unit now exercises the HTTP handler and checks the response,
single call, single script advance and empty fixture-error list. The original
response fails that assertion; the prepared overlay passes. The existing
Runtime regression independently confirms that the retry-suppression header
does not change final failure classification. The full process gate retained and passed the original package A/B selection,
new StageExecution, disclosure, Runtime slot reuse and redaction assertions.

## Verification

The correction is committed as `ab5aca11`; the integrated release is frozen
on `747839c255fc3eef800281d837b1c57ab87f892d`. The focused HTTP regression
also passed on the applied source. The actual full process target passed on
that frozen commit; no process assertion was removed or relaxed.

| Check | Result |
| --- | --- |
| Original release target | Failed: one process test, 36.36 s; exact log section preserved. |
| HTTP assertion with original response | Failed as expected: HTTP 400 and absent retry-suppression header; 3.375 s. |
| HTTP assertion with prepared overlay | Passed: one case, zero skips; 3.312 s. |
| Existing Runtime classification regression | Passed: three cases, zero skips; 1.396 s including process startup. |
| Focused HTTP regression on applied source | Passed: one case, zero skips; 2.96 s. |
| Full `make test-agent-skills-mvp` on frozen integrated source | Passed: one process test, zero skips; test 146.95 s, Go package 146.964 s. |

Exact commands, source/overlay hashes and immutable logs are recorded in
[V60-037 evidence](../../tasks/evidence/v60-037.json). Overlay success is not
counted as execution of the full Skill process journey.

The target log is an immutable section of the shared integrated release. Make
advanced to the next target after its success. The complete release remained
in progress at extraction; this task does not claim that aggregate run passed.
