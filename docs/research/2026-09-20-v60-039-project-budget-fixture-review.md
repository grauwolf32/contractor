# V60-039: Validate Project Workflow budgets against their pinned policy

The required Project Workspaces process gate reached a succeeded
`dependency_discovery` Stage, then failed its durable Worker report assertion.
The report contained limits `200/200/2500000` and the expected observed model,
tool and token counts `5/3/80`. The assertion still required `24/96/500000`.

The original assertions came from `94264867` and subsequent fixture history.
Commit `394018a04a57c52a1e4cdce8d0c4ef63edd74114` consolidated the production
Project Workflow selections on `worker@2`, whose current limits match the
report. This is a stale process assertion, not evidence of incorrect Runtime
budget enforcement. [Spec 04](../spec/04-execution-lifecycle-and-metrics.md)
binds ModelPolicy attribution to immutable execution configuration.

## Correction

The durable report check reads the retained `StageSpecSnapshot`, selects its
effective ModelPolicy by the allocated logical Agent name, and compares all
three report limits exactly. It retains exact observed model/tool/token
counts, missing-usage and exhaustion assertions. It does not read mutable
authoring defaults to reinterpret a completed execution. All durable result,
publication, lineage, restart, placement and cleanup assertions remain intact.

A focused validator regression loads the actual OpenAPI Workflow and covers
each changed limit and counter, unavailable usage, exhaustion, missing budget,
missing logical Agent and malformed snapshot. It also distinguishes an
effective execution override from template defaults and retains the original
snapshot independently of later authoring changes.

## Verification

The unmodified required process failed in 56.97 s; its error and command are
retained separately. To keep the ongoing release source frozen, focused checks
use explicit Go overlays. An overlay extracting the original numeric comparison
reproduced its rejection in 3.304 s. The proposed correction passed in 3.076 s:
13 Go PASS events, zero skips. This extracted comparison is not presented as an
original checked-in regression.

A separate local configuration preflight checked all eight OpenAPI/LikeC4
stages against their existing fake Gateway scripts: current model alias,
exact tool surfaces, sufficient policy budgets and model-call counts including
the finalizer. It passed in 2.760 s: nine Go PASS events, zero skips. The counts
remain `5,5,9,9` and `5,5,11,9`; all stages select `worker@2`. No additional
policy-dependent stale expectation was found in the remaining scenario.

Implementation `ef7defc766b3c70b568829dcfc5d8bc0777dbcb7` applies the correction.
The focused command passed on actual tracked source, without an overlay,
in 2.79 s: 13 Go PASS events and zero skips. Full
`make test-project-workspaces-e2e` ran at source
`747839c255fc3eef800281d837b1c57ab87f892d`. Corrected budget assertions passed,
but the process failed later in 56.39 s because its exact physical-binding
expectation omitted the required `contractor-system/repeat-request` record.
That distinct stale assertion is tracked as V60-040. The failed target stopped
before its Python workspace-process recipe; no full-gate success is claimed.
The immutable segment is
`.local/v60-review/v60-039-integrated-repeat-binding-failure.log`.
After V60-040, the full target passed at source
`e49c65230b185c0daf575a202a22e4508bb7ed4b`: the real process completed in
164.99 s (Go package 164.995 s), followed by four Python workspace-process
passes in 0.95 s, with zero failures or skips. All original budget, result,
publication, lineage, restart, placement and cleanup assertions passed.
These focused configuration checks do not establish process success. Exact
commands and hashes are in [V60-039 evidence](../../tasks/evidence/v60-039.json).
No live model or production service is used.

The immutable full-target log is `.local/v60-review/v60-039-040-full-target.log`
(SHA256 `fa758c8c0b838243c3d319894728d86f62b6f07d542acd3ec949f68fee2661a0`), captured
from lines 1–13 before the Audit target. Required task verification is complete;
completion metadata is recorded separately. The overall release is still in
progress and is not claimed passed.
