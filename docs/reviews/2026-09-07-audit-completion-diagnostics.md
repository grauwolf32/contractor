# V39-006: Audit function review and completion diagnostics

Reviewed the v2 tools, collector admission/revisions, deterministic encoder and
publisher, Runtime completion/cancellation ownership, Server contract pinning,
placement, importer acceptance, and report/public diagnostic projections.

## Findings and corrections

1. Late State synchronization/session-release failure or cancellation could leave
   completion diagnostics at `published` after Worker success was withdrawn.
   Three regression cases reproduced this. The final facts now become `failed`
   with the actual failure code and retained item counts. The exact written ZIP
   remains untouched. Cancellation facts are recorded while invocation ownership
   is held; terminal State is not completed a second time. A returned failure
   references the updated State revision.
2. The new optional diagnostic parser disagreed across languages: Python treated
   empty kind/phase as future data, while Go accepted an empty optional failure
   code. Shared JSON fixtures reproduced both mismatches. Both consumers now
   reject those malformed known envelopes, including through live Worker State.
3. Repository catalog checks assumed the old Workflow/Worker/policy counts, and
   two public projection tests assumed one version per Workflow name. The new
   catalog entry is covered explicitly; projection tests select exact versions.

No remaining blocking defect was found in the reviewed collection/publication
path. Its authority remains the immutable AuditProfile check binding. Empty
evidence cannot satisfy a conclusive checklist result. CAS/batch conflicts keep
earlier valid results, sealing requires the full assignment, and publication
never overwrites different bytes. The Go importer still independently checks
the frozen result, exact manifest membership, evidence and Run ownership before
atomic collection. A published archive is not accepted Audit coverage.

## Delivered behavior

Reports and live allocation metrics carry optional bounded `completion` facts:
kind, phase, accepted/total item counts, reminders, and stable failure code.
Public attempt diagnostics expose a detached copy with a bounded explanatory
message. There are no new model/tool calls, token charges or LLM errors for
programmatic phase changes and publication failures. Unknown optional kinds or
phases are discarded; historical and ordinary reports retain omission behavior.
The shared fixtures live under `api/testdata/audit-completion`, separate from
operator configuration.

The new opt-in closure is `source-checklist@3` → `audit-source-check@4` →
`audit_source_checker@4`, with `audit_completion_worker@1`. It uses one
passthrough Worker, `audit-results@2`, and no target summarizer. Failure and
interruption end the child Run, allowing Audit policy to decide on a fresh Run.
The isolated example test proves that adding this closure leaves the old
profile's digest and completion protocol unchanged.

## Verification

Passed in the working tree:

```sh
go test ./internal/contracts ./internal/config ./internal/telemetry ./internal/httpapi/public
go test ./internal/contracts ./internal/auditimport ./internal/auditcontroller ./internal/runservice ./internal/scheduler ./internal/planner ./internal/artifacts
cd runtime
uv run --frozen pytest tests/test_audit_completion_contracts.py tests/test_audit_result_collector.py tests/test_audit_result_publication.py tests/test_audit_results_toolset.py tests/test_audit_completion_continuation.py tests/test_audit_completion_runtime.py tests/test_audit_completion_diagnostics.py tests/test_adk_runtime.py tests/test_worker_state.py tests/test_metrics.py tests/test_instrumentation.py tests/test_contracts.py
uv run --frozen pytest tests/test_audit_completion_diagnostics.py tests/test_audit_completion_runtime.py
```

The broad Python run passed 373 tests; after extending shared live-State cases,
the focused run passed 84 tests. Ruff passed for the changed Python code.
Both public clients were regenerated from the OpenAPI contract.

An isolated checkout containing only this task's changes also passed the same
broad Python command (398 tests), the Go contract/config/telemetry/public API
packages, and `go test ./internal/publicclient/...`. This separates the review
from concurrent operator catalog and telemetry work in the shared workspace.
`make verify-public-api` and `cd ui && corepack pnpm typecheck` also passed.

This review used deterministic offline models and unit/test API transports.
The commands above did not require PostgreSQL; optional database tests can skip.
They do not close V39-007's mandatory real artifact API/Runtime ZIP/Go importer
bridge and persistence/restart release gate. No live Audit, service deployment,
credential or running catalog was changed by this task.
