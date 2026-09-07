# V39-001: Audit completion contracts and ADK continuation

Status: inert contract/interface implementation validated offline on 2026-09-07.
The [target design](../spec/25-audit-worker-finalization.md) remains opt-in.
No production selector, running Audit, Runtime capability advertisement or
completion dispatch has been enabled by this task.

The protocol version split described below was removed by
[V50-001](../../tasks/v50-001-unified-private-contracts.yml). Current registration
and allocation models and fixtures live only under `v1alpha1`; this review records
the earlier V39-001 implementation boundary.

## Trusted configuration and wire shapes

An AuditProfile check binding can select `workerCompletion` with
`kind: audit-check-results@1`, one Stage and one logical Agent. Validation requires
single-Worker `passthrough@1`, both `audit-results@2` tools, no @1 selection and no
terminal summarizer. The selected Stage must own the required canonical ZIP result
and receive exactly one mapped required task package and execution manifest.
Input slot names are derived from mappings. Another Stage cannot supply the same
canonical output. Unknown fields/kinds and invalid owners are rejected.

The contract is included in the canonical AuditProfile digest only when present.
Omitted contracts preserve old bytes/digests. Snapshot decoding checks the retained
closure; no catalog lookup participates. Legacy role digests cannot authenticate a
new completion selection. Cloning does not share the mutable completion pointer.

All reachable retry/escalation branches are checked against their fully effective
execution configuration. Current escalation changes model access, not AgentTemplate
or result mappings; unknown authoring fields cannot introduce replacement templates.
The retained template/output constraints apply to every effective variant. Authored
escalation and a tampered effective Worker selection have regression tests.

Both AllocationSpec versions accept optional `completionContract` with exact,
distinct `inputs` refs and a versionless output in the allocated Worker's namespace.
The allocation template must select @2 and omit a summarizer. Private v2 registration
adds optional `capabilities.completionContracts`; omitted means unsupported. Declared
lists are closed, bounded and unique. Production registration still omits this
capability and no @2 execution factory is registered. Descriptor parity explicitly
distinguishes an inert authoring descriptor from an installed Runtime factory.

Production private-v2 allocation/registration schemas, the legacy allocation schema
and dedicated contract/authoring schema fragments accompany 27 shared Go/Python
cases. Actual AuditExecution ownership, Run grants, StageContentRequest binding,
Server propagation and unsupported-fleet handling belong to V39-002/V39-005.
The new types alone do not grant execution authority.

## Independent collector/encoder/publisher interfaces

`runtime/src/contractor_runtime/toolsets/audit_results/contracts.py` defines immutable,
invocation-owned normalized items, positive revisions, trusted-order snapshots,
sealed complete snapshots and separate recorded/publication receipts. Fields are
bounded and sensitive bytes/summaries are excluded from representation. None of
these values is stored in model-editable ADK State.

`AuditTrustedInputs` retains exact task-package and execution-manifest bytes. Its
owner pins the task-package digest. Both are needed by the pure encoder: requested
coverage and task-local evidence rules live in the task package, while the manifest
pins execution membership. Collector admission and final publication must call the
same publisher-owned encoder over these immutable inputs; a second approximate
size calculation is outside the contract.

Limits are 64 assigned items, 16 KiB UTF-8 summaries, 512 coverage/proposal values
per list, 256 evidence records across the snapshot, 8 MiB collected canonical data,
and 16 MiB per member/final package. Revisions are positive exact JSON integers
bounded by `2^53-1`. The encoder returns actual package bytes and measured sizes
including overhead. Concrete package validation/encoding is V39-004, and collector
acceptance/CAS/replay/sealing is V39-003. The interfaces specify exact update replay,
atomic full-batch admission and rejection of all updates after sealing.

The common boundary returns `ContinueCompletion`, `CompleteCompletion` or
`FailCompletion`. The reminder allowance is two per logical invocation; V39-005
owns enforcing it and choosing the ordinary versus Audit strategy. A recorded
receipt proves local acceptance; a publication receipt identifies verified exact
ZIP bytes. Audit importer acceptance remains a later independent fact.

## Real ADK continuation proof

The pinned environment uses **google-adk 2.8.0**. Runtime uv.lock SHA-256:
`54638e90aa96b146bc02215106272fc74cd8404fdc8890f24758a066c963ba0c`.
`test_audit_completion_continuation.py` uses the real `Runner`, `App`, `LlmAgent`,
`InMemorySessionService` and production `WorkerInstrumentationPlugin`, with a
scripted `BaseLlm` and artifact-observation callback. No network model is used.

The proof executes two Runner turns under the same invocation/session: four model
calls and two tool calls. It checks one begin, continuous tool ordinals, all observed
artifact refs, cumulative model/tool budgets and one terminal State completion.
Repeated completion is inert; stale IDs, another session, missing/reused continuation
tokens and continuation after completion fail. The next model call at the exact
budget cap raises the existing WorkerBudgetExceeded signal.

Runtime must explicitly arm `prepare_continuation(invocation_id=...)` after the
previous turn settles. One `before_run_callback` consumes that token and installs
the current trusted State snapshot without initializing a reducer, fetching fresh
workspace observation metadata or resetting counters. The token pins the original
session identity and cannot be used while callbacks remain pending. Terminal/close
cleanup clears it.

V39-005 must place the Runner loop inside the existing logical invocation/session
lifecycle, after one budget start and one observed-ref reset, before terminal State
and session release. It must call the seam only for a Runtime-authored reminder,
retain the same deadline/policy, enforce the reminder bound and preserve fatal
budget/cancellation/sandbox failures. This proof establishes the instrumentation
seam; production completion dispatch and deterministic ZIP publication are pending.

## Verification

The following gates passed in the shared worktree and in a separate checkout of
this change, preserving unrelated edits:

```sh
make verify-wire-contracts
go test ./internal/config ./internal/contracts
cd runtime
uv run --frozen pytest tests/test_audit_completion_contracts.py \
  tests/test_audit_completion_continuation.py tests/test_allocation_contracts.py \
  tests/test_contracts.py tests/test_instrumentation.py tests/test_capabilities.py
uv run --frozen pytest tests/test_adk_runtime.py
```

The required Runtime/extra capability selection contains 162 passing tests; the
wire gate contains 134 passing Python tests plus the Go contracts package. No
required case is skipped. The isolated ADK regression adds 62 passing tests;
the shared worktree also passes its five additional pre-existing ADK tests. These are contract/lifecycle tests, not instruction-quality
or security-accuracy evaluations.
