# V39-005: common Worker completion boundary

Normal ADK finishes now pass through one dispatch before terminal Worker State,
WorkerCompletion, workspace export and session cleanup. Omitted completion
contracts retain the ordinary one-shot result finalizer and optional summarizer.
An explicitly prepared Audit binding instead checks the invocation's collector,
issues at most two deterministic reminders, and seals/publishes a complete set.
Audit Workers construct no LLM result finalizer. Empty terminal text is sufficient
when collection is complete; terminal prose cannot satisfy missing items.

Preparation reads both input artifacts by the exact contract revisions, checks
the returned exact receipts, media types, task/manifest digests and membership,
and retains immutable bytes. Invocation begin derives a fresh owner and collector
from those bytes. Both tools resolve the active collector and verify the ADK
invocation ID. Shared sessions retain no accepted drafts across invocations.

Continuation uses the V39-001 production instrumentation seam unchanged:
prepare_invocation runs once, prepare_continuation admits each additional Runner
turn, and the reducer, tool ordinals, counters and artifact observation list remain
continuous. State completion and session cleanup run once. A single Audit deadline
is fixed from the existing request timeout at invocation start and is shared by
all turns and publication; existing external cancellation remains authoritative.

An actual budget exception is retained separately from usage counts. A complete
normal finish on the last permitted call/token may publish, while a forbidden
next call, token overshoot, model error event, sandbox failure, timeout or
cancellation prevents success. Programmatic completion/publication errors are
translated before generic model-error accounting. No model/tool call is invented
for publication. Discard is shielded and awaited before releasing invocation
ownership, including cancellation during normal cleanup.

Verified publication receipts join existing trusted artifact observations in the
result projection. Model-authored refs do not enter this path. The deterministic
summary says results were recorded and published; it does not claim Audit
acceptance. Retention and semantic acceptance remain Server importer work.

`audit-results@2` is installed in the factory registry, but its probe requires
artifact transport. Completion advertisement additionally requires a successful
ADK probe, the implementing runtime factory, and both v2 tools. Allocation
preparation rejects a v2 selection without a trusted contract, unsupported
completion capability, invalid target selection or a terminal summarizer.
Registration preserves omission when completion support is unavailable.

Validation uses the pinned real ADK Runner with scripted models and offline
artifact IO. It covers missing/partial/invalid results, reverse-order completion,
exact model/tool/token limits, continuous artifact observations, publication
failure, shared-session reuse, cancellation during continuation/publication and
cleanup, sandbox failure, timeout and allocation preparation/registration.

```sh
cd runtime
uv run --frozen pytest tests/test_audit_completion_runtime.py tests/test_audit_completion_continuation.py tests/test_adk_runtime.py tests/test_instrumentation.py tests/test_result_finalizer.py tests/test_app.py tests/test_capabilities.py tests/test_worker_summarizer.py tests/test_allocation.py tests/test_audit_completion_contracts.py tests/test_audit_result_collector.py tests/test_audit_results_toolset.py tests/test_audit_result_publication.py
```

All 295 regression tests pass; changed Python files also pass Ruff. No live model
or deployment is involved. Phase diagnostics and new opt-in catalog versions
remain V39-006; the database/importer release gate remains V39-007.
