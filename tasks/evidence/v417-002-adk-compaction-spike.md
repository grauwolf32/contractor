# V417-002: accepted Worker checkpoint adapter

Decision date: 2026-10-26. This is test-local design evidence, not a shipped
`compact@1` capability. Executable proof: `runtime/tests/test_compaction_spike.py`.
Exact inputs, commands and outcomes are in the adjacent JSON evidence.

## Selected boundary

Use the public `LlmAgent.before_model_callback` request projection, with a
Runtime-owned immutable checkpoint. Do not enable native automatic, token or
post-invocation compaction, resumability, semantic retries or SDK-private helpers.

The native public `EventCompaction` interval is inclusive and timestamp-only.
Real Runner/Gateway cases show it absorbs an unselected retained neighbor with
an equal start/end timestamp, or an event whose timestamp is out of source order.
A later subsuming marker replaces the older memory. These properties do not
establish exact source identity or complete FC/FR group ownership. Native
compaction is therefore not the v1 adapter, even for ordinarily distinct clocks.

The chosen projection keeps exact source event IDs in ingestion order. It
rejects duplicate IDs, partial/error events, interleaved/incomplete/orphan calls,
foreign branches/isolation scopes and native markers before any checkpoint write.
A group comprises the entire model response and every matching tool response;
the fixture includes two calls in each model response and a 16-group recent tail.
Do not sort source history by timestamp. Reject a foreign session before model
transport. Production must also fence pending partial events at ingestion:
SessionService does not necessarily retain them for a later scan.

A checkpoint consists of the same pending identity, original session/branch/
isolation owner, source IDs/ordinals, generation and one validated summary. The
coordinator appends an ordinary content-free Event through SessionService, with
native compaction absent, before publishing the new request projection. Exact
checkpoint metadata remains invocation-local; this is not a new durable store.
A production adapter must retain the entire pending identity and compare it
when reconciling; an Event ID alone is insufficient. The spike's single owner
and local pending variables supply that identity without concurrent mutation.

## Logical invocation and continuation

The executable proof uses three real Runner turns, two accepted checkpoints,
five main calls, two independent tool-free compactor calls and eight tool effects.
The same logical invocation ID, State, `_InvocationBudget`, deadline admission
closure, tool instances and exact observation collector survive. SDK IDs differ;
`prepare_continuation` authorizes the next turn without beginning State again.
The actual next Gateway request has the exact protected task, exactly the latest
rolling memory and intact recent FC/FR groups; older groups and memory disappear.
Raw source events remain available to the owning session.

The coordinator closes each async generator in its owner task after completed
FunctionResponses, outside instrumentation callbacks. It then calls the
compactor, commits the checkpoint and supplies a fresh explicit Runtime-authored
user message. No nested callback model call and no `new_message=None` are needed.
Compactor calls consume the original model/token budget while preserving the
latest main prompt measurement. Exact observations use the existing bounded
result projection; parallel observation-log overlap does not create new refs.

This is real SDK/session/Gateway/tool integration with controlled HTTP responses,
not a Server/Runtime/A2A process or semantic model-quality claim. Shared-session
cross-invocation behavior and the 4096-group archive belong to subsequent tasks.

## Failure and cleanup contract

Empty/None/oversized/schema-invalid/output-limited/tool-bearing or demonstrably
ineffective output is rejected before publication of a checkpoint. Exceptions
propagate; native silent skip is forbidden. A received rejected Gateway response
still contributes its known usage; an unreceived response consumes its admitted
call, without inventing token usage. Existing Gateway transport policy owns any
physical transport retry; the negative property fixtures disable physical retries
to establish that the coordinator adds no semantic retry.

After an accepted checkpoint, a newly measured main response can demonstrate
ineffectiveness. The fixture retains the marker and both charged responses, then
raises before the response can dispatch another tool. Never roll back accepted
observations, memory or charges to hide that outcome.

Acknowledgement loss is reconciled by inspecting exactly one equal pending marker
in the same session; inspection unavailability fences continuation. Do not blindly
append again. ADK 2.13's in-memory service deduplicates equal redelivery, but keeps
a different event with the same ID: ID reuse does not establish idempotency.
Cancellation before admission, after append and while provider/append/continuation
work is blocked prevents later semantic calls. The task owner cancels and joins
the blocked operation and closes Runner/model handles before fixture cleanup.
Cancellation/expiry wins over a late compactor success; already committed markers
and already dispatched remote effects are not promised rollback.

## Reversal condition

Reopen on an SDK/lock change, a public callback moving after semantic dispatch,
changed session-copy/append behavior, or a grouping/scope case not represented by
this proof. A future exact-ID native API may replace projection only after running
these cases against the new API. Production integration must reuse or replace
this test-local adapter rather than ship it as a second compaction engine.
