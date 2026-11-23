# V420-001: frozen Planner compaction integration contract

Decision date: 2026-10-26. This is the implementation contract for V420-002,
not accepted production configuration. The numbered owning specifications and
registries change with that implementation. The spike makes no Scheduler,
production Runtime, API, generated-client, dependency or model-selection change.

## Adapter and ownership

Use one bounded request projection through the public Go `BeforeModelCallbacks`.
The callback obtains the current session through the public SessionService:
Go ADK 2.6 callback `Session()` is unsupported. No SDK-private package is imported.
One outer Planner owner closes/consumes the Runner iterator at a complete tool
group, performs one tool-free compactor call outside callbacks, writes one
checkpoint, then starts another explicit SDK user-message turn in the same
Planner/session. Do not recursively invoke `Planner.Run`. Native compaction,
its estimator, tail-retention repair, post-invocation work and resumability stay off.

Keep the original absolute deadline, model/Worker/token counters, plan instance,
prepared Worker handles and current task authority. SDK invocation IDs need not
stay equal. Check cancellation/admission before every semantic request and
checkpoint publication. No work detached from the Planner may outlive return.
The compactor request timeout is min(remaining absolute deadline, effective
Gateway requestTimeout); session writes and cleanup retain their existing budgets.

Source IDs, ingestion ordinals, session/branch/isolation and generation define the
checkpoint. Complete FC/FR groups are indivisible in the active prompt. Inclusive
or equal timestamps cannot choose membership. Reject incomplete/interleaved,
foreign, error or stale selections before mutation; a pending partial event is
fenced by its ingestion owner even when ADK omits it from storage. Previous
summary markers are bookkeeping, never source groups; replacing memory also
covers the earlier prefix without resummarizing its entire raw archive.

A content-free ordinary Event goes through the existing required reduced-fact
append before publication of the invocation-local checkpoint projection. This
adds no PostgreSQL entity or transcript. Lost acknowledgement fences continuation
unless the same pending identity/generation/source set and existing session/fact
sequence can be proved. No new marker or Worker is dispatched to resolve doubt.
A restart still cannot resume a partial Planner; completed-result recovery is
unchanged and model-free. The tested conservative Go choice is to fence every
required append error, including an acknowledgement lost after its real commit.

## Configuration shape and selection

```yaml
spec:
  stages:
    review:
      planner: streamline@1
      plannerContextManagement:
        strategy: compact@1
        modelPolicy: compactor@1
        instructions:
          ref: instructions/planner-compaction.md
        contextWindowRatio: 0.75
        keepRecentGroups: 4
        maxCompactions: 8
```

An omitted authoring field normalizes to `{strategy: none}`. A resolved snapshot
always contains exactly `none` or the complete `compact@1` branch, with all defaults
explicit. Explicit null, unknown/duplicate fields and mixed branches fail closed.
`none` accepts no compact fields. `compact@1` requires an exact ModelPolicy ref and
resolved nonempty instructions, a modeled Streamline/Router and a main-model
context window. Passthrough/model-free stages cannot select it.

The independent reference-only execution leaf is
`executionConfig.planner.compactorModelPolicy`. It follows existing Planner
selection precedence (catalog defaults, Workflow, Run, selected escalation),
exact dependency pinning and effective Planner Gateway/credential. The strategy,
instructions and thresholds remain Workflow-owned. Inline model, URL, credentials,
limits or timeouts are forbidden. An inapplicable selection is rejected.

Compactor ModelPolicy requires contextWindowTokens, positive maxOutputTokens below
that window, maxModelCalls exactly 1 per cycle and positive finite maxTotalTokens.
maxToolCalls/maxWorkerCalls are forbidden. It has no domain/Memory/State/Worker tools
or inherited Skills. The model response schema is exactly `{summary: string}`.
Reasoning/thought content is excluded in this first Planner slice.

There is no Planner terminal summarizer. Reject `terminal@1`,
`summarizerModelPolicy`, terminal fallback and mixed fields on this owner.
Workers independently keep none/terminal/compact policy, so a compacting Planner
can dispatch a terminal-summarizing Worker. Existing passthrough summary escalation
keeps an explicit `none` target in the first slice; it neither inherits hidden
compaction nor needs new Scheduler state or transitions.

## Trigger, accounting and failure precedence

Only latest actual main-model prompt usage triggers at
`min(floor(window * ratio), window - maxOutputTokens)`, after the tool group and
before another main turn. Valid ordinary terminal completion wins. Summary usage,
cumulative spend and input estimates never replace that measurement. After a
checkpoint, another cycle requires a fresh main measurement in the new generation
and a newly eligible prefix. Hard cumulative limits never trigger compaction or
reset. Every compactor admission and known token usage also consumes the original
Planner totals, once, including rejected responses. Missing/invalid usage is
`planner_gateway_invalid_response`: charge the call, retain previously known usage,
never fabricate tokens, and stop under the existing Gateway classification.

Precommit invalid/empty/declined/tool-bearing/truncated/oversized/schema-invalid or
ineffective summary produces nonretryable `planner_compaction_failed`, no marker
and no continuation. Bounded reasons: `empty`, `invalid`, `output_limit`,
`oversized`, `tool_output`, `ineffective`, `scope`, `input_capacity`.
Postcommit ineffectiveness retains the checkpoint and charges and stops before
another semantic dispatch/cycle. Exhausted compaction, group, retained-byte, tail
or accounted-peak ceilings use nonretryable `planner_compaction_limit_exceeded`.
No terminal fallback, semantic retry or model substitution is allowed.

Existing classified Gateway outages/rejections, session unavailable/rejected,
cancellation/deadline and PLN-R106 outcome retryability remain authoritative.
A sticky required-write failure is checked before the next semantic call AND
before accepting completion, even if the iterator ended after a valid finish.
The ordinary/final PostgreSQL precedence and cancellation regression suites in
this package are required alongside the new checkpoint fixtures. No checkpoint
or repair starts after a terminal decision. Earlier task wording about finish
precedence describes a historical finding already repaired before this spike.

## Bounded authority and recovery

Every projected request reconstructs original objective, parameters, exact inputs,
result contract and distinct saved continuation from their existing authority.
Add only a bounded current-plan view (current subtask/active dispatch/revision),
one rolling untrusted summary and intact recent groups. A summary cannot create
an ArtifactRef, override an observation or rewrite a plan/result contract. Do not
inject 1000 subtasks or an entire result index into a prompt.

Keep an invocation-local index of exact validated Worker results/refs and allowed
source projections, backed by canonical data/offsets rather than gratuitous full
copies. Expose `read_context_history` only to the live modeled Planner under
compact@1. The actual functiontool/Runner proof delivers an old exact result in
the next model request despite contradictory lossy memory. No public endpoint or
durable raw history is added.

A page returns UTF-8 document fragments with source ordinal, byte offset and an
opaque next cursor. They are data, never protocol FC/FR messages. Complete group
ownership is preserved in source descriptors, including when a large group must
span pages. Cursors are random 128-bit capabilities stored in a bounded local map,
bound to session, context generation, index revision, view and logical Worker.
Invalid, foreign, stale or expired cursors return no data. Advancing a cursor
consumes it; at most 8 chains are outstanding. Index changes invalidate cursors.
All cursors and index contents expire when the invocation releases.

## Frozen ceilings and measurement definitions

| Quantity | Frozen design value |
| --- | --- |
| Context ratio | Strict (0,1), default 0.75 |
| Cycles per invocation | 1..16, default 8 |
| Recent complete groups | 1..16, default 4 |
| Accepted summary | 16 KiB UTF-8 |
| Required recent tail | 1 MiB serialized complete source groups |
| Aggregate retained history/results/summaries | 32 MiB and 4096 complete source groups |
| Accounted peak | 64 MiB |
| Actual encoded Gateway request | Existing 4 MiB ceiling |
| History page | 32 KiB encoded JSON and 8 source groups/fragments |
| One archive fragment | At most 16 KiB UTF-8, reduced at a rune boundary |
| Outstanding cursor chains | 8 per live invocation |

Count UTF-8/serialized backing once when actually shared; count every raw ADK,
canonical archive, request, tail, summary, page and transient copy separately when
it has its own backing. Include a bounded per-event/index structural reservation.
Admission is before allocation/copy when possible. The 4096-group fixture counts
both a raw and canonical representation plus requests/tail/pages/summary and a
1-KiB structural reservation per group; it remains below 64 MiB. These are
accounted-data ceilings, not an exact Go heap/RSS claim. Production must maintain
this ledger as data is ingested; a transient copy that exceeds it is a finite
limit failure, even if its eventual retained result would fit.

The large-result fixture exercises the independent 64-KiB result text, 256-KiB
payload and 128-ref cardinality envelope. It does not assert those maxima must
fit simultaneously in the existing result contract. Fragmented retrieval is the
selected alternative when an otherwise valid group cannot fit a 32-KiB page.
A requested recent tail exceeding 1 MiB fails; do not silently lower keepRecentGroups
or split active prompt groups. A full maximum plan/result set is never injected.

The compactor window is independently admitted. The tested conservative text
capacity rule counts the complete serialized tool-free request bytes as input
tokens plus a protocol reserve and output reserve; it is intentionally stricter
than the 4-MiB request bound. It is justified for non-overlapping, byte-preserving
text tokenizers (BPE/byte-fallback routes), where each token consumes at least one
input byte. The operator's pinned Gateway route must preserve that property and
have bounded framing; injected route prompts are part of its capacity metadata.
Use a 1024-token framing reserve only for at most 8 messages; otherwise use
128 tokens per message plus 1024. A route with different token expansion/framing
requires a supported exact tokenizer or fails admission; it cannot silently use
the bound or a different model. ContextWindowTokens already trusts the pinned
operator route metadata; misdeclared route capacity remains an external context
rejection, never a retry-with-truncation. Reopen this assumption for multimodal
input, opaque route prompt injection or tokenizer changes.

This task freezes values and counting definitions here because spec conventions
forbid describing an unimplemented capability as current behavior. V420-002 must
install the numbered owning limits/errors/failure/lifecycle entries together with
production enforcement, using this table. This resolves R10's registry wording
with R12's explicit implementation boundary; no pending-only spec requirement is
introduced by the spike.

## Alternatives and reversal

Native timestamp markers are rejected as the default adapter: equal-time and
out-of-order membership and native repair/callback/error behavior are not the
application contract. Full raw transcript reinjection violates the intended
bounds; summary-as-authority loses exact facts; new persistent checkpoint stores
and native restart resume violate existing ownership. The tested public request
projection is sufficient without changing Scheduler.

Reopen on SDK/model-route changes, changed callback timing or append semantics,
an unsupported complete-group shape, measured valid input outside this envelope,
or inability to maintain the retained-data ledger. Rerun both SDK integration
fixtures before changing adapters. V420-002 consumes/replaces these test-local
helpers rather than publishing a parallel engine or unused exports.
