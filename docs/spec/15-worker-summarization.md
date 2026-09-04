# 15 — Optional terminal Worker summarization

Status: **Working agreement; post-observation increment**

Depends on: [01](01-agent-template.md),
[04](04-execution-lifecycle-and-metrics.md) and
[14](14-worker-results-and-live-state.md)

## Goal

A long-running Worker should be able to stop before its hard context/token
budget and still return one useful typed semantic result. This document defines
an optional, deterministic terminal summarizer owned by Runtime Agent.

It is deliberately different from conversation compaction:

- **terminal summarization** stops the normal Worker loop, makes one separate
  tool-free model call and returns a final `WorkerResult` with
  `summarized: true`;
- **context compaction** rewrites older events and lets the same Worker continue
  using tools. Google ADK has experimental compaction machinery, but Contractor
  does not enable or wrap it in this increment.

## Configuration

An `adk@1` AgentTemplate may contain one optional block:

```yaml
spec:
  summarizer:
    modelPolicy: worker-summarizer@1
    contextWindowRatio: 0.9
    cumulativeBudget: 220000
```

Omission disables summarization. The object is closed and contains:

- mandatory exact `modelPolicy` selector;
- optional `contextWindowRatio`, defaulted to `0.9` during resolution and
  required in the normalized AllocationSpec, strictly between zero and one;
- optional positive `cumulativeBudget`, lower than the normal Worker's hard
  `maxTotalTokens`.

The block has fixed `terminal@1` semantics in this API version. It is not a
generic bag of compaction options and will not be silently reinterpreted as a
continuing strategy. When a second strategy is implemented, authoring will gain
an explicit strategy/version discriminator and a strategy-specific closed
schema.

The summarizer uses the same pinned Worker LLMGatewayConfig and credential as
the normal Worker but its own exact ModelPolicy/model alias. That policy is a
tool-free Worker summarizer policy: it requires `contextWindowTokens`,
`maxOutputTokens` and `maxModelCalls: 1`, and permits no
`maxToolCalls`/`maxWorkerCalls`. `maxTotalTokens` is optional because the call
count is already one; when supplied, Runtime checks it against reported usage
after that call. Runtime also enforces the one-call bound; larger or missing
`maxModelCalls` is a configuration error. Multiple AgentTemplates may reuse one
exact common summarizer ModelPolicy. ExecutionConfig override of only the
summarizer model or use of a different Gateway/credential is deferred.

The resolved summarizer block and exact ModelPolicy are included in the
AgentTemplate digest, immutable Run snapshot and AllocationSpec. Runtime
verifies both digests before Worker construction. Labels may select the same
physical Worker Gateway route under [07], but cannot enable summarization or
change its policy/soft limits.

## Trigger semantics

An enabled normal Worker ModelPolicy must pin `contextWindowTokens`. Runtime
derives one prompt boundary from the effective Worker policy and resolved
summarizer configuration:

```text
context_boundary = min(
  floor(contextWindowTokens * contextWindowRatio),
  contextWindowTokens - maxOutputTokens,
)
```

The second term reserves the configured maximum normal response. Configuration
is invalid when `maxOutputTokens >= contextWindowTokens`. Runtime requests
summarization when the most recent completed normal response reports
`prompt_token_count >= context_boundary`.

The optional `cumulativeBudget` independently compares the sum of
provider-reported `total_token_count` across all completed normal-loop model
calls in this A2A invocation. It is a soft cumulative spending boundary which
can stop a long sequence of individually small calls before its expensive tail.
It does not describe model context capacity and does not replace the normal
Worker policy's hard cumulative `maxTotalTokens`.

Missing provider usage never invents a token estimate and therefore cannot
trigger the affected rule. A zero-valued total produced by an adapter for an
omitted provider `usage` object is also unavailable. Non-negative counters are
internally inconsistent when `prompt + completion > total` or cached input
exceeds prompt input; Runtime drops that response's token counters closed,
increments `tokenUsageUnavailable`, and lets independent model/tool-call
limits continue to bound execution. The context rule observes the last
completed valid provider prompt, not an exact tokenization of the prospective
prompt after new tool results. Its ratio and output reserve are a deterministic
pre-emptive boundary, not a guarantee that every provider will accept the next
request. Independent model/tool/hard-token budgets remain mandatory.

The trigger is checked at a safe boundary:

1. a model response completes;
2. every tool call selected by that response either completes or fails;
3. if the main Worker response contains ordinary terminal semantic text,
   Runtime completes it through the mandatory result finalizer from
   [14](14-worker-results-and-live-state.md) and returns it with
   `summarized: false`;
4. otherwise, before starting another normal model call, reaching either the
   derived context boundary or optional cumulative budget requests terminal
   summarization.

No in-flight tool is cancelled merely because a completed response crossed a
soft threshold. No new normal tool or model call starts after the request.
Hard exhaustion or provider context failure that occurs before this boundary
uses the existing Worker failure path and does not retroactively invoke the
summarizer.

The instrumentation plugin records only a private
`summarization_requested` control signal. It must not make a nested LLM call
inside an ADK callback. The Runtime invocation coordinator unwinds the normal
Runner and invokes the independent summarizer afterward.

## Summarizer input and output

The summarizer has no Contractor/domain tools, Agent Skills, Artifact client,
MemoryTools or workspace handles. Its input is one Runtime-built bounded
document containing:

- authoritative subtask ID, objective and task instructions;
- immutable string parameters and named inputs already visible to Worker;
- a chronological, non-thought projection of model-visible normal-loop events;
- the deterministic latest `WorkerObservations`.

Trusted result bindings remain private Runtime data and are not supplied as a
separate summarizer input. An ArtifactRef may reappear only if it was already
part of a bounded model-visible event projection.

Runtime never includes system/runtime internals, hidden thought, credentials,
headers, provider exceptions, host paths or arbitrary ADK State. The event
projection is capped at 512 KiB. Immutable task data is retained first; the
newest complete event groups that fit are retained next, preserving their
original order, and a deterministic `transcriptTruncated` marker reports any
omission. Tool payloads remain subject to their existing model-visible bounds.

The separate model uses the same strict two-field `WorkerModelResult` schema:

```python
class WorkerModelResult(BaseModel):
    subtask_id: str
    result: str
```

Runtime performs the same exact subtask check, secret scan, UTF-8/size bound,
artifact projection and final wire validation as for a normal result. It then
sets `summarized: true`; the model cannot set or clear that flag. Deterministic
observations are copied from the completed normal loop rather than authored by
the summarizer. Because this tool-free terminal summarizer already produces the
strict `WorkerModelResult`, Runtime does not invoke the ordinary result
finalizer after it.

Once terminal summarization is requested, Runtime makes exactly one summarizer
attempt. Missing/invalid/mismatched/oversized output, provider failure or its
independent budget exhaustion returns one safe retryable
`worker_summarization_failed` (with a more specific bounded cause in metrics),
never resumes the normal Worker and never attempts repair.

## State, metrics and lifecycle

The Contractor Worker State records:

- whether summarization is disabled, not requested, requested, succeeded or
  failed;
- the normal-loop state revision that caused the request;
- bounded summarizer model/token counters and one stable error code.

It does not retain the summarizer input or output separately. The semantic
output exists only in the returned `WorkerResult`/live Planner conversation;
durable Stage content remains Planner-owned.

`ExecutionMetrics` gains a separate optional summarizer aggregate so normal
Worker usage and terminal summarization cost can be evaluated independently.
The aggregate contains attempt/success/failure counts, model calls,
provider-reported input/output/total tokens, missing-usage count and bounded
stable failure-code counts. It contains no transcript/result text or
endpoint/model-provider body. The immutable AllocationSpec supplies exact
summarizer ModelPolicy and Gateway attribution.

Summarization is semantic work and therefore occurs only inside the active A2A
invocation while allocation writes are allowed. It never runs during
finalize, abort, lease-loss drain, report collection or release. Cancellation
winning before its result stops the call and produces no WorkerResult; normal
bounded abort semantics apply.

## Deliberately deferred

- ADK `EventsCompactionConfig` and continuing after compaction;
- rolling, hierarchical, extractive and artifact-backed summarization
  strategies; their introduction requires an explicit discriminator rather
  than changing `terminal@1` semantics;
- summarizer retries, tool access or model-authored observations;
- a separate summarizer Gateway/credential or executionConfig override;
- estimated-token triggers when provider usage is absent;
- reactive recovery after a provider has already rejected context length;
- UI mutation of summarizer policy for an existing Run/allocation.

## Invariants

1. Absence of `summarizer` preserves hard-budget behavior exactly.
2. A soft trigger starts no additional normal Worker side effect.
3. Runtime, never either model, owns `summarized`.
4. The summarizer has one call, no tools and an independent pinned ModelPolicy.
5. Summarizer failure cannot resume the normal loop or invoke a repair loop.
6. Normal and summarizer usage remain separately attributable and secret-free.
7. Finalize/abort/release never invoke the summarizer.
