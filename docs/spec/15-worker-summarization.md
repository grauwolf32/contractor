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
    softTotalTokens: 220000
    softPromptTokens: 100000
```

Omission disables summarization. The object is closed and contains:

- mandatory exact `modelPolicy` selector;
- at least one of `softTotalTokens` or `softPromptTokens`;
- positive soft limits lower than the corresponding hard Worker bound when the
  comparison exists.

The summarizer uses the same pinned Worker LLMGatewayConfig and credential as
the normal Worker but its own exact ModelPolicy/model alias. That policy is a
tool-free Worker summarizer policy: it requires `maxOutputTokens` and
`maxTotalTokens`, requires `maxModelCalls: 1`, and permits no
`maxToolCalls`/`maxWorkerCalls`. Runtime also enforces this one-call bound;
larger or missing `maxModelCalls` is a configuration error. ExecutionConfig
override of only the summarizer model or use of a different
Gateway/credential is deferred.

The resolved summarizer block and exact ModelPolicy are included in the
AgentTemplate digest, immutable Run snapshot and AllocationSpec. Runtime
verifies both digests before Worker construction. Labels may select the same
physical Worker Gateway route under [07], but cannot enable summarization or
change its policy/soft limits.

## Trigger semantics

Runtime evaluates soft limits only from non-negative provider usage already
observed on completed normal Worker responses:

- `softTotalTokens` compares cumulative normal-loop total tokens;
- `softPromptTokens` compares the most recent normal-loop prompt token count.

Missing provider usage never invents a token estimate and therefore cannot
trigger a token-based summarization rule. Independent model/tool/hard-token
budgets remain mandatory.

The trigger is checked at a safe boundary:

1. a model response completes;
2. every tool call selected by that response either completes or fails;
3. if the response already contains a valid `WorkerModelResult`, Runtime uses
   it normally with `summarized: false`;
4. otherwise, before starting another normal model call, reaching either soft
   threshold requests terminal summarization.

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
the summarizer.

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
The aggregate contains attempted/succeeded, model calls, provider-reported
input/output/total tokens, missing-usage count and a stable failure code. It
contains no transcript/result text or endpoint/model-provider body. The trusted
Server envelope supplies exact ModelPolicy and Gateway attribution.

Summarization is semantic work and therefore occurs only inside the active A2A
invocation while allocation writes are allowed. It never runs during
finalize, abort, lease-loss drain, report collection or release. Cancellation
winning before its result stops the call and produces no WorkerResult; normal
bounded abort semantics apply.

## Deliberately deferred

- ADK `EventsCompactionConfig` and continuing after compaction;
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
