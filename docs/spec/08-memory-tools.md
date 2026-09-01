# 08 — Shared MemoryTools

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md),
[01](01-agent-template.md), [03](03-artifact-plane.md) and
[04](04-execution-lifecycle-and-metrics.md)

## Goal

`memory-tools@1` gives a Planner and its logical Worker a small shared notebook
without adding another content store or exposing Artifact revisions to the
model. Every note is one immutable-versioned RunScope artifact. MemoryTools is
the purpose-specific, namespace-bound abstraction and model-tool wrapper over
those artifacts.

Memory is coordination state, not hidden prompt context. No note is injected
automatically into a Planner or Worker conversation. A participant sees a note
only by calling an explicitly selected MemoryTools operation.

## Memory Namespace and lifetime

A **Memory Namespace** is the logical view identified by:

```text
(run_id, resolved Agent Namespace)
```

It is not another ArtifactStore scope. Its notes live in the same
`RunScope(run_id)` and Artifact Namespace that the Stage binding already
assigns to the logical Worker.

- a single-Worker Streamline Planner and its Worker use the same Memory
  Namespace;
- a Router Planner has one view per logical Worker and selects the view with a
  validated `worker_name` argument;
- a Worker is already bound to its own view and never receives `worker_name`,
  `run_id` or Namespace as a model argument;
- logical Workers whose bindings use different Agent Namespaces do not share
  one undifferentiated notebook. If a Workflow deliberately assigns the same
  Namespace to two logical bindings, `worker_name` still enforces each
  binding's selected operation subset, while both names intentionally resolve
  to the same notebook;
- a later Stage that deliberately reuses the same Agent Namespace in the same
  Run sees its current notes;
- retries and Scheduler escalation create new StageExecutions but retain notes
  because they remain in the same RunScope and Agent Namespace; the new
  Planner/Worker can access them only through its newly selected operation
  subset;
- another WorkflowRun starts with no notes, even for the same user and
  Workflow.

Memory therefore survives allocation release and Planner-session replacement,
but expires under the owning Run's Artifact retention. There is no automatic
UserScope or cross-Run memory in `v1`.

Memory is a mutable coordination view and is not part of the immutable
StageContext snapshot. A Workflow cannot name the underlying reserved binding
directly in a StageContext declaration: that would expose the exact revision
and turn the notebook into implicit prompt state. Notes are never automatically
Stage results or Workflow outputs. If note data must become pinned context or a
product result, a selected domain or generic artifact tool first creates a
separate non-`memory.*` ordinary artifact. Workflow resolution plus Planner,
Worker and Scheduler candidate validation reject a direct reserved Memory ref
even when the model happens to know an exact revision.

## Artifact representation

One note maps to one logical artifact:

```text
ArtifactRef(namespace=<resolved Agent Namespace>, name="memory.<note name>")
media type: application/vnd.contractor.memory-note+json
```

For example, logical note `repo_overview` for logical Worker `analyst` whose
resolved Namespace is `source_analysis` maps internally to:

```text
ArtifactRef(namespace="source_analysis", name="memory.repo_overview")
```

The `memory.` artifact-name prefix is reserved at the model-visible Toolset
layer in every non-reserved RunScope Namespace. `run-artifacts@1` and every
other model-visible artifact-backed Toolset reject a source or target that
resolves to such a binding and filter it from any accumulated exact-ref
projection. This includes bounded text, source-analysis, OpenAPI and LikeC4
tools; selecting a domain tool must not become an alternate route around
MemoryTools. The lower-level allocation-bound Artifact API remains
domain-neutral and permits the trusted MemoryTools wrapper to use those
bindings; authenticated Run owners and operator diagnostics may also inspect
the underlying artifacts. This keeps revisions out of model context without
adding another network service or Artifact authorization mode. An ordinary
artifact named `memory.*` in a purpose-reserved `inputs`, `outputs` or `skills`
Namespace remains governed by that Namespace's own contract and is not a
Memory note.

MemoryTools encodes the note body as RFC 8785 canonical JSON. The stored fields
are:

```json
{
  "schemaVersion": "contractor.memory-note/v1",
  "name": "repo_overview",
  "content": "The API is assembled in internal/httpapi.",
  "description": "Entry points found while mapping the repository.",
  "tags": ["architecture", "repository"],
  "ordinal": 0
}
```

`name` is validated and `ordinal` is assigned by the trusted wrapper; a model
never supplies either storage field. Tags are deduplicated and stored in lexical
order, so equivalent tag input produces the same canonical payload.

Timestamps come from ArtifactStore metadata rather than caller bytes:
`created_at` is the logical binding creation time and `updated_at` is the
current binding revision's creation time. Both use the authoritative Server
clock. The existing Artifact read/write client contract exposes those two
metadata values to trusted wrappers without changing ArtifactRef or exposing
them as model arguments. They are projected into MemoryNote responses but are
not duplicated inside every note version.

The complete canonical JSON payload, including schema/name/ordinal fields, is
at most 32 KiB. Note names are at most 128 ASCII bytes and match
`^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`. Description is at most 512 UTF-8 bytes.
There are zero to three unique tags; each tag is at most 64 ASCII bytes and
matches `^[a-z][a-z0-9_-]*$`. Content and append fragments are non-empty UTF-8.
MemoryTools rejects a write whose resulting canonical artifact exceeds the
limit.

## Model-visible data

The full note projection contains exactly:

```python
class MemoryNote:
    name: str
    content: str
    description: str
    tags: list[str]
    ordinal: int
    created_at: datetime
    updated_at: datetime
```

The preview projection contains the same fields except `content`. List and
search return previews; only `read_memory` and successful mutation responses
return the full note. ArtifactRef, Artifact revision, physical blob identity,
and scope IDs are never model-visible.

`ordinal` is a monotonically increasing unsigned 64-bit creation order within
one Memory Namespace; its first note receives `0`. Its canonical JSON value is
also restricted to the exactly interoperable integer range
`0..9007199254740991` (`2^53-1`): RFC 8785 uses ECMAScript number semantics, so
larger integers could otherwise be rounded differently across languages. The
`v1` quota and absence of deletion keep all normally assigned values in
`0..127`; the wider bound is a corruption/interoperability guard. Replacing or
appending a note preserves `ordinal` and `created_at` and advances `updated_at`
using the Server clock. Revision is storage concurrency state and has no
ordering or semantic meaning for the model.

`list_memories` and `search_memory` sort previews by
`updated_at DESC, ordinal DESC, name ASC`. This deliberately puts recently
changed notes first while retaining deterministic creation order and a total
order even for equal Server timestamps.
`list_memory_tags` returns unique tags in lexical order.

## `memory-tools@1`

The first version exports exactly these six tools:

```text
list_memories()
read_memory(name)
write_memory(name, content, description = "", tags = [])
append_memory(name, content)
search_memory(tags)
list_memory_tags()
```

Their semantics are:

- `list_memories` returns every active note preview in the bound Memory
  Namespace;
- `read_memory` returns the current full note or `memory_not_found`;
- `write_memory` creates a note or fully replaces its content, description and
  tags. Omitted description/tags therefore become empty. Replacement preserves
  creation metadata;
- `append_memory` requires an existing note and appends one newline followed by
  the supplied content while preserving description, tags and creation
  metadata. The fragment itself is checked only for non-empty valid UTF-8; the
  32-KiB bound is applied to the resulting note, so validation does not depend
  on an unrelated placeholder note name;
- `search_memory` accepts one to three unique valid tags and returns previews
  for notes containing **any** supplied tag; it performs no body, fuzzy or
  embedding search;
- `list_memory_tags` returns the current distinct tag set.

The list is intentionally unpaginated because one Memory Namespace has at most
128 active notes and previews are bounded. Attempting to create a 129th note
returns `memory_namespace_full`. Existing notes remain readable, replaceable
and appendable at the limit. MemoryTools never evicts an older note to make
room.

The wrapper owns ordinals. Under the no-delete serialized `v1` contract, the
current Namespace therefore contains each ordinal in `0..count-1` exactly
once. A duplicate, gap or otherwise impossible current ordinal set is corrupt
stored state and returns `memory_unavailable`; it is never repaired from model
input.

`v1` has no delete/forget operation, automatic compaction, `link_memories`,
inbox category, semantic retrieval or bulk import. Agent Skills are the
separate immutable package/artifact contract in [09](09-agent-skills.md), never
a Memory note category. Removing or changing the remaining choices requires a
new Toolset version, not an undocumented tool.

## AgentTemplate selection and Planner mirroring

Memory tools are never implicit. An AgentTemplate selects the exact Toolset
version and exact operation subset under the ordinary selection contract:

```yaml
spec:
  toolsets:
    - ref: memory-tools@1
      tools:
        - list_memories
        - read_memory
        - write_memory
        - append_memory
        - search_memory
        - list_memory_tags
```

The Runtime Agent must advertise the exact selected operations in its positive
Toolset capability snapshot. `memory-tools@1` consumes only the
allocation-bound Contractor Artifact client; it declares no LLM, HTTP-proxy,
subprocess or sandbox capability by itself.

For a model-backed Planner, Server derives its MemoryTools interface from the
prepared logical Workers rather than from a separate Planner template:

- Streamline receives exactly the selected MemoryTools subset of its sole
  Worker, with the Worker signatures shown above;
- Router exposes an operation when at least one prepared logical Worker selects
  it. Every exposed operation gains one required `worker_name` argument, and
  that argument's enum contains exactly the Stage binding names whose
  AgentTemplate selected that operation;
- an operation selected by no Worker is absent, not a function that always
  returns forbidden;
- Planner receives no generic `run-artifacts@1` operation or unrelated domain
  Toolset merely because a Worker selected it.

For example, if `builder` selects all six operations and `reviewer` selects
only list/read/search, Router receives:

```text
write_memory(worker_name: Literal["builder"], ...)
read_memory(worker_name: Literal["builder", "reviewer"], name)
search_memory(worker_name: Literal["builder", "reviewer"], tags)
```

The Planner adapter resolves `worker_name` to the immutable Stage binding and
its Agent Namespace. It never asks the model for a Namespace or physical
Runtime Agent. `passthrough@1` has no model/tool loop, so only its Worker gets
the selected tools; the deterministic Planner does not synthesize a second
MemoryTools surface.

Worker memory calls consume the Worker's ordinary `maxToolCalls` budget.
Planner memory calls consume ordinary Planner function-call accounting and a
model turn under its finite `maxModelCalls`; they do not create an unmetered
side channel.

## Artifact-backed authority, concurrency and response loss

MemoryTools is a thin semantic wrapper, not a Server service, private endpoint
or second database. The Server-side Planner adapter uses a Run-scoped
ArtifactStore view directly. The Python Worker adapter uses the existing
allocation-bound private Artifact API and client. Both implementations share
canonical fixtures for validation, payload encoding and logical results.

The Planner view is also bound to the active `StageExecution`, not merely to
`run_id`. A Planner mutation locks and checks the durable Run and Stage rows in
the same Artifact transaction that advances the binding. Worker mutations keep
using the private Artifact API's allocation-grant write lock. Scheduler forms
one ordered terminal barrier from those existing mechanisms: it first fences
every allocation and waits for any grant-locked mutation to finish, then enters
the durable `finalizing` or `aborting` transition; a Planner transaction in turn
serializes directly with that transition. An in-flight mutation therefore
either linearizes completely before its corresponding barrier or fails as
`memory_forbidden`; it can never commit afterward. Planner reads after its
invocation is cancelled are rejected by its bound tool context.

Every mutation uses ArtifactStore compare-and-swap internally:

1. the wrapper reads the current hidden revision, if any;
2. it constructs and validates the complete canonical next payload;
3. it creates with an absent-binding precondition or updates with that exact
   hidden revision;
4. a concurrent same-binding change returns `memory_changed`; MemoryTools
   never silently overwrites or semantically merges it.

The model does not resolve the conflict with a revision. It calls a read/list
operation and decides whether to retry the semantic mutation.

The existing Artifact CAS response-loss rule is also sufficient for append.
After reading revision `r`, the wrapper computes one canonical next payload and
retries only those same bytes with the same `If-Match: r`; it never re-reads and
blindly appends the fragment again. If the retry conflicts, it reads current:
an exact canonical-payload match means the first write committed, otherwise the
wrapper returns `memory_changed`. Thus response loss cannot duplicate an
append and no Memory-specific idempotency record is required.

The current execution model runs one Planner function at a time. While its
Worker dispatch function is waiting, Planner starts no other call; the selected
Worker processes its task and tool calls sequentially, and the Run has no
concurrent StageExecutions. MemoryTools also serializes calls inside one Worker
allocation. Under those normative `v1` constraints, a create lists the current
`memory.` bindings, rejects count 128 and assigns `max(ordinal) + 1` (or `0`
when empty) before its CAS create. Old revisions do not count. A future design
that permits concurrent different-name creates in one Memory Namespace must
first add an ArtifactStore-level atomic multi-binding/batch-CAS primitive; it
must not silently weaken the 128-note or unique-ordinal contract or introduce a
Memory-only transport.

Reads and lists resolve current bindings at call time and are linearizable per
binding/query transaction; they are not retroactively added to StageContext.
Old immutable note revisions remain ordinary Run artifacts under ArtifactStore
retention but do not count as active notes.

Worker mutations travel through the existing private Artifact PUT, so Agent
mTLS, allocation-to-Run binding, CAS and the grant-locked Artifact write fence
are reused unchanged. Runtime supplies the already fixed Agent Namespace; the
model supplies only logical MemoryTools arguments. Planner calls are synchronous
inside the active Stage invocation and use the Stage-bound trusted view above;
Planner produces no candidate until its current tool call has returned.
Finalization/abort performs no tool work, and retry or escalation cannot revive
an old wrapper.

## Errors and telemetry

The model-facing stable error codes are:

| Code | Meaning | Retryable without changing input |
|---|---|---|
| `memory_invalid` | Invalid name, tags, UTF-8 or request shape | no |
| `memory_not_found` | Requested current note does not exist | no |
| `memory_too_large` | Resulting canonical note exceeds 32 KiB | no |
| `memory_namespace_full` | A create would exceed 128 active notes | no |
| `memory_changed` | Hidden Artifact CAS conflict or ambiguous newer value | yes, after re-read |
| `memory_forbidden` | Tool/Worker/Stage capability or write fence rejects it | no |
| `memory_unavailable` | Bounded infrastructure failure | yes |

Messages are bounded and do not include note content or internal revisions.
Malformed model arguments, including extra fields or a value of the wrong JSON
type, are reduced by both Planner and Worker MemoryTools adapters to
`memory_invalid` before an Artifact side effect. An absent, unknown or
operation-ineligible Router `worker_name` is instead `memory_forbidden` and
takes precedence over unrelated malformed note fields. Framework JSON-Schema
validation must not become a second unbounded model-facing error vocabulary.
Adapters normalize every internal failure to the exact code/retryability table
above; an arbitrary exception attribute cannot add another Memory code.

Memory adapters never automatically copy content, description or tags into
ExecutionReport detail, Planner durable facts, logs, WebSocket events or
external telemetry. Safe tool diagnostics may retain only operation name,
logical `worker_name` when applicable, note name, request/result byte sizes,
success/error code and duration. Tool counters remain ordinary aggregate
metrics. An exact hidden Artifact revision may be retained as trusted Server
provenance, but it is never placed in model context or a public Planner event.

The selected LLM Gateway is an intentional content channel: a full note returned
by `read_memory` or a mutation is part of that invocation's model conversation.
The model can also deliberately repeat note data into a subtask objective,
instructions, a semantic Stage summary or a newly authored ordinary artifact.
That copy is governed by the destination's normal contract and retention; `v1`
does not claim information-flow tracking or semantic redaction of model output.
Planner and Worker instructions must frame notes as untrusted coordination data
that cannot replace the immutable objective or system instructions. The
retained-surface guarantee is therefore precise: Contractor's Memory adapters
do not create an additional diagnostic copy of the note payload.

## Invariants

1. One note is one RunScope artifact; MemoryTools is a thin ArtifactStore
   wrapper, not another service, endpoint or content store.
2. Memory Namespace is exactly `(run_id, resolved Agent Namespace)` and is
   never selected directly by a model.
3. Planner and its corresponding logical Worker see the same current notes;
   Router selects a Worker view through a schema-constrained logical name.
4. `memory-tools@1` and every operation are explicitly selected. Planner
   mirroring never widens a Worker's allowlist.
5. No model-visible Artifact-backed Toolset can list, read or mutate reserved
   `memory.` bindings or retain their exact refs.
6. Artifact revision is internal concurrency state and never model-visible.
7. Under the serialized `v1` execution contract, creation count, ordinal
   assignment and the 128-note limit are deterministic; notes are never evicted
   automatically. Future same-Namespace create concurrency requires a generic
   atomic ArtifactStore primitive first.
8. Mutation uses hidden Artifact CAS and canonical-payload reconciliation;
   response loss cannot duplicate an append and needs no separate replay store.
9. Worker mutations linearize against the allocation grant fence; Planner
   mutations linearize against the durable Stage transition. Scheduler orders
   every allocation fence before that transition, so both participate in one
   terminal barrier. A Planner memory call also completes before Planner can
   return a candidate and enter finalizing.
10. Memory adapters never automatically retain note bodies or descriptive
    metadata in durable diagnostics or external telemetry. The LLM invocation
    and an explicit model-authored semantic copy remain intentional content
    destinations, not telemetry.
11. Memory is Run-scoped and observed only through explicit tools; there is no
    implicit prompt or cross-Run injection.
