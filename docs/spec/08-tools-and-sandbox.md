# 08 — Tools and sandbox

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [07](07-agent-runtime-and-a2a.md)

## Tool boundary

For the ADK-backed Worker strategy, ADK provides tool declaration, schema
generation, dispatch, `ToolContext`, callbacks and tracing. Contractor does not
build a second generic agent framework. It defines typed tool, artifact, policy
and sandbox ports. Every framework dispatcher, including ADK and non-ADK
bindings, adapts into one Attempt-scoped Contractor `ToolInvoker`; a strategy
never receives the underlying tool implementation or an unscoped dispatcher.

`ToolInvoker` validates the exact versioned contract and effective `TaskSpec`
policy on every call, applies cancellation/deadline and artifact limits, records
usage and invokes the registered implementation. Prompt/tool declaration
filtering is discovery only; authorization is enforced again at invocation.

Tool inputs and small results are JSON-compatible DTOs. Large results are
streamed through an `ArtifactWriter` into PostgreSQL chunks and returned as
`ArtifactRef`. A configured absolute artifact limit fails with a stable
`artifact_too_large` error without buffering the entire value.

## Sandbox lifecycle

Every sandbox is owned by exactly one `SandboxLease` containing `attempt_id`,
workspace identity, creation time, expiry and fencing token.

```text
acquire(AttemptId, policy) → SandboxLease
execute/read/write through lease
cancel or release exact lease
```

Global `teardown_all()` is forbidden in normal run cleanup.

## Attempt-scoped workspace views

A Worker strategy may create private child workspace views to explore alternate
changes without copying them into Server RunState. The versioned local contract
is:

```text
fork(immutable_base_ref, parent_view?, policy_version) -> WorkspaceView
diff(view) -> immutable PatchArtifact
apply(view, PatchArtifact) -> WorkspaceView
merge(base_ref, ordered PatchArtifact refs, conflict_policy_version)
  -> merged view | typed conflicts
```

Every view is owned by the enclosing Attempt and references one immutable base
workspace/project-snapshot version. Patches are immutable, hash-verified
artifacts; merge order and conflict policy are versioned and deterministic.
Private views, branches and conflicts are Agent-local execution state, never
Server Tasks or Attempts. Only exact final staged outputs named by
`WorkerResult` are eligible for Server promotion. Cancellation/deadline releases
or quarantines all child views and processes; committed patch artifacts remain
in inaccessible Attempt staging and follow normal retention rather than becoming
run-visible.

## External side effects and retry

Fencing protects result acceptance and artifact promotion; it cannot undo an
email, payment, deployment or other external effect. Every tool contract
therefore declares a versioned effect class:

- `read_only` — no external mutation;
- `idempotent` — replay with the same operation key is contractually one effect;
- `reconcilable` — an exact provider operation reference supports status lookup
  before retry;
- `non_idempotent` — dispatch may create another effect and has no reliable
  reconciliation protocol.

Every effecting call carries a stable `tool_invocation_id`/idempotency key for
one logical effect. In one short transaction, `ToolInvoker` locks/checks the
Server-owned Attempt `operation_start_gate` and records intent in the Attempt's
durable mapping before provider I/O. A non-open gate rejects the call before any
external bytes are sent. The existing record is updated after provider I/O only
through monotonic outcome/reconciliation transitions. A tool-specific retry
protocol must state whether the key is stable across WorkerJob retries for the
same Task, how an ambiguous result is reconciled, and which terminal evidence
permits another call. Once a non-idempotent request may have left the process,
or a reconcilable request remains unknown, the Worker reports stable
`external_effect_unconfirmed`; Server must not automatically retry the Task.
Cancellation stops further calls but never claims to roll back an already
dispatched external effect.

Agent loss is itself ambiguous. If the `TaskSpec` permits a non-idempotent tool,
Server assumes it may have been invoked unless durable effect evidence proves
otherwise. A failed, lost or cancelled Attempt is not automatically retried
merely because its result was fenced; retry additionally requires the allowed
tool contracts and their durable effect records to prove that another execution
is safe. For cross-Attempt idempotent replay, the logical effect key is derived
from stable Task/effect identity rather than `attempt_id`.

For retry reconciliation, Server first seals or revokes the old Attempt's
operation-start gate in the same RunState/Attempt Unit of Work used to authorize
overlap. Intent creation serializes against this transition. Server then reads
one exact version/hash of the sealed journal through a read-only repository and
records the decision in RunState/audit. A pre-closure intent is visible and must
be reconciled; a post-closure intent cannot commit or reach provider I/O.
Existing calls may still append monotonic outcome observations during the
bounded evidence window, so absent, changing or conflicting evidence remains
`external_effect_unconfirmed`. Server cannot update an Agent-owned outcome, and
a snapshot hash without gate closure is never sufficient retry proof.

## Live-target safety

Network, exploit-oriented and effecting tools are denied by default. Enabling
one requires an exact authorized policy referenced by `TaskSpec` that pins the
target identity/environment, allowed operations, outbound destinations and
protocols, credential scope and effective time/deadline. Replay services and
disposable targets are distinct from live targets; authorization for one does
not imply authorization for another.

`ToolInvoker` validates target and operation scope before dispatch, while the
sandbox/network adapter independently enforces egress and time scope before
DNS, socket or subprocess I/O. Redirects, resolved addresses and provider
references are rechecked so they cannot escape the authorized target. Worker
prompts, model output and Agent-private planning cannot add targets, operations,
egress or time. Every allow/deny decision for these tools is appended to the
security audit; an outside-scope request fails before external I/O.

## Requirements

- **TLS-001** — Tool names, argument schemas, result schemas and required
  capabilities MUST be versioned and registered explicitly.
- **TLS-002** — Worker-visible declarations, including prompts where applicable,
  MUST expose only tools allowed by the selected capability provider and exact
  `TaskSpec` policy. `ToolInvoker` MUST independently enforce that policy on
  every invocation.
- **TLS-003** — Filesystem tools MUST resolve paths inside the leased workspace
  and reject traversal, symlink escape and host absolute paths.
- **TLS-004** — Sandbox processes MUST have configurable CPU, memory, process,
  disk, wall-time and network limits.
- **TLS-005** — Cancellation or timeout MUST terminate the full subprocess tree,
  not only the immediate shell process.
- **TLS-006** — Cleanup MUST be idempotent and scoped to one lease. Cleaning one
  Attempt MUST NOT affect another concurrent Attempt.
- **TLS-007** — Secrets MUST be injected only into tools that declare them and
  MUST be redacted from results, artifacts and telemetry.
- **TLS-008** — Every tool contract MUST declare its versioned effect class and
  retry/reconciliation behavior. Every effecting invocation requires a stable
  logical operation key; framework-native call IDs alone are insufficient.
- **TLS-009** — Sandbox output streaming MUST be bounded; backpressure cannot
  grow Agent memory without limit.
- **TLS-010** — ADK code executors or Environment APIs MAY back the port only
  after capability tests prove the same isolation and cancellation contract.
- **TLS-011** — Tool output above the inline threshold MUST stream through the
  chunked ArtifactStore API with bounded memory; output above the absolute
  artifact limit MUST fail deterministically.
- **TLS-012** — Every Worker strategy, whether ADK-backed or custom, MUST obey
  the same versioned tool schemas, policy checks, artifact limits and sandbox
  lease contract when it invokes Contractor tools.
- **TLS-013** — Agent-internal planning MUST NOT widen the enclosing
  `WorkerJob`'s tool, secret, artifact, filesystem, network or sandbox grants.
  Every internal action remains attributable to the boundary Attempt.
- **TLS-014** — Contractor tool implementations MUST be reachable by Worker
  strategies only through the scoped `ToolInvoker`. A custom framework adapter
  MUST NOT bypass policy, cancellation, usage, audit or effect-state recording.
- **TLS-015** — `ToolInvoker` MUST durably record an effecting invocation before
  provider I/O in a transaction that serializes with the Server-owned
  operation-start gate, then persist confirmed, failed or unknown outcome plus
  any opaque provider operation reference. Records MUST be scoped to the
  Attempt and stable logical tool invocation.
- **TLS-016** — Automatic retry after ambiguous external I/O is allowed only
  when a versioned tool protocol proves replay idempotent with the same key or
  reconciles the exact provider operation to a safe state. A non-idempotent or
  still-unconfirmed effect MUST produce `external_effect_unconfirmed` and block
  automatic Task retry. Agent loss for a TaskSpec that permits such an effect
  MUST be treated as unconfirmed unless durable evidence proves it was not
  dispatched or is safe to replay.
- **TLS-017** — Cancellation and deadline MUST reject every later invocation in
  the scoped `ToolInvoker`; they do not imply rollback of an already dispatched
  external effect.
- **TLS-018** — Every workspace child view MUST be Attempt-scoped and derived
  from an immutable, hash-verified base reference. A view MUST NOT mutate its
  base or another concurrent Attempt's view.
- **TLS-019** — Workspace diffs and merge inputs MUST be immutable versioned
  patch artifacts. Merge order, path normalization and conflict policy MUST be
  deterministic and identified by policy version; unresolved conflicts MUST be
  returned as typed data rather than silently selecting a side.
- **TLS-020** — Agent-private workspace forks, patches and merges MUST NOT create
  Server Task/Attempt records or become authoritative progress. Only final exact
  output refs returned by the boundary WorkerResult may be promoted.
- **TLS-021** — Cancellation, deadline and restart reconciliation MUST release
  or quarantine every recorded child view and subprocess. Any committed patch
  artifacts remain staged/inaccessible and are collected by Attempt retention.
- **TLS-022** — Network, exploit and effecting tools MUST default deny. Their
  exact target/environment, operation, egress, credential and time scope MUST be
  authorized by immutable policy refs in `TaskSpec`; framework or model content
  MUST NOT widen that scope.
- **TLS-023** — `ToolInvoker` MUST reject an outside-scope target or operation
  before provider/network I/O. The sandbox/network adapter MUST independently
  enforce destination/protocol/port and deadline scope, including redirect and
  resolved-address checks.
- **TLS-024** — Every live-target allow/deny decision MUST append an authorized
  audit record with Task/Attempt, policy revision, requested target/operation and
  outcome, without leaking credentials or sensitive payload content.
- **TLS-025** — Automated evaluation or retry MUST use a replay service or
  disposable authorized target unless the exact live operation is explicitly
  approved and its effect protocol proves repetition safe.
- **TLS-026** — Before automatic retry or overlapping replacement of an
  effect-capable Attempt, Server MUST seal or revoke its operation-start gate
  and read the resulting final evidence snapshot. After that transition, Agent
  may only finalize or reconcile a pre-existing record and MUST NOT initiate a
  new business/effecting operation. Exact-reference status/read/cancel I/O
  required by the pinned reconciliation protocol remains allowed and cannot
  change the original request identity. The gate check, intent insert and
  external-call ordering MUST make a late-intent race impossible.

## Acceptance

1. Traversal and symlink-escape tests cannot read a host fixture outside the
   workspace.
2. Cancelling a tool with grandchildren leaves no process or container running.
3. Concurrent Attempts use different workspaces; releasing one preserves the
   other.
4. A tool returning a large file publishes an artifact rather than embedding
   the file in A2A/session state.
5. Policy tests prove that an unlisted tool cannot be called by the Worker.
6. A large-output fixture above the inline threshold stays within the memory
   bound, while a fixture above the absolute limit returns
   `artifact_too_large` and leaves no readable partial artifact.
7. Equivalent ADK and non-ADK Worker strategies cannot call an unlisted tool or
   escape the same Attempt-scoped sandbox and artifact limits.
8. A custom framework attempting to call an implementation directly cannot
   obtain it; invocation through `ToolInvoker` rejects an unlisted contract even
   when the framework exposed it in its own prompt or registry.
9. Fault injection or Agent death after an external request is sent but before
    its response never creates an automatic duplicate: an idempotent tool reuses
    the exact cross-Attempt key, a reconcilable tool inspects the stored provider
    reference using only its declared status/read/cancel protocol, and a non-
    idempotent tool returns or is conservatively classified
    `external_effect_unconfirmed` and blocks retry.
10. Cancellation after a confirmed external effect prevents later tool calls,
    retains the audit/effect record and does not report that the external effect
    was rolled back.
11. Two child views from one immutable base make conflicting edits; the same
    ordered patches and policy version produce the same typed conflict on every
    run, while reversing an explicitly order-sensitive policy changes its hash.
12. A Worker creates several private forks and patches but Server observes one
    boundary Attempt and promotes only the exact final output refs returned in
    WorkerResult.
13. Cancellation during merge stops all child processes, releases every view,
    leaves committed patch artifacts staged/inaccessible and does not alter the
    immutable base or another Attempt's workspace.
14. With no live-target policy, network, exploit and effecting tools fail before
    DNS, socket, subprocess or provider I/O and append a redacted denial audit.
15. A prompt or private planning step substitutes a different host, operation,
    redirect or egress destination; both `ToolInvoker` and sandbox enforcement
    reject it and the fixture observes no outside-scope bytes.
16. An authorized replay/disposable target permits only its exact operations and
    egress until the TaskSpec deadline; the same request against a live target or
    after expiry fails before I/O.
17. Automated evaluation of an effecting workflow uses the replay fixture and
    never repeats the live external effect.
18. A concurrent loss/retry fixture pauses an old `ToolInvoker` before intent
    creation while Server seals or revokes the operation-start gate. If intent
    commits first, retry sees it and follows its protocol; if the gate transition
    commits first, the tool sends no provider request. A snapshot followed by a
    late external effect is impossible.
