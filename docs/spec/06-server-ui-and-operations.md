# 06 — Server UI and Operations

Status: **Working agreement; being refined in dialogue**

Depends on: [00](00-workflow-and-planner.md),
[01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[03](03-artifact-plane.md), [04](04-execution-lifecycle-and-metrics.md)

## Purpose

Contractor Server exposes one Web UI for ordinary Workflow use and for
single-VM operational visibility. The UI is an API client; it does not become a
second Scheduler, configuration resolver or source of lifecycle truth.

The initial navigation contains:

```text
Workflows
Artifacts
Runs
  -> Run details
     -> Stages / attempts
     -> Inputs / outputs
     -> Metrics

Operations
  -> Runtime Agents
  -> Allocations
  -> LLM configurations
  -> Credentials
```

The UI remains domain-neutral. OpenAPI, LikeC4, code analysis and security are
Workflow/configuration examples rather than hard-coded application modes.

## Workflow user surface

The first useful user surface supports:

- listing published Workflow names and exact versions;
- rendering Run string parameters and input Artifact slots from the selected
  Workflow contract;
- uploading a UserScope Artifact or selecting an existing exact version;
- selecting published execution configurations when an override is desired;
- creating an idempotent WorkflowRun and cancelling a non-terminal Run;
- listing Runs and showing their durable lifecycle;
- showing ordered Stages, attempts, retry/termination reasons and the currently
  active StageExecution;
- showing exact input, intermediate, frozen output and lineage metadata;
- previewing bounded UTF-8 Markdown, YAML, JSON, OpenAPI and LikeC4 artifacts,
  while retaining an explicit download action for original bytes;
- showing bounded Planner, Worker and Runtime metrics without exposing prompts,
  tool payloads, provider bodies or secrets.

UI labels must preserve the execution vocabulary: a Runtime Agent is the
long-running single-slot process, while Worker is its temporary
allocation-scoped role. The UI must not present them as two independently
deployed services.

## Operations surface

Operations is part of the first UI increment rather than a later product. It is
read-only for execution state and shows at least:

- Runtime Agent instance ID, software version, observed state, last accepted
  heartbeat, confirmed lease horizon and current allocation ID;
- `idle`, `reserved`, `busy`, `draining` and `fenced` slot state using the
  authoritative/observed reconciliation vocabulary from [02];
- allocation ID, WorkflowRun, StageExecution, logical Worker binding,
  AgentTemplate, effective ModelPolicy and LLMGatewayConfig refs;
- allocation prepare/finalize/abort/release state and bounded failure reason;
- safe aggregate call/tool/token/error counters and whether a budget dimension
  was exhausted.

The page is an observation surface. It cannot force a Runtime Agent to idle,
reassign an allocation, mark a Stage successful or bypass bounded abort/release.
Explicit administrative recovery operations, if later required, need separate
idempotent Server commands and audit contracts.

## Published LLM configuration

Run creation never accepts free-form model names, budgets, Gateway URLs or
tokens. Operations manages and Run forms select already validated published
configurations:

- one shared `ModelPolicy` kind is usable by Planner and Worker consumers;
- role-specific fields are omitted when not used, while each consumer rejects
  a policy missing one of its required finite limits;
- `LLMGatewayConfig` versions carry protocol, URL and an optional logical
  credential reference;
- secret values live behind `LLMCredentialRef`, never in Workflow,
  AgentTemplate, ModelPolicy, LLMGatewayConfig or Run JSON;
- an immutable published version is never edited in place.

The Operations authoring flow is conceptually:

```text
published configuration
  -> clone as draft
  -> edit typed fields
  -> validate references and consumer compatibility
  -> publish new immutable name@version + digest
  -> optionally disable it for selection by future Runs
```

Disabling a configuration removes it from new-selection lists but does not
invalidate a Run that already pinned it. Deletion while any retained Workflow
or Run snapshot references the version is forbidden. Whether drafts and
published YAML are persisted in a managed filesystem root or a PostgreSQL
configuration registry remains an explicit open decision below; both must
produce the same normalized document and digest.

Publication atomically adds a complete validated version to the Server's
current configuration set; readers observe either the set before publication
or the set including the new version, never partial dependency resolution. It
does not mutate an existing version and does not require re-resolving existing
Runs. The current startup-only file loader remains a valid bootstrap/import
path but is not by itself sufficient for writable Operations UI behavior.

## Credential handling

The UI may create a credential and rotate its secret value, but it receives the
value only in the write request. Read responses contain only:

- stable credential ID;
- non-secret revision ID;
- created/rotated timestamps;
- enabled/revoked state;
- optional safe label and consumption aggregates.

The Server never returns a stored secret, even to the principal that created
it. UI forms do not preserve it in browser storage or logs. Rotation creates a
new revision; Run initialization pins the non-secret revision selected through
its exact LLMGatewayConfig. Active allocation RuntimeSettings remain unchanged.

Attribution records ModelPolicy, LLMGatewayConfig and credential refs together
with WorkflowRun, StageExecution and Planner/Worker role. Provider-reported
tokens and calls are useful operational measurements, not billing-grade proof.
Future quotas may reject new Run/Stage work by credential or policy without
changing the executionConfig shape.

## ExecutionConfig

Workflow YAML carries default execution selections. The Run form may submit
only exact published refs in the reference-only `executionConfig` override
defined by [00](00-workflow-and-planner.md). It may choose Planner and Worker
policies independently and may target a particular Stage/Agent binding.

For example, the UI may leave discovery on a local Worker policy while choosing
a stronger published policy and a separately metered credential for the
OpenAPI builder. A `streamline@1` Stage may independently use a stronger Planner
policy. `passthrough@1` exposes no Planner model selection because it makes no
Planner LLM call.

Before returning a successful Run-create response, Server expands defaults and
overrides into a complete immutable per-consumer ResolvedExecutionConfig. The
Run detail page shows both the selected refs and their origin (`workflow` or
`run override`), but never reconstructs authority from browser state.

## API boundary

The UI uses authenticated public Server APIs. HTML routes do not receive direct
database, Control Plane or Runtime Agent access. UI-oriented query endpoints
may aggregate existing durable/read-model state, but mutations must call the
same domain application services as non-UI API clients.

Run and Artifact mutations retain their existing idempotency, ownership and CAS
requirements. Configuration publication, credential rotation and future
administrative commands require their own idempotency keys and audit actor.

The exact browser authentication mechanism, authorization roles, pagination,
filter grammar and live-update transport are not selected yet. The current
single configured user may initially own both user and Operations surfaces, but
the API keeps configuration/credential mutations distinct so later RBAC does
not require changing Run semantics.

## Invariants

1. UI can request domain operations but cannot manufacture execution state.
2. Run overrides contain exact published refs only; no inline model, budget,
   URL, token or provider parameter is accepted.
3. The immutable ResolvedExecutionConfig is authoritative after Run creation.
4. Planner and every logical Worker resolve their model policy and Gateway
   independently.
5. Published configuration versions and their digests never change in place.
6. Tokens are write-only secrets and never appear in durable execution state,
   API reads, metrics, logs or browser persistence.
7. Operations reflects both observed Runtime state and authoritative Control
   Plane state without conflating them.
8. Existing API clients can perform the same domain operations without using
   the Web UI.

## Open decisions for the next dialogue steps

- managed YAML directory versus PostgreSQL Config Registry as the writable
  source of truth for UI-authored drafts and published configuration;
- embedded same-origin UI versus separately deployed frontend;
- browser authentication and the first user/operations permission split;
- polling, Server-Sent Events or WebSocket updates for Run and Agent state;
- exact list/filter/pagination contracts and retention window;
- whether the initial editor covers Workflow and AgentTemplate or only
  ModelPolicy, LLMGatewayConfig and credentials;
- artifact preview size limits and whether OpenAPI/LikeC4 receive specialized
  renderers in the first UI increment.
