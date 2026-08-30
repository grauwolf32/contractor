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
```

Published YAML is the source of truth. Server has two configured roots that
form one logical configuration namespace:

```text
/etc/contractor/configs/       operator/bootstrap root, normally read-only
/var/lib/contractor/configs/   Server-managed root for UI publications
```

The paths are deployment defaults, not hard-coded identities. Local development
may point both flags at workspace-specific directories. A duplicate
`kind + metadata.name + metadata.version` across either root invalidates the
complete set; there is no root precedence and no implicit override. The first
UI increment publishes only ModelPolicy and LLMGatewayConfig manifests. Existing
Workflow, AgentTemplate and instruction resources remain operator-authored and
read-only until their editors receive a separate contract.

UI publication is create-only. It cannot edit, replace, disable or delete an
existing published identity. Retiring configurations and safe garbage
collection remain a later operation; an implementation must not approximate
retirement by silently hiding a file still referenced by a Workflow or Run.
Drafts are not executable configuration and may be stored separately without
becoming a source of truth.

Publication atomically adds a complete validated version to the Server's
current configuration set; readers observe either the set before publication
or the set including the new version, never partial dependency resolution. It
does not mutate an existing version and does not require re-resolving existing
Runs. The current startup-only file loader remains a valid bootstrap/import
path but is not by itself sufficient for writable Operations UI behavior.

For one publication Server:

1. acquires the single configuration-publication lock;
2. normalizes the document and validates the complete union of operator,
   managed and candidate manifests, including refs, consumer compatibility and
   digests;
3. writes canonical YAML to a new same-directory temporary file in the managed
   kind subtree, flushes it, atomically renames it to its stable path and
   flushes the parent directory;
4. atomically swaps the in-memory configuration snapshot;
5. records non-authoritative audit metadata.

No successful response is returned before the file and in-memory snapshot are
both published. A crash after rename but before snapshot swap is recovered by
the ordinary startup load. Publication requires an Idempotency-Key: retrying an
identical identity/body returns that published version, while the same identity
with different normalized content conflicts. PostgreSQL audit loss cannot
remove or reinterpret the YAML version.

The client never supplies a filesystem path. Server derives a stable relative
path from validated kind/name/version, creates parent directories inside the
dedicated managed root, and rejects symlinks or any resolved escape. Temporary
and final creation use exclusive/no-follow semantics; an existing destination
is handled only by the idempotency comparison above.

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

Credential metadata and encrypted revisions live in PostgreSQL. Published YAML
contains only the stable `credentialRef`; PostgreSQL is authoritative for the
secret value and its revision lifecycle, never for ModelPolicy or
LLMGatewayConfig bodies. The logical schema is:

```text
llm_credentials
  credential_id
  label
  current_revision
  created_at

llm_credential_revisions
  credential_id
  revision
  key_id
  nonce
  ciphertext
  created_at
  lifecycle_state
```

`credential_id` uses the shared configuration-ID grammar. Revisions are
positive monotonically increasing integers scoped to that ID. One revision
accepts a token of 1 through 16,384 UTF-8 bytes through a write-only request;
Server performs no trimming. It encrypts the exact bytes with AES-256-GCM using
a fresh 96-bit cryptographically random nonce. Canonical schema version,
credential ID and revision are authenticated additional data, so ciphertext
cannot be moved to another identity or revision. Credential creation/rotation
and advancing `current_revision` are one PostgreSQL transaction. Plaintext is
retained only for the bounded encryption/decryption operation and the
model-client settings that actively need it.

The 256-bit master key is a bootstrap secret outside PostgreSQL. Server receives
only an absolute `--credential-master-key-file` path. The file contains RFC
4648 base64 for exactly 32 bytes, with at most one trailing newline; command-line
literal and environment-variable key values are forbidden. Server rejects a
missing, malformed, non-regular, symlinked, group-readable or world-readable
key file and reads it once during startup. The database stores a non-secret
`key_id` formatted as the lowercase `sha256:<hex>` digest of the decoded key
bytes with each ciphertext, so a future keyring migration does not require a
schema change. The fingerprint is not secret because the key has 256 bits of
random entropy. The first slice accepts one active key and does not implement
master-key rotation.

When any published Gateway or retained non-terminal Run references a credential,
Server startup requires the credential store and key to be available. A
decryption/authentication failure is a bounded internal configuration error and
never falls back to plaintext, another revision or an unauthenticated request.
Run initialization pins the then-current revision, while Planner construction
and Control Plane allocation preparation decrypt exactly that pinned revision
just in time. The token is never copied into WorkflowRun, StageExecution,
Planner Session, audit rows or metrics.

Encrypting the database protects a PostgreSQL dump without the master-key file;
it does not protect against compromise of the running Server process or host.
Database and master-key backups must be protected and stored separately. A
future Vault/KMS adapter may implement the same revision interface without
changing LLMGatewayConfig or executionConfig.

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

- credential disable/revoke behavior for new Runs, pinned non-terminal Runs and
  already active allocations;
- master-key rotation/re-encryption and a future Vault/KMS adapter;
- embedded same-origin UI versus separately deployed frontend;
- browser authentication and the first user/operations permission split;
- polling, Server-Sent Events or WebSocket updates for Run and Agent state;
- exact list/filter/pagination contracts and retention window;
- whether the initial editor covers Workflow and AgentTemplate or only
  ModelPolicy, LLMGatewayConfig and credentials;
- artifact preview size limits and whether OpenAPI/LikeC4 receive specialized
  renderers in the first UI increment.
