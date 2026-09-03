# 06 — Server UI and Operations

Status: **Working agreement; being refined in dialogue**

Depends on: [00](00-workflow-and-planner.md),
[01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[03](03-artifact-plane.md), [04](04-execution-lifecycle-and-metrics.md) and
[09](09-agent-skills.md)

## Purpose

Contractor deployment includes one separately deployable Node.js Web UI for
ordinary Workflow use and single-VM operational visibility. The UI uses the
Server's public API; it does not become a second Scheduler, configuration
resolver or source of lifecycle truth.

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

## Deployment boundary

Web UI is its own Node.js build artifact and runtime service. Go Server embeds
and serves no HTML, JavaScript, CSS or other frontend asset. Rebuilding,
releasing, restarting or rolling back Web UI therefore does not rebuild or
restart Server, and Server startup/health never depends on UI availability.
Both services may still run on the same VM.

Their only product contract is the versioned authenticated public Server API.
The UI build records its own version and supported Server API versions; it must
show an explicit incompatible-version error rather than guessing around a
missing or changed field. Additive API changes within a supported version do
not require a coordinated release.

The delivered UI is a client-rendered frontend. Node serves its static assets
and a non-secret `/runtime-config.json`; it performs no SSR, BFF, API proxying,
session storage or domain operation. Runtime config contains exactly the UI
version, supported API versions and one absolute `apiBaseUrl`, allowing the same
UI build to target another Server without rebuilding. The URL has no userinfo,
query or fragment and uses HTTPS; HTTP is accepted only for a loopback origin
in local development. The browser calls that Go Server URL directly, including
for Artifact upload/download and Operations mutations. Node receives no Server
credential and has no database, Control Plane or Runtime Agent access.

## Frontend implementation contract

The first UI implementation is a React single-page application written in
TypeScript with strict type checking. Vite builds the browser bundle. React
Router is used in Data Mode with browser history; Framework Mode, SSR and
pre-rendering are out of scope. The Node static server returns `index.html` for
known client-route fallbacks, but never uses that fallback for assets,
`/runtime-config.json` or unknown API-looking paths.

TanStack Query is the only shared cache of Server state. It owns API queries,
mutations, invalidation, refetch and the selected live-update policy. Router,
form and transient presentation state stay local to the narrowest route or
component that needs them. The first slice has no Redux, Zustand or second
client-side copy of the Workflow/Run domain model.

In particular, the UI never predicts an authoritative Run, StageExecution,
allocation, configuration-publication or credential-operation transition. A
successful mutation causes the affected query keys to be invalidated and
refetched; only a subsequent Server response establishes the displayed domain
state. Optimistic updates are allowed for purely local drafts, but not for
execution lifecycle, published configuration or credential state.

The browser API has one committed OpenAPI 3.1 contract at
`api/openapi/contractor-public-v1.yaml`. It describes the public `/v1` auth,
Workflow, Run, Artifact and Operations resources, their error envelope and the
mutation headers used by browser and non-UI clients. Private Control Plane,
Runtime Agent and A2A wire contracts remain separate and are not made public by
including them in this document. Incompatible changes require a new public API
version; regenerating a client from an incompatible document does not make the
change backward compatible.

The UI build pins the exact public contract and a deterministic TypeScript
client generator. Generated DTO/client files are not hand-edited, and CI fails
when regeneration produces an uncommitted diff. A small handwritten transport
adapter around that generated client is the only place that:

- resolves `apiBaseUrl` from runtime config;
- opts into the session cookie and attaches the current CSRF token to unsafe
  requests;
- supplies operation-specific `Idempotency-Key` and `If-Match` values provided
  by the caller;
- maps the specified success/error envelopes into one bounded client error
  type without logging response bodies or secrets.

The adapter does not implement domain transitions, retry unsafe mutations with
a new idempotency key or persist credentials/session material. Server contract
tests exercise request and response examples against the same OpenAPI document.

Vite's transform is not the type-checking gate: CI separately runs strict
TypeScript checking. Vitest and React Testing Library cover units and routed
components; Playwright covers browser behavior against the real Node static
service and Go Server, including login/session recovery, CORS and CSRF failure,
one Run lifecycle, Artifact transfer and an authorized Operations mutation.
The Node major version, package-manager version, generator and all frontend
dependencies are lockfile-pinned. Their upgrades and the UI release remain
independent of the Go Server release while the supported API version overlaps.

## Workflow user surface

The first useful user surface supports:

- listing published Workflow names and exact versions;
- rendering Run string parameters and input Artifact slots from the selected
  Workflow contract;
- uploading a UserScope Artifact or selecting an existing exact version;
- selecting published ModelPolicy, LLMGatewayConfig and credential refs when a
  Run execution override is desired, and showing any Workflow-pinned
  ExecutionConfig escalation profiles;
- selecting zero or more active `runtimeLabels` such as `debug` or `caido` for
  the concrete Run and showing their exact pinned RuntimeConfig refs;
- attaching bounded immutable key/value metadata labels to a Run, displaying
  them separately from Runtime configuration and filtering Runs by exact label
  conjunctions;
- creating an idempotent WorkflowRun and cancelling a non-terminal Run;
- listing Runs and showing their durable lifecycle;
- showing ordered Stages, attempts, retry/escalation decisions, effective
  executionConfig refs, termination reasons and the currently active
  StageExecution;
- showing the active Stage's immutable objective as the global task, its ordered
  validated Planner subtasks, current subtask and optional active logical Worker
  dispatch; Planner subtasks are not rendered as additional Workflow Stages;
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
- the Runtime Agent's frozen positive capability snapshot: exact runtime and
  SandboxProfile refs, each exact Toolset ref and available tool names, plus
  exact RuntimeAdapter refs;
- the certificate-derived Runtime Agent principal and its authoritative durable
  label set, without exposing certificate bytes or label-supplied secrets;
- `idle`, `reserved`, `busy`, `draining` and `fenced` slot state using the
  authoritative/observed reconciliation vocabulary from [02];
- allocation ID, WorkflowRun, StageExecution, logical Worker binding,
  AgentTemplate, effective ModelPolicy and LLMGatewayConfig refs;
- allocation prepare/finalize/abort/release state and bounded failure reason;
- safe aggregate call/tool/token/error counters and whether a ModelPolicy
  execution-safety dimension was exhausted.

The page is an observation surface. It cannot force a Runtime Agent to idle,
reassign an allocation, mark a Stage successful or bypass bounded abort/release.
Explicit administrative recovery operations, if later required, need separate
idempotent Server commands and audit contracts.

The REST snapshot is mutex-consistent with the in-process Control Plane
registry and carries a random process generation plus an unsigned decimal
revision. Observable registration, heartbeat, reservation, lifecycle, report
or release changes advance that revision. A Server restart changes the
generation. Pagination cursors pin both values and fail closed when either is
stale, causing the UI to fetch a new snapshot. This is a current-state view,
not durable allocation history: an allocation disappears after authoritative
release.

The durable Runtime Agent principal/label configuration in [07] is a separate
Operations resource. It remains listable while no process for that certificate
principal is connected and does not claim liveness. When an instance is live,
the current-state projection links it to that principal and its pinned active
allocation settings; process-local Operations revisions still govern only the
live registry view.

`lastAcceptedHeartbeat` and `confirmedLeaseUntil` are absent between a new
registration and the corresponding first accepted/confirmed heartbeat; the
snapshot does not manufacture either timestamp from registration time.

Observed and authoritative fields remain separate. A reserved allocation with
an idle observation is `reserved`; an observed `allocated` Runtime proves that
the Worker role was prepared but does not prove that an A2A call is currently
executing. Therefore allocation `observedPhase` reports `prepared`, not a
guessed `busy`. The derived slot may still be `busy` because the allocation
exclusively owns that single slot. A fenced observation remains visible beside
an unreleased authoritative allocation during reconciliation. Until a final
Runtime report arrives, allocation metrics are an incomplete zero aggregate;
only safe counters and the exhausted budget dimension are retained in this
live projection.

When a process registration supersedes an older registration at the same
Runtime endpoints, the older process remains visible only while it owns an
authoritative allocation that still requires reconciliation. After that
allocation is released—or immediately when it owned none—it is retired from
the current-state snapshot rather than shown indefinitely as a fenced slot.
Control-lease-expired process entries follow the same rule. Retirement removes
the full entry from the bounded in-memory live Registry, not merely from REST
serialization; a new process-scoped UUID can therefore register after any
number of historical restarts. Durable Runtime Agent principals remain visible
through their separate configuration resource.

## Published LLM configuration

Run creation never accepts free-form model names, budgets, Gateway URLs or
tokens. Operations manages and Run forms select already validated published
configurations:

- one shared `ModelPolicy` kind is usable by Planner and Worker consumers;
- role-specific fields are omitted when not used, while each consumer rejects
  a policy missing one of its required finite limits;
- `LLMGatewayConfig` versions carry protocol, inference URL and an optional
  credential-manager declaration; executionConfig selects an optional
  credential independently for that exact Gateway;
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
complete set; there is no root precedence and no implicit override.

The executable settings are `--operator-config-root` /
`CONTRACTOR_OPERATOR_CONFIG_ROOT` and `--managed-config-root` /
`CONTRACTOR_MANAGED_CONFIG_ROOT`. `CONTRACTOR_CONFIG_ROOT` and
`--config-root` remain temporary aliases for the operator root. When the
managed root is omitted, local configuration derives a sibling
`managed-configs/` directory from the final operator-root path.

The first UI increment publishes only ModelPolicy and LLMGatewayConfig
manifests. Existing
Workflow, AgentTemplate, ExecutionConfig and instruction resources remain
operator-authored and read-only until their editors receive a separate
contract. Operations lists the version-indexed configuration kinds here;
instruction refs and digests remain visible through their consuming resources,
while a standalone path-indexed instruction API is deferred rather than
inventing a `name@version` identity for them.

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

## Agent Skill artifacts

Agent Skills use the ordinary owner Artifact API and existing Artifacts UI;
there is no Skill-specific API or Operations page. A package is created or
updated at `skills/<name>` with the same strong ETag preconditions, version
history and exact `ArtifactRef` response as any other UserScope artifact.
[09](09-agent-skills.md) owns package validation and Run forking.

The browser may upload a ZIP with
`application/vnd.contractor.agent-skill+zip`, inspect its ordinary metadata and
history, and download bytes under existing owner authorization. It does not
preview instructions, validate or edit package structure, enable scripts, or
claim that a successful upload is runnable: authoritative validation happens
when a referencing Run initializes and again in Runtime. Bundled
`configs/skills` sources merely create missing owner artifacts at startup and
never overwrite an Artifact API update.

## Credential handling

If a credential row exists, its key is active. The model has no `disabled`,
`revoked`, `superseded` or mutable-current state. The token is immutable;
replacing it means creating another credential ID and selecting that ID in a
new Workflow default or Run executionConfig. The UI may create and delete
credentials but cannot update or rotate one in place. `active` means that the
Contractor row exists and its managed Gateway key is expected to exist. There
is no disabled/revoked row state: removing access deletes the key from the
Gateway and the database through the fenced lifecycle below. LiteLLM may still
reject a call because its spend or rate limit is exhausted. That is Gateway
policy state, not a Contractor credential lifecycle state.

The first slice accepts no token in the public create request and never reveals
the token generated by the Gateway. Credential creation accepts a new
credential ID, one exact digest-bearing LLMGatewayConfig ref and an optional
safe label plus the typed Gateway policy below. Read responses contain:

- stable credential ID;
- exact bound LLMGatewayConfig ref;
- created timestamp;
- optional safe label;
- the immutable effective key policy returned by LiteLLM;
- optional live consumption aggregates reported by LiteLLM.

The Server never returns a stored secret, even to the principal that created
it. UI forms do not preserve it in browser storage or logs. Run initialization
pins the non-secret credential ID selected beside the exact LLMGatewayConfig.
Active allocation RuntimeSettings remain unchanged.

Credential metadata and encrypted tokens live in PostgreSQL.
LLMGatewayConfig contains no credential selector; PostgreSQL is authoritative
only for the immutable credential record, never for ModelPolicy or
LLMGatewayConfig bodies. The logical schema is:

```text
llm_credentials
  credential_id
  llm_gateway_id
  llm_gateway_version
  llm_gateway_digest
  remote_key_id
  label
  gateway_policy                   selected refs + canonical effective key policy
  key_id
  nonce
  ciphertext
  created_at

credential_operations              internal recovery journal, not a credential
  operation_id
  idempotency_key
  request_hash
  credential_id
  operation_kind                   create | delete
  phase                            prepared | completed
  created_at
  updated_at
```

`credential_id` uses the shared configuration-ID grammar. The Gateway manager
must return a secret token of 1 through 16,384 UTF-8 bytes and a bounded stable
non-secret remote key identifier; Server performs no trimming and rejects a
missing, malformed or oversized response. It encrypts the exact token bytes
with AES-256-GCM using a fresh 96-bit cryptographically random nonce.
Canonical schema version, credential ID and exact Gateway ref are authenticated
additional data, so ciphertext cannot be moved to another identity or route.
Credential creation inserts the active row only after the Gateway
credential-management boundary has returned the bounded non-secret
`remote_key_id` and token. Plaintext is retained only for the bounded
encryption/decryption operation and the model-client settings that actively
need it.

Credential IDs are never reused. Successful deletion removes the remote key and
encrypted credential row but retains a non-secret audit tombstone containing
only the ID, actor and deletion time; that tombstone is not a credential and
cannot be selected for execution.

The 256-bit Contractor credential-encryption master key is a bootstrap secret
outside PostgreSQL. Server receives only an absolute
`--credential-master-key-file` path. The file contains RFC 4648 base64 for
exactly 32 bytes, with at most one trailing newline; command-line literal and
environment-variable key values are forbidden. Server rejects a missing,
malformed, non-regular, symlinked, group-readable or world-readable key file
and reads it once during startup. The database stores a non-secret `key_id`
formatted as the lowercase `sha256:<hex>` digest of the decoded key bytes with
each ciphertext, so a future keyring migration does not require a schema
change. The fingerprint is not secret because the key has 256 bits of random
entropy. The first slice accepts one active key and does not implement
master-key rotation.

When any credential row exists, Server startup requires the master key. A
decryption/authentication failure is a bounded internal configuration error and
never falls back to plaintext, another credential or an unauthenticated
request. Planner construction and Control Plane allocation preparation decrypt
exactly the credential ID pinned by the Run or durable allocation-config
snapshot, just in time. The token is never
copied into WorkflowRun, StageExecution, Planner Session, audit rows or metrics.
A YAML default naming a deleted or missing credential remains inspectable but
is not runnable unless a valid higher-precedence Run/Agent infrastructure layer
under [07] replaces that physical Worker credential before prepare.

There is no disable operation. Credential deletion is idempotent and allowed
only when no non-terminal WorkflowRun, active RuntimeConfig label binding or
live allocation snapshot pins/references that credential; otherwise it returns
`credential_in_use` with safe referencing Run IDs/label names and performs no
side effect. Under a credential lock, Server first asks the bound Gateway credential
manager to delete `remote_key_id` (`already absent` counts as success), then
deletes the encrypted PostgreSQL row. A Gateway error retains the database
record. Contractor never automatically aborts a Run or silently substitutes a
different key to make deletion succeed.

### LiteLLM virtual-key manager

`litellm-virtual-keys@1` is the only managed credential implementation in the
first slice. It follows LiteLLM's documented virtual-key API: creation calls
`POST /key/generate` and deletion calls `POST /key/delete`. The Server sends the
LiteLLM master key as the management request's Bearer credential. It never
sends that key to a Planner or Runtime Agent; executions receive only the
generated virtual key selected by their immutable ResolvedExecutionConfig.

The LLMGatewayConfig provides the non-secret management origin and declares the
adapter implementation. A separate operator-owned bootstrap document passed as
`--llm-gateway-admin-bindings-file` binds an exact digest-bearing
LLMGatewayConfig ref to one absolute `adminKeyFile` path:

```yaml
bindings:
  - llmGateway:
      gatewayId: local-litellm
      version: "1"
      digest: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
    adminKeyFile: /etc/contractor/secrets/local-litellm-master-key
```

The bootstrap document and key files are read once at startup and cannot be
read or changed through Operations. Duplicate Gateway refs are invalid. Each
admin key file must be absolute, regular, non-symlinked and not readable by
group or world. After removing at most one trailing LF, its exact content must
be 1 through 16,384 UTF-8 bytes and start with `sk-`. The management client uses
only the binding whose complete ref equals the selected immutable Gateway ref;
a missing binding makes credential creation/deletion unavailable and never
falls back to another key or sends an admin key to an unbound URL.

For creation, Server derives a bounded LiteLLM `key_alias` as `contractor-` plus
the lowercase hex value of
`SHA-256(UTF-8(gateway_digest) || 0x00 || UTF-8(credential_id))`. It sends the
alias with bounded metadata containing the credential ID and exact Gateway ref.
The request sets `key_type: llm_api`, so the generated key cannot call LiteLLM
management routes. The first-slice create request also contains one immutable
`gatewayPolicy`:

```text
gatewayPolicy
  modelPolicies                   non-empty set of exact ModelPolicyRefs
  maxBudget                       optional positive finite JSON number
  budgetDuration                  optional LiteLLM duration; only with maxBudget
  tpmLimit                        optional positive integer <= 2^31-1
  rpmLimit                        optional positive integer <= 2^31-1
  maxParallelRequests             optional positive integer <= 2^31-1
```

Server resolves every exact ModelPolicyRef and sends the sorted distinct values
of their `model` fields as LiteLLM `models`. It maps the remaining fields to
`max_budget`, `budget_duration`, `tpm_limit`, `rpm_limit` and
`max_parallel_requests`. Unknown fields, duplicate refs, an empty model set,
non-finite/zero/negative values and an invalid duration are rejected before any
Gateway call. Absence of an optional field lets LiteLLM apply its configured
default. Omitting `budgetDuration` with a present `maxBudget` creates the
Gateway's non-resetting budget; `budgetDuration` without `maxBudget` is invalid.
The first slice does not expose key TTL, automatic rotation, teams, users,
custom aliases or provider-specific extension fields.

LiteLLM is the sole enforcement authority for that policy. Contractor neither
mirrors its spend/rate counters nor rejects Run/Stage admission from a locally
calculated quota. After `/key/generate`, Server validates and stores the
effective key-level policy returned by LiteLLM rather than assuming that the
requested values are effective: Gateway defaults or operator upper bounds may
have changed them. The canonical stored object contains the selected
ModelPolicyRefs, derived requested model aliases and effective key-level fields.
It is informational provenance and drives the Credentials display, not
authorization. Changing policy means creating and selecting another credential
ID; the old credential can be deleted after no non-terminal Run pins it.
Contractor does not call `/key/update`.

Server validates that the response's plaintext `key` starts with `sk-` and that
its non-secret hashed `token_id` is exactly 64 lowercase hexadecimal characters.
It stores `token_id` as `remote_key_id` and uses that exact identifier in the
ordinary `/key/delete` `keys` list. A confirmed `not found` for that identifier
is normalized to success.

Credential create requires an Idempotency-Key. A small durable internal
operation record stores its request hash and phase before the outbound call;
it is not a credential and is neither listable nor selectable by a Run. If a
process stops after LiteLLM created a key but before Contractor committed its
encrypted token and `token_id`, startup recovery removes the deterministic
alias and retries creation instead of creating an untracked second key. An
unfinished deletion is likewise completed before Server accepts Run or
Operations traffic. The pinned LiteLLM compatibility envelope must enforce
unique aliases; an adapter that cannot prove this fails its startup conformance
check. Thus Contractor-managed transitions expose only active credentials, not
a user-visible provisioning, deleting or partially active state. Out-of-band
LiteLLM mutation is unsupported drift and produces a bounded operational error.

Run initialization, RuntimeConfig binding/Agent-label mutation, allocation
config pinning and credential deletion share one credential-lifecycle barrier.
Readers hold it from current credential revalidation through their immutable
Run/allocation commit; deletion holds the write side from all durable/live
reference checks through remote deletion and local tombstone commit. If a
delete call has an ambiguous Gateway or database result,
the encrypted row remains active and inspectable but the write intent remains
prepared and new Run initialization is rejected until the same request or
startup recovery resolves it. This prevents a newly committed Run from pinning
a remote key that may already have been deleted.

Management calls have finite connect/request timeouts and bounded bodies. Their
Authorization header, generated token and raw provider error body are always
redacted. LiteLLM must have its own persistent database for virtual-key state;
Contractor accesses that state only through the management API, never by
reading or modifying LiteLLM tables. Contract tests use an explicitly pinned
LiteLLM version or image digest rather than a mutable `main-stable` tag.

The external protocol references for this adapter are LiteLLM's
[Virtual Keys](https://github.com/BerriAI/litellm-docs/blob/main/docs/proxy/virtual_keys.md)
and [access-control API](https://github.com/BerriAI/litellm-docs/blob/main/docs/proxy/access_control.md)
documentation. Contractor's adapter tests, not an unpinned current LiteLLM
release, define the supported `litellm-virtual-keys@1` compatibility envelope.

Encrypting the database protects a PostgreSQL dump without the master-key file;
it does not protect against compromise of the running Server process or host.
Database and master-key backups must be protected and stored separately. A
future Vault/KMS adapter may implement the same credential interface without
changing LLMGatewayConfig or executionConfig.

Attribution records ModelPolicy, LLMGatewayConfig and credential refs together
with WorkflowRun, StageExecution and Planner/Worker role. Provider-reported
tokens and calls are useful operational measurements, not billing-grade proof.
Gateway-reported spend/rate state may be displayed as operational information,
but a Gateway quota rejection follows the ordinary bounded LLM error path.
ModelPolicy call/token ceilings remain independently enforced because they
bound one Planner or Worker execution rather than account consumption.

## ExecutionConfig

Workflow YAML carries default execution selections. The Run form may submit
only exact published refs in the reference-only `executionConfig` override
defined by [00](00-workflow-and-planner.md). It may choose Planner and Worker
policies independently and may target a particular Stage/Agent binding.

For example, the UI may leave discovery on a local Worker policy while choosing
a stronger published policy and a separately metered credential for the
OpenAPI builder. A `streamline@1` or `router@1` Stage may independently use a
stronger Planner policy. `passthrough@1` exposes no Planner model selection
because it makes no Planner LLM call.

A Workflow escalation may instead select one immutable Stage-local
ExecutionConfig profile by exact `name@version`. Operations exposes that
profile's digest and safe ModelPolicy/LLMGatewayConfig/credential IDs read-only;
the Run form cannot edit its body or mix browser-supplied inline fields into the
referenced profile.

Before returning a successful Run-create response, Server expands defaults and
overrides into immutable base and escalation-variant per-consumer
ResolvedExecutionConfigs. Planner selections are complete; a Worker's physical
Gateway route may remain incomplete until the default/Run/Agent configuration
layers in [07] are resolved for placement. The Run detail page shows the
selected refs and their origin (`workflow`, `run override`, inline escalation
or exact ExecutionConfig ref), but never reconstructs authority from browser
state.

## Runtime labels and infrastructure configuration

Operations manages database-backed immutable RuntimeConfig versions,
revisioned label bindings, write-only encrypted adapter credentials and durable
Runtime Agent label assignments under the contract in
[07](07-runtime-labels-and-infrastructure-config.md). These resources are not
part of the six YAML configuration subtrees and have no disabled state.

The Run form submits Runtime label names only through `runtimeLabels`. The
successful Run response and detail
surface show the exact pinned binding revision, RuntimeConfig ref/digest and
safe adapter refs. They never expose resolved secret headers, proxy passwords,
tokens or allocation RuntimeSettings. Rebinding a label uses an idempotency key
and `If-Match`; the UI reports a stale revision instead of overwriting another
operator's update.

The reserved default binding is displayed separately and is never a checkbox:
it is pinned for every Run even when the explicit label selection is empty.
Before placement the Run page can show only default/Run layers. Once a
StageExecution pins an allocation, its detail shows the selected Agent labels
and final safe provenance, making any higher-precedence Agent override explicit.

Runtime Agent detail permits replacing the complete authoritative label set.
Startup labels seed only an unseen certificate-derived principal, so the UI has
one set to explain rather than separate declared/managed/effective views. A
busy agent retains its active allocation snapshot; the page makes clear that
the changed labels apply to a future allocation.

## WorkflowRun metadata labels

Run-create, list and detail expose the separate immutable `labels` map owned by
[16](16-run-metadata-labels.md). Its key/value editor and exact filters never
share controls with RuntimeConfig selection. Conventional `eval.*` shortcuts
are presentation over the generic label query and do not create an eval-only
Server execution path.

Planner and Worker traces may carry these labels as bounded attributes. The UI
therefore treats their values as telemetry-visible safe metadata and warns
against secrets. It does not use labels as authorization, scheduling or Run
success signals.

## API boundary

Browser code uses authenticated public Server APIs directly. UI-oriented query
endpoints may aggregate existing durable/read-model state, but mutations must
call the same domain application services as non-UI API clients.

Because UI and API may have different origins, Server has an operator-configured
exact browser-origin allowlist. Wildcard and reflected origins and the opaque
`null` origin are invalid. A matching preflight exposes only implemented public
API methods and required headers, including `Content-Type`, `Idempotency-Key`,
`If-Match` and `X-CSRF-Token`; browser-readable response headers include
`X-Request-ID`, `ETag` and `Content-Disposition`. Credentialed CORS is enabled
only for an exact allowed origin. CORS grants no authorization and non-browser
clients remain subject to the public API's authentication contract.

### Live event WebSocket

Live UI updates use one read-only WebSocket endpoint on Go Server:
`GET /v1/events/ws`, negotiated with subprotocol `contractor.events.v1`. The
browser connects directly to the configured API origin; Node does not proxy the
upgrade. Production uses `wss`, with `ws` accepted only for the same loopback
development exception as HTTP runtime config.

The upgrade authenticates the ordinary Server session cookie and rejects every
`Origin` other than the exact configured UI origin before accepting the socket.
Authorization is rechecked for each subscription, so a principal may observe
only its own Runs and its permitted Operations surface. Session expiry, logout
or revocation closes its sockets. Periodic socket checks validate the existing
session handle without extending its idle deadline; revocation notifications
are only a fast process-local edge and do not replace those checks. No
credential, CSRF token or cursor is placed in the URL. Because this channel is
strictly observational, subscription and unsubscribe frames require no CSRF
token and cannot cancel a Run, publish a configuration, manage a credential or
perform any other domain mutation. Those operations remain authenticated HTTP
requests with their existing CSRF, idempotency and CAS contracts.

One connection multiplexes explicit `run` and `operations` subscriptions. All
client and Server frames are bounded JSON and carry the protocol version. A
Server event envelope contains:

```json
{
  "type": "event",
  "stream": {"kind": "run", "id": "run_..."},
  "cursor": {"generation": "run-generation-...", "sequence": "42"},
  "kind": "planner.event",
  "occurredAt": "2026-08-30T12:00:00Z",
  "data": {}
}
```

The cursor combines an opaque generation with an unsigned decimal string
sequence, so a Server restart or browser number precision cannot silently alter
its meaning. Unknown protocol versions, stream kinds, event kinds and fields
are rejected rather than guessed. The message contract is committed at
`api/events/contractor-events-v1.schema.json`; the OpenAPI document describes
the HTTP upgrade and links that schema. Normal generated HTTP methods and the
browser WebSocket adapter remain separate code paths.

For a `run` stream, every durable Run/StageExecution lifecycle change and every
public Planner fact receives one monotonically increasing per-Run sequence in
the same transaction as its source record. Its generation is stable for that
Run. A Run detail response includes the corresponding event cursor. Subscribing
with `after` replays committed events after that cursor in order and then
follows new commits; reconnect uses the last fully processed cursor. Run events
remain replayable for as long as the Run itself is retained. `planner.event`
carries exactly the reduced durable fact or typed Planner-plan projection
change defined by [04](04-execution-lifecycle-and-metrics.md), never a live raw
ADK or model stream. Lifecycle events are invalidation hints: the UI refetches
the affected TanStack Query and does not treat their payload as a replacement
authoritative aggregate.

`workflow_run_events` is the sole Run replay authority. Its insert trigger
emits a PostgreSQL notification that becomes visible only if the source
transaction commits. That notification is only a coalescible wake-up hint;
each subscription also performs bounded periodic catch-up from its last
delivered sequence, so a lost or duplicated notification cannot create a gap.
A subscription without `after` tails from the cursor captured after
authorization and does not replay older history. Server sends `subscribed`
before replay, while `unsubscribed` is sent only after that subscription pump
has stopped, so no later frame can reuse the acknowledged subscription ID.

The `operations` stream reports Runtime Agent, allocation, configuration and
credential changes, but its cursor is process-local and its notifications are
not a new audit log. The Operations snapshot response includes its current
generation and revision. A new subscription is established from that cursor;
Server restart creates another generation, while a missed revision or an
unavailable cursor produces `resync_required`. The UI then obtains another
authenticated REST snapshot before continuing. Server retains only the latest
255 typed Operations invalidations; an older cursor also requires that snapshot
resynchronization.

Delivery may be duplicated across disconnects. The client deduplicates by
stream and cursor, processes events in order and treats any gap as a resync,
not as permission to infer missing state. Each connection has a bounded output
queue; a slow consumer is closed with retryable overload semantics instead of
blocking Scheduler or Planner persistence. Client reconnect uses bounded
exponential backoff with jitter and always supports explicit manual refresh.
Server ping/control frames detect dead connections.

The first-slice limits are 16 KiB per client frame, 64 KiB per Server frame,
eight sockets per authenticated session, 32 subscriptions per socket and an
output queue of at most 256 frames or 1 MiB, whichever is reached first. Server
sends a ping after 20 seconds without outbound traffic and closes a connection
that has not answered within 60 seconds. UI reconnect starts at 500 ms, doubles
up to 30 seconds and applies full jitter. Exceeding an input or subscription
limit closes or rejects only that socket; filling the output queue closes it
with WebSocket status `1013`, after which the client resynchronizes.

Contract tests cover invalid Origin/session, unauthorized Run subscription,
ordered replay after reconnect, duplicate delivery, Operations resync after
Server restart and slow-consumer closure. They also assert that prompts, model
responses, reasoning, tool payloads, provider bodies and credentials cannot
appear in a Planner event frame.

### Local browser session

The first slice has exactly one local principal. Server reads it once at startup
from the absolute `--local-auth-file` bootstrap path:

```yaml
user:
  userId: local-admin
  username: admin
  passwordHash: $argon2id$v=19$m=65536,t=3,p=1$...
```

`userId` uses the shared ID grammar and is the immutable UserScope owner and
audit actor. `username` is case-sensitive, 1 through 64 ASCII characters and
matches `[A-Za-z0-9][A-Za-z0-9_.-]*`. The file must be absolute, regular,
non-symlinked and unreadable by group or world. It contains one Argon2id PHC
string with version 19, 64 MiB memory, three iterations, parallelism one, a
16-byte random salt and 32-byte output. Server rejects another algorithm or
parameter set rather than silently weakening or unexpectedly amplifying login
cost.

`contractor-server auth hash-password` reads the password twice from an
interactive terminal without echo and emits the bootstrap YAML; it accepts no
password command-line argument or environment variable. Passwords are 12
through 1,024 UTF-8 bytes, are never trimmed and are never logged. Login uses a
2 KiB maximum JSON body, `Cache-Control: no-store` and one generic failure
response for unknown username and bad password. Before Argon2 verification, an
in-memory limiter permits at most five failed attempts per rolling minute for
one socket-peer IP and 30 process-wide; it does not trust forwarded-IP headers
in the first slice. Excess returns `429` plus bounded `Retry-After` without
creating an account lockout.

The browser auth API is:

```text
POST /v1/auth/login       username + password; create session
GET  /v1/auth/session     return safe principal + session-bound CSRF token
POST /v1/auth/logout      require CSRF; destroy session
```

Successful login always creates a fresh cryptographically random 32-byte value,
puts its unpadded base64url representation only in a
`__Host-contractor_session` cookie and stores only its SHA-256 digest in Server
memory. The cookie is `HttpOnly`, `Secure`,
`SameSite=Lax`, `Path=/`, has no `Domain`, and is sent by frontend requests with
`credentials: include`. Production UI and API must therefore be HTTPS and
same-site even when they have different origins. A separately named insecure
cookie is permitted only behind an explicit loopback-development mode and is
never accepted on a non-loopback listener.

The concrete bootstrap settings are the required absolute
`--local-auth-file`, repeatable exact `--browser-origin`, and optional
`--insecure-loopback-cookie`. Their environment equivalents are
`CONTRACTOR_LOCAL_AUTH_FILE`, comma-separated `CONTRACTOR_BROWSER_ORIGINS`, and
`CONTRACTOR_INSECURE_LOOPBACK_COOKIE`. The insecure mode uses only
`contractor_loopback_session` and requires an IP-literal loopback Server
listener plus loopback HTTP origins; the production cookie name is never
downgraded. The retained non-browser `Authorization: Bearer` path maps to this
same local-auth principal and requires neither Origin nor CSRF. The old
`CONTRACTOR_PUBLIC_USER_ID`, when present, is only a compatibility assertion
that must equal local-auth `userId`, not a second principal source.

Sessions have an eight-hour idle limit, a 24-hour absolute limit and a maximum
of eight live sessions; a ninth successful login revokes the oldest. An
accepted authenticated request advances only the idle deadline. Restarting
Server invalidates every session; no browser session, password hash or CSRF
token is written to PostgreSQL. `GET /v1/auth/session` uses
`Cache-Control: no-store` and lets a reloaded SPA recover a session-bound
32-byte random CSRF token without exposing the session cookie. Every
cookie-authenticated non-safe request requires both an exact allowed `Origin`
and `X-CSRF-Token`; missing or mismatched values have no side effect. Login
itself requires JSON plus an exact allowed browser Origin when an Origin header
is present. Non-browser authentication does not gain authority from CORS or
bypass the domain's owner/audit rules.

The local principal owns both ordinary user and Operations capabilities in the
first slice. API components receive `userId` from the authenticated request
context rather than a request field or UI state. Authentication failure is
`401`, authenticated lack of a future permission is `403`, and ownership checks
retain their non-disclosing `404` behavior. A future OIDC/session adapter and
RBAC reuse the same principal boundary rather than changing Workflow, Artifact
or credential semantics.

Run and Artifact mutations retain their existing idempotency, ownership and CAS
requirements. Configuration publication, credential creation/deletion and
RuntimeConfig/label/Agent-label administrative commands require
their own idempotency keys, revision preconditions where specified and audit
actor.

Pagination, filter grammar and live-update transport are not selected yet. The
API keeps configuration/credential mutations distinct from ordinary Run and
Artifact use so later RBAC does not require changing domain semantics.

## Invariants

1. UI can request domain operations but cannot manufacture execution state.
2. Run overrides contain exact published refs only; no inline model, budget,
   URL, token or provider parameter is accepted.
3. Immutable base and escalation-variant ResolvedExecutionConfigs are
   authoritative after Run creation; an ExecutionConfig ref is never resolved
   during an attempt. The default/Run/Agent infrastructure layers in [07] may
   complete or override only a Worker's physical connection settings under
   their separately pinned provenance.
4. Planner and every logical Worker resolve their model policy, Gateway and
   optional matching credential independently.
5. Published configuration versions and their digests never change in place.
6. Public read APIs never return raw key material. LiteLLM virtual keys and its
   master key never appear in durable execution state, metrics, logs or browser
   persistence; only generated virtual keys are encrypted in the LLM
   credential store. Adapter-specific runtime credentials may be submitted
   once through the authenticated write-only Operations boundary defined by
   [07], then are likewise encrypted and never readable.
7. Operations reflects both observed Runtime state and authoritative Control
   Plane state without conflating them.
8. LiteLLM, not Contractor, is authoritative for Gateway-wide model access,
   spend and rate limits; ModelPolicy call/token ceilings remain authoritative
   only for bounded execution of their selected Planner or Worker.
9. Existing API clients can perform the same domain operations without using
   the Web UI.
10. Web UI is an independently releasable Node.js service; Go Server never
    embeds frontend assets or requires UI availability.
11. Agent Skill packages use ordinary owner Artifact CAS/history; Runtime sees
    only exact RunScope forks and no owner Artifact authority.
12. Node serves only client assets and public runtime configuration; browser
    code, not Node, directly invokes the configured Go Server API.
13. Browser passwords, session cookies and CSRF tokens never reach Node or
    durable execution/storage; Go Server derives the sole local principal from
    its authenticated in-memory session.

## Open decisions for the next dialogue steps

- LiteLLM key TTL/automatic rotation and policy fields beyond the explicit
  first-slice allowlist; LiteLLM remains their enforcement authority;
- Contractor credential-encryption master-key rotation/re-encryption and a
  future Vault/KMS adapter;
- component library/design system and the production Node static-server
  implementation;
- future OIDC authentication, multiple users and user/Operations RBAC;
- exact list/filter/pagination contracts and retention window;
- whether the initial editor covers Workflow and AgentTemplate or only
  ModelPolicy, LLMGatewayConfig and credentials;
- artifact preview size limits and whether OpenAPI/LikeC4 receive specialized
  renderers in the first UI increment.
