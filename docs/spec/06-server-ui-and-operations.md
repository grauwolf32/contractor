# 06 — Server UI and Operations

Status: **Working agreement; being refined in dialogue**

Depends on: [00](00-workflow-and-planner.md),
[01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[03](03-artifact-plane.md), [04](04-execution-lifecycle-and-metrics.md)

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
- safe aggregate call/tool/token/error counters and whether a ModelPolicy
  execution-safety dimension was exhausted.

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

If a credential row exists, its key is active. The model has no `disabled`,
`revoked`, `superseded` or mutable-current state. The token is immutable;
replacing it means creating another credential ID and selecting that ID in a
new Workflow default or Run executionConfig. The UI may create and delete
credentials but cannot update or rotate one in place. `active` means that the
key has not been disabled, revoked or deleted; LiteLLM may still reject a call
because its spend or rate limit is exhausted. That is Gateway policy state, not
a Contractor credential lifecycle state.

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
exactly the credential ID pinned by the Run, just in time. The token is never
copied into WorkflowRun, StageExecution, Planner Session, audit rows or metrics.
A YAML default naming a deleted or missing credential remains inspectable but
is not runnable until the request supplies another valid credential override.

There is no disable operation. Credential deletion is idempotent and allowed
only when no non-terminal WorkflowRun pins that credential; otherwise it
returns `credential_in_use` with safe referencing Run IDs and performs no side
effect. Under a credential lock, Server first asks the bound Gateway credential
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
OpenAPI builder. A `streamline@1` Stage may independently use a stronger Planner
policy. `passthrough@1` exposes no Planner model selection because it makes no
Planner LLM call.

Before returning a successful Run-create response, Server expands defaults and
overrides into a complete immutable per-consumer ResolvedExecutionConfig. The
Run detail page shows both the selected refs and their origin (`workflow` or
`run override`), but never reconstructs authority from browser state.

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
future administrative commands require their own idempotency keys and audit
actor.

Pagination, filter grammar and live-update transport are not selected yet. The
API keeps configuration/credential mutations distinct from ordinary Run and
Artifact use so later RBAC does not require changing domain semantics.

## Invariants

1. UI can request domain operations but cannot manufacture execution state.
2. Run overrides contain exact published refs only; no inline model, budget,
   URL, token or provider parameter is accepted.
3. The immutable ResolvedExecutionConfig is authoritative after Run creation.
4. Planner and every logical Worker resolve their model policy, Gateway and
   optional matching credential independently.
5. Published configuration versions and their digests never change in place.
6. Public APIs neither accept raw key material during credential creation nor
   return it on any read. LiteLLM virtual keys and its master key never appear
   in durable execution state, metrics, logs or browser persistence; only
   generated virtual keys are encrypted in the credential store.
7. Operations reflects both observed Runtime state and authoritative Control
   Plane state without conflating them.
8. LiteLLM, not Contractor, is authoritative for Gateway-wide model access,
   spend and rate limits; ModelPolicy call/token ceilings remain authoritative
   only for bounded execution of their selected Planner or Worker.
9. Existing API clients can perform the same domain operations without using
   the Web UI.
10. Web UI is an independently releasable Node.js service; Go Server never
    embeds frontend assets or requires UI availability.
11. Node serves only client assets and public runtime configuration; browser
    code, not Node, directly invokes the configured Go Server API.
12. Browser passwords, session cookies and CSRF tokens never reach Node or
    durable execution/storage; Go Server derives the sole local principal from
    its authenticated in-memory session.

## Open decisions for the next dialogue steps

- LiteLLM key TTL/automatic rotation and policy fields beyond the explicit
  first-slice allowlist; LiteLLM remains their enforcement authority;
- Contractor credential-encryption master-key rotation/re-encryption and a
  future Vault/KMS adapter;
- frontend framework, build tool and client-side state/query library;
- future OIDC authentication, multiple users and user/Operations RBAC;
- polling, Server-Sent Events or WebSocket updates for Run and Agent state;
- exact list/filter/pagination contracts and retention window;
- whether the initial editor covers Workflow and AgentTemplate or only
  ModelPolicy, LLMGatewayConfig and credentials;
- artifact preview size limits and whether OpenAPI/LikeC4 receive specialized
  renderers in the first UI increment.
