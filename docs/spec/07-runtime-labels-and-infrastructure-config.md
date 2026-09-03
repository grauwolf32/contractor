# 07 — Runtime labels and infrastructure configuration

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md),
[01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[04](04-execution-lifecycle-and-metrics.md) and
[06](06-server-ui-and-operations.md)

## Purpose

Contractor must be able to redirect allocation-scoped infrastructure without
editing a Workflow or AgentTemplate and without rebuilding or restarting every
Runtime Agent. Typical examples are:

- a Run carrying `debug` emits Worker and Planner telemetry to one configured
  OTLP/Langfuse deployment;
- a Run carrying `caido` sends eligible Worker HTTP traffic through one
  configured Caido proxy;
- a Runtime Agent carrying a site-specific label obtains the LLM Gateway,
  proxy or telemetry settings appropriate for that agent;
- rebinding `debug` from `langfuse-a@3` to `langfuse-b@1` affects new Run
  snapshots and future Agent-label allocation boundaries, but does not mutate
  an active allocation.

A label is therefore a short Control Plane alias for an immutable, typed
`RuntimeConfig`; it is not configuration itself. Runtime Agent never resolves
or interprets strings such as `debug` or `caido`. Control Plane resolves labels,
merges their typed settings, resolves credential references, and sends one
complete allocation-scoped `RuntimeSettings` snapshot.

This boundary deliberately keeps infrastructure knowledge on the Server:

```text
Agent Runtime labels + pinned Run-selected Runtime labels
  -> Control Plane starts with the pinned default RuntimeConfig
  -> resolves exact Run-selected and Agent RuntimeConfig versions
  -> validates and merges typed non-secret settings
  -> selects a Runtime Agent supporting the required adapter versions
  -> resolves referenced secrets just in time
  -> sends one immutable RuntimeSettings snapshot over mTLS
  -> Runtime Agent constructs allocation-scoped adapter instances
  -> finalization/abort bounded-flushes and destroys adapters
  -> release erases the retained RuntimeSettings secrets
```

Runtime Agent code still contains adapter implementations such as
`otlp-http@1` and `http-proxy@1`. It does not contain deployment endpoints,
tokens, concrete Langfuse/Caido instance names, or a mapping from label names
to behavior.

## Vocabulary

| Concept | Meaning |
|---|---|
| Runtime label | A bounded name such as `debug` or `caido` selected through `runtimeLabels` on a Run or assigned to a Runtime Agent |
| Default binding | The reserved `default` binding used as the base for every Run, including a Run with no labels |
| RuntimeConfig | One immutable versioned typed non-secret infrastructure configuration stored by Server |
| Label binding | The mutable, revisioned Control Plane pointer from one label to one exact RuntimeConfig version |
| Run Runtime-label snapshot | Exact label-binding/config versions pinned when the WorkflowRun is created |
| Agent label set | The authoritative durable labels assigned to one stable Runtime Agent principal |
| ResolvedRuntimeConfig | The merged, non-secret, exact configuration and provenance for one consumer/allocation |
| RuntimeSettings | The secret-bearing wire snapshot created for one allocation after placement |
| Runtime adapter | Versioned Runtime-owned code that consumes one typed settings block, for example `otlp-http@1` |

The model has one label set on each subject. It has no `declaredLabels`,
`managedLabels`, `effectiveLabels` or per-label `source` fields. Startup labels
only seed a previously unseen Runtime Agent principal; after that first
registration the Control Plane database is authoritative.

Runtime labels are configuration selection, not authorization roles,
scheduling scores, arbitrary user tags or Workflow graph conditions.
Independent immutable WorkflowRun metadata labels are owned by
[16](16-run-metadata-labels.md). All valid Runtime
Agent certificates have the same Runtime role eligibility under [02]; access
to one allocation's settings, A2A route and Artifact scope remains bound to
that allocation's authenticated principal and instance grant.

## Label and configuration identity

A label uses the existing configuration-ID grammar:

```text
label = [a-z][a-z0-9_-]*
```

It is 1–63 ASCII bytes. `default` is reserved for the base binding and cannot
be selected on a Run or assigned to an Agent. A Run and a Runtime Agent may
each carry at most 32 unique labels. Request order has no semantics; Server
normalizes successful response and durable forms lexicographically. Duplicates,
unknown labels and malformed names are rejected. A label is active when a
binding exists. There is no `enabled` or `disabled` state.

Control Plane has exactly one active `default` binding. It is resolved for
every Run, not only when the explicit label list is empty. This makes
RuntimeConfig versions useful as typed patches: `debug` may add only telemetry
and inherit the default Worker Gateway, while `caido` may add only a proxy.
When neither the Run nor the selected Runtime Agent has labels, the resolved
configuration is exactly the pinned default plus executionConfig.

The database migration idempotently creates the immutable built-in
`contractor-empty@1` RuntimeConfig with `spec: {}` and points a missing default
binding to it. That no-op baseline preserves existing Workflows whose
executionConfig already supplies a complete Worker route. An operator may
atomically rebind `default` to a non-empty version. The built-in identity cannot
be replaced through publication, and its normalized digest is fixed by the
schema implementation. Its exact JCS bytes and digest are:

```text
{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"contractor-empty","version":"1"},"spec":{}}
sha256:80a1754c01f8443c29fdc8f650a2254b2461694819918b204a55a7ad3425dc5f
```

A RuntimeConfig uses the exact `<id>@<version>` selector grammar from [00]. A
published version is immutable and has a SHA-256 digest over its normalized
document, encoded as lowercase `sha256:<64 hex>`. RuntimeConfig versions and
label bindings are stored in PostgreSQL; they do not add a seventh YAML
configuration subtree. A binding contains:

```text
RuntimeLabelBinding
  label
  exact RuntimeConfigRef(name, version, digest)
  monotonically increasing uint64 revision (starts at 1)
  created/updated timestamp and audit actor
```

Creating another RuntimeConfig version does nothing by itself. Rebinding a
label is one compare-and-set transaction against the binding revision. It
either installs the complete new exact ref or leaves the old binding intact;
readers can never observe a partially updated configuration.

Compact `worker.llmGateway.gateway` input such as `local-litellm@1` is an
authoring selector only. Before computing the RuntimeConfig digest and
committing publication, Server resolves it to the exact current
`LLMGatewayConfigRef(gateway_id, version, digest)` and the normalized immutable
document contains that expanded ref. Exact publication replay is looked up by
idempotency key and canonical author-request digest before this mutable
dependency resolution; a lost-response replay therefore returns the already
published version even if the Gateway catalog later changed. RuntimeConfig read
models expose the expanded safe ref rather than reconstructing it from current
catalog state.

API ETags encode revisions as a quoted unsigned decimal value. A successful
semantic mutation increments by exactly one; an idempotency replay or rejected
precondition does not consume a revision.

The default binding may be atomically rebound but not removed. Publication and
binding validate that every referenced credential currently exists and matches
the typed adapter/Gateway. An immutable RuntimeConfig version or encrypted
credential cannot be deleted while an active label binding, non-terminal Run
snapshot or live allocation pins/references it. Removing a label binding first
requires removing that label from every Runtime Agent. Existing Run
snapshots do not depend on the mutable binding and remain executable. Physical
garbage collection of unreferenced immutable versions is deferred; the first
implementation may retain them indefinitely.

## RuntimeConfig shape

RuntimeConfig is an API-managed typed document. Its first schema is
conceptually:

```yaml
apiVersion: contractor/v1alpha1
kind: RuntimeConfig
metadata:
  name: observable-through-caido
  version: "1"
spec:
  worker:
    llmGateway:
      gateway: local-litellm@1
      credential: worker-local
    telemetry:
      adapter: otlp-http@1
      endpoint: https://telemetry.internal/v1/traces
      credential: langfuse-otel
      captureContent: false
      flushTimeoutSeconds: 3
    httpProxy:
      adapter: http-proxy@1
      proxyUrl: http://caido.internal:8080
      credential: caido-proxy
      caBundlePem: |-
        -----BEGIN CERTIFICATE-----
        ...
        -----END CERTIFICATE-----
      targets: [llm-gateway, tool-http, tool-subprocess]
  planner:
    telemetry:
      adapter: otlp-http@1
      endpoint: https://telemetry.internal/v1/traces
      credential: langfuse-otel
      captureContent: false
      flushTimeoutSeconds: 3
```

Every section is optional. A user-published RuntimeConfig must contain at least
one setting or explicit-null patch operation; only the built-in
`contractor-empty@1` has an empty `spec`. Unknown fields and adapter versions
are invalid. Endpoints are absolute `http` or
`https` URLs without userinfo or fragments. Bounds on URL, PEM and complete
document size are enforced before persistence. The normalized body contains
credential IDs but no secret bytes.

The first schema fixes those bounds so publication is independently testable:

- the RFC 8785 JCS encoding of the complete normalized document is at most
  128 KiB; its SHA-256 digest covers `apiVersion`, `kind`, normalized metadata
  and `spec`, excluding only the derived digest itself;
- an endpoint/proxy URL is at most 2,048 UTF-8 bytes, has no query unless the
  adapter schema explicitly permits one, and is normalized without resolving
  DNS or contacting it;
- `flushTimeoutSeconds` is an integer from 1 through 10 and defaults to 3;
  omitted `captureContent` is `false`, while `true` is invalid in `v1alpha1`;
- proxy `targets` is a unique non-empty subset of the three values shown above;
- `caBundlePem` is at most 64 KiB, contains one through eight parseable X.509
  certificates and no private-key PEM block; the exact validated UTF-8 string
  bytes participate in the RuntimeConfig digest without Unicode/line-ending
  rewriting;
- every credential ID uses the configuration-ID grammar and is validated
  against the credential kind required by its containing adapter block.

Normalization materializes schema defaults such as `captureContent: false` and
`flushTimeoutSeconds: 3`, sorts set-valued `targets`, rejects duplicate object
keys before decoding and otherwise preserves scalar bytes under RFC 8785. Thus
omitting a default and spelling it explicitly produce the same digest, while a
semantic endpoint, credential, target or trust change produces a new digest.

The first typed leaves are:

- `worker.llmGateway`: a non-empty typed patch containing an exact existing
  `LLMGatewayConfig` ref, a matching LLM credential ref, or both; after all
  precedence layers the resulting Worker route must be complete;
- `worker.telemetry`: one exact Runtime adapter, OTLP endpoint, optional typed
  credential, safe capture policy and bounded flush timeout, or explicit null
  in an overlay;
- `worker.httpProxy`: one exact Runtime adapter, proxy endpoint, optional typed
  credential, optional bounded CA bundle and an explicit non-empty target set,
  or explicit null in an overlay;
- `planner.telemetry`: the Server-side exporter configuration used by a
  Planner invocation for that Run, or explicit null in an overlay.

`RuntimeConfig` cannot select a ModelPolicy, Planner implementation,
AgentTemplate, Toolset, SandboxProfile, Stage deadline, retry/escalation rule or
Workflow transition. Stronger models and larger call/token budgets remain an
explicit `executionConfig`, for example `high-budget@1`; they are not labels.
Planner LLM Gateway/model selection also remains entirely in executionConfig.

The initial telemetry adapter is backend-neutral `otlp-http@1`. It can target
a compatible Langfuse deployment without adding LangChain to Contractor. A
future backend receives another versioned adapter rather than a hard-coded
meaning for `debug`.

`otlp-http@1` exports OTLP/HTTP protobuf traces to the configured complete
traces endpoint. It does not export OpenTelemetry metrics/logs and does not
replace Contractor's durable ExecutionReport. Its bounded resource attributes
contain Contractor service/version, Run/StageExecution/allocation or Planner
session correlation IDs, logical Worker name, exact adapter/config refs and
safe label names. Span attributes may contain operation kind, model alias,
tool name, outcome, duration and token/count aggregates, but never prompts,
responses, artifact content, tool arguments/results, URLs with query/userinfo,
credentials or provider error bodies. Export uses a bounded queue whose
overflow increments adapter failures without blocking model/tool execution.
The first queue holds at most 2,048 span records and 2 MiB of encoded pending
data; a record that would exceed either bound is dropped and counted. One span
has at most 64 attributes and one string attribute is at most 256 UTF-8 bytes.
There is no durable exporter spool or infinite retry; a process crash may lose
unflushed external telemetry while Contractor's own durable lifecycle remains
authoritative.

`http-proxy@1` configures ordinary HTTP proxying. A model-visible Caido API tool
would be a separately selected Toolset plus a future typed `caido-api@1`
adapter. Merely applying a `caido` label never adds that tool to an
AgentTemplate.

## Credentials

RuntimeConfig stores only exact credential references. Secret material lives
in the existing encrypted PostgreSQL credential boundary, generalized for
versioned adapter-owned credential schemas. Initial runtime credential kinds
are:

- `otlp-headers@1`: a bounded secret HTTP header mapping for `otlp-http@1`;
- `http-proxy-basic@1`: username/password for `http-proxy@1`;
- `http-proxy-bearer@1`: a bearer token for `http-proxy@1`.

Header names are validated and hop-by-hop, routing and body-framing headers are
forbidden. Secret values are accepted once through an authenticated Operations
mutation over HTTPS; cookie-authenticated browser calls additionally require
the existing session-bound CSRF proof. Values are encrypted with the Contractor
credential master key and never returned by any read API. Existing managed
LiteLLM credentials keep their current create/delete lifecycle and gateway
binding.

One encrypted runtime-credential plaintext is canonical JSON of at most 32 KiB.
`otlp-headers@1` contains 1–32 unique RFC 9110 field names, each at most 64
ASCII bytes, with values from 1 through 4,096 bytes and at most 16 KiB total.
It rejects `host`, `content-length`, `connection`, `transfer-encoding`,
`upgrade`, proxy-routing headers and CR/LF. Basic-auth username is 1–256 UTF-8
bytes and its password is 1–8,192 bytes; a bearer token is 1–8,192 bytes.
Read projections expose only credential ID, kind and creation metadata.

Server decrypts only the credentials required for a Planner invocation or a
selected allocation after non-secret resolution and placement succeed. Secret
values enter RuntimeSettings over the private mTLS channel, remain in memory,
never enter AgentTemplate/instructions/model context/durable provenance, and
are erased during release. The active-binding/Run/allocation deletion fence is
defined above and applies to both LLM and adapter credentials.

## Runtime Agent principal and labels

Control Plane needs a stable subject for labels assigned through Operations,
while `instance_id` intentionally changes on every process restart. The stable
`runtime_agent_id` is therefore derived by Server as lowercase hex
`SHA-256(DER SubjectPublicKeyInfo)` of the authenticated Runtime Agent leaf
certificate. Runtime Agent does not claim this value in JSON.

Each logical Runtime Agent must use its own certificate/key pair. Two processes
using the same key are the same principal and cannot concurrently register as
two slots; a second live `instance_id` is rejected. Operators running two agents
on one VM issue two certificates from the same deployment CA. This changes no
agent's authorization role: the fingerprint only binds durable configuration
and observed process incarnations to an authenticated peer.

Control Plane and the private Artifact API bind every live `instance_id`,
allocation and report to that same principal. A different CA-valid Runtime
Agent cannot claim the allocation or its RuntimeSettings. This is consistency
and secret-recipient binding, not a label-derived permission model.

Renewing a certificate with the same public key retains the principal. Rotating
the key creates a new principal whose initial labels are seeded normally; the
first slice does not copy labels automatically between fingerprints. Operations
may assign the same desired label set to the new principal before retiring the
old certificate.

Registration adds a sorted `initialLabels` list. On the first registration of
an unseen principal, Control Plane validates that every binding exists and
atomically creates the principal record with that label set. On later
registrations it ignores `initialLabels` for state mutation and returns the
database-authoritative label set. Changing startup arguments after first
registration therefore does not silently overwrite an Operations decision;
the operator changes the durable set through Control Plane.

The private registration request also adds mandatory
`supportedRuntimeAdapters`, a sorted unique array that may be empty. The
registration response returns safe `runtimeAgentId`, authoritative `labels` and
`labelRevision` beside the existing heartbeat/lease settings. Runtime may show
them in local diagnostics but does not use the returned labels to configure the
process or active allocation. Registration request and response contain
mandatory integer `privateProtocolVersion: 2`; the shared document
`apiVersion: contractor/v1alpha1` remains unchanged. An older peer fails version
negotiation rather than silently treating missing arrays as defaults.

Runtime enables every installed built-in adapter by default. An immutable
startup allowlist may narrow that installed surface with repeated
`--runtime-adapter <ref>` flags or the comma-separated
`CONTRACTOR_RUNTIME_ADAPTERS` environment value. Only factories in that set are
probed and only successful probes are advertised. The allowlist cannot add an
implementation, manufacture a positive capability, or change after
registration; changing it means starting a new process incarnation. This is an
environment/deployment boundary, not a label or allocation setting.

Operations may replace the complete agent label set with compare-and-set. A
change while a Worker is allocated affects only a later allocation. The active
Worker retains its immutable RuntimeSettings snapshot. The Runtime Agent does
not receive a mutable label-update command and does not restart or re-probe its
environment.

The durable record is conceptually:

```text
RuntimeAgentPrincipal
  runtime_agent_id
  sorted labels
  monotonically increasing uint64 label_revision (starts at 1)
  created/updated timestamp and audit actor
```

First creation and every replacement validate binding existence plus all
same-layer merge conflicts. Rebinding a label likewise validates every current
Agent principal that contains it against that principal's other labels; a
rebind that would make any durable Agent label set internally conflicting is
rejected atomically. Adapter-capability mismatch is not a mutation error
because the principal may be offline or restart with a different immutable
environment; it simply makes a live instance ineligible and is visible in
Operations.

Only `spec.worker` participates when a config is reached through an Agent
label; `spec.planner` is Run/default-only. Assigning a label whose current
RuntimeConfig has no Worker setting is rejected as not Agent-applicable rather
than being accepted as a silent no-op. A combined `debug` config may still
configure Worker telemetry when assigned to an Agent and both Worker/Planner
telemetry when selected on a Run.

Project filesystem configuration remains entirely Runtime-local under
[10](10-runtime-filesystems-and-edit-tools.md). A label cannot supply a host
path, source artifact, workspace mode or storage override. Runtime advertises
its immutable local/memory workspace capability; Workflow selects direct or
overlay semantics and Scheduler pins exact Run artifacts. The existing
`local-workdir@1` SandboxProfile continues to own general allocation scratch.

Caido is different: its GraphQL endpoint and credential are infrastructure and
therefore belong to one atomic `spec.worker.caido` RuntimeConfig field under
[11](11-http-and-caido-tools.md). A label can retarget that client, but cannot
make `caido@1` model-visible when AgentTemplate omitted it.

## Runtime labels selected by a Run

`POST /v1/runs` accepts one optional top-level `runtimeLabels` array:

```json
{"workflow":"likec4-from-source@1","runtimeLabels":["caido","debug"],"labels":{"purpose":"eval"},"parameters":{},"artifacts":{}}
```

Run-selected Runtime labels are immutable. During Run creation Server first
pins the `default` binding and then resolves every selected label binding to its exact
RuntimeConfig ref. It validates the Run-level configuration, pins every
binding revision/config digest and non-secret credential ref, and stores that
snapshot in the same transaction as the Run. The canonical idempotency digest
includes the sorted explicit `runtimeLabels` list independently of the
metadata-label map defined by [16]; the pinned default is a resolved dependency
rather than caller input. Exact replay returns the existing Run before
consulting current bindings, including `default`.

Run creation locks the default and selected binding rows in lexical order in
the same PostgreSQL transaction that stores the Run snapshot. Binding rebind
uses the same row lock plus revision compare-and-set. A race therefore pins
either the complete old exact ref or the complete new exact ref, never a mix.
Idempotency replay lookup and request-digest comparison occur before these
mutable dependency reads.

Rebinding `debug` after Run creation has these effects:

- a new Run resolves the new binding;
- an existing non-terminal Run continues to use its pinned version for every
  later Stage and retry;
- an active allocation is unchanged;
- no Workflow, AgentTemplate or Runtime Agent restart is required.

The Server applies the pinned default plus Run-selected Runtime-label
`planner.telemetry` settings to Planner invocations, with that block replacing the default
block. Agent labels can never configure Planner, because Planner runs in Server
and no physical Worker placement is part of its model-client contract.

Workflow definitions carry no Runtime labels in `v1alpha1`. A caller chooses
`runtimeLabels` for one concrete Run. If a future product needs
Workflow-recommended Runtime labels, that requires an explicit default/override contract rather than making
an infrastructure alias part of Workflow semantics.

## Resolution and merge rules

Run-selected Runtime-label settings are resolved and pinned at Run creation. Agent-label
settings are read at allocation time because the physical principal is selected
only then. Control Plane builds a candidate configuration in explicit layers;
a higher layer replaces only the typed leaves it supplies:

1. the exact `default` RuntimeConfig pinned by the Run;
2. the Workflow's resolved Worker executionConfig defaults;
3. all exact RuntimeConfigs pinned by the Run's explicit `runtimeLabels`;
4. the Run-request executionConfig override;
5. the already pinned escalation executionConfig patch for an escalated
   attempt, when present;
6. all exact RuntimeConfigs currently assigned to the candidate Runtime Agent
   principal;
7. completeness/capability validation, non-secret provenance persistence and
   just-in-time secret resolution into one allocation RuntimeSettings snapshot.

Agent-label configuration is intentionally the highest infrastructure layer.
It can, for example, replace the common Worker Gateway credential with a token
metered for that physical agent or route that agent through its local proxy.
It cannot replace ModelPolicy, budgets, Planner access or any semantic field.
An omitted higher-layer leaf inherits the lower value.

Run and Agent label names do not need to match. For example:

```text
pinned default
  worker.llmGateway = local-litellm@1 / worker-local

pinned Run label debug
  worker.telemetry = otlp-http@1 / langfuse-a

candidate Runtime Agent labels
  []

resolved allocation
  worker.llmGateway = local-litellm@1 / worker-local
  worker.telemetry = otlp-http@1 / langfuse-a
```

That candidate is eligible when its frozen capabilities include
`otlp-http@1`; it does not need a `debug` Agent label. If it lacks the adapter,
Control Plane tries another candidate or reports temporary insufficient
compatible capacity. If the candidate instead carries an Agent label whose
config supplies `worker.telemetry`, that higher layer replaces the Run's
telemetry block. Labels are configuration inputs, never an equality/affinity
predicate between Run and Runtime Agent.

Conversely, an Agent label applies to every future allocation placed on that
principal even when the Run did not select the same label. Thus an Agent-wide
`site-debug` can export all of that agent's Worker telemetry, but it still does
not enable Planner telemetry because Agent labels consume only `spec.worker`.

There is no ordering or hidden priority among multiple labels within the same
Run-selected Runtime-label layer or within the same Agent-label layer. Same-layer configurations
may populate different merge units. The same exact normalized value in one
unit deduplicates. Different same-layer values in one unit are a deterministic
`runtime_config_conflict`; lexicographic label order never chooses a winner.
An Agent label replacing a Run-selected Runtime-label/default value is not a conflict because
the cross-layer precedence is explicit. Diagnostics report only safe paths and
label/config refs, not endpoint userinfo, headers or secret values.

ExecutionConfig is not another label. It alone selects ModelPolicy, budgets and
Planner model access. Its Worker Gateway/credential fields participate only in
the explicit infrastructure layers above, so an Agent label may override those
physical connection settings but cannot change the selected model policy. Run
initialization may store an incomplete optional Worker Gateway route; each
placement candidate must complete it from the default, pinned Run-selected Runtime labels,
executionConfig and that principal's Agent labels. Model-backed Planner routes
remain complete and pinned at Run creation.

Merge units are explicit rather than inferred from arbitrary JSON depth:

- `worker.llmGateway.gateway` and `worker.llmGateway.credential` are independent
  leaves; `credential: null` explicitly clears an inherited credential and an
  omitted value inherits it;
- `worker.telemetry`, `worker.httpProxy` and `planner.telemetry` are atomic
  blocks. A higher layer replaces the complete block, so endpoints, credentials
  and trust/capture policy cannot be assembled accidentally from different
  deployments. An explicit `null` clears an inherited block, while omission
  inherits it;
- two labels in one layer that provide the same atomic block must provide
  byte-equivalent normalized values or conflict.

After merging, the final Gateway must satisfy the Worker model client's
protocol requirements. A present LLM credential must be bound to that exact
Gateway and its effective policy must authorize the already selected exact
ModelPolicy/model alias. This revalidation cannot change the ModelPolicy or its
budgets. Every adapter block must pass its adapter-owned schema as a whole.

An explicit null is a typed patch operation, not a disabled label or credential
state. For example an Agent `direct-network` config may clear an inherited
`worker.httpProxy` because the Agent layer has higher priority; removing the
label on the next allocation restores the lower Run/default block.

A conflict between explicit Run-selected Runtime labels rejects Run creation. A malformed or
conflicting Agent label set is rejected by the label-assignment mutation. A
same-layer Agent conflict cannot be hidden by a lower Run/default value.
Cross-layer replacement follows the precedence above and is valid. Operations
exposes a bounded safe reason when no currently connected slot can complete the
configuration or adapter requirements.

### Agent-label race and allocation pinning

Agent labels may change while the process is busy, but allocation resolution
has one durable linearization point before any Runtime prepare request. Control
Plane first reads candidate principal label-set revisions and the current
binding revisions for those Agent labels; default/Run-selected Runtime-label refs already come
from the immutable Run snapshot. It then computes a complete compatible
matching. After reserving the complete slot set in memory, it locks the relevant
Agent-label binding rows in lexical order and selected principal rows by
`runtime_agent_id`, verifies those revisions, re-resolves the settings and
durably records each exact revision plus the non-secret allocation config
snapshot. It sends no AllocationSpec before that transaction commits.

All Run creation, binding mutation, Agent-label mutation and allocation-pinning
transactions use the same binding-then-principal lock order. If any revision
changed or the recomputed settings no longer fit the selected adapter
capabilities, Control Plane releases the unprepared reservation and retries
placement for the same StageExecution. A concurrent Agent-label or binding
mutation serializes on those rows: if allocation pinning commits first, the
mutation affects the next allocation; if mutation commits first, this
allocation observes the new set/binding. No database or network operation
occurs while holding the in-memory Runtime Registry lock.

The exact resolution result stored for StageExecution/allocation provenance
contains:

```text
ResolvedRuntimeConfigProvenance
  sorted Run-selected Runtime labels and pinned binding revisions
  pinned default binding revision
  sorted Agent labels and allocation-time binding revisions
  exact RuntimeConfig refs and digests
  exact Runtime adapter refs
  non-secret LLMGatewayConfig and credential refs
```

It never contains resolved tokens, secret headers/passwords, CA private
material or model-visible configuration text.

AllocationSpec may carry the safe label names and exact refs above as telemetry
provenance. Runtime code may attach them as bounded attributes but must not
branch on their spelling; only the typed RuntimeSettings blocks activate
behavior.

The private Go/Python wire contract extends the existing RuntimeSettings
without sending a RuntimeConfig document:

```text
RuntimeSettings
  llmGatewayUrl
  llmGatewayToken?                   secret
  artifactApiUrl
  telemetry?
    adapter                          exact RuntimeAdapterRef
    endpoint
    headers                          secret map resolved from credential
    captureContent                   false in v1alpha1
    flushTimeoutSeconds
  httpProxy?
    adapter                          exact RuntimeAdapterRef
    proxyUrl
    basicAuth? | bearerToken?        secret; mutually exclusive
    caBundlePem?
    targets
  requestTimeoutSeconds

AllocationSpec
  ...existing fields...
  runtimeSettings
  resolvedRuntimeConfigProvenance    safe refs/revisions/digests only
```

Unknown fields, duplicate JSON keys, an unrecognized exact adapter ref,
multiple proxy credential variants or any secret in provenance fail closed
before adapter/workspace construction. Go and Python fixtures must round-trip
the same canonical examples and prove that object formatting/logging redacts
all marked fields.

## Adapter capabilities and placement

Startup capability discovery in [02] adds one fourth composable dimension:

```text
supported_runtime_adapters: sorted set of exact RuntimeAdapter refs
```

Initial refs are `otlp-http@1` and `http-proxy@1`. Their bounded startup probes
verify only that local implementation dependencies can construct the adapter.
They do not contact an OTLP endpoint, proxy, LLM Gateway or credential service.
The snapshot remains immutable for `instance_id`; changing adapter code or
local dependencies requires a Runtime Agent restart.

For each candidate slot, Control Plane first resolves Run-selected plus Agent Runtime labels and
derives the required adapter set. Capability-aware placement from [02] then
requires set containment in addition to the AgentTemplate's runtime, sandbox
and selected Toolset tools. Extra adapter capabilities do not activate an
adapter and do not become model-visible.

After reservation, Runtime Agent validates the exact typed settings against its
frozen capability snapshot before creating a workspace, tool or Worker.
Malformed settings, a missing required secret, digest mismatch or local adapter
construction failure is a bounded preparation failure. Remote OTLP/proxy
availability is not a startup capability. Telemetry delivery failure is
best-effort as defined below; an unavailable required HTTP proxy produces the
ordinary bounded LLM/tool HTTP failure and can therefore affect the semantic
Stage outcome.

The failure mapping is deterministic:

| Failure | Boundary result |
|---|---|
| Unknown Runtime label, Run-selected Runtime-label conflict or invalid referenced config at Run creation | Public validation/conflict error; no Run is created |
| Agent-label conflict or non-Agent-applicable assignment | Operations mutation is rejected atomically |
| Label/binding revision changes during reservation | Unprepared batch is released and placement retries the same StageExecution |
| No live candidate supports the resolved adapter set | Temporary insufficient compatible capacity; Stage remains `preparing` until its existing deadline/cancellation bound |
| Candidate-specific Gateway/credential does not support the selected Worker ModelPolicy | That candidate edge is ineligible; other compatible capacity is considered |
| Missing/decryption-failed secret, schema/digest mismatch or unsupported adapter in AllocationSpec | Bounded preparation interruption; no Planner starts and the whole reservation batch drains |
| Locally advertised adapter cannot construct because of a transient allocation resource error | Retryable preparation interruption with safe code `runtime_adapter_prepare_failed` |
| Required proxy cannot service a Worker request | Ordinary bounded model/tool failure observable to Worker/Planner policy |
| OTLP delivery, queue overflow or final flush failure | Adapter metrics only; semantic result and release continue |

Schema, digest, credential-kind and capability mismatches are non-retryable for
that immutable configuration. Transient local construction failure is
retryable. Existing Stage deadline, cancellation and abort bounds remain the
outer authority; label resolution introduces no unbounded wait.

## Allocation-scoped Runtime behavior

Runtime Agent constructs an `AllocationAdapterHost` before constructing the
sandbox, any Toolset or the Worker. It owns only adapters present in the
resolved settings and injects
explicit clients/handles into the model client and selected Toolset factories.
Label names and complete RuntimeConfig documents are not supplied to the model.

HTTP proxy configuration is never applied through process-global environment,
global certificate stores or module-global clients. Its target set has these
meanings:

- `llm-gateway`: the allocation's Worker model client uses the proxy;
- `tool-http`: HTTP clients explicitly created for selected tools use it;
- `tool-subprocess`: a selected tool may pass a child-only proxy environment
  and CA file to its own bounded subprocess.

These targets are injection contracts, not transparent network interception.
A Toolset descriptor declares for each exported tool whether it consumes the
Runtime-owned HTTP client and/or bounded subprocess launcher. Runtime validates
that declaration against the factory registration. When a selected proxy target
applies, such a tool must use the injected handle and cannot create a second
ambient HTTP client or invoke a network subprocess outside that launcher. A
local/artifact-only tool may declare neither channel and is unaffected. The UI
and provenance can therefore say exactly which selected tool channels were
proxied without claiming that arbitrary process traffic was intercepted.

For `tool-subprocess`, the bounded launcher constructs a child-only
`HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` and trust-bundle environment, closes it
after process exit and treats every proxy-auth value as secret. Private
Contractor hosts are in the bypass set and the child receives no Runtime client
certificate, Artifact grant or Control Plane token unless its separate selected
tool contract explicitly requires an allocation-scoped handle.

The `v1alpha1` generic subprocess environment can carry unauthenticated or
Basic-authenticated proxy URLs. It cannot faithfully encode Bearer proxy
authentication for an arbitrary executable. If `tool-subprocess` is selected
with `http-proxy-bearer@1`, the launcher therefore returns a bounded
fail-closed tool error; it never retries directly and never translates the
Bearer token into Basic credentials. Model and tool HTTP clients support both
credential kinds. Supporting Bearer for arbitrary child processes requires a
future versioned local relay contract.

The private registration/heartbeat/control endpoint, Runtime Agent A2A server,
Artifact API client and any Server-internal traffic always bypass the Worker
proxy. Runtime never mutates its own control-channel trust roots. A temporary
CA bundle is allocation-scoped, readable only by the process/child that needs
it, and removed with the allocation workspace.

For a selected target, proxying is fail-closed: connection, authentication or
TLS failure is returned as the bounded model/tool error and never retries the
same request directly. Proxy credentials are sent only to the configured proxy
authority. The optional CA bundle augments the Runtime's ordinary public trust
roots only for targeted allocation clients; it never replaces or augments the
private Contractor mTLS context.

Telemetry instrumentation emits bounded framework-neutral spans/events for
model calls, selected tool calls, A2A task lifecycle and Worker errors. By
default `captureContent=false`: prompts, model responses, artifact bytes, tool
arguments/results and RuntimeSettings secrets are absent. Enabling future
content capture requires a separate explicit schema and redaction contract;
the first version rejects `captureContent=true` rather than silently exporting
payloads.

An OTLP exporter delivery error, timeout or unavailable backend increments
bounded adapter error metrics but does not change StageResult or
StageTermination. Finalization/abort performs one best-effort flush bounded by
the configured timeout and the remaining lifecycle deadline. It then returns
the Contractor ExecutionReport and destroys the adapter regardless of export
success. Release erases all credential values, closes clients and removes
temporary proxy trust material before the slot can become idle.

Local confirmed-lease loss invokes the same bounded adapter teardown while
self-terminating Worker. It never waits for an exporter beyond the remaining
local grace deadline and remains fenced under [02] even when telemetry cannot
be delivered.

Planner telemetry uses the same pinned Run configuration but is constructed in
Server, not Runtime Agent. Its exporter follows the same no-content default,
bounded flush and best-effort delivery rule. It never receives Agent labels or
Worker proxy settings. Server owns a versioned PlannerTelemetryAdapter registry;
publication rejects a `planner.telemetry.adapter` unknown to that registry.
This Server capability is deployment code, not a Runtime Agent placement fact.

One selected `otlp-http@1` adapter is invocation-local. It does not install a
global OpenTelemetry provider, read ambient proxy variables, follow redirects
or retry. Passthrough, Streamline and Router use a closed instrumentation
vocabulary for invocation/session, model, Worker dispatch, subtask transition
and finish spans. Planner session IDs are attached only after durable session
creation; prompts, model responses, Stage objective/instructions, subtask
content and tool arguments/results cannot enter that vocabulary. Every span of
one invocation shares safe Run/Stage/Planner and exact pinned RuntimeConfig
resource correlation; Agent-label provenance is absent.

After Planner returns, Server makes at most one OTLP request. Its deadline is
the minimum of the configured flush timeout, remaining Stage deadline and the
finalization bound. Adapter creation, credential decryption, encoding,
delivery, queue overflow and flush failure are supplementary: they are logged
with a closed safe code and, when a Planner report identity exists, recorded as
the `telemetry.export` tool outcome in the durable ExecutionReport. They never
change the candidate, retry/escalation decision, StageTermination or allocation
release. No selected Planner telemetry means no exporter/client/request is
created.

## Operations and UI

Operations adds the following authenticated, audited surfaces:

- create/list/get immutable RuntimeConfig versions and their safe digests;
- create write-only typed runtime credentials and delete them when unpinned;
- create/rebind/remove labels with `If-Match` compare-and-set;
- inspect a Runtime Agent principal, its current `instance_id`, frozen adapter
  capabilities and authoritative labels;
- replace one principal's complete label set with compare-and-set;
- remove an offline principal record only after its label set is empty and no
  live instance/allocation refers to it; a later registration seeds it anew;
- select zero or more labels on the Run creation form;
- inspect a Run's pinned labels and a StageExecution's safe resolved config
  provenance and adapter status.

There is no disable toggle. A label without a binding cannot be selected, and a
credential row that exists is active. Read APIs never return secret payloads or
secret-derived hashes. UI previews endpoints and non-secret policy only on the
Operations surface; the owner Run surface shows labels and exact refs/digests,
not resolved credentials or physical Runtime Agent configuration.

Binding and Agent-label mutations use independent idempotency keys plus
revision preconditions. WebSocket Operations notifications remain hints: after
a process-generation/revision gap the UI refetches the authoritative REST
snapshot. No mutation is sent over WebSocket.

### Minimal public API contract

The committed OpenAPI `/v1` document owns JSON spelling. Its first endpoint set
has these semantics:

```text
GET/POST /v1/operations/runtime-configs
GET      /v1/operations/runtime-configs/{name}/versions/{version}

GET      /v1/operations/runtime-labels
GET/PUT/DELETE
         /v1/operations/runtime-labels/{label}

GET/POST /v1/operations/runtime-credentials
GET/DELETE
         /v1/operations/runtime-credentials/{credentialId}

GET      /v1/operations/runtime-agent-principals
GET/DELETE
         /v1/operations/runtime-agent-principals/{runtimeAgentId}
PUT      /v1/operations/runtime-agent-principals/{runtimeAgentId}/labels
```

RuntimeConfig publication and runtime-credential creation are create-only and
require `Idempotency-Key`. Binding `PUT` and Agent-label-set `PUT` require an
idempotency key plus `If-Match` with the current strong revision ETag; initial
non-default binding creation uses `If-None-Match: *`. Delete is idempotent and
also requires the current ETag when the resource exists. The default binding
rejects delete. Stale revisions return `412` without mutation.

Runtime-credential create is write-only: the request contains the typed secret
body, while success contains only safe metadata. To make response-loss replay
safe without retaining a secret-derived public hash, Server authenticates the
canonical request with an HMAC subkey derived from the credential master key
and stores that MAC only in the internal idempotency record. Same key/body
returns the original safe result; same key with different body conflicts. The
MAC is never returned, logged or used as a credential.

`POST /v1/runs` adds `runtimeLabels`, a sorted-unique array in the canonical
request; responses expose both explicit Runtime labels and the pinned
default/label config refs. The independent `labels` map belongs to [16].
Run detail exposes final Agent-label/config provenance only after an allocation
snapshot commits. The public `StageAttempt.runtimeConfiguration` is omitted
before that boundary. Once present, its per-logical-Worker entries contain
only the sorted Agent-label pins, required RuntimeAdapter refs, closed field
origins and `pinned | release_pending | released` cleanup status. It omits
allocation IDs, physical Runtime Agent IDs, endpoints, credentials and the
complete `RuntimeSettings`; current Operations state is never joined into
historical Run detail. Stable errors include `runtime_label_unknown`,
`runtime_config_conflict`, `runtime_config_invalid`,
`runtime_credential_in_use`, `runtime_label_in_use`,
`runtime_agent_label_not_applicable` and the existing precondition/idempotency
errors. Error bodies remain bounded and secret-free.

## Invariants

1. Runtime Agent never interprets a label name or stores deployment topology.
2. One label binding always names one exact immutable RuntimeConfig version.
3. Default and Run-selected Runtime labels are pinned at Run creation; Agent labels are resolved
   at allocation; an active RuntimeSettings snapshot never mutates.
4. Multiple labels in the same layer conflict on different values for one
   merge unit; Agent-label units explicitly override Run-selected Runtime-label/default units.
5. ExecutionConfig alone selects ModelPolicy and budgets. Labels can affect
   physical Worker connection settings but never those semantic limits.
6. Labels configure adapters and already selected tools; they never add a
   model-visible tool, Agent Skill, AgentTemplate or Worker binding.
7. Runtime adapter availability is an immutable positive startup capability,
   while remote endpoint health is an allocation preparation/execution fact.
8. Worker HTTP proxying is allocation-scoped and cannot intercept Contractor
   control, heartbeat, A2A or Artifact API traffic.
9. Secret material is encrypted at rest, decrypted just in time, transported
   only over mTLS, held in memory for the consumer lifetime and never durable
   execution provenance.
10. Telemetry export is bounded and best-effort; exporter failure never changes
    semantic execution outcome or prevents allocation release.
11. A stable certificate-key fingerprint binds durable Agent labels to process
    incarnations but grants no additional private-API permission.
12. Rebinding a label affects only future resolution boundaries and requires no
    Workflow/AgentTemplate edit or Runtime Agent restart.

## Deferred

- label-based affinity, anti-affinity, cost/scoring or authorization policy;
- Workflow-declared default labels and mutable labels on an existing Run;
- arbitrary user tags unrelated to RuntimeConfig;
- explicit priorities among multiple labels inside one layer instead of
  conflict-on-write;
- dynamic Runtime adapter download, re-probe or capability update;
- model-visible Caido API tools and a `caido-api@1` adapter;
- content-bearing telemetry and its separate consent/redaction/retention model;
- cross-process W3C parent/child trace topology beyond safe correlation
  attributes;
- external Vault/KMS-backed runtime credentials and automatic secret rotation;
- immutable RuntimeConfig version garbage collection.
