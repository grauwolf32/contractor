# Run-scoped toolset configuration and MCP Streamable HTTP

Plan dated 2026-09-19. Series V61, status: planned, not implemented.
On 2026-09-20, tasks were moved into main and renumbered from V56 to V61 without
changing scope. Statuses, dependencies, readiness criteria and verification
commands are in [`tasks/index.yml`](../../tasks/index.yml) and `tasks/v61-*.yml`.

The user requested a general tool-configuration mechanism through AgentTemplate
and existing bindings, selection at launch in the UI, and access to arbitrary
MCP endpoints. Accepted constraints: Streamable HTTP only, one configuration per
toolset for the entire Run, maximum reuse of RuntimeConfig and minimal YAML changes.

This document records implementation decisions. V61-001 transfers normative
contracts into the existing AgentTemplate, Runtime, RuntimeConfig and Audit
specifications and shared schemas/fixtures. Implementation must follow those
contracts; the plan and tasks are not evidence that the feature is ready.

## User-facing contract

AgentTemplate retains its current structure and explicitly selects tools:

```yaml
spec:
  toolsets:
    - ref: mcp@1
      tools: [search, fetch]
```

`mcp@1` is a new toolset implementation. Endpoint and credentials are supplied
through a new section of the existing RuntimeConfig:

```yaml
apiVersion: contractor/v1alpha1
kind: RuntimeConfig
metadata:
  name: knowledge-production
  version: "1"
spec:
  worker:
    toolsets:
      mcp@1:
        endpoint: https://knowledge.example.com/mcp
        credential: knowledge-access
```

The existing runtime label `knowledge-prod` binds to the document's exact name,
version and digest. Run creation uses the existing field:

```json
{"runtimeLabels": ["knowledge-prod"]}
```

Workflow and Stage agent bindings retain their current shape. No new
ToolConfiguration, configurationInputs, mandatory configBinding or toolset fields
in executionConfig are introduced. Configuration differs from call arguments:
existing `tool@1.execution.arguments` continues to define literal/parameter/artifact
arguments for the operation.

The map key is an exact `<toolset-id>@<version>`. For native toolsets, contents
are defined by their registered schema; the mechanism is not endpoint-specific.
Individual tools may have nested settings where their toolset schema permits it.
The first delivery adds only one user-facing schema, MCP; extensibility is
verified with a test native toolset.

`mcp@1` has one endpoint per Run. Two agents may select different tools from that
connection. Multiple instances of one ref, different endpoints for the same ref
per agent, and dynamic registration of arbitrary aliases are outside this delivery.
Different future registered toolset refs may share one client implementation.

## Descriptor, typing and compatibility

Extend the existing code-owned ToolsetDescriptor with tool source `static` or
`allocation`, configuration schema/requirement, supported Worker runtimes and
shared infrastructure channels for dynamic tool lists. These properties are not
duplicated in user YAML. Each schema registers typed normalization/validation
and credential-reference extraction. Unknown refs, unsupported configuration and
unknown fields are rejected.

`mcp@1` requires MCP configuration; its tool list is checked at allocation time.
Initially, only `adk@1` is supported; HTTP uses the existing `runtime-http-client`
channel. Native static validation, visible names, collision prevention and
special Agent Skill names are preserved. Shared native configuration delivery
also works for model-free Workers without introducing model credentials.

Contracts are extended together in Go, Python, JSON Schema, OpenAPI and strict
UI parsers. Absent new fields are not serialized: old canonical bytes, digests,
the empty built-in RuntimeConfig and static capabilities are preserved.
The unified V50 private protocol remains `contractor/v1alpha1`.

New required configuration for an existing native toolset needs a separate
contract/capability version: the old static ref alone does not establish that
an older Runtime supports new settings. The test native descriptor has its own
ref; this series does not redefine existing production native toolsets.

## MCP schema and credentials

| RuntimeConfig field | Contract |
| --- | --- |
| `endpoint` | Required HTTP/HTTPS URL, up to 2048 bytes; no userinfo, query or fragment; the client does not append a path |
| `credential` | Optional immutable credential ID of kind `mcp-headers@1` |
| `caBundlePem` | Optional additional CA, existing 64 KiB limit; normal TLS verification preserved |
| `connectTimeoutSeconds` | Integer 1–60, default 10 |
| `requestTimeoutSeconds` | Integer 1–600, default 60; total budget for one MCP RPC |

An earlier enclosing deadline always bounds the nested operation. Individual
HTTP connect, full RPC, allocation preparation and cleanup timeouts have distinct
owners; MCP settings do not extend the Scheduler deadline. A long-lived inbound
stream is not an unlimited tools/call budget.

The toolsets map is limited to 128 refs and the existing 128 KiB total RuntimeConfig
bound. Objects are normalized before comparison and digest calculation; optional
defaults are materialized consistently across layers. URL normalization preserves
meaningful path semantics and does not arbitrarily rewrite the user's endpoint.

The new credential stores `{"headers": {"Authorization": "Bearer ..."}}`.
Reuse encryption, write-only creation, idempotency and the reference barrier.
Header normalization is shared with existing header credentials: at most 32
headers, names up to 64 bytes, values up to 4096 bytes, total values up to 16 KiB;
case-insensitive duplicates and CR/LF are forbidden. Besides hop-by-hop/proxy
headers, SDK-managed Accept, Content-Type, MCP-Session-Id, MCP-Protocol-Version
and Last-Event-ID are forbidden. No authorization is supplied without a credential.

Implementation defines new defaults and bounds as named constants. URL, PEM and
RuntimeConfig size bounds reuse existing constants. Shared header limits receive
names without an OTLP prefix and are used by both credential kinds without changing
values. V61-001 records the values in normative specifications and shared fixtures
for Go, Python and JSON Schema default/boundary checks.

RuntimeConfig, Run drafts, provenance, descriptions and tool arguments contain no
plaintext secrets. Headers are materialized only immediately before private
allocation delivery. Extend allowed-credential-kind CHECK constraints, metadata
validators, SQL lookup of referencing bindings, Run/Audit holds and allocation
references. All secret-redaction collectors, safe repr, cleanup and SDK logging
account for new values and session IDs. Rotation uses a new credential ID and
config version. Interactive OAuth and automatic refresh are outside the first delivery.

## Merge, bindings and Run snapshot

Only `default` and explicit Run `runtimeLabels` layers apply to `worker.toolsets`.
The object for one ref is an atomic setting.

| Input | Result |
| --- | --- |
| Ref absent from the Run layer | Inherit the value from `default` |
| Object for a ref in the Run layer | Replace that ref's entire `default` value; object fields are not merged |
| `ref: null` | Clear the ref's value in this layer |
| `toolsets: {}` | Leave the layer unchanged; normalize to an absent section |
| `toolsets: null` | Reject as invalid |
| Different refs in labels of one layer | Combine into one map |
| Identical normalized values for one ref in labels of one layer | Merge without conflict, preserving sources |
| Different values for one ref in labels of one layer, including object vs `null` | Produce a stable conflict identifying ref/path and sources |

Label order does not determine a winner. Conflicts are checked across all selected
labels before projecting onto an individual agent. If required configuration is
missing after merging layers, Run creation fails.

Labels with a nonempty toolsets patch, including clearing, cannot be assigned to
a physical Runtime Agent. Rebinding a label already assigned to agents to such a
document is also rejected under existing locks/revision checks. Mixed configs
follow the same rule. The resolver additionally rejects such an agent layer.
Existing proxy/telemetry rules do not change Run-selected endpoint and credential;
the physical route may differ under current policy.

Run creation collects the union of requirements from all Workflow AgentTemplates,
including later Stages. Within the binding-pinning transaction, validate merge,
required configurations and credential kinds using the same exact refs that will
be stored in the Run snapshot. No network calls to MCP occur in that transaction.
Reuse existing immutable-document storage, without a new configuration table or
copies of document bodies in the Run. Credentials participate in existing holds.

Replay of an accepted idempotent request returns the original Run before checking
current aliases. Repeat creates a new Run and retains existing review of changed
or deleted bindings. Changing a default or label after Run creation does not alter
the existing Run, its later Stages or replacement allocations. Pinning an endpoint
does not pin the remote server implementation.

Audit reuses baseline/credential holds and validates requirements across all its
workflow bindings. Until trusted classification exists, generic MCP has unknown
external effects and is incompatible with Audit profiles requiring action
classification/approval. This is an explicit compatibility reason through existing
admission, not a new approval flow. MCP annotations and names are not trusted
classifications. Native Audit behavior is preserved.

## Allocation and capabilities

The shared pure toolsets-map resolver is used at Run creation and placement.
For an allocation, intersect the effective map with AgentTemplate refs before
loading secrets and computing that allocation's requirements. Additional known
settings in the Run are not delivered to unrelated agents. The existing snapshot
may conservatively retain all selected credential references.

Add `toolsets` to the existing `AllocationSpec.runtimeSettings`.
For MCP, the private value contains endpoint, materialized secret headers, CA
and timeouts. Public and stored projections contain only safe refs/origins.
`origins.toolsets[ref]` uses the existing value-provenance format.
Default/run pin refs and binding revisions are preserved; deduplication records
sources deterministically without inventing precedence.

Proposed new capability:

```json
{"ref": "mcp@1", "tools": [], "toolDiscovery": "allocation"}
```

Absent `toolDiscovery` means the existing `static` mode, where tools is nonempty.
Allocation mode is allowed only for a matching Server descriptor and local
factory. Wildcard names and fabricated startup inventories are forbidden.
The startup probe checks local client readiness without a user endpoint.
The production factory must not advertise MCP before the complete ADK path is
implemented. Placement checks the generic capability; preparation checks actual
selected tools. Required proxy adapters still pass existing capability checks.

## Runtime: transport, lifecycle and Worker

Use the official Python MCP SDK at a compatible version pinned in uv.lock and
the existing ADK BaseTool. The SDK handles JSON-RPC, initialization, session
headers and Streamable HTTP JSON/streaming responses. Legacy HTTP+SSE, stdio,
transport fallback and server subprocess launch are not implemented.

The MCP client receives separate HTTP state/headers/cookies. A configured
`tool-http` route uses existing proxy policy and host restrictions. SDK integration
must not bypass those checks through a raw underlying AsyncClient or allow closing
the MCP client to close the allocation's shared adapter. Redirects are disabled;
CA, forbidden internal Contractor origins and deadlines also apply to reconnect
requests. Existing restrictions are not replaced by a blanket ban on all
private/local endpoints.

The factory creates ToolInstance and a shared session owner without network waits.
The allocation first takes ownership through the current rollback/cleanup path,
then runs optional `prepare(deadline)`, and only then creates the Worker.
An equivalent design is acceptable if ownership before the first cancellable await
is demonstrated; a separate resource supervisor is unnecessary.

Each allocation has its own session. Selected tools of one ref share an owner.
The owner enters and exits SDK async contexts in the same task, tracks in-flight
calls, closes idempotently and obeys existing deadlines and abort/release/lease-loss
fencing. Suppressing an SDK exception does not turn unconfirmed cleanup into
success. Partial preparation remains accounted for until release is confirmed.

Initialization and complete paginated tools/list have finite limits on time,
pages, tool count, message/description/schema bytes and schema depth. Code and
normative specifications define these limits before enabling the factory.
Check missing/duplicate/reserved names, exact allowlists and supported schemas
for selected tools. Arbitrary remote schema refs are not fetched. Unselected
tools do not become model-visible. Selected schemas/tool lists are frozen per
allocation; list-changed cannot expand authority.

BaseTool uses the selected tool's description and JSON Schema; the owner performs
tools/call. Native callables continue through FunctionTool. Worker callbacks,
tool budgets, metrics, observation/completion semantics and the tool-free
finalizer/summarizer are preserved. Native typed effects/evidence are not inferred
from remote names or text. Generic MCP under tool@1 is rejected before allocation;
model-free native tool parameters do not become LLM settings.

Results preserve isError, structuredContent and supported content blocks in a
bounded representation. Unsupported blocks or limit violations produce explicit
outcomes, without hidden URI fetching, loss of the error flag or unbounded base64
in context. Tool business errors differ from transport/protocol errors.
V61-001 defines the exact supported schema/content subset and bounds; V61-006
adds fixtures and implementation before advertising the capability.

Setup/discovery may retry within bounded preparation deadlines. After an
ambiguous tools/call, no new POST is sent automatically. Stream recovery differs
from repeating a remote operation; a new session rechecks frozen selected schemas.
Cancellation/closure does not promise remote rollback. Sampling, prompts, automatic
resource fetching and server requests for local-root access are not enabled.
Session IDs and credentials do not appear in logs.

## Public API, UI and observability

Workflow detail gains computed `toolsetConfigurationRequirements`: ref, required,
a union of tools and bounded consumers (stage/agent). The label picker receives
safe ref coverage (configured/cleared), a `valueDigest` for each configured ref's
normalized non-secret value, modified sections, an exact config ref and binding
revision. The value digest excludes document metadata: identical settings in
different RuntimeConfigs must be recognized as identical. Forms do not fetch a
full config for every label; projection/query count stays bounded. Endpoint and
secret values are unnecessary for selection/comparison. Existing APIs remain
the write surface.

The Run form shows required toolsets, tools/consumers, selected profile, inherited
defaults, missing settings and conflicts. Draft/POST stores only existing
runtimeLabels. The shared profile is selected once; mixed LLM/telemetry effects
are displayed rather than silently discarded.

Connection creation uses existing Operations mutations: create a credential if
needed, publish RuntimeConfig, create a binding, select a label. Each step has a
stable idempotency identity; after failure, obtained refs are preserved and only
the unfinished step is retried. An existing shared label is not implicitly rebound.
Incomplete publication does not delete someone else's resources. Secrets are not
written to URLs/localStorage/Run drafts or read back. The current global
Operations/auth model is preserved.

Initially, use a dedicated MCP form within the existing UI framework. No new
universal schema-form platform or separate network preflight job/RPC is required.
Runtime checks actual availability during preparation; the UI reports that state
accurately rather than simulating a browser probe.

Diagnostics distinguish missing/conflicting config, incompatible credentials,
unsupported capability, connect/auth/schema/missing-tool errors, tool business
errors, timeout/ambiguous calls and unconfirmed cleanup. Setting provenance,
selected names, selected-schema digests and bounded timings/counters are safe;
endpoint/session/credential plaintext is not added as metric labels.

## Tasks and dependencies

| ID | Outcome | New dependencies |
| --- | --- | --- |
| V61-001 | Normative contracts, schemas, typed models and golden fixtures | — |
| V61-002 | RuntimeConfig normalization/merge, MCP credentials and reference lifecycle | 001 |
| V61-003 | Run/Audit pinning, allocation projection, secret materialization and origins | 002, 004 |
| V61-004 | Dynamic tool discovery, template/placement validation and Audit admission | 001 |
| V61-005 | Allocation-owned MCP transport/session, proxy integration and preparation/cleanup | 003 |
| V61-006 | MCP ToolsetFactory, ADK BaseTool, exact selection, calls and accounting | 005 |
| V61-007 | Public requirements/label coverage/provenance projections | 003 |
| V61-008 | Operations and Run UI, connection creation/selection and draft recovery | 007 |
| V61-009 | End-to-end release gate, examples and documentation | 006, 008 |

Detailed depends_on also lists required completed tasks from earlier series.
V61 order: 001 → (002, 004) → 003 → (005 → 006, 007 → 008) → 009.
This does not change other queues' priorities in tasks/index.yml.
All V61 tasks remain pending until their implementation begins.

## Delivery criterion

A controlled local MCP fixture and scripted model execute an ordinary Run through
real Server/PostgreSQL/Scheduler/private allocation/Runtime/ADK boundaries.
Two agents use one pinned configuration, different allowlists and separate
sessions; a third native agent receives no MCP settings or secrets. Rebinding
does not affect a later Stage/replacement allocation of the original Run;
Repeat for a new Run shows the change and pins the new config.

Fixtures cover JSON and streaming transport, pagination, missing/duplicate tools,
incompatible schemas, auth/proxy/TLS failures, oversized content, ambiguous calls,
cancellation during setup/call, lease loss, failed prepare, concurrent close and
unconfirmed cleanup. The browser journey checks connection creation, conflicts
and recovery between publication steps. A regression native Run and model-free
Worker pass with the old settings.

V61-009 adds `make test-toolset-runtime-configuration-release` with an isolated
database, MCP fixture and browser prerequisites. A missing prerequisite,
unselected required case or skipped mandatory case fails the gate. External MCP,
a paid LLM and production deployment are unnecessary for verification. Evidence
precisely records boundaries, executed cases and absence of resource leaks.

Rollout: update schema/server, then Runtime with the implemented capability,
then enable new templates and connections. The new Server accepts old static
registrations; old Runtimes do not receive allocations with unknown fields.
Rollback after publishing new documents requires retaining a compatible reader
or explicitly disabling/draining new runs; an old binary is not declared
compatible with documents it cannot understand.

## Main reuse points

- `internal/config/descriptors.go`, `template.go`: toolset and selection contracts.
- `internal/runtimeconfig`: immutable versions, normalization, merge, bindings,
  transaction pinning, credential references and origins.
- `internal/credentials`: encrypted write-only store and usage barriers.
- `internal/runservice`, `auditservice`: shared Run pinning and Audit baseline.
- `internal/controlplane`, `scheduler`: placement, allocation and materialization.
- `runtime/src/contractor_runtime/factories.py`, `allocation`: ownership/cleanup.
- `runtime/src/contractor_runtime/adapters/http_proxy.py`: HTTP route and guards.
- `runtime/src/contractor_runtime/worker`, `llm/openai.py`: ADK and JSON Schema.
- `ui/src/routes/operations`, `routes/workflows/run-form.tsx`, `run-drafts`:
  existing CRUD, dialogs, labels and idempotent draft lifecycle.

Protocol sources: [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
and the [official Python SDK](https://github.com/modelcontextprotocol/python-sdk).
The SDK version and actual supported protocol/schema/content features are pinned
by contract tests; latest-SDK examples do not supersede uv.lock.
