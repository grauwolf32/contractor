# 01 — AgentTemplate

Status: **Working agreement**

The authenticated Catalog UI exposes exact template versions and their loaded
base instruction text under [06](06-server-ui-and-operations.md). This read-only
view does not expose invocation context or introduce a prompt editor.

Depends on: [00](00-workflow-and-planner.md)

## Purpose

`AgentTemplate` is the reusable definition of a Worker's intended
behavior. It separates three things that the original monolith partially
conflated:

- the Stage-local logical name used by Planner;
- the semantic Worker configuration reused across Workflows;
- the allocation-scoped Worker runtime instance created from that configuration
  inside one Runtime Agent process.

The template is a Contractor contract, not an ADK object and not a running
Agent.

## Minimal contract

```python
class AgentTemplateRef(BaseModel):
    template_id: str
    version: str
    digest: str


class WorkerRuntimeRef(BaseModel):
    runtime_id: str
    version: str


class ModelPolicyRef(BaseModel):
    policy_id: str
    version: str
    digest: str


class ModelPolicy(BaseModel):
    ref: ModelPolicyRef
    model: str
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    max_model_calls: int | None = None
    max_tool_calls: int | None = None
    max_worker_calls: int | None = None
    max_total_tokens: int | None = None
    temperature: float | None = None


class LLMGatewayConfigRef(BaseModel):
    gateway_id: str
    version: str
    digest: str


class LLMCredentialRef(BaseModel):
    credential_id: str


class ToolsetRef(BaseModel):
    toolset_id: str
    version: str


class ToolsetSelection(BaseModel):
    ref: ToolsetRef
    tools: frozenset[str]


class SandboxProfileRef(BaseModel):
    sandbox_profile_id: str
    version: str


class WorkerSummarizerConfig(BaseModel):
    model_policy: ModelPolicyRef
    context_window_ratio: float = 0.9
    cumulative_budget: int | None = None


class AgentTemplate(BaseModel):
    ref: AgentTemplateRef
    description: str
    runtime: WorkerRuntimeRef
    instructions: ResolvedInstructions
    skills: list[ArtifactRef] = Field(default_factory=list)
    model_policy: ModelPolicyRef
    summarizer: WorkerSummarizerConfig | None = None
    toolsets: list[ToolsetSelection]
    sandbox_profile: SandboxProfileRef
```

The complete first-slice AgentTemplate YAML is:

```yaml
apiVersion: contractor/v1alpha1
kind: AgentTemplate

metadata:
  name: oas_builder
  version: "2"

spec:
  description: Builds and updates an OpenAPI description
  runtime: adk@1
  instructions:
    ref: instructions/oas-worker.md
  modelPolicy: code-analysis@1
  toolsets:
    - ref: run-artifacts@1
      tools:
        - list_artifacts
        - read_artifact
        - write_artifact
    - ref: openapi@1
      tools:
        - read_operation
        - update_schema
        - validate_openapi
  sandboxProfile: local-workdir@1
```

The document follows the shared configuration-file and envelope contract in
[00](00-workflow-and-planner.md). `metadata` contains exactly `name` and
`version`; their pair is the AgentTemplateCatalog lookup key regardless of the
file name. `spec` contains the six mandatory fields and optional `skills` and
`summarizer` fields.
`description` is a mandatory non-empty purpose string for human-facing
configuration and for the deterministic `router@1` agent roster. `runtime`,
`modelPolicy` and `sandboxProfile` are mandatory exact selectors;
`instructions` contains exactly the mandatory `ref`; `skills` is the optional
versionless ArtifactRef set defined by [09](09-agent-skills.md), with omission
equivalent to an empty set; `summarizer` is the optional one-shot terminal
summarization policy defined by [15](15-worker-summarization.md); and
`toolsets` is the mandatory, possibly empty selection list defined below.
Unknown fields and duplicate YAML mapping keys are invalid.

In AgentTemplate YAML, the mandatory Worker instructions use the same
configuration-root-relative text-resource ref contract as Stage Planner
instructions:

```yaml
spec:
  instructions:
    ref: instructions/oas-worker.md
```

AgentTemplateCatalog resolves the UTF-8 text at load time. The normalized ref,
content digest and resolved text form `ResolvedInstructions`; its resolved
content contributes to the enclosing AgentTemplate digest and is sent inside
the complete template in AllocationSpec. Runtime Agent therefore requires no
catalog or filesystem access. ModelPolicy, Toolset selections and
SandboxProfile are framework-neutral Contractor data; an ADK Worker adapter
translates them into runtime configuration only when an allocation is
prepared.

`skills` contains sorted unique logical Agent Skill ArtifactRefs. Namespace is
exactly `skills`, revision is omitted, and the owner UserScope is supplied by
the Run rather than encoded in the ref. The normalized `(namespace, name)`
pairs contribute to AgentTemplate digest. Their current exact revisions are
resolved and pinned only while WorkflowRun initializes, so configuration load
validates syntax but does not require current packages. Ordinary Artifact
update, Run fork, AllocationSpec and native ADK behavior belong to
[09](09-agent-skills.md). One AgentTemplate contains at most 32 refs; one Run
snapshot contains at most 128 distinct refs.

```yaml
spec:
  skills:
    - namespace: skills
      name: likec4
```

### ModelPolicy

For `runtime: adk@1`, `spec.modelPolicy` is a mandatory exact selector using the
shared `<id>@<version>` grammar:

```yaml
spec:
  modelPolicy: code-analysis@1
```

ModelPolicyCatalog resolves it before the AgentTemplate is accepted. The
resolved `ModelPolicyRef(policy_id, version, digest)` and complete validated
policy body are embedded in the resolved AgentTemplate and AllocationSpec;
Runtime Agent performs no policy-catalog lookup. Its exact ref contributes to
the enclosing AgentTemplate digest.

ModelPolicy is the shared, immutable model-loop policy for both Planner and
Worker consumers. It owns the logical LLM Gateway model alias and the portable
generation or cumulative execution-safety fields used by that consumer. These
bounds stop one Planner or Worker model loop deterministically; they are not a
Gateway-wide spend, rate or account quota. ModelPolicy does not own the Gateway
endpoint, provider credentials or provider routing. Planner never inherits a
Worker's policy implicitly: `executionConfig` selects its own exact ModelPolicy
when the chosen Planner implementation uses an LLM.

The complete first-slice ModelPolicy YAML is:

```yaml
apiVersion: contractor/v1alpha1
kind: ModelPolicy

metadata:
  name: code-analysis
  version: "1"

spec:
  model: code-analysis
  maxOutputTokens: 16000
  maxModelCalls: 24
  maxToolCalls: 96
  maxTotalTokens: 250000
  temperature: 0.1
```

The same kind can describe a Streamline or Router Planner without inventing a
Worker tool limit:

```yaml
apiVersion: contractor/v1alpha1
kind: ModelPolicy

metadata:
  name: planner-strong
  version: "1"

spec:
  model: planning-strong
  maxOutputTokens: 16000
  maxModelCalls: 32
  maxWorkerCalls: 64
  maxTotalTokens: 500000
  temperature: 0.1
```

`model` is a mandatory non-empty opaque Gateway alias; Contractor does not
parse it as a provider/model pair. Every other field is structurally optional:
a policy omits a limit or parameter that its intended consumer does not use.
Every present numeric limit is a positive integer. `contextWindowTokens` pins
the model route's advertised input-plus-output context capacity;
`maxOutputTokens` caps one model response and must be smaller than that capacity
when both are present; `maxModelCalls`, `maxToolCalls`, `maxWorkerCalls`, and
`maxTotalTokens` are cumulative across one complete Planner or Worker
invocation. `maxModelCalls` is bounded by 1,000, `maxToolCalls` and
`maxWorkerCalls` by 10,000, and `maxTotalTokens` by 100,000,000. These maxima
reject configuration mistakes and are not recommended operating values.
`contextWindowTokens` is likewise bounded by 100,000,000. Contractor treats it
as pinned operator metadata and does not ask the provider to discover it at
Runtime.

Optional in the shared schema does not mean unbounded by default. Each model
consumer declares the fields it requires and configuration resolution fails
before Run execution if the selected policy is incompatible:

- an `adk@1` Worker requires `maxOutputTokens`, `maxModelCalls` and
  `maxTotalTokens`; it additionally requires `maxToolCalls` when the resolved
  AgentTemplate exposes any model-visible Contractor tool or Agent Skill and
  does not use `maxWorkerCalls`. `maxModelCalls` and `maxTotalTokens` include
  the mandatory one-shot result-finalizer call after every ordinary terminal
  Worker response. The explicitly pinned Audit-check strategy in
  [25](25-audit-worker-finalization.md) instead spends model/tool budget only on
  main turns and bounded completion reminders; its programmatic ZIP finalizer
  makes no model call. That strategy is not inferred from AgentTemplate tools;
- a Worker whose AgentTemplate enables terminal summarization additionally
  requires `contextWindowTokens` so Runtime can derive its soft context
  boundary;
- an optional Worker summarizer policy requires `contextWindowTokens` and
  `maxOutputTokens`, requires `maxModelCalls: 1`, permits neither
  `maxToolCalls` nor `maxWorkerCalls`, and is invoked at most once. Its
  `maxTotalTokens` remains optional and, when present, is checked against the
  provider usage of that single call;
- a `streamline@1` or `router@1` Planner requires `maxOutputTokens`,
  `maxModelCalls`, `maxWorkerCalls` and `maxTotalTokens`; it does not use
  `maxToolCalls` in the first UI/configuration slice;
- `passthrough@1` does not use an LLM and therefore has no Planner ModelPolicy
  selection.

This consumer validation lets one kind represent both roles without inventing
zero values or silently treating an omitted safety limit as infinity.

Worker Runtime checks call and tool capacity before starting the next
operation. It adds provider-reported `total_token_count` after each completed
model response;
crossing the token ceiling stops the loop before a response tool call is
executed, while reaching the ceiling permits a valid structured semantic result
but no later operation. Missing token usage is recorded explicitly and never
disables the independent model/tool-call ceilings. Exhaustion makes Runtime
construct one safe `WorkerFailure` with code `worker_budget_exhausted`, the
exhausted dimension in bounded metrics, and `retryable: true`. The model does
not serialize that failure. Passthrough therefore supplies a normal failed
candidate to Scheduler, whose Workflow transition owns whole-Stage retry.

`temperature` is optional; when absent, Worker omits the parameter instead of
inventing a default. When present, it is a finite JSON number greater than or
equal to zero. Gateway remains responsible for whether that value and the
per-response output limit are supported by the selected route.

The `spec` object contains only `model` and the seven optional portable fields
shown by the model above; arbitrary provider-specific parameters are invalid.
ModelPolicy contains no URL, token, retry policy, request timeout, Stage
deadline, organization quota, or pricing configuration. Those concerns remain
in the selected LLMGatewayConfig, RuntimeSettings or the enclosing execution
contract.

ModelPolicyCatalog validates and normalizes the document, then computes
`ModelPolicyRef.digest` as SHA-256 over its RFC 8785 JCS manifest using the same
`sha256:<64 lowercase hex>` representation as AgentTemplate. Runtime Agent
verifies the resolved policy body against that digest before configuring the
Worker model client. Future schema versions may add explicit portable fields
without changing the meaning of `contractor/v1alpha1`.

### LLMGatewayConfig and credentials

`LLMGatewayConfig` is an immutable, published description of one
OpenAI-compatible Gateway endpoint. It is deliberately separate from
ModelPolicy so the same execution bounds/model alias can use different
endpoints or tokens and the same endpoint can serve multiple policies.

```yaml
apiVersion: contractor/v1alpha1
kind: LLMGatewayConfig

metadata:
  name: local-litellm
  version: "1"

spec:
  protocol: openai-compatible@1
  url: http://127.0.0.1:4000/v1
  credentialManager:
    implementation: litellm-virtual-keys@1
    managementUrl: http://127.0.0.1:4000
```

`protocol` and `url` are mandatory. The first slice accepts exactly
`openai-compatible@1`. URL userinfo, query and fragment components are invalid;
credentials never travel inside the URL or Gateway manifest. A modeled
consumer selects an optional credential independently beside `llmGateway` in
executionConfig. Omitting it means an unauthenticated local Gateway request.

`credentialManager` is optional. When present, it contains exactly
`implementation` and `managementUrl`; the first slice accepts only
`litellm-virtual-keys@1`. Its management URL is an absolute HTTPS origin with
no userinfo, query, fragment or non-root path; HTTP is accepted only for a
loopback IP origin in the single-VM first slice. It is used only by Server
Operations, never by Planner or Runtime Agent. Absence means that the Gateway
can still be selected for unauthenticated execution, but Contractor cannot
create a credential for it. The declaration contains no LiteLLM master key:
Server deployment separately binds the exact digest-bearing Gateway ref to an
operator-owned admin-key file as specified by [06](06-server-ui-and-operations.md).

Resolving model access for a Run produces an exact `LLMGatewayConfigRef` and,
when selected, an `LLMCredentialRef(credential_id)` bound to that Gateway. A
credential is immutable: its token is never replaced in place. The immutable
Run snapshot stores those non-secret refs but never the token. Replacing a key
means creating another credential ID and selecting it in executionConfig;
changing URL or protocol creates a new LLMGatewayConfig version. An active
allocation continues with the exact in-memory RuntimeSettings snapshot it
received.

LLMGatewayConfig manifests live under `configs/llm-gateways/` in either
configured root. Configuration loading validates their exact schema and digest
before resolving `executionConfig`. Operations publishes new immutable versions
only into the managed YAML root according to [06](06-server-ui-and-operations.md).
Encrypted PostgreSQL credentials and the external bootstrap master-key file are
also specified there and remain separate from this manifest contract.

### Toolset and tool selection

A Toolset is a versioned group of model-visible tools implemented by the
selected Worker runtime. AgentTemplate chooses both the exact group and an
explicit allowlist within it:

```yaml
spec:
  toolsets:
    - ref: run-artifacts@1
      tools:
        - list_artifacts
        - read_artifact
    - ref: source-analysis@1
      tools:
        - read_source
        - search_symbols
    - ref: openapi@1
      tools:
        - read_operation
        - update_schema
        - validate_openapi
```

`toolsets` is a mandatory list but may be empty. Every entry contains exactly a
versioned `ref` and a non-empty `tools` list. Tool names are case-sensitive,
must be unique within that entry and must be exported by the selected Toolset
version. A Toolset ref may appear only once in an AgentTemplate. List order has
no meaning: normalization sorts selections by exact ref and sorts tool names
before computing the AgentTemplate digest.

There is no wildcard, omitted-list-means-all behavior or exclusion mode in
`v1alpha1`. To expose every tool, the author explicitly lists every name. This
prevents a newly added Toolset function from becoming model-visible to an
existing AgentTemplate without a template change and new digest. Omitting a
Toolset exposes none of its tools.

The native ADK Agent Skill functions are the one bounded first-class exception
to this generic Toolset rule. A non-empty `skills` field constructs ADK's
standard SkillToolset under [09](09-agent-skills.md); it does not select a
Contractor Toolset ref and cannot add any domain tool. The initial accepted
packages contain no scripts or ADK metadata capable of extending the tool
surface. This exception is explicit in the AgentTemplate body rather than
being inferred from Runtime installation or a label. When `skills` is
non-empty, AgentTemplateCatalog rejects any selected Contractor Toolset
operation named `list_skills`, `load_skill` or `load_skill_resource`; the
model-visible function namespace remains unambiguous. Contractor filters ADK's
`run_skill_script` and its script-bearing prompt under [09], so that function
is not part of the reserved visible set.
It also validates the Worker ModelPolicy as tool-using even when the ordinary
`toolsets` list is empty. Native Skill calls consume the same `maxToolCalls`
counter as Contractor Toolset calls.

`run-artifacts@1` is the built-in current-Run Artifact Toolset for the first
slice. It exports exactly `list_artifacts`, `read_artifact` and
`write_artifact`. These tools are not injected implicitly: an AgentTemplate
must select the Toolset and explicitly list each operation it wants the model
to see. For example, selecting only `list_artifacts` and `read_artifact`
creates a model-visible read-only artifact interface; omitting
`run-artifacts@1` exposes no generic Artifact tools.

`memory-tools@1` is the built-in shared-note Toolset. It exports exactly
`list_memories`, `read_memory`, `write_memory`, `append_memory`,
`search_memory` and `list_memory_tags`. An AgentTemplate selects any exact
non-empty subset in the same way as every other Toolset. Its Worker tools are
bound to the allocation's resolved Agent Namespace; its model receives no Run,
Namespace, ArtifactRef or revision argument. Model-backed Planner mirroring,
the Router `worker_name` schema, note limits and the artifact wrapper are owned
solely by [08](08-memory-tools.md).

The optional project-filesystem Toolsets are `filesystem@1`, `edit-files@1`
and `workspace-changes@1`. They follow the same explicit operation allowlist;
none is injected by a SandboxProfile or instruction. Workflow Stage owns the
logical Run-artifact sources and `direct`/`overlay` semantics; Runtime startup
owns private `local`/`memory` storage under
[10](10-runtime-filesystems-and-edit-tools.md). AgentTemplate selects only the
model-visible operations. It cannot name an artifact, host path or backend,
change workspace mode, or request persistence outside declared Stage result
slots.

`code-analysis@1` follows the same workspace and allowlist rule. Its shallow
`search_def`/`list_symbols` operations may be reported by local or memory
Runtime Agents, while its Trailmark graph operations are a positive local-only
capability. A template names the exact operations it needs; Scheduler never
silently substitutes shallow analysis for a selected graph operation. The
complete snapshot, child-process, bound and lifecycle contract is owned by
[12](12-code-analysis-tools.md).

`http-tools@1` and `caido@1` use the same exact allowlist rule. A Runtime label
may configure HTTP routing or a Caido endpoint/credential, but labels do not
add either Toolset. `caido@1` requires the private `caido-graphql@1` adapter
resolved for the allocation. The complete transport and bounded operation
contract is owned by [11](11-http-and-caido-tools.md).

Tool selection controls model-visible interface construction, not
authorization. Selecting `write_artifact` cannot broaden the allocation's
Server-side grant, bypass the `outputs` Namespace reservation or cross the
write fence established when the StageExecution enters `finalizing` or
`aborting`. Conversely, a grant does not make a tool model-visible unless the
AgentTemplate selects it. Domain Toolset implementations may use the private
Artifact client internally within their own contract without exposing the
generic `run-artifacts@1` operations.

Server has a lightweight Toolset descriptor registry for authoring-time
validation. Each Runtime Agent has its own installed and enabled
ToolsetFactory registry, then derives a positive capability snapshot from that
registry through the startup probes defined by [02](02-runtime-and-a2a.md).
Installed code alone is not a capability: the Runtime Agent advertises a
Toolset ref only when at least one exported tool can honor its complete
contract in that process environment.

Each exported-tool descriptor also declares a fixed subset of infrastructure
channels it consumes: `runtime-http-client` and/or
`runtime-subprocess-launcher` and/or `caido-graphql-client`. This is registered
code metadata under the exact Toolset version, not an AgentTemplate option. A
selected tool declaring a channel uses only the allocation-owned handle
supplied by Runtime; a tool declaring none remains local/artifact-only. Generic
HTTP may use its optional proxy channel or a bounded direct client; Caido's
channel is mandatory. HTTP-proxy targeting is owned by [07], and the HTTP/Caido
tool boundary by [11](11-http-and-caido-tools.md).

A Toolset capability contains the exact ref and the subset of its exported
tools whose prerequisites passed. This deliberately has the same granularity
as `ToolsetSelection`. For example, an environment without the LikeC4
executable may advertise the editing operations of `likec4@1` but omit
`validate_likec4` when only that operation depends on the executable. A
template that selects `validate_likec4` cannot be placed there, while a
template selecting only the advertised editing operations can.

AgentTemplateCatalog rejects unknown refs, unknown selected tool names and
cross-Toolset collisions in final model-visible tool names independently of
the currently connected fleet. Runtime Agent validates every AllocationSpec
against its immutable startup capability snapshot, then asks each factory to
construct only the selected tools with the allocation's Namespace, Artifact
client, sandbox and safe RuntimeSettings. Unselected tools are neither
constructed nor exposed.

Toolset refs identify registered code; Workflow and AgentTemplate cannot name a
Python module, callable, executable or shell command. Internal helpers used by
a selected tool are not themselves model-visible tools and do not need to
appear in the allowlist.

### SandboxProfile

`spec.sandboxProfile` is a mandatory exact selector using the shared
`<id>@<version>` grammar:

```yaml
spec:
  sandboxProfile: local-workdir@1
```

A SandboxProfile identifies registered Runtime Agent behavior, not an
arbitrary mapping of host settings. Its exact ID and version form
`SandboxProfileRef`, contribute to the AgentTemplate digest and travel in the
resolved AgentTemplate inside AllocationSpec. Server validates the ref against
its SandboxProfile descriptor registry; Runtime Agent must have the matching
implementation and rejects allocation preparation if it does not. A template
cannot provide a host path, container image, command, mount or implementation
module through this field.

The first slice defines exactly one built-in profile: `local-workdir@1`. It:

- creates a new empty directory owned by one allocation before constructing
  the selected Toolsets;
- chooses that directory internally below the operator-configured Runtime Agent
  work root and never serializes its host path into AgentTemplate,
  StageExecution, A2A messages or ArtifactRefs;
- provides an allocation-local workspace handle only to runtime and Toolset
  factories that need it, and never reuses the directory for another
  allocation;
- keeps the directory until release, then acknowledges release only after its
  contents are removed; cleanup failure leaves the Runtime Agent fenced and
  unavailable for placement so that the slot cannot reuse dirty state;
- removes recognized orphan allocation directories below its dedicated work
  root on Runtime Agent startup before the slot can report `idle`.

Despite the type name, `local-workdir@1` is lifecycle isolation, not an OS
security boundary. Worker and tools still run in the Runtime Agent process with
that process's filesystem, network and subprocess privileges. The profile
promises no container, mount, syscall, process or network isolation. Selected
Toolsets remain responsible for safe input handling, while stronger future
profiles require new exact refs rather than changing `local-workdir@1`
semantics.

The implemented extension [21](21-podman-sandbox.md) provides `podman@1` and the
explicit `code-execution@1` Toolset. It requires a local direct project
workspace and cross-validates profile/tool compatibility. Runtime advertises
both capabilities only after an opt-in real startup probe and confirmed cleanup;
registering authoring refs alone does not make a Runtime eligible.

An AgentTemplate body is immutable. `template_id + version + digest` identifies
that exact body, including its logical skill ArtifactRef set and optional
summarizer policy. Workflow authoring
uses the exact `<id>@<version>` selector defined
in [00](00-workflow-and-planner.md); `latest`, ranges and unversioned aliases are
invalid. Any template instruction, policy, summarizer, tool or normalized
logical skill-ref set change creates a new version and digest. Omitted `skills` and explicit
`skills: []` are the same empty body. Writing a new current UserScope artifact
revision for an already selected logical ref is deliberately resolved as
separate Run-pinned behavior under [09](09-agent-skills.md) and does not mutate
this body.

### Canonical digest

`AgentTemplateRef.digest` is encoded as `sha256:<64 lowercase hex>` and is
computed from a resolved manifest, not from the source YAML bytes:

```text
sha256(JCS(normalized resolved AgentTemplate manifest))
```

The manifest is an I-JSON object containing the normalized `apiVersion`, `kind`,
template name and version, plus the canonical `spec` representation below.
Resolved Worker instructions appear as their normalized
configuration-root-relative `ref` and
exact instruction `digest`; their full text is carried beside the manifest but
need not be duplicated inside it. A non-empty `skills` set appears as sorted
logical `{namespace,name}` refs with no revision; the property is omitted for
the empty set. An optional summarizer appears with its exact ModelPolicy ref,
normalized context-window ratio and optional cumulative budget. Every model,
toolset and sandbox reference appears in its normalized exact form. The computed AgentTemplate digest itself is the only
AgentTemplate field excluded from the input.

The manifest is serialized with the JSON Canonicalization Scheme from
[RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html), then its canonical UTF-8
bytes are hashed with SHA-256. Object property order is canonical; array order
is preserved unless a field such as `toolsets` or `skills` is explicitly
defined as an unordered set and sorted during normalization; Unicode strings are not
normalized. YAML comments, mapping order, anchors, aliases and scalar
presentation therefore do not independently affect identity after parsing and
validation. Changing any normalized value or the bytes of the resolved
instruction resource changes the digest. This empty-set normalization preserves
the digest of an existing pre-Agent-Skills template.

AgentTemplateCatalog computes and stores the digest with the resolved template.
AllocationSpec carries the normalized manifest, resolved instruction text,
resolved default ModelPolicy dependency, optional summarizer ModelPolicy and
the separately selected effective Worker ModelPolicy body with their exact
refs. Before creating Worker, Runtime Agent verifies the instruction, every
policy dependency and enclosing AgentTemplate digests. A mismatch fails
preparation with non-retryable
`template_digest_mismatch`; Runtime Agent never silently recomputes a new
identity or fetches replacement configuration content.

The digest is an integrity and identity fingerprint, not an authorization
signature. Trust still comes from the configured catalog boundary and the mTLS
control channel.

## What the template does not contain

An AgentTemplate contains no:

- Workflow, Stage objective/instructions, Stage inputs or product success
  criteria;
- Planner strategy or Planner agent tree;
- physical Runtime Agent, allocation, endpoint or Agent Card;
- process ID, host path, Python class/module or executable;
- project filesystem backend, mount source or host-write switch;
- credential or provider secret;
- retry, deadline or Stage budget state;
- Run-specific Namespace or artifact grant.

Workflow owns product semantics. A Stage binding owns its logical name and
Namespace. Control Plane owns placement and sends the resolved template to one
free Runtime Agent. Allocation identity and scoped access context are generated
for that assignment.

`AgentTemplate.instructions` defines reusable Worker behavior. It is distinct
from a Workflow Stage's `objective` and `instructions`, which tell Planner what
that particular Stage must accomplish and how to orchestrate its prepared
Workers. Resolving a Stage never concatenates those fields into or mutates the
stored AgentTemplate.

The template selects an in-process Worker runtime by stable ref but does not
contain host-specific process details. Runtime Agents may run the same
Contractor code in different immutable process environments and therefore
advertise different runtime, Toolset/tool, SandboxProfile and RuntimeAdapter
capability snapshots. AgentTemplate expresses semantic requirements only; it
never names an environment or physical process. Infrastructure-channel
metadata belongs to the exact registered Toolset descriptor and does not add
fields to the AgentTemplate manifest.

## Stage binding

Each Stage uses a non-empty mapping from logical Agent name to an object-valued
binding:

```yaml
agents:
  oas_builder:
    template: oas_builder@2
    namespace: oas
  reviewer:
    template: oas_reviewer@1
```

The authoring schema has no list or scalar shorthand. The YAML mapping key
becomes `name` in the normalized internal binding, while the versioned
`template` selector is resolved to an exact ref:

```python
class StageAgentBinding(BaseModel):
    name: AgentName
    template: AgentTemplateRef
    namespace: NamespaceId | None = None
```

The mapping must contain at least one binding and its order has no execution
semantics. `namespace` defaults to `name`. Names are unique by construction;
WorkflowCatalog also rejects duplicate YAML keys before ordinary schema
decoding. Specialized namespace-bound toolsets must resolve to unique
Namespaces within that Stage; a later Stage may deliberately reuse a Namespace
with another binding. The Run-reserved `inputs` and `outputs` Namespaces cannot
be used as an Agent Namespace; their access rules are owned by
[03](03-artifact-plane.md).

Before a Run starts, Workflow Scheduler resolves every authoring selector
through `AgentTemplateCatalog` to the exact digest-bearing ref and validated
template body stored with the Run snapshot. Accepted execution state never
retains an unresolved selector. When a StageExecution is created, its template
comes from that snapshot rather than a fresh catalog lookup.

## Resolution and pinning

The resolution path is deliberately short:

```text
StageAgentBinding
  -> AgentTemplateCatalog: resolve exact id/version/digest
  -> Run owner UserScope: resolve/fork each logical Skill ArtifactRef exactly
  -> WorkflowRun snapshot retains the resolved template and skill dependencies
  -> Workflow Scheduler records it for the StageExecution
  -> Control Plane builds AllocationSpec
  -> Runtime Agent creates one in-process Worker instance from that template
```

After Stage preparation begins, Planner, Control Plane recovery, Runtime Agent
and Worker do not consult a mutable template alias. Changing or removing the
catalog entry cannot retarget an already prepared or recorded StageExecution.
Likewise, they never re-resolve a current owner skill binding: exact skill
artifacts come from the immutable WorkflowRun snapshot defined by [09].

`AgentTemplateCatalog` is an in-process Server configuration boundary in the
single-VM baseline. It loads through the shared all-or-nothing configuration
root defined in [00](00-workflow-and-planner.md); it is not a new network
service, Kubernetes CRD or runtime service locator.

## Worker runtime and allocation

`AgentTemplate` is the complete declarative definition selected by Workflow.
Its `runtime` field chooses the generic code that can execute that definition;
the first supported value is `adk@1`. Many different AgentTemplates may use the
same runtime. A running Worker is an ephemeral in-process instance created from
one template for one allocation; it is not another service or operating-system
process.

There is no separate domain `WorkerImplementation` or
`WorkerImplementationCatalog`, and Control Plane does not select an executable
or image. Runtime Agent code contains the supported in-process runtime factories,
conceptually:

```text
adk@1 -> AdkWorkerRuntimeFactory
```

Control Plane verifies that the deployment supports the template's runtime ref
and places that ref directly in `AllocationSpec`. The selected Runtime Agent
validates it against its own build and creates the Worker instance. Workflow
YAML, AgentTemplate, Planner and public API input cannot supply executable
paths, modules, images or raw process arguments.

Conceptually, the complete assignment is:

```text
AllocationSpec
  globally unique allocation_id
  run and stage identity
  logical Agent name and resolved Namespace
  complete AgentTemplate + exact ref
  exact resolvedSkills refs and package digests selected by WorkflowRun
  effective ModelPolicy + exact ref
  exact WorkerRuntimeRef
  RuntimeSettings supplied by Control Plane
  exact lease_expires_at, deadline and resource limits
```

`lease_expires_at` is copied from Control Plane's confirmed allocation lease;
Runtime Agent does not derive a later value from its local heartbeat settings.
Private wire timestamps are normalized to UTC microsecond precision, which is
preserved exactly by Go, Python and PostgreSQL.

`RuntimeSettings` contains resolved connection and execution settings, not
Worker semantics. The initial settings include at least:

```text
RuntimeSettings
  llm_gateway_url
  llm_gateway_token               secret, preferably allocation-scoped
  artifact_api_url
  typed allocation adapter settings (telemetry / HTTP proxy when selected)
  request timeouts and size/resource limits
```

`RuntimeSettings.requestTimeoutSeconds` is the allocation-local outbound
request budget used by Worker model and tool clients. The Server rejects an
operator value below 120 seconds and uses 180 seconds by default. This budget
is independent of the shorter Control Plane-to-Runtime management request
timeout and of the enclosing Stage/Planner deadline; increasing it must not
make terminal cleanup or Scheduler claim progress wait for a model-sized
timeout.

AgentTemplate selects its default exact ModelPolicy, but neither object carries
the LLM Gateway URL, token, provider routing or credential. Run initialization
resolves the Workflow defaults plus the request's reference-only
`executionConfig` overrides. Control Plane obtains the selected URL from the
exact LLMGatewayConfig and the pinned credential from the trusted
Server secret store, then delivers their values over the private mTLS control
channel. When executionConfig deliberately omits the Worker Gateway, Control
Plane may instead complete it from pinned Run-selected Runtime labels or
allocation-time Agent Runtime labels; those higher infrastructure layers may
also override an explicitly
selected physical route without changing ModelPolicy. The exact merge, adapter
and secret lifecycle rules are owned by
[07](07-runtime-labels-and-infrastructure-config.md). `LLM Gateway` is the
Contractor role: LiteLLM is the initial backend,
but another backend may replace it when it satisfies the configured
model-client protocol. An active allocation uses one resolved settings
snapshot; configuration changes or label rebinding never mutate it.

AgentTemplate remains the sole authored reusable Worker-behavior contract. The
Runtime Agent allocation layer consumes the resolved template and builds a minimal
Worker context containing description, instruction text, card version, effective
model policy, optional resolved summarizer policy, instantiated tools and
prepared skills. `WorkerBuildContext` does not
contain an `AgentTemplate`; neither the ADK implementation nor its model sees a
template ref, digest, manifest or allocation projection. Stage objective/instructions
remain separate per-task semantic input under [00]. Its explicit Toolset selections
own domain tools and its explicit `skills` refs
own the bounded native ADK SkillToolset described by [09]. A Runtime label may
configure a proxy, telemetry exporter or an already selected Toolset adapter,
but cannot add a tool, Toolset, skill, instruction or Worker behavior. Runtime
Agent receives typed settings and never interprets label names.

Durable provenance records the exact AgentTemplate, Run-pinned Agent Skill
artifact refs/digests, effective ModelPolicy, LLMGatewayConfig and non-secret
credential refs, WorkerRuntimeRef, allocation ID and Runtime Agent process
identity. It never records ephemeral access material or secret-bearing
RuntimeSettings values.

The PostgreSQL connection URL is Server bootstrap configuration. It never
appears in AgentTemplate, AllocationSpec or Runtime Agent configuration supplied
by Control Plane. Artifact access uses the allocation-bound Server API; model
access uses the supplied LLM Gateway settings. Runtime Agent receives no database
or S3 credential.

Runtime Agent control code validates the AllocationSpec but does not interpret
the template's domain instructions. Its selected in-process Worker runtime
consumes those instructions and configures the same process's A2A Server; an
incompatible runtime/card fails preparation before Planner starts.

## Invariants

1. Workflow Stage binding selects the AgentTemplate, Workflow Scheduler resolves
   its exact body and Planner cannot change it.
2. One Stage binding produces one allocation and one in-process Worker instance
   in one Runtime Agent.
3. The complete template is resolved before capacity preparation and is passed
   in AllocationSpec rather than fetched by Worker at task time.
4. ExecutionConfig may select another compatible published ModelPolicy without
   mutating AgentTemplate; AllocationSpec carries that effective exact policy
   separately and Runtime verifies its digest.
5. WorkerHandle reports the exact template ref, runtime ref and allocation
   identity returned by the prepared Runtime Agent.
6. An Agent Card incompatible with the required A2A protocol fails preparation
   before Planner starts.
7. AgentTemplate never becomes a physical deployment or long-lived Agent
   identity.
8. Runtime Agent and Worker functions execute in one process; one Runtime Agent
   has at most one active Worker instance.
9. Workflow/API input cannot select an executable, module, image or raw process
   arguments.
10. Runtime labels configure allocation infrastructure around the resolved
    template and never change its model-visible instructions or selected tools.
11. AgentTemplate declares versionless Agent Skill ArtifactRefs; WorkflowRun,
    not Runtime, resolves them in owner UserScope and pins exact immutable
    revisions under [09].
