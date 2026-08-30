# 01 — AgentTemplate

Status: **Working agreement**

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
    max_output_tokens: int
    max_model_calls: int
    max_tool_calls: int
    max_total_tokens: int
    temperature: float | None = None


class ToolsetRef(BaseModel):
    toolset_id: str
    version: str


class ToolsetSelection(BaseModel):
    ref: ToolsetRef
    tools: frozenset[str]


class SandboxProfileRef(BaseModel):
    sandbox_profile_id: str
    version: str


class AgentTemplate(BaseModel):
    ref: AgentTemplateRef
    description: str
    runtime: WorkerRuntimeRef
    instructions: ResolvedInstructions
    model_policy: ModelPolicyRef
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
file name. `spec` contains exactly the six fields shown above.
`description` is a mandatory non-empty informational string. `runtime`,
`modelPolicy` and `sandboxProfile` are mandatory exact selectors;
`instructions` contains exactly the mandatory `ref`; and `toolsets` is the
mandatory, possibly empty selection list defined below. Unknown fields and
duplicate YAML mapping keys are invalid.

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

### Worker ModelPolicy

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

ModelPolicy owns the logical LLM Gateway model alias and bounded generation
settings, not the Gateway endpoint, provider credentials or provider routing.
Changing the deployment route behind the same model alias does not change the
policy or AgentTemplate digest. Planner model selection is separate and is
never inherited from a Worker AgentTemplate.

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

`model` is a mandatory non-empty opaque Gateway alias; Contractor does not
parse it as a provider/model pair. `maxOutputTokens` is a mandatory positive
integer and caps the generated output of each individual Worker LLM call.
`maxModelCalls`, `maxToolCalls`, and `maxTotalTokens` are mandatory positive
integer ceilings for one Worker A2A invocation. They are cumulative across the
complete ADK tool loop, including the optional tool-free result-finalization
call. The first two fields are bounded by 1,000 and 10,000 respectively;
`maxTotalTokens` is bounded by 100,000,000. These maxima reject configuration
mistakes and are not recommended operating values.

Runtime checks call and tool capacity before starting the next operation. It
adds provider-reported `total_token_count` after each completed model response;
crossing the token ceiling stops the loop before a response tool call is
executed, while reaching the ceiling permits a final text result but no later
operation. Missing token usage is recorded explicitly and never disables the
independent model/tool-call ceilings. Exhaustion returns one safe failed
`StageContentResult` with code `worker_budget_exhausted`, the exhausted
dimension in bounded metrics, and `retryable: true`. Passthrough therefore
supplies a normal failed candidate to Scheduler, whose Workflow transition owns
whole-Stage retry.

`temperature` is optional; when absent, Worker omits the parameter instead of
inventing a default. When present, it is a finite JSON number greater than or
equal to zero. Gateway remains responsible for whether that value and the
per-response output limit are supported by the selected route.

The `spec` object contains exactly these six fields in `v1alpha1`; arbitrary
provider-specific parameters are invalid. ModelPolicy contains no URL, token,
retry policy, request timeout, Stage deadline, organization quota, or pricing
configuration. Those concerns remain in RuntimeSettings or the enclosing
execution contract.

ModelPolicyCatalog validates and normalizes the document, then computes
`ModelPolicyRef.digest` as SHA-256 over its RFC 8785 JCS manifest using the same
`sha256:<64 lowercase hex>` representation as AgentTemplate. Runtime Agent
verifies the resolved policy body against that digest before configuring the
Worker model client. Future schema versions may add explicit portable fields
without changing the meaning of `contractor/v1alpha1`.

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

`run-artifacts@1` is the built-in current-Run Artifact Toolset for the first
slice. It exports exactly `list_artifacts`, `read_artifact` and
`write_artifact`. These tools are not injected implicitly: an AgentTemplate
must select the Toolset and explicitly list each operation it wants the model
to see. For example, selecting only `list_artifacts` and `read_artifact`
creates a model-visible read-only artifact interface; omitting
`run-artifacts@1` exposes no generic Artifact tools.

Tool selection controls model-visible interface construction, not
authorization. Selecting `write_artifact` cannot broaden the allocation's
Server-side grant, bypass the `outputs` Namespace reservation or cross the
write fence established when the StageExecution enters `finalizing` or
`aborting`. Conversely, a grant does not make a tool model-visible unless the
AgentTemplate selects it. Domain Toolset implementations may use the private
Artifact client internally within their own contract without exposing the
generic `run-artifacts@1` operations.

The homogeneous deployment provides a lightweight Toolset descriptor registry
to Server and the matching ToolsetFactory implementations to Runtime Agent.
AgentTemplateCatalog rejects unknown refs, unknown selected tool names and
cross-Toolset collisions in final model-visible tool names. Runtime Agent
revalidates the same facts, then asks each factory to construct only the
selected tools with the allocation's Namespace, Artifact client, sandbox and
safe RuntimeSettings. Unselected tools are neither constructed nor exposed.

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

An AgentTemplate is immutable. `template_id + version + digest` identifies one
exact body. Workflow authoring uses the exact `<id>@<version>` selector defined
in [00](00-workflow-and-planner.md); `latest`, ranges and unversioned aliases are
invalid. A behavior or policy change creates a new version and digest.

### Canonical digest

`AgentTemplateRef.digest` is encoded as `sha256:<64 lowercase hex>` and is
computed from a resolved manifest, not from the source YAML bytes:

```text
sha256(JCS(normalized resolved AgentTemplate manifest))
```

The manifest is an I-JSON object containing the normalized `apiVersion`, `kind`,
template name and version, plus every validated `spec` field. Resolved Worker
instructions appear as their normalized configuration-root-relative `ref` and
exact instruction `digest`; their full text is carried beside the manifest but
need not be duplicated inside it. Every model, toolset and sandbox reference
appears in its normalized exact form. The computed AgentTemplate digest itself
is the only AgentTemplate field excluded from the input.

The manifest is serialized with the JSON Canonicalization Scheme from
[RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html), then its canonical UTF-8
bytes are hashed with SHA-256. Object property order is canonical; array order
is preserved unless a field such as `toolsets` is explicitly defined as an
unordered set and sorted during normalization; Unicode strings are not
normalized. YAML comments, mapping order, anchors, aliases and scalar
presentation therefore do not independently affect identity after parsing and
validation. Changing any normalized value or the bytes of the resolved
instruction resource changes the digest.

AgentTemplateCatalog computes and stores the digest with the resolved template.
AllocationSpec carries the normalized manifest, resolved instruction text,
resolved ModelPolicy body and exact refs. Before creating Worker, Runtime Agent
verifies the instruction, ModelPolicy and enclosing AgentTemplate digests. A
mismatch fails preparation with non-retryable `template_digest_mismatch`;
Runtime Agent never silently recomputes a new identity or fetches replacement
catalog content.

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
contain host-specific process details. Every Runtime Agent in the initial
homogeneous fleet runs the same code and supports the same runtime refs.

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
  -> WorkflowRun snapshot retains the resolved template dependency
  -> Workflow Scheduler records it for the StageExecution
  -> Control Plane builds AllocationSpec
  -> Runtime Agent creates one in-process Worker instance from that template
```

After Stage preparation begins, Planner, Control Plane recovery, Runtime Agent
and Worker do not consult a mutable template alias. Changing or removing the
catalog entry cannot retarget an already prepared or recorded StageExecution.

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
  exact WorkerRuntimeRef
  RuntimeSettings supplied by Control Plane
  exact lease_expires_at, deadline and resource limits
```

`lease_expires_at` is copied from Control Plane's confirmed allocation lease;
Runtime Agent does not derive a later value from its local heartbeat settings.
Private wire timestamps are normalized to UTC microsecond precision, which is
preserved exactly by Go, Python and PostgreSQL.

`RuntimeSettings` contains deployment-owned connection and execution settings,
not Worker semantics. The initial settings include at least:

```text
RuntimeSettings
  llm_gateway_url
  llm_gateway_token               secret, preferably allocation-scoped
  artifact_api_url
  request timeouts and size/resource limits
```

AgentTemplate selects an exact ModelPolicy, but neither object carries the LLM
Gateway URL, token, provider routing or credential. Control Plane takes those
values from trusted Server configuration (or a future secret provider) and
delivers them over the private mTLS control channel. `LLM Gateway` is the
Contractor role: LiteLLM is the initial deployment backend, but another backend
may replace it when it satisfies the configured model-client protocol. An
active allocation uses one resolved settings snapshot; ordinary configuration
changes apply to the next allocation.

Durable provenance records the exact AgentTemplate ref, WorkerRuntimeRef,
allocation ID and Runtime Agent process identity. It never records ephemeral
access material or secret-bearing RuntimeSettings values.

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
4. WorkerHandle reports the exact template ref, runtime ref and allocation
   identity returned by the prepared Runtime Agent.
5. An Agent Card incompatible with the required A2A protocol fails preparation
   before Planner starts.
6. AgentTemplate never becomes a physical deployment or long-lived Agent
   identity.
7. Runtime Agent and Worker functions execute in one process; one Runtime Agent
   has at most one active Worker instance.
8. Workflow/API input cannot select an executable, module, image or raw process
   arguments.
