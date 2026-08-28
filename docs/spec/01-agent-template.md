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


class AgentTemplate(BaseModel):
    ref: AgentTemplateRef
    description: str
    runtime: WorkerRuntimeRef
    instruction: AgentInstruction | None = None
    model_policy: ModelPolicyRef | None = None
    toolsets: list[ToolsetRef] = Field(default_factory=list)
    sandbox_profile: SandboxProfileRef
```

The exact DTO shapes for instruction and policy refs are intentionally deferred.
Their architectural ownership is not: they are framework-neutral Contractor
data, and an ADK Worker adapter translates them into ADK configuration only
when an allocation is prepared.

An AgentTemplate is immutable. `template_id + version + digest` identifies one
exact body; `latest` is authoring convenience at most and is never part of an
accepted StageExecution. A behavior or policy change creates a new version and
digest.

## What the template does not contain

An AgentTemplate contains no:

- Workflow, Stage inputs or product success criteria;
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

The template selects an in-process Worker runtime by stable ref but does not
contain host-specific process details. Every Runtime Agent in the initial
homogeneous fleet runs the same code and supports the same runtime refs.

## Stage binding

Each Stage Agent entry binds one logical name to one exact template:

```python
class StageAgentBinding(BaseModel):
    name: AgentName
    template: AgentTemplateRef
    namespace: NamespaceId | None = None
```

`namespace` defaults to `name`. Names are unique within the Stage. Specialized
namespace-bound toolsets must also resolve to unique Namespaces within that
Stage; a later Stage may deliberately reuse the Namespace with another binding.
The Run-reserved `inputs` and `outputs` Namespaces cannot be used as an Agent
Namespace; their access rules are owned by [03](03-artifact-plane.md).

A concise Workflow authoring form may use `oas_builder@2`; it means
`name=oas_builder` plus that versioned template. Before a Run starts, the
Workflow Scheduler resolves it through `AgentTemplateCatalog` to the exact
digest. Accepted execution state never retains an unversioned alias.

## Resolution and pinning

The resolution path is deliberately short:

```text
StageAgentBinding
  -> AgentTemplateCatalog: resolve exact id/version/digest
  -> Workflow Scheduler records the complete template for the StageExecution
  -> Control Plane builds AllocationSpec
  -> Runtime Agent creates one in-process Worker instance from that template
```

After Stage preparation begins, Planner, Control Plane recovery, Runtime Agent
and Worker do not consult a mutable template alias. Changing or removing the
catalog entry cannot retarget an already prepared or recorded StageExecution.

`AgentTemplateCatalog` is an in-process Server configuration boundary in the
single-VM baseline. It may load validated files at startup; it is not a new
network service, Kubernetes CRD or runtime service locator.

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
  lease, deadline and resource limits
```

`RuntimeSettings` contains deployment-owned connection and execution settings,
not Worker semantics. The initial settings include at least:

```text
RuntimeSettings
  llm_gateway_url
  llm_gateway_token               secret, preferably allocation-scoped
  artifact_api_url
  request timeouts and size/resource limits
```

AgentTemplate may select a model policy, but it never carries the LLM Gateway
URL or token. Control Plane takes those values from trusted Server configuration
(or a future secret provider) and delivers them over the private mTLS control
channel. `LLM Gateway` is the Contractor role: LiteLLM is the initial deployment
backend, but another backend may replace it when it satisfies the configured
model-client protocol. An active allocation uses one resolved settings snapshot;
ordinary configuration changes apply to the next allocation.

Durable provenance records the exact AgentTemplate ref, WorkerRuntimeRef,
allocation ID and Runtime Agent process identity. It never records ephemeral
access material or secret-bearing RuntimeSettings values.

The PostgreSQL connection URL is Server bootstrap configuration. It never
appears in AgentTemplate, AllocationSpec or Runtime Agent configuration supplied
by Control Plane. Artifact access uses the allocation-bound Server API; model
access uses the supplied LLM Gateway settings. Runtime Agent receives no database
or S3 credential.

Runtime Agent control code validates the AllocationSpec but does not interpret
the template's domain instruction. Its selected in-process Worker runtime
consumes that instruction and configures the same process's A2A Server; an
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
