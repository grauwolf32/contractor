# 00 — Product scope

Status: **Draft**  
Depends on: none

## Product definition

Contractor is an AI-assisted software-understanding and application-security
product. It turns an exact source-project snapshot and an explicit analysis
objective into versioned, inspectable deliverables: API descriptions,
architecture models, source/data-flow traces, vulnerability findings,
verification verdicts and, when explicitly authorized, live-target
exploitability evidence.

A Contractor Run is a finite assessment or artifact-production job. A user or
automation client selects a versioned workflow profile, supplies its exact
inputs and receives typed output artifacts plus status, provenance and usage.
The long-running Server and distributed Agents are the v2 delivery mechanism;
they are not the product purpose. Planner/Worker topology may change without
changing what a workflow promises to consume and produce.

The primary users and jobs are:

- application and API engineers reconstruct or enrich OpenAPI descriptions from
  implementation evidence;
- architects and security reviewers build a security-focused LikeC4 model from
  the source project;
- application-security engineers trace entry points and data flows, discover
  vulnerabilities and independently verify findings;
- authorized security testers assess whether a specific finding is exploitable
  against an explicitly scoped live or replay target;
- CI/release automation reruns pinned workflows and evaluations to compare
  output quality, reliability and resource use.

## Product capability families

| Family | Principal inputs | Product deliverables |
|---|---|---|
| API discovery and enrichment | project snapshot; optional existing OpenAPI artifact | validated OpenAPI artifact, source-coverage/provenance facts and validation results |
| Architecture modelling | project snapshot; optional prior model | validated security-focused LikeC4 model and validation results |
| Operation and data-flow tracing | project snapshot plus OpenAPI operations | trace annotations, isolated source overlay/patch/diff, supported finding records and optional independent verifications |
| Vulnerability discovery and assessment | project snapshot; optional OpenAPI/finding seeds | typed findings, deduplicated/confirmed assessment artifacts and verification verdicts |
| Authorized exploitability assessment | exact finding plus target, credential, egress, operation and time authorization | bounded HTTP/tool evidence and a typed exploitability verdict |
| Prompt-directed specialist analysis | project snapshot, free-form objective and bounded workflow policy | typed artifacts declared by the selected specialist/output contract |

The versioned workflow catalog defines which of these capabilities are enabled
in a deployment and their exact objective/input/output schemas. A model's prose
or a terminal Agent status is not by itself a product deliverable: required
artifacts must satisfy the accepted workflow output contracts. Intermediate
plans, prompts, sessions and telemetry are implementation or diagnostic data
unless an output contract explicitly promotes them.

The product contract is the selected workflow's objective, input/output,
authorization, outcome and provenance semantics. Server/Agent placement, A2A,
Planner strategy, Worker framework, database layout and telemetry are runtime
choices. A Server decomposer and passthrough plus Agent-local ReAct are valid
substitutes only when both satisfy the same pinned product contract; an execution
experiment does not silently redefine the workflow.

## v2 engineering goal

Build a cleaner Contractor runtime in which orchestration, agent execution,
persistence and infrastructure are separated by serializable contracts. The
contracts separate where a task is coordinated from how an Agent decides to
fulfil it.

## Target topology

- one Server process owns the public API, run lifecycle and the selected
  Server-side Planner strategy;
- N Agent processes accept Worker attempts over A2A and own their execution
  logic and sandboxes;
- one PostgreSQL database stores all durable state in logical table groups;
- one external LLM Proxy is used by every model-backed Server/Agent strategy;
- an external OpenTelemetry Collector/backend is optional for sampled traces;
- Server-to-Agent execution uses A2A.

## Planner and Worker composition

The Server supports interchangeable Planner strategies behind the same domain
boundary:

- a **decomposing Planner** maintains the current subtask/DAG state and emits
  one or more typed Task definitions;
- a **static Planner** instantiates a versioned one-or-more-Task template
  manifest against the accepted RunSpec;
- a **passthrough Planner** is a deterministic stub that creates one root Task,
  without decomposing the objective. Authorized exact/capability routing remains
  in RunSpec/TaskSpec policy, not Planner configuration. Passthrough is the
  one-node specialization of static planning; each Attempt of that Task has its
  own immutable `WorkerJob`.

All strategies propose typed TaskSpecs/edges through the same state machine.
The deterministic scheduler commits Attempt state and dispatch through the same
RunState CAS/outbox path. Passthrough does not create a second Planner protocol:
the scheduler materializes its root Task into the same Attempt-specific
`WorkerJob` sent through the same A2A boundary.

A Worker Agent is not assumed to be a passive or atomic executor. Its local
implementation may perform a direct operation, maintain its own subtask state,
or alternate between planning and working (for example, a ReAct loop). This
choice is opaque behind `WorkerJob`/`WorkerResult`; the Agent may use ADK,
another framework or custom code.

## Requirements

- **SCP-001** — Components MUST communicate across process boundaries using
  versioned, serializable DTOs. Python callables, `LlmAgent`, filesystem
  objects and SQLAlchemy sessions MUST NOT cross a process boundary.
- **SCP-002** — The initial release MUST run with one Server, one PostgreSQL
  database and at least one Agent. Adding internal microservices is out of
  scope.
- **SCP-003** — Adding more Agent instances MUST NOT require a Server code
  change or a database schema per Agent.
- **SCP-004** — Execution MUST survive a Server restart from committed state.
  Every Attempt/result boundary MUST be fenced. Automatic retry after an
  in-flight external effect is allowed only when its durable effect protocol
  proves replay safe; result fencing alone is insufficient.
- **SCP-005** — Loss of telemetry MUST NOT change run correctness or block the
  main execution loop indefinitely.
- **SCP-006** — The v2 core MUST NOT import v1 workflow implementations. v1
  behavior is migrated one workflow at a time behind v2 contracts.
- **SCP-007** — Decomposing, static and passthrough Planner strategies MUST use
  the same validated Planner-command, RunState CAS, outbox and A2A dispatch
  path.
- **SCP-008** — One versioned `WorkerJob`/`WorkerResult` profile MUST support
  both a narrow planned subtask and a high-level root objective; it MUST NOT
  prescribe whether the receiving Agent plans, works or alternates between the
  two internally.
- **SCP-009** — Conformance with the Contractor A2A and domain contracts MUST
  NOT require the Worker implementation to use Google ADK.
- **SCP-010** — Contractor MUST support reproducibly specified, rerunnable and
  independently verifiable evaluation of substitutable coordination
  configurations on the same immutable cases. Comparison MUST include workflow-
  specific output quality as well as reliability, latency and resource metrics;
  a result is attributed to the Planner alone only when Worker behavior and all
  other material dimensions are held fixed.
- **SCP-011** — Source projects MUST enter as immutable, checksummed project
  snapshot artifacts. A client, Git importer or future connector may produce a
  snapshot, but a Run MUST NOT depend on a Server-local project path.
- **SCP-012** — Every enabled workflow profile MUST name its product capability,
  objective contract, exact accepted input kinds and required typed output
  contracts. A successful transport or WorkerResult MUST NOT make the Run a
  product success when a required deliverable is absent or invalid.
- **SCP-013** — Every accepted product deliverable MUST be attributable to the
  exact project snapshot, input artifacts, `WorkflowProfileRef`,
  `PlannerStrategyRef`, Worker execution provenance and effective
  model/tool/prompt/sandbox policy versions that produced it. Optional diagnostic
  detail MUST NOT be required to establish that provenance.
- **SCP-014** — Live-target interaction is a distinct, explicitly authorized
  product capability. Source analysis, API/architecture generation, tracing and
  static verification MUST NOT acquire live-target or unrestricted egress
  authority implicitly.
- **SCP-015** — Replacing a Planner strategy, Worker implementation, Agent
  framework or A2A adapter MUST NOT change the semantics of a pinned workflow-
  profile version. A semantic objective/input/output/authorization change
  requires a new cataloged workflow-profile version.
- **SCP-016** — Every accepted Run MUST pin an immutable
  `WorkflowProfileRef`. The workflow key is a stable public product name, not a
  mutable semantic contract: changing objective, input, output, completion or
  authorization semantics requires a new profile version and digest. Disabling
  a profile or changing the catalog default MUST NOT change recovery or output
  acceptance for an existing Run.

## Non-goals for the first release

- active-active or multi-leader Server;
- a mandatory Contractor telemetry service or separate artifact server (an
  optional external OpenTelemetry sink remains supported);
- support for databases other than PostgreSQL;
- arbitrary third-party Worker protocols besides A2A;
- a generic distributed workflow product independent of Contractor;
- a guarantee that model-generated specifications or findings are correct;
  validators, verification and evaluation expose and measure their quality;
- autonomous, unrestricted penetration testing, browser automation or general
  remote code execution;
- byte-for-byte compatibility with v1 internal artifacts or telemetry events.

## Acceptance

1. A local deployment starts one Server, one PostgreSQL and two Agents.
2. The same run can dispatch attempts to either Agent without sharing Python
   objects or host filesystem paths.
3. Restarting Server after a committed task result resumes the run without
   executing that completed task again.
4. Disabling telemetry persistence does not prevent the run from reaching a
   terminal state.
5. The decomposing Planner dispatches multiple narrow Tasks, while replacing it
   with the passthrough Planner creates one root Task and dispatches its Attempts
   through the same outbox and A2A adapter.
6. The root job completes on an Agent that alternates planning and working
   internally, and the same protocol/conformance fixture passes on a non-ADK
   Agent without changing Server domain or Planner code.
7. A versioned evaluation fixture runs both strategies on the same inputs and
   produces a durable comparison report without treating sampled telemetry as
   authoritative evidence.
8. A static manifest creates multiple typed Tasks, while its one-node
   passthrough specialization uses the identical scheduler and Worker path.
9. A client publishes a project snapshot, submits its exact ArtifactRef and the
   selected Agent materializes it without either wire contract carrying a host
   filesystem path.
10. The enabled workflow catalog describes each entry in product terms and
    exposes exact objective/input/output contracts rather than Planner or Python
    implementation details.
11. A representative API workflow produces a schema-valid OpenAPI artifact from
    a project snapshot, and a representative security workflow produces valid
    typed finding/verification artifacts or an explicit evidence-backed empty
    result; terminal Agent prose alone cannot satisfy either Run.
12. A static-analysis Run has no live-target capability. An exploitability Run
    sends traffic only when its exact target/operation/egress/time authorization
    is valid and returns a typed verdict plus bounded evidence.
13. A Run submitted through a catalog default records the exact resolved
    `WorkflowProfileRef`; changing that default or disabling the old profile
    does not change its RunSpec, recovery behavior or required outputs.
