# Contractor v2 — working specifications

Status: **Working agreement**

This directory is the canonical working specification for Contractor v2. It
captures the deliberately small execution model from the current design
discussion; removed historical documents do not supply implicit requirements.

The target deployment is one VM or one physical host. Server, PostgreSQL and
one or more lightweight Runtime Agent processes may all run on that host.
Nothing in this model requires Kubernetes, a controller or a separate control-
plane service.

The execution core is domain-neutral. Code understanding and application
security are initial Workflow examples, not a closed product-capability enum or
a restriction on future Workflows. Domain semantics live in YAML Workflow,
Planner and AgentTemplate definitions rather than in Scheduler branching.

## Reading order

| Document | Owns |
|---|---|
| [00](00-workflow-and-planner.md) | Workflow Scheduler, Stage, Planner and StageResult |
| [01](01-agent-template.md) | Reusable Worker behavior and Stage binding |
| [02](02-runtime-and-a2a.md) | Allocation and the single-slot Runtime Agent acting as Worker/A2A Server |
| [03](03-artifact-plane.md) | ArtifactStore scopes, RunArtifactSpace, input/output forks, Namespace and CAS |
| [04](04-execution-lifecycle-and-metrics.md) | StageExecution identity, sessions, finalization, recovery and metrics |
| [05](05-first-slice-and-open-decisions.md) | First implementation slice and deliberately deferred decisions |
| [LikeC4](architecture.c4) | Component map and focused architecture views |

[`core-execution-model.md`](core-execution-model.md) is a short navigation entry
point for links that previously targeted the monolithic working agreement.

## Smallest useful mental model

```text
Workflow
  -> Workflow Scheduler forks exact UserScope inputs into RunArtifactSpace
  -> selects a ready Stage
  -> resolve its AgentTemplate bindings
  -> Control Plane prepares Worker allocations
  -> PlannerFactory creates one Planner ADK agent
  -> Planner talks through A2A to allocated Runtime Agents acting as Workers
  -> all participants exchange durable data through RunArtifactSpace
  -> Planner returns one candidate StageResult
  -> Workflow Scheduler persists finalizing, drains Workers and collects reports
  -> Workflow Scheduler accepts the terminal result, binds outputs and releases allocations
```

The boundaries are deliberately narrow:

| Concept | Responsibility |
|---|---|
| Workflow | Product-specific Stage graph and transition/acceptance policy |
| Workflow Scheduler | WorkflowRun progression and durable StageExecution lifecycle |
| Planner | Stage-local decomposition and routing among prepared Workers |
| AgentTemplate | Immutable, reusable Worker behavior/configuration |
| Control Plane | Capacity and allocation lifecycle |
| Runtime Agent | One process and one slot: control client, A2A Server and one in-process Worker runtime while allocated |
| ArtifactStore | One physical artifact service, registry and blob boundary |
| UserScope | Authenticated user's durable artifact library |
| RunArtifactSpace | RunScope view with mutable inputs, intermediates and declared outputs |

## Specification rule

Each decision has one owning document. Other documents link to it instead of
repeating a second normative version. Unresolved behavior stays in
[05](05-first-slice-and-open-decisions.md) rather than being inferred from
historical designs.
