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

The configuration/UI target loads YAML/text definitions from `configs/` with
seven fixed subtrees: `workflows`, `agent-templates`, `model-policies`,
`llm-gateways`, `execution-configs`, `audit-profiles` and `instructions`.
YAML document identity comes from
`kind + metadata.name + metadata.version`, not its file name;
[00](00-workflow-and-planner.md) owns the complete loading contract.
Reviewable built-in Agent Skill sources live in the additional non-YAML
`configs/skills` subtree. It is the create-only source for initial
`SkillCatalog` population of the local owner's ordinary `skills/*` UserScope
artifacts, not live desired state; restart never overwrites a binding advanced
through the Artifact API. [09](09-agent-skills.md) owns that lifecycle.

## Reading order

| Document | Owns |
|---|---|
| [00](00-workflow-and-planner.md) | Workflow Scheduler, Stage, Planner and StageResult |
| [01](01-agent-template.md) | Reusable Worker behavior and Stage binding |
| [02](02-runtime-and-a2a.md) | Allocation and the single-slot Runtime Agent acting as Worker/A2A Server |
| [03](03-artifact-plane.md) | ArtifactStore scopes, RunArtifactSpace, input/output forks, Namespace and CAS |
| [04](04-execution-lifecycle-and-metrics.md) | StageExecution identity, StageTermination, sessions, finalization, recovery and metrics |
| [05](05-first-slice-and-open-decisions.md) | First implementation slice and deliberately deferred decisions |
| [06](06-server-ui-and-operations.md) | Separate Node.js Web UI, Operations visibility and published execution configuration selection |
| [07](07-runtime-labels-and-infrastructure-config.md) | Run-selected/Agent Runtime labels, database-backed infrastructure configs, adapter placement and allocation-scoped settings |
| [08](08-memory-tools.md) | Run-scoped shared Memory Namespace and the artifact-backed `memory-tools@1` contract |
| [09](09-agent-skills.md) | AgentTemplate-selected, artifact-pinned Agent Skills loaded through native Google ADK |
| [10](10-runtime-filesystems-and-edit-tools.md) | Run-artifact workspaces, Runtime local/memory storage, overlay export and filesystem/Edit tools |
| [11](11-http-and-caido-tools.md) | Allocation-scoped HTTP exploration and label-configured Caido GraphQL tools |
| [12](12-code-analysis-tools.md) | Workspace Tree-sitter analysis and local-only allocation-scoped Trailmark graph tools |
| [13](13-taint-annotations.md) | Structured, atomic `@trace`/`@validate`/`@sink` workspace annotations |
| [14](14-worker-results-and-live-state.md) | Typed Worker results, deterministic observations, volatile Worker State and explicit Planner projections |
| [15](15-worker-summarization.md) | Optional one-shot terminal Worker summarization at a deterministic soft limit |
| [16](16-run-metadata-labels.md) | Immutable queryable WorkflowRun metadata labels and the eval correlation convention |
| [17](17-projects-and-queue.md) | Optional Projects, ProjectScope artifact reuse, Workflow recommendations, Evals and the global Queue |
| [18](18-run-and-workspace-lifecycle-controls.md) | Consolidated Runs UI, durable queue pause/resume and safe Run/Project deletion |
| [19](19-audits.md) | Project-bound multi-Run Audits, deterministic inventories, findings, review, coverage and recovery |
| [20](20-scheduler-concurrency-control.md) | Durable Operations-controlled concurrency across ordinary WorkflowRuns and Audit dispatch backpressure |
| [21](21-podman-sandbox.md) | Implemented: opt-in local Podman execution, direct workspace sharing and verified container cleanup |
| [22](22-performance-metrics-and-profiling.md) | Implemented: Server/DB performance charts, completed allocation resources, retained history and independent Go profiling |
| [23](23-artifact-blob-backends.md) | Startup-selected PostgreSQL/filesystem blobs, Kubernetes without PVC, bounded memory and deferred S3 |
| [24](24-git-artifacts.md) | Owner SSH-key Settings, bounded in-memory Git snapshots, Workflow/Project import UI and immutable commit provenance |
| [25](25-audit-worker-finalization.md) | Planned: explicit Audit-check completion contracts, incremental results and deterministic Runtime ZIP publication; ordinary Workflows unchanged |
| [LikeC4](architecture.c4) | Component map and focused architecture views |

[`core-execution-model.md`](core-execution-model.md) is a short navigation entry
point for links that previously targeted the monolithic working agreement.

## Smallest useful mental model

```text
Workflow
  -> Workflow Scheduler validates immutable parameters and forks exact UserScope or ProjectScope inputs and selected owner skills into RunArtifactSpace
  -> selects a ready Stage
  -> resolve its AgentTemplate bindings
  -> Control Plane resolves pinned default/Run-selected Runtime labels plus Agent Runtime labels and prepares compatible Worker allocations
  -> PlannerFactory creates the selected Stage-local Planner
  -> Planner tools talk through WorkerInvoker/A2A to allocated Runtime Agents acting as Workers
  -> Runtime validates one typed semantic Worker result and attaches deterministic observations
  -> explicit Planner state tools may inspect safe live projections from that allocation
  -> explicitly selected MemoryTools let Planner and each logical Worker share that Worker's Run-scoped notes
  -> all participants exchange durable data through RunArtifactSpace
  -> Planner returns one candidate StageResult
  -> Workflow Scheduler persists finalizing, drains Workers and collects reports
  -> Workflow Scheduler accepts the terminal result, binds outputs and releases allocations
  -> an explicit successful Workflow transition freezes required outputs and commits Run success
```

If Scheduler-owned cancellation or interruption wins before `finalizing`, the
alternative path is `aborting`: Scheduler stores StageTermination, fences
writes, performs a bounded best-effort drain and reaches `cancelled` or
`interrupted` without waiting for terminal A2A Task state.

WorkflowRun separately owns `initializing`, `running`, `cancelling` and terminal
`succeeded`/`failed`/`cancelled` states. Cancel and success serialize on the
durable Run row; the first committed transition wins.

The boundaries are deliberately narrow:

| Concept | Responsibility |
|---|---|
| Workflow | Product-specific Stage graph and transition/acceptance policy |
| Workflow Scheduler | WorkflowRun progression and durable StageExecution lifecycle |
| Planner | Stage-local decomposition and routing among prepared Workers |
| AgentTemplate | Immutable, reusable Worker behavior/configuration |
| Worker session mode | Immutable Stage policy selecting isolated-by-default or explicit shared sequential ADK conversation semantics inside each allocation |
| Control Plane | Capacity and allocation lifecycle |
| Runtime Agent | One process and one slot: control client, A2A Server and one in-process Worker runtime while allocated |
| Runtime label | Control Plane alias selecting an immutable typed infrastructure config for a Run or Runtime Agent |
| Run metadata label | Immutable queryable key/value metadata that groups Runs without changing execution |
| Runtime adapter | Allocation-scoped Runtime code configured by Control Plane without adding model-visible tools |
| ArtifactStore | One physical artifact service, registry and blob boundary |
| Agent Skill | Owner UserScope guidance artifact selected by AgentTemplate and pinned/forked per Run |
| SkillCatalog | Internal validator/resolver over ordinary Skill artifacts; no separate API or storage |
| MemoryTools | Thin Planner/Worker wrapper over reserved RunScope note artifacts; hidden CAS and logical note projection |
| WorkspaceFS | Allocation-scoped Run artifact workspace with Runtime-local local/memory storage and optional overlay export |
| HTTP/Caido tools | Explicit Worker tools over allocation-owned direct/proxied HTTP and typed label-configured Caido clients |
| Code analysis | Read-only shallow Tree-sitter tools on local/memory workspaces and local-only killable Trailmark graph tools |
| Taint annotations | Explicitly selected structured source mutations over a narrowed workspace Writer |
| Worker State | Bounded volatile Runtime-owned metrics/observation state, readable only through private Control Plane transport |
| Planner state tools | Explicit typed projections over a logical Worker's newest correlated live snapshot; never a generic State query |
| UserScope | Authenticated user's durable artifact library |
| Project | Optional owner-scoped organization of reusable artifacts and ordinary Runs; never an execution state machine |
| ProjectScope | Long-lived Project artifact view from which exact inputs are forked into a Run |
| RunArtifactSpace | RunScope view with mutable inputs, intermediates and declared outputs |
| Queue | Owner-scoped read projection over nonterminal WorkflowRuns; never a second Scheduler |
| Queue control | Durable owner-scoped admission gate; never a WorkflowRun state or process-local switch |
| Audit | Project-bound durable coordinator of ordinary Runs, evidence, findings and review; never another Scheduler |

## Specification rule

Each decision has one owning document. Other documents link to it instead of
repeating a second normative version. Unresolved behavior stays in
[05](05-first-slice-and-open-decisions.md) rather than being inferred from
historical designs.
