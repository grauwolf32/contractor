# Contractor v2 documentation

The canonical working specification is [`spec`](spec/README.md). Its
[`architecture.c4`](spec/architecture.c4) model and focused documents describe
the same execution architecture; there is no second legacy specification in
this tree.

The smallest deployment is one VM containing Contractor Server, PostgreSQL and
one or more single-slot Runtime Agent processes running the same Contractor
code. Their immutable environments and positive tool capabilities may differ.
Each allocated Runtime Agent acts as its one Worker in-process and exposes that
Worker's A2A endpoint. External dependencies such as the LLM Gateway and an S3
blob backend remain replaceable deployment adapters.

## Entry points

- [Command-line client](cli.md)
- [Local development and end-to-end MVP](development.md)
- [Artifact blob storage and no-PVC deployment](artifact-blob-storage.md)
- [Git repository artifacts](git-artifacts.md) — SSH-key Settings, Workflow/Project import, deployment and release verification; [specification](spec/24-git-artifacts.md).
- [Specification index](spec/README.md)
- [Portable evaluation format](spec/26-portable-evaluation-format.md) — versioned datasets, execution bindings and evaluator records; [offline readiness and implementation pins](reviews/portable-eval-format-readiness.md). Model quality runs require a separate frozen budget.
- [UI user stories and improvement roadmap](ui-user-stories.md) — planned scenarios, acceptance and V37/V38 tasks.
- [UI use-case and usability review, 2026-09-06](reviews/2026-09-06-ui-use-cases-and-usability.md) — observed obstacles and supporting evidence.
- [Workflow and AuditProfile mapping to contractor-old](reviews/2026-09-06-workflows-audits-legacy-mapping.md) — catalog updated 2026-09-07, concrete legacy counterparts and remaining gaps.
- [Production scenario catalog](reviews/2026-09-06-production-scenario-catalog.md) — purpose, production use and proposed composition of security scenarios; updated 2026-09-07.
- [Findings tools plan, 2026-09-06](reviews/2026-09-06-findings-tools-plan.md) — shared creation/reading tools, producer/analyst roles and V43 tasks.
- [Findings tools and collection format](spec/27-findings-tools-and-collections.md) — implemented creation/reading tools, Server collection publication and Runtime materialization; [offline process validation](reviews/2026-09-06-findings-tools-validation.md).
- [Unfinished-task consistency review, 2026-09-06](reviews/2026-09-06-unfinished-tasks-consistency.md) — historical review of the pending plan, corrected contract conflicts and verification gaps; current statuses are in tasks/index.yml.
- [Annotations as optional Workflow artifacts](reviews/2026-09-06-trace-annotation-contract.md) — artifact boundary and when a structured index is useful.
- [LikeC4 architecture](spec/architecture.c4)
- [Workflow Scheduler and Planner](spec/00-workflow-and-planner.md)
- [AgentTemplate](spec/01-agent-template.md)
- [Runtime and A2A](spec/02-runtime-and-a2a.md)
- [Artifact plane](spec/03-artifact-plane.md)
- [Execution lifecycle and metrics](spec/04-execution-lifecycle-and-metrics.md)
- [First slice and open decisions](spec/05-first-slice-and-open-decisions.md)
- [Server UI and Operations](spec/06-server-ui-and-operations.md)
- [Runtime labels and infrastructure configuration](spec/07-runtime-labels-and-infrastructure-config.md)
- [Shared MemoryTools](spec/08-memory-tools.md)
- [Agent Skills](spec/09-agent-skills.md)
- [Runtime filesystems and Edit tools](spec/10-runtime-filesystems-and-edit-tools.md)
- [HTTP and Caido tools](spec/11-http-and-caido-tools.md)
- [Workspace code-analysis tools](spec/12-code-analysis-tools.md)
- [Projects, reusable artifacts and global Queue](spec/17-projects-and-queue.md)
- [Run and workspace lifecycle controls](spec/18-run-and-workspace-lifecycle-controls.md)
- [Project-bound multi-Run Audits](spec/19-audits.md)
- [Audit Worker completion contracts](spec/25-audit-worker-finalization.md) — inert contracts and deterministic result publisher implemented; collector, integration and rollout remain in V39.
- [Workflow Scheduler concurrency control](spec/20-scheduler-concurrency-control.md)
- [Allocation-scoped Podman execution](spec/21-podman-sandbox.md) — implemented, opt-in local/direct sandbox with verified cleanup.
- [Operations performance and profiling](operations-performance.md) — startup
  switches, retention, missing-data semantics, upgrade order and release evidence;
  [specification](spec/22-performance-metrics-and-profiling.md).
- [V8–V11 implementation decision log](implementation-decisions-v8-v11.md)

## Research drafts

Research drafts are non-normative, not production-ready, and do not describe
registered configuration unless promoted into the focused specification.

- [`stateflow@1` Planner research draft](stateflow-1-research-draft.md) — explores
  bounded explicit Planner state, deterministic Worker-context construction and
  allocation-scoped Worker-session modes (isolated by default) selected from
  Stage configuration, inspired by the SKILL.state
  paper.

## Local commands

```shell
likec4 validate docs/spec
likec4 start docs/spec
```
