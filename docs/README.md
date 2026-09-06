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

- [Local development and end-to-end MVP](development.md)
- [Artifact blob storage and no-PVC deployment](artifact-blob-storage.md)
- [Git repository artifacts](spec/24-git-artifacts.md) — planned SSH-key Settings and Workflow/Project snapshot import.
- [Specification index](spec/README.md)
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
- [Workflow Scheduler concurrency control](spec/20-scheduler-concurrency-control.md)
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
