# Contractor v2 documentation

The canonical working specification is [`spec`](spec/README.md). Its
[`architecture.c4`](spec/architecture.c4) model and focused documents describe
the same execution architecture; there is no second legacy specification in
this tree.

The smallest deployment is one VM containing Contractor Server, PostgreSQL and
one or more identical single-slot Runtime Agent processes. Each allocated
Runtime Agent acts as its one Worker in-process and exposes that Worker's A2A
endpoint. External dependencies such as the LLM Gateway and an S3 blob backend
remain replaceable deployment adapters.

## Entry points

- [Specification index](spec/README.md)
- [LikeC4 architecture](spec/architecture.c4)
- [Workflow Scheduler and Planner](spec/00-workflow-and-planner.md)
- [AgentTemplate](spec/01-agent-template.md)
- [Runtime and A2A](spec/02-runtime-and-a2a.md)
- [Artifact plane](spec/03-artifact-plane.md)
- [Execution lifecycle and metrics](spec/04-execution-lifecycle-and-metrics.md)
- [First slice and open decisions](spec/05-first-slice-and-open-decisions.md)

## Local commands

```shell
likec4 validate docs/spec
likec4 start docs/spec
```
