# Contractor documentation

The documentation is grouped by purpose so that current behavior, proposals,
and research notes are not mistaken for one another.

## Start here

- [Project README](../README.md) — installation, configuration, and CLI usage.
- [Architecture overview](architecture/README.md) — contributor-oriented tour of
  the runtime, workflows, artifacts, memory, and planner.
- [Reconstruction specification](spec/README.md) — normative behavior and
  acceptance criteria for a compatible implementation.

## Architecture

- [Core architecture and planner internals](architecture/README.md)
- [Streamline planner and task runner](architecture/planner.md)
- [Architecture diagrams and LikeC4 source](architecture/diagrams/)

## Guides and references

- [Tuning and performance](guides/tuning.md) — practical tuning workflow and
  the highest-impact controls.
- [Evaluation tuning](guides/evaluation-tuning.md) — parameter sweeps through
  the evaluation harness.
- [Tunable parameters](guides/tunable-parameters.md) — exhaustive settings and
  configuration reference.

## Designs and proposals

These documents describe intended or experimental work, not necessarily the
current runtime.

- [Code-as-agent harness improvement plan](design/code-as-agent-harness.md)
- [Shannon workflow design](design/shannon-workflow.md)

## Insights and research

- [Parallel vulnerability-pipeline insights](insights/parallel-vulnerability-pipelines.md)
- [Research index](research/README.md) — tracked reports, audits, source notes,
  and the interactive research memo.

When a proposal becomes implemented, update the applicable specification and
tests in the same change. Generated or historical research should remain under
`research/`, rather than being presented as normative project behavior.
