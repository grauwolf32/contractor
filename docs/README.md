# Documentation

Contractor is an Application Security orchestration platform. Workflows define
extensible checks and supporting analyses; Audits coordinate them into assessments.

| Start here | Contents |
| --- | --- |
| [Deployment](deployment.md) | Install, configure and run Server, Runtime, UI and their dependencies |
| [User guides](guides/README.md) | CLI, Projects, Audits, artifacts and Skills |
| [Development](development.md) | Dependencies, building the CLI and local checks |
| [Specification](spec/README.md) | Execution model, component contracts and UI acceptance scenarios |
| [Testing](testing/README.md) | Deterministic checks, integration gates and live-model evaluations |

The [configuration catalog](../configs/README.md) lists shipped Workflows and
policies. [Operations](operations/README.md) contains focused references used by
the deployment guide.

[Research](research/README.md) holds unimplemented proposals and hypotheses.
Accepted contracts belong in the specification; implementation status belongs in
[tasks/index.yml](../tasks/index.yml). A historical test report does not establish
current readiness.

[Evals experience design](evals-experience-design.md) records the selected full
browser journey and independent producer boundary; V38 implementation is planned.

[Legacy compatibility removal](plans/2026-09-20-legacy-removal.md) records
confirmed implementation shims, 41 superseded catalog entries and a staged
removal plan. Python shims, deprecated process settings and obsolete UI
redirects have been removed. The project does not require backward compatibility;
remaining readers will move to strict current formats in subsequent increments.

[Architecture review — 2026-09-15](reviews/architecture-review-2026-09-15.md)
records the per-view scenario checks, fixes and verification limits.
