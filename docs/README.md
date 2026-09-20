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
browser journey and independent producer boundary. V38-001–010 are delivered,
including collection, comparison, the native UI and the release gate.
The [open-task review](plans/2026-09-20-open-task-review.md) records the dated
main/worktree status, remaining scope and parallel work at the time of that review.

[Legacy compatibility removal](plans/2026-09-20-legacy-removal.md) records
confirmed implementation shims, 41 superseded catalog entries and a staged
removal plan. Python shims, deprecated process settings and obsolete UI
redirects have been removed. The project does not require backward compatibility;
remaining readers will move to strict current formats in subsequent increments.

[Autonomous pentest Audit specification](spec/33-autonomous-pentest-audits.md)
defines source-optional web/API auditing, enforced scope, isolated identities,
typed evidence, independent replay, recovery, budgets and acceptance gates.
It specifies the implementation target and maps existing work to delivery
stages; the new capabilities are not yet implemented.

[Audit preparation and Workflow composition](plans/2026-09-20-audit-workflow-composition.md)
tracks V62: first supplied-OpenAPI SQLMap and pinned-URL Nuclei checks, then
preparation before initial inventory. The scan adapter is in progress;
assessment snapshots, cross-round routing and standalone analysis are explicitly
deferred. Candidate Audit scan profiles are present; their production-process
acceptance remains open in V62-009.

[Architecture evolution review — 2026-09-19](research/2026-09-19-architecture-evolution-review.md)
rechecks all 19 recommendations against original decisions and current contracts.
The [follow-up plan](plans/2026-09-19-architecture-review-followups.md) separates
confirmed corrections, a completion decision, an optional refactor and a transport
experiment. Both documents are non-normative.
