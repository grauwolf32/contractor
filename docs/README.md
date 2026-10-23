# Documentation

Contractor is an Application Security orchestration platform. Workflows define
extensible checks and supporting analyses; Audits coordinate them into assessments.

| Start here | Contents |
| --- | --- |
| [Deployment](deployment.md) | Install, configure and run Server, Runtime, UI and their dependencies |
| [User guides](guides/README.md) | CLI, Projects, Audits, artifacts and Skills |
| [Development](development.md) | Dependencies, building the CLI and local checks |
| [Specification](spec/README.md) | Reference specification by domain: contracts, limits, error codes, failure model, entity lifecycle and UI journeys |
| [Testing](testing/README.md) | Deterministic checks, integration gates and live-model evaluations |

The [configuration catalog](../configs/README.md) lists shipped Workflows and
policies. [Operations](operations/README.md) contains focused references used by
the deployment guide.

[Proposals](proposals/) hold designs of capabilities that are not implemented
yet; each is folded into the specification once implemented.
[Compressed HTTP traffic export](proposals/0036-compressed-http-traffic-export.md)
is a deferred idea. The
[autonomous pentest Audit proposal](proposals/0033-autonomous-pentest-audits.md)
defines source-optional web/API auditing, enforced scope, isolated identities,
typed evidence, independent replay, recovery, budgets and acceptance gates; it
is not yet implemented.

[Runtime placement labels](proposals/0043-runtime-placement-labels.md) records
the selected `cloud` and `local` namespaces, startup declarations and dynamic
parent matching for WorkflowRun and Audit launches; it is not yet implemented.

Accepted contracts belong in the specification; implementation status belongs
in [task files](../tasks/) governed by [tasks/index.yml](../tasks/index.yml).
The following are non-normative, and a historical report does not establish
current readiness:

- [UI design](design/ui/README.md) records the implemented V3B layout, its
  themes, vocabulary and the Workflow/Audit/Agent editor studio.
- [Plans](plans/) hold the plans of open or deferred task series:
  [ScanTools](plans/2026-09-19-scan-tools.md) (V55),
  [run-scoped toolset configuration](plans/2026-09-19-toolset-runtime-configuration.md) (V61),
  [Audit preparation and Workflow composition](plans/2026-09-20-audit-workflow-composition.md) (V62),
  [contextual checklist prioritization](plans/2026-09-20-audit-check-prioritization.md) (V64)
  and [schema reduction](plans/2026-10-04-schema-reduction.md) (V365).
- [Research](research/README.md) holds dated reviews that open or deferred
  tasks still cite.
- [Evals experience design](evals-experience-design.md) records the selected
  full browser journey and independent producer boundary. V38-001–010 are
  delivered, including collection, comparison, the native UI and the release
  gate.
