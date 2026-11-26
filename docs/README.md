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

[Proposals](proposals/) contain designs and decisions for unfinished or partly
delivered capabilities. Their status is stated in each document; task files own
delivery status. For example, the [pentest Audit design](proposals/0033-autonomous-pentest-audits.md)
contains historical design and a revised delivery plan; its implemented parts
are specified by their owning contracts. [Project sitemap](proposals/0042-live-project-sitemap.md)
is partially implemented, with its first-slice release gate still pending.
The [common Audit hypothesis model](proposals/0050-audit-common-hypothesis-model.md)
revises the superseded research-only [proposal 0045](proposals/0045-audit-hypothesis-search.md).
The [Runtime Planner and recovery proposal](proposals/0058-runtime-planner-and-recovery.md)
builds on the completed multi-allocation Runtime and separates Planner relocation
from checkpoint recovery, with proposed boundaries, diagrams and open decisions.

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
  [Audit redesign](plans/2026-10-31-audit-redesign.md) (common Audit model and V64),
  [schema reduction](plans/2026-10-04-schema-reduction.md) (V365)
  and [schema reduction, round two](plans/2026-10-23-schema-reduction-round-2.md) (analysis, no task yet).
- [Research](research/README.md) holds dated reviews and experiment planning.
- [Historical documents](history/README.md) preserve completed proposals and
  superseded plans/reviews; original paths remain available for task references.
