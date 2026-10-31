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
[Target scanning through Audit](proposals/0049-audit-owned-target-scanning.md)
records the decision to remove independent local and durable sweeps and use the
existing Project Audit and selected check Workflow for a concrete target scan.
[Compressed HTTP traffic export](proposals/0036-compressed-http-traffic-export.md)
is a deferred idea. The
[autonomous pentest Audit proposal](proposals/0033-autonomous-pentest-audits.md)
defines source-optional web/API auditing, enforced scope, isolated identities,
typed evidence, independent replay, recovery, budgets and acceptance gates; it
is not yet implemented.

[Hypothesis search within Audits](proposals/0045-audit-hypothesis-search.md)
records an optional research pool, provenance, experiment selection, exact
cross-Run context and the revised V64 direction. Direct deterministic execution
remains independent; attack trees are an optional analytical layer. It is a
draft and does not enable the capability.

Runtime placement labels are implemented: see the [Control Plane contract](spec/execution/control-plane.md)
for `cloud` and `local` namespaces, startup declarations and matching for Run
and Audit launches.

[Workflow context compaction](proposals/0046-workflow-context-compaction.md)
proposes continuing a Worker within one invocation, an explicit Workflow
binding policy, configurable reasoning input for summarization and compaction,
and bounded history recovery. It is a draft and does not enable the capability.

[Worker model consultation](proposals/0047-worker-model-consultation.md)
proposes advice from an explicitly pinned tool-free model within the same
Worker, with shared budgets, bounded context and repeated comparative Eval.
It remains a proposal without implementation tasks or activation.

[OpenCode and retained Audit environments](proposals/0048-opencode-audit-environments.md)
proposes an autonomous OpenCode backend behind the existing A2A boundary,
an Audit-owned writable Podman environment, placement on `local` or a concrete
cloud VM, and named service publications across Runs. It is a draft; session
retention, lifecycle details and port forwarding still require contract decisions.

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
  [optional hypothesis search](plans/2026-10-24-audit-hypothesis-search.md) (revised V64 and shared V62 dependencies),
  [schema reduction](plans/2026-10-04-schema-reduction.md) (V365)
  and [schema reduction, round two](plans/2026-10-23-schema-reduction-round-2.md) (analysis, no task yet).
- [Research](research/README.md) holds dated reviews and experiment planning.
- [Historical documents](history/README.md) preserve completed proposals and
  superseded plans/reviews; original paths remain available for task references.
- [Evals experience design](evals-experience-design.md) records the selected
  full browser journey and independent producer boundary. V38-001–010 are
  delivered, including collection, comparison, the native UI and the release
  gate.
