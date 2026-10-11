# Specification migration

Temporary working area for moving the legacy feature-ordered specifications
(`docs/spec/NN-*.md` and friends) into the domain layout described in
[conventions](../conventions.md). Delete this directory when the last legacy
document is gone.

## Process

1. `split.py` cuts every legacy document into blocks (paragraph, list, table,
   code block) with IDs `<doc>#<n>`; it accounts for every non-heading line.
2. `ledger.json` classifies every block: target domain document (or several,
   for a mixed block), kind, and registry tags. Block `sha` values identify
   blocks edited after classification.
3. One domain at a time: write the domain documents from their blocks,
   extract registry entries, move history and open questions, verify that no
   fact was lost or changed, delete the migrated blocks from the legacy
   documents (leaving a pointer), and repoint links.
4. A legacy document is deleted once none of its blocks remain.

## Target documents

| Target | Prefix | Scope |
| --- | --- | --- |
| `overview.md` | — | System purpose, deployables, single-host topology, trust boundaries, end-to-end path of one Run |
| `glossary.md` | — | One definition per term, linking to its owner |
| `platform/limits.md` | LIM | Registry of numeric limits and cardinality targets |
| `platform/errors.md` | ERR | Registry of error and stop codes |
| `platform/failure-model.md` | FAIL | Outcome classes, retryability, idempotency and receipts, ambiguous writes, cancellation ownership |
| `platform/lifecycle.md` | LIFE | Entity deletion, cascade and retention matrix |
| `platform/security.md` | SEC | Trust boundaries, mTLS identity, authentication, secrets and redaction, target policy |
| `platform/observability.md` | OBS | Metrics, OTLP traces, telemetry content rules, metadata propagation |
| `platform/persistence.md` | PER | PostgreSQL ownership, transactions, migrations, schema conventions, leases in the database |
| `configuration/catalog.md` | CAT | Configuration kinds, `(kind, name, version)` identity, selectors, digests, loading, validation, managed configuration roots, immutability |
| `configuration/workflows.md` | WFL | Workflow definitions: Stages, transitions, bindings, input and output slots, presentation |
| `configuration/agent-templates.md` | TPL | AgentTemplate, ModelPolicy, ExecutionConfig layering, toolset selection |
| `configuration/runtime-config.md` | RCF | RuntimeConfig, runtime labels, label resolution and merge, credentials, LLM Gateways, adapters |
| `execution/runs.md` | RUN | WorkflowRun resource and lifecycle, creation and admission, metadata labels, cancellation, outputs |
| `execution/scheduler.md` | SCH | StageExecution, claims, lanes and concurrency, retry and escalation, owner pause, recovery |
| `execution/control-plane.md` | CTL | Runtime Agent registration, principals, mTLS identity, heartbeat and lease, capability discovery, placement, allocation and release |
| `execution/runtime-agent.md` | RTA | Runtime process, startup probes, allocation lifecycle on the Runtime side, A2A endpoint, work root |
| `execution/planners.md` | PLN | Passthrough, Streamline and Router Planners, plans, budgets, Planner tools |
| `execution/worker.md` | WRK | Worker model boundary, result finalizer, structured results and observations, live state, summarization, session modes, Gateway client |
| `artifacts/artifact-store.md` | ART | ArtifactStore scopes, namespaces, refs, revisions, CAS, tool authority, Workflow input forks and output publication |
| `artifacts/projects.md` | PRJ | Project resource, ProjectScope, recommendations, output publication, the global queue view, Project target configuration |
| `artifacts/retention-and-deletion.md` | DEL | Run hard deletion, Project deletion, purge and garbage collection |
| `artifacts/blob-storage.md` | BLB | Payload backends (PostgreSQL, filesystem), transfer slots, orphan cleanup |
| `artifacts/git-import.md` | GIT | Git repositories as artifact inputs, source bundles, SSH trust |
| `tools/toolsets.md` | TLS | Common toolset contract: selection, capabilities, tool errors versus exceptions, bounds, telemetry |
| `tools/memory.md` | MEM | Memory notes and memory tools |
| `tools/skills.md` | SKL | Agent Skill packages, catalog, pinning, ADK Skill toolset |
| `tools/workspace.md` | WSP | Run workspaces, hydration, direct and overlay modes, filesystem and edit tools |
| `tools/code-analysis.md` | CAN | Tree-sitter and Trailmark tools, taint annotations |
| `tools/http-caido.md` | HTP | HTTP tools, forward proxy, Caido tools |
| `tools/scanners.md` | SCN | Tool Workers, scanner adapters, scan request preparation and scan planning |
| `tools/code-execution.md` | EXE | Podman sandbox and code execution |
| `audits/audits.md` | AUD | AuditProfile, baseline and scope, durable model, packages and schemas, inventories, Audit and Round lifecycle |
| `audits/execution.md` | AUX | Audit Controller: submission, dispatch, collection, import, recovery, budgets |
| `audits/review-and-reports.md` | REV | Human review, evidence, coverage, reports, Audit retention |
| `audits/worker-completion.md` | AWC | Audit check result collection and completion inside the Worker |
| `audits/scans.md` | ASC | Audit preparation and OpenAPI scan checks |
| `findings/findings.md` | FND | Finding proposals, collections, locations and finding tools |
| `evals/format.md` | EVF | Portable evaluation format |
| `evals/managed-evals.md` | EVM | Managed Evals and external producers |
| `api/public-api.md` | API | Public API conventions: authentication, sessions, CSRF, pagination, idempotency, error envelope, versioning, events |
| `ui/ui.md` | UI | UI architecture, information architecture, language, data freshness, accessibility |
| `ui/journeys.md` | UIJ | User journeys the UI must support |
| `operations/operations.md` | OPS | Operations surface and settings, performance metrics, profiling, local PKI, deployment boundary, CLI |

Non-reference destinations:

| Target | Receives |
| --- | --- |
| `proposals/0028-finding-analysis-and-sarif.md` | Unimplemented SARIF and analysis design |
| `proposals/0033-autonomous-pentest-audits.md` | Unimplemented autonomous pentest design |
| `proposals/0034-audit-check-prioritization.md` | Prioritization design whose integration is not implemented |
| `decisions/spec-history.md` | Delivery plans, increments, acceptance and verification narratives, implementation status, task references |
| `decisions/open-questions.md` | Open design questions |
| `drop` | Pure navigation with no information (tables of contents, "see above") |

## Block kinds

| Kind | Meaning |
| --- | --- |
| `normative` | Rule, contract or behaviour |
| `rationale` | Why a rule exists |
| `definition` | Defines a term |
| `example` | Worked example |
| `non-goal` | Deliberately unsupported or deferred behaviour |
| `history` | Delivery, increment, acceptance, verification, status, task reference |
| `open-question` | Unresolved design question |
| `proposal` | Design of an unimplemented capability |
| `navigation` | No information beyond structure |

Registry tags mark blocks that contain entries for a platform registry:
`limit`, `error`, `failure`, `lifecycle`, `security`, `observability`.

## Domain status

| Domain | Status |
| --- | --- |
| artifacts | pilot |
| all others | not started |
