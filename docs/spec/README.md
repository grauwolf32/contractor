# Contractor specification

The reference specification of Contractor: how the system is meant to behave,
organised by domain. It is normative. When code and specification disagree,
find out which one is wrong and fix it in the same change.

[Documentation index](../README.md) · [Conventions](conventions.md) · [Glossary](glossary.md)

Contractor is an Application Security orchestration platform. Versioned
Workflows define security checks and supporting analyses; a Run executes one
Workflow; an Audit coordinates many ordinary Runs inside a Project against a
pinned baseline and tracks coverage, evidence, findings and human review.
Everything runs on one host: Server, PostgreSQL and one or more single-slot
Runtime Agents.

## Start here

| Document | Read it for |
| --- | --- |
| [System overview](overview.md) | Purpose, deployables, single-host topology, trust boundaries and the end-to-end path of one Run |
| [Conventions](conventions.md) | What belongs in the specification, document shape, requirement IDs, registries |
| [Glossary](glossary.md) | One definition per term, linked to the document that owns it |

Each document starts with its requirement prefix. Requirements are numbered
`**PREFIX-Rn.**`, never renumbered, and cited as `[RUN-R12](execution/runs.md)`.
Numeric bounds, error codes, failure outcomes and deletion rules are defined
once in the platform registries and cited by ID (`LIM-…`, error code,
`FAIL-…`, `LIFE-…`).

## Domains

### Platform

Rules shared by every domain.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Limits](platform/limits.md) | `LIM` | Registry of numeric limits and cardinality targets |
| [Error and stop codes](platform/errors.md) | `ERR` | Registry of error and stop codes |
| [Failure model](platform/failure-model.md) | `FAIL` | Outcome classes, retryability, idempotency and receipts, ambiguous writes, cancellation ownership |
| [Entity lifecycle](platform/lifecycle.md) | `LIFE` | Entity deletion, cascade and retention matrix |
| [Security](platform/security.md) | `SEC` | Trust boundaries, mTLS identity, authentication, secrets and redaction, target policy |
| [Observability](platform/observability.md) | `OBS` | Metrics, OTLP traces, telemetry content rules, metadata propagation |
| [Persistence](platform/persistence.md) | `PER` | PostgreSQL ownership, transactions, migrations, schema conventions, leases in the database |

### Configuration

What can be configured and how a configuration is identified, loaded and selected.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Configuration catalog](configuration/catalog.md) | `CAT` | Configuration kinds, `(kind, name, version)` identity, selectors, digests, loading, validation, managed configuration roots, immutability |
| [Workflows](configuration/workflows.md) | `WFL` | Workflow definitions: Stages, transitions, bindings, input and output slots, presentation |
| [Agent templates](configuration/agent-templates.md) | `TPL` | AgentTemplate, ModelPolicy, ExecutionConfig layering, toolset selection |
| [Runtime configuration, labels, credentials and Gateways](configuration/runtime-config.md) | `RCF` | RuntimeConfig, runtime labels, label resolution and merge, credentials, LLM Gateways, adapters |

### Execution

How a Run executes: Runs, the Scheduler, the Control Plane, Runtime Agents, Planners and Workers.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Runs](execution/runs.md) | `RUN` | WorkflowRun resource and lifecycle, creation and admission, metadata labels, cancellation, outputs |
| [Scheduler](execution/scheduler.md) | `SCH` | StageExecution, claims, lanes and concurrency, retry and escalation, owner pause, recovery |
| [Control Plane](execution/control-plane.md) | `CTL` | Runtime Agent registration, principals, mTLS identity, heartbeat and lease, capability discovery, placement, allocation and release |
| [Runtime Agent](execution/runtime-agent.md) | `RTA` | Runtime process, startup probes, allocation lifecycle on the Runtime side, A2A endpoint, work root |
| [Planners](execution/planners.md) | `PLN` | Passthrough, Streamline and Router Planners, plans, budgets, Planner tools |
| [Worker](execution/worker.md) | `WRK` | Worker model boundary, result finalizer, structured results and observations, live state, summarization, session modes, Gateway client |

### Artifacts

Durable data: the ArtifactStore, Projects, retention and deletion, blob storage and Git import.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Artifact store](artifacts/artifact-store.md) | `ART` | ArtifactStore scopes, namespaces, refs, revisions, CAS, tool authority, Workflow input forks and output publication |
| [Projects](artifacts/projects.md) | `PRJ` | Project resource, ProjectScope, recommendations, output publication, the global queue view, Project target configuration |
| [Retention and deletion](artifacts/retention-and-deletion.md) | `DEL` | Run hard deletion, Project deletion, purge and garbage collection |
| [Artifact blob storage](artifacts/blob-storage.md) | `BLB` | Payload backends (PostgreSQL, filesystem), transfer slots, orphan cleanup |
| [Git import](artifacts/git-import.md) | `GIT` | Git repositories as artifact inputs, source bundles, SSH trust |

### Tools

Toolsets available to Workers.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Toolsets](tools/toolsets.md) | `TLS` | Common toolset contract: selection, capabilities, tool errors versus exceptions, bounds, telemetry |
| [Memory](tools/memory.md) | `MEM` | Memory notes and memory tools |
| [Agent Skills](tools/skills.md) | `SKL` | Agent Skill packages, catalog, pinning, ADK Skill toolset |
| [Run workspaces and filesystem tools](tools/workspace.md) | `WSP` | Run workspaces, hydration, direct and overlay modes, filesystem and edit tools |
| [Code analysis](tools/code-analysis.md) | `CAN` | Tree-sitter and Trailmark tools, taint annotations |
| [HTTP and Caido tools](tools/http-caido.md) | `HTP` | HTTP tools, forward proxy, Caido tools |
| [Scanners](tools/scanners.md) | `SCN` | Tool Workers, scanner adapters, scan request preparation and scan planning |
| [Code execution](tools/code-execution.md) | `EXE` | Podman sandbox and code execution |

### Audits

Project-bound assessments that coordinate many Runs.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Audits](audits/audits.md) | `AUD` | AuditProfile, baseline and scope, durable model, packages and schemas, inventories, Audit and Round lifecycle |
| [Audit execution](audits/execution.md) | `AUX` | Audit Controller: submission, dispatch, collection, import, recovery, budgets |
| [Review and reports](audits/review-and-reports.md) | `REV` | Human review, evidence, coverage, reports, Audit retention |
| [Audit Worker completion](audits/worker-completion.md) | `AWC` | Audit check result collection and completion inside the Worker |
| [Audit preparation and OpenAPI scans](audits/scans.md) | `ASC` | Audit preparation and OpenAPI scan checks |

### Findings

Finding proposals, collections and finding tools.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Findings](findings/findings.md) | `FND` | Finding proposals, collections, locations and finding tools |

### Evals

The portable evaluation format and managed Evals.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Portable evaluation format](evals/format.md) | `EVF` | Portable evaluation format |
| [Managed Evals](evals/managed-evals.md) | `EVM` | Managed Evals and external producers |

### API

Conventions shared by every public resource.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Public API](api/public-api.md) | `API` | Public API conventions: authentication, sessions, CSRF, pagination, idempotency, error envelope, versioning, events |

### UI

The web UI.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Web UI](ui/ui.md) | `UI` | UI architecture, information architecture, language, data freshness, accessibility |
| [UI journeys](ui/journeys.md) | `UIJ` | User journeys the UI must support |

### Operations

Operations settings, performance metrics, profiling, PKI, deployment and the command line.

| Document | Prefix | Owns |
| --- | --- | --- |
| [Operations](operations/operations.md) | `OPS` | Operations surface and settings, performance metrics, profiling, local PKI, deployment boundary, CLI |

## Registries

| Registry | Defines |
| --- | --- |
| [Limits](platform/limits.md) | Every numeric bound and cardinality target (`LIM-<area>-<n>`) |
| [Error and stop codes](platform/errors.md) | Every error and stop code, its HTTP status and retryability |
| [Failure model](platform/failure-model.md) | Outcome classes and how each external call and mutation behaves (`FAIL-<area>-<n>`) |
| [Entity lifecycle](platform/lifecycle.md) | Who deletes each entity, what blocks it, what cascades and what is retained (`LIFE-<area>-<n>`) |

## Contracts and models outside this directory

| Source | Contents |
| --- | --- |
| [`api/openapi/contractor-public-v1.yaml`](../../api/openapi/contractor-public-v1.yaml) | Public HTTP contract |
| [`api/events/`](../../api/events/) | Public WebSocket event schemas |
| [`api/v1alpha1/`](../../api/v1alpha1/) | Server ↔ Runtime wire contract and shared fixtures |
| [Architecture model](artitecture.likec4) | LikeC4 component map and focused views (`make verify-architecture`) |
| [Database ERD](database-erd.html) | Interactive PostgreSQL schema: tables, keys, purpose and the processes that use them (`make docs-erd`) |
| [Audit and Workflow dependencies](audit-workflow-dependencies.html) | Offline interactive architecture, shipped configuration dependencies and Workflow Stages (`make docs-dependencies`) |
| [`configs/`](../../configs/README.md) | Shipped Workflows, templates, policies and Skills |

## Outside the specification

| Location | Contents |
| --- | --- |
| [`docs/proposals/`](../proposals/) | Designs of capabilities that are not implemented yet; folded into the specification once implemented |
| `tasks/` | What is done, in progress or planned; tasks cite the specification, never the reverse |
