# Specification conventions

[Specification index](README.md)

`docs/spec/` is the reference specification of Contractor: how the system is
meant to behave now, organised by domain. It is normative. This document says
what belongs in it, how each document is shaped and how documents refer to
each other.

## What belongs where

| Content | Location |
| --- | --- |
| Current intended behaviour, contracts, invariants, limits, errors | `docs/spec/<domain>/` |
| Cross-domain registries (limits, error codes, failure outcomes, entity lifecycle, security, observability) | `docs/spec/platform/` |
| Design of a capability that is not implemented yet | `docs/proposals/NNNN-slug.md` |
| What is done, in progress or planned, and verification evidence | `tasks/` |
| How to use the system | `docs/guides/`, `docs/operations/` |

A specification never refers to tasks: no task IDs, no links into `tasks/`, no
implementation status ("implemented", "verified by"), no delivery increments
or release plans. Tasks refer to the specification, never the reverse.
`make lint` enforces this for every migrated document.

## Domains

Each domain is a directory. A concept has exactly one owning document; other
documents link to it instead of restating it.

| Directory | Owns |
| --- | --- |
| [`platform/`](platform/) | Rules shared by every domain: limits, error codes, failure outcomes, entity lifecycle, security model, observability, persistence conventions |
| `configuration/` | The configuration catalog: identities and selectors, Workflows, AgentTemplates, ModelPolicies, ExecutionConfigs, RuntimeConfig, labels, credentials and Gateways |
| `execution/` | WorkflowRuns, the Scheduler, the Control Plane, Runtime Agents, Planners and Workers |
| `artifacts/` | The ArtifactStore, Projects, retention and deletion, blob storage and Git import |
| `tools/` | Toolsets available to Workers: memory, Skills, workspaces, code analysis, HTTP and Caido, scanners, code execution |
| `audits/` | Audits: profiles, baselines, rounds, inventories, results, review and reports |
| `findings/` | Findings, collections and their tools |
| `evals/` | The portable evaluation format and managed Evals |
| `api/` | Public API conventions shared by every resource |
| `ui/` | The web UI: architecture, language, navigation and data freshness |
| `operations/` | Operations settings, performance metrics, profiling, PKI and deployment |

## Document shape

Every domain document uses these sections in this order, omitting the ones
that do not apply:

1. **Purpose and scope.** What the document owns and what it leaves to others.
2. **Concepts.** Terms used here, linked to [the glossary](glossary.md).
3. **Model.** Entities, their fields and state machines.
4. **Requirements.** Numbered normative statements (see below).
5. **Interfaces.** Wire formats and APIs, linking to the schemas in `api/`.
6. **Limits, failures and errors.** References to the platform registries.
7. **Security.**
8. **Observability.**
9. **Invariants.** Properties that always hold, each naming its owning test
   when one exists.
10. **Examples.**
11. **Non-goals.** What is deliberately not supported.

Rationale may follow a requirement as a short "Rationale:" paragraph.

## Requirements and IDs

A requirement is one paragraph that starts with its bold ID, for example
`**ART-R12.** A revision is immutable once published.` IDs use the document's
prefix (listed at the top of each document), are never renumbered and are
never reused after deletion. Use "must", "must not", "should" and "may" with
their RFC 2119 meaning.

Refer to a requirement as `[ART-R12](../artifacts/artifact-store.md)`; tasks
refer to it as `docs/spec/artifacts/artifact-store.md#ART-R12` in
`context.specification`.

## Registries

A value or rule that more than one component enforces is defined once, in a
registry under `platform/`, and every document links to the entry instead of
repeating it:

| Registry | Entry ID | One row per |
| --- | --- | --- |
| [limits](platform/limits.md) | `LIM-<area>-<n>` | Numeric bound or cardinality target, with its value, unit, enforcement points and owner requirement |
| [errors](platform/errors.md) | the error code itself | Error or stop code, with HTTP status, retryability and meaning |
| [failure model](platform/failure-model.md) | `FAIL-<area>-<n>` | External call or mutation, with its outcome classes and reconciliation |
| [lifecycle](platform/lifecycle.md) | `LIFE-<area>-<n>` | Entity, with who deletes it, what blocks deletion, cascades and retention |

`<area>` is the requirement prefix of the owning document (`LIM-ART-3` is the
third limit owned by `artifacts/artifact-store.md`).

Registries are Markdown tables today; the intent is that validators and
shared test fixtures in Go, Python and the UI read machine-readable versions of
them, so a rule exists in exactly one place.

## Proposals

A capability that is designed but not implemented is a proposal in
`docs/proposals/NNNN-slug.md`. A proposal may track status and link tasks. The
task that completes the capability folds its normative content into the
reference specification and marks the proposal as folded.
