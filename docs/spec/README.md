# Contractor v2 implementation specifications

These documents define what must be implemented. They are normative design
inputs, not a description of the v1 codebase.

The terms **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT** and **MAY** have
their usual RFC 2119 meaning. Every normative requirement has a stable ID so it
can later be linked to code and tests.

## Status model

- `Draft` — can still change during implementation discovery.
- `Accepted` — implementation may depend on the contract.
- `Implemented` — all mandatory acceptance scenarios pass.

## Specification index

| Spec | Scope | Depends on |
|---|---|---|
| [00](00-product-scope.md) | Product scope and non-goals | — |
| [01](01-architecture-boundaries.md) | Components, ports and allowed dependencies | 00 |
| [02](02-domain-contracts.md) | IDs, DTOs and state machines | 00–01 |
| [03](03-adk-integration.md) | Optional Google ADK adapter profile | 01–02 |
| [04](04-database-runtime.md) | PostgreSQL, engine lifecycle and migrations | 01–03 |
| [05](05-artifacts-and-run-state.md) | PgArtifactService and authoritative RunState | 02, 04 |
| [06](06-planner-and-workflow.md) | Planner, DAG and recovery semantics | 02–05 |
| [07](07-agent-runtime-and-a2a.md) | Agent fleet, A2A and attempt lifecycle | 02–06 |
| [08](08-tools-and-sandbox.md) | Framework-neutral tools and attempt-scoped sandbox | 02, 07 |
| [09](09-telemetry.md) | Derived observability and bounded projections | 02, 04, 07 |
| [10](10-control-plane-api.md) | Public API and status projection | 02, 05–07 |
| [11](11-llm-proxy.md) | Shared model gateway contract | 02–03 |
| [12](12-operations-security.md) | Startup, shutdown, failure and security | 04–11 |
| [13](13-testing-and-roadmap.md) | Test matrix, vertical slices and v1 migration | all |

## Dependency order

```text
scope → architecture → contracts → framework / ADK adapter boundary
                                  ↓
                              database → artifacts / RunState → planner / DAG
                                  │                              ↓
                                  ├→ telemetry              agent / A2A → sandbox
                                  └→ LLM Proxy                   ↓
                                           control plane → operations → migration
```

The first executable milestone proves the portable seam with the smallest
deterministic configuration:

```text
publish project snapshot
  → POST /runs
  → create PlannerRunState artifact
  → select a one-node static/passthrough plan
  → commit an Attempt and dispatch outbox record
  → deliver one A2A attempt idempotently
  → run one real Worker strategy
  → stage a Worker result artifact
  → accept/promote it in a Server RunState CAS
  → expose final status and derived observability
```

The same port/conformance suites then add decomposing planning and alternate
ADK/non-ADK Worker implementations without changing the domain or wire DTOs.

## Definition of done for a spec

A spec can become `Implemented` only when:

1. every `MUST` has a code or configuration implementation;
2. every listed acceptance scenario is automated;
3. public DTO fixtures are versioned;
4. failure and cancellation paths are tested;
5. architecture and operational documentation remain consistent.
