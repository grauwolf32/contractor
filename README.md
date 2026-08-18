# Contractor v2

Contractor is an AI-assisted software-understanding and application-security
product that produces OpenAPI descriptions, architecture models, source/data-
flow traces, vulnerability findings and verification/exploitability evidence
from a project snapshot and explicit analysis objective.

Contractor v2 is a specification-first rewrite focused on low coupling and
explicit process boundaries while preserving those product outcomes.

The Server can use a decomposing Planner, a static workflow manifest, or a
deterministic one-root passthrough Planner. The receiving A2A Worker may execute
directly or plan and work internally, using ADK or another implementation
behind the same portable contracts.

The target deployment is deliberately small:

- one Contractor Server process;
- one PostgreSQL database;
- one or more Contractor Agent processes;
- one shared external LLM Proxy for enabled model-backed strategies;
- optionally, an external OpenTelemetry sink for sampled traces.

The repository currently contains the target architecture and implementation
specifications. Application code will be added through the vertical slices
defined in the roadmap.

## Documentation

- [Architecture model](docs/architecture.c4)
- [Architecture and ownership guide](docs/README.md)
- [Implementation specifications](docs/spec/README.md)
- [Implementation roadmap](docs/spec/13-testing-and-roadmap.md)

## Validate the architecture

```shell
likec4 validate docs
```
