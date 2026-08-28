# Contractor v2

Contractor is an AI-assisted software-understanding and application-security
product that produces OpenAPI descriptions, architecture models, source/data-
flow traces, vulnerability findings and verification/exploitability evidence
from a project snapshot and explicit analysis objective.

Contractor v2 is a specification-first rewrite focused on low coupling and
explicit process boundaries while preserving those product outcomes.

The current design track models a Workflow as product-specific Stages. Workflow
Scheduler selects and executes each ready Stage; one Planner ADK agent works
with a fixed set of ephemeral A2A Worker Agents prepared before Planner starts.
A versioned `AgentTemplate` describes reusable Worker behavior without becoming
a deployment or physical Agent. One `ArtifactStore` serves authenticated user
artifacts and per-Run working data through separate `UserScope` and `RunScope`
views; Run inputs are version-pinned forks, not mutable aliases to user data.

The target deployment is deliberately small:

- one Contractor Server process;
- one PostgreSQL database;
- one or more lightweight single-slot Runtime Agent processes, potentially on
  the same VM;
- ephemeral Worker child processes or containers;
- one shared external LLM Proxy for enabled model-backed strategies;
- optionally, an external OpenTelemetry sink for sampled traces.

The repository currently contains competing design iterations. `docs/spec-2`
is the current working set; the earlier detailed candidate is retained for
comparison rather than inherited implicitly.

## Documentation

- [Current working specifications](docs/spec-2/README.md)
- [Current LikeC4 model](docs/spec-2/architecture.c4)
- [Earlier candidate architecture](docs/README.md)
- [Earlier candidate specifications](docs/spec/README.md)

## Validate the architecture

```shell
likec4 validate docs/spec-2
```
