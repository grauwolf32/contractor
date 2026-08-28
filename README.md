# Contractor v2

Contractor is a runtime for complex AI-assisted tasks that benefit from explicit
Workflows and non-trivial planning. Code understanding and application security
are the initial use cases, but the execution model is domain-neutral.

Contractor v2 is a specification-first rewrite focused on narrow component
boundaries and a deployment that remains practical on one VM.

The current design track models a Workflow as product-specific Stages. Workflow
Scheduler selects and executes each ready Stage; one Planner ADK agent works
with a fixed set of Workers prepared before Planner starts. A versioned
`AgentTemplate` describes reusable Worker behavior without becoming a deployment
or physical Agent. A Worker is an allocation-scoped in-process role of a
single-slot Runtime Agent, not a child service or process.

One `ArtifactStore` serves authenticated user artifacts and per-Run working data
through separate `UserScope` and `RunScope` views. Run inputs are version-pinned
forks, not mutable aliases to user data.

The target deployment is deliberately small:

- one Contractor Server process;
- one PostgreSQL database;
- one or more lightweight single-slot Runtime Agent processes, potentially on
  the same VM;
- one shared external LLM Gateway, initially LiteLLM or another compatible
  backend;
- PostgreSQL or S3-backed artifact payload storage;
- optionally, an external telemetry backend.

## Documentation

- [Documentation overview](docs/README.md)
- [Working specifications](docs/spec/README.md)
- [LikeC4 architecture model](docs/spec/architecture.c4)

## Validate the architecture

```shell
likec4 validate docs/spec
```

## Implementation commands

The implementation is being built incrementally in the order recorded under
`tasks/`. The current Go server and Python Runtime Agent checks run with:

```shell
make verify
go run ./cmd/contractor-server config validate --root ./configs
```

Server and migration commands share `CONTRACTOR_DATABASE_URL` (or the
`--database-url` flag). Migrations are forward-only and safe to invoke again:

```shell
CONTRACTOR_DATABASE_URL='postgres://...' \
  go run ./cmd/contractor-server migrate
```

PostgreSQL integration tests create and remove isolated schemas inside the
caller-provided test database:

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://...' make test-postgres
```
