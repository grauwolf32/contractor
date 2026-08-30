# Contractor v2

Contractor is a runtime for complex AI-assisted tasks that benefit from explicit
Workflows and non-trivial planning. Code understanding and application security
are the initial use cases, but the execution model is domain-neutral.

Contractor v2 is a specification-first rewrite focused on narrow component
boundaries and a deployment that remains practical on one VM.

The current design track models a Workflow as product-specific Stages. Workflow
Scheduler selects and executes each ready Stage; one Planner works with a fixed
set of Workers prepared before Planner starts. The deterministic
`passthrough@1` Planner invokes one Worker, while the model-backed
`streamline@1` Planner uses Google ADK Go and a bounded typed subtask plan to
execute one prepared logical Worker. Multi-Worker selection belongs to the
separate `router@1` Planner contract. A
model-backed Planner reports either success or semantic failure through one
validated `finish` candidate; Workflow Scheduler alone chooses retry,
escalation, or another Workflow transition. A
versioned `AgentTemplate` describes reusable Worker behavior without becoming a
deployment or physical Agent. A Worker is an allocation-scoped in-process role
of a single-slot Runtime Agent, not a child service or process.

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
- [Local development and end-to-end MVP](docs/development.md)
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
CONTRACTOR_TEST_DATABASE_URL='postgres://...' make test-e2e
```

For a local private mTLS deployment, generate the CA and both node identities
without OpenSSL-specific shell scripts:

```shell
go run ./cmd/contractor-pki init-ca
go run ./cmd/contractor-pki issue-control-plane
go run ./cmd/contractor-pki issue-agent --name agent-local
make test-mtls
```

Generated certificates and 0600 private keys live under `.local/pki/` and are
ignored by Git. `init-ca` refuses to replace an existing CA unless `--force` is
explicitly supplied.
