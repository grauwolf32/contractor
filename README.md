# Contractor

Contractor runs AI-assisted Workflows for code understanding and application
security: generating OpenAPI and LikeC4 documents, analyzing source code, and
coordinating Audits with findings and evidence.

A Workflow defines Stages and their inputs, outputs and transitions. The Go
Server schedules each Stage; its Planner coordinates Workers running inside
single-slot Python Runtime Agents. Versioned configuration and pinned artifact
revisions make each Run's inputs and execution settings explicit.

The application can run on one host: Server, PostgreSQL, one or more Runtime
Agents, and an optional independently built React UI served by Node. Model
calls go through an external OpenAI-compatible Gateway. Artifact payloads use
PostgreSQL by default or a configured filesystem backend.

[![Contractor architecture: control plane, Runtime Agents, allocations, and infrastructure adapters](docs/assets/architecture-overview.png)](docs/assets/architecture-overview.png)

## Get started

Follow [deployment](docs/deployment.md) to install and run Contractor, or the
[CLI guide](docs/guides/cli.md) to connect to an existing Server. To build from
source, start with [development setup](docs/development.md).

## Documentation

[Guides and references](docs/README.md) · [Specification](docs/spec/README.md) ·
[Configuration catalog](configs/README.md)

## Repository map

| Path | Contents |
| --- | --- |
| [`cmd/`](cmd/) and [`internal/`](internal/) | Go CLI, Server and execution services |
| [`runtime/`](runtime/README.md) | Python Runtime Agent and toolsets |
| [`ui/`](ui/README.md) | React UI and Node static service |
| [`api/`](api/README.md) | Public OpenAPI, private schemas and shared fixtures |
| [`configs/`](configs/README.md) | Versioned Workflows, policies, instructions and Skills |
| [`deploy/`](deploy/) | Deployment examples |
| [`docs/`](docs/README.md) | Guides, specifications and research |
| [`tests/`](tests/) and [`testdata/`](testdata/) | Integration tests, evaluations and fixtures |
| [`tasks/`](tasks/index.yml) | Implementation status and verification evidence |

The v2 [working specifications](docs/spec/README.md) define accepted contracts;
individual task files record their implementation and verification status.
