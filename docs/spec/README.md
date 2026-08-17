# Contractor reconstruction specification

This directory is the implementation-derived specification for Contractor. It
defines behavior, data contracts, state transitions, safety boundaries, and
acceptance criteria independently of the programming language used to build it.
A conforming implementation may use different libraries or runtime technology,
but it must preserve the observable contracts described here.

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. Examples
are illustrative unless explicitly called normative.

## Reading order

1. [Product and requirements](01-product-and-requirements.md)
2. [System architecture](02-system-architecture.md)
3. [Runtime orchestration](03-runtime-orchestration.md)
4. [Agents, callbacks, and skills](04-agents-callbacks-skills.md)
5. [Workflow catalog and algorithms](05-workflows.md)
6. [Tools and filesystems](06-tools-and-filesystems.md)
7. [Artifacts, HTTP, OpenAPI, and security records](07-artifacts-http-openapi-security.md)
8. [CLI and explorer interfaces](08-cli-and-explorer.md)
9. [Configuration, observability, and deployment](09-configuration-observability-deployment.md)
10. [Testing and acceptance](10-testing-and-acceptance.md)
11. [Reconstruction checklist](11-reconstruction-checklist.md)
12. [Implementation inventory](12-implementation-inventory.md)

## Specification boundary

The specification covers:

- the command-line product and its local explorer;
- model-agent, planner, worker, workflow, tool, memory, artifact, and checkpoint
  behavior;
- all registered workflows and their persistent inputs and outputs;
- source-tree sandboxing, overlay editing, live-target HTTP probing, and optional
  code-execution isolation;
- configuration, event telemetry, evaluation envelopes, and deployment helpers.

It deliberately does not prescribe the source language, web framework, agent
SDK, serialization library, or database driver. Names such as `TaskRunner`,
`trace-graph`, and `vulnerability-reports` are retained because they are stable
product and persistence identifiers, not implementation-language choices.

## Sources and precedence

This specification was derived from executable modules, task and prompt
manifests, workflow configuration, tests, and deployment files. If two sections
appear inconsistent, use this precedence order:

1. data/storage and safety invariants;
2. subsystem-specific specification;
3. architecture overview;
4. examples and explanatory notes.

A future implementation change is incomplete until the affected specification
and acceptance tests are updated together.

## Target contract and compatibility notes

Normative requirements describe the behavior a safe reconstruction must
provide. Files 05–07 also contain explicitly labeled **compatibility gaps**:
these record observable weaknesses or inconsistencies in the inspected working
tree so an implementer can reproduce or migrate existing state knowingly. They
are not permission to weaken a new implementation. When a compatibility note
and a normative safety requirement differ, preserve wire/storage readability
where necessary but implement the normative safety requirement and document the
migration.
