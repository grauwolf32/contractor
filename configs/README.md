# Contractor configuration

This directory contains the executable default configuration. It includes the
small artifact-copy fixture and the four-Stage `openapi-from-source@1` and
`likec4-from-source@1` project workflows. Validate the complete set from the
repository root with:

```sh
go run ./cmd/contractor-server config validate --root ./configs
```

Manifest identity comes from `kind`, `metadata.name`, and `metadata.version`;
file names and nesting are only organizational. Instruction references are
paths relative to this directory.

`llm-gateways/` contains immutable, non-secret endpoint descriptions. The
shipped `local-litellm@1` config uses the OpenAI-compatible `/v1` inference
path and a loopback-only HTTP LiteLLM management origin. Tokens and LiteLLM
admin keys are never valid fields in these manifests.

Workflow `spec.executionConfig` contains reference-only defaults. A modeled
consumer resolves an exact `modelPolicy`, `llmGateway`, and optional non-secret
credential ID; per-Stage and per-Agent leaves override Workflow-wide defaults.
`POST /v1/runs` accepts the same shape as an override, with
`credential: null` as the only explicit clear operation. The resolved bodies,
digests, origins, and credential IDs are pinned in the Run snapshot, while
token bytes are resolved only when constructing a Planner client or Worker
allocation.

For local development only, `CONTRACTOR_LLM_GATEWAY_TOKEN` and
`CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN` bootstrap the fixed credential IDs
`development-worker` and `development-planner` against `local-litellm@1`.
They do not override the Gateway URL or model selected by configuration.

`examples/` contains copyable multi-Stage, bounded-retry, single-Worker
`streamline@1`, and multi-Worker `router@1` Workflow manifests. They are
intentionally outside `workflows/`, so they document supported shapes without
changing the default end-to-end fixture.
