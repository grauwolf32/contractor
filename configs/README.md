# Contractor configuration

This directory contains the executable default configuration. It includes the
small artifact-copy fixture and the four-Stage `openapi-from-source@1` and
`likec4-from-source@1` project workflows. The original LikeC4 identity remains
available for reproducibility. New project-sized variants are:

- `likec4-from-source@2`: the same deterministic `passthrough@1` graph with
  `project_worker@1` (48 model calls, 256 tool calls, 1,000,000 cumulative
  provider-reported tokens, and 32,768 output tokens per response);
- `likec4-from-source-streamline@1`: the same four Stages and artifact handoffs,
  but each Stage uses a model-backed `streamline@1` Planner with
  `project_planner@1` (48 model calls, 64 Worker calls, 500,000 cumulative
  tokens, and 8,192 output tokens per response) and the same project Worker
  policy.

All limits remain finite and are enforced per Planner or Worker invocation.
Select the passthrough variant when one Worker can follow the complete Stage
contract directly; select Streamline when the Planner should decompose that
Stage into ordered subtasks for its one fixed logical Worker. Validate the
complete set from the repository root with:

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

Managed credentials bind that exact digest-bearing Gateway ref to an
owner-only admin-key file through the separate process bootstrap document
passed with `--llm-gateway-admin-bindings-file`. The binding document is not a
configuration resource, is never served by the public API, and must be updated
explicitly when the immutable Gateway digest changes.

Workflow `spec.executionConfig` contains reference-only defaults. A modeled
consumer resolves an exact `modelPolicy`, `llmGateway`, and optional non-secret
credential ID; per-Stage and per-Agent leaves override Workflow-wide defaults.
`POST /v1/runs` accepts the same shape as an override, with
`credential: null` as the only explicit clear operation. The resolved bodies,
digests, origins, and credential IDs are pinned in the Run snapshot, while
token bytes are resolved only when constructing a Planner client or Worker
allocation.

`execution-configs/` contains immutable Stage-local escalation patches. A
Workflow `failed` or `interrupted` action may select one exact profile ref, or
declare the same `planner`/`agents` patch inline. The loader resolves and
validates every possible variant, and Run creation reapplies it over that
Run's base overrides before pinning the complete effective configuration.
Credential `null` is an explicit clear in this patch shape.

For local development only, `CONTRACTOR_LLM_GATEWAY_TOKEN` and
`CONTRACTOR_PLANNER_LLM_GATEWAY_TOKEN` bootstrap the fixed credential IDs
`development-worker` and `development-planner` against `local-litellm@1`.
They do not override the Gateway URL or model selected by configuration.

`examples/` contains copyable multi-Stage, bounded-retry, single-Worker
`streamline@1`, and multi-Worker `router@1` Workflow manifests. They are
intentionally outside `workflows/`, so they document supported shapes without
changing the default end-to-end fixture.
