# Contractor configuration

This directory contains the executable default configuration. It includes the
small artifact-copy fixture and the four-Stage `openapi-from-source@1` and
`likec4-from-source@1` project workflows. Those IDs retain their archive-local
`source-analysis@1` behavior. Explicit workspace-backed variants are:

- `openapi-from-workspace@1`;
- `likec4-from-workspace@1`.

Each workspace variant hydrates the exact `inputs/source` ZIP below the private
workspace root for every Stage, uses only bounded `filesystem@1` reads, and
exports exact cumulative `workspace_state` plus checkpoint `workspace_diff`
results. A later Stage imports the exact prior state revision; no allocation
workspace is shared or synchronized implicitly. The domain outputs are the same
as the corresponding source workflow and both workspace results are additionally
frozen as Workflow outputs.

The original LikeC4 identity remains available for reproducibility. Other
project-sized variants are:

- `likec4-from-source@2`: the same deterministic `passthrough@1` graph with
  `project_worker@1` (48 model calls, 256 tool calls, 1,000,000 cumulative
  provider-reported tokens, and 32,768 output tokens per response);
- `likec4-from-source-streamline@1`: the same four Stages and artifact handoffs,
  but each Stage uses a model-backed `streamline@1` Planner with
  `project_planner@1` (48 model calls, 64 Worker calls, 500,000 cumulative
  tokens, and 8,192 output tokens per response) and the same project Worker
  policy.

The LikeC4 Agent Skill is an additive compatibility boundary:

- `likec4_builder@1` and `likec4_validator@1` keep their original, self-contained
  instruction files and declare no Skill;
- `likec4_builder@2` and `likec4_validator@2` keep the mandatory artifact,
  evidence, validation, and completion procedure in always-on instructions and
  select the versionless logical artifact `skills/likec4` for detailed DSL
  guidance;
- `likec4-from-source@1`, `likec4-from-source@2`,
  `likec4-from-source-streamline@1`, and `likec4-from-analysis@1` remain legacy
  selectors using the @1 templates;
- `likec4-from-source@3`, `likec4-from-source-streamline@2`, and
  `likec4-from-analysis@2` preserve their predecessor graphs and select only the
  @2 LikeC4 templates in build and validation Stages.

The package source is `skills/likec4/SKILL.md` plus fourteen on-demand
references. It is packaged and published into the owner-scoped `skills/likec4`
artifact by the Skill Catalog path; configuration refers to the logical binding,
not a checked-in digest.

`security-analysis@1` is an opt-in single-Stage workflow for authorized HTTP
and Caido analysis. It requires the caller to provide `objective`, `target` and
`authorization_scope` string parameters, accepts one optional text context
artifact, and freezes one Markdown report. Its `caido_analyst@1` template pins
the versionless `skills/caido` artifact and explicitly selects every HTTP and
Caido operation named by that package. `http_explorer@1` provides a reusable
HTTP-only template without the Caido Skill or adapter requirement.

Infrastructure is not embedded in those manifests. Before starting the
workflow, publish and bind a RuntimeConfig such as the secret-free
[`caido-analysis@1` example](../deploy/runtime-labels/runtime-configs.example.yaml),
create its write-only credential separately, and add the resulting `caido`
label to the Run. Omitting the label leaves the Stage waiting for a compatible
allocation configuration; it does not silently fall back to an unconfigured or
direct Caido client.

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
The shipped `workflows/` manifests select these IDs explicitly and separately
for Workers and modeled Planners. They do not override the Gateway URL or model
selected by configuration. If the corresponding token is absent, Run creation
fails while resolving the selected credential, before a Stage or allocation is
started. Deployments using different credentials must override these
reference-only selections in the Run `executionConfig` or publish their own
Workflow versions.

`examples/` contains copyable multi-Stage, bounded-retry, single-Worker
`streamline@1`, and multi-Worker `router@1` Workflow manifests. They are
intentionally outside `workflows/`, so they document supported shapes without
changing the default end-to-end fixture.
