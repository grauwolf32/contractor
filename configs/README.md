# Contractor configuration

This directory contains the executable default configuration. Its current
user-facing Workflow set is:

[`server.local.yaml`](server.local.yaml) is a separate, non-secret
`ServerConfig` for the loopback demo process. It is not a published catalog
resource and the Workflow configuration loader ignores root-level YAML files.
Start the process with `contractor-server serve --config
./configs/server.local.yaml`; relative paths in that document resolve from the
document's directory. Database URLs, public bearer tokens, development LLM
tokens and all key bytes remain environment/file secrets and are deliberately
not valid `ServerConfig` fields.

- `openapi-from-workspace@5` and `likec4-from-workspace@5` for four-Stage local
  graph-backed analysis and document generation;
- `openapi-from-workspace-streamline@1` and `likec4-from-workspace-streamline@2`
  for the same document contracts with a modeled single-Worker Planner;
- `openapi-from-analysis@2` and `likec4-from-analysis@3` when the caller already
  has exact dependency and project reports;
- `security-analysis@2` and `taint-trace-from-workspace@2` for their explicit
  focused analysis contracts;
- `audit-source-check@1` for bounded task sets in `source-checklist@1`;
- `audit-openapi-operation-trace@1` for graph-backed operation analysis in
  the limited `openapi-operation-trace@2` comparison variant;
- `audit-openapi-operation-trace@2` for graph-backed analysis and finding proposals
  in `openapi-operation-trace@3`, plus `findings-review@1` for a supplied collection;
- `audit-top10-source-risk@1` and `audit-asvs-source-verification@1` for the
  standard-specific AuditProfiles;
- `artifact-copy@1` as the ordinary artifact smoke fixture;
- `podman-python-check@1` as the explicitly selected offline Podman execution fixture.

Historical source-analysis and earlier workspace/Skill versions are deleted,
not archived or republished. Existing Runs retain their complete resolved
Workflow snapshots; a deleted exact identity is never reused. Process-only
Router/Streamline Memory fixtures live under `configs/e2e` and do not enter the
default catalog.

The two initial AuditProfiles are deliberately generic, one-round and
non-certifying. They require an exact source ZIP plus either a custom checklist
or OpenAPI document. Their child Worker reads the Controller-generated task and
execution manifest, performs bounded source analysis and uses
`audit-results@1/read_audit_task` to obtain its validated ordered JSON task set
and `audit-results@1/submit_check_result` to package one complete strict result
set without asking the model to decode ZIP bytes, reproduce item identities, or
construct ZIP bytes manually. The checklist profile currently batches up to two
compatible items; the OpenAPI profile retains `batchSize: 1`.

`openapi-operation-trace@2` selects `audit_openapi_operation_tracer@1`, with
all eleven `code-analysis@1` operations and bounded filesystem reads. Its
read-only tools inspect an overlay workspace hydrated from the exact Audit
`source` input. The existing task/execution-manifest and result ZIP contracts
are retained. Graph coverage and source evidence inform the trace; the current
coverage key remains `operation-resolution`. This limited comparison variant
does not export annotations or propose findings. Version 1 is
superseded in the default catalog; existing Audits retain their pinned snapshot.
A Runtime must advertise local workspace and graph capabilities to run version 2.

`openapi-operation-trace@3` selects `audit_openapi_operation_tracer@2`, adding
`security-findings@2.finding` and `text-artifacts@1.write_text_artifact` with
`findingConfirmation: human-required`. It preserves the operation inventory and
prohibits active checks. Findings do not turn `operation-resolution` into full
trace coverage. `findings-review@1` independently reads a pinned collection with
`list_findings` and publishes an analysis report; it does not confirm findings.

`audit-standards/` is the operator-owned source for immutable curated standard
packages. Each immediate package directory contains only a strict
`standard.json`; Server startup validates and canonicalizes the complete set
before create-only seeding into the protected owner catalog. AuditProfiles name
only `(scheme, version)`. Audit start resolves and retains the exact package
revision and license provenance; changing content under an existing identity
is fatal drift, so changed content must use a new version.

The document-generation overlay workspace Workflows hydrate the exact `inputs/source` ZIP below a
private project-workspace root for every Stage and uses bounded filesystem/code
tools. It exports exact cumulative `workspace_state` plus checkpoint
`workspace_diff`; a later Stage imports the exact prior state revision. With no
imported state, the cumulative overlay is the canonical empty delta over the
hydrated sources. An unchanged Stage therefore exports an identity state and
empty diff. These are internal lineage artifacts retained for future explicit
Workflow composition; no join behavior or implicit workspace sharing exists.

`podman-python-check@1` is a separate opt-in **direct** workspace example. It
selects `podman_python_fixer@1`, fixes a small offline Python fixture and publishes
only an explicitly written JSON report. It neither exports nor imports overlay
state, mounts Skills nor downloads packages. See the [local Podman setup and
source ZIP recipe](../runtime/README.md#local-podman-workflow). Existing agents
need no policy changes; only agents advertising verified local/direct Podman
execution can accept this workflow.

The discovery Stages use `workspace_source_graph_analyst@1` and select all
eleven bounded structural operations. They wait for a Runtime Agent whose
positive capability includes the complete local Trailmark graph surface; there
is no automatic downgrade to shallow analysis.

The passthrough workspace Workflows pin `domain_worker@2`. It retains the
24-model-call, 96-tool-call, and 16,384-token per-response bounds from
`domain_worker@1`, while raising the cumulative provider-reported token budget
from 250,000 to 500,000 so a completed artifact still has room for terminal
result finalization.

`openapi-from-workspace-streamline@1` and `likec4-from-workspace-streamline@2`
retain the exact graph-backed workspace, artifact handoffs, cumulative state/diff
and output contracts of their respective `*-from-workspace@5` variants, with
`streamline@1` in every Stage. Their modeled Planners use `project_planner@1`;
their Workers use `project_worker@1`. OpenAPI retains the `strong-oas-review@1`
validator escalation, including the modeled Planner in its effective policy. Choose these
when each semantic Stage benefits from explicit ordered subtask decomposition;
the passthrough variant remains the simpler default when one Worker invocation
can own the complete Stage objective.

For larger local runs, the opt-in `local_project_worker@1` policy permits
128 model calls, 512 tool calls, 8,000,000 cumulative tokens and 32,768 output
tokens per response. `local_project_planner@1` permits 96 model calls, 64 Worker
calls, 2,000,000 cumulative tokens and 8,192 output tokens per response. Select
these through Run execution overrides; they do not replace shared Workflow
defaults. Passthrough has no modeled Planner and accepts only the Worker
override. These budgets assume a local deployment; they are not cost-safe
defaults for a metered provider. Model context capacity, individual request
timeouts and Stage deadlines remain separate bounds. See the
[local stability run ledger](../docs/reviews/2026-09-07-local-workflow-stability.md)
for the tested model configuration and actual outcomes.
Version 2 of both local policies retains those budgets and uses temperature
1.0 for Qwen3.8 thinking-mode sampling; version 1 retains the diagnostic 0.1
setting. Sampling parameters are model-specific, not universal defaults.

Both current LikeC4 template families select the versionless logical
`skills/likec4` artifact for detailed DSL guidance while retaining mandatory
artifact, evidence, validation and completion rules in always-on instructions.
The package source is `skills/likec4/SKILL.md` plus its on-demand references. It
is packaged and published into the owner-scoped `skills/likec4` artifact by the
Skill Catalog path; configuration refers to the logical binding, not a
checked-in digest.

`security-analysis@2` is an opt-in single-Stage workflow for authorized HTTP
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
