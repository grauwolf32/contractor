# Project and document-generation Workflows

[Documentation index](../README.md) · [Local stack](local-stack.md)

These examples use the current Memory-enabled versions in the
[configuration catalog](../../configs/README.md). Start the local Server and set
`CONTRACTOR_API_TOKEN` to its bearer token. Run commands from the repository
root; the HTTP examples also need `curl`, `jq` and Git.

- [Reusable Project workspace](#project-workspace-walkthrough)
- [OpenAPI from source](#openapi-from-a-standalone-run-workspace)
- [Runtime prerequisites](#runtime-workspace-requirements)
- [LikeC4 from source](#likec4-from-a-standalone-run-workspace)
- [Reuse analysis reports](#reusing-explicit-analysis-reports)
- [Budgets and cancellation](#worker-invocation-budgets)

For automated verification, see [testing](../testing/README.md).

## Project workspace walkthrough

Open the independent UI, choose **Projects**, and create a Project. The Project
detail page is the reusable workspace; creating a Project does not start a Run
and a standalone Workflow remains available from **Workflows**.

Use the **Sources** shortcut to upload a ZIP to a stable Project binding such as
`sources/service`. Optional **OpenAPI**, **LikeC4**, **Docs**, **Diffs**, and
**Other** shortcuts only prefill namespace and media type—the identity remains
editable and arbitrary Project artifacts are allowed. Skills remain global
UserScope artifacts and are managed from **Skills**, never copied into a
Project.

The **Recommended Workflows** panel enables only Workflows whose required input
media types can be satisfied by current Project bindings. Select
`openapi-from-workspace@7` or `likec4-from-workspace@7`, review the exact
revision chosen for each input, and start the Project Run. While it is active,
the same Run appears in the top-level **Queue** with its safe Project identity.
Queue is a lifecycle read model, not a second scheduler or a promise of numeric
position.

After success, return to the Project. Frozen Run outputs are published
create-only under `outputs/<workflow-output-name>` and retain exact lineage to
the Run revision. If a primary output already exists, the Workflow moves from
the recommended list to **All workflows** as an explicit **Run again** action;
the UI never prevents deliberate recomputation.

An optional Project application target is configured from **Application
target**. The URL and RuntimeCredential reference are safe metadata. Basic or
Bearer material is accepted only by the write-only credential form, encrypted
in PostgreSQL, and resolved into allocation-private settings only when the
selected AgentTemplate actually exposes `http_request`. It is never a Project
artifact and is not sent to Planner, unrelated Workers, model requests or UI
query state.

## OpenAPI from a standalone Run workspace

`openapi-from-workspace@7` runs four serial Stages: graph-backed dependency
discovery, graph-backed project discovery, incremental OpenAPI construction,
and final validation/repair. Each Stage gets its own allocation and reconstructs
one private workspace from the exact source and cumulative overlay state. The
reports, document and workspace state move between Stages as exact RunScope
artifact revisions rather than model memory or a shared directory.

The catalog also includes `openapi-from-workspace-streamline@2` with a modeled
`streamline@1` Planner in each Stage and the same artifact contracts.

The Runtime Agent host must have `vacuum` on its `PATH`:

```shell
command -v vacuum
vacuum version
```

Create a ZIP with normalized relative entries. For a Git project without
tracked symbolic links, `git archive` is the simplest safe option; it includes
only the committed tree, so commit or otherwise package intentional local
changes first.

```shell
export PROJECT_ROOT='/absolute/path/to/project'
export SOURCE_ZIP='/tmp/contractor-project-source.zip'
git -C "$PROJECT_ROOT" archive --format=zip --output="$SOURCE_ZIP" HEAD
```

Workspace hydration deliberately rejects path traversal, absolute/backslash
paths, duplicate entries, symbolic/special entries, encrypted members and
configured file/archive limits.

This CLI example intentionally exercises the standalone path. Upload the source
as a UserScope artifact; the Run creation transaction copies
the exact revision to `inputs/source`; Workers never read the UserScope binding
directly. For reusable ProjectScope inputs and automatic output publication,
upload the source and optional seed into the Project, then submit their exact
ProjectScope refs to `POST /v1/projects/{project_id}/runs` as in the Project flow.

```shell
SOURCE_REF="$(curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: application/zip' -H 'If-None-Match: *' \
  --data-binary "@$SOURCE_ZIP" \
  http://127.0.0.1:8080/v1/artifacts/projects/project-source | jq -c .artifact)"
```

An existing OpenAPI 3.0/3.1 YAML or JSON document is optional. When supplied,
the build Stage reads its exact input revision and creates an independent
`openapi/openapi` Run binding; the uploaded UserScope value is never mutated.

```shell
OPENAPI_SEED_REF="$(curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: application/yaml' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/openapi.yaml' \
  http://127.0.0.1:8080/v1/artifacts/projects/existing-openapi | jq -c .artifact)"
```

Submit a Run with the optional seed:

```shell
RUN_ID="$(jq -n \
  --argjson source "$SOURCE_REF" \
  --argjson seed "$OPENAPI_SEED_REF" \
  --arg objective 'Document the implemented public HTTP API' \
  '{workflow:"openapi-from-workspace@7",parameters:{objective:$objective},artifacts:{source:$source,existing_openapi:$seed}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: openapi-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"
```

For a new document, omit `existing_openapi` and build the artifact map as
`{source:$source}`. Poll `/v1/runs/$RUN_ID`; after success, retrieve the
document/report outputs. The same Run also freezes `workspace_state` and
`workspace_diff` for exact lineage and future explicit composition:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/openapi" \
  --output /tmp/generated-openapi.yaml

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/openapi_validation_report" \
  --output /tmp/openapi-validation-report.md
```

Intermediate Run bindings are `analysis/dependencies`, `analysis/project`,
the per-namespace cumulative `workspace_state`/checkpoint `workspace_diff`,
`openapi/openapi`, and `openapi/validation-report`. The Workflow succeeds only
when the final `validate_openapi` call reports a structurally clean document and
no serious Vacuum findings. Missing Vacuum is an explicit failed validation,
never an implicit pass.

### Runtime workspace requirements

Every Stage receives a new isolated workspace reconstructed from the exact
source ref and, after the first Stage, the exact cumulative state exported by
its predecessor. With no imported state, the overlay is the canonical empty
delta over the hydrated source. An unchanged Stage still exports an identity
state and an empty diff; there is no implicit join or live workspace sharing.

Workspace support is an immutable Runtime startup capability. The shipped
four-Stage Workflows require **local** storage and the complete Trailmark graph
capability. Their validators additionally require `vacuum` for OpenAPI or
`likec4` for LikeC4 on the Runtime's startup `PATH`. The locked Runtime
environment includes Trailmark; inspect Operations to confirm the positive
startup probes before submitting a Run.

Enable local storage by adding these flags to the complete Runtime command in
the [local-stack guide](local-stack.md#start-a-runtime-agent). Memory storage is
available for other Workflows that do not require the local graph tools:

```shell
# Disposable files below a dedicated local root; binary ZIP members are retained.
contractor-runtime \
  --workspace-storage local \
  --workspace-work-root /var/lib/contractor/project-workspaces \
  ...

# Isolated in-process fsspec storage; binary ZIP members are intentionally skipped.
contractor-runtime --workspace-storage memory ...
```

The optional `--workspace-max-files`, `--workspace-max-expanded-bytes`,
`--workspace-max-managed-text-bytes`, and `--workspace-max-file-bytes` flags
replace backend defaults. They are probed and advertised at registration and
cannot change for the lifetime of that Runtime process. Physical workspace
paths never cross the private Runtime boundary. Direct-mode edits affect only
the disposable hydrated copy; overlay-mode changes become durable only through
the Workflow-declared state/diff Artifact slots.

Text tools operate on managed text files. Binary members may be retained by
local storage, but text tools cannot edit them. Command execution requires the
separately enabled [Podman sandbox](../../runtime/PODMAN.md) and an explicit
`code-execution@1` tool selection; the overlay Workflows described here do not
select it. Allocations, including Router siblings, exchange exact exported
artifacts rather than sharing a live workspace. Keep the local `workRoot` private to the Runtime OS identity: a
same-UID process with write access is part of that host's trust boundary.

## LikeC4 from a standalone Run workspace

`likec4-from-workspace@7` reuses the same graph-backed dependency and project
discovery contracts as the OpenAPI Workflow, then builds and repair-validates
one single-file architecture model. Reports, the model and cumulative overlay
state cross Stage boundaries as exact artifact revisions; no Worker relies on
another allocation's memory or live workspace. The final Run freezes
`likec4`, `likec4_validation_report`, `workspace_state`, and `workspace_diff`.

`likec4-from-workspace-streamline@4` has the same artifact and Stage contract,
but each Stage uses a modeled `streamline@1` Planner. Use it when explicit
subtask decomposition is useful; the `likec4-from-workspace@7` passthrough Workflow is the simpler
default.

Install the LikeC4 CLI directly on every Runtime Agent host and make it visible
on `PATH`. The Runtime never invokes `npx` or downloads a validator while a Run
is executing.

```shell
command -v likec4
likec4 version
```

Create and upload `SOURCE_REF` with the safe ZIP procedure in the OpenAPI
section above. A pre-existing single-file model is optional. When supplied, the
build Stage copies its exact revision into `likec4/architecture`; it never
modifies the UserScope artifact. Both the canonical LikeC4 media type and plain
UTF-8 text are accepted as seeds and normalized to `text/vnd.likec4`.

```shell
LIKEC4_SEED_REF="$(curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/vnd.likec4' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/architecture.c4' \
  http://127.0.0.1:8080/v1/artifacts/projects/existing-likec4 | jq -c .artifact)"
```

Submit a Run with the optional seed:

```shell
RUN_ID="$(jq -n \
  --argjson source "$SOURCE_REF" \
  --argjson seed "$LIKEC4_SEED_REF" \
  --arg objective 'Model the implemented architecture and trust boundaries' \
  '{workflow:"likec4-from-workspace@7",parameters:{objective:$objective},artifacts:{source:$source,existing_likec4:$seed}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: likec4-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"
```

For a new model, omit `existing_likec4` and use
`artifacts:{source:$source}`. Poll `/v1/runs/$RUN_ID`; after success, retrieve
the immutable Workflow outputs:

```shell
curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/likec4" \
  --output /tmp/generated-architecture.c4

curl --fail --silent --show-error \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  "http://127.0.0.1:8080/v1/runs/$RUN_ID/outputs/likec4_validation_report" \
  --output /tmp/likec4-validation-report.md
```

Intermediate bindings are `analysis/dependencies`, `analysis/project`,
`likec4/architecture`, and `likec4/validation-report`. The supported MVP is one
self-contained file: includes, multi-file projects, rendering, and layout
artifacts are outside this workflow. It succeeds only after direct
`likec4 validate` reports a clean model. A missing or failed CLI is an explicit
retryable validation failure, never an implicit pass.

## Reusing explicit analysis reports

`openapi-from-analysis@4` and `likec4-from-analysis@5` are two-Stage variants
for callers that already have reviewed dependency and project reports. They do
not search prior Runs or choose a current artifact implicitly. The caller must
upload exact `text/markdown` reports to UserScope and select their revisions
together with the exact source revision. Run creation copies them to
`inputs/dependency_report` and `inputs/project_report`; Workers may mutate only
RunScope artifacts and the uploaded values remain unchanged.

```shell
DEPENDENCY_REPORT_REF="$(curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/markdown' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/dependency-report.md' \
  http://127.0.0.1:8080/v1/artifacts/projects/dependency-report | jq -c .artifact)"

PROJECT_REPORT_REF="$(curl --fail --silent --show-error -X PUT \
  -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
  -H 'Content-Type: text/markdown' -H 'If-None-Match: *' \
  --data-binary '@/absolute/path/to/project-report.md' \
  http://127.0.0.1:8080/v1/artifacts/projects/project-report | jq -c .artifact)"

RUN_ID="$(jq -n \
  --argjson source "$SOURCE_REF" \
  --argjson dependencies "$DEPENDENCY_REPORT_REF" \
  --argjson project "$PROJECT_REPORT_REF" \
  '{workflow:"openapi-from-analysis@4",parameters:{},artifacts:{source:$source,dependency_report:$dependencies,project_report:$project}}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H "Idempotency-Key: openapi-analysis-run-$(date +%s)" \
    -H 'Content-Type: application/json' --data-binary @- \
    http://127.0.0.1:8080/v1/runs | jq -r .runId)"
```

Use `likec4-from-analysis@5` in the same request to produce LikeC4. Optional
`existing_openapi` and `existing_likec4` inputs retain the contracts described
above. Contractor v1alpha1 does not prove that a report revision was derived
from the selected source revision; the caller owns that compatibility decision.
A future provenance/cache policy can automate it without changing these
explicit workflow contracts.

## Worker invocation budgets

Worker ModelPolicies declare `maxModelCalls`, `maxToolCalls`, and
`maxTotalTokens` in addition to per-response `maxOutputTokens`. The Runtime
applies the cumulative limits to one Worker A2A invocation, including its
mandatory tool-free result-finalizer call after an ordinary terminal response.
Model and tool capacity is checked before starting the next operation. Provider
`total_tokens` is accumulated
after each response; a response that crosses the limit cannot execute its tool
call. Missing usage is reported as unavailable and model/tool limits remain
active.

Budget exhaustion returns a retryable failed Worker result with stable code
`worker_budget_exhausted`. The Worker report includes configured and observed
limits plus `model_calls`, `tool_calls`, or `total_tokens` as the exhausted
dimension, without retaining prompts or payloads. Workflow `maxAttempts` still
owns whole-Stage retry and the Server Planner deadline remains the outer wall
limit. Tune the YAML policy and create a new policy digest; do not patch Runtime
constants or treat `maxOutputTokens` as a cumulative ceiling.

Cancellation is an idempotent durable request. It interrupts an active local
Planner immediately; another Server process observes the same `cancelling`
state while renewing its claim. To cancel instead of waiting for the result:

```shell
jq -n --arg reason 'no longer needed' '{reason:$reason}' | \
  curl --fail --silent --show-error \
    -H "Authorization: Bearer $CONTRACTOR_API_TOKEN" \
    -H 'Content-Type: application/json' --data-binary @- \
    "http://127.0.0.1:8080/v1/runs/$RUN_ID/cancel"
```

The response is `202 Accepted` while bounded cleanup is in progress and `200 OK`
when the Run was already terminal. Repeating the request never replaces
the first cancellation reason or changes a successful terminal Run.

Using a real LiteLLM/provider is deliberately opt-in. Publish its non-secret
endpoint as an exact `LLMGatewayConfig`, publish model aliases and bounds as
exact `ModelPolicy` documents, then select them independently for Planner and
Workers through `executionConfig`. For local bootstrap, the Worker and Planner
may use the separately named `development-worker` and `development-planner`
credentials described in the [local-stack guide](local-stack.md). CI and `make test-e2e` use a temporary authored
Gateway manifest and a deterministic fake model rather than an external
provider.
