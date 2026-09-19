# Live-model evaluation

[Testing overview](README.md) · [Local stack](../guides/local-stack.md)

These opt-in commands call a real Gateway/model and can consume its budget.
They are separate from deterministic release checks. Run each command block
from the repository root unless it begins with `cd runtime`. Install the
locked Runtime dependencies first.

- [Local LiteLLM and Router evaluation](#live-router-evaluation-through-litellm)
- [Planner dialect check](#planner-gateway-dialect)
- [Worker and summarizer checks](#worker-and-summarizer-gateway-checks)
- [Project-workflow quality evaluation](#live-project-workflow-quality-evaluation)

## Live Router evaluation through LiteLLM

Provision the Gateway using [the deployment guide](../deployment.md#llm-gateway).
The checked-in LiteLLM profile exposes `planner-model` and `worker-model`.

In another terminal, run the bounded real-model Router scenario:

```shell
CONTRACTOR_LIVE_LLM_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_LIVE_LLM_TOKEN='replace-with-litellm-master-key' \
CONTRACTOR_LIVE_LLM_MODEL='planner-model' \
  make test-live-routing
```

The wrapper `scripts/test-live-routing.sh` invokes the same target. Set
`CONTRACTOR_LIVE_LLM_MODEL=planner-model` when using it with the checked-in
LiteLLM profile; its older default model name is not a published proxy alias.

Use a token accepted by the Gateway; the local profile above requires one. The
evaluation uses the production Go ADK Router and exact four-tool contract,
offers builder and reviewer descriptions, and requires exactly one dispatch to
reviewer. The Worker and artifact lookup are deterministic because the normal
E2E gate already covers A2A and mTLS. Success prints only a model-name SHA-256,
bounded call count, selected logical Worker, and outcome. Failure reports a
stable Contractor error code; inspect local LiteLLM/LM Studio logs separately
when provider diagnostics are needed. Provider bodies and tokens are never
printed by the test command.

The live check found and now guards an important dialect detail: no-argument
tools still carry an explicit closed JSON object schema with `properties: {}`.
LM Studio rejects the otherwise equivalent schema when that field is omitted.

## Planner Gateway dialect

The live Gateway dialect check is opt-in so normal tests remain deterministic.
It verifies that a deployed LiteLLM or LM Studio model returns a real function
tool call through the same Go adapter:

```shell
CONTRACTOR_STREAMLINE_LIVE_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_STREAMLINE_LIVE_GATEWAY_TOKEN='replace-with-gateway-token' \
CONTRACTOR_STREAMLINE_LIVE_MODEL='planner-model' \
  go test -count=1 -run TestLiveGatewayToolCall ./tests/integration/streamline
```

## Worker and summarizer Gateway checks

To smoke-test the real ADK/LiteLLM adapter and provider-supplied token usage
without making the deterministic suite depend on a live model, point the
opt-in test at any OpenAI-compatible gateway:

```shell
cd runtime
CONTRACTOR_LIVE_LLM_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_LIVE_LLM_GATEWAY_TOKEN='replace-with-gateway-token' \
CONTRACTOR_LIVE_LLM_MODEL='worker-model' \
  uv run pytest tests/test_live_gateway.py
```

The same opt-in module can evaluate the fixed `terminal@1` summarizer contract
against LM Studio through LiteLLM. This is a dialect/quality check, not a
release dependency:

```shell
cd runtime
CONTRACTOR_LIVE_LLM_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_LIVE_LLM_GATEWAY_TOKEN='replace-with-gateway-token' \
CONTRACTOR_LIVE_LLM_MODEL='worker-model' \
  uv run pytest -v tests/test_live_gateway.py \
    -k live_terminal_summarizer_returns_one_strict_worker_result
```

A pass means the Gateway/model accepts ADK structured output, preserves the
exact subtask ID and returns one non-empty `WorkerModelResult` in one logical
call. Provider usage may either be present or be explicitly counted as
unavailable. Schema rejection, free text, a changed subtask ID, a second call,
or a Gateway timeout is a failed evaluation. The deterministic release proof
remains `make test-worker-summarizer-e2e`; it never contacts LM Studio.

To exercise the ordinary Worker boundary against a backend that cannot combine
tools and JSON Schema, run the separated tool-bearing main Agent plus mandatory
tool-free result finalizer:

```shell
cd runtime
CONTRACTOR_LIVE_LLM_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_LIVE_LLM_GATEWAY_TOKEN='replace-with-gateway-token' \
CONTRACTOR_LIVE_LLM_MODEL='worker-model' \
  uv run pytest -v tests/test_live_gateway.py -k result_finalizer
```

A pass proves that the main ADK request contains tools without an output
schema, the following isolated ADK Agent uses the strict schema without tools,
and Runtime accepts the exact copied text as an unsummarized Worker result.

## Live project-workflow quality evaluation

This gate is deliberately excluded from `make verify` and ordinary CI. It
requires PostgreSQL, `vacuum`, `likec4`, the locked Python environment, and a
model with reliable OpenAI-compatible function calling. The fixture is a small
FastAPI service with authenticated GET/POST routes, PostgreSQL persistence, and
an outbound Inventory HTTP client. Both workflows start from source only, with
no prebuilt OpenAPI or LikeC4 seed.

`CONTRACTOR_WORKFLOWS_LIVE_MODEL` selects the Worker model alias sent to the
Gateway. The harness resolves the selected Workflows, updates their effective
Worker policies and Gateway URLs in an isolated catalog, then reloads that
catalog to recompute exact identities. Selection uses manifest identities,
not policy filenames or the previous model alias. Production catalog files,
policy limits, sampling settings and separate Planner/summarizer models stay
unchanged. An unresolved policy/route fails before stack startup; a Worker
policy shared with another model role is rejected instead of silently changing
that role. The selected alias must exist at the Gateway; rejection does not
switch to the default model. The checked-in
[LiteLLM profile](../../deploy/litellm/litellm_config.yaml) exposes `worker-model`.

```shell
CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable' \
CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL='http://127.0.0.1:4000/v1' \
CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_TOKEN='replace-with-litellm-master-key' \
CONTRACTOR_WORKFLOWS_LIVE_MODEL='worker-model' \
  make test-project-workflows-live
```

Allow up to 30 minutes for each four-Stage workflow, plus setup and cleanup.
The harness still selects `openapi-from-workspace@5` and
`likec4-from-workspace@5`. They currently resolve `worker@2`; inspect the exact
[policy source](../../configs/model-policies/worker.yaml) for its limits.
Actual model/tool/token counters are reported by Stage in the Run status.
On failure, the harness writes bounded generated documents, analysis reports, predicate
codes, and counters below ignored `.local/eval-results/`. It deliberately does
not persist the source archive, Gateway URL/token, prompts, or provider response
bodies. A successful run removes its isolated schema, certificates, processes,
and Runtime workspaces without retaining evaluation artifacts.

Failure-summary schema `1.1` replaces the ambiguous `modelSha256` field with
`requestedModelAliasSha256` and per-Workflow/Stage/Worker `modelSelections`:
effective ModelPolicy and Gateway refs/digests plus `modelAliasSha256`.
`modelSelectionBasis: resolved_configuration` identifies what was observed;
`upstreamModelRevision: null` means the harness cannot attest backend weights
or revision from a Gateway alias. These are configuration provenance, not
proof of model quality or evidence that every configured Stage executed.

The selection regression can be checked offline with the locked Runtime
environment installed:

```shell
go test -count=1 ./tests/eval/project_workflows -skip '^TestLive'
```

It sends all eight resolved Worker selections through the actual Python
Gateway factory using an in-memory HTTP transport, checks successful requests
and model-not-found responses without fallback, and verifies safe provenance
and unchanged catalog bytes. It requires no database or model service and
fails, rather than skips, if the Python bridge prerequisites are absent.

During local diagnosis only, set `CONTRACTOR_WORKFLOWS_LIVE_ONLY` to either
`openapi-from-workspace@5` or `likec4-from-workspace@5`; the default and documented
quality gate always execute both.
