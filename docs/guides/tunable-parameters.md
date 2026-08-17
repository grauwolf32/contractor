# Tunable Parameters Reference

A consolidated catalog of every tunable parameter in Contractor — env-driven
`Settings`, per-workflow `config.yaml`, task templates, planner/subtask caps,
callbacks, tool limits, CLI flags, eval overrides, and the LiteLLM deploy.

Generated 2026-06-07, last synced 2026-06-11. Line numbers are indicative;
treat the source file as authoritative.

**Where config lives (mental model):**

| Layer | Mechanism | Scope | Edit by |
|---|---|---|---|
| Global defaults | `Settings` (pydantic-settings) | env / `cli/.env` | env var or `.env` |
| Per-workflow | `contractor/workflows/<mode>/config.yaml` | one workflow | edit YAML |
| Per-task | `contractor/tasks/*.yml` + `add_task(...)` | one task | YAML / assembler |
| Planner/subtask | function-default args | per planner | code |
| Callbacks | constructor args (mostly hardcoded) | per agent | code |
| Run-time | CLI flags | one invocation | CLI |
| Eval | `CONTRACTOR_EVAL_*` env | eval suite | env var |

> ⚠️ **Not env-routed (hardcoded magic numbers):** the summarization trigger
> `max_tokens=80000` (`worker_factory.py:57`) and the planner subtask cap
> `max_steps=15` (repeated in `task_runner.py`, `models.py`,
> `planning_agent/agent.py`, `workflows/config.py`). To change globally you must
> edit code or override per-workflow YAML.

---

## 1. Global Settings (env / `cli/.env`)

All in `contractor/utils/settings.py` (`Settings`, pydantic-settings,
**case-insensitive** — env var = uppercased field name). Loaded once
(`@lru_cache get_settings()`); `.env` resolved relative to CWD (convention:
`cli/.env`). Tool defaults are global fallbacks — each has a per-call override.

### 1.1 LLM / model / proxy

| Env var | Default | Controls |
|---|---|---|
| `DEFAULT_MODEL_NAME` | `lm-studio-qwen3.6` | LiteLLM proxy alias when `--model` omitted |
| `DEFAULT_MODEL_TIMEOUT` | `300` s | Per-request LLM timeout |
| `MODEL_TEMPERATURE` | `None` (model default) | Sampling temperature (forwarded only when set) |
| `MODEL_TOP_P` | `None` | Nucleus sampling top_p (forwarded only when set) |
| `LITELLM_API_BASE` | `None` | LiteLLM proxy base URL |
| `LITELLM_API_KEY` | `None` | LiteLLM proxy API key |

> Sampling surface is intentionally minimal: only temperature + top_p. No
> `max_tokens` / `top_k` / penalties at the Settings layer.

### 1.2 Observability / external services

| Env var | Default | Controls |
|---|---|---|
| `USE_LANGFUSE` | `False` | Master switch; off ⇒ all observability calls are no-ops |
| `LANGFUSE_HOST` / `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | `None` | Langfuse connection |
| `CAIDO_URL` / `CAIDO_AUTH_TOKEN` | `None` | Caido proxy (exploitability workflow) |
| `CONTRACTOR_ARTIFACTS_DIR` | `None` (→ `./artifacts`) | Artifact store base dir (explicit env alias) |

### 1.3 GitLab FS auth

Two places: `Settings` (`GITLAB_PRIVATE_TOKEN`, `GITLAB_OAUTH_TOKEN`,
`CI_JOB_TOKEN`) **and** a separate `GitlabFileSystemSettings`
(`contractor/tools/fs/gitlabfs.py`, `env_prefix="GITLAB_FS_"`):

| Env var | Default | Controls |
|---|---|---|
| `GITLAB_FS_GITLAB_URL` | `https://gitlab.com` | Instance base URL |
| `GITLAB_FS_REF` | `master` | Branch/tag/commit |
| `GITLAB_FS_PER_PAGE` | `100` (1–100) | API pagination size |
| `GITLAB_FS_TIMEOUT` | `60.0` s | Total HTTP timeout |
| `GITLAB_FS_MAX_CONCURRENT` | `3` | Max parallel downloads |
| `GITLAB_FS_MAX_FILE_SIZE` | `52_428_800` (50 MiB) | Skip larger files |
| `GITLAB_FS_MAX_RETRIES` | `5` | HTTP retry attempts |
| `GITLAB_FS_RETRY_BACKOFF_FACTOR` | `5` | `sleep = factor · 2^attempt` |
| `GITLAB_FS_RETRY_STATUSES` | `{429,500,502,503,504}` | Retry-triggering statuses |

---

## 2. Tool parameters

Tool limits/defaults are Settings-backed (env-overridable) global fallbacks;
every one has a per-call override in the tool signature.

### 2.1 Filesystem tools (`contractor/tools/fs/`)

| Env var | Default | Controls |
|---|---|---|
| `FS_MAX_ITEMS` | `100` | Max directory entries returned by listings |
| `FS_MAX_OUTPUT` | `50_000` bytes | `read_file` (and write) output byte cap |
| `FS_MAX_READ_LINES` | `2000` (`None`=off) | Default per-read line cap when `limit` omitted |
| `FS_MAX_FILES_PER_WALK` | `100_000` | Max files scanned per fs glob/grep tree walk (walk stops early + truncation notice; mirrors `CODE_MAX_FILES_PER_WALK`) |
| `FS_HEAVY_KEEP_BUDGET_CHARS` | `0` (disabled) | Char budget for retained heavy-tool results (elision) |
| `FS_HEAVY_KEEP_LAST_N` | `0` (→ caller's, ~15) | Override count cap for retained heavy-tool results |

### 2.2 Code-walk & graph tools (`contractor/tools/code/`)

| Env var | Default | Controls |
|---|---|---|
| `CODE_MAX_WALK_DEPTH` | `50` | Max recursion depth for code/file walk |
| `CODE_MAX_FILES_PER_WALK` | `100_000` | Max files visited per walk |
| `GRAPH_MAX_RESULTS` | `200` | Max nodes/results per graph query |
| `GRAPH_MAX_PATHS` | `25` | Max paths returned by graph path search |
| `GRAPH_MAX_PATH_DEPTH` | `30` | Max depth for graph path traversal |

### 2.3 HTTP tool (`contractor/tools/http.py`)

| Env var | Default | Controls |
|---|---|---|
| `HTTP_TIMEOUT` | `30.0` s | Request timeout |
| `HTTP_BODY_PREVIEW_CHARS` | `2048` | Response body preview length |
| `HTTP_HISTORY_SIZE` | `20` | Requests retained in history ring buffer |
| `HTTP_RETRY_ATTEMPTS` | `3` | Retry count |
| `HTTP_RETRY_BASE_DELAY` | `0.5` s | Base backoff delay |
| `HTTP_RETRY_MAX_DELAY` | `8.0` s | Backoff cap |

### 2.4 LikeC4 tool

| Env var | Default | Controls |
|---|---|---|
| `LIKEC4_VALIDATE_TIMEOUT` | `120.0` s | LikeC4 validation subprocess timeout |

### 2.5 Code-exec sandbox (podman — `contractor/tools/podman.py`)

Not env-routed (module constants / factory args):

| Name | Default | Controls |
|---|---|---|
| `_CONTAINER_TTL` | `2h` | Container self-expiry backstop |
| `_DEFAULT_TIMEOUT_S` | `120` s | Default wall-clock for `run_python` / `run_bash` |
| exec grace | `timeout_s + 15` | Host-side kill margin over in-container timeout |

### 2.6 Subtask / planner tool factory (`contractor/tools/tasks/`)

| Name | Default | Controls |
|---|---|---|
| `max_records` | `20` | `get_records` returns last N records |
| `n_retries` | `3` | Worker re-run budget on empty/unparseable/mismatched output |
| `_MAX_LITERAL_EVAL_LEN` | `50_000` | Cap on literal-eval parsing of worker output |
| decompose children | `1–3` | Subtasks per `decompose_subtask` (min enforced) |
| `task_id` pattern | `^\d+(\.\d+)*$` | Dotted numeric IDs |
| factory toggles | `use_skip=T`, `use_type_hint=F`, `use_input_schema=T`, `use_output_schema=T`, `use_summarization=T`, `worker_instrumentation=T` | Tool-surface switches |

---

## 3. Per-workflow config (`contractor/workflows/<mode>/config.yaml`)

Schema in `contractor/workflows/config.py`. Four optional top-level blocks:

- **`budgets:`** — free-form `dict[str,int]` (`CFG.budgets.<name>`). Convention:
  `*_max_tokens` = summarization-trigger budget (context retained before
  compression), **not** a generation cap. Also `max_steps`, `max_concurrency`.
- **`tasks.<name>:`** — `TaskBudget`: `iterations` (def 1), `max_attempts`
  (def 1), `max_steps` (def 15). Splatted into `add_task()`.
- **`agents.<name>:`** — `AgentToolConfig`: `output_format` (def `json`;
  `json|xml|yaml|markdown`), `with_graph_tools` (def F), `with_code_exec` (def F).
- **`observations:`** — `ObservationConfig` (projects worker-usage facts into the
  planner). Default disabled. Fields: `enabled`, `track_tools`, `tracked_tools`,
  `include_tool_errors`, `track_skills`, `track_files`, `track_file_paths`,
  `track_coverage_gap`, `track_memories`, `malformed_only`, `in_record`,
  `in_result`. Env overlay: `CONTRACTOR_EVAL_OBSERVATIONS` (JSON).

### 3.1 Budgets per workflow (token / scalar)

| Workflow | Budgets |
|---|---|
| oas_building | swe 100k, builder 100k, validator 100k |
| oas_enrichment | builder 120k, validator 120k |
| likec4_building | swe 100k, builder 120k |
| router | max_tokens 120k, max_steps 20 |
| exploitability | max_tokens 80k |
| trace_annotation | max_tokens 80k |
| trace_annotation_direct | max_tokens 100k |
| trace_graph | max_tokens 100k |
| trace_graph_pathpar | max_tokens 100k, **max_concurrency 3** |
| trace_verify | max_tokens 80k |
| vuln_assess | swe 100k, builder 100k, validator 100k |
| vuln_scan | scan 80k |
| vuln_scan_fast | scan 80k, swe 100k |
| vuln_scan_trace | scan 80k, trace 80k |

### 3.2 Task budgets per workflow (`iterations` / `max_attempts` / `max_steps`)

| Workflow.task | iters | attempts | steps |
|---|---|---|---|
| oas_building.dependency_information | 1 | 2 | 20 |
| oas_building.project_information | 1 | 2 | 20 |
| oas_building.oas_update | 2 | 4 | 20 |
| oas_building.oas_validate | 1 | 1 | 20 |
| oas_enrichment.oas_enrich | **3** | **6** | **30** |
| oas_enrichment.oas_validate | 2 | 2 | 20 |
| likec4_building.dependency_information | 1 | 3 | 20 |
| likec4_building.project_information | 1 | 3 | 20 |
| likec4_building.likec4_build | **3** | **6** | 20 |
| likec4_building.likec4_validate | 1 | 2 | 20 |
| exploitability.assess | 1 | 2 | 25 |
| trace_annotation.annotate | 1 | 3 | 20 |
| trace_verify.verify | 1 | 2 | 20 |
| vuln_assess.* | (identical to oas_building) | | |
| vuln_scan.scan | 1 | 2 | **75** |
| vuln_scan_fast.dependency_information | 1 | 2 | 20 |
| vuln_scan_fast.project_information | 1 | 2 | 20 |
| vuln_scan_fast.scan | 1 | 2 | **50** |
| vuln_scan_trace.scan | 1 | 2 | **75** |
| vuln_scan_trace.trace | 1 | 1 | 30 |

> `trace_annotation_direct`, `trace_graph`, `trace_graph_pathpar` have no `tasks`
> block (they run agents directly, not via TaskRunner).

### 3.3 Agent tool options per workflow

- `with_graph_tools: true` — every declared `trace_agent` and `codereview_agent`
  (router, trace_annotation, trace_*, vuln_scan*, vuln_scan_trace).
- `with_code_exec: true` — only `exploitability_agent` (exploitability).
- `output_format: json` — everywhere; no workflow uses xml/yaml/markdown.

### 3.4 Observations per workflow

11 workflows enable the "lean + file-paths" config (`enabled:true`,
`include_tool_errors:false`, `track_file_paths:true`). Disabled (no block):
`trace_annotation_direct`, `trace_graph`, `trace_graph_pathpar`. No workflow
enables `track_coverage_gap`, `track_memories`, `malformed_only`, or
`include_tool_errors` (A/B showed error counts hurt).

---

## 4. Task templates (`contractor/tasks/*.yml`)

`TaskTemplate.load` parses 8 body fields (`contractor/runners/models.py`):
`name`, `objective`, `instructions`, `output_format`, `artifacts` (def `[]`),
`skills` (def `[]`), `iterations` (def `1`), `format` (def `json`). Keys
`context`/`constraints` appear in some bodies but are **ignored by the loader**
(prose only). Manifest = `active:` + `versions:` map of `vN → {file}`. Version
resolution: explicit arg > `CONTRACTOR_TASK_VERSION_<NAME>` env override (e.g.
`CONTRACTOR_TASK_VERSION_TRACE_ANNOTATION=v3`) > manifest `active:`.

Run-time budgets (set by assembler / `add_task`, not the YAML body):
`iterations` (def 1), `max_attempts` (def `max(1, iterations)`), `max_steps`
(def 15). Resolution: `eff_max_attempts < iterations` raises.

| Template | Active | iters | format | skills |
|---|---|---|---|---|
| dependency_information | v1 | 1 | json | — |
| project_information | v1 | 1 | json | — |
| oas_enrich | v2 | 1 | json | — |
| oas_update | v2 | 1 | json | — |
| oas_validate | v1 | 1 | json | — |
| likec4_build | v1 | 1 | json | likec4 |
| likec4_validate | v2 | 1 | json | likec4 |
| threat_analysis | v1 | 1 | json | stride |
| trace_annotation | **v3** | 1 | json | — |
| trace_verify | v1 | 1 | json | — |
| exploitability_assessment | v4 | 1 | json | — |
| vuln_scan | v3 | 1 | json | vuln_scan |
| vuln_scan_fast | v1 | 1 | json | vuln_scan |

> No template declares `iterations > 1` or `max_attempts` — those come from
> per-workflow `config.yaml` (§3.2). `format` is `json` everywhere.

Non-active versions exist for: `exploitability_assessment` (v1–v4),
`oas_enrich`/`oas_update` (v1–v2), `likec4_validate` (v1–v2), `vuln_scan` (v1–v3),
`trace_annotation` (v1, v2, `shannon`).

---

## 5. Planner & subtask state machine

`contractor/agents/planning_agent/agent.py` + `contractor/tools/tasks/`:

| Name | Default | Controls |
|---|---|---|
| `max_steps` | `15` | `task_tools(max_tasks=…)` + `<<MAX_SUBTASKS>>` prompt token |
| `_format` | `json` | Subtask/memory serialization |
| `use_output_schema` | `False` | Schema-constrained worker output |
| `worker_instrumentation` | `True` | Attach Subtask I/O schemas + trailer |
| `RepeatedToolCallCallback(threshold=2)` | `2` | Blocks identical repeated tool calls (planner) |
| decompose children | `1–3` | Children per decompose |
| `finish` guard | — | Requires ≥1 `done`, no `new` subtasks remaining |

Flow: `CFG.budgets.max_steps` → `TaskInvocation.max_steps` (15) → `TaskRunner`
→ `build_planning_agent(max_steps)` → `task_tools` + `<<MAX_SUBTASKS>>` prompt token.

Subtask states: `new, done, incomplete, malformed, skipped, decomposed`.
Transitions: `new`→[done,incomplete,malformed,skipped];
`malformed`/`incomplete`→[skipped,decomposed]; `done`/`decomposed`/`skipped`
terminal. Active planner prompt: **v5** (versions: pentestgpt, v5, v4, v3, v2, v1).

---

## 6. Callbacks (`contractor/callbacks/`)

The **live** worker callback stack: `TokenUsage → SummarizationLimit →
[FunctionResultsRemoval] → InvalidToolCallGuardrail → RepeatedToolCall`
(`worker_factory.py`). Most knobs are constructor args, **not** env-routed.

### 6.1 Active

| Name | Value | Controls |
|---|---|---|
| `SummarizationLimitCallback.max_tokens` | **80000** (worker_factory) | Token threshold → injects "summarize progress" message |
| `SummarizationLimitCallback.summarization_key` | `total` | Which counter the threshold measures |
| `RepeatedToolCallCallback.threshold` | worker **5**, planner **2** | Identical-call advisory trigger |
| `MandatoryToolCallback.max_nudges` | def 2, exploit **3** | Nudges to call required verdict tool |
| `FunctionResultsRemovalCallback.keep_last_n` | 0 (→ worker 15 via `elide_keep_last_n`) | Count axis of heavy-result retention |
| `FunctionResultsRemovalCallback.keep_budget_chars` | 0 (← `FS_HEAVY_KEEP_BUDGET_CHARS`) | Char-budget axis |
| `FunctionResultsRemovalCallback.deduplicate` | True | Elide stale duplicate tool results |
| `build_worker(with_elide=…)` | True | Register the elision callback at all |

### 6.2 Defined but **NOT wired** (dormant knobs)

These have full implementations but no callsite in any agent factory — token /
rate hard-caps are effectively inactive; only the soft 80k summarization nudge
is live.

| Name | Knobs | Status |
|---|---|---|
| `ThinkingBudgetGuardrailCallback` | `token_budget`, `token_budget_key=total` | Not instantiated |
| `ToolMaxCallsGuardrailCallback` | `max_calls`, `rvalue` | Not instantiated |
| `TpmRatelimitCallback` | `tpm_limit`, `tpm_limit_key=input`, window 60s (hardcoded) | Not instantiated |
| `RpmRatelimitCallback` | `rpm_limit`, window 60s (hardcoded) | Not instantiated |

### 6.3 Plugins (`contractor/runners/plugins/`)

No numeric tunables. `AdkMetricsPlugin`: `result_error_detector` (heuristic),
`_args_hash` digest length 16 (must match `analyze_metrics.py`), prefix
`metrics`. `AdkTracePlugin` prefix `trace`. `SandboxCleanupPlugin` name
`sandbox_cleanup`.

---

## 7. CLI flags (`cli/main.py`)

| Flag | Default | Controls |
|---|---|---|
| `--workflow` | `oas_build` | Workflow to run (Choice from registry) |
| `--project-path` | **required** | Target root = FS sandbox root + default output dir |
| `--folder-name` | `/` | Project-relative folder injected into templates |
| `--artifact` | `None` | Existing OpenAPI seed (UTF-8 validated) |
| `--user-id` | `cli-user` | ADK session / artifact store key |
| `--model` | `DEFAULT_MODEL_NAME` | LiteLLM proxy alias |
| `--timeout` | `DEFAULT_MODEL_TIMEOUT` (300) | Per-request model timeout (s) |
| `--prompt` | `None` | Prompt for router; required with `--no-ui` |
| `--rm` | `False` | Remove prior artifacts (excl. with `--resume`) |
| `--resume` | `False` | Resume from `<output>/checkpoint.json` |
| `-o/--output` | `None` (→ `<project>/.contractor`) | Artifacts + `metrics.jsonl` dir |
| `--no-ui` | `False` | Plain stdout instead of live UI |

> No per-budget/iteration CLI flags — those are per-workflow `config.yaml` only.
> FS sandbox (`cli/fs.py`) takes only `root_path`; symlink-follow hard-disabled,
> `..` rejected. `MetricsSink` has no tunables (fixed `metrics.jsonl`).

---

## 8. Eval overrides (env, `tests/eval/`)

### 8.1 Gating / selection

| Env var | Default | Controls |
|---|---|---|
| `CONTRACTOR_RUN_EVAL` | unset (skip) | Enable eval suite (`-m eval` also bypasses) |
| `CONTRACTOR_EVAL_MODEL` | project default | Override eval model alias (timeout 600) |
| `CONTRACTOR_EVAL_RUN_STAMP` | `mmdd-HHMMSS` UTC | Per-run archive namespace |
| `CONTRACTOR_EVAL_RESULTS_DIR` | `eval_runs/` | Results output dir |
| `CONTRACTOR_EVAL_CASE_IDS` | all | Comma-separated case-id subset |
| `CONTRACTOR_EVAL_OBSERVATIONS` | unset | JSON overlay of `observations:` block (A/B) |

### 8.2 Trace agent eval

| Env var | Default | Controls |
|---|---|---|
| `CONTRACTOR_EVAL_TRACE_PROMPT_VERSION` | per-case | Pin trace_agent prompt version |
| `CONTRACTOR_EVAL_TRACE_PASS_AT` | `1` | pass@N loop count |
| `CONTRACTOR_EVAL_WITH_OAS` | off | Feed the OpenAPI spec into the prompt as an attack-surface map (X1 A/B) |
| `CONTRACTOR_TASK_VERSION_TRACE_ANNOTATION` | unset (→ `active: v3`) | Pin trace task-template version (generic mechanism, §4) |

### 8.3 Vuln detection eval

| Env var | Default | Controls |
|---|---|---|
| `CONTRACTOR_EVAL_VULN_AGENT` | `vuln_scan` | Which vuln agent to eval |
| `CONTRACTOR_EVAL_VULN_PASS_AT` | `3` | pass@N |
| `CONTRACTOR_EVAL_VULN_PROMPT_VERSION` | unset | Pin prompt variant |
| `CONTRACTOR_EVAL_VULN_MIN_RECALL` | `0.15` | Min recall to pass |
| `CONTRACTOR_EVAL_VULN_MIN_PRECISION` | `0.10` | Min precision to pass |
| `CONTRACTOR_EMITTED_VS_READ` | off | Emitted-vs-read scoring mode |
| `CONTRACTOR_VULN_DEDUP` | off | Finding dedup |

### 8.4 Exploitability & XBOW

| Env var | Default | Controls |
|---|---|---|
| `CONTRACTOR_EVAL_EXPLOIT_PROMPT_VERSION` | per-case | Pin exploit prompt |
| `CONTRACTOR_EVAL_PROXY` | unset | HTTP proxy for exploit attempts |
| `CONTRACTOR_XBOW_BENCHMARKS` | `DEFAULT_XBOW_IDS` | Benchmark id subset |
| `CONTRACTOR_XBOW_AGENT` | all | Restrict to one agent |
| `XBOW_MAX_TOKENS` | `80_000` (hardcoded) | XBOW token cap |

### 8.5 Runtime targets (used by exploit/vuln workflows)

| Env var | Default | Controls |
|---|---|---|
| `CONTRACTOR_TARGET_URL` | unset (stage skipped) | Target base URL for exploit stage |
| `CONTRACTOR_PROXY` | unset | HTTP proxy (exploitability workflow) |

### 8.6 Harness arg defaults (not env, but tunable)

harness `run_agent` default `timeout_s=600.0`; per-eval overrides: trace 900,
planner / project-info / vuln-detection 1800, threat / oas_enrich / oas_build
2400, oas_analyzer / xbow 1500, exploitability 300–900 (per-case `timeout_s`
in meta.yaml wins where supported). A/B drivers (`scripts/ab_*.py`):
`AB_FIXTURE(S)`, `AB_ARMS`, `AB_TIMEOUT` (21600/3600/3000), `AB_PER_PATH_TIMEOUT`
(900/420), `AB_MAX_ATTEMPTS` (2).

---

## 9. LiteLLM deploy (`deploy/litellm/`)

### `litellm_config.yaml`

- **model_list** aliases: `lm-studio-nemotron`, `lm-studio-openai`,
  `lm-studio-qwen3.5`, `lm-studio-glm`, `lm-studio-qwen3.5-opus`,
  `lm-studio-qwen3.5-hauhau`, `lm-studio-qwen3.6` (**project default**),
  `lm-studio-qwen3.6-mtp`, `lm-studio-qwen3.6-27b-mtp`.
- Per-model `litellm_params`: `model`, `api_key` (`lm-studio`), `api_base`
  (`http://localhost:1234/v1`), `tpm` (`1000000`), `rpm` (`20`).
- `litellm_settings`: `num_retries` `3`, `request_timeout` `300`.

### `run.sh`

`LITELLM_MASTER_KEY` (`sk-litellm-changeme`), `LITELLM_SALT_KEY`
(`sk-random-hash-changeme`), `--network=host`, config bind-mount. Image
`ghcr.io/berriai/litellm:main-stable`.
