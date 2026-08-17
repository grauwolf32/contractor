# Tuning & Performance Knobs

A consolidated inventory of every tunable parameter in Contractor — CLI flags,
global settings, LiteLLM proxy config, per-workflow agent budgets, task/planner
limits, callback thresholds, and tool caps — plus a tuning playbook at the end.

All file:line references are against the tree as analyzed; treat them as the
"where to change it" pointer.

> **TL;DR levers, highest impact first**
> 1. `--model` / `default_model_name` — the single biggest quality+cost+latency lever.
> 2. Per-workflow `*_max_tokens` (summarization trigger) — context retained before compression.
> 3. Per-task `iterations` / `max_attempts` / `max_steps` — convergence vs. cost.
> 4. Planner `max_steps` (subtask budget) — decomposition granularity.
> 5. **Sampling params (`MODEL_TEMPERATURE` / `MODEL_TOP_P`) — wired via `Settings`, default unset; see [§5](#5-sampling-params-settings-routed-default-unset).**
> 6. Context-elision (`elide_keep_last_n`) and rate limits (`tpm`/`rpm`).

---

## 1. How parameters flow

```
CLI flag (cli/main.py)
   └─► WorkflowContext (contractor/workflows/__init__.py)            # model, timeout, paths, ui
         └─► Workflow assembler (contractor/workflows/<mode>/workflow.py)
               │     reads CFG = WorkflowConfig.load(__file__)       # ← contractor/workflows/<mode>/config.yaml
               ├─► build_worker(...) (agents/worker_factory.py)      # token budget (CFG.budgets.*), elision, guardrails
               ├─► build_<agent>(...) (agents/<agent>/agent.py)      # tool opts (CFG.agent(...).output_format / with_graph_tools)
               └─► TaskRunner.add_task(..., **CFG.tasks.<n>.as_kwargs())  # retry/iteration semantics
Settings (.env via contractor/utils/settings.py)          # model alias default, proxy, langfuse, caido, sampling, tool caps
LiteLLM proxy (deploy/litellm/litellm_config.yaml)        # model aliases, rpm/tpm, retries, timeout
```

Three editing surfaces:
- **Per-run**: CLI flags.
- **Per-environment**: `cli/.env` (`Settings`) and `litellm_config.yaml`.
- **Per-workflow / per-task / per-agent (data)**: the `budgets:` (`*_max_tokens`,
  `max_steps`, …), `tasks:` (`iterations` / `max_attempts` / `max_steps`), and
  `agents:` (`output_format` / `with_graph_tools`) blocks in each
  `contractor/workflows/<mode>/config.yaml`. The workflow reads them via
  `WorkflowConfig.load(__file__)` — edit the YAML, not the assembler. (The
  `build_worker(...)` defaults remain the last-resort fallback.)

---

## 2. CLI flags (`cli/main.py`)

| Flag | Type | Default | Effect on perf/quality/cost |
|------|------|---------|------------------------------|
| `--workflow` | choice | `oas_build` | Selects agent chain + task templates. |
| `--project-path` | path | **required** | Sandbox root; scope of analysis = cost driver. |
| `--folder-name` | str | `/` | Narrows template focus; does not change file-scan scope. |
| `--artifact` | path | `None` | Seeds from a prior artifact; optional for enrich/trace/exploit/vuln (reuses store seed). |
| `--user-id` | str | `cli-user` | Session/Langfuse/artifact namespace. |
| `--model` | str | `Settings.default_model_name` | **Top lever.** LiteLLM alias, not a raw model name. |
| `--timeout` | int (s) | `Settings.default_model_timeout` (300) | Per-request model timeout → completion-vs-timeout tradeoff. |
| `--prompt` | str | `None` | Prompt-driven workflows (router); required with `--no-ui`. |
| `--rm` | flag | `False` | Clears prior artifacts (mutually exclusive with `--resume`). |
| `--resume` | flag | `False` | Skips completed tasks via `checkpoint.json` — biggest re-run cost saver. |
| `-o/--output` | path | `<project>/.contractor` | Artifact/checkpoint/metrics dir. |
| `--no-ui` | flag | `False` | Disables live UI (CI mode). |

---

## 3. Global settings (`contractor/utils/settings.py`, read from `cli/.env`)

| Field | Env var | Default | Notes |
|-------|---------|---------|-------|
| `default_model_name` | `DEFAULT_MODEL_NAME` | `lm-studio-qwen3.6` | Default for `--model`. |
| `default_model_timeout` | `DEFAULT_MODEL_TIMEOUT` | `300` | Default for `--timeout` (seconds). |
| `litellm_api_base` | `LITELLM_API_BASE` | `None` | Proxy URL (e.g. `http://localhost:4000`). |
| `litellm_api_key` | `LITELLM_API_KEY` | `None` | Proxy auth. |
| `use_langfuse` | `USE_LANGFUSE` | `False` | Observability; adds per-call overhead when on (no-op when off). |
| `langfuse_host` / `langfuse_public_key` / `langfuse_secret_key` | resp. | `None` | Required iff `use_langfuse`. |
| `caido_url` / `caido_auth_token` | resp. | `None` | Routes HTTP-tool traffic through Caido proxy (latency cost; needed for exploit proof chains). |
| `gitlab_private_token` / `gitlab_oauth_token` / `ci_job_token` | resp. | `None` | Repo access for remote/CI projects. |
| `artifacts_dir` | `CONTRACTOR_ARTIFACTS_DIR` | `None` → `<repo>/artifacts` | Shared cross-project artifact store. |
| `target_url` / `proxy` | `CONTRACTOR_TARGET_URL` / `CONTRACTOR_PROXY` | `None` | Live target base URL + outbound HTTP proxy for exploit/vuln workflows. |
| `model_temperature` / `model_top_p` | `MODEL_TEMPERATURE` / `MODEL_TOP_P` | `None` | Sampling defaults applied by `build_model()` to every `LiteLlm`; `None` keeps backend defaults (see §5). |

**Tool-default fields** (same file — global baselines, overridable per call-site;
defaults equal the historical hardcoded constants unless noted):

| Field | Env var | Default |
|-------|---------|---------|
| `http_timeout` / `http_body_preview_chars` / `http_history_size` | `HTTP_*` | `30.0` / `2048` / `20` |
| `http_retry_attempts` / `http_retry_base_delay` / `http_retry_max_delay` | `HTTP_RETRY_*` | `3` / `0.5` / `8.0` |
| `fs_max_output` / `fs_max_read_lines` / `fs_max_items` | `FS_MAX_OUTPUT` / `FS_MAX_READ_LINES` / `FS_MAX_ITEMS` | `50_000` chars / `2000` lines / `100` |
| `fs_max_files_per_walk` | `FS_MAX_FILES_PER_WALK` | `100_000` |
| `fs_heavy_keep_last_n` / `fs_heavy_keep_budget_chars` | `FS_HEAVY_KEEP_LAST_N` / `FS_HEAVY_KEEP_BUDGET_CHARS` | `0` (= use caller's `elide_keep_last_n`) / `0` (budget axis off) |
| `code_max_walk_depth` / `code_max_files_per_walk` | `CODE_*` | `50` / `100_000` |
| `graph_max_results` / `graph_max_paths` / `graph_max_path_depth` | `GRAPH_*` | `200` / `25` / `30` |
| `likec4_validate_timeout` | `LIKEC4_VALIDATE_TIMEOUT` | `120.0` s |

---

## 4. LiteLLM proxy (`deploy/litellm/litellm_config.yaml`)

**Global** (`litellm_settings`):

| Param | Default | Effect |
|-------|---------|--------|
| `num_retries` | `3` | Auto-retry on failed calls → reliability vs. latency on persistent failures. |
| `request_timeout` | `300` | Global request ceiling (mirror of `--timeout`). |

**Per-model alias** — each defines `tpm: 1000000`, `rpm: 20`. Available aliases:
`lm-studio-nemotron`, `lm-studio-openai`, `lm-studio-qwen3.5`, `lm-studio-glm`,
`lm-studio-qwen3.5-opus`, `lm-studio-qwen3.5-hauhau`, `lm-studio-qwen3.6` (default),
`lm-studio-qwen3.6-mtp`, `lm-studio-qwen3.6-27b-mtp`.

> `rpm: 20` is the binding throughput limit for parallel/multi-task workflows; raise it
> if the backend can sustain more concurrency, otherwise tasks serialize behind it.

---

## 5. Sampling params (Settings-routed, default unset)

`temperature` / `top_p` are now wired through `Settings` (`model_temperature` /
`model_top_p`, env `MODEL_TEMPERATURE` / `MODEL_TOP_P`). `build_model()` in
`contractor/utils/settings.py` forwards them to every `LiteLlm` it constructs —
including the shared `DEFAULT_MODEL` — but only when set; the default `None`
keeps LiteLLM/backend defaults, so behaviour is unchanged until you opt in.

Tuning notes:
- Lower `temperature` (e.g. 0–0.3) for the deterministic structured-output tasks
  (OAS build/enrich, trace annotation, vuln verdicts) likely improves schema
  adherence and reduces retry/`malformed` churn — set it in `cli/.env`.
- `reasoning_effort` / thinking budgets remain **unwired** — adding one (where the
  backend supports it) is the natural place to trade latency for depth on
  `vuln_scan` / `exploitability`. Per-alias defaults in `litellm_config.yaml` are
  the alternative injection point.

---

## 6. Per-workflow agent budgets & task limits

The `build_worker` factory defaults (`contractor/agents/worker_factory.py`):

| Param | Default | Effect |
|-------|---------|--------|
| `max_tokens` | `80000` | Token budget before the **summarization** message is injected (context compression trigger). |
| `with_elide` | `True` | Register tool-result elision callback. |
| `elide_keep_last_n` | `15` | Recent heavy-tool results kept un-elided; lower = cheaper, less recall. (`FS_HEAVY_KEEP_LAST_N` > 0 overrides it globally.) |
| `elide_keep_budget_chars` | `None` → `Settings.fs_heavy_keep_budget_chars` (0 = off) | Optional cumulative char budget for retained heavy-tool results (evicts oldest-first). |
| `repeated_call_threshold` | `5` | Identical-consecutive-call count before loop advisory. |
| `model` | `DEFAULT_MODEL` | Per-agent model override. |

Per-workflow overrides live in `contractor/workflows/<mode>/config.yaml`
(`budgets:` for the `*_max_tokens` token budgets, `tasks:` for the retry/iter/step
budgets). The workflow reads them via `WorkflowConfig.load(__file__)`; edit the YAML
to tune. `iter` = `iterations`, `att` = `max_attempts`, `steps` = planner subtask
budget (`max_steps`):

| Workflow (`<mode>/config.yaml`) | `budgets.*_max_tokens` | Task budgets (`iter`/`att`/`steps`) |
|-----------------|----------------|--------------------------------------|
| `oas_enrichment` | 120k (builder/validator) | enrich `3/6/30`; validate `2/2/20` |
| `oas_building` | 100k (swe/builder/validator) | update `2/4/20`; dep/proj info `1/2/20`; validate `1/1/20` |
| `likec4_building` | 100k swe / 120k builder | build `3/6/20`; dep/proj info `1/3/20`; validate `1/2/20` |
| `trace_annotation` | 80k | annotate `1/3/20` |
| `trace_annotation_direct` | 100k | — |
| `trace_graph` / `trace_graph_pathpar` | 100k (pathpar adds `budgets.max_concurrency: 3`) | — |
| `trace_verify` | 80k | `1/2/20` |
| `vuln_scan` | 80k | `1/2/75` |
| `vuln_scan_fast` | 80k scan / 100k swe | scan `1/2/50`; dep/proj info `1/2/20` |
| `vuln_scan_trace` | 80k scan / 80k trace | scan `1/2/75`; trace `1/1/30` |
| `vuln_assess` | 100k (swe/builder/validator) | update `2/4/20`; dep/proj info `1/2/20`; validate `1/1/20` |
| `exploitability` | 80k | assess `1/2/25` |
| `router` | 120k | `budgets.max_steps` (20) |

Notes:
- **`*_max_tokens` here is a *summarization trigger*, not a generation cap** — raising it
  retains more context (better cross-file reasoning) at higher per-call token cost and
  risk of hitting the model's true context window.
- `vuln_scan` uses a large `steps=75` budget (deep, single-pass exploration); the
  `_fast` variant cuts it to 50.

### Per-agent tool options (`config.yaml` `agents:` block)

Each `config.yaml` may carry an `agents:` map keyed by agent name; the workflow reads
it via `CFG.agent("<name>")` and threads it into the `build_<agent>` factory. Agents
not listed fall back to defaults (`output_format: json`, `with_graph_tools: false`),
so behaviour is unchanged unless tuned.

| Key | Default | Effect |
|-----|---------|--------|
| `output_format` | `json` | The shared `_format` knob for fs/memory/openapi/report tool output (`json` / `xml` / `yaml` / `markdown`; unsupported renderers fall back to json). |
| `with_graph_tools` | `false` | Attach the trailmark call-graph tools (callers/callees/paths/attack-surface). Enabled for the `codereview_agent` / `trace_agent` in the scan + trace workflows. |
| `with_code_exec` | `false` | Attach the podman-backed `run_python` / `execute_bash` sandbox tools (exploit agents only; enabled in `exploitability`). |

### Worker observations (`config.yaml` `observations:` block)

Each `config.yaml` may also carry a workflow-global `observations:` block
(`CFG.observations`, an `ObservationConfig` from `contractor/tools/observations.py`):
deterministic worker-usage facts (tool/file/skill counts, optional unread-file
coverage gap) injected back into the planner's task records/results. All-default is
disabled; most workflows now enable the "lean + file-paths" arm
(`enabled: true`, `include_tool_errors: false`, `track_file_paths: true`) — A/Bs
showed a consistent vuln-detection F1 lift at roughly neutral cost, while tool
*error* counts hurt. The `CONTRACTOR_EVAL_OBSERVATIONS` env var (JSON object)
overlays the block field-by-field for A/B runs without editing YAML.

---

## 7. Task runner & retry semantics (`runners/models.py`, `runners/task_runner.py`)

| Param | Default | Semantics |
|-------|---------|-----------|
| `iterations` (`models.py:100`) | `1` | **Successful** runs required before a task is "done". |
| `max_attempts` (`models.py:101`) | `1` (resolved to `max(1, iterations)`) | Upper bound on tries; exhausting it without enough successes → `TaskNotCompletedError`. |
| `max_steps` (`models.py:102`) | `15` | Per-attempt planner subtask budget (overridden per task above). |
| `default_iterations` / `format` (template) | `1` / `json` | From task YAML; `task_runner.py` `_resolve_retry_params` enforces `max_attempts ≥ iterations ≥ 1`. |

> **Resilience gap:** default `max_attempts == iterations`, so a single transient failure
> kills a task unless the workflow explicitly sets a buffer (as enrich/build do). Raising
> `max_attempts` above `iterations` is the cheap reliability knob.

Task templates are now **versioned** like agent prompts: `contractor/tasks/<name>.yml`
is a manifest (`active:` + `versions:`) selecting a body from `contractor/tasks/<name>/v*.yml`
(e.g. `trace_annotation` active is `v3`). `CONTRACTOR_TASK_VERSION_<NAME>` (e.g.
`CONTRACTOR_TASK_VERSION_TRACE_ANNOTATION=v3`) overrides the active version per task —
the A/B lever for task-prompt variants. All template bodies currently use
`iterations: 1` and `format: json`; `skills:` is set on `likec4_*`, `vuln_scan*`,
and `threat_analysis`.

---

## 8. Planner / subtask budget (`agents/planning_agent/`, `tools/tasks/`)

| Param | Default | Effect |
|-------|---------|--------|
| `max_steps` (planner, `agent.py:34`) | `15` | Total subtask budget (`add_subtask` + `decompose_subtask` share it). Substituted into `<<MAX_SUBTASKS>>`. |
| Bootstrap ratio (prompt `v5.md:73`) | `0.7` | ≤70% of budget for initial subtasks; reserves 30% for mid-run decomposition. |
| Decomposition cardinality | `1–3` children | Branching width when refining an `incomplete`/`malformed` subtask. |
| `max_records` (`tools/tasks/tools.py:254`) | `20` | Subtask history records returned to planner — context vs. recall. |
| `n_retries` (worker parse, `tools.py:255`) | `3` | Parse-retry budget for malformed worker output before decompose/skip. |
| `_MAX_LITERAL_EVAL_LEN` (`tools/tasks/models.py:100`) | `50000` | Char cap for literal-eval JSON recovery of large outputs. |

Subtask state machine (`tools/tasks/models.py`) is strict: `incomplete`/`malformed`
can only be decomposed or skipped, never re-executed; `finish` requires no `new` subtasks.

---

## 9. Callbacks (`contractor/callbacks/`)

**Context management** (`context.py`):

| Callback / param | Default | Effect |
|------------------|---------|--------|
| `SummarizationLimitCallback.max_tokens` | per-agent (see §6) | Compression trigger. |
| `summarization_key` | `total` | Which counter (`input`/`output`/`total`) drives it. |
| `FunctionResultsRemovalCallback.keep_last_n` | `0` (disabled) → `15` via factory | Count of tool results kept. |
| `keep_budget_chars` | `0` (disabled) | Alternative char-budget cap. |
| `deduplicate` | `True` | Elide stale duplicate tool results. |

**Guardrails** (`guardrails.py`):

| Callback / param | Default | Effect |
|------------------|---------|--------|
| `ThinkingBudgetGuardrailCallback.token_budget` | required | Hard stop (forces MAX_TOKENS finish). |
| `ToolMaxCallsGuardrailCallback.max_calls` | per-tool | Caps invocations of one tool. |
| `MandatoryToolCallback.max_nudges` | `2` (3 in exploit agents) | Nudges toward a required tool (e.g. verdict) before giving up. |
| `RepeatedToolCallCallback.threshold` | `5` (3 for router) | Identical-call loop guard. |

**Rate limits** (`ratelimits.py`) — opt-in per agent, no defaults:

| Callback / param | Effect |
|------------------|--------|
| `TpmRatelimitCallback.tpm_limit` (+ `tpm_limit_key`) | Tokens/min cap; sleeps `60-elapsed+1`s when breached. |
| `RpmRatelimitCallback.rpm_limit` | Requests/min cap; same sleep. |

> Note: the TPM/RPM callbacks throttle with a *blocking* `time.sleep`, which stalls the
> whole asyncio event loop — wire them up only for single-agent runs (see the class
> docstring in `ratelimits.py`).

---

## 10. Tool caps (`contractor/tools/`)

Most caps are now **Settings-routed** (env-tunable via `cli/.env`, see §3); the
constructors fall back to `get_settings()` when no explicit value is passed, so
production defaults are unchanged unless tuned.

| Tool / knob | Default (Settings field / env) | Effect |
|--------------|---------|--------|
| `fs` read/output byte cap | `50_000` chars (`fs_max_output` / `FS_MAX_OUTPUT`) | `read_file` / listing output budget; binds together with the line cap, whichever first. |
| `fs` read line cap | `2000` lines (`fs_max_read_lines` / `FS_MAX_READ_LINES`; `None` disables) | Default per-read `limit`. |
| `fs` `max_items` | `100` (`fs_max_items`) | Listing pagination. |
| `fs` walk ceiling | `100000` (`fs_max_files_per_walk`) | Hard cap on files scanned per glob/grep tree walk (truncation notice on hit). |
| code walk depth / files | `50` / `100000` (`code_max_walk_depth` / `code_max_files_per_walk`) | Dir nesting cap (symlink-loop guard) / runaway-scan guard. |
| graph max results | `200` (`graph_max_results`) | Symbol search results. |
| graph max paths | `25` (`graph_max_paths`) | Call-path enumeration cap. |
| graph path depth | `30` (`graph_max_path_depth`) | Path depth cap (exponential-blowup guard). |
| `list_symbols` page size | `300` (hardcoded) | Symbol-listing pagination. |
| HTTP `timeout` | `30.0`s (`http_timeout`) | Per-request. |
| HTTP `body_preview_chars` | `2048` (`http_body_preview_chars`; 512 in exploit agents) | Inline body preview; rest via `http_read_body`. |
| HTTP `history_size` | `20` (`http_history_size`) | Session request history. |
| HTTP `verify_ssl` | `True` (False behind Caido) | TLS verification. |
| HTTP `RetryConfig` | `attempts=3`, `base_delay=0.5`, `max_delay=8.0` (`http_retry_*`), statuses `(408,425,429,500,502,503,504)` | Transient-failure backoff. |
| likec4 `validate` timeout | `120.0`s (`likec4_validate_timeout`) | Linter subprocess ceiling. |

---

## 11. Prompt versions (quality lever, zero token cost)

Each agent dir has `prompt.yml` with an `active:` version selecting `prompts/v*.md`
(`load_prompt(name)` / `load_prompt_with_version(name, version)`). Switching the active
version is a pure quality lever. The trace eval already A/Bs versions via
`CONTRACTOR_EVAL_TRACE_PROMPT_VERSION`; the same pattern works for any agent.
Versioned agents and their current actives: `planning_agent` (`v5`), `trace_agent`
(`converge`), `codereview_agent` (`v3`), `exploitability_agent` (`shannon`),
`web_exploitability_agent` (`v4`), `swe_edit_agent` (`v2`), `http_agent` (`v1`),
`threat_model_agent` (`v1`), `oas_builder_agent` (`v4`), `oas_linter_agent` (`v1`).

Task templates carry the same mechanism (§7): manifest `active:` per
`contractor/tasks/<name>.yml`, overridable via `CONTRACTOR_TASK_VERSION_<NAME>`.

---

## 12. Tuning playbook

**Speed / cost down (sacrifice depth):**
- Use a smaller/faster `--model` alias; keep `default_model_timeout` modest.
- `--resume` aggressively on re-runs.
- Lower per-workflow `*_max_tokens` (earlier summarization) and `elide_keep_last_n` (e.g. 8–10).
- Trim `max_steps` on exploratory workflows (`vuln_scan` 75→50, à la `_fast`).
- Keep `use_langfuse=False`.

**Quality / completeness up (accept cost):**
- Set `MODEL_TEMPERATURE≈0` for structured tasks (§5) — now a one-line `.env` change;
  it directly cuts `malformed`/retry churn.
- Raise `iterations` (2–3) on the tasks whose output you most need to converge
  (enrich already uses 3; trace/vuln verdicts are candidates).
- Raise `max_attempts` above `iterations` everywhere to survive transient failures cheaply.
- Raise planner `max_steps` for large codebases (more decomposition headroom).
- Raise `*_max_tokens` toward the model's real context window for cross-file reasoning.

**Throughput / parallelism:**
- Raise per-alias `rpm` (default `20`) in `litellm_config.yaml` if the backend sustains it —
  it's the binding limit for multi-task workflows.
- Add `RpmRatelimitCallback`/`TpmRatelimitCallback` only where you need to *protect* a
  shared backend; they introduce blocking sleeps.

**Reliability:**
- `num_retries` (LiteLLM) and HTTP `RetryConfig.attempts` for transient infra errors;
  `max_attempts > iterations` for agent-level non-determinism.

---

*Generated from a code sweep of `cli/main.py`, `contractor/utils/settings.py`,
`deploy/litellm/litellm_config.yaml`, `contractor/workflows/*/workflow.py`, `contractor/agents/worker_factory.py`,
`contractor/runners/*`, `contractor/agents/planning_agent/*`, `contractor/callbacks/*`,
and `contractor/tools/*`. Verify line numbers before editing — they drift.*
