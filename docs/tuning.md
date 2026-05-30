# Tuning & Performance Knobs

A consolidated inventory of every tunable parameter in Contractor — CLI flags,
global settings, LiteLLM proxy config, per-workflow agent budgets, task/planner
limits, callback thresholds, and tool caps — plus a tuning playbook at the end.

All file:line references are against the tree as analyzed; treat them as the
"where to change it" pointer.

> **TL;DR levers, highest impact first**
> 1. `--model` / `default_model_name` — the single biggest quality+cost+latency lever.
> 2. Per-workflow `*_MAX_TOKENS` (summarization trigger) — context retained before compression.
> 3. Per-task `iterations` / `max_attempts` / `max_steps` — convergence vs. cost.
> 4. Planner `max_steps` (subtask budget) — decomposition granularity.
> 5. **Sampling params (temperature, top_p, reasoning_effort) — currently unset; see [§5](#5-sampling-params--currently-unset-lever).**
> 6. Context-elision (`elide_keep_last_n`) and rate limits (`tpm`/`rpm`).

---

## 1. How parameters flow

```
CLI flag (cli/main.py)
   └─► WorkflowContext (contractor/workflows/__init__/workflow.py)        # model, timeout, paths, ui
         └─► Workflow assembler (contractor/workflows/<mode>/workflow.py) # *_MAX_TOKENS, max_steps, iterations
               ├─► build_worker(...) (agents/worker_factory.py)   # token budget, elision, guardrails
               └─► TaskRunner.add_task(...) (runners/task_runner.py) # retry/iteration semantics
Settings (.env via contractor/utils/settings.py)          # model alias default, proxy, langfuse, caido
LiteLLM proxy (deploy/litellm/litellm_config.yaml)        # model aliases, rpm/tpm, retries, timeout
```

Three editing surfaces:
- **Per-run**: CLI flags.
- **Per-environment**: `cli/.env` (`Settings`) and `litellm_config.yaml`.
- **Per-workflow / per-task (code)**: the `*_MAX_TOKENS` / `max_steps` / `iterations`
  constants inside each `contractor/workflows/<mode>/workflow.py`, and the `build_worker(...)` defaults.

---

## 2. CLI flags (`cli/main.py`)

| Flag | Type | Default | Effect on perf/quality/cost |
|------|------|---------|------------------------------|
| `--workflow` | choice | `build` | Selects agent chain + task templates. |
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

---

## 4. LiteLLM proxy (`deploy/litellm/litellm_config.yaml`)

**Global** (`litellm_settings`):

| Param | Default | Effect |
|-------|---------|--------|
| `num_retries` | `3` | Auto-retry on failed calls → reliability vs. latency on persistent failures. |
| `request_timeout` | `300` | Global request ceiling (mirror of `--timeout`). |

**Per-model alias** — each defines `tpm: 1000000`, `rpm: 20`. Available aliases:
`lm-studio-nemotron`, `lm-studio-openai`, `lm-studio-qwen3.5`, `lm-studio-glm`,
`lm-studio-qwen3.5-opus`, `lm-studio-qwen3.5-hauhau`, `lm-studio-qwen3.6` (default).

> `rpm: 20` is the binding throughput limit for parallel/multi-task workflows; raise it
> if the backend can sustain more concurrency, otherwise tasks serialize behind it.

---

## 5. Sampling params — currently UNSET lever

Grep across `contractor/`, `cli/`, and `deploy/` finds **no** `temperature`, `top_p`,
`top_k`, `reasoning_effort`, `thinking`, or `GenerateContentConfig` configuration.
Every `LlmAgent` is built with `LiteLlm(model=...)` only (`worker_factory.py:127-134`),
so sampling falls entirely to LiteLLM/backend defaults.

This is an **unexploited tuning surface**:
- Lower `temperature` (e.g. 0–0.3) for the deterministic structured-output tasks
  (OAS build/enrich, trace annotation, vuln verdicts) would likely improve schema
  adherence and reduce retry/`malformed` churn.
- A reasoning/thinking budget (where the backend supports it) is the natural place to
  trade latency for depth on `vuln_scan` / `exploitability`.

To wire it in: pass a `generate_content_config` (ADK) or LiteLLM `extra`/sampling
kwargs through `build_worker(...)`, or set defaults per-alias in `litellm_config.yaml`.

---

## 6. Per-workflow agent budgets & task limits

The `build_worker` factory defaults (`contractor/agents/worker_factory.py`):

| Param | Default (line) | Effect |
|-------|----------------|--------|
| `max_tokens` | `80000` (52) | Token budget before the **summarization** message is injected (context compression trigger). |
| `with_elide` | `True` (54) | Register tool-result elision callback. |
| `elide_keep_last_n` | `15` (56) | Recent heavy-tool results kept un-elided; lower = cheaper, less recall. |
| `repeated_call_threshold` | `5` (57) | Identical-consecutive-call count before loop advisory. |
| `model` | `DEFAULT_MODEL` (53) | Per-agent model override. |

Per-workflow overrides (verified). `iter` = `iterations`, `att` = `max_attempts`,
`steps` = planner subtask budget (`max_steps`):

| Workflow (file) | `*_MAX_TOKENS` | Task budgets (`iter`/`att`/`steps`) |
|-----------------|----------------|--------------------------------------|
| `oas_enrichment.py` | 120k (enrich agents) | enrich `3/6/30`; update `2/2/20` |
| `oas_building.py` | 100k (swe/builder) | build `2/4/20`; others `1/2/20` |
| `likec4_building.py` | 100k | `1/2/20`-class stages |
| `trace_annotation.py` | 80k | `1/3/20` |
| `trace_annotation_direct.py` | 100k | — |
| `trace_graph.py` / `trace_graph_pathpar.py` | 100k | — |
| `trace_verify.py` | 80k | `1/2/20` |
| `vuln_scan.py` | 80k | `1/2/75` |
| `vuln_scan_fast.py` | 80k scan / 100k assess | scan `1/2/50`; assess `1/2/20` |
| `vuln_scan_trace.py` | 80k scan / 80k trace | scan `1/2/75`; trace `1/1/30` |
| `vuln_assess.py` | 100k | assess `1/2/20`; one stage `2/4/20`; final `1/1/20` |
| `exploitability.py` | 80k | `1/2/25` |
| `router.py` | 120k | `ROUTER_MAX_STEPS` |

Notes:
- **`max_tokens` here is a *summarization trigger*, not a generation cap** — raising it
  retains more context (better cross-file reasoning) at higher per-call token cost and
  risk of hitting the model's true context window.
- `vuln_scan` uses a large `steps=75` budget (deep, single-pass exploration); the
  `_fast` variant cuts it to 50.

---

## 7. Task runner & retry semantics (`runners/models.py`, `runners/task_runner.py`)

| Param | Default | Semantics |
|-------|---------|-----------|
| `iterations` (`models.py:90`) | `1` | **Successful** runs required before a task is "done". |
| `max_attempts` (`models.py:91`) | `1` (resolved to `max(1, iterations)`) | Upper bound on tries; exhausting it without enough successes → `TaskNotCompletedError`. |
| `max_steps` (`models.py:92`) | `15` | Per-attempt planner subtask budget (overridden per task above). |
| `default_iterations` / `format` (template) | `1` / `json` | From task YAML; resolution logic at `task_runner.py:366-381` enforces `max_attempts ≥ iterations ≥ 1`. |

> **Resilience gap:** default `max_attempts == iterations`, so a single transient failure
> kills a task unless the workflow explicitly sets a buffer (as enrich/build do). Raising
> `max_attempts` above `iterations` is the cheap reliability knob.

Task templates (`contractor/tasks/*.yml`) all currently use `iterations: 1` and
`format: json`; `skills:` is set only on `likec4_*` (likec4) and `vuln_scan*` (vuln_scan).

---

## 8. Planner / subtask budget (`agents/planning_agent/`, `tools/tasks/`)

| Param | Default | Effect |
|-------|---------|--------|
| `max_steps` (planner, `agent.py:35`) | `15` | Total subtask budget (`add_subtask` + `decompose_subtask` share it). Substituted into `<<MAX_SUBTASKS>>`. |
| Bootstrap ratio (prompt `v5.md:72`) | `0.7` | ≤70% of budget for initial subtasks; reserves 30% for mid-run decomposition. |
| Decomposition cardinality | `1–3` children | Branching width when refining an `incomplete`/`malformed` subtask. |
| `max_records` (`tools/tasks/tools.py:217`) | `20` | Subtask history records returned to planner — context vs. recall. |
| `n_retries` (worker parse, `tools.py:218`) | `3` | Parse-retry budget for malformed worker output before decompose/skip. |
| `_MAX_LITERAL_EVAL_LEN` (`models.py:88`) | `50000` | Char cap for literal-eval JSON recovery of large outputs. |

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

> Note: `RepeatedToolCallCallback` uses an `assert threshold > 1` — per project rule
> [no `assert` in production code], that's a latent cleanup target, not a tuning knob.

---

## 10. Tool caps (`contractor/tools/`)

| Tool / const | Default | Effect |
|--------------|---------|--------|
| `fs` `max_output` (`read_tools.py:58`) | `80000` chars | Truncates directory listings. |
| `fs` `max_items` (`read_tools.py:59`) | `100` | Listing pagination. |
| code `_MAX_WALK_DEPTH` | `50` | Dir nesting cap (symlink-loop guard). |
| code `_MAX_FILES_PER_WALK` | `100000` | Runaway-scan guard. |
| graph `DEFAULT_MAX_RESULTS` | `200` | Symbol search results. |
| graph `DEFAULT_MAX_PATHS` | `25` | Call-path enumeration cap. |
| graph `_MAX_PATH_DEPTH` | `30` | Path depth cap (exponential-blowup guard). |
| `list_symbols` page size | `300` (hardcoded) | Symbol-listing pagination. |
| HTTP `timeout` | `30.0`s | Per-request. |
| HTTP `body_preview_chars` | `2048` (512 in exploit agents) | Inline body preview; rest via `http_read_body`. |
| HTTP `history_size` | `20` | Session request history. |
| HTTP `verify_ssl` | `True` (False behind Caido) | TLS verification. |
| HTTP `RetryConfig` | `attempts=3`, `base_delay=0.5`, `max_delay=8.0`, statuses `(408,425,429,500,502,503,504)` | Transient-failure backoff. |
| likec4 `validate` timeout | `120.0`s | Linter subprocess ceiling. |

---

## 11. Prompt versions (quality lever, zero token cost)

Each agent dir has `prompt.yml` with an `active:` version selecting `prompts/v*.md`
(`load_prompt(name)` / `load_prompt_with_version(name, version)`). Switching the active
version is a pure quality lever. The trace eval already A/Bs versions via
`CONTRACTOR_EVAL_TRACE_PROMPT_VERSION`; the same pattern works for any agent.
Versioned agents include: `planning_agent`, `trace_agent`, `vuln_scan_agent` (active `v2`),
`exploitability_agent`, `web_exploitability_agent`, `swe_edit_agent`, `http_agent`,
`threat_model_agent`, `oas_*`.

---

## 12. Tuning playbook

**Speed / cost down (sacrifice depth):**
- Use a smaller/faster `--model` alias; keep `default_model_timeout` modest.
- `--resume` aggressively on re-runs.
- Lower per-workflow `*_MAX_TOKENS` (earlier summarization) and `elide_keep_last_n` (e.g. 8–10).
- Trim `max_steps` on exploratory workflows (`vuln_scan` 75→50, à la `_fast`).
- Keep `use_langfuse=False`.

**Quality / completeness up (accept cost):**
- Set `temperature≈0` for structured tasks (§5) — likely the highest ROI change available,
  since it directly cuts `malformed`/retry churn.
- Raise `iterations` (2–3) on the tasks whose output you most need to converge
  (enrich already uses 3; trace/vuln verdicts are candidates).
- Raise `max_attempts` above `iterations` everywhere to survive transient failures cheaply.
- Raise planner `max_steps` for large codebases (more decomposition headroom).
- Raise `*_MAX_TOKENS` toward the model's real context window for cross-file reasoning.

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
