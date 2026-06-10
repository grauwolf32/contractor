# Eval Tuning Configs

A design for parameter-sweep configurations runnable through the eval suite
(`tests/eval/`), so tuning decisions (sampling, tool params, budgets) are backed
by precision/recall/verdict numbers instead of guesses.

Pairs with [tuning.md](tuning.md) (the full knob inventory). This doc is narrower:
**which knobs are worth sweeping in evals, where to inject them, and the concrete
experiments to run** — prioritizing sampling (temperature/top_p) and tool params.
Status: the Tier 1–2 plumbing has since landed via `Settings` env vars (see §2/§5);
the harness `max_tokens` literal is the one remaining gap.

---

## 1. What the eval harness already gives us

| Capability | Where | Note |
|------------|-------|------|
| Eval gate | `conftest.py` `pytest_collection_modifyitems` | Eval tests auto-skip unless `-m eval` (word-matched in the `-m` expression) or `CONTRACTOR_RUN_EVAL` is truthy (`1/true/yes/on` — `0`/`false` stay off). |
| Model override | `conftest.py:279` `eval_model` fixture | `CONTRACTOR_EVAL_MODEL` → `LiteLlm(model=..., timeout=600)`; unset → the shared `DEFAULT_MODEL`. **Single chokepoint — every eval agent gets this object.** |
| Sampling override | `Settings` (`contractor/utils/settings.py`) | `MODEL_TEMPERATURE` / `MODEL_TOP_P` env vars flow through `build_model()` into `DEFAULT_MODEL`. Caveat: the `CONTRACTOR_EVAL_MODEL` path builds a bare `LiteLlm` that does **not** carry them — sweep sampling via `DEFAULT_MODEL_NAME` + `MODEL_*` instead. |
| Prompt-version override | trace/exploit/vuln tests | `CONTRACTOR_EVAL_*_PROMPT_VERSION` (env wins over the case's `prompt_version`, else the manifest's `active`). |
| Task-version override | `runners/models.py` | `CONTRACTOR_TASK_VERSION_<NAME>` (e.g. `..._TRACE_ANNOTATION=v3`) pins a task-template version for task/pipeline evals. |
| Toolset toggle | `trace_harness.py:228`, `vuln_scan_harness.py:62` | `with_graph_tools: bool`. Trace cases set it per-case (`with_graph_tools`, default **true**); the vuln test passes `True`. |
| Pass@N | vuln + trace tests | `CONTRACTOR_EVAL_VULN_PASS_AT` (default 3), `CONTRACTOR_EVAL_TRACE_PASS_AT` (default 1); passes if any attempt clears thresholds. |
| Score thresholds | vuln test | `CONTRACTOR_EVAL_VULN_MIN_RECALL` (0.15) / `_MIN_PRECISION` (0.10). |
| Observations A/B | `tools/observations.py` | `CONTRACTOR_EVAL_OBSERVATIONS` (JSON) overlays any workflow's `observations:` block — flip arms without editing config.yaml. |
| OAS-in-prompt arm | `test_trace_agent_eval.py` | `CONTRACTOR_EVAL_WITH_OAS=1` feeds the expected OpenAPI spec as an attack-surface map (default off). |
| Agent select / proxy / subsets | vuln + exploit tests | `CONTRACTOR_EVAL_VULN_AGENT` (`vuln_scan`\|`trace`), `CONTRACTOR_EVAL_PROXY`, `CONTRACTOR_EVAL_CASE_IDS` (exploit task eval). |
| Result envelope | `tests/eval/results.py` (`EvalSink` / `write_eval_results`) | One `eval/v1` envelope per `(scenario, unit)` under `eval_runs/`; `CONTRACTOR_EVAL_RUN_STAMP` names the archive dir, `CONTRACTOR_EVAL_RESULTS_DIR` redirects vuln records. `scripts/rebuild_eval_envelope.py` re-aggregates a unit's envelope from per-case `metrics.json` when fixtures ran in separate sessions. |
| Timeout handling | `tests/eval/harness.py` | `run_agent(timeout_s=...)` (trace cases: per-case `timeout_s`, default 900) raises `AgentRunTimeout` carrying the **partial** `AgentRun` (`timed_out=True`) — a timeout is always a failed attempt, never a silent pass, but stays inspectable. |
| N-run aggregation | `scripts/probe_variance.py` | mean/stdev/min/max per metric across samples. |

**Scoring is deterministic** (set-matching precision/recall/f1; verdict equality) —
no LLM judge — so any sweep produces directly comparable numbers. Variance comes
only from the model's own non-determinism, which is exactly what the sampling sweep
should quantify.

**Remaining gap:** `max_tokens` is still hardcoded `80_000` in all three harnesses
(`trace_harness.py:264`, `vuln_scan_harness.py:83,100`, `exploitability_harness.py:86,101,116`).
Sampling and tool caps are no longer gaps — both are Settings/env-routed (see below).

---

## 2. The eval-config surface (prioritized)

Each config is a bundle of overrides. Group by tuning ROI:

### Tier 1 — Sampling (DONE — Settings-routed)

| Knob | Env var (implemented) | Sweep values | Hypothesis |
|------|------------------|--------------|------------|
| `temperature` | `MODEL_TEMPERATURE` | `0.0, 0.2, 0.4, 0.7` | Low temp ↑ schema adherence on structured tasks (OAS/trace/verdict) → fewer `malformed`/retries; high temp ↑ recall on open-ended vuln scan. |
| `top_p` | `MODEL_TOP_P` | `1.0, 0.9, 0.8` | Nucleus trim stabilizes output without flattening exploration as hard as temperature. |
| (optional) `reasoning_effort` / thinking budget | still unwired | backend-specific | Trade latency for depth on vuln_scan / exploitability. |

**Injection landed in `Settings`, not the eval fixture.** `model_temperature` /
`model_top_p` (`contractor/utils/settings.py`) are forwarded by `build_model()` to
every `LiteLlm` it constructs — including `DEFAULT_MODEL`, which `eval_model`
returns when `CONTRACTOR_EVAL_MODEL` is unset. So sampling sweeps work across
**all** eval types *and* production runs with two env vars; default `None` keeps
backend defaults. Caveat: the `CONTRACTOR_EVAL_MODEL` override path constructs a
bare `LiteLlm(model=..., timeout=600)` without the sampling kwargs — to sweep
sampling on a non-default model, set `DEFAULT_MODEL_NAME` instead.

### Tier 2 — Tool params (DONE — Settings-routed)

These shape what the agent can *see* and how much it costs. All caps below now live
in `Settings` (tool constructors fall back to `get_settings()` when no explicit value
is passed), so each is sweepable via plain env vars — no `CONTRACTOR_EVAL_*` plumbing:

| Knob | Env var (implemented) | Default | Sweep values | Exercised by | Hypothesis |
|------|---------------|------------------|--------------|--------------|------------|
| `with_graph_tools` | — (per-case key in `trace-cases.json`, default **on**; vuln test passes `True`) | on | `false` per case to ablate | trace, vuln (trace mode) | Graph/trailmark tools ↑ cross-file recall on large fixtures (the v7+graph result — now the production default). |
| graph max results | `GRAPH_MAX_RESULTS` | `200` | `100, 200, 400` | trace, vuln | More symbol hits ↑ recall but ↑ context/token cost. |
| graph max paths / depth | `GRAPH_MAX_PATHS` / `GRAPH_MAX_PATH_DEPTH` | `25` / `30` | paths `15/25/40`, depth `20/30` | trace | Deeper call-path enumeration ↑ taint-flow recall; risk of blow-up/noise. |
| HTTP `body_preview_chars` | `HTTP_BODY_PREVIEW_CHARS` (512 in exploit agents) | `2048` | `256, 512, 2048` | exploitability, web | Bigger preview ↑ evidence quality for verdicts vs. token burn. |
| fs `max_items` / `max_output` / line cap | `FS_MAX_ITEMS` / `FS_MAX_OUTPUT` / `FS_MAX_READ_LINES` | `100` / `50k` / `2000` | items `100/250`, output `50k/80k/160k` | all code evals | Wider listings ↑ discovery on big repos vs. context cost. (The 80KB-vs-50KB/2000-line A/B was inconclusive — the cap rarely binds on small fixtures.) |

### Tier 3 — Agentic budgets (already partly wired)

| Knob | Source | Env var | Sweep values | Note |
|------|--------|------------------|--------------|------|
| `max_tokens` (summarization trigger) | still hardcoded `80_000` in 3 harnesses | *(proposed)* `CONTRACTOR_EVAL_MAX_TOKENS` | `60k, 80k, 100k, 120k` | Replace the literals; bigger budget ↑ cross-file reasoning on large fixtures, ↑ cost. **The one un-landed item.** |
| `elide_keep_last_n` / char budget | `worker_factory.py` (15 / off) | `FS_HEAVY_KEEP_LAST_N` / `FS_HEAVY_KEEP_BUDGET_CHARS` (implemented; >0 overrides the caller) | keep `8, 15, 25`; budget per QW3 | Recall of earlier tool output vs. token cost (QW3 byte-retention: −44% tokens). |
| planner `max_steps` | per-pipeline `config.yaml` | — (edit YAML) | `15, 30, 75` | Only relevant for task-runner evals (oas/vuln_assess), not single-agent harnesses. |
| `pass@N` | vuln + trace | `CONTRACTOR_EVAL_VULN_PASS_AT` (default 3) / `CONTRACTOR_EVAL_TRACE_PASS_AT` (default 1) | `1, 3, 5` | Wired for both; extend the pattern to exploit if you want pass@N there. |
| observations arms | workflow `observations:` blocks | `CONTRACTOR_EVAL_OBSERVATIONS` (JSON overlay, implemented) | lean / +file-paths / +tool-errors / off | Lean+file-paths is the current production arm; tool-error counts measurably hurt. |
| task-prompt version | `contractor/tasks/<name>.yml` manifests | `CONTRACTOR_TASK_VERSION_<NAME>` (implemented) | registered versions | A/B task-template variants in task/pipeline evals (e.g. trace_annotation `v3`, now active). |

---

## 3. Named configs to run (the experiment matrix)

Each row is a config worth committing as a reproducible experiment. Run with
`CONTRACTOR_RUN_EVAL=1` and the listed env vars; aggregate ≥3 samples with
`probe_variance.py` (or pass@N for vuln) since sampling adds variance.

| Config name | Overrides | Target fixtures | Metric to watch | Question it answers |
|-------------|-----------|-----------------|-----------------|---------------------|
| `baseline` | none (backend-default sampling, graph tools on) | all | p/r/f1, success rate, malformed count | Current production behavior + its variance floor. |
| `det-t0` | `MODEL_TEMPERATURE=0` | trace, oas, exploit | malformed/retry count, verdict accuracy | Does greedy decoding cut format churn without hurting recall? |
| `det-t0-p09` | `MODEL_TEMPERATURE=0`, `MODEL_TOP_P=0.9` | trace, oas | f1, variance (stdev across N) | Best stability/quality point for structured output. |
| `explore-t07` | `MODEL_TEMPERATURE=0.7` | vuln_scan (pass@5 via `CONTRACTOR_EVAL_VULN_PASS_AT=5`) | recall@5, unique findings union | Does higher temp + more attempts ↑ vuln coverage? |
| `graph-off` | `with_graph_tools: false` per trace case | trace, vuln-trace (large: cloud-core, crapi) | recall, annotation count | Ablate the now-default graph tools; quantify the token tax they pay for. |
| `graph-rich` | `GRAPH_MAX_PATHS=40`, `GRAPH_MAX_RESULTS=400` | large trace fixtures | recall vs. precision (noise) | Diminishing returns / precision loss from deeper graph. |
| `lean-context` | `MAX_TOKENS=60000` (once landed), `FS_HEAVY_KEEP_LAST_N=8` | all | f1 delta vs. baseline, tokens/run | How cheap can we go before quality drops? |
| `rich-context` | `MAX_TOKENS=120000` (once landed), `FS_HEAVY_KEEP_LAST_N=25` | large fixtures | recall delta, tokens/run | Is the cost of more context justified on big repos? |
| `evidence-rich` | `HTTP_BODY_PREVIEW_CHARS=2048` | exploitability, web | verdict accuracy, evidence-present rate | Does more response body ↑ correct verdicts? |

Run sampling × context as a small grid on a fixed fixture subset first (cheap, fast
fixtures like `realvuln-vampi`, `realvuln-dsvw`, `vulnyapi`), then promote the winning
config to the expensive large fixtures (`crapi-*`, `cvebench-*`).

---

## 4. Suggested workflow

1. **Plumbing is landed for Tiers 1–2** (Settings env vars) — only the harness
   `max_tokens` literal (Tier 3) remains.
2. **Pin a fixture subset** for fast iteration (small/medium fixtures) via
   `-k` (any eval) or `CONTRACTOR_EVAL_CASE_IDS` (exploit task eval).
3. **Sample ≥3× per config** — sampling sweeps are meaningless single-shot; use
   pass@N (`CONTRACTOR_EVAL_VULN_PASS_AT` / `CONTRACTOR_EVAL_TRACE_PASS_AT`) or
   `probe_variance.py` aggregation and report mean ± stdev, not one number.
4. **Compare on deterministic metrics** the harness already emits (precision/recall/f1,
   success rate, verdict accuracy, annotation count, tokens/run); results land as
   `eval/v1` envelopes under `eval_runs/` for analytics-ui. Run a unit's fixtures in
   one pytest session (one combined envelope); if they ran separately, consolidate
   with `scripts/rebuild_eval_envelope.py` — don't cp-snapshot run dirs.
5. **Promote the winner** to large fixtures, then update the production defaults
   (`Settings` field defaults / `config.yaml` budgets / litellm config) — and the
   [Don't switch models] rule still stands: tune sampling/tools/budgets, keep the
   project's default model.

---

## 5. Plumbing checklist (smallest viable change set)

- [x] Sampling env-driven — landed as `Settings.model_temperature` / `model_top_p`
      (`MODEL_TEMPERATURE` / `MODEL_TOP_P` via `build_model()`), not in `eval_model`.
      **(Tier 1 done.)**
- [ ] Replace hardcoded `max_tokens=80_000` in `trace_harness.py:264`, `vuln_scan_harness.py:83,100`, `exploitability_harness.py:86,101,116` with an env-driven default. **(Tier 3 — the only open item.)**
- [x] `with_graph_tools` — per-case key in trace cases (default on); vuln eval runs with it on. No env var; flip per case to ablate. **(Tier 2 done, differently than proposed.)**
- [x] Graph / HTTP / fs caps — landed as `Settings` fields (`GRAPH_MAX_*`, `HTTP_BODY_PREVIEW_CHARS`, `FS_MAX_*`); constructors fall back to `get_settings()`, production defaults unchanged. **(Tier 2 done.)**
- [x] Env vars documented in test-module header docstrings (e.g. `test_vuln_detection_eval.py`).

> Guardrails to keep: production defaults must not change from these eval hooks (env-gated,
> default = current constant), and no `assert` in the new override code — use explicit
> `if ...: raise` per project rule.

---

*Injection points verified against `tests/eval/conftest.py`, `tests/eval/harness.py`,
`tests/eval/results.py`, `trace_harness.py`, `vuln_scan_harness.py`,
`exploitability_harness.py`, `contractor/utils/settings.py`, and `tests/eval/scorers.py` /
`scoring.py`. Confirm `LiteLlm` kwarg forwarding in the installed ADK before trusting
sampling numbers.*
