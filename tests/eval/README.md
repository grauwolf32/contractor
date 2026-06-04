# Contractor evals

End-to-end, LLM-bound evaluations of Contractor's agents, tasks, and
workflows against intentionally-vulnerable target codebases. Evals measure
real model behavior (not unit logic), so they are **slow, non-deterministic,
and opt-in**, and every producer converges on one on-disk shape — the
`eval/v1` envelope — that [analytics-ui](../../analytics_ui) reads.

- Eval code: `tests/eval/`
- Standalone scripts: `scripts/run_*_eval.py`, `scripts/score_*`, `scripts/compare_*`
- Results: `eval_runs/**/eval_results.json` (gitignored)
- Viewer: `python -m analytics_ui`

---

## 1. Prerequisites

1. **LiteLLM proxy up.** `--model` is an alias resolved by the proxy, not a
   model name. Start it: `cd deploy/litellm && bash run.sh` (needs Podman).
2. **Fixtures submodule.** Eval fixtures resolve `source_root` into the
   `security-playground` submodule:
   `git submodule update --init --recursive`.
3. **Model.** Defaults to `Settings.DEFAULT_MODEL`; override with
   `CONTRACTOR_EVAL_MODEL`. Eval `LiteLlm` instances use a 600 s per-request
   timeout (raise the timeout, don't switch models, when runs are slow).
4. **Opt in.** Eval tests auto-skip in normal `pytest` runs.

---

## 2. The three scenarios

Every eval is exactly one scenario; they differ only in *what one attempt
runs*. Each case is repeated **pass@X** times via
`results.pass_at(...)` — the case passes iff **any** attempt passes.

| Scenario   | One attempt drives…                            | pass@ | Harness |
|------------|------------------------------------------------|-------|---------|
| `agent`    | a single `LlmAgent`                            | N     | `harness.run_agent` |
| `task`     | one `TaskRunner` task (planner + worker chain) | M     | `task_harness.run_task_pipeline` |
| `pipeline` | a whole workflow / CLI run                      | K     | the workflow's own `.run()` |

`scenarios.py` exposes thin wrappers (`run_agent_eval` / `run_task_eval` /
`run_pipeline_eval`) over a shared `run_eval`, but most evals call
`pass_at` (or `eval_sink.record`) directly.

### Pass isolation (important)

One harness call = **one pass**. Within a pass, all `max_attempts` retries
and `iterations` **share** the artifact tree, so memory is reused across
retries (the planner relies on this). **Across passes you must isolate** —
otherwise pass 2 reads pass 1's memory (`user:memory/{ns}`).

- `run_agent` / `run_task_pipeline` create a fresh `FileArtifactService`
  rooted at `artifact_dir` (or a per-call `TemporaryDirectory` when omitted).
- For pass@M, give each pass a **distinct artifact root**. The agent evals do
  this by suffixing namespace + `artifact_dir` per attempt (see
  `test_vuln_detection_eval`). Task evals can pass
  `run_task_pipeline(..., run_id=attempt)` — the tree nests under
  `<artifact_dir>/pass-<run_id>` with a per-pass Langfuse session, isolating
  memory even when per-task namespaces are identical.

---

## 3. The `eval/v1` envelope

`tests/eval/results.py` is the single source of truth. Producers build an
`EvalRun` and `write_eval_results(run, out_dir)` serializes
`<out_dir>/eval_results.json`:

```jsonc
{
  "schema": "eval/v1",
  "scenario": "agent|task|pipeline",
  "unit": "trace_agent",          // agent / task-template / workflow name under test
  "metric_kind": "detection|verdict|capture|diff|generic",
  "pass_at": 1,
  "model": "lm-studio-qwen3.6",
  "prompt_version": "v7",
  "timestamp": "2026-06-03T…Z",
  "meta": { },                    // free-form run-level extras
  "fixtures": [
    { "slug": "vulnyapi",
      "cases_total": 1, "cases_passed": 1,
      "cases": [
        { "id": "vulnyapi", "passed": true, "pass_count": 1, "attempts": 1,
          "metrics": { "total_tokens": …, "total_tool_calls": …, "tool_counts": {…} },
          "detail":  { /* domain-specific, see metric_kind */ },
          "runs":    [ /* per-attempt breakdown, only when attempts > 1 */ ] }
      ] }
  ],
  "headline": { /* derived snapshot, see below */ },
  "totals":   { /* tokens, tool calls, errors, skill reads, … */ }
}
```

`headline`/`totals` are derived from `fixtures` and embedded so the file is
self-describing; the UI recomputes them for legacy files.

### `metric_kind` → `detail` fields + headline

| `metric_kind` | case `detail` carries…                       | headline adds |
|---------------|----------------------------------------------|---------------|
| `detection`   | `tp,fp,fn,tn,precision,recall,f1,per_cwe,reported_findings,matches` | micro `precision/recall/f1` |
| `verdict`     | `expected_verdict,actual_verdict,has_evidence` | `evidence_rate` |
| `capture`     | `captured,chain,tags`                        | `chain_rate` |
| `diff`        | `precision,recall,f1,matched,missing,extra`  | `mean_f1` |
| `generic`     | anything (pass/fail only)                    | — |

All headlines always include `pass_rate / passed / total`.

### Metrics helpers

- `metrics_from_events(events)` — fold agent plugin events (tool calls,
  errors, llm usage) into the analytics bag.
- `metrics_from_task(task_metrics)` — fold `TaskRunner` per-task `TaskMetrics`.
- `derive_totals(fixtures)` — sum tokens / tool calls / `tool_errors` /
  `skill_reads` across cases.

---

## 4. Directory layout

```
tests/eval/
  README.md                 ← this file
  results.py                ← eval/v1 envelope, pass_at, EvalSink, case_artifact_dir
  scenarios.py              ← agent/task/pipeline wrappers over run_eval
  conftest.py               ← eval marker skip, fixture loader, eval_model/eval_sink
  harness.py                ← run_agent (agent scenario)
  task_harness.py           ← run_task_pipeline (task scenario) + per-pass run_id
  trace_harness.py          ← trace_agent driver + annotation extraction/scoring
  vuln_scan_harness.py      ← codereview_agent / trace_agent vuln scan
  exploitability_harness.py ← exploitability agents
  trace_vuln_scoring.py     ← vuln-finding scorer for trace_annotation task runs
  scorers.py / scoring.py   ← OAS / trace / vuln scoring primitives
  test_*_eval.py            ← one file per eval (see §6)
  fixtures/<slug>/          ← targets + ground truth (see §5)

eval_runs/                  ← results root (gitignored), one eval_results.json per run
  <unit>/eval_results.json            (from eval_sink)
  <unit>/cases/<fixture>__<case>/…    (live artifact trees, from case_artifact_dir)
  <script-run>/eval_results.json      (from scripts)
```

`EVAL_ROOT` is `eval_runs/`. analytics-ui `rglob`s it for `eval_results.json`.

---

## 5. Fixtures & ground truth

Each fixture is `tests/eval/fixtures/<slug>/` with a `meta.yaml`:

```yaml
slug: vulnyapi
language: python
framework: fastapi
source_root: tests/playground/python/vulnyapi   # into the submodule
description: >
  Mid-sized intentionally-vulnerable FastAPI service…
```

Ground-truth files (present per the eval that consumes them):

| File                              | Consumed by | Shape |
|-----------------------------------|-------------|-------|
| `vuln-cases.json`                 | vuln detection | per-vuln `{id, is_vulnerable, primary_cwe, file, function, …}` |
| `vulnerabilities.expected.json`   | trace-task vuln scoring | `{vulnerability(class), method, path, severity}` |
| `trace-cases.json`                | trace agent / parallel | entrypoint + `expected_annotated [{file,function}]` |
| `oas.expected.yaml`               | oas build / trace task | expected OpenAPI spec |
| `exploitability-cases.json`       | exploitability | per-finding verdict expectations |
| `swe-/planner-/task-cases.json`   | swe / planner / task | scenario-specific |

Per-case fixtures (`trace_case`, `swe_case`, `planner_case`,
`exploitability_case`) are auto-parametrized in `conftest.pytest_generate_tests`
— each `(slug, case_id)` becomes its own pytest item.

---

## 6. The evals

| Test file | scenario | unit | metric_kind |
|-----------|----------|------|-------------|
| `test_trace_agent_eval` | agent | trace_agent | diff (annotations) |
| `test_vuln_detection_eval` | agent | codereview_agent / trace_agent | detection |
| `test_exploitability_eval` | agent | exploitability_agent | verdict |
| `test_oas_builder_eval` / `test_oas_analyzer_eval` | agent | oas_* | diff |
| `test_swe_agent_eval` | agent | swe_agent | generic |
| `test_oas_build_task_eval` / `oas_enrich` / `project_information` / `likec4` | task | task template | diff |
| `test_exploitability_task_eval` | task | exploitability_assessment | verdict |
| `test_threat_analysis_task_eval` | task | threat_analysis (STRIDE) | diff (structural + endpoint coverage) |
| `test_planner_eval` | task | planner | diff |
| `test_xbow_eval` | task | xbow:web_exploit / xbow:exploit | capture |
| `test_trace_parallel_eval` | pipeline | trace (graph variants) | diff |

---

## 7. Running

```bash
# whole suite (opt-in, slow)
poetry run pytest -m eval
CONTRACTOR_RUN_EVAL=1 poetry run pytest tests/eval/

# one eval / one fixture / one case
CONTRACTOR_RUN_EVAL=1 poetry run pytest tests/eval/test_trace_agent_eval.py
poetry run pytest -m eval tests/eval/test_vuln_detection_eval.py -k vulnyapi
poetry run pytest -m eval tests/eval/test_trace_agent_eval.py -k "vulnyapi/sqli" -s
```

`-s` streams the per-case scoring tables. Results land in
`eval_runs/<unit>/eval_results.json` (written by `eval_sink` at session end).

### Environment knobs

| Var | Effect |
|-----|--------|
| `CONTRACTOR_RUN_EVAL=1` | run eval tests without `-m eval` |
| `CONTRACTOR_EVAL_MODEL` | model alias override (else `DEFAULT_MODEL`) |
| `CONTRACTOR_EVAL_TRACE_PROMPT_VERSION` | pin trace_agent prompt across all cases |
| `CONTRACTOR_EVAL_VULN_AGENT` | `vuln_scan` (default) or `trace` |
| `CONTRACTOR_EVAL_VULN_PASS_AT` | N for vuln-detection pass@N (default 3) |
| `CONTRACTOR_EVAL_VULN_PROMPT_VERSION` | vuln eval prompt version |
| `CONTRACTOR_EVAL_VULN_MIN_PRECISION` / `_MIN_RECALL` | pass thresholds |
| `CONTRACTOR_EVAL_CASE_IDS` | comma-list subset (exploitability task) |
| `CONTRACTOR_XBOW_BENCHMARKS` / `CONTRACTOR_XBOW_AGENT` | xbow subset / single agent |
| `CONTRACTOR_EVAL_RESULTS_DIR` | override the results root (vuln eval) |

---

## 8. Standalone scripts

Richer, report-generating drivers that bypass pytest. All write the same
`eval/v1` envelope plus extra artifacts.

| Script | What it runs |
|--------|--------------|
| `scripts/run_vuln_eval.py` | trace/codereview agent vuln scan + HTML report (agent scenario, detection) |
| `scripts/run_trace_task_eval.py` | `trace_annotation` **task** (planner+worker per OpenAPI path); emits annotation `diff` + vuln `detection` envelopes; per-path timeout salvages partial overlays |
| `scripts/score_trace_task_vulns.py` | post-hoc vuln scoring of a finished trace-task run dir |
| `scripts/run_exploit_eval.py` | exploitability agents (verdict / capture) |
| `scripts/compare_eval_runs.py` | A/B two trace_agent configs on one fixture/case |
| `scripts/compare_task_runs.py` | A/B two task-pipeline configs |
| `scripts/compare_exploit_variants.py` | HTML comparison across exploit variants |
| `scripts/eval_sweep.py` | Cartesian sweep over `--vary name=v1,v2` prompt/task axes |
| `scripts/analyze_metrics.py` / `analyze_vuln_eval.py` | post-run charts/tables |
| `scripts/prepare_vuln_benchmarks.py` | import external benchmarks (CVE-Bench, …) as fixtures |

Example — `trace_annotation` task eval, v7 vs shannon (run sequentially; one
local model):

```bash
poetry run python scripts/run_trace_task_eval.py --prompt v7 \
  --per-path-timeout 700 --output eval_runs/trace-task-v7
poetry run python scripts/run_trace_task_eval.py --prompt shannon \
  --per-path-timeout 1000 --output eval_runs/trace-task-shannon
```

---

## 9. Vuln-finding scoring (trace_annotation task)

`trace_vuln_scoring.py` scores the vulnerabilities a `trace_annotation` run
surfaces (the task's real output, beyond annotation placement). It unions
findings from `report_vulnerability` artifacts **and** the Shape A/B/C/D
blocks in each path's result text, normalizes each to a **general AppSec
family** (`sqli, ssrf, idor, csrf, path-traversal, sensitive-data,
auth-crypto, rate-limit-abuse, business-logic, …` — not benchmark-specific),
attributes it to the operation path via `@trace target=` markers, and matches
against `vulnerabilities.expected.json` (family + path, greedy one-to-one,
exact-path preferred) → detection `precision/recall/f1`.

Keep the taxonomy general — encoding a fixture's specific sinks is overfitting
and invalidates the eval.

---

## 10. Viewing results

```bash
python -m analytics_ui            # serves http://127.0.0.1:8765 (auto-opens)
python -m analytics_ui --port 9000 --no-browser
```

It reads `eval_runs/**/eval_results.json` live per request (browser caches —
hit ↻ Refresh after a new run). Each run is a card grouped by scenario, with
the metric_kind domain panel (PRF / verdict matrix / capture chain / F1),
token/tool/error charts, and skill-usage. Non-envelope JSON is ignored.

---

## 11. Adding a new eval

1. Pick the scenario (agent / task / pipeline) and `metric_kind`.
2. Write `test_<thing>_eval.py`: build the unit, run it through the matching
   harness with `artifact_dir=case_artifact_dir(unit, fixture, case_id)`
   (live trace), score it, and emit one `eval_sink.record(scenario=…, unit=…,
   metric_kind=…, fixture=…, case=CaseResult(...), artifacts=…)` per case.
3. For pass@X, repeat the attempt and isolate per pass (distinct
   `artifact_dir` / `run_id`; see §2).
4. Add ground-truth files under `fixtures/<slug>/`.
5. Mark the test `@pytest.mark.eval` and verify it appears in analytics-ui.

Keep scoring inside the test; the harnesses only standardize repetition,
metrics capture, artifact persistence, and the result shape.
