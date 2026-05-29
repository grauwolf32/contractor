# scripts/

Operator & developer tooling — **not** part of the `contractor` package. These
drive evals, analyze run output (`metrics.jsonl`), and produce comparison
reports. They are not imported by `cli/` or `contractor/`, are not covered by
the test suite, and may lag behind format changes faster than the library.

Run any of them with Poetry; most accept `--help`:

```bash
poetry run python scripts/<name>.py --help
```

## Run analysis & observability

Operate on a finished run's `<output_dir>/metrics.jsonl` (default
`<project>/.contractor/metrics.jsonl`).

| Script | Purpose |
| --- | --- |
| `analyze_metrics.py` | Charts, tables, and a report from `metrics.jsonl` (referenced in CLAUDE.md). |
| `diagnose.py` | Surface concrete failure-mode fingerprints from `metrics.jsonl`. |
| `visualize_subtasks.py` | Render subtask graphs from a run's `metrics.jsonl`. |
| `dump_langfuse_trace.py` | Dump a Langfuse trace into compact agent-grouped JSON (referenced in CLAUDE.md). |

## Eval running

Drive the agents over fixtures (slow, LLM-bound; need the LiteLLM proxy up).

| Script | Purpose |
| --- | --- |
| `run_vuln_eval.py` | Run trace-agent vulnerability-detection evals + HTML report. |
| `run_exploit_eval.py` | Run exploitability-agent evals across all fixtures + HTML report. |
| `eval_sweep.py` | Run a pipeline across multiple prompt/task version combinations. |
| `analyze_vuln_eval.py` | Run vuln_scan + trace agents on a fixture and dump detailed traces. |
| `probe_trace.py` | Probe trace_agent on an arbitrary project + entrypoint (no fixture). |
| `probe_variance.py` | Run `probe_trace` N times and aggregate (mean/std/min/max). |

## Eval reporting (HTML)

Render results produced by the eval runners above.

| Script | Purpose |
| --- | --- |
| `vuln_eval_report.py` | Self-contained HTML report from vuln-eval results. |
| `exploit_eval_report.py` | Self-contained HTML report from exploitability eval results. |

## A/B comparison

| Script | Purpose |
| --- | --- |
| `compare.py` | Unified A/B comparison CLI for eval runs. |
| `compare_eval_runs.py` | Compare two `trace_agent` configurations on a single fixture. |
| `compare_task_runs.py` | A/B two task-pipeline configurations on a single fixture. |
| `compare_exploit_variants.py` | HTML comparison across exploitability-agent variants. |

## Eval fixture prep

| Script | Purpose |
| --- | --- |
| `prepare_vuln_benchmarks.py` | Prepare vulnerability benchmark fixtures for the eval harness. |
| `precompute_task_artifacts.py` | Precompute `dependency_information` + `project_information` artifacts for fixtures. |

## Misc utilities

| Script | Purpose |
| --- | --- |
| `caido_auth.py` | Exchange a Caido PAT for an access token and print it. |
| `measure_trailmark.py` | Measure trailmark cost on a target codebase. |
