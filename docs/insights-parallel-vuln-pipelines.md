# Key Insights: Parallel Trace Pipelines & Vulnerability Detection

*Session: 2026-05-27 → 2026-05-28*

---

## 1. Parallelizing the Trace Pipeline

### Problem
The trace pipeline processes API operations sequentially — each operation runs a full LLM agent session (multiple tool calls, file reads, annotation writes). With 10-20 operations, wallclock time scales linearly.

### Approach: Path-Level Parallelism
Each API path gets its own forked `MemoryOverlayFileSystem`. Paths run concurrently via `asyncio.TaskGroup` with a semaphore (`max_concurrency=3`). Operations within a path remain sequential so sibling operations see each other's annotations.

```
Before:  path1 → path2 → path3 → path4  (serial, 4x wallclock)
After:   path1 ┐
         path2 ├→ merge → save  (parallel, ~1.3x wallclock)
         path3 ┘
         path4 (waits for semaphore)
```

### Key Technical Decisions

**Overlay fork/merge pattern.** Each parallel fork starts from a snapshot of the shared overlay. After all forks complete, writes are merged back. Conflict resolution: when two forks modify the same file, take the version with the most content (more `@trace` annotations = more complete trace). In practice, conflicts are rare — different API paths trace different code.

**Shared graph tools eliminate re-parse overhead.** Trailmark (call-graph engine) parses the project via tree-sitter on first use. Naively, each forked overlay triggers a separate parse — causing 7.7x slowdown in eval. Fix: build graph tools once from the base FS before forking, pass them to all agents via a new `graph_tools` parameter on `build_trace_agent`. Trailmark only reads the base FS (read-only), so sharing is safe. Result: overhead dropped from 637s to 72s.

**Per-fork `AgentRunner` instances.** `AgentRunner` stores `_on_event` as instance state — two concurrent `.run()` calls would overwrite each other. Each parallel path creates its own runner.

### Eval Results (fastapi fixture, 2 paths)

| Variant | Wallclock | Annotations | Recall |
|---|---|---|---|
| trace-graph (baseline) | 63.5s | 8 | 1.00 |
| trace-graph-pathpar | 72.3s | 8 | 1.00 |

Identical annotation quality. The ~9s overhead is fork/merge cost. With a concurrent LLM backend (cloud API, multi-GPU), wallclock would halve.

### Why Operation-Level Parallelism Was Rejected
Operations within a path often share code (middleware, services). Sequential execution lets later operations see annotations from earlier ones, improving trace quality. Forking per-operation also means each operation starts blind — no cross-operation context. Tested as `trace-graph-opspar`, showed no quality benefit and significant overhead on small fixtures.

---

## 2. The Cascade Decay Problem

### The Math
In a sequential N-stage pipeline where each stage has accuracy `p`, end-to-end accuracy is `p^N`.

| Stages | p=0.80 | p=0.90 | p=0.95 |
|---|---|---|---|
| 2 | 0.64 | 0.81 | 0.90 |
| 3 | 0.51 | 0.73 | 0.86 |
| 5 | 0.33 | 0.59 | 0.77 |

At 80% per-stage, a 5-step pipeline delivers only 33% end-to-end accuracy.

### Research Findings (2025-2026)
- Google DeepMind: unstructured multi-agent networks amplify errors **up to 17.2x** vs single-agent baselines
- ICLR 2026 ("From Spark to Fire"): minor inaccuracies solidify into false consensus through "consensus inertia"
- Production rule of thumb: **keep chains under 5 steps, insert verification at steps 3 and 5**
- PwC: independent judge agents improved accuracy **7x** (10% → 70%)
- VULSOLVER: decomposing vuln detection into binary constraint checks per call-chain hop achieved **100% recall, 96.3% accuracy** on OWASP benchmarks

### Key Insight: Verification Steps Don't Multiply Error
A verification step is a *filter*, not a *generator*. It doesn't produce new information that can be wrong — it validates existing information. So the cascade formula doesn't apply:

```
Generator → Generator → Generator     = p₁ × p₂ × p₃     (compounds)
Generator → VERIFY → Generator → VERIFY = p₁ × p₃ × filter  (filters reduce FP, don't reduce TP)
```

---

## 3. Pipeline Architecture: Two Designs

### Design Rule
> Keep sequential chains under 5 steps. Insert verification agents at steps 3 and 5.

### Pipeline A: `vuln-assess` (full assessment, highest quality)

```
Step 1: project_discovery     — SWE agent: deps + structure
Step 2: oas_build             — OAS builder: extract API surface
Step 3: oas_validate  [VERIFY]— linter: verify spec, fix errors
Step 4: trace_vuln            — trace agent per-operation (parallel paths)
Step 5: exploit       [VERIFY]— exploitability agent: verify each finding
```

- Steps 1-2 skipped if artifacts already exist (idempotent)
- Step 3 catches bad OAS before expensive trace stage
- Step 5 requires `CONTRACTOR_TARGET_URL` (live target); skipped if unset
- Artifact bridge copies `user:oas-openapi-building` → `oas-openapi-building` between OAS and trace stages

### Pipeline B: `vuln-scan-fast` (high recall, no OAS dependency)

```
Step 1: project_discovery     — SWE agent (skip if exists)
Step 2: vuln_scan_fast        — breadth-first scan, intentionally over-reports
Step 3: dedup         [VERIFY]— programmatic: merge by file+CWE, keep highest confidence
Step 4: trace_confirm         — trace agent targeted per finding (confirm or deny)
Step 5: exploit       [VERIFY]— exploitability agent per finding
```

- Step 2 uses a **separate task template** (`vuln_scan_fast`) with tuned instructions for high recall — "under-reporting is worse than over-reporting"
- Step 3 is programmatic (no LLM), acts as cheap noise filter
- Step 4 uses trace agent in targeted mode: given a specific finding, trace the code path to confirm or refute

### Why Two Pipelines
Pipeline A is thorough but slow (OAS build + per-operation trace). Pipeline B skips OAS entirely and uses a fast scan + targeted confirmation, trading coverage depth for speed. For CI integration, B runs in minutes; A runs in tens of minutes but produces richer output (annotated code + OAS + verified findings).

---

## 4. Optimizing the Funnel: Over-Report Early, Filter Late

### Anti-Pattern: Balanced Precision/Recall at Every Stage
If every stage targets 80% precision and 80% recall, the funnel loses information at each step. Findings missed at stage 1 are **permanently lost** — no downstream stage can recover them.

### Pattern: Asymmetric Thresholds

```
Stage 1 (scan):    target 95% recall, accept 40% precision  → cast wide net
Stage 2 (dedup):   programmatic filter, zero recall loss     → remove noise cheaply  
Stage 3 (trace):   targeted confirmation, high precision      → kill false positives
Stage 4 (exploit): final verification against live target     → highest confidence
```

The key: **recall can only decrease through the pipeline; precision can increase**. So maximize recall at the top, refine precision at the bottom.

### Concrete Implementation
The `vuln_scan_fast` task template instructs the agent:
> "You are the first stage of a multi-stage pipeline. Your job is to MAXIMIZE RECALL. Report every plausible vulnerability, even at low confidence. A later stage will verify and filter. Missing a real vulnerability here means it is lost forever."

### Pass@K Alternative
Run the scan stage K times with prompt/temperature variation, union the findings:
- At 80% recall per run, `pass@3` union gives `1 - (1-0.8)³ = 99.2%` theoretical recall
- Cost: K× scan tokens, but only 1× verification tokens
- The parallel pipeline infrastructure already supports concurrent execution

---

## 5. Vulnerability Detection Eval Results

### Trace Agent Performance (3 benchmark fixtures, pass@1)

| Fixture | Vulns | FP Traps | Precision | Recall | F1 | Tokens |
|---|---|---|---|---|---|---|
| realvuln-pythonssti | 2 | 1 | 1.00 | 0.50 | 0.67 | 172K |
| realvuln-vfapi | 9 | 2 | 0.60 | **1.00** | 0.75 | 157K |
| realvuln-vampi | 15 | 4 | 0.62 | 0.67 | 0.65 | 190K |
| **Aggregate** | **26** | **7** | **0.68** | **0.77** | — | **519K** |

### What Works
- **Perfect recall on single-file apps** (vfapi: 9/9 found)
- **CWE mapping skill adopted immediately** when available
- **Skills references loaded consistently** (4-5 refs per fixture)
- **Structured finding format** (shape A/B/C, CWE ID, evidence lines) works well

### What Doesn't
- **Missed CWEs**: CWE-79 (XSS), CWE-770 (resource exhaustion) — detection heuristics missing from sink catalogue
- **FP rate on larger codebases** — agent over-reports when scanning many files
- **Annotation tools underused** — agent often skips `annotate_trace`/`annotate_sink` and goes straight to `report_vulnerability`
- **No "trace before report" discipline** — on 2/3 fixtures, zero annotation calls before first vulnerability report

### Recommendations
1. Enforce annotation-before-reporting in prompt (require `annotate_trace` before `report_vulnerability`)
2. Add CWE-79 XSS detection heuristics to sink catalogue (template rendering without escaping)
3. Add dedup instruction: call `list_vulnerabilities` before reporting to avoid duplicates
4. Expand CWE mapping reference with detection patterns, not just ID mapping

---

## 6. HTML Analytics Report

Self-contained HTML report (`scripts/vuln_eval_report.py`) with matplotlib charts embedded as base64 PNGs. Sections:

1. **Executive Summary** — aggregate scores, traffic-light recall indicator, scores-by-fixture bar chart
2. **Per-Fixture Cards** — CWE detection chart, tool sequence timeline (swim-lane), finding classification table, file coverage checklist
3. **Finding Classification** — confusion matrix, CWE detection heatmap across fixtures, severity accuracy, detailed FP/FN analysis with "was the file even read?"
4. **Agent Behavior Analysis** — workflow phase balance (discovery/analysis/reporting), annotation-before-reporting check
5. **Tool Usage Analytics** — tool category pie chart, per-fixture comparison, skills loading frequency
6. **Token Efficiency** — tokens per TP, fixture comparison radar (precision/recall/F1/coverage/efficiency)
7. **Auto-Generated Recommendations** — severity-tagged improvement cards based on eval results

---

## 7. Technical Artifacts

| File | Purpose |
|---|---|
| `cli/pipelines/trace_graph_pathpar.py` | Path-parallel trace pipeline |
| `cli/pipelines/vuln_assess.py` | Pipeline A: full vulnerability assessment |
| `cli/pipelines/vuln_scan_fast.py` | Pipeline B: high-recall fast scan |
| `contractor/tasks/vuln_scan_fast/v1.yml` | Separate task template for over-reporting scan |
| `contractor/tools/fs/merge.py` | Overlay fork/merge utilities |
| `contractor/skills/trace/references/cwe-mapping.md` | CWE quick-reference for agents |
| `scripts/run_vuln_eval.py` | Eval runner with metrics capture |
| `scripts/vuln_eval_report.py` | HTML report generator |
| `tests/eval/test_trace_parallel_eval.py` | A/B eval for parallel vs sequential |
| `tests/units/.../test_overlay_merge.py` | Unit tests for fork/merge |

---

## References

- [From Spark to Fire: Error Cascades in LLM Multi-Agent Collaboration](https://arxiv.org/html/2603.04474v1) — genealogy-graph governance, defense success 0.32 → 0.89
- [The Multi-Agent Trap](https://towardsdatascience.com/the-multi-agent-trap/) — keep chains under 5 steps, insert verification at 3 and 5
- [VulSolver: LLM-Driven Constraint Solving](https://arxiv.org/html/2509.00882v1) — decompose vuln detection into binary checks, 100% recall / 96% accuracy
- [Multi-role Consensus for Vulnerability Detection](https://arxiv.org/pdf/2403.14274) — +18% recall via role diversity
- [SAST + LLM Hybrid](https://conf.researchr.org/details/icst-2026/iteqs-2026-papers/1/) — F1 improvement 17-60% when combining
