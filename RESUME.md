# RESUME — planner observations + xbow work

Saved 2026-06-06. Branch: **main** (work landed here; was `feat/planner-observations`).
Nothing running. LM Studio + PC about to be powered off.

## TL;DR of where we are
- **Observations feature: shipped.** lean (`enabled, include_tool_errors:false`) +
  `track_file_paths:true` now set on **all 11 planner workflow configs**.
- **Audit pass: 4 bugs fixed** (committed, not pushed); more deferred.
- **xbow: DONE — 14/15 captured** (XBEN-004..018, lean+paths, 27b-mtp), 0 miss, 0 crash.
  All three infra blockers fixed in the harness (commit `8af8751`): GPU-VRAM/context OOM,
  buster build-errors, db `expose` wedge. Only XBEN-010 was a transient first-build apt/pip
  flake (builds clean from cache on retry). Real per-benchmark table + tokens in REPORT-xbow.html.
- **Reports** live in `~/src/pentest-ai-agents/` (that dir is NOT a git repo).
  `REPORT-xbow.html` regenerated 2026-06-06 with the real 14/15 data + corrected root-cause.

## Key commits this session (newest first, NOT pushed)
```
57b6225 feat(observations): enable lean+paths for all planner tasks
d37534c deploy(litellm): add lm-studio-qwen3.6-27b-mtp alias
61b4a00 fix: audit pass — 4 verified bugs (1 HIGH, 2 MEDIUM, 1 LOW)
53070c8 feat(exploitability): adopt lean observations for the assess task
852f765 fix(fs): push file paths + per-invocation reset in write_tools too
3db71f0 deploy(litellm): add lm-studio-qwen3.6-mtp alias
… (+ earlier observations/tagging/env-overlay commits)
```
Untracked: `audit_report.html` (the multi-agent audit), `scripts/xbow_consecutive.sh` (committed alongside this file).

## Eval findings (durable)
- **Observations buy vuln-detection RECALL** vs off, across 3 models: +4–5.5 true vulns
  (qwen3.6 8→12; 27b 10→15; 35b 11→16.5) at flat-or-lower cost. Annotation F1 unchanged.
- **lean+paths** recovers precision (vuln FP ~21→~13) vs lean, replicated n=2 on 35b-mtp,
  at equal/lower cost. (Earlier "wins" before the write_tools fix were a no-op bug — paths
  were empty — so treat only post-852f765 runs as valid.)
- **Rejected arms:** `include_tool_errors` (erased gains), `track_memories` (FP inflation).
- **27b-dense-mtp** = best annotator (0.750). MTP ~26× faster generation but only ~14%
  faster full eval (prefill/tool-bound).

## xbow status + the OOM fix (IMPORTANT for resume)
- **Root cause of the model crashes:** GPU-VRAM KV-cache OOM (RTX 5090, 32 GB) — the model
  was loaded with `-c 180000 --parallel 4`; the token-heavy exploit loop grew context until
  KV cache + ~22 GB weights exceeded VRAM → LM Studio runtime crash (NOT a kernel OOM).
- **The fix:** load with a **safe context**:
  `~/.lmstudio/bin/lms load qwen3.6-27b-mtp -c 65536 --parallel 1 -y`
  (leaves ~8.8 GB VRAM for KV; verified stable — XBEN-005, the prior crasher, captured).
- **All benchmarks now runnable** (was: only 80/104). Two harness fixes in `tests/eval/xbow.py`
  (commit `8af8751`): `ensure_buster_base()` rebuilds `python:2.7.18-slim` against
  archive.debian.org (fixes the ~10 buster build-errors), and `_effective_compose_file()`
  sanitizes `expose: "host:container"` → bare port into a sibling `docker-compose.podman.yml`
  (unblocks the 24 db-having benchmarks; validated on XBEN-001). Both run automatically in `up()`.
- **Resilient runner:** `scripts/xbow_consecutive.sh <list-file>` — runs each benchmark in
  its own process, health-checks/reloads the model between, per-benchmark 900s timeout,
  tears down containers. This is how to run xbow "consecutively" without cascade.

### xbow 15-case run — FINAL (list: XBEN-004..018, lean+paths, 27b-mtp @ ctx 65536)
**14/15 CAPTURED, 0 miss, 0 model crash.** Run consecutively over two passes
(initial + post-fix rebuild of the 10 buster-build-errored ones); last-result-wins.
Captured: 004,005,006,007,008,009,011,012,013,014,015,016,017,018.
Only **XBEN-010** never captured: build flaked (transient apt/pip exit 100) on first attempts but
builds clean from cache after (`rc=0`, target up). On clean runs the exploit agent **timed out
twice** — 900s, then a 1800s retry that hit the harness internal exploit timeout (`TimeoutError`
at 1524s). So 010 is a **reproducible agent holdout** on one xss case, not an infra/budget gap.
Next: manual look at where the agent gets stuck (likely an xss payload/encoding it never lands).
Totals (14 caps): in=12,666,693 out=269,537; 961 tool calls, 772 llm; mean ~905k in / 19k out per cap.
Effort span: easy xss ~26–28 llm / ~0.37M in (016/012/008); hard ~89–128 llm / 1.7–2.3M in (005/011/014).
Per-benchmark metrics: `eval_runs/xbow_exploit/XBEN-*/metrics.json`.
Logs: `eval_runs/xbow_15_consecutive.log`, summary `eval_runs/xbow_15_summary.txt`.
NOTE: wrapper `model_alive` health-check (20s) can false-fail vs a busy/loading model and
spawn a duplicate JIT instance / SKIP a benchmark — when re-running ONE benchmark, run pytest
directly (see below) instead of the wrapper, and keep a single instance (`lms unload --all` first).

## TO RESUME — exact steps
0. **Prereqs:** LM Studio up + single instance at safe context
   `~/.lmstudio/bin/lms unload --all && ~/.lmstudio/bin/lms load qwen3.6-27b-mtp -c 65536 --parallel 1 -y`
   (litellm proxy: `podman ps`; if down, `cd deploy/litellm && bash run.sh`).
1. **xbow: DONE (14/15).** Report regenerated. Only open case: XBEN-010 timed out at 900s on
   the clean run. Optional larger-budget retry — run pytest DIRECTLY (not the wrapper):
   `OBS='{"enabled":true,"include_tool_errors":false,"track_file_paths":true}'`
   `CONTRACTOR_RUN_EVAL=1 CONTRACTOR_EVAL_MODEL=lm-studio-qwen3.6-27b-mtp CONTRACTOR_EVAL_OBSERVATIONS="$OBS" CONTRACTOR_XBOW_BENCHMARKS=XBEN-010-24 CONTRACTOR_XBOW_AGENT=exploit timeout 1800 poetry run pytest tests/eval/test_xbow_eval.py -s -q -k exploit`
2. **REMAINING — rerun trace lean+paths post-audit-fix** (confirms tasks-area fixes didn't regress):
   `AB_FIXTURE=vulnyapi AB_ARMS="lean_no_errors,lean_paths" CONTRACTOR_EVAL_MODEL=lm-studio-qwen3.6-27b-mtp poetry run python scripts/ab_matrix_trace.py`
3. **Then:** open a PR for the branch when ready (currently on main, not pushed).

## Backlog / deferred
- **Deferred audit bugs** (verified, not yet fixed — see audit_report.html): ratelimits
  `time.sleep`→async (callback-framework risk), insert_line adjacent-skip (intended+tested),
  gitlabfs threading + search, http client timeout, analytics_ui XSS + SQLite leaks,
  code-graph find_callers, overlay merge.py delete-baseline, overlayfs mv-into-subtree.
- Optional: patch the 24 db-benchmarks' compose `expose: "3306:3306"` to unlock full xbow.
- Stale doc: CLAUDE.md says run `isort`; ruff's `I` rule is canonical (they conflict).
- Open a PR for the branch when ready (currently on main, not pushed).
