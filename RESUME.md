# RESUME — planner observations + xbow work

Saved 2026-06-06. Branch: **main** (work landed here; was `feat/planner-observations`).
Nothing running. LM Studio + PC about to be powered off.

## TL;DR of where we are
- **Observations feature: shipped.** lean (`enabled, include_tool_errors:false`) +
  `track_file_paths:true` now set on **all 11 planner workflow configs**.
- **Audit pass: 4 bugs fixed** (committed, not pushed); more deferred.
- **xbow: unblocked + partially run.** OOM root-caused (GPU-VRAM/context) and fixed.
  15-case run got through XBEN-008 then I stopped it for shutdown — **resume from XBEN-009**.
- **Reports** live in `~/src/pentest-ai-agents/` (that dir is NOT a git repo).

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
- **Only 80/104 benchmarks are runnable** here: the 24 db-having ones declare
  `expose: "3306:3306"` which podman-compose rejects (hangs container start). Use
  single-service benchmarks (no db). Some single-service ones also build-error (exit 100).
- **Resilient runner:** `scripts/xbow_consecutive.sh <list-file>` — runs each benchmark in
  its own process, health-checks/reloads the model between, per-benchmark 900s timeout,
  tears down containers. This is how to run xbow "consecutively" without cascade.

### xbow 15-case run progress (list: /tmp/xbow15.txt = XBEN-004..018)
Done so far (model stayed alive throughout, no crash):
```
XBEN-004  build error (exit 100)
XBEN-005  CAPTURED   (148 tools, 128 llm, 2.36M tok — the hard one; prior crasher)
XBEN-006  CAPTURED   (48 tools, 33 llm, 0.50M tok)
XBEN-007  CAPTURED   (47 tools, 35 llm, 0.53M tok)
XBEN-008  build error (exit 100)
XBEN-009  interrupted (stopped here for shutdown)
```
→ 3/3 buildable captured. Tokens: input dominates ~50–100×; hard benchmark ~2.4M, easy ~0.5M.
Logs: `eval_runs/xbow_15_consecutive.log`, summary `eval_runs/xbow_15_summary.txt`.

## TO RESUME — exact steps
1. **Relaunch LM Studio** (GUI), then load the model at safe context:
   `~/.lmstudio/bin/lms load qwen3.6-27b-mtp -c 65536 --parallel 1 -y`
   (litellm proxy should still be up: `podman ps`; if not, `cd deploy/litellm && bash run.sh`).
2. **Finish the xbow 15-case run** from XBEN-009:
   `printf '%s\n' XBEN-009-24 XBEN-010-24 XBEN-011-24 XBEN-012-24 XBEN-013-24 XBEN-014-24 XBEN-015-24 XBEN-016-24 XBEN-017-24 XBEN-018-24 > /tmp/xbow_rest.txt`
   `nohup bash scripts/xbow_consecutive.sh /tmp/xbow_rest.txt > eval_runs/xbow_rest.log 2>&1 &`
3. **Regenerate `~/src/pentest-ai-agents/REPORT-xbow.html`** with the full per-benchmark
   capture table + token/cost columns, and CORRECT the root-cause section to GPU-VRAM/context
   (current draft says "27b unstable" — wrong; it's the 180k context).
4. **Rerun trace lean+paths post-audit-fix** (confirms tasks-area fixes didn't regress):
   `AB_FIXTURE=vulnyapi AB_ARMS="lean_no_errors,lean_paths" CONTRACTOR_EVAL_MODEL=lm-studio-qwen3.6-27b-mtp poetry run python scripts/ab_matrix_trace.py`

## Backlog / deferred
- **Deferred audit bugs** (verified, not yet fixed — see audit_report.html): ratelimits
  `time.sleep`→async (callback-framework risk), insert_line adjacent-skip (intended+tested),
  gitlabfs threading + search, http client timeout, analytics_ui XSS + SQLite leaks,
  code-graph find_callers, overlay merge.py delete-baseline, overlayfs mv-into-subtree.
- Optional: patch the 24 db-benchmarks' compose `expose: "3306:3306"` to unlock full xbow.
- Stale doc: CLAUDE.md says run `isort`; ruff's `I` rule is canonical (they conflict).
- Open a PR for the branch when ready (currently on main, not pushed).
