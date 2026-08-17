<!-- Generated 2026-06-21 by multi-agent whole-project-review workflow: 22 subsystems, 66 agents, adversarial verification. -->

# Contractor Project Review — `llamacpp-serving` branch

## Executive Summary

The codebase is fundamentally sound and unusually well-documented: the runner core, subtask state machine, runner plugins, agent factories, and the filesystem-sandbox containment boundary are all carefully built and well-tested (1299 unit tests, all green on the branch). The defects that exist are concentrated, not systemic, and cluster around a few recurring themes: **silent failure modes in a security tool** (wrong matchers, swallowed errors, empty-substituted inputs that let a run "succeed" on garbage), **secret/egress handling on the live-target surface**, and **latent config-coupling that only holds because `APP_NAME` happens to be a constant today**. The single most important fix is the **credential leak in metrics_plugin.py**: HTTP bearer tokens, Authorization headers, basic-auth passwords, and `Set-Cookie` headers are written verbatim and unredacted to `metrics.jsonl` (and Langfuse) on every normal run — the http client redacts these in agent-facing views but the telemetry path bypasses that entirely. Close behind it are two correctness bugs that defeat the tool's core purpose: the **inverted Spectral/vacuum severity filter** (the linter keeps only info-level notes and drops every real schema error) and the **GitLab fs `fnmatch` matcher** that silently drops every top-level file from recursive globs. A recurring "diff vs apply disagree / two code paths that must match but were written separately" pattern shows up in the overlay save-vs-diff asymmetry, the gitlabfs in-memory-vs-fallback grep split, and the group-depth probe mismatch. Recommend prioritizing the secret-redaction fix, the OpenAPI linter/links bugs, and the overlay type-change crash before further RAG/serving work.

## Top Findings

### Critical / High

**1. HTTP auth tokens, Authorization headers, and Set-Cookie persisted in cleartext to metrics.jsonl** — `contractor/runners/plugins/metrics_plugin.py:401-411,451-497` — **high**
`before_tool_callback` emits raw `arguments` and `after_tool_callback` emits raw `result`; `cli/metrics._append_jsonl` `json.dump`s them with no redaction. `http_session_set(auth={...})` and `http_request(headers={Authorization:...})` put secrets in tool args, and the ResponseRecord includes `dict(response.headers)` (http.py:604) with `Set-Cookie`. The client's `_REDACTED` redaction only covers agent-facing session views, not telemetry. Live-target credentials land in `<project>/.contractor/metrics.jsonl` and Langfuse. **Fix:** centralize a redactor in `_emit`/`resolve_tool_args`/`resolve_tool_response` scrubbing keys matching `authorization|cookie|set-cookie|token|password|secret` (case-insensitive) from `arguments`, `result`, and response `headers`, so both metrics + trace plugins inherit it.

**2. Linter filters to severity-2 (info) and drops all real errors (severity 0)** — `contractor/tools/openapi/vacuum.py:239-302,305-355` — **high**
Vacuum/Spectral severity is inverted from the code's assumption: 0=error, 1=warn, 2=info, 3=hint (verified against vacuum 0.25.2). `lint_openapi` passes `include_severities=(2,)`, defaults are `(1,2)` "excluding severity 0", and the sort is `reverse=True` — so the linter returns only the least-serious tier, drops every schema-breaking error, and sorts least-serious first. The `oas_linter_agent` reports invalid specs as clean; the lint/repair loop is a no-op for the problems it exists to catch. **Fix:** `include_severities=(0,1)`, sort ascending (severity 0 first), correct the docstrings, and add a test asserting a known severity-0 error (e.g. missing `info.version`) appears in output.

**3. `read_file` returns ZERO content when a file's first line exceeds `fs_max_output`** — `contractor/tools/fs/format.py:133-178` — **high**
When line 0 overflows `max_output` (default 50KB), `cut_at_line=0`, `out_parts` stays empty, and the `emitted > 0` guard at line 161 suppresses the resume offset — the function returns only `"### truncated at line: 0 ### lines left in the file: 1 ###"`. Minified JS/JSON/packed CSS and one-line generated files become entirely unreadable with no recovery path. **Fix:** when zero lines fit, fall back to byte-level truncation — emit the first `max_output - footer_bytes` of the line and advertise a byte/char resume offset.

**4. `glob()`/`grep()` in gitlabfs use raw `fnmatch` — `**` recursive globs silently drop every top-level file** — `contractor/tools/fs/gitlabfs.py:694,775,1305,1382` — **high**
All four matching sites use stdlib `fnmatch.fnmatch` instead of the project's path-aware `glob_to_regex` (the rest of the fs subsystem deliberately migrated off `fnmatch`; read_tools.py:205 documents the fix). `fnmatch('README.md','**/*')==False` and `fnmatch('a.py','**/*.py')==False`, while `*` over-matches across `/`. The default grep path is `**/*`, so a bare grep silently skips every root-level file (Dockerfile, manage.py, root `*.yml`) — false-negative security findings with no error. The same default also yields **inconsistent results in-memory vs the API fallback** (medium #5, same root cause). **Fix:** replace `fnmatch.fnmatch(p,pat)` with `glob_to_regex(pat).match(p)` at all four sites (`_normalize_path` already strips the leading `/`).

**5. ~~`forced_tool_choice` ContextVar leaks from worker to planner~~ — REFUTED (FALSE POSITIVE)** — `contractor/callbacks/context.py` — ~~high~~ → **not a bug**
**Retracted after a dedicated empirical probe (2026-06-21).** The claimed leak does not occur. ADK's `Runner._run_node_async` wraps each agent's flow — including `before_model` callbacks — in `asyncio.create_task(_drive_root_node())` (`.venv/.../google/adk/runners.py:564`), which snapshots `contextvars.copy_context()`. The worker's `forced_tool_choice.set("none")` therefore mutates the *child task's* context copy, never the planner's. The planner resumes its own unmutated context (`None`) and `_apply_forced_params` injects nothing. Verified with the real project ContextVar and with a faithful `AgentTool.run_async` + fake-LLM repro (4 reproductions, all NO LEAK). The isolation is intentional and documented at `llm_compat.py:43-45` ("A ContextVar is copied per asyncio task, so each invocation sees only the value its own callback set"). The original static check missed it because the isolation lives in the ADK dependency, not in contractor code (hence "no `copy_context` in contractor" was a misleading signal). **Only actionable item:** an optional regression test pinning the ADK `create_task` assumption, so a future ADK upgrade that ran sub-agents inline would fail loudly instead of silently reintroducing the risk.

**6. `Response.links` typed `dict[str,str]` rejects valid OpenAPI Link Objects** — `contractor/tools/openapi/models.py:98` — **high**
OpenAPI 3.x `links` is `{linkName: Link Object | $ref}` (each a dict). `upsert_component` validates `responses` against this model (openapi.py:409-412), so any response component declaring `links` returns an error envelope and wastes agent retries on an unfixable validation error. **Fix:** `links: dict[str, Any] | None` (matching the sibling `headers`/`content`).

**7. `save()`/patch builder crashes on dir→file type change, silently drops file→dir change** — `contractor/tools/fs/overlay_patch.py:80-117` — **high**
`build_overlay_patch` lacks the type-change branch that `overlay_diff.render_overlay_diff` has (overlay_diff.py:147), so `save()` and `diff()` disagree. `rm <dir>` then `write_file <same path>` raises `RuntimeError("Type mismatch")`; `rm <file>` then `mkdir <same path>` silently emits `patches: []`. `save()` materializes the result artifact in all trace workflows and fork merge, so the first case crashes whole-task artifact production and the second diverges the diff from the applied patch. **Fix:** for any path in `base_paths & visible_paths` where types differ, emit a `delete_path` for the base entry then the normal `create_dir`/`write_file` — mirroring `render_overlay_diff`.

**8. `ls()` crashes (OSError) on any directory containing a symlink** — `cli/fs.py:122-149` (interacts with `_strip_protocol` 65-81) — **high**
The rooted `_strip_protocol` resolves symlinks via `realpath`; fsspec's `info()` sets `link=True` for a symlink DirEntry but then rewrites the path to the resolved target and calls `os.readlink()` on a non-link → `OSError [Errno 22]`. `ls()` always calls `super().ls(detail=True)` and only catches `FileNotFoundError`; `MemoryOverlayFileSystem.ls` (overlayfs.py:1106) also catches only `FileNotFoundError`, so this breaks directory listing for trace/likec4/vuln_scan whenever the tree contains a symlink (node_modules, vendored deps). **Fix:** override `info()`/build entries via `os.scandir`/`os.lstat` without re-resolving the entry, and add an `OSError` catch in `overlayfs.ls` as defense-in-depth.

**9. One malformed stored record poisons the entire vuln/verification store** — `contractor/tools/vuln.py:114-136` — **high**
`_load_artifact_records` calls `normalize()` per item with no try/except and no non-dict guard on the top-level YAML. One out-of-vocab Literal (a hand/externally-edited artifact, or schema drift in the user-scoped persistent store) raises `ValidationError`, and a non-dict top-level YAML raises `AttributeError` on `raw.items()`. Since every CRUD op (`write_report`/`get_report`/`list_reports`/`delete_report`) loads first, one bad row turns every subsequent call into an error envelope — including writing new valid reports and deleting the bad row itself. *(Verifier adjusted to medium: the in-band write path constructs the same frozen model, so the realistic ingress is edited artifacts / schema drift, not normal LLM writes — but the un-deletable poison row + loss of existing findings keeps it serious.)* **Fix:** wrap `normalize()` in try/except inside the loop (skip+log bad rows); add `if not isinstance(raw, dict): return {}`.

**10. HTTP request tool has no SSRF guard** — `contractor/tools/http.py:510-557,710-757` — **high**
`http_request` takes an arbitrary `url` with `follow_redirects` defaulting True at both the client (line 190) and per-call (718) level, and no scheme/host/IP validation anywhere. The URL is LLM-chosen over untrusted target source, so prompt-injected targets can reach `169.254.169.254` (cloud metadata), `127.0.0.1` host services (LiteLLM proxy, llama-server, Caido, pgvector), and RFC1918 hosts; redirects can pivot even an allowlisted target into them. Wired live into `http_agent` and `exploitability_agent`. **Fix:** resolve host, reject loopback/link-local/private/reserved by default with an opt-in target allowlist via Settings/workflow config, and re-validate every redirect hop (or disable redirects by default).

### Medium (grouped)

**Sandbox network/memory exposure (podman):** `--network host` + root with no `--cap-drop`/`--user`/`--read-only` (`contractor/tools/podman.py:123-133`) exposes all host-loopback backplane services to attacker-influenced exploit scripts wired via `with_code_exec: true`; and `_read_file` (`podman.py:168-171`) buffers the full container file into host RAM before applying the 1MB cap (post-read truncation → host OOM, no subprocess timeout). *Fix: bridged/dedicated network scoped to the target; cap inside the container via `head -c N` + add subprocess timeouts.*

**HTTP robustness:** no response-size cap (`http.py:536-557`) — full body materialized in memory, OOM/artifact-bloat from a hostile target (add `http_max_response_bytes` + stream/abort); retry layer re-sends non-idempotent POST/PUT/PATCH/DELETE on 5xx/timeout (`http.py:471-508`) — duplicate writes corrupt the proof chain (gate retries to idempotent methods).

**Latent artifact-scope / namespace coupling (workflows):** most workflows hardcode `TaskRunner(name="contractor")` instead of `ctx.app_name` (`vuln_scan`, `vuln_scan_fast`, `vuln_assess`, `trace_verify`, `trace_annotation`, `vuln_scan_trace`, `vuln_sweep`, `exploitability`) while `artifact_exists()` probes `ctx.app_name` — works only because `APP_NAME` is constant; `vuln_assess`/`trace_verify` hardcode group-key probe depths `(1,2)` that drift from the producer's configurable `group_depth` (`vuln_assess:299-304`, `trace_verify:112-114`). Both re-introduce the exact namespace-mismatch class the shared modules were built to prevent. *Fix: use `name=ctx.app_name` everywhere; derive `group_depth` from a single shared source or assert `<=2`.*

**OpenAPI authoring constraints:** `set_info` replaces the whole `info` block dropping required `version` (`openapi.py:546-547`, merge instead); `Operation.operationId` is required though spec-optional (`models.py:220`, make optional or document).

**Subtask completion edges:** `finish("done")` doesn't block on unresolved `incomplete`/`malformed` subtasks (`tools/tasks/tools.py:788-797`) — ships partial work as complete; and `finish("done")` is impossible for a legitimately all-skipped plan, forcing spurious failure + retry exhaustion (`795-797`).

**Code-tools correctness/perf:** tree-sitter row line resolved then re-indexed with `str.splitlines()` (`code/annotations.py:189-215`) mis-places annotations in files with form-feed/NEL separators (use `content.split("\n")`); `search_definition` buffers every needle-matching file's full text before any cap (`code/tools.py:864-879`) — OOM risk on common short symbols.

**Silent-degradation / observability (runners):** a missing/typo'd input artifact is substituted with `""` and the task proceeds (`task_runner.py:620-638`) — violates the artifacts-only invariant with only a log line; *fix: hard-fail when a queued sibling's `effective_artifact_key` is the missing ref, emit a `TaskRunnerEvent`.*

**GitLab fs reliability:** search-API grep fallback swallows all errors into `[]` (`gitlabfs.py:443-445,1344-1350`) — false "no matches" on 403/unsupported instances.

**RAG config/recall (latent, unwired):** `rag_embedding_dim` never wired into `PgVectorPoolBackend.dim` (`artifact_rag.py:80`) — silent config trap → Postgres dimension error; pgvector `search` post-filters masks after a fixed `k*4` over-fetch (`205-253`) — under-returns on fenced namespaces.

**Caido tech-debt:** persistent `httpx.AsyncClient` leaked with no teardown seam reachable from agent factories (`caido.py:331-341,1045-1062`) — the exact leak http.py was refactored to fix.

**Agent wiring:** `build_oas_builder_agent` advertises `with_graph_tools` but no OAS caller threads `CFG.agent('oas_builder').with_graph_tools` (`oas_builder_agent/agent.py:44,56`) — the knob is a silent no-op.

**fsspec compat:** `walk(self, path="", **kwargs)` override (`cli/fs.py:97-99`) breaks fsspec's positional-`maxdepth` `find()`/`du()` callers with TypeError, and drops `maxdepth` even when passed by keyword. *Fix: `def walk(self, path="", maxdepth=None, **kwargs)` and honor maxdepth.*

**Architecture / docs drift:** `contractor/runners/agio.py:21` imports `from cli.utils` (the lone reverse layer edge; a byte-identical helper exists in `contractor.utils`); the `trace_graph` workflow is a ~210-line verbatim copy of `trace_annotation_direct` differing only in two constants already config-driven (`trace_graph/workflow.py:46-226`, plus a `trace_graph_pathpar` sibling) — collapse to one base + config; `gitlabfs.py` (1508 lines) is fully dead **and** introduces a forbidden second `BaseSettings` config surface (`gitlabfs.py:55,73`) duplicating creds already in canonical Settings — delete or fold.

## Per-Subsystem Health

| Subsystem | Health | Note |
|---|---|---|
| FS: overlay/memory | minor-issues | Solid COW core; one high type-change save/diff asymmetry. |
| FS: gitlab backend | needs-attention | Structurally sound but lone `fnmatch` holdout → silent glob/grep truncation. |
| FS: read/write tools | minor-issues | Good core; high zero-content bug on long first lines; no binary detection. |
| Code: search/AST/graph | minor-issues | Well-tested; line-index mismatch on annotation + unbounded candidate buffering. |
| OpenAPI: build/lint/vacuum | needs-attention | Two high bugs: inverted severity filter + `links` type reject valid specs. |
| Tasks: state machine | minor-issues | Strict & well-defended; two completion-edge gaps in `finish()`. |
| Memory + artifact pool + RAG | minor-issues | Keyword path solid/tested; pgvector path dead + untested with latent bugs. |
| Vuln tool surface | minor-issues | Clean models; load path has no per-record isolation (store poisoning). |
| HTTP + Caido | needs-attention | Well-structured but unrestricted egress: SSRF, no size cap, TLS-off, client leak. |
| Podman + LikeC4 + observations | minor-issues | likec4/observations solid; podman `--network host` + pre-cap read are under-mitigated. |
| Runners: core | solid | Carefully built; one silent empty-input substitution edge. |
| Runner plugins | solid | Intricate but sound correlation; latent SSE over-count + unbounded payloads. |
| Callbacks | solid | Clean chain; the suspected `forced_tool_choice` worker→planner leak was empirically REFUTED (ADK `create_task` isolates per-invocation context). |
| Workflow assemblers | minor-issues | Follow the patterns; latent app_name + group_depth coupling. |
| Agent factories | solid | Uniform; one `with_graph_tools` wiring no-op. |
| CLI entrypoint + fs root | needs-attention | Entry/validation solid; `ls()` symlink crash + `walk()` signature break. |
| Utils: settings/llm_compat/obs | solid | Sanitizer + forced-tool-choice correct; only polish-level findings. |
| Branch correctness (xcut) | minor-issues | Production-path changes correct + tested; RAG half unwired/untested. |
| Security posture (xcut) | needs-attention | FS sandbox excellent; secret-in-logs + egress gaps are the real weaknesses. |
| Test suite (xcut) | solid | 1299 tests, high-signal new branch tests; main gap is untested pgvector RAG. |
| Architecture (xcut) | minor-issues | Coherent; concentrated copy-paste/dead-code/reverse-edge issues. |
| Deploy/build/docs (xcut) | minor-issues | serve.sh + litellm + pyproject sound; doc drift on tasks.py + workflow list. |

## Cross-Cutting Themes

- **Silent failure is the dominant risk class for a security tool.** Wrong matchers (`fnmatch`), swallowed errors (`grep`→`[]`), empty-substituted inputs (`_load_artifacts`→`""`), inverted severity (linter), and store poisoning all cause the tool to report success/clean while producing zero or garbage findings, with only a log line. Several are silent false-negatives in vulnerability detection — the worst possible failure for this product.
- **"Two code paths that must agree but were written separately."** Recurring root pattern: overlay `save()` vs `diff()` (type changes), gitlabfs in-memory grep vs API fallback, `vuln_assess`/`trace_verify` consumer probe depth vs producer `group_depth`. Wherever a writer and a reader/materializer are separate, they have drifted.
- **Security posture is bimodal.** The filesystem sandbox (realpath containment, symlink non-following, `..` rejection, overlay write-isolation) is genuinely excellent and verified escape-resistant. The outbound/secret surface is the opposite: cleartext secrets in telemetry, `--network host`, unrestricted SSRF, hard-disabled TLS. The boundary that was hardened is strong; the one pointed at untrusted targets is wide open.
- **Latent config coupling held only by constants.** `APP_NAME="contractor"` and `group_depth<=2` are the load-bearing accidents keeping the workflow artifact-scope and namespace probes correct — both documented as the very failure they'd reintroduce.
- **Test gaps track the unwired RAG work.** The pgvector backend ships with zero tests including two pure functions, and cross-component flows (callback→client `forced_tool_choice`, nested-`$defs` sanitizer) are only tested in isolation.
- **Duplication / dead code:** 1508-line dead `gitlabfs.py` (+ forbidden 2nd BaseSettings), the trace-direct workflow trio, duplicated `_xml_attr`/`utc_now_iso` helpers, and reverse `cli`-import edge.
- **Low/info findings (~30):** clustered in FS edge cases (offset clamping, EOF signaling, footer self-truncation), RAG perf/recall, doc drift (`tasks.py` path in CLAUDE.md/docs/README/insights.md, missing workflows, serve.sh help leak), and minor plugin/settings validation gaps — none blocking.

## Prioritized Action List

1. Redact secrets (`authorization|cookie|token|password|secret`) from tool args/results/headers before they reach `metrics.jsonl`/Langfuse — `metrics_plugin.py` + shared `_emit`.
2. Fix the OpenAPI linter severity mapping (`include_severities=(0,1)`, ascending sort) so real errors surface — `vacuum.py`.
3. Add an SSRF host policy + redirect re-validation to `http_request`, threaded as an opt-in target allowlist via Settings — `http.py`.
4. ~~Isolate `forced_tool_choice` across the worker→planner boundary~~ — **REFUTED** (see finding #5; no leak — ADK `create_task` already isolates it). Optional: add a regression test pinning that ADK assumption.
5. Handle type changes in `build_overlay_patch` (delete-then-create) so `save()` mirrors `diff()` — `overlay_patch.py`.
6. Fix `ls()` OSError on symlinked directories (override `info()`/scandir) + catch `OSError` in overlay ls — `cli/fs.py`, `overlayfs.py`.
7. Switch gitlabfs `fnmatch` → `glob_to_regex` at all four sites; distinguish search-unavailable from no-match — `gitlabfs.py`.
8. Byte-level truncation fallback in `format_output` when the first line overflows — `format.py`.
9. Change `Response.links` to `dict[str, Any]` and merge (not replace) `info` in `set_info` — `openapi/models.py`, `openapi.py`.
10. Isolate per-record failures in `_load_artifact_records` (try/except + non-dict guard) — `vuln.py`.
11. Replace `name="contractor"` with `name=ctx.app_name` in all TaskRunner workflows; derive consumer `group_depth` from one shared source — `workflows/*`.
12. Harden the podman sandbox (scoped network, cap-drop/non-root/read-only, in-container `head -c N` read cap, subprocess timeouts) — `podman.py`.