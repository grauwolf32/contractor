# `shannon` workflow — design doc (not yet implemented)

> **Status:** design / proposal. No code in this folder yet — this documents how to
> port the **Shannon** white-box AI pentester (`KeygraphHQ/shannon`) onto contractor's
> workflow framework. Derived from the comparative analysis in
> `pentest-ai-agents/REPORT-agent-design-shannon-pentestgpt-vs-contractor.md` and a
> multi-agent feasibility pass over both codebases.

## TL;DR

A faithful, **source-centric** Shannon port is roughly **70% the existing `vuln-assess`
workflow**. Build it as a thin re-composition of vuln-assess's three sub-stages plus
**two new pieces** (a report-finalization stage and an auth-preflight stage), and
explicitly **drop Shannon's live-browser stages** (Playwright) as out-of-scope for
contractor's sandboxed, HTTP-only model.

---

## 1. Stage-by-stage mapping (Shannon → contractor)

Classes: **REUSE** (existing as-is) · **ADAPT** (existing + config/prompt change) ·
**BUILD** (new agent/skill/stage) · **DOES-NOT-FIT** (outside the source-centric model).

| # | Shannon stage | Contractor primitive | Class | Notes |
|---|---|---|---|---|
| 1 | Preflight validation | CLI flag wiring in `cli/main.py` + `WorkflowContext`; sandbox root check in `cli/fs.py` | ADAPT | No live-target reachability check (no mandatory live target). Add a config-parse guard. |
| 2 | Playwright stealth config | — | DOES-NOT-FIT | Anti-detection browser config is meaningless without a browser. |
| 3 | Authentication validation (live login) | `auth` skill + on-demand discovery in `exploitability_agent`/`web_exploitability_agent` | DOES-NOT-FIT (live Playwright) / ADAPT (HTTP auth fixtures) | Today auth is discovered ad-hoc per probe and stored in `auth/*` memory. A `auth_preflight_agent` (BUILD) closes this for HTTP APIs. |
| 4 | Deliverables git init | `TaskRunner(checkpoint_path=...)` + artifact store (`runners/artifacts.py`) | REUSE | Per-task artifacts (`{key}/result|summary|records`) + checkpoint resume replace the git-repo-of-deliverables. |
| 5 | SDK deny-rules sync | `RootedLocalFileSystem` sandbox (`cli/fs.py`) | REUSE | The rooted/virtual FS subsumes deny-rules. |
| 6 | Pre-recon (static code analysis) | `swe_agent` via `dependency_information` + `project_information` | REUSE | Exactly vuln-assess's OAS-stage prelude. |
| 7 | Recon (live crawl + endpoint enum) | `oas_builder_agent` + `oas_linter_agent` (source-derived OpenAPI) | ADAPT (static) / DOES-NOT-FIT (live crawl) | Attack surface derived from source, not by crawling a running target. Cannot find runtime-only undocumented endpoints. |
| 8/10/12/14/16 | Vuln analysis: injection / xss / auth / ssrf / authz (static) | `trace_agent` per-path + `trace` skill; breadth via `codereview_agent` + `vuln_scan` skill | REUSE | One generic tracer covers all 5 classes; the `trace` skill carries the sink/source/control/CWE taxonomy. Class-split is a prompt/namespace decision, not new agents. |
| 9/11/13/15/17 | Exploitation (live, per class, conditional) | `exploitability_agent` / `web_exploitability_agent` + `exploit` skill + `ExploitabilityWorkflow` | REUSE (HTTP) / DOES-NOT-FIT (browser/DOM) | HTTP PoC + verdict + evidence (`request_ids`) fully present, with code-exec sandbox + Caido. DOM-XSS/SPA/localStorage-JWT/visual oracles need a browser. |
| Gate | `checkExploitationQueue` (queue>0 → exploit) | `vuln_assess._run_exploit_stage`: `_collect_vuln_reports()` + `CONTRACTOR_TARGET_URL` guard | REUSE | Equivalent to Shannon's `shouldExploit` + `exploit_mode_enabled`. |
| 18 | External findings merge (SARIF/SAST) | seed-artifact bridge (`persist_seed_artifact`, `WorkflowContext(artifact=...)`) | ADAPT | Seed mechanism can inject findings YAML; no SARIF parser yet (small adapter). |
| 19 | Report assembly (concat per class) | artifact aggregation, analog to `_collect_vuln_reports()` | ADAPT | Merge logic exists; needs markdown assembly rather than YAML merge. |
| 20 | Report agent (exec summary) | `triage_agent` (orphaned) | ADAPT (reactivate) / partial BUILD | triage does dedup/rank/severity but not exec-summary authoring; needs a report prompt + task template. |
| 21 | Report metadata injection (models) | `cli/metrics.MetricsSink` + observability | REUSE | Model/usage already captured; injecting into a report is formatting. |
| 22 | Report output (CSV/HTML/SOAR hook) | — | BUILD-S / OPTIONAL | No `ReportOutputProvider` analog; a post-workflow emitter. |
| — | Resume / skip-gate (`completedAgents`) | per-task `artifact_exists()` skip + `checkpoint_path` | REUSE | Structurally identical to `shouldSkip`/`completedAgents`. |
| — | Concurrency (5 parallel pipelines) | `asyncio.TaskGroup` + `Semaphore(CFG.budgets.max_concurrency)` + `fork_overlay`/`merge_overlay_forks` | REUSE | `trace_graph_pathpar` already runs N concurrent per-path analyses with overlay forking. |

### How much `vuln-assess` already equals Shannon

vuln-assess already implements **stages 4, 6, 7 (static), 8–17 (static + HTTP), the
exploit gate, and resume** — the entire static-analysis → trace → conditional HTTP
exploitation spine:

- Shannon 6 (pre-recon) ≡ `dependency_information` + `project_information` (swe_agent).
- Shannon 7 (recon) ≡ `oas_update` + `oas_validate` (oas_builder + oas_linter).
- Shannon 8/10/12/14/16 (5 vuln agents) ≡ the trace stage (`TraceGraphPathParWorkflow`).
- Shannon gate ≡ `_run_exploit_stage` (`_collect_vuln_reports()` + `CONTRACTOR_TARGET_URL`).
- Shannon 9/11/13/15/17 (exploitation) ≡ `ExploitabilityWorkflow` per finding.
- Shannon 4 + resume ≡ `checkpoint_path` + `artifact_exists()` skipping.

It does **not** equal: live-browser stages (2, live-3, 7-crawl, DOM exploitation), the
upfront auth-preflight fixtures, SARIF ingestion (18), and the polished report stages
(19–22). **A faithful port = `vuln-assess` + a report stage + an auth-preflight stage,
with the browser stages dropped.**

## 2. Gap list

| Gap | Effort | Req/Opt | Detail |
|---|---|---|---|
| **Report-finalization agent/stage** (19/20) | M | REQUIRED | Reactivate `triage_agent` for dedup/rank, add a report prompt + assembly merging `user:vulnerability-reports/*` and `user:vulnerability-verifications/*` into one markdown doc. |
| **Auth-preflight agent + fixture artifact** (HTTP subset of 3) | M | REQUIRED | New `auth_preflight_agent` driving the `auth` skill: discover signup/login, create N role/tenant accounts, emit an `auth-fixtures` artifact the exploit agents consume. Needed for multi-user IDOR/role tests. |
| Per-class vuln-analysis namespacing/prompts (8/10/12/14/16 as 5 stages) | S | OPTIONAL | Generic `trace_agent` already covers all classes; fan out 5 namespaced trace tasks for cosmetic fidelity. Quality-neutral. |
| Per-class exploit specialists | L | OPTIONAL | Generic `exploitability_agent` + class refs already cover these. |
| SARIF/SAST ingestion (18) | S | OPTIONAL | SARIF→`vulnerability-reports` adapter feeding the seed bridge. |
| Report output emitter (22) | S | OPTIONAL | Post-workflow CSV/JSON/HTML formatter over the final markdown. |
| Live endpoint crawl (7 dynamic) | L | OPTIONAL | Outside source-centric model; partly mitigated by `http_agent`. |
| **Browser / Playwright** (2, live-3, 7-crawl, DOM exploitation) | L | DOES-NOT-FIT (drop) | DOM-XSS, SPA, localStorage-JWT, visual oracles. Declare out of scope. |
| Config-parse / target preflight guard (1) | S | OPTIONAL | Early config/credential validation in `cli/main.py`. |

---

## 3. Implementation blueprint

### (a) Feasibility verdict

The Shannon port is ~**70% existing `vuln-assess`** and is best implemented as a thin
re-composition of its three sub-stages. `vuln-assess` already chains discovery
(`swe_agent`), source-derived attack-surface inventory (`oas_builder`/`oas_linter`),
per-path parallel taint tracing (`TraceGraphPathParWorkflow`), the exploitation gate, and
conditional HTTP exploitation (`ExploitabilityWorkflow`), with checkpoint/`artifact_exists()`
resume. What **must be built**: a **report-finalization stage** (reactivate the orphaned
`triage_agent` — `build_triage_agent` exists but has *no task template* and is flagged
unused — plus a markdown-assembly step), and an **auth-preflight stage** (new
`auth_preflight_agent` + task minting role/tenant fixtures). What **cannot be faithfully
ported**: Shannon's entire live-browser surface (Playwright stealth, browser-validated
login, live crawl, DOM/SPA/localStorage-JWT exploitation) — incompatible with the
sandboxed, source-centric, HTTP-only model; explicitly dropped.

### (b) `workflow.py` skeleton

```python
"""Shannon-style end-to-end assessment (source-centric port).

Stage order:
  0. auth_preflight   — [Phase 3] mint auth fixtures from source (new agent)
  1. discovery        — swe_agent: deps + project structure   (reuse vuln_assess)
  2. recon            — oas_builder + oas_linter: attack surface (reuse)
  3. analysis         — per-path / per-class parallel trace    (pathpar primitive)
  4. exploit gate     — collect reports + CONTRACTOR_TARGET_URL  (reuse)
  5. exploitation     — ExploitabilityWorkflow per finding      (reuse)
  6. report           — triage_agent dedup/rank + markdown assembly (new)
"""
from __future__ import annotations

import asyncio
import logging
import os
from functools import partial
from typing import Any, Optional

import yaml

from contractor.agents.oas_builder_agent.agent import build_oas_builder_agent
from contractor.agents.oas_linter_agent.agent import build_oas_linter_agent
from contractor.agents.swe_agent.agent import build_swe_agent
from contractor.agents.trace_agent.agent import TraceFormat, build_trace_agent
from contractor.agents.triage_agent.agent import build_triage_agent
from contractor.runners.agent_runner import AgentRunner
from contractor.runners.models import (RenderedTask, TaskRunnerEventHandler,
                                       TaskTemplate)
from contractor.runners.plugins.metrics_plugin import AdkMetricsPlugin
from contractor.runners.plugins.trace_plugin import AdkTracePlugin
from contractor.runners.skills import inject_skills
from contractor.runners.task_runner import TaskRunner
from contractor.tools.code import attach_graph_tools_if_local
from contractor.tools.fs import MemoryOverlayFileSystem
from contractor.tools.fs.merge import fork_overlay, merge_overlay_forks
from contractor.utils.settings import build_model
from contractor.workflows import (Workflow, WorkflowContext,
                                   persist_seed_artifact)
from contractor.workflows.config import WorkflowConfig
from contractor.workflows.trace_annotation import extract_openapi_paths
from contractor.workflows.trace_graph_pathpar import TraceGraphPathParWorkflow

CFG = WorkflowConfig.load(__file__)
logger = logging.getLogger(__name__)

OAS_ARTIFACT = "user:oas-openapi-building"
TRACE_OAS_ARTIFACT = "oas-openapi-building"
REPORT_ARTIFACT = "user:shannon-report"


class ShannonWorkflow(Workflow):
    def __init__(self, ctx: WorkflowContext) -> None:
        super().__init__(ctx)
        self.llm = build_model(ctx.model, ctx.timeout)

    async def _run_impl(self, *, user_id: str,
                        on_event: Optional[TaskRunnerEventHandler]) -> Any:
        await self._run_auth_preflight(user_id=user_id, on_event=on_event)   # Phase 3
        await self._run_discovery_recon(user_id=user_id, on_event=on_event)  # Phase 1
        await self._bridge_oas_artifact(user_id=user_id)                     # Phase 1
        await self._run_analysis(user_id=user_id, on_event=on_event)         # Phase 1/2
        await self._run_exploit_stage(user_id=user_id, on_event=on_event)    # Phase 1
        await self._run_report_stage(user_id=user_id, on_event=on_event)     # Phase 1

    # ── Stage 0: auth preflight (Phase 3) ─────────────────────────────
    async def _run_auth_preflight(self, *, user_id, on_event) -> None:
        if not os.environ.get("CONTRACTOR_TARGET_URL"):
            await self.emit_task_skipped(on_event, "auth_preflight",
                                         reason="no live target")
            return
        if await self.artifact_exists(user_id=user_id,
                                      filename="auth_preflight/result"):
            await self.emit_task_skipped(on_event, "auth_preflight")
            return
        ctx = self.ctx
        runner = TaskRunner(name="contractor",
                            artifact_service=ctx.artifact_service,
                            checkpoint_path=ctx.checkpoint_path)
        runner.add_variable(name="project_path", value=ctx.folder_name)
        runner.add_variable(name="target_base_url",
                            value=os.environ["CONTRACTOR_TARGET_URL"])
        # NEW agent — see part (e)
        from contractor.agents.auth_preflight_agent.agent import \
            build_auth_preflight_agent
        builder = partial(build_auth_preflight_agent, name="auth_preflight_agent",
                          _format=CFG.agent("auth_preflight_agent").output_format,
                          fs=ctx.fs, model=self.llm,
                          max_tokens=CFG.budgets.auth_max_tokens)
        runner.add_task(name="auth_preflight", worker_builder=builder,
                        **CFG.tasks.auth_preflight.as_kwargs(),
                        namespace="auth-preflight",
                        skills=["auth"], model=self.llm)
        await runner.run(user_id=user_id, on_event=on_event)

    # ── Stage 1-2: discovery + recon (reuse vuln_assess shape) ────────
    async def _run_discovery_recon(self, *, user_id, on_event) -> None:
        ctx, fs = self.ctx, self.ctx.fs
        await persist_seed_artifact(ctx, filename=TRACE_OAS_ARTIFACT)
        if await self.artifact_exists(user_id=user_id, filename=OAS_ARTIFACT):
            await self.emit_task_skipped(on_event, "recon",
                                         reason="OAS artifact already exists")
            return
        runner = TaskRunner(name="contractor",
                            artifact_service=ctx.artifact_service,
                            checkpoint_path=ctx.checkpoint_path)
        swe = partial(build_swe_agent, name="swe_agent",
                      _format=CFG.agent("swe_agent").output_format, fs=fs,
                      model=self.llm, max_tokens=CFG.budgets.swe_max_tokens)
        oas = partial(build_oas_builder_agent, name="oas_builder",
                      _format=CFG.agent("oas_builder").output_format, fs=fs,
                      model=self.llm, max_tokens=CFG.budgets.builder_max_tokens)
        lint = partial(build_oas_linter_agent, name="oas_validator",
                       _format=CFG.agent("oas_validator").output_format, fs=fs,
                       model=self.llm, max_tokens=CFG.budgets.validator_max_tokens)
        runner.add_variable(name="project_path", value=ctx.folder_name)

        if not await self.artifact_exists(user_id=user_id,
                                          filename="dependency_information/result"):
            runner.add_task(name="dependency_information", worker_builder=swe,
                            **CFG.tasks.dependency_information.as_kwargs(),
                            namespace="dependency_information", model=self.llm)
        else:
            await self.emit_task_skipped(on_event, "dependency_information")

        if not await self.artifact_exists(user_id=user_id,
                                          filename="project_information/result"):
            runner.add_task(name="project_information", worker_builder=swe,
                            **CFG.tasks.project_information.as_kwargs(),
                            artifacts=["dependency_information/result"],
                            namespace="project_information", model=self.llm)
        else:
            await self.emit_task_skipped(on_event, "project_information")

        runner.add_task(name="oas_update", worker_builder=oas,
                        **CFG.tasks.oas_update.as_kwargs(),
                        artifacts=["dependency_information/result",
                                   "project_information/result"],
                        namespace="openapi-building", model=self.llm)
        runner.add_task(name="oas_validate", worker_builder=lint,   # [VERIFY]
                        **CFG.tasks.oas_validate.as_kwargs(),
                        artifacts=["dependency_information/result",
                                   "project_information/result",
                                   "oas_update/result"],
                        namespace="openapi-building", model=self.llm)
        await runner.run(user_id=user_id, on_event=on_event)

    async def _bridge_oas_artifact(self, *, user_id) -> None:
        # identical to vuln_assess._bridge_oas_artifact: copy
        # user:oas-openapi-building -> oas-openapi-building for the trace stage.
        ctx = self.ctx
        if await self.artifact_exists(user_id=user_id, filename=TRACE_OAS_ARTIFACT):
            return
        part = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name, user_id=user_id, filename=OAS_ARTIFACT)
        if part is None:
            logger.warning("No OAS artifact found after recon stage")
            return
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name, user_id=user_id,
            filename=TRACE_OAS_ARTIFACT, artifact=part)

    # ── Stage 3: analysis ─────────────────────────────────────────────
    async def _run_analysis(self, *, user_id, on_event) -> None:
        ctx = self.ctx
        if not await self.artifact_exists(user_id=user_id,
                                          filename=TRACE_OAS_ARTIFACT):
            logger.error("Cannot run analysis — no OAS artifact")
            return
        if CFG.budgets.per_class_analysis:                 # Phase 2 toggle
            await self._run_analysis_per_class(user_id=user_id, on_event=on_event)
        else:                                              # Phase 1: reuse as-is
            await TraceGraphPathParWorkflow(ctx)._run_impl(
                user_id=user_id, on_event=on_event)

    async def _run_analysis_per_class(self, *, user_id, on_event) -> None:
        """Phase 2: fan out one namespaced trace pass per vuln class, each a
        per-path parallel sweep — same fork_overlay/merge primitive as pathpar,
        nested under a per-class Semaphore so classes also run concurrently."""
        ctx = self.ctx
        raw = await ctx.artifact_service.load_artifact(
            app_name=ctx.app_name, user_id=user_id, filename=TRACE_OAS_ARTIFACT)
        openapi = yaml.safe_load(raw.text or "")
        paths = extract_openapi_paths(openapi=openapi)
        template = TaskTemplate.load("trace_annotation")

        overlayfs = MemoryOverlayFileSystem(fs=ctx.fs)
        shared_graph_tools = attach_graph_tools_if_local(overlayfs)
        pre_fork_patch = overlayfs.save()
        pre_fork_files = dict(overlayfs._files)

        classes = CFG.budgets.vuln_classes        # e.g. [injection,xss,authz,ssrf,auth]
        # one fork per (class, path) work item — isolated overlays merged at end
        work = [(c, p) for c in classes for p in paths]
        forks = [fork_overlay(ctx.fs, pre_fork_patch) for _ in work]
        sem = asyncio.Semaphore(CFG.budgets.max_concurrency)

        async def _run_item(vuln_class, api_path, overlay):
            async with sem:
                runner = AgentRunner(name=ctx.app_name,
                                     artifact_service=ctx.artifact_service)
                ns = f"shannon-trace:{vuln_class}:{api_path.path_key}"
                await inject_skills(["trace"], namespace=ns,
                                    artifact_service=ctx.artifact_service,
                                    app_name=ctx.app_name, user_id=user_id)
                for idx, op in enumerate(api_path.operations):
                    schema = yaml.safe_dump(
                        {op.path: {op.method: op.schema}}, sort_keys=False)
                    rendered = RenderedTask.from_template(
                        template=template,
                        variables={"project_path": ctx.folder_name,
                                   "operation_id": op.operation_id,
                                   "operation_schema": schema,
                                   "vuln_class": vuln_class},  # class-scoped prompt
                        params={}, artifacts={})
                    agent = build_trace_agent(
                        name="trace_agent", fs=overlay, namespace=ns,
                        _format="json", model=self.llm,
                        max_tokens=CFG.budgets.trace_max_tokens,
                        enable_vuln_reporting=True,
                        graph_tools=shared_graph_tools)
                    sid = __import__("uuid").uuid4().hex
                    ev = f"shannon:{vuln_class}:{op.operation_id}"
                    await runner.run(agent=agent, message=rendered._format_task(),
                                     user_id=user_id, session_id=sid,
                                     initial_state={},
                                     plugins=[AdkTracePlugin(task_name=ev, task_id=idx,
                                                iteration=1, session_id=sid,
                                                emit=runner._emit),
                                              AdkMetricsPlugin(task_name=ev, task_id=idx,
                                                iteration=1, session_id=sid,
                                                emit=runner._emit)],
                                     on_event=on_event, event_name=ev)

        async with asyncio.TaskGroup() as tg:
            for (vc, p), ovl in zip(work, forks):
                tg.create_task(_run_item(vc, p, ovl))

        conflicts = merge_overlay_forks(overlayfs, forks, pre_fork_files)
        if conflicts:
            logger.warning("merge produced %d conflicts: %s",
                           len(conflicts), conflicts)

    # ── Stage 4-5: gate + exploitation (handoff via seed artifact) ────
    async def _run_exploit_stage(self, *, user_id, on_event) -> None:
        if not os.environ.get("CONTRACTOR_TARGET_URL"):
            await self.emit_task_skipped(on_event, "exploit",
                                         reason="CONTRACTOR_TARGET_URL not set")
            return
        from contractor.workflows.exploitability import ExploitabilityWorkflow
        findings_yaml = await self._collect_vuln_reports(user_id=user_id)
        if not findings_yaml:
            await self.emit_task_skipped(on_event, "exploit", reason="no_findings")
            return
        # analysis -> exploitation handoff: collected reports become the seed
        # artifact of a fresh WorkflowContext, exactly as vuln_assess does.
        exploit_ctx = WorkflowContext(
            project_path=self.ctx.project_path, folder_name=self.ctx.folder_name,
            model=self.ctx.model, timeout=self.ctx.timeout,
            app_name=self.ctx.app_name, user_id=self.ctx.user_id,
            artifact_service=self.ctx.artifact_service, fs=self.ctx.fs,
            artifact=findings_yaml, checkpoint_path=self.ctx.checkpoint_path)
        await ExploitabilityWorkflow(exploit_ctx)._run_impl(
            user_id=user_id, on_event=on_event)

    async def _collect_vuln_reports(self, *, user_id) -> str:
        # Same merge logic as vuln_assess._collect_vuln_reports, but scan the
        # shannon-trace:{class}:{path_key} namespaces (or trace-annotation:* in
        # Phase 1). Returns merged YAML or "".
        ...

    # ── Stage 6: report finalization (new) ────────────────────────────
    async def _run_report_stage(self, *, user_id, on_event) -> None:
        ctx = self.ctx
        if await self.artifact_exists(user_id=user_id, filename=REPORT_ARTIFACT):
            await self.emit_task_skipped(on_event, "report")
            return
        # seed the triage agent's memory with all reports + verifications
        merged = await self._collect_vuln_reports(user_id=user_id)
        if not merged:
            await self.emit_task_skipped(on_event, "report", reason="no_findings")
            return
        await ctx.artifact_service.save_artifact(
            app_name=ctx.app_name, user_id=user_id,
            filename="shannon-findings-seed",
            artifact=__import__("google.genai.types", fromlist=["types"]
                                ).Part.from_text(text=merged))
        runner = TaskRunner(name="contractor",
                            artifact_service=ctx.artifact_service,
                            checkpoint_path=ctx.checkpoint_path)
        builder = partial(build_triage_agent, name="triage_agent",
                          _format=CFG.agent("triage_agent").output_format,
                          fs=ctx.fs, namespace="shannon-report",
                          model=self.llm, max_tokens=CFG.budgets.report_max_tokens)
        runner.add_variable(name="project_path", value=ctx.folder_name)
        runner.add_task(name="security_report", worker_builder=builder,   # NEW template
                        **CFG.tasks.security_report.as_kwargs(),
                        artifacts=["shannon-findings-seed"],
                        namespace="shannon-report", model=self.llm)
        await runner.run(user_id=user_id, on_event=on_event)
```

> Note: `build_triage_agent`'s signature takes `namespace=` as a keyword at build time
> (unlike the `swe`/`oas` factories), so the `partial` binds it directly rather than
> relying on `add_task(namespace=...)` to thread it into the builder; `add_task`'s
> `namespace` only scopes the runner's skills/artifacts.

### (c) `config.yaml`

```yaml
budgets:
  # token (summarization-trigger) budgets
  swe_max_tokens: 100000
  builder_max_tokens: 100000
  validator_max_tokens: 100000
  trace_max_tokens: 100000
  auth_max_tokens: 80000
  report_max_tokens: 80000
  # concurrency for the per-path / per-class fan-out
  max_concurrency: 3
  # Phase 2 toggle + class set (read in _run_analysis)
  per_class_analysis: false
  vuln_classes: [injection, xss, authz, ssrf, auth]

tasks:
  auth_preflight:        { iterations: 1, max_attempts: 2, max_steps: 25 }
  dependency_information:{ iterations: 1, max_attempts: 2, max_steps: 20 }
  project_information:   { iterations: 1, max_attempts: 2, max_steps: 20 }
  oas_update:            { iterations: 2, max_attempts: 4, max_steps: 20 }
  oas_validate:          { iterations: 1, max_attempts: 1, max_steps: 20 }
  security_report:       { iterations: 1, max_attempts: 2, max_steps: 25 }

agents:
  swe_agent:            { output_format: json, with_graph_tools: false }
  oas_builder:          { output_format: yaml }
  oas_validator:        { output_format: yaml }
  auth_preflight_agent: { output_format: json }
  triage_agent:         { output_format: markdown, with_graph_tools: true }
```

> `max_tokens`/`max_concurrency` for the inner `TraceGraphPathParWorkflow` come from
> *its own* sibling `config.yaml` in Phase 1 — the nested workflow loads its own `CFG`;
> the `shannon` budgets above only drive the Phase-2 in-line `_run_analysis_per_class`.

### (d) Registration + CLI alias

In `contractor/workflows/__init__.py`, inside `get_workflows()`:

```python
    from .shannon import ShannonWorkflow
```
and add to the returned dict:
```python
        "shannon": ShannonWorkflow,
```

No CLI change needed: `cli/main.py` builds `--workflow` choices from
`sorted(get_workflows().keys())` (≈ line 230), so `"shannon"` becomes a valid alias
automatically:

```bash
poetry run contractor --workflow shannon --project-path ./target --folder-name src --model lm-studio-qwen3.6
```

This folder's `__init__.py` re-exports: `from .workflow import ShannonWorkflow` /
`__all__ = ["ShannonWorkflow"]`.

### (e) New agents / skills / tasks

- **`contractor/agents/auth_preflight_agent/`** *(new agent)* — drives the `auth` skill
  against the live target to discover signup/login and mint N role/tenant accounts,
  emitting a structured `auth-fixtures` result the exploit agent consumes. Phase 3.
- **`contractor/tasks/auth_preflight.yml`** *(new task template)* — objective/instructions
  for upfront auth-fixture creation; vars `project_path`, `target_base_url`;
  `format: json`; `skills: [auth]`.
- **`contractor/tasks/security_report.yml`** *(new task template)* — the missing template
  for the reactivated `triage_agent`: dedup/rank/severity over the seeded findings and
  assemble one executive-summary markdown doc; `format: markdown`; consumes the
  `shannon-findings-seed` artifact.
- **`contractor/agents/triage_agent/prompts/v*.md`** *(new prompt version)* — add a
  report-finalization prompt variant; the current triage prompt does dedup/rank but not
  exec-summary authoring.
- *(Optional, Phase 2)* **per-class trace prompt variant** under
  `contractor/agents/trace_agent/prompts/` — a `{vuln_class?}`-scoped instruction so each
  fan-out pass focuses one class. Mind the ADK brace rules (`{var?}` / `<<TOKEN>>`); do
  not author a bare `{vuln_class}`.
- *(Optional, Phase 3)* **SARIF→reports adapter** — a small pure-Python function feeding
  `persist_seed_artifact` / `WorkflowContext(artifact=...)`; no new agent.

### (f) Phased build plan

**Phase 1 — minimal faithful port (reuse the vuln-assess spine).**
Create the folder + `config.yaml` + `__init__.py`; implement `_run_discovery_recon`,
`_bridge_oas_artifact`, `_run_analysis` (Phase-1 branch = call
`TraceGraphPathParWorkflow` unchanged), `_run_exploit_stage` + `_collect_vuln_reports`
(copied from `vuln_assess`, scanning `trace-annotation:*`), and `_run_report_stage` with
the new `security_report.yml` + reactivated `triage_agent`. Register `"shannon"`. Skip
`auth_preflight`. Delivers Shannon stages 4/6/7-static/8–17/gate/resume + a report stage.
Verify on an existing eval fixture.

**Phase 2 — per-vuln-class specialization.**
Add `per_class_analysis: true` + `vuln_classes`; implement `_run_analysis_per_class`
(in-line `TaskGroup`+`Semaphore`+`fork_overlay`/`merge_overlay_forks` fan-out over
`(class, path)` work items into `shannon-trace:{class}:{path_key}` namespaces). Add the
class-scoped trace prompt variant; update `_collect_vuln_reports` to scan the new
namespaces. Mirrors Shannon's 8/10/12/14/16 five-way split (cosmetic fidelity;
quality-neutral per the v5/v7 finding). Per-class exploit specialists optional/later.

**Phase 3 — dynamic-PoC gate + recon attack-surface deliverable + auth preflight.**
Build `auth_preflight_agent` + `auth_preflight.yml`, wire `_run_auth_preflight` as stage
0 (gated on `CONTRACTOR_TARGET_URL`), thread the `auth-fixtures` artifact into the exploit
context for multi-user IDOR/role tests. Harden the exploit gate with the `auth-fixtures`
precondition. Emit the recon attack surface as a first-class report section. Optional:
SARIF ingestion + a post-workflow CSV/HTML emitter. Live-browser stages remain out of
scope.

---

## Key files to copy from

- `contractor/workflows/vuln_assess/workflow.py` — the spine to copy.
- `contractor/workflows/trace_graph_pathpar/workflow.py` — the parallelism primitive.
- `contractor/workflows/exploitability/workflow.py` — the exploitation handoff target.
- `contractor/agents/triage_agent/agent.py` — orphaned; `build_triage_agent` has no task
  template yet (reactivate for the report stage).
- `contractor/workflows/__init__.py` — `get_workflows()` registration.
- `cli/main.py` (≈ line 230) — `--workflow` alias auto-derived from registry keys.
