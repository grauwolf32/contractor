# Local workflow stability tuning

Status: primary DVWA workflow pair passed with all stages on first attempt;
LikeC4 repetition passed; both medium workflows passed with retries; large
Froxlor LikeC4 and DVWA Streamline passed without retries.
Do not treat a running or manually cancelled run as passing. OpenAPI schema validity and lint cleanliness
are separate outcomes.

## Handoff — 04:46 MSK

- Seven successful validation runs in the selected series: DVWA OpenAPI,
  DVWA LikeC4 twice, DVWA LikeC4 Streamline, crAPI OpenAPI + LikeC4, and Froxlor
  LikeC4. All published outputs were downloaded and independently validated.
  Medium crAPI required retries; do not call that a clean first-attempt pass.
- Final test policies are opt-in `local_project_worker@2` and, for Streamline,
  `local_project_planner@2`. Shared workflow defaults and old run snapshots
  remain unchanged. Select the explicit policies for comparable new runs.
- Worker limits: 8M cumulative tokens / 128 model calls / 512 tool calls;
  max response 32,768 tokens. Planner limits: 2M tokens / 96 model calls /
  64 worker calls. Sampling temperature 1.0, reasoning medium, top-p .95,
  top-k 20, min-p 0. Model: Qwen3.8 27B Q4_K_M, 262,144 context, Q8 K/V,
  Flash Attention, one inference slot; scheduler concurrency is one run.
- Backend: `telemetry-buffer-cancel-20260907`; all four runtimes:
  `runtime-likec4-diagnostics-20260907`. These isolated snapshots include only
  verified changes atop the prior demo release. UI was not changed; no DB
  migration was needed. The shared worktree contains unrelated changes and
  was not deployed wholesale. These stability changes are committed separately
  from unrelated ongoing work; machine-local demo settings remain local.
- Queue open (`paused=false`, revision 22), no active allocations, backend
  health OK; UI, LAN proxy and four runtimes active. Model reloaded from saved
  defaults without context/cache overrides and gateway smoke test passed.
- OpenAPI outputs pass schema validation, **not** the strict playground lint
  gate. Reports retain warnings and source-backed exceptions. LikeC4 checks
  validate syntax/semantics, not visual layout or exhaustive source coverage.
- Targeted runtime release regression suite: 129 passed; relevant Go telemetry
  and scheduler tests passed on the isolated backend source. No blanket claim
  about all tests in the concurrently modified main worktree is made.

## Baseline and environment

- Test Project: `project_4e3627571d8c5087b88819dc4e64122d` (DVWA).
- Exact source: `sources/source@rev_82d8df19ab9111c37395e355a0c1cf5e`.
- Source SHA-256: `6884fad74d1b445d46d0d692e4210e21fca6b7c37b0017012f1fc5bd8ea9a7a0`.
- Original failed LikeC4 run: `run_521bad70ecdd3eaecfbb99beddc3f08e`.
- Diagnostic replay: `run_8d84e1caf854dd63ada994dfb88dec70`.
  Reproduced `worker_budget_exhausted (total_tokens)` in dependency discovery.
  Cancelled deliberately before another identical-budget retry completed, to
  release the allocation and deploy the larger exporter buffer.
- Baseline worker trace: `82a20b151ce9a16a5c94174c88f1533c` in the local
  Langfuse project `contractor-debug-demo`. Captured model inputs grew from
  5,302 to 68,301 tokens across the first 15 retained model observations.
  Repeated source context contributes to the cumulative token budget.
- Run one workflow at a time; the local inference server has one generation
  slot. Avoid unbounded test/build concurrency on this 32 GiB host.

## Changes

- Planner and worker OTLP pending bytes: 2 MiB → 64 MiB **per exporter**.
  The 2,048-span cap and 256 KiB per-content-field cap remain finite.
- `debug` now resolves to `langfuse-debug@3`: capture content on both consumers,
  with a 10-second flush bound. Older runs retain their pinned configuration.
- New opt-in `local_project_worker@1`: 8,000,000 cumulative tokens,
  128 model calls, 512 tool calls, 32,768 maximum output tokens per response.
- New opt-in `local_project_planner@1`: 2,000,000 cumulative tokens,
  96 model calls, 64 worker calls, 8,192 maximum output tokens per response.
- Demo worker request timeout: 180 → 300 seconds. Stage/planner timeout remains
  30 minutes (current server validation maximum).
- Existing shared model policies and workflow definitions are unchanged.
- Fixed demo Runtime `PATH`: include `/home/ruslan/.npm-global/bin`, where
  LikeC4 is installed. Without it every idle Runtime advertised LikeC4 editing
  but omitted `validate_likec4`, so the build Stage waited in `preparing` with
  zero allocations. Only idle Runtime services were restarted; the same Run
  and its completed discovery stages were preserved.
  Registration recovered after the preceding leases expired (about 60 seconds).
  All four replacement runtimes now advertise `validate_likec4`, and the same
  build attempt entered `running` at 01:15 MSK.
- Fixed planner export after run cancellation: the flush request detaches the
  cancelled context while preserving its values and the minimum of configured
  flush, finalization and remaining stage-deadline bounds. An already-expired
  stage still does not start an export. This fix is tested and deployed.
- Explicit `worker_output_limit_exceeded` for a `MAX_TOKENS` finish in the
  worker or one-shot finalizer. Token usage remains accounted; truncated text
  is rejected even when it resembles valid final JSON. Runtime does not silently
  continue or accept a partial semantic result. Core model spans mark the limit
  failure instead of an ordinary success.
- Additional gateway hardening (rolled out at 01:57 MSK): a `length` finish
  suppresses all tool calls, including parseable partial calls, and preserves
  usage/error classification for malformed arguments or an empty response.
  Targeted gateway/runtime/finalizer suite: 77 passed.
  Release snapshot including instrumentation/OTLP: 87 passed. Runtime source is
  pinned to `.local/demo/releases/runtime-output-limit-20260907/runtime-src`;
  the backend release is unchanged.
- Local LiteLLM `worker-model` / `planner-model` use
  `extra_body.reasoning_effort: medium`. LiteLLM's supported-parameter lookup
  returned false for top-level `reasoning_effort` on this custom OpenAI model,
  so passing it via `extra_body` is necessary with `drop_params: true`.
  Their secondary routes now point to the loaded `qwen/qwen3.8-27b`, not the
  absent `qwen/qwen3.8-27b:2`. The unrelated strong-model alias was not changed.

The model policies were published through the public API and are also stored
in the operator catalog as reproducible YAML definitions. Runs explicitly
select these policies through `executionConfig.planner/workers.modelPolicy`.

## Deployment and verification

The current demo backend/runtime release is
`.local/demo/releases/telemetry-buffer-cancel-20260907` (rolled out at 01:40 MSK).
The original buffer-only release was `telemetry-buffer-20260907`.
It was built from the same
isolated V32-integrated source as the preceding backend, with the two buffer
constants patched. This avoids deploying unrelated ongoing work from the shared
working tree. Runtime units pin the release's Python source through `PYTHONPATH`.
No database migration was needed. UI was not changed by this rollout.
The newer runtime snapshot adds only the checked output-limit handling modules
to that stable baseline; unrelated main-worktree modules are not deployed.

- Go telemetry tests passed.
- Runtime OTLP/content tests: 21 passed, including retaining a late span after
  more than 2 MiB of captured model content.
- Go telemetry/config tests passed against the isolated deployed source.
- A broader main-worktree config test attempt encountered a concurrent,
  unrelated compile error in `findings_catalog_test.go` (`Ref.Name`).
- After rollout, server health and all four runtime registrations recovered.
- Expanded-buffer worker trace `2364914b199d1e55ff8cd0c39851e8f4` contains
  75 observations and 2,728,006 bytes of input/output content, including the late
  report write and finalizer. Dependency discovery succeeded on its first try:
  22 model calls and 919,802 tokens. This particular sample remained below the
  old token cap; repetition is needed to assess the larger budget's effect.
- Cancellation-flush tests pass in both main and isolated deployment sources.
- The complete isolated scheduler and telemetry test packages pass.
- Project discovery succeeded on attempt 1 with both subtasks successful;
  published `analysis/project@rev_ad4bba4442e74b6fcd0303f934ddabb6`
  (29,384 bytes). Its main subtask used 1,165,997 tokens, exceeding the former
  one-million-token cap. Across the allocation: 33 model calls, 186 tool calls,
  1,578,182 tokens, successful telemetry export with zero failed operations.
  Four recoverable reads requested lines beyond EOF; they did not fail the run.
- Two LikeC4 subtasks in the `xhigh` baseline each consumed exactly 32,768 output
  tokens in a single response and failed without a terminal result. The run was
  deliberately cancelled after reproduction to apply the reasoning adjustment.
  This baseline is not a full workflow pass.
- 76 runtime model/finalizer/instrumentation/gateway tests passed in main;
  84 including OTLP passed with the deployed runtime snapshot.
- Verified `medium` at the actual LM Studio input via its model-input log stream:
  the template's default `xhigh` instruction was absent. The small controlled
  probe returned `7`, finish `stop`, 17 input / 38 output tokens. The log stream
  was stopped before resuming workflow tests.
- The owner queue was temporarily paused during rollout and restored to
  `paused=false`, revision 8. Scheduler concurrency remains 1.
- With `medium`, OpenAPI dependency discovery passed on attempt 1. Worker trace
  `4e7c438ae128c3dd2e78885a238f4044` includes the finalizer: 424 input / 667 output
  tokens, 5.8 seconds, versus the excessively long finalizer observed with xhigh.
  This is a single sample, not a controlled performance comparison. Project
  discovery is in progress and has exceeded the old cumulative token cap.
- OpenAPI discovery and build all passed on attempt 1. Project discovery used
  1,476,819 tokens and published its report; its 1 failed `ls` was recovered.
  Build recovered from 4 rejected path payloads. Investigation subsequently
  found that 3 partial updates were legitimate under the documented recursive
  merge contract but rejected by premature pre-merge validation. Fixed paths
  and components to validate the merged candidate before CAS publication;
  invalid merged objects still leave the artifact and revision unchanged.
  All 24 OpenAPI toolset tests pass. Deployed at 02:09 MSK in
  `.local/demo/releases/runtime-openapi-merge-20260907/runtime-src`;
  full selected release snapshot suite: 111 passed.
- Rollout incident at 01:57 MSK: the operator (this session) resumed the queue
  before replacement runtimes had registered. Stale live leases allowed
  allocation attempts to the just-stopped endpoints; `connection refused`
  immediately exhausted two validation-stage retries and the queued LikeC4's
  two initial retries. This is not a model failure or clean workflow pass.
  Paused again, waited for all four replacement instance IDs to be `idle` with
  fresh accepted heartbeats, then resumed the same runs through the public
  resume API at 01:59 MSK. OpenAPI retained all three completed stages.
  Queue restored to `paused=false`, revision 12. Future rollouts must gate
  unpause on replacement registrations, not just systemd process activity.
- Resumed OpenAPI succeeded at 02:01 MSK, emitting 21,364-byte OpenAPI 3.0.3
  with 11 paths / 17 operations and a 5,226-byte validation report. Independent
  `openapi-spec-validator` validation passed. Strict playground Vacuum gate
  remains **failed** on 12 severity-1 style/example warnings (case conventions,
  trailing slashes, missing examples), with zero severity-0 findings. Do not
  rename source-defined API fields/paths to silence these. The workflow report
  explicitly says NOT clean and documents each remaining warning. This is a
  successful resumed workflow and schema-valid output, not a clean repeat or
  a zero-warning lint pass.
- `medium` with temperature 0.1 still failed on a LikeC4 dependency-discovery
  response: trace `3a15a6e448351d0674202db96285e4e0`, observation
  `8cad1dd4ea8e9a81`, 22,186 output tokens / 204.8 seconds, only thought content
  with extensive repetition and no tool call or final answer. No MAX_TOKENS
  marker was present; `worker_result_missing` is accurate in this case.
- Next sampling variant: `local_project_worker@2` and
  `local_project_planner@2`, same budgets, temperature 1.0. Gateway retains
  medium and now explicitly supplies top_p 0.95, top_k 20, min_p 0.0.
  These match the [official Qwen thinking sampling recommendation](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices).
  The loaded llama-server template confirms xhigh/medium/low are supported;
  medium adds no extra reasoning-depth instruction. Low is not enabled yet.
  The temperature-0.1 OpenAPI repeat was deliberately cancelled to use the new
  variant. No success is claimed for that incomplete repeat.
- The 02:09 rollout waited for four replacement runtime IDs with fresh accepted
  heartbeats and idle slots plus gateway liveness before unpausing revision 14.
  No active allocation was restarted in this rollout.
- Verified the actual llama-server active slot sampling on the new run:
  temperature 1.0, top_p ≈0.95, top_k 20, min_p 0.0, presence_penalty 0.0.
  OpenAPI dependency discovery passed on its first attempt at 02:14 MSK.
- Both v2-sampling dependency stages passed on first attempt: OpenAPI 807,110
  tokens / 23 model calls / 47 tools / 208.7 seconds, LikeC4 749,529 tokens /
  21 model calls / 51 tools / 179.6 seconds. Both reports published and OTLP
  flush succeeded with zero failed adapter operations. Project discovery next.

## Run ledger

| Run | Workflow | Configuration | Outcome |
| --- | --- | --- | --- |
| `run_8d84e1caf854dd63ada994dfb88dec70` | `likec4-from-workspace-streamline@2` | Original budgets; debug content on; 2 MiB queue | Budget failure reproduced; manually cancelled |
| `run_c191147af58bf01b8506429b53f58e2f` | `likec4-from-workspace-streamline@2` | Expanded local policies; debug@3; 64 MiB queue; xhigh | Discovery passed; response-length failures in build; manually cancelled |
| `run_eab2e9ddee0befc8c59c79ba65517e87` | `openapi-from-workspace@5` | Expanded worker policy; debug@3; 64 MiB queue; medium | Succeeded after rollout-incident resume; schema valid, 12 lint warnings |
| `run_b4b316a793009fc30e2c0cb6e644230b` | `likec4-from-workspace@5` | Expanded worker policy; debug@3; 64 MiB queue; medium, temp 0.1 | Failed: repetitive thought-only result |
| `run_2afdf585d3591f82795071d251f312d4` | `openapi-from-workspace@5` | Same medium configuration | Cancelled for sampling adjustment |
| `run_8cb309470fff89756039453b61c5d00f` | `likec4-from-workspace@5` | Local policies v2, medium, temp 1.0, OpenAPI merge fix | Succeeded 02:41; all 4 stages attempt 1; independent DSL validator passed |
| `run_3a09b16043c8897f246df49d001d43a6` | `openapi-from-workspace@5` | Same v2 configuration | Succeeded 02:40; all 4 stages attempt 1; independent schema validator passed |
| `run_001c9afe80705c95f7070001650c73b3` | `likec4-from-workspace@5` | Same v2 configuration, DVWA; 160k for final validation | Succeeded 03:18; all stages attempt 1; independent DSL validation passed |
| `run_0825ac057b9b2460fdea8dc54b6c7a13` | `openapi-from-workspace@5` | v2 policies, crapi-identity; context raised to 160k before build | Succeeded 03:19; project discovery retried at 128k; schema validation passed |
| `run_0ad8a9f964dcaaf042005db816003f26` | `likec4-from-workspace@5` | v2 policies, crapi-identity, 160k discovery / 262k Q8 build retry | Succeeded 04:09; discovery and build each retried; DSL validates |
| `run_557704f579d409e24c1d311eebed3157` | `likec4-from-workspace@5` | v2 policies, Froxlor 2.1.8, 262k Q8, diagnostic fix | Succeeded 04:26; all stages attempt 1; independent DSL validation passed |
| `run_c37c0789b18d43ae01453176c6a53d87` | `likec4-from-workspace-streamline@2` | v2 worker/planner policies, DVWA, 262k Q8 | Succeeded 04:44; all stages and 8 subtasks first attempt; independent DSL validation passed |

The successful v2 OpenAPI is 22,756 bytes, 11 paths / 17 operations; independent
schema validation passes. Vacuum has 0 severity-0 issues, 61 severity-1
style/example warnings and 33 severity-2 hints, so the strict playground lint
gate still fails. The output includes a 4,994-byte validation report.
LikeC4 is 15,013 bytes, independent CLI validation passes without diagnostics,
and its validation report is 2,722 bytes. The builder corrected invalid `**`
view predicates through its own validate/edit loop; no operator artifact edits
were made.

`crapi-identity` is packaged only from its declared `source` directory with
fixture exclusions: 121 files, 370,615 source bytes, 143,349 ZIP bytes,
SHA-256 `f1e55fa870780d5afa58b5e7002ccc8becc3aa559c5870e36437c9d1274795ab`.
Project: `project_815bacb3805def328398394c8c6ea7b1`. Hidden ground truth is not
included. The local helper initially used `@` in an idempotency key and received
400 before Run creation; normalizing the key resolved this harness-only error.
Its OpenAPI dependency discovery passed on attempt 1. The repeated DVWA
dependency discovery also passed on attempt 1.

Medium-project context boundary: crAPI project discovery attempt 1 failed with
`worker_output_limit_exceeded`; trace `cc6c9a55aead4585da45d4e446c21502`, model
observation `1c1a2bd7d1a803b3`, 119,883 input + 8,114 output = 127,997 tokens.
The provider's 128k context, not the 32,768 configured response cap or 8M
cumulative budget, left insufficient output room. The limit span is correctly
marked failed and the truncated tool call was not executed. Attempt 2 reached
125,839 input tokens, after which the active llama-server prompt dropped to
69,774 tokens (consistent with provider history truncation), and reached
finalizing. This is not a clean medium-project pass.

At 03:07 MSK, with owner queue paused (revision 15), all allocations gone and
LM Studio idle/queued=0, unloaded only `qwen/qwen3.8-27b` and began reloading the
same identifier at context length 160,000, GPU=max, parallel=1. Earlier actual
GPU usage was about 27.9/32.6 GB; KV cache was f16. The estimator reported LOW
confidence, so actual post-load memory and a probe must pass before unpausing.
Runtime/backend services are unchanged by this model reload.
Reload completed successfully. Actual GPU usage was 29,763/32,607 MiB;
LM Studio reports 160,000 context and one parallel slot. A gateway `worker-model`
probe returned HTTP 200, answer `7`, finish `stop`, 17 input / 39 output tokens.
Queue resumed (`paused=false`, revision 16). New llama-server PID is 2762598;
process argument dumps remain forbidden because they contain its API key.

Prepared, but not yet submitted, a larger source from the same playground:
`cvebench-cve-2024-34070` / Froxlor 2.1.8. The corpus fetch selected only
`src/critical/challenges/CVE-2024-34070/target/froxlor-2.1.8.tar.gz` at corpus
revision `4ed2d80ba5752ee958d624fa581242744e8a8a5d`, with the existing bounded
extractor and declared exclusions. Result: 1,928 files, 11,933,845 source bytes,
3,596,111 ZIP bytes, SHA-256
`d84b9cc3ff235c698f6d3eef3954219d3e3e144330793f33afaa17066b47cb98`.
No vulnerable application, exploit, target container or installer was run.
Submitted LikeC4 for that exact source at 03:21 MSK in Project
`project_25deec47c6f5bad8062f3f0c44d96aed` after the medium OpenAPI finished.

Repeated DVWA LikeC4: 16,240-byte DSL, zero independent validation errors,
2,008-byte validation report. All stages passed on first attempt. crAPI OpenAPI:
57,298 bytes, 30 paths / 32 operations, independent schema validation passed;
Vacuum severity counts 0:0, 1:242, 2:5. The strict lint gate still fails;
codes include example/description omissions, naming conventions and
`operation-success-response`, so do not reduce all findings to cosmetic style.
Its validation report is 5,184 bytes. The 128k discovery retry remains recorded.
Spot-checked the two no-success-response findings directly in the fixture:
`UserServiceImpl.loginWithEmailToken` returns only 400/403, and the user-facing
`ProfileController.deleteVideo` returns only 403/404. The report correctly
explains these source-backed exceptions; adding fabricated 2xx would be wrong.

The medium LikeC4 project discovery also exhausted available response room at
160k (`stage_execution_067b2f777c02afac90e3bcc738f33a55`,
`worker_output_limit_exceeded`). Attempt 2 passed; build is running. A larger
cumulative budget alone does not solve this context-window boundary.
Queue paused at revision 17 to drain before a possible Q8 KV-cache / native
262,144-context load. The official LM Studio SDK exposes independent K/V cache
quantization; this is a provider experiment, not a Contractor rolling-memory
feature. No active model was interrupted.

At 03:46 the medium LikeC4 build attempt 1 exhausted 8M cumulative tokens:
73 model calls, 7,935,643 input / 70,117 output tokens, 28 validations,
25 writes. Many later writes were small syntax experiments in a separate
`likec4/test` artifact, not repeated overwrites of the main architecture.
Several experiments validated, but the main diagram was still invalid and
was correctly not published as the workflow output. OTLP flush succeeded.

At 03:47, after draining allocations and checking LM Studio idle, reloaded
the same model through its SDK with context 262,144, Flash Attention, GPU ratio
1.0 and Q8_0 for both K and V. Actual llama-server flags and `/slots` confirm
Q8_0 and one 262,144-token slot. GPU usage: 30,309 / 32,607 MiB. A gateway probe
returned HTTP 200, answer `7`, finish `stop`. The loaded weights and sampling
are unchanged. Queue resumed at revision 18. This is an experiment pending
full workflow validation, not yet a proven cure for the DSL repair loop.
Provider API reference: https://lmstudio.ai/docs/typescript/api-reference/llm-load-model-config

Additional LikeC4 diagnostic fix: the CLI can emit hundreds of parser token
alternatives before its `but found` suffix. Prefix-only clipping removed that
actionable suffix. Bounded text now preserves head and tail, with an explicit
middle truncation marker included in the 4 KiB UTF-8 limit. Short messages are
unchanged; validation remains fail-closed. Added ASCII/multibyte boundary tests
and a full validator-path regression test. LikeC4 suite: 20 passed; release
snapshot regression selection: 129 passed. Snapshot:
`.local/demo/releases/runtime-likec4-diagnostics-20260907/runtime-src`.
At 03:54 queue paused (revision 19), idle runtimes a/b/d restarted onto it;
runtime c remains on the previous source while its allocation is active.
At 04:06 that allocation ended successfully and runtime c was restarted onto
the same diagnostic-fix release. Queue stays paused pending its new heartbeat.
Medium LikeC4 build attempt 2 succeeded at native Q8 context; the main artifact
`rev_11062a3ffea431a203274bb33f30f973` is 17,792 bytes and independently validates
with zero issues. The agent repaired the missing nested-element braces and
deployment/view syntax itself. Full run succeeded at 04:09:12. Independent
validation of the published output passed (17,792 bytes, zero issues), and the
validation report is 2,422 bytes. Build retry used 4,785,290 tokens / 54 model
calls / 1,038 seconds; OTLP flush succeeded with zero failed operations.
All four runtimes were freshly registered before queue resume at 04:08
(`paused=false`, revision 20). The Froxlor run started after medium completion.

Queued a fresh DVWA `likec4-from-workspace-streamline@2` run with both local
policies @2 and native Q8 context: `run_c37c0789b18d43ae01453176c6a53d87`.
The earlier clean DVWA results were passthrough workflows, not Streamline.

End-to-end buffer evidence on the long successful medium build retry:
Langfuse trace `530dd543a17c002a801e7b120ea904b8` retains all 54 model observations,
81 tool observations and the task span (136 total), with 8,122,449 bytes of
captured input/output content, including its late model calls. This exceeds
the former 2 MiB pending buffer substantially; export reports zero failures.

Froxlor finished at 04:26:28: all four stages succeeded on their first attempt.
Independent validation of the published 26,111-byte DSL passed with zero issues;
validation report is 2,425 bytes. Stage token totals / elapsed seconds:
dependency 591,145 / 176; project 1,304,387 / 353; build 1,170,666 / 458;
validation 128,301 / 47. The builder corrected syntax through its own tool loop.
No operator artifact edits and no application/exploit execution were involved.

LM Studio persistence check found its saved Qwen-specific load profile still
at 128k despite the live 262k/Q8 load. Updated only
`~/.lmstudio/.internal/user-concrete-model-default-config/qwen/qwen3.8-27b.json`:
262,144 context, parallel 1 retained, Flash Attention enabled, K/V cache
quantization `{checked: true, value: "q8_0"}`. Original small profile retained at
`.local/demo/qwen-default-before-tuning.json`. Field names/types were checked
against LM Studio's published config schema. This does not restart or change
an active inference. Other model profiles are unchanged. After all workflow
runs finished, paused the queue (revision 21), unloaded only this idle model,
and ran `lms load qwen/qwen3.8-27b --identifier qwen/qwen3.8-27b --yes` with no
context/cache/GPU/parallel overrides. Load completed successfully in 60 seconds;
actual llama-server flags confirm context 262,144, parallel 1 and both caches
Q8_0. Gateway probe returned HTTP 200 / answer `7` / finish `stop`.
Queue resumed at 04:46:22, revision 22. Model profile persistence is verified.

DVWA Streamline finished at 04:44:03. All four stages and all eight planned
subtasks succeeded without retry or operator resume. Its published DSL is
19,970 bytes, independent validation passed; report is 1,953 bytes. Build was
decomposed into specification, model, and views; per-worker history remained
bounded naturally between these tasks. This is a clean pass of the original
workflow variant under the new explicit worker/planner policies.

Follow-up candidates, not part of a completed stability claim: repeat both
medium workflows from scratch on the final native-context configuration;
evaluate source coverage and visual layout; address OpenAPI lint findings
without renaming real APIs or fabricating source behavior. This session only
packaged declared source roots and never used hidden evaluator ground truth.
