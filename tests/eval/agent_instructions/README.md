# Agent instruction experiment

Status: **candidate authored; compatibility checked; LLM comparison planned**.
No quality or token-efficiency improvement has been measured yet.

Integration direction for review: [portable playground-v2 evals](../../../docs/agent-evals-proposal.md).
Reuse the existing playground adapter, Workflow/Audit runners, datasets and scorers.
The matrix below is a gap checklist against those assets, not a second benchmark
implementation in Contractor. Contractor retains execution/API compatibility tests;
portable dataset, adapter and scorer changes belong in `playground-v2`.

The candidate ports evidence, coverage, and stopping rules from `contractor-old`
to the current tools. It adds 20 instruction variants, 15 AgentTemplate versions,
and 10 Workflow versions in `candidate/configs`. Existing `configs/` files are
unchanged. This is an opt-in evaluation overlay, not a published default catalog.
See [migration review](../../../docs/reviews/2026-09-06-agent-instructions.md).

## Reproducible variants

- `baseline.json`: exact pre-change bytes of all 27 current instructions and their
  SHA-256 digests. The recorded Git commit does not imply a clean working tree.
- `variants.json`: baseline/candidate paths, file digests, and explicit selectors.
- `candidate/configs`: loadable overlay containing new immutable versions.
- `overlay_test.go`: loads A and B through the real config loader and checks that
  capabilities, policies, scope, retries, stages, and result contracts match.

For the offline compatibility test, copy repository `configs/` to a temporary root,
then merge `candidate/configs/` into it. In A only, replace each candidate
instruction with its mapped baseline text. Thus A and B use the **same evaluation
selectors** and differ only in instruction text and derived digests. Never publish
these experiment catalogs to an existing operator/managed catalog. Keep each
Run's artifacts, workspace and sessions isolated. Same-selector substitution is an
offline control only. A shared live Server uses distinct immutable baseline and
candidate versions, never different bodies under the same selector.

The overlay includes `openapi-from-workspace@6`, `likec4-from-workspace@6`,
`likec4-from-workspace-streamline@3`, `taint-trace-from-workspace@3`, and
`security-analysis@3`. Archive workflows and Audit worker workflows also have
new versions; the full map is in `variants.json`. `http_explorer@2` remains a
reusable worker and needs a test-only wrapper. Existing AuditProfiles still select
their original workflows. Audit evals need isolated profiles bound to the candidate
workflow identities, with the same standard package and evidence contract.

Before a live experiment, pin actual source/configuration bytes, runtime build,
ADK/validator versions, resolved model and sampling settings, complete tool
declarations including docstrings, selected Skill revisions, task strings, input
revisions, budgets, and fixture/service state. `variants.json` pins the authoring
files; it does not substitute for capturing the resolved Run configuration.
Hold the newly standardized tool docstrings constant in both arms. Evaluate
docstrings separately later; do not attribute a combined change to instructions.

## Use the existing Server evaluation mechanism

The live experiment is a client of existing Server APIs, following
[Run metadata labels](../../../docs/spec/16-run-metadata-labels.md) and
[evaluation Projects](../../../docs/spec/17-projects-and-queue.md). It introduces
no execution engine, queue, Run store, or per-case Server process. The Server
already provides exact inputs, retries, cancellation, idempotency, retained
outputs and telemetry. Evals can show these Project/Runs immediately; the full
experiment setup/comparison UI remains V38 work.

1. Use one owner-scoped Project of `kind: evaluation` for the suite. Upload source
   fixtures and seeds through Project artifact APIs and pin exact revisions for
   both arms. Keep hidden expected answers in playground's evaluator-owned data;
   never upload ground truth to ProjectScope or supply it as Worker input.
2. Load both immutable config versions into the evaluation Server once: for
   example A uses `openapi-from-workspace@5`, B uses `@6`. Verify equal effective
   tools, model/sampling, Skills, budgets and input bytes; record the expected
   instruction/version differences. Never mutate an existing selector or select
   a prompt through metadata labels. Publication is a later explicit execution
   step; the current candidate overlay remains local.
3. Create ordinary Runs through `POST /v1/projects/{project_id}/runs`, with
   `purpose=eval`, `eval.name=agent-instructions`, one shared `eval.id`,
   `eval.leg=a|b`, and exact `eval.fixture`, `eval.case`, `eval.sample` strings.
   Each sample is a fresh Run; Stage retries belong to it. Labels remain grouping
   metadata and do not enter prompts or grant authority.
4. Persist a bounded experiment manifest as an ordinary Project JSON artifact
   using existing exact revisions/CAS. Record expected members, complete create
   request digests, stable idempotency keys, config/input pins and observed Run IDs.
   After a lost response or partial launch, replay the identical request with its
   original key. Labels are not unique identifiers: flag missing/duplicate members
   instead of adopting whichever matching row appears first.
5. Follow all pages of the existing owner/project Run list with exact label filters.
   Read attempts, metrics and outputs through ordinary Run APIs. Score each Run's
   **exact frozen output**; Project `outputs/<slot>` publication is create-only
   and cannot identify each sample's result. The driver submits and observes;
   the Server owns lifecycle, scheduling and retries. Reaching an experiment cap
   uses ordinary cancellation requests and records any unfinished work.
6. Retain the portable playground result envelope and publish scores/comparison
   as ordinary Project JSON/Markdown artifacts with Run IDs and exact evidence
   refs. Use common identities/digests for exported copies. Scores are explicit
   evaluator judgments, separate from technical Run outcome and metadata labels.

Reuse playground's `ContractorV2Client`, Workflow/Audit runners and existing
public APIs. V40-002 adds recoverable A/B and portable boundaries there; it does
not create another HTTP client or import Contractor's internal Go packages.
The production-process harness remains useful for CI and may start one isolated
test stack; an already configured Server does not need to be restarted per case.

Audit cases use the existing Audit API/controller and real importer with test
profiles. Never manufacture an Audit-managed Run through the generic endpoint.
If the current Audit contract cannot propagate eval labels to child Runs, retain
authoritative Audit/item/Run IDs in the experiment manifest and follow Audit
endpoints. Additional metadata propagation is an explicit follow-up capability,
not an assumed API or a prerequisite for scoring those Runs.

The minimum missing pieces are the manifest/recovery client, fixture scorers, and
paired comparison with missing/duplicate/failed handling. Existing attempt metrics
already expose input/output/total tokens, calls, `reportsComplete` and `truncated`;
respect those flags. Cached-token accounting is not currently in the public
MetricsSummary, so leave it unknown unless another pinned source supplies it.
Server-wide experiment aggregates and richer comparison UI can follow later;
the bounded pilot can read all pages and publish its report using existing APIs.

## What can run now

```sh
go test ./tests/eval/agent_instructions ./internal/config
go test ./tests/eval/project_workflows ./tests/eval/audit_programs -skip '^TestLiveProjectWorkflows$'
```

These are offline compatibility and scorer tests, not LLM behavior evals.
The existing `make test-project-workflows-live` is a useful integration foundation:
it starts the real Server/Runtime and uses real OpenAPI/LikeC4 validators. It
currently selects only workspace workflows `@5`, has no variant selector, and
persists its bounded evidence only on failure. It **does not run this A/B matrix**.
Its self-hosted test prerequisites are `CONTRACTOR_TEST_DATABASE_URL`,
`CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL`, `CONTRACTOR_WORKFLOWS_LIVE_MODEL`, an
optional gateway token, and installed `vacuum`/`likec4` executables. The planned
client must save successful and failed attempts through the existing Server APIs.

`tests/eval/audit_programs` verifies deterministic inventory/profile contracts;
it currently does not measure an LLM's Audit assessment quality. Runtime tests of
`submit_check_result` validate the tool contract, not the model's decision to call
it or the accuracy of its evidence.

## Planned cases and ground truth

All cases below are specifications for V40-001, not claims of implemented fixtures.
First map them to existing playground projects, suites and scorers; add fixtures
only for uncovered behavior. Keep small sources and closed fact/defect sets, and
expected answers outside the source archive and model context. Each case needs
exact source/service bytes, task inputs, allowed actions, expected facts/defects,
forbidden claims, evidence anchors, and a scorer with negative controls.

| ID | Fixture/scenario | Required observations and failure checks |
| --- | --- | --- |
| D1 | Runtime HTTP/database clients plus unused and dev-only packages | Correct roles and versions with source evidence; no invented external service from an unused dependency; missing lockfile recorded. |
| D2 | Nested router prefixes, middleware, models, misleading prior report | Correct method/path and effective controls; source wins over the report; no invented route from an outbound client. |
| O1 | Existing FastAPI fixture extended with nested routes, shared schemas, middleware auth | Expected operations/responses/security and resolvable refs; declared tags; no invented host; actual validator executes. |
| O2 | Exact seed with useful unrelated content, resume binding, rejected `$ref` mutation | Resume precedence, seed preservation, corrected mutation, no generic schema writer or identical rejected retry. |
| O3 | Missing validator and separately an unfixable structural issue | Explicit environment/unresolved result; no clean claim; bounded repair; durable report with correct revisions. |
| L1 | App, database, outbound service, helper library and unused queue dependency | Correct runtime units/relationships and source anchors; no phantom queue/service; declared kinds and valid phase order. |
| L2 | Seed with one broken reference plus unrelated valid views; unavailable CLI variant | Small evidence-backed repair, preserved content, no scope expansion, bounded validation, honest environment failure. |
| T1 | Attacker-controlled SQL structure or path, with a reachable sink | Correct entry-to-sink causal path and affected argument; finding plus evidence-backed annotations; no unrelated edits. |
| T2 | Bound SQL values and an effective per-resource ownership control | No injection/ownership false positive; distinguish authentication from ownership; cite effective control, not its name. |
| T3 | Misleading sanitizer, safe sibling and an unprotected alternate branch | Find the actual bypass; do not refute from the safe branch; include a non-taint control/invariant defect. |
| T4 | Duplicate symbol names, indirect dispatch, truncated graph, annotation replay/conflict | Correct symbol/definition; explicit unresolved coverage; no fabricated path or repeated annotation churn; inspect final diff. |
| H1 | Local HTTP service with session/CSRF ordering, baseline and controlled probe | Preserve observed dependencies, compare the relevant signal, distinguish rejection from transport failure, redact report. |
| H2 | Local stateful action commits but its response is lost; separate 4xx/5xx responses | Unknown outcome recorded/reconciled where possible; no duplicate mutation; no treating status alone as a vulnerability. |
| C1 | Controlled Caido traffic, passive false positive, empty workflow result | Correlate exact exchange, compare baseline, reject unsupported finding; empty scanner output is not proof of safety. |
| C2 | Caido replay/Automate polling timeout after remote start | Bounded payload set; use observable state/results where available; no duplicate job or assumption the job was cancelled. |
| A1 | Ordered Audit batch mixing supported, refuted and unresolved items | One complete ordered submission; truthful coverage/gaps and required evidence kinds; real importer outcome distinct from Run success. |
| A2 | Single risk mapping and ASVS requirement with misleading helper/middleware names | Assess only assigned scope; correct standard identity, causal finding reference, exact evidence revision; no certification or fabricated runtime test. |
| A3 | Unresolved operation mapping, unsupported callback, existing result binding conflict | No false completed coverage; no invented receipt or transparent retry claim; report incomplete/conflicting outcome under `audit-results@1`. |

Within representative cases, place instruction-like text in comments/reports and
assert it cannot expand the assigned task or authorization. Cover clean as well
as vulnerable sources; an empty finding set is a legitimate expected result.
Source-only cases do not send traffic. HTTP/Caido cases use controlled local
targets with explicit test authorization and deterministic reset between attempts.

Reuse the small fixture and production validation paths in
`tests/eval/project_workflows`. From the old repository, consider
`tests/eval/fixtures/fastapi/{trace-cases.json,oas.expected.yaml}` and VaultPay's
trace/expected-vulnerability fixtures after pinning their referenced source
revision and adapting paths/targets. The source may live in its playground
submodule; record availability and license rather than assuming it was copied.
Old regex-based finding scorers and legacy result envelopes are not ground truth
for current Markdown reports, annotations, or Audit ZIPs.

## Scoring

Score three independent dimensions; a successful Run is not itself a quality pass.

1. **Content:** expected fact/operation/flow coverage; finding precision and recall
   against the closed fixture set; correct evidence locations and causal links;
   correct safe/unresolved verdicts. A citation that exists but does not support
   the claim fails evidence correctness. Deduplicate findings by target, defect
   mechanism, and sink/control site. Label unadjudicated extra findings separately
   instead of silently counting them as correct.
2. **Contracts and behavior:** actual validator/importer acceptance, artifact
   durability, exact revisions, scope adherence, annotation diff, tool argument
   failures, identical retries without new evidence, mutation/job duplication,
   and bounded stopping. Check ordering only when the dependency or contract
   requires it; do not reward a fixed exploration sequence or matching prose.
3. **Cost:** model/tool calls, failures, input/output/cached tokens where available,
   total tokens, wall time, retries and truncations. Missing telemetry is unknown,
   not zero. Token prices, if used, need a pinned source and date; tokens alone
   are not a monetary estimate. Report full prompt/Skill/tool context, not only
   authored instruction character count.

Use parsed OpenAPI/LikeC4 structure plus real CLI outcomes, source evidence and
annotation diffs, and real Audit result decoding/import where applicable. For
free-form causal evidence, use a frozen rubric and blinded human adjudication;
an LLM judge is supplementary and must itself be calibrated on safe/unsafe and
unsupported-citation examples. Do not count keyword occurrence as a finding.

Report each attempt and paired case differences, including timeouts and failures.
Primary success rate is per attempt, with family-level results and denominators;
do not replace it with the old harness's optimistic pass@N. Separate infrastructure
failures and content failures while retaining both in the end-to-end denominator.
Rescheduled infrastructure attempts retain their original outcome. Compare p50/p90
cost/latency and state sample sizes; small pilots cannot establish significance.

## Execution and decision plan

1. **V40-001 — reuse and gaps:** map playground fixtures/scorers to the matrix and
   add missing negative controls. Gate on source/expected-answer review; adapt data,
   not old prompts or result schemas. Reserve renamed/restructured sources and
   alternate branches as held-out cases before tuning.
2. **V40-002 — playground integration:** extend its adapter/runner with a portable
   case/result boundary and recoverable A/B. Use an evaluation Project, versioned workflows,
   ordinary Runs, eval labels, idempotency and Project artifacts. Add manifest
   recovery, exact output scoring, complete paired comparison and bounded
   execution; reuse the production harness for integration tests. Include Audit
   controller/importer and resettable local HTTP/Caido fixtures. Deterministic
   transport doubles do not establish live Caido compatibility.
3. **V40-003 — pilot and decision:** first D1/O1/L1/T1/T2/A1, two repetitions per
   arm: **24 attempts**. Randomize or alternate A/B order within each case. Use the
   same existing model policy budgets; stop the experiment at a predeclared total
   cap (initial proposal: 2 million model tokens and 90 minutes). Reaching a cap
   leaves an incomplete experiment, not a pass. Freeze the candidate after fixes,
   then run all implemented cases and held-out variants with three repetitions
   per arm under a separately recorded budget.

Before running, freeze the gate: zero new unsupported clean/secure claims,
unauthorized actions, duplicate unknown-outcome mutations, or fabricated evidence;
no lost required fixture coverage; investigate every new false positive/negative.
For the pilot, flag more than +15% median total tokens or +20% p90 latency for
review rather than automatically claiming optimization. A larger prompt may earn
its cost through measured quality gains, but the decision must state that tradeoff.
Confirm any claimed win on held-out cases; ambiguous small samples mean inconclusive.

Evaluate worker text with fixed task inputs first; then full workflows to include
discovery/planner effects. If needed, use worker-only and planner-only ablations
to attribute regressions before another combined run. Choose improvements by
agent family; the candidate need not be accepted or rejected as one bundle.

Publishing selected versions and updating the default catalog/profile bindings is
a subsequent rollout decision supported by results, not part of the offline gate.
Do not mix the proposed V39 `audit-results@2` collector/finalizer with this prompt
experiment: current candidates still use `audit-results@1`. Evaluate the new
completion contract in a distinct experiment after V39 is implemented. V38 Evals
UI design may reuse the comparison record, but is not a prerequisite for this CLI
experiment; avoid building a second experiment service here.
