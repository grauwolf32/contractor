# Evals experience: product direction

Status: **Draft for V38-001; product review in progress.**
No API/UI implementation or model experiment is delivered by this document.

Task: [V38-001](../tasks/v38-001-evals-experience-contract.yml).
Contracts: [US-10](spec/ui-user-stories.md#us-10--compare-variants-through-evals),
[portable format](spec/26-portable-evaluation-format.md),
[Projects and Queue](spec/17-projects-and-queue.md),
[Run labels](spec/16-run-metadata-labels.md).

## Problem established in code

The current `/evals` list shows evaluation Projects. Its detail reuses Project
sections and groups only the loaded page of ordinary Runs by `eval.id`.
`ui/src/routes/projects/evaluation-groups.ts` cannot know the planned denominator,
missing submissions, receipt-backed membership or scorer judgments.
`ui/src/routes/projects/runs-region.tsx` consequently presents execution history,
not a complete experiment comparison.

V41 supplies the portable plan, single-writer recovery journal, execution adapters,
assessment/comparison APIs and safe publication in `playground-v2`. It does not
supply a browser-facing evaluator service, suite catalog API or managed experiment
control API. Its current publication projection also omits some details needed by
the proposed UI: resolved variant summaries, approved execution navigation refs,
paired cost/latency summaries and their exclusion reasons. These need a deliberate
versioned presentation contract; the browser cannot infer them from Run labels.

## Recommended product choices

1. **One experiment is the primary Evals item.** An evaluation Project remains its
   storage/owner container and may contain several experiments. The experiment ID
   and frozen plan digest identify a run of the comparison. Existing evaluation
   Projects and unmatched labelled Runs remain accessible as legacy history.
2. **Start with an explicit A/B comparison.** A is the baseline; B is the candidate.
   Both use the same selected cases and repeat count. Show exact Workflow versions
   and applicable model/config differences before launch. An AgentTemplate is
   selected through its Workflow binding; an arbitrary template cannot run alone
   without an explicit executable wrapper. Selecting the latest available version
   in a new draft freezes that exact version at preparation, never follows `latest`
   during execution or recovery.
3. **Use a setup flow with four steps:** variants; cases and shared inputs;
   evaluation criteria and repeats; readiness and launch. Show the expanded count
   as `cases × 2 variants × repeats`, along with concurrency, time allowance and
   the user's token observation threshold. This UI work does not tighten existing
   numeric limits. Explain any unavailable model/tool/Skill pins before launch.
4. **Keep experiment detail focused:** Overview, Comparison, Attempts and Setup.
   Comparison is a side-by-side case/sample matrix with A/B outputs and evidence;
   Attempts supplies execution diagnostics and links. Setup is read-only after
   freezing; Duplicate creates a new draft and experiment identity.
5. **Separate execution, assessment and conclusion.** A completed Run is not a
   quality pass. Show unscored/error/partial states, the selected scorer or human
   review rubric, and evidence coverage. No implicit LLM judge or inferred winner.
6. **Base totals on the entire frozen experiment.** Filters and pagination change
   visible rows, not experiment denominators. Missing and failed attempts remain
   visible. Compare tokens/time only on pairs with complete comparable measures;
   show covered pair counts beside deltas.

## Screen structure

`/evals` should show experiment name, A/B variants, suite/case count, progress,
assessment coverage, conclusion, last update and one primary New experiment
button. Filters cover status, suite and evaluation workspace. An empty state
explains that a case set plus two variants is needed.

An experiment page could read:

```text
Trace instructions                   Interrupted · 7 / 8 attempts terminal
Baseline @6  /  Candidate @7          2 cases × 2 variants × 2 repeats

Overview    Comparison    Attempts    Setup

Quality passed       A 3/4          B 2/4
Scored               A 3/4          B 3/4
Token coverage       A 4/4          B 2/4
Conclusion           Insufficient evidence

Case / sample        Baseline       Candidate       Difference / evidence
unsafe-query / 1     Pass · 80      Pass · 70        −10 tokens; open pair
unsafe-query / 2     Pass · 100     Not submitted    Pending
safe-query / 1       Pass · 50      Fail · —         False positive; open pair
safe-query / 2       Run failed     Pass · 60        Failure remains in denominator
```

The displayed numbers illustrate the canonical trace example, not measured
instruction quality.
Use the same navigation, tables, status components and action treatment as existing
Project and Operations views. On narrow screens each case/sample becomes a paired
card preserving both variant labels, with details opening a dedicated page.

The default comparison emphasizes regressions and unresolved pairs. A toggle
reveals all pairs. Selecting a row opens exact result artifacts side by side;
returning preserves the experiment, tab, filters and cursor. Raw IDs/digests and
execution logs belong in details rather than the primary comparison columns.

## Launch ownership: product choice under review

**Recommended: setup, launch and comparison are available in the browser.**
That requires a managed evaluator service built around the existing Playground
planner/runner/journal/scorers. Contractor authenticates owner-scoped commands;
ordinary Contractor Run/Audit APIs, Scheduler and Runtime remain execution owners.
The evaluator keeps private cases, expected data, frozen plans and journals in its
own durable store. The browser receives bounded safe projections and command
receipts, not hidden truth, credentials or arbitrary server filesystem paths.
Closing the browser must not stop the experiment or lose its recovery state.

This is an explicit extension beyond V41's first CLI release. It must define
service availability, owner isolation, durable command identity, execution leases,
restart recovery and project deletion before the Start button is implemented.
It must reuse the existing runner, not copy its execution policy into a second
Go experiment engine or submit the whole matrix from browser JavaScript.

The smaller alternative is a report/comparison viewer for CLI-owned experiments.
It can consume safe publication without controlling execution; launch and recovery
stay in CLI. It does not fulfill the complete browser setup-and-launch journey.
The requested product preference is pending, so this draft does not yet settle the
managed-service/API contract or mark its implementation tasks ready.

## Worked example and invariant walkthrough

Reuse spec 26's `trace-small` fixture: two cases, two repeats and two variants,
eight expected members. A succeeds and passes three members; its fourth execution
fails. B succeeds three members, one of which fails quality with missing usage;
its fourth member was not submitted after interruption.

| Observation | Required presentation and recovery |
| --- | --- |
| Partial submission | A has 4/4 submitted; B has 3/4. Continue resumes the same frozen journal and submits only the outstanding member, subject to its original limits. |
| Lost create response | Retain an uncertain state; reconcile/replay the identical durable intent and key. No fresh member or sample is allocated. |
| Failed A execution | Remains in the expected denominator. Its known usage is retained even though quality is unscored. |
| Matching labels on an unrelated Run | Keep it in legacy/unassociated history. Labels do not prove membership or influence the comparison. |
| Two valid receipts claiming one member | Expose a conflict and suppress the affected comparison until explicitly resolved; never select the best result. |
| Missing B usage | Display unavailable, not zero. Only two pairs have complete comparable token data. |
| Scorer fails after a successful Run | Show assessment error separately; reassess retained output into a new immutable Assessment without rerunning the model. |
| Changed inputs/model/tools at resume | Stop with a pin mismatch and preserve prior attempts. A changed experiment starts with a new plan/identity. |
| Browser reload or evaluator restart | Restore the same command receipt and durable journal; no re-submission from labels or page state. |
| Stop requested | Stop dispatch and reconcile cancellation of owned work; show draining until remote terminal state is confirmed. |
| Report published but private bundle lost | Published results remain readable; execution recovery is unavailable. Safe reports cannot reconstruct the private journal. |
| Evidence deleted | Report unavailable evidence and incomplete comparison. A retained score does not recreate its supporting artifact. |

The full-experiment execution success is A 3/4, B 3/4; end-to-end quality pass is
A 3/4, B 2/4. Conditional scored quality is A 3/3, B 2/3 and must not replace those
denominators. B's observed 130 tokens cover only two expected members and do not
establish lower total cost. Overall conclusion is inconclusive.

## Implementation decomposition to finalize after the launch choice

The likely order for the recommended browser flow is:

1. Specify the managed evaluator and safe catalog/control/read contracts, including
   snapshot-bound pagination, example requests/responses and private-data boundaries.
2. Add durable evaluator commands and recovery around existing Playground APIs.
3. Add authenticated Contractor integration and bounded experiment read models;
   index a complete selected publication generation, not the first Run page.
4. Build the setup/readiness flow and explicit start/stop/resume actions.
5. Build comparison, assessment/evidence drilldown and legacy-history navigation.
6. Verify the eight-member example, failures, owner isolation, restart, deletion,
   large paginated experiments and desktop/mobile journeys using deterministic
   providers. Live instruction-quality measurement remains V40-003.

Before V38-001 can be completed, turn this outline into indexed, self-contained
API/evaluator/UI/verification tasks, settle launch ownership, specify exact API
examples and review all lifecycle and recovery states. Existing V40 live-readiness
gaps remain independently visible; this UI design does not make those bindings runnable.
