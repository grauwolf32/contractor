# Contractor UI: user stories and improvement plan

Status: **V37 implemented and verified; separate follow-up scope remains**.
Updated: 2026-09-19.

This document records user goals and usability criteria. The
[UI specification](06-server-ui-and-operations.md) describes current behavior
and protocols. A story's presence here does not mean it is fully implemented.
Implementation status is tracked in the [tasks](../../tasks/index.yml).
The [V37 acceptance report](../plans/2026-09-19-v37-journey-verification.md)
records connected browser journeys and real Server/Runtime/PostgreSQL
verification. Fixture evidence is separate from real API evidence. No participant
usability study has been performed; US-10 remains outside V37.

## Users and scope

- **Analyst / researcher** prepares materials, starts analyses, reads results
  and makes Audit decisions.
- **Workflow / agent author** selects and checks published definitions, reads
  prompts and compares behavior variants.
- **Operator** monitors execution and resources, changes settings and resolves
  infrastructure issues that cause waits or failures.

These are roles in user scenarios, not new authorization roles. One person can
perform all three functions; Server continues to determine access.

The primary journey is **Project → materials → Workflow → Run setup → execution →
result → next Run or Audit review**. Starting a Run without a Project remains a
supported standalone scenario. Catalog groups Workflows, Agents and Skills;
Runtime Agent processes belong in Operations. No additional main navigation
items are needed; Evals remains after Artifacts.

## User stories

Identifiers US-01…US-11 correspond to UC-01…UC-11 from the usability study. They
remain stable as the interface changes and are used in tasks and acceptance checks.

### US-01 — Prepare a Project

As an analyst, I want to collect source code, the target and results in one Project
so that I can continue and repeat work on the same subject.

Done when Project sections provide actions to add materials and choose an
analysis, with clear section navigation; after an upload or Git import, the storage scope, exact revision and
next step are clear. Deletion is available through secondary actions with the
existing confirmation.

The Project root is a short Overview of decisions, recent results and activity.
Artifacts, Workflows, Runs, Audits, Findings and Settings are independent screens
with common navigation. The Run history has Server-backed state filters and
compact rows; upload and exact Project Run setup open on demand. Filters,
selected versions and return context survive the corresponding navigation.
See the [Project section plan](../plans/2026-09-project-section-navigation.md).

Project cards also expose a named Delete icon. Confirmation requires the
current Project name; pending deletion disables repeat submission and provides
a link to deletion status. A stale revision requires refreshed data and a new
confirmation.

Implementation tasks: V37-002, V37-003. Git import: existing V35-004.

### US-02 — Choose an analysis by its intended result

As a user, I want to find a Workflow by its purpose and expected result so that
I can choose an analysis without first learning about Stages and internal names.

Done when the catalog shows descriptions, required materials and outputs, search
covers all published entries, and the exact selected version is visible before
launch. Execution details are available separately. A missing description is not
replaced by a guess based on the name; the default version follows the explicit shared UI ordering described below.
Numeric dot-separated versions sort by integer components and default to the
highest version after the complete inventory loads. Non-numeric labels remain
selectable in deterministic natural order without a Latest claim. A deliberate
older selection and an open Run draft never change silently on refresh. This
policy supersedes the earlier version-order prohibition at the user’s request.

Implementation tasks: V37-003, V37-006, V37-007.

### US-03 — Configure and start a Run without losing input

As an analyst, I want to select suitable inputs, add a file if necessary and
return to unfinished setup so that I do not have to assemble the request again.

Done when local upload, the library and existing Git import are available from
the relevant input slot; navigating through the application and closing the form
preserve the draft in the current tab. Returning restores it for the same user,
Workflow/version and Project. Explicit reset requires confirmation when input is
present; successful launch completes the draft. Persistence across a tab reload
is not guaranteed: sensitive input is not automatically written to browser storage.

MIME-based suggestions are labeled as format matches and require user review.
The interface must not claim that a document's semantic suitability is proven.
The form and nested dialogs are fully usable with the keyboard.

Implementation tasks: V37-001, V37-002, V37-004.

### US-04 — Understand execution progress and waits

As a user, I want to see the current work and the reason for a wait so that I can
decide whether to intervene or wait for execution to continue.

Done when Run and Queue show the reason known to Server, the next available step
and a link to the relevant diagnostics. An already scheduled Scheduler retry,
a wait for Runtime capacity and the option to create another Run are distinct.
An unknown reason remains unknown; an idle Runtime is not presented as a
guarantee of compatible capacity.

Implementation tasks: V37-008, V37-011. Existing Home, Queue and Run triage remain.

### US-05 — Retrieve and use a result

As an analyst, I want to find the primary result immediately and open a specific
document so that I can assess it and continue working with it.

Done when primary outputs are highlighted using the published `primary` flag,
and a supported small file opens through one explicit action on that file.
Unsupported previews offer an explanation and Download; all outputs and exact
revisions remain available. A missing primary output is not hidden by selecting
an arbitrary file. Opening a Run does not require downloading every output.

Implementation task: V37-009.

### US-06 — Fix the cause and repeat an analysis

As a user, I want to open a new draft from a previous Run with the same inputs
and parameters so that I can repeat the work after fixing an error.

Done when the action prefills the available original settings, preserves the
Project context and prompts the user to review them before launch. Unavailable
versions, credentials or inputs are clearly marked. The old Run and its history
remain unchanged; repeating creates a new Run only after explicit submission.
For an Audit-managed Run, new execution goes through the controlling Audit,
without copying its internal labels into an ordinary user request.

Implementation tasks: V37-002, V37-008.

### US-07 — Conduct an Audit and make decisions

As a researcher, I want to see check progress, gaps and pending decisions so that
I can assess findings against their evidence and accept the report.

Done when the summary links to the corresponding complete filtered lists, and
pending review opens a specific finding or report with exact evidence in one
action. Returning preserves filters. Technical success, coverage completeness
and the researcher's decision are shown separately. Audit cancellation and
deletion require confirmation that explains the consequences; dismissing the
confirmation changes nothing.

The Audit draft loads all Project Artifact pages before preselecting a sole
media-compatible input. Ambiguous matches require an explicit choice; user
choices and cleared slots are preserved. A load failure offers retry and does
not imply uniqueness. The exact selection stays visible and editable before
draft creation and remains subject to Server validation.

Start and Continue open the time-limit dialog specified in
[19](19-audits.md#17-public-api-and-ui): seven days by default, 24 hours, a
custom duration or no time limit. Ordinary paused Audits default to keeping
their remaining allowance. The UI explains that queue waiting consumes time,
pauses preserve it, and reaching the limit pauses new Runs while existing work
finishes. Eligible legacy deadline closures offer Continue with accepted
results and previous reports retained. Dismissing time settings sends no
mutation; icon actions retain accessible names and explanatory tooltips.

Implementation tasks: V37-001, V37-005, V37-010.

### US-08 — Inspect an agent and its prompt

As a Workflow author, I want to open an exact agent version, read its base prompt,
Skills and tools, and see the Workflows that use it so that I can understand its
behavior.

Done when versions are selected from published definitions, search is not limited
to the current page, and usage links lead to exact Workflow versions. Existing
Preview / Source / Copy actions remain. The base prompt is clearly distinguished
from the actual context of an individual invocation. Comparing prompts across
versions is a later phase, outside V37.

Implementation tasks: V37-006, V37-007. Basic functionality is implemented in V33-001.

### US-09 — Find and inspect the right file

As an analyst, I want to add a file and see its purpose, revision and provenance
so that I can select the right material and reproduce the result.

Done when a file can be added directly during Run setup, inspected before binding
to a slot, and opened without unnecessary expansion steps. The exact revision
and Git commit, when present, are available; full history, lineage and diagnostics
do not obscure the primary actions. An update conflict offers a way to inspect
the current state or choose a new name and never results in a silent overwrite.

Implementation tasks: V37-002, V37-004, V37-009; existing V34 and V35 preserve the
storage contract.

### US-10 — Compare variants through Evals

As an agent author, I want to define variants, shared cases and a repeat count so
that I can compare results, duration and token usage using comparable data.

Target outcome: experiment setup and comparison by case/sample, with missing,
repeated and failed Runs shown explicitly. Missing metrics are not zero; a
successful Run alone does not imply high quality. Quality assessment requires
an explicitly selected evaluator or a human decision.

**Selected design, implementation pending:** V38-001 defines the
[full browser journey](../evals-experience-design.md): configure, prepare, start,
review and compare experiments. [Spec 29](29-managed-evals.md) keeps native
Contractor execution independent of Playground, which can import data and drive
external experiments through the same public API. V38-002 through V38-010 own
implementation and verification. Existing grouping of ordinary Runs by `eval.*`
labels remains supported; completing V37 or this design does not complete US-10.

### US-11 — Monitor and configure execution

As an operator, I want to see Runtime readiness, Server load and database state,
and understand the impact of settings before changing them, so that I can resolve
execution blockers and manage resources.

Done when the overview explains state and availability, forms open through
Create/Edit actions, and changes and their scope are visible before saving.
Publishing a RuntimeConfig is not presented as changing an allocation already
in progress. Revision/snapshot details are available in diagnostics. Permissions
and the disabled state of metrics reflect Server's state.

The readiness overview distinguishes observed idle/busy/reserved/draining/fenced
states; idle alone is not a promise of compatible capacity. Binding forms show
the current exact version beside the proposal. A stale revision blocks saving
until an explicit authoritative reload, which retains the proposed version for
review and retry. Scope copy distinguishes new Run snapshots from future
Agent-label allocation resolution; prepared allocations keep pinned settings.

In Runs Configuration, named icon actions open RuntimeConfig publication and
Runtime credential dialogs. Bindings open for the exact selected version and
show whether bindings already exist. Closing a credential dialog clears its
unsaved secret fields. Lists and detail preserve Caido configuration and its
explicit-null overlay semantics under
[06](06-server-ui-and-operations.md#runtime-labels-and-infrastructure-configuration)
and [11](11-http-and-caido-tools.md).

Layout task: V37-011. Server/PostgreSQL charts and final allocation metrics are
already planned in V32-006; the 15-second / 60-second / 5-minute intervals,
disabling metrics and independent Go profiling are defined by the
[separate specification](22-performance-metrics-and-profiling.md).
This plan does not add a new pprof toggle to the UI.

## Implementation order

The original implementation waves below are complete for V37. New tasks start
with `status: pending`. Their `priority: P2` follows the
repository policy of "after the first releasable slice". The urgency identified
in the UX-01…UX-11 study is expressed by the waves below, without overriding
that policy. Each YAML file contains the exact dependencies, scope, criteria
and verification commands.

| Wave | Task | Outcome | Rationale |
| --- | --- | --- | --- |
| 1 | [V37-001](../../tasks/v37-001-accessible-dialogs.yml) | Shared Dialog, focus management and nested forms | UX-05 |
| 1 | [V37-002](../../tasks/v37-002-run-draft-continuity.yml) | Drafts within the current tab, local upload to an input slot and Git integration | UX-02 |
| 1 | [V37-003](../../tasks/v37-003-project-workflow-primary-actions.yml) | Add-materials and launch actions visible at the top of the page | UX-01 |
| 1 | [V37-004](../../tasks/v37-004-input-suggestion-review.yml) | Explicit review of inputs suggested by format | UX-03 |
| 1 | [V37-005](../../tasks/v37-005-audit-action-confirmations.yml) | Cancel/Delete Audit confirmations | UX-04 |
| 2 | [V37-006](../../tasks/v37-006-catalog-discovery-contracts.yml) | Metadata and bounded search, version and usage-link APIs | UX-06 |
| 2 | [V37-007](../../tasks/v37-007-catalog-discovery-ui.yml) | Workflow selection by purpose, agent version selection and Where used | UX-06 |
| 2 | [V37-008](../../tasks/v37-008-run-repeat-and-next-actions.yml) | Prefilled repeat and next steps for errors/waits | UX-07 |
| 2 | [V37-009](../../tasks/v37-009-primary-result-preview.yml) | Primary results and preview in one action | UX-08 |
| 3 | [V37-010](../../tasks/v37-010-audit-review-workspace.yml) | Audit summary and decision work queue | UX-09 |
| 3 | [V37-011](../../tasks/v37-011-operations-progressive-forms.yml) | Operations overview, forms opened by action and the impact of changes | UX-10 |
| V37 acceptance | [V37-012](../../tasks/v37-012-ui-journey-verification.yml) | Verification of connected journeys, keyboard access and mobile viewport | US-01…09, US-11 within V37 |
| Separate design | [V38-001](../../tasks/v38-001-evals-experience-contract.yml) | Selected native experiment and independent producer contract | UX-11 / US-10 |
| Planned implementation | [V38-002–010](../evals-experience-design.md#implementation-plan-and-closure-of-the-design-task) | Full Evals setup, launch, review, comparison and release verification | UX-11 / US-10 |

The implemented sequence began with independent Dialog and Project actions,
then added draft continuity, confirmations and input review. V37-006 added the
catalog contract; V37-008 and V37-010 include Server read projections for repeat
setup and bounded Audit review. The final V37-012 gate verifies these behaviors
together.

V35-004 already adds Git Settings/import; V37 uses that implementation. V32-006
continues to own the Performance UI. V37-011 neither depends on charts being
available nor duplicates them. V37 verification covers the implemented repeat
and bounded review contracts; it does not establish performance at production
inventory sizes.

## Verification

The completed [acceptance report](../plans/2026-09-19-v37-journey-verification.md)
records 379 UI unit tests, 36 fixture browser scenarios, the real UI-stack test
and PostgreSQL/API checks. Run drafts, primary previews, repeat setup, nested
keyboard dialogs, Audit review and Operations forms pass at 1440×1000 and
390×844. These results verify behavior, not human task-success rates.

The repeatable acceptance approach is to verify a connected journey rather than
only an individual screen:
choose a Workflow → missing file → fill in the form → navigate away and return →
launch; error → new draft; completed Run → primary result; waiting_review →
evidence → decision; dismiss Audit confirmation without mutation.

Audit checks also cover artifact pagination and ambiguous matches, starting
without a time limit, preserving time through a pause, and continuing a legacy
deadline closure without changing accepted results. Configuration checks cover
keyboard opening/closing of dialogs, exact-version binding selection, Caido
read projections and clearing an unsaved credential on close.

Acceptance covers keyboard access, 1440×1000 and 390×844 viewports, real data
constraints, errors, empty states and recovery after an ambiguous response.
Synthetic browser fixtures verify reproducible interactions; integration checks
verify API, ownership and exact-revision boundaries.

To assess usability, have a new user and an experienced user perform the same
tasks before and after the changes. Measure success without hints, time spent,
backtracking, incorrect input choices and lost input. Set numerical targets
after the baseline measurement; expert review does not replace user testing.
