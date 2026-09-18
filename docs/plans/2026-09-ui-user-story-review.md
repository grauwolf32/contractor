# UI usability review — 2026-09-19

Branch: `feat/ui-user-story-usability`, based on the two completed design commits.
Reference: [US-01…US-11](../spec/ui-user-stories.md). This is a scenario walkthrough
and implementation review, not a measured usability study with participants.

Evidence: route definitions, request/interaction code, existing scenario tests,
and 41 populated routes at 1440px/390px using live stand REST data and a fixture
session. Browser writes are blocked; mutation behavior is tested with fixtures.

## All-view assessment

| Views / route families | Stories | Assessment and planned action |
| --- | --- | --- |
| Home | 02, 04, 11 | Useful triage, but Quick start only considered the first Workflow page and the active list linked to only running Runs. Use complete inventory and the shared version policy; link to the whole queue. |
| Login, guard, not found, application shell | All | Clear authentication errors and recovery navigation. Shared keyboard skip link and modal isolation were fixed in stage 2. Keep server authorization. |
| Projects list / detail | 01, 03 | Contextual upload, target setup, exact revisions and confirmed deletion are present. Workflow matching unnecessarily requires manual Artifact pagination; load complete inventory with retry before offering unique input matches. |
| Project Workflow Run setup / standalone Workflow detail | 02, 03, 06 | Exact version, input review, session-scoped drafts, reset confirmation and nested upload/Git dialogs support the journey. Preserve these contracts. |
| Project Audits / Project Findings | 07 | Complete collections and URL filters support research. Preserve the originating list when opening an Audit, evidence or Run. |
| Audit overview | 07 | Setup/digests/baseline dominate; no immediate linked progress/decision summary. Add complete current-round coverage and pending-review counts with exact filtered destinations; keep stop reason near the top and move technical baseline into disclosure. |
| Audit Coverage / Findings / Checks / Reviews | 07 | Evidence, assessments and decisions remain distinct. Add a URL-backed pending/all review filter and anchored review destinations; maintain return context across section links and drill-downs. |
| Audit Runs / Report | 05, 07 | Exact report artifacts and execution links exist. Restore originating Audit section/filter after visiting a Run or Project Artifact. |
| Runs Queue / Completed / Run detail | 04, 05, 06 | Server triage, retry semantics, rerun drafts and primary outputs work. Back link currently loses Completed state/cursor/labels; carry the source URL and nested return state. |
| Runs Configuration / RuntimeConfig detail | 11 | Publication, exact bindings, secret-clearing dialogs and impact copy are present; shared navigation/dialog fixes apply. No new configuration semantics needed. |
| Catalog Workflows / Workflow detail | 02, 03 | Shared cards now show exact latest numeric selection, contracts and complete search; retain authored-only presentation and explicit older choices. |
| Catalog Agents / Agent detail / Skills | 08, 09 | Server-wide search, exact usage and prompt Preview/Source/Copy are present. Preserve catalog/Skills context on Artifact visits. Cross-version prompt comparison remains a separately planned feature. |
| Artifact library / User Artifact detail | 09 | Exact metadata and explicit previews/downloads exist. Namespace and cursor are local state and disappear after inspection; move list state to URL and restore it via back links. |
| Project / Eval / Run Artifact detail | 05, 09 | Correct scope, exact revisions and provenance are retained. Back links must restore source context; Rendered/Source tabs need arrow/Home/End navigation and one tab stop. |
| Evals list / detail | 10 | Current label-based grouping remains useful, but does not meet the experiment/variant/repeat comparison story. V38-001 must define the API contract first; do not invent metrics or claim this feature is complete. |
| Operations overview / Runtime Agents / active and completed Allocations | 04, 11 | State, mismatch, activity and capacity distinctions are visible. Stage 2 removed snapshot internals from the primary reading path. |
| Operations Performance | 11 | Metrics intervals, unavailable/disabled states and scoped read-only charts are present. No new controls needed. |
| LLM configurations / detail, Credentials / detail, Settings | 11 | Forms, permissions, exact versions and write scope are explicit. Shared controls/dialogs apply. Credential detail inspected in code/tests because stand has no active sample. |

## Implementation order

1. Correct Home defaults and automatically complete Project matching inventory;
   errors offer retry and never imply a sole match from partial data.
2. Preserve context through Run/Artifact/Audit drill-down and back links; keep
   Artifact library namespace and pagination in the URL.
3. Put Audit progress and decisions first; counts and destination filters use the
   same assessment groups, all pages, current round and pending review state.
4. Complete keyboard behavior of Artifact preview tabs.
5. Run scenario regressions, full UI suite/typecheck/lint/build, repeat desktop/
   mobile route review and selected keyboard/return journeys; commit, deploy an
   immutable UI release, verify local and LAN assets while preserving backend,
   Runtime processes and Scheduler state.

## Remaining product work

US-10 experiment comparison and cross-version prompt comparison require their
separately scoped contracts. Large-inventory performance should be measured with
production-sized data before replacing complete client inventory with a server
family/search endpoint. No invented descriptions or implicit writes are added.

## Implemented and verified

All five implementation steps are complete through pre-deployment validation.
Library and Project Artifact filters now survive inspection; nested return state
survives Run → Artifact → Run → original list. Coverage drill-down and pending
review destinations retain their source context. Explicit version selection,
input review, draft retention and destructive confirmation regressions still pass.

Validation: 57 test files / 348 tests passed, including complete/cyclic Project
inventory, latest Home version on a later page, complete Audit coverage/review
counts, pending-review anchors, nested return context, library filter restoration,
and preview keyboard navigation. Typecheck, full source ESLint, changed-file
formatting and production build passed. Read-only browser review covered all 41
available routes at desktop/mobile sizes, plus seven live-data journeys. No JS
errors, attempted writes or root horizontal overflow. The existing deadline stop
notice remains visible and is expected. Browser authentication uses a fixture;
real login and production mutations were not exercised.

Deployment uses a separate immutable release directory and restarts only the UI
service. The release manifest records the exact commit, verification artifacts,
previous UI override, health checks, backend PIDs and Scheduler state.
