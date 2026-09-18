# UI improvements — implementation and review plan

Requested sequence, 2026-09-19:

1. Implement the approved Project header and shared Workflow cards; commit on main.
   - Contextual Project actions and a compact secondary menu.
   - One Workflow family per card; explicit selected version, latest numeric
     version by default, previous versions available, no silent selection change.
   - Catalog contracts remain visible in a compact layout; Project cards show
     artifact matching, ambiguity and the selected version's primary outputs.
   - Preserve exact-version links, Run drafts, all published versions and pagination.
2. Inventory all route families, record consistency findings, apply shared UI fixes
   and commit on main. Review headings, actions, navigation, forms, tables,
   loading/error/empty states, mobile layout and keyboard interaction.
3. Create a separate usability branch. Map every route family to US-01…US-11,
   record evidence and priorities, implement actionable journey improvements,
   validate and deploy that branch to the stand. Keep separately scoped Evals
   experiment work distinct from the currently supported label-based grouping.

Validation: targeted behavior tests for version selection, exact links and
context actions; full UI tests, typecheck, lint/format and production build;
read-only browser review against stand data on desktop/mobile. Deployment
preserves Server, Runtime processes and the current scheduler state.

## Progress

- Planning: approved design and existing User Stories/specifications reviewed.
- Implementation and review findings will be recorded below as each stage completes.

### Stage 1 — implemented

Shared WorkflowCard and version grouping now serve both Project and Catalog.
Catalog reads complete inventory with cursor-cycle validation; Project matching
uses the same complete Workflow list. Numeric components compare as arbitrary
precision integers. Opaque labels have no Latest badge. Older explicit choices
survive refresh, and configuration opens the exact selected version.

Project actions moved to their sections; confirmed deletion remains in a compact
menu. Catalog cards expose compact contracts without placeholder descriptions.
No immutable published definition was rewritten to invent presentation metadata.

Verification: 53 files / 340 tests passed; typecheck, targeted ESLint, production
build and diff whitespace checks passed. Live read-only browser review: 15 catalog
families from 33 versions, desktop/mobile catalog fits, no JS errors or writes.
Project mobile has pre-existing horizontal overflow; tracked for stage 2.

### Stage 2 — consistency audit and implementation plan

Baseline: 41 available routes, each at 1440px and 390px, against live REST data
with a fixture browser session and writes blocked. Includes Home/login/404,
Project and Eval lists/details, all seven Audit sections and Project findings,
Run queue/history/detail/configuration, Catalog lists/details, Artifact previews
at each scope, and every Operations section. No active credential was available
for a live credential detail screenshot; its form is covered by component tests.
No JavaScript errors or unnamed buttons were observed.

Findings and fixes:

| Area | Evidence | Change |
| --- | --- | --- |
| Section navigation | Catalog uses underlines, Operations/Run views use pills, Project/Audit use a third shape | Shared section navigation sizing, active/focus state and scrollable mobile behavior; retain local sticky positioning |
| Headings and actions | Agents heading smaller than sibling lists; primary/secondary controls differ in height | Consistent list heading and control tokens; keep deliberately compact embedded card controls |
| Dialogs | Run deletion and HTTP target lack shared focus containment; Skill upload has a separate partial implementation | Use existing Dialog for focus trap, inert background, Escape, initial focus and focus restoration; pending writes cannot dismiss |
| Project menu | Native disclosure has no Escape/outside dismissal | Reusable action disclosure with keyboard and outside dismissal, preserving native Tab navigation |
| Mobile Project | Long hash in description widens document from 390 to 399px | Wrap metadata values; constrain grid children rather than hide overflow |
| Cross-view feedback | Generic failures incorrectly say Artifact request failed | Neutral request failure fallback |
| Global keyboard navigation | Repeated sidebar must be traversed before content | Visible-on-focus Skip to content link; close mobile menu via Settings too |
| Operations overview | Snapshot internals occupy a full first-screen panel | Collapsible snapshot details; task-focused introductory copy |

The shared styles apply to the route families above. Tables retain horizontal
scrolling, Artifact previews keep their media-specific layout, and operational
state distinctions/destructive confirmations remain intact.

Implemented all fixes above. Validation: full suite exercised 341 tests across
54 files; the Project refresh regression was updated to simulate a background
query refresh while the dialog is inert, then all 12 Project tests passed.
Typecheck, ESLint, production build and whitespace checks passed. The repeated
41-route / 82-viewport read-only browser review has no root horizontal overflow,
JavaScript errors, unexpected error alerts or unnamed buttons. Scheduler settings
review preserves the API's Cache-Control header; authentication remains a fixture,
so this review is not an end-to-end login or mutation acceptance test.

### Stage 3 — separate branch

Usability review, route/story matrix, implementation choices and validation are
recorded in [the all-view User Story review](2026-09-ui-user-story-review.md).
Branch: `feat/ui-user-story-usability`. Full final UI suite: 57 files / 348 tests.
