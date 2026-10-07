# V3B build contract

[V3B implementation plan](v3b-implementation.md) · [Coverage map](v3b-coverage.html) ·
[V3B Panes](v3b-panes.md) · [Themes](themes.md)

This is the working agreement for everyone (people and agents) building the V3B
redesign of `ui/`, developed on `feat/ui-v3b` and now integrated into `main`.
The plan says *what* and *why*;
this contract says *how*: file ownership, shared building blocks, copy, keyboard,
layout and the checks every change must pass. When the code and this contract
disagree, fix one of them in the same commit.

## 1. Sources of truth

| Question | Read |
| --- | --- |
| Agreed decisions, constraints, build order | [v3b-implementation.md](v3b-implementation.md) |
| Every capability today and where it lands | [v3b-coverage.html](v3b-coverage.html) (open in a browser; the data is the `ROWS` array in the page source) |
| Visual reference | `docs/design/ui/mockups/v3b/{home,project,start,check,issue}.html` (open in a browser; `?theme=dark` / `?theme=black`) and [v3b-panes.md](v3b-panes.md) |
| Tokens and palette | `ui/src/app/theme.css`, [themes.md](themes.md) |
| User goals and acceptance | `docs/spec/ui-user-stories.md`, `docs/spec/06-server-ui-and-operations.md`, `docs/spec/18-…`, `docs/spec/19-audits.md` (UI parts), `docs/spec/24-git-artifacts.md`, `docs/spec/30-managed-evals.md` |
| API | `api/openapi/contractor-public-v1.yaml`, wrappers in `ui/src/api/*.ts` |

## 2. Ground rules

1. **Lose nothing.** Every row of the coverage map stays reachable. A
   capability may move (behind *Technical details*, into a menu, to another
   page) but never disappears. If you remove a control, name where it went in
   your report.
2. **Existing URLs keep loading directly**, with their query and hash
   semantics. New URLs are only the ones in §5. No compatibility shims for old
   internal APIs (the project is pre-production), but no URL breaks.
3. **No optimistic lifecycle state.** After a mutation, invalidate and refetch;
   rows change only when the server confirms (S06:225-230).
4. **Separate actions stay separate.** Repeat, Continue from failed stage and
   Retry model connection are three actions with their own confirmations.
   Confirm, Not an issue, Needs evidence, Duplicate and Reopen are five
   decisions. Cancel and Delete keep their confirmation dialogs.
5. **Colours only from tokens** (`var(--…)` from `theme.css`). No raw hex, rgb
   or named colours in CSS or TSX. Every screen works in light, dark and black.
6. **No browser storage** except the existing theme preference. No secrets in
   the DOM after submit, in URLs, or in storage.
7. **Accessible by default.** Real `<button>` and `<a>`; icon-only controls have
   `aria-label`; status is never colour alone (icon + word); visible focus; the
   shared `Dialog` for every modal; layouts at 1440, 390 and 320 px without
   horizontal page overflow.
8. **Keep repo conventions.** TypeScript strict, ESLint and Prettier clean,
   Testing Library for unit tests, no new runtime dependencies without a reason
   stated in the commit message.
9. **Visible identifiers required by the specs stay visible** as `IdChip`s:
   check ID in the check header, `name@version` before a Workflow Run and on
   check types, Git commit after import and on materials, Runtime Agent ID.

## 3. Vocabulary and copy

The UI is in English. One term, one meaning.

| Internal | User-facing |
| --- | --- |
| Audit | Check |
| AuditProfile, audit preset | Check type |
| Audit work item / coverage item | Endpoint (OpenAPI traces), Requirement (Top 10, ASVS, standards), Scenario (WSTG); **Item** when the kind is unknown |
| "N checks" in presets | "N requirements" or "N scenarios" |
| Eval assessment check | Criterion |
| Finding proposal (state `proposed`) | Possible issue |
| Confirmed finding | Issue |
| Artifact | Material (project pages), File (Library) |
| Run, Workflow, Runtime Agent | unchanged |
| Home | Inbox |
| Catalog | Library |

State labels (use `src/app/vocabulary.ts`, never ad-hoc strings):

| Audit state | Label | Tone |
| --- | --- | --- |
| draft | Draft | idle |
| active | Running | progress |
| waiting_review | Waiting for you | review |
| paused | Paused | warning |
| finalizing | Finishing | progress |
| cancelling | Stopping | warning |
| completed | Finished | done |
| cancelled | Stopped | neutral |
| failed | Failed | blocked |
| deleting | Deleting | neutral |

| Coverage status | Label | Tone |
| --- | --- | --- |
| not-tested | Not checked yet | idle |
| inconclusive | Inconclusive | warning |
| satisfied | Met | done |
| violated | Issue found | blocked |
| not-applicable | Not applicable | neutral |
| blocked | Blocked | blocked |
| excluded | Excluded | neutral |
| traced-complete | Fully traced | done |
| traced-partial | Partially traced | partial |
| unmapped | Unmapped | neutral |

| Finding state | Label | Tone |
| --- | --- | --- |
| proposed | Needs review | review |
| confirmed | Confirmed | success |
| rejected | Not an issue | neutral |
| duplicate | Duplicate | neutral |
| needs-evidence | Needs evidence | warning |

Decision verdicts map onto the API as: Confirm issue → `true_positive` (+
severity), Not an issue → `false_positive`, Needs evidence → `needs_evidence`,
Duplicate → `duplicate` (+ `duplicateTargetId`), Reopen → `reopen`. Every
decision needs a non-empty rationale (`minLength: 1`).

Wording that must not change in meaning:

- suggestions say "format matches", never semantic fit (UUS:80-81);
- an unknown wait reason stays unknown; idle slots do not mean capacity;
- publishing does not change prepared allocations;
- "Meets declared gates" for evals; a missing metric is not zero;
- a proposed report never reads as accepted; reports keep "not a security or
  compliance certification";
- possible issues stay separate from confirmed issues; technical outcome,
  coverage and decision stay separate.

## 4. Layout, keyboard and visual language

- **Shell.** Rail on the left (§6), content on the right. ≤ 820 px: a top bar
  with logo, a command button and a **Menu** / **Close menu** toggle that opens
  the destinations as a drawer.
- **Panes.** Destinations with lists use `PaneLayout`: list pane (360 px; 300 px
  between 821 and 1099 px) and detail pane. ≤ 820 px shows one pane: the list,
  or the detail with a "Back to …" link when something is selected. Selection
  lives in the URL (path segment or query), never only in state.
- **Keyboard.** J / K (and ↓ / ↑ inside a focused list) move the selection;
  Enter opens; C / R / E decide on a possible issue; Ctrl+Enter / ⌘+Enter records
  a decision from the rationale field; Ctrl+K / ⌘+K opens the command palette;
  Esc closes. Use `useShortcuts` / `useListNavigation` from `src/ui`: they
  ignore events from inputs, textareas, selects and contenteditable, ignore
  key repeats with other modifiers, and stay off while a modal dialog is open.
  Declare keys with `aria-keyshortcuts` and show them with `Kbd`.
- **Visual language** (match the V3B mockups): Manrope for text,
  JetBrains Mono for endpoints, file:line, code and IDs. Flat surfaces with
  1 px hairlines (`--line`), 12–14 px radii for panels, 8–10 px for controls.
  Primary action: `--primary` fill. Selection: `--accent-dark` fill with a
  3 px `--accent` bar on the left. Status: `StatusGlyph`/`StatusChip` tones.
  No gradients, no coloured left borders on cards (the selection bar is the
  only one), no emoji.
- **CSS.** Put styles in your area's own CSS file with an area class prefix
  (`inbox-`, `checks-`, `issues-`, `reports-`, `start-`, `projects-`,
  `materials-`, `runs-`, `library-`, `evals-`/`eval-`, `ops-`, `shell-`, `ui-`).
  You may move or delete rules in `src/styles.css` that only style your area;
  never edit rules that style other areas.

## 5. Routes

Existing routes stay. New routes (registered by the foundation with stub
components that the owning agent replaces):

| Path | Component (file) | Owner |
| --- | --- | --- |
| `/` | `InboxRoute` (`src/routes/inbox/index.tsx`) | Inbox |
| `/checks` | `ChecksRoute` (`src/routes/checks/index.tsx`) | Checks |
| `/checks/new` | `StartCheckRoute` (`src/routes/checks/start.tsx`) | Start |
| `/issues`, `/issues/:auditId/:findingId` | `IssuesRoute` (`src/routes/issues/index.tsx`) | Issues |
| `/reports`, `/reports/:auditId` | `ReportsRoute` (`src/routes/reports/index.tsx`) | Reports |

Deep-link parameters:

- `/checks/new?project=<projectId>&objective=<text>&type=<profile name>` prefills
  Start; without `project` it asks for one.
- `/projects?new=1` opens the New project dialog.
- `/checks?state=<filter>&project=<projectId>&check=<auditId>` selects a filter
  and a check in the list.
- `/issues?state=<finding state>&project=<projectId>&severity=<…>` filters;
  the selected issue is the path.

The check page stays at `/projects/:projectId/audits/:auditId[/:section]`
(sections `coverage`, `findings`, `reviews`, `runs`, `report`, `overview`;
`checks` → `coverage` redirect stays). The rail shows **Checks** as active
there.

## 6. Shell (foundation)

- **Rail groups** (`<nav aria-label="Primary navigation">`, links with labels):
  1. Inbox `/` (badge: items needing the user), Projects `/projects`,
     Checks `/checks`, Issues `/issues`, Reports `/reports`;
  2. divider, then Runs `/runs`, Library `/catalog`, Evals `/evals`;
  3. bottom: Operations `/operations` (only with the `operations` capability),
     account menu button.
- **Active state**: Inbox only on `/`; Checks on `/checks*` and
  `/projects/:id/audits/:auditId*`; Projects on other `/projects*`; Issues on
  `/issues*`; Reports on `/reports*`; Runs on `/runs*`; Library on `/catalog*`
  and `/artifacts*`; Evals on `/evals*`; Operations on `/operations*`.
- **Top bar**: command button "Search or start a check…" with `Ctrl K` hint
  (opens the command palette dialog).
- **Account menu**: signed-in user, Settings (`/operations/settings`), Theme
  (System / Light / Dark / Black), Sign out, `UI <version>`.
- Skip link to `#main-content`; `<main id="main-content" tabIndex={-1}>`.

## 7. Foundation building blocks

Built first; page work starts from them. The foundation records the final
signatures in §11 "As built"; use those.

- `src/ui/` primitives (`src/ui/index.ts` exports all; styles in
  `src/ui/ui.css`, imported eagerly from `src/main.tsx`):
  `PaneLayout`, `ListPane`, `ListSection`, `ListRow`, `FilterChips`,
  `DetailPane`, `DetailHeader`, `StatusGlyph`, `StatusChip`, `MethodChip`,
  `IdChip`, `Kbd`, `ProgressSegments`, `TechnicalDetails`, `ActivityLog`,
  `DecisionBar`, `EmptyState`, `useShortcuts`, `useListNavigation`.
  `StatusTone = "done" | "partial" | "progress" | "blocked" | "idle" |
  "review" | "warning" | "success" | "info" | "neutral"`.
- `src/app/vocabulary.ts`: labels and tones of §3; `itemNoun(kind, count)`;
  `checkItemKind(…)`.
- `src/api/cross-project.ts`: owner-wide cursor lists (every page of projects,
  checks, findings and reviews; polling 20 s; `refetchOnWindowFocus` off;
  partial results retained when a continuation fails):
  `useProjectsIndex`, `useAllChecks`, `usePendingDecisions`,
  `useAllPossibleIssues`, `useAllReports`, `useInboxSummary`,
  `invalidateCrossProject`.
- `src/routes/decisions/`: `FindingSummary` (what the AI found, code lines,
  impact, evidence, uncertainty), `FindingDecision` (opens a review request
  when needed and records true positive + severity, false positive, needs
  evidence, duplicate with target picker, reopen; rationale required),
  `ActionDecision` (active-check approval and requirement applicability:
  approve / reject / not applicable), `ReportDecision` (report acceptance).
  Shared by Inbox, Issues, Checks and Reports.
- Tokens added to `theme.css` for V3B parts that were missing (`--chrome`,
  `--chrome-2`, `--faint`, `--sel`, `--sel-bar`, `--sel-line`, `--code-bg`,
  `--code-hl`, `--badge-bg`, `--badge-ink`, `--m-idle`, `--m-partial`,
  `--m-progress`, `--m-blocked`, `--m-done`).

## 8. File ownership (page stage)

Edit only what you own. Read anything. If you need a change in a file you do
not own, describe it in your report (`shared_file_requests`). `query-keys.ts`
is append-only: add your keys in a new top-level group named after your area.
Do not edit existing `src/api/*.ts` wrappers; add new wrappers in a new
`src/api/<area>.ts` file. Do not edit `ui/e2e/**`, `router.tsx`,
`server/static-server.mjs` or `src/app/shell.*`: the integration step owns them.

| Area | Owns |
| --- | --- |
| Inbox | `src/routes/inbox/**`; deletes `src/routes/home.tsx`, `src/routes/home.test.tsx` |
| Checks | `src/routes/checks/index.tsx` and `src/routes/checks/list/**`; `src/routes/projects/audits/**` except the files owned by Issues, Reports and Start below; `audits.test.tsx` (shared: edit only tests of your features) |
| Issues | `src/routes/issues/**`; `projects/audits/{audit-findings.tsx,finding-card.tsx,findings.tsx,finding-locations.tsx,finding-locations.test.tsx,finding-options.ts,queue.tsx,queue-state.ts}` |
| Reports | `src/routes/reports/**`; `projects/audits/report.tsx` |
| Start | `src/routes/checks/start.tsx`, `src/routes/checks/start/**` |
| Projects | `src/routes/projects/**` except `audits/**`, `artifact-region.tsx`, `artifact-detail.tsx` |
| Materials | `src/routes/artifacts/**`, `projects/artifact-region.tsx`, `projects/artifact-detail.tsx` |
| Runs | `src/routes/runs/**`, `src/routes/queue.tsx`, `src/routes/queue.test.tsx` |
| Library | `src/routes/catalog/**`, `src/routes/workflows/**`, `src/routes/skills*.tsx`, `skill-*.tsx`, `skills.css` |
| Evals | `src/routes/evals/**`, `src/routes/evals-*.test.tsx` |
| Operations | `src/routes/operations/**`, `src/routes/settings/**`, `src/routes/login*.tsx`, `session-error.tsx`, `placeholders.tsx`, `guard*.tsx` |

Interfaces that cross owners and must keep their export names and props:
`AuditFindingsSection`-like exports used by the check page for its `findings`
section (owned by Issues), the report section export used for `report` (owned
by Reports), and the components in `src/routes/decisions/`.

## 9. Checks before you commit

From `ui/` (first run `corepack pnpm install --frozen-lockfile` in a fresh
worktree):

```shell
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm test --run --maxWorkers=2
corepack pnpm build
```

All four must pass. Browser specs (`ui/e2e`) are reconciled and run by the
integration step; list every accessible name, label, heading, route or test id
you changed so it can update them.

## 10. Report format (agents)

Return: worktree branch and commit; one-paragraph summary; files added and
deleted; accessible names and visible labels changed (old → new, where);
routes changed; e2e specs likely affected; shared-file requests; coverage-map
rows you implemented and any row you could not; open questions.

## 11. As built

### Primitives (`ui/src/ui`)

The full API with one example per component is in `ui/src/ui/README.md`;
read it before using a primitive. Highlights:

- `PaneLayout({ list, detail, showDetail, backLink?, listLabel, detailLabel })`
  manages focus when switching panes on narrow screens; selection must be in
  the URL so Back restores it.
- `ListRow`: the title covers the row as one click area; put actions in
  `children` or `trailing`, never inside `meta`. An array `meta` renders "·"
  separators that start the part they precede; the meta line clips the one
  that starts a wrapped line (`data-parts`), so no line ends in "·".
  `ariaKeyShortcuts` declares the row's keys; `ListSection` takes
  `aria-keyshortcuts` and `aria-label` for its list.
- `ListPane` and `DetailHeader` take `titleRef`: the heading becomes a focus
  target (`tabIndex={-1}`) for when the selected item goes away.
- `FilterChips`: pressing the current chip does nothing.
- `TechnicalDetails` takes `className` and `onToggle(open)`; `IdChip` takes
  `wrap` for long values.
- `DecisionBar` is presentational: the caller owns verdict, severity and
  rationale state and performs the mutation in `onSubmit`.
- `useShortcuts` / `useListNavigation` implement the keyboard guards of §4; a
  binding can be `{ handler, when }` to decline an event before
  `preventDefault` (e.g. Ctrl/⌘+Enter on a focused link).
- `DetailPane`'s pinned footer caps a recorded decision (`ui-footer-record`)
  at `min(45vh, 24rem)`; the pending bar is never capped.
- All optional props accept `undefined` (`exactOptionalPropertyTypes`).

### App parts (`ui/src/app`)

- `ActionMenu({ label, children })` is the "⋯" menu. Its panel is a manual
  popover in the top layer, placed next to the trigger in viewport
  coordinates (`menu-placement.ts`), so scrolling panes never clip it; its
  whole look is in `action-menu.css`. Do not restyle it per area.
- `usePageLocationNow()` returns where the user is now while that is still
  this page (the destination of a navigation in progress, else the router's
  location), or `undefined`. React Router applies navigations in a
  transition, so a page being left stays mounted with its old location; ask
  this before navigating on the user's behalf after a late answer (a
  recorded decision, a started check).
- Route CSS loads with its route chunk. A component imports the stylesheet
  that defines its classes (shared parts keep theirs next to them), never
  relying on CSS another route happens to have loaded, or its look depends
  on which page the user opened first.

### Vocabulary (`ui/src/app/vocabulary.ts`)

Tables keyed by the generated API enums (`CHECK_STATE_LABELS`,
`COVERAGE_STATUS_LABELS`, `FINDING_STATE_LABELS`, `VERDICT_LABELS`,
`REVIEW_ACTION_LABELS`, `REVIEW_STATE_LABELS`, `REPORT_STATUS_LABELS`,
`REVIEW_KIND_LABELS`, `SEVERITY_LABELS`) and lookups returning
`{ label, tone }`; `TERMS` for shared nouns; `ItemKind`, `itemNoun`,
`itemCount` and `checkItemKind(source)` (work item kind → profile inventory
and standards → audit baseline standards → profile name → "item").

### Cross-project data (`ui/src/api/cross-project.ts`)

- `CROSS_PROJECT_LIMITS = { pageSize: 200, pollMs: 20_000 }`.
- `INBOX_CHECK_STATES = ["active", "waiting_review"]`: pass it as
  `checkStates` on the Inbox so lists match the rail badge and share reads.
- Hooks return `CrossProjectStatus` (`truncated`, `partial`, `errors` keyed by
  `scope: "index" | "list" | "project" | "check"`, `error`, `isPending`, `refetch`) plus:
  `useProjectsIndex() → projects`; `useAllChecks({ states? }) → checks`;
  `useAllPossibleIssues({ checkStates?, states?, verdicts?, severities? }) → issues, truncatedAuditIds, total`;
  `usePendingDecisions({ checkStates?, kinds? }) → decisions, truncatedAuditIds`;
  `useAllReports({ statuses? }) → reports` (default proposed and ready; 404 = no report);
  `useInboxSummary() → { needsDecision, possibleIssues, otherDecisions, partial, truncated }`.
- Checks, findings and reviews use `/v1/audits`, `/v1/findings` and
  `/v1/reviews`. Collections follow every cursor sequentially and have no
  fixed cap; report payloads remain pinned to each check revision.
- `invalidateCrossProject(queryClient)` after every decision or lifecycle
  mutation. Keys live in `queryKeys.crossProject`.

### Shell (`ui/src/app/shell.tsx`, `destinations.ts`, `account-menu.tsx`, `command-palette.tsx`)

- The shell sets `--shell-rail-width: 76px` and `--shell-topbar-height: 56px`
  on `.application`; `PaneLayout` uses the latter. `.content` has no padding
  on pane pages; give your page its own padding when it is not a
  `PaneLayout`.
- Rail destinations and active rules live in `destinations.ts`
  (`DESTINATIONS`, `destinationsFor(capabilities)`, `activeDestination(path)`,
  `projectIdOf(path)`). The Inbox badge reads `useInboxSummary()`.
- The account menu (`AccountMenu`, `AccountPanel`) holds Settings, Theme,
  Sign out and the UI version; at ≤ 820 px the same panel is in the "Menu"
  drawer.
- The command palette (Ctrl/⌘+K) offers Start a check (with `?project=`
  inside a project), New project (`/projects?new=1`), Go to destinations plus
  Files (`/artifacts`), and searches projects, checks, check types
  (`useAuditPresets` from `routes/catalog/audit-preset-data.ts`) and workflows
  (`useWorkflowInventory` from `routes/workflows/inventory.ts`). Keep those
  two export names and shapes.
- `/artifacts` has no rail item; the Library page must link to Files.

### Decisions (`ui/src/routes/decisions`, README.md there)

- `FindingSummary({ auditId, finding, variant?: "full" | "compact", titleAs? })`
  renders only sections backed by `AuditFinding` fields.
- `FindingDecision({ auditId, finding, onDecided?, next?, autoFocus?, pendingReview?, shortcuts? = true })`
  records Confirm issue (C, severity), Not an issue (R), Needs evidence (E),
  Duplicate… and Reopen (More), creating the review request when needed
  (If-Match + Idempotency-Key), refreshes the check, project lists and
  cross-project queries, announces "Decision recorded" and calls
  `onDecided`. Pass `shortcuts={false}` when several decisions share a page.
- `ActionDecision({ auditId, review, onDecided?, onRecording? })` (approvals
  and applicability) and `ReportDecision({ auditId, review, report?,
  onDecided?, onRecording? })` (report acceptance) follow the same rules;
  `DecisionRecord({ decision })` shows a recorded decision. While a decision
  is being recorded, and until the refreshed request arrives, the bar is not
  swapped for a "no longer matches" state; `onRecording(recording)` tells the
  page.
- If your page unmounts the decision on success (for example it moves to the
  next item), announce the outcome yourself in a page-level status region and
  move focus to the next item.

### Routes

New routes (§5) are registered with stubs; owners replace the stub body and
keep the export name (`ui/src/app/router.test.tsx` checks identity). List and
item routes share one component, so selection keeps the list mounted; read
`auditId` / `findingId` with `useParams()`. The static server allows the new
paths; query strings are ignored when matching. Unknown shapes
(`/checks/:x`, `/issues/:a`, …) are 404.

## Check history API follow-up (2026-10-07)

All activity uses `GET /v1/audits/{auditId}/events` and its immutable sequence
order. Read 50 events per page and expose Load older activity. Continuations
keep the same `throughSequence` and `total`; a refresh starts a new prefix and
rebuilds every loaded continuation. Keep settled pages visible when a later
request fails, with an explicit retry. Poll at most once every 5 seconds while
the check can change; stop off-route or terminal, with a final refresh when it
becomes terminal and immediate invalidation after mutations. The current-state
sentence is not a recorded event. Never reconstruct missing check events from
current timestamps. Per-item activity still summarizes retained attempts and
results; WebSocket transport is deferred by S19.
