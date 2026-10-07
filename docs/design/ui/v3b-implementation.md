# V3B implementation plan

[UI redesign explorations](README.md) · [V3B Panes](v3b-panes.md) ·
[Coverage map](v3b-coverage.html)

V3B Panes is the implemented direction for the `ui/` redesign. This document
records the decisions agreed on 2026-10-06, the constraints the redesign must
keep, and the build order. The [coverage map](v3b-coverage.html) records the
original UI inventory (118 rows) with its intended place in V3B, the API it
uses, and its status. Open it in a browser.

## Implementation status (2026-10-07)

All nine steps in the build order below are implemented and integrated into
`main`. The order records how the redesign was built; it is no longer an open
task list. The shared component contract is in
[v3b-build-contract.md](v3b-build-contract.md), with the component APIs in
[`ui/src/ui/README.md`](../../../ui/src/ui/README.md).

Browser journeys now follow the V3B names, account menu, Materials and Library
tabs, check item panes and decision bars. They retain exact request payload,
revision, evidence, browser storage and destructive-action checks.

The API gaps below remain: global lists still use bounded client fan-out,
activity is assembled from snapshots, and suggestions use static rules.

The map was built from four read-only inventories of `ui/` (projects and
audits; runs, artifacts, catalog and evals; shell, operations and the visual
system; specs and browser gates). Each row was checked against
`api/openapi/contractor-public-v1.yaml`.

## Decisions

### Information architecture

| Rail group | Destinations |
| --- | --- |
| Main | Inbox (`/`) · Projects · Checks · Issues · Reports |
| Under a divider | Runs · Library · Evals |
| Bottom | Operations (only with the `operations` capability) · account menu |

- **Library** has the tabs Check types · Workflows · Agents · Skills · Files.
- **Account menu** holds Settings (`/operations/settings`, including the Git
  SSH key for every user), Theme, Sign out, the signed-in user and the UI
  version.
- **URLs.** Every existing URL keeps loading directly. New list routes are
  `/checks`, `/issues` and `/reports`; the Inbox replaces Home at `/`. New
  routes are added to `CLIENT_ROUTES` / `CLIENT_ROUTE_PATTERNS` in
  `ui/server/static-server.mjs`.

### Vocabulary

The UI stays in English. Each user-facing term has one meaning.

| Internal | User-facing |
| --- | --- |
| Audit | Check |
| AuditProfile, preset | Check type |
| Audit work item | Endpoint (API traces), Requirement (Top 10, ASVS), Scenario (WSTG); Item when the kind is unknown |
| "N checks" in presets | "N requirements" or "N scenarios" |
| Eval assessment check | Criterion |
| Finding proposal / confirmed finding | Possible issue / Issue |
| Artifact | Material on project pages; File in Library |
| Run, Workflow | Unchanged; they live in Runs and Library |

### Identifiers the specification requires

These stay visible as small monospace chips with a copy action:

- the check ID in the check header, short form with the full ID on hover (S19:1816-1817);
- `name@version` before a Workflow Run and on check types (UUS:62-63);
- the Git commit after an import and on the material (S24:26-27);
- the shortened Runtime Agent ID in Operations (S06:943).

Revisions, digests, rounds, slots, tokens and allocations move behind
*Technical details*.

### API gaps and how V3B ships without new server work

| Gap | First version | Follow-up |
| --- | --- | --- |
| No cross-project lists (Inbox, Checks, Issues, Reports) | Client fan-out over the first page of projects (≤ 50). The Inbox reads only active and `waiting_review` checks. Polling every 15–30 s, refetch after each action. | Task for `GET /v1/audits`, `GET /v1/reviews?state=pending`, `GET /v1/findings?verdict=unreviewed` |
| Decisions require a rationale (`minLength: 1`) | Required short field in the decision bar. C / R / E choose the verdict and focus the field; Enter records. "Use AI summary" copies the conclusion on request; no automatic prefill. | — |
| Duplicate and Reopen | In the decision bar's More menu. Duplicate opens a finding search within the same check. | — |
| No per-item retry | No Retry button on an endpoint. Show "Retried automatically · attempt N of M". After the last attempt, link to the child Run and the recovery actions the server offers there. | Item retry API only if this proves too indirect |
| No Audit event stream | The activity log is built from item, attempt, finding and review timestamps. | Audit events (S19:1858-1860) |
| No check-type suggestions | Static rules: project materials plus keywords in the objective map to a check type; "Why this fits" is the rule's fixed text. Never claim more than the rule checks. | Optional model call |

Review kinds other than finding triage (active-check approval, requirement
applicability, report acceptance) are served by
`/v1/audits/{auditId}/reviews` and have V3B decision screens.

### Smaller defaults

- **Sign out** moves into the account menu. `ui/e2e/stack.spec.ts` opens the menu first.
- **Theme** follows the system (light or dark) by default; black is chosen explicitly. The choice is stored in `localStorage`: it is a browser preference, not a secret or a draft.
- **Fonts.** Manrope and JetBrains Mono ship as bundled woff2 files (SIL OFL). The CSP stays `font-src 'self'`.
- **Evaluation workspaces** stay: experiments and datasets live inside them.
- **Inbox · Unblock** lists failed standalone Runs and blocked check items. Each item offers its own action; there is no single generic Retry.

## Must keep

- Every existing capability in the coverage map. Admin-only areas stay
  observation-first: no force, release or reassign controls.
- Repeat, Continue from failed stage and Retry model connection stay separate
  actions with their own confirmations (S06:332-379, spec 04:89-92).
- No optimistic lifecycle state: act-in-place actions refetch (S06:225-230).
- Possible issues stay separate from confirmed issues; technical outcome,
  coverage and decision stay separate (S19:1791-1793, UUS:127-128). A proposed
  report never reads as accepted.
- Wording constraints:
  - "format matches", never semantic fit (UUS:80-81);
  - an unknown wait reason stays unknown, and idle slots do not mean capacity (UUS:91-93);
  - publishing does not change prepared allocations (UUS:198-199);
  - "Meets declared gates" (S30:529-530);
  - "not a security or compliance certification" on reports.
- Secrets handling: password inputs with `autoComplete=new-password`, wiped
  after submit; the Git key never enters the query cache; confirmations for
  destructive actions; 409/412 reconcile notices.
- Accessibility and layout: the shared dialog contract, skip link, keyboard
  tabs. Layouts at 1440, 390 and 320 px with no horizontal overflow.
- No secrets or drafts in browser storage; the stack gate scans DOM, storage
  and screenshots.

## Specification and test changes that travel with the UI

- S06, S18 and UUS now describe the V3B navigation and Inbox.
- Browser specs follow the V3B accessible names and account menu while
  retaining the "Primary navigation", "Project sections" and "Run views"
  landmarks. Keep them aligned with the screens they test and the browser
  gates in `ui/server/mocked-browser-gate.mjs`, `tests/ui-stack/stack_test.go`
  and the `tests/e2e/*matrix*.yml` bindings.
- Any `ui/src` change runs both live browser shards in PR CI
  (`release-verify-browser-a`, `-b`).

## Build order

1. **Visual foundation** (implemented on `main`).
   - Bundled fonts and the token set for light, dark and black (see [Themes](themes.md)), replacing every hard-coded colour in the CSS.
   - Theme preference with a control in Settings.
   - Scalar, LikeC4, the GPU palette, the eval A/B colours, `index.html` and the server 404 page follow the theme.
   - Today's layout is kept; it only loses the lime accent.
2. **Shell.** Rail with three groups, account menu, command bar, phone layout,
   the list + detail pane frame. Spec rewrite for navigation.
3. **Checks.** Running check (API trace and requirement checks), all check
   states and controls, Decisions, Technical details.
4. **Issues.** List across projects, detail tabs, complete decision bar.
5. **Inbox.** Decide, Unblock, Ready, Running with fan-out.
6. **Start a check.** Check types with readiness, rules for suggestions,
   advanced options, time limit.
7. **Projects.** Overview, Materials, material detail, settings.
8. **Reports.** List and detail with acceptance.
9. **Runs, Library, Evals, Operations, sign-in.** Restyled into the pane
   frame with every action kept.

These steps were developed on `feat/ui-v3b` and integrated into `main`.
Verification uses `make ui-verify`, the mocked browser gate and both live
browser shards.

## Remaining work

1. **Cross-project APIs.** Add owner-scoped, paginated lists for checks,
   pending decisions and findings, then replace the client fan-out. This
   removes the current first-page limits and reduces polling requests.
2. **Check events.** Provide the Audit event stream described by S19 before
   replacing the activity assembled from item, attempt and review timestamps.
3. **Visual editor.** Implement the chosen
   [B node studio](editor-studio.md) as a separate phase: authored YAML import,
   graph and inspector, validation and YAML export, using the V3B themes.

Per-item retry and model-assisted suggestions remain conditional on a proven
need. Borrowing V3A's quieter rail and typography or adding V3C's column view
still needs a design decision; neither is part of the completed V3B rollout.
