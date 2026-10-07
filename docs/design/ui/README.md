# UI redesign explorations

[Documentation index](../../README.md) · [UI user stories](../../spec/ui-user-stories.md)

This directory records the 2026-10 exploration of a simpler Web UI: the main
user journey redrawn in several directions, and a visual editor for
Workflows, Audits and Agent templates. The static mockups remain design
exploration; the chosen V3B journey is now implemented in `ui/` on `main`.
Accepted behaviour belongs in the [specification](../../spec/README.md).

## Decision status (2026-10-07)

| Track | Variants | State |
| --- | --- | --- |
| Main journey, first round | [V1 Guided](v1-guided.md), [V2 Workspace](v2-workspace.md), [V3 Inbox](v3-inbox.md) | **V3 Inbox** chosen as the base |
| Main journey, V3 variations | [V3A Focus](v3a-focus.md), [V3B Panes](v3b-panes.md), [V3C Board](v3c-board.md) | **V3B Panes** implemented on `main` |
| Palette and themes | [Themes](themes.md) | Accepted: neutral accent instead of lime; light, dark and black themes |
| Workflow / Audit / Agent editor | [Editor studio A, B, C](editor-studio.md) | **B (node studio)** implemented: local YAML authoring, automatic graph layout, catalog selectors and live Check view; advanced follow-ups documented |

V3B Panes is the implemented direction for the `ui/` redesign. The
[comparison](#variants-at-a-glance) gives the reasoning, and the [V3B
implementation plan](v3b-implementation.md) records the agreed decisions and
build order.

## What is wrong today

[Today's UI](original.md), captured from the demo stand, exposes the execution
machinery on every page:

- Run, Audit and revision IDs, artifact digests and `name@N` Workflow names appear as page titles and labels.
- Home is an operator dashboard showing queue admission, runtime slots and published versions. It does not answer "what needs me, and what is running".
- Starting a check means choosing a Workflow and filling string parameters, MIME artifact slots and runtime placement.
- The Audit page shows six counters, semantic disclaimers and a limits table.
- Run detail is about 5,000 px of scheduler and allocation diagnostics.

## Principles shared by every variant

1. **Plain language, outcome first.** Each page answers one question: what
   needs me, what is this project, what should I check, how is it going, is this
   issue real.
2. **Technical detail on demand, not deleted.** IDs, revisions, digests, limits,
   token counts and runtime data stay reachable behind a *Technical details*
   disclosure, or in an *Operations* area for admins. They never appear on the
   main path.
3. **One primary action per page**, visually distinct from everything else.
4. **Act in place.** Confirm or reject a possible issue, and retry a blocked
   endpoint, where the item is shown, without leaving the page.
5. **Status is never colour alone.** Every state has an icon and a word as well
   as a colour; "Not checked yet" is a dashed outline, not a pale fill.
6. **Themes through tokens.** Every colour in a page is a CSS custom property set
   on a root theme class; see [Themes](themes.md).

### Vocabulary

| Internal term | User-facing term |
| --- | --- |
| Audit | Check (a security check of a project) |
| AuditProfile, preset | Check type |
| Workflow, Run, Stage | Hidden; at most "steps" inside a check |
| Artifact (source ZIP, OpenAPI) | Materials: "Source code", "API spec" |
| Coverage or check item | Endpoint (API checks) or requirement (standards) |
| Finding proposal (unreviewed) | Possible issue · needs your review |
| Confirmed finding | Issue |
| Report acceptance | Report |
| Runtime, allocation, queue, slots | Operations area only (admins) |

The review decisions in the mockups map onto `AuditFindingState`:

| Button | State |
| --- | --- |
| Confirm issue | `confirmed` |
| Not an issue | `rejected` |
| Needs evidence | `needs-evidence` |

## The journey every variant draws

Each variant draws the same five pages with the same real data from the demo
stand: project `crapi-workshop`, a running *API endpoint trace* over five
endpoints, and one possible IDOR issue (CWE-639) on
`GET /workshop/api/mechanic/mechanic_report`.

| Page | Question it answers | Replaces |
| --- | --- | --- |
| Home | What needs me, and what is running? | Operator dashboard |
| Project | What is this project, and what can I check with its materials? | Artifact, Run and Workflow lists |
| Start a check | What should I check, and with what? | "Configure Run" form |
| Check in progress | How far along is it, and what is stuck? | Audit overview, coverage and Run detail |
| Review a possible issue | Is this real, and how bad is it? | Audit findings page |

## Variants at a glance

| Variant | Idea | Navigation | Type | Strength | Trade-off |
| --- | --- | --- | --- | --- | --- |
| [V1 Guided](v1-guided.md) | Task-first, wizards, generous spacing | Top bar | Geist | Easiest first journey | Low density; power users click more |
| [V2 Workspace](v2-workspace.md) | Project-centric workspace | Project sidebar, split panes | IBM Plex | Calm, dense triage inside one project | Assumes one project at a time; dense shell |
| [V3 Inbox](v3-inbox.md) | Decisions first, act in place, intent-driven start | Icon rail and command bar | Manrope | Fastest path to decisions | Browsing is second-class |
| [V3A Focus](v3a-focus.md) | V3 without containers; typographic, keyboard-first | Rail; rows expand inline | Manrope | Calmest, fastest to scan from the keyboard | Less self-explanatory for occasional mouse users |
| [V3B Panes](v3b-panes.md) | Rail · list · detail on every page | Rail and list pane | Manrope | Triage without navigation; one layout everywhere | Narrower detail column; sparse lists look empty |
| [V3C Board](v3c-board.md) | Lanes of cards; detail as a sheet | Rail; board lanes | Manrope | Whole state in one glance | Narrow lanes wrap long paths; sideways scroll on phones |

The V3 variations all keep V3's core: an inbox of decisions, acting in place,
an intent-driven start, a plain-language live narrative, a focused review with
keyboard decisions (C confirm, R not an issue, E needs evidence, J next), a
command bar, and Operations visible only to admins.

## Viewing the mockups

`mockups/<variant>/<page>.html` are static, self-contained pages. They need only
Google Fonts and have no build step. Open one in a browser. For V1–V3C, append
`?theme=dark` or `?theme=black` to switch the theme; light is the default. The
editor studio mockups are dark-only. Pages are drawn for a 1440 px frame and
reflow down to phone width.

`screenshots/` holds the light-theme renders used in these documents. They were
rendered at 1440 px with Playwright as full-page captures and stored as lossless
WebP. `original-*.webp` are screenshots of the demo stand UI as of 2026-10-04.

The mockups were authored on a design canvas, which remains the interactive
source with a per-board theme switch:

- Contractor UX Redesign: <https://claude.ai/artifact/7zVAWUFEEeHSS3ftEEuBx1> (private to the owner).
- Workflow & Audit Visualizer: <https://claude.ai/artifact/7tLrop53SQTf73dZS9cWWi> (private to the owner).

The files here are a frozen copy as of 2026-10-05.

## Data and placeholders

All names, counts, times, file:line references and texts come from the demo
stand, either from the brief or from read-only API reads. Where a fact was
missing, the mockups show a visible placeholder (`[TEAM NAME]`, `[USER NAME]`).
They do not invent a value. V2 labels two project rows "Nothing running" and
"No checks yet" without a source. V1's collapsed *Technical details* keeps the
demo stand's run and Audit IDs to show where that information moves.

## Implementation

The decisions on navigation, vocabulary, visible identifiers and API gaps were
agreed on 2026-10-06 and are recorded in the [V3B implementation
plan](v3b-implementation.md). The plan also lists the constraints to keep and
the original build order and remaining work. The [coverage map](v3b-coverage.html)
is the original inventory of 118 capabilities and their intended place in V3B;
it is a design inventory rather than a current verification report.

Still open: whether V3B borrows V3A's quieter rail and typography, and V3C's
lanes as a second view of a running check.
