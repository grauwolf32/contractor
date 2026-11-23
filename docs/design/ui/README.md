# UI design

[Documentation index](../../README.md) · [UI user stories](../../spec/ui/journeys.md)

This directory records the design of the current Web UI: the V3B Panes layout
implemented in `ui/`, its themes and the Workflow, Audit and Agent editor
("studio"). Accepted behaviour belongs in the
[specification](../../spec/README.md); these documents are non-normative.

| Document | Contents |
| --- | --- |
| [V3B Panes](v3b-panes.md) | The chosen layout: rail, list pane and detail pane on every page |
| [V3B build contract](v3b-build-contract.md) | How `ui/` is built: file ownership, building blocks, copy, keyboard, layout and checks |
| [V3B implementation plan](v3b-implementation.md) | Decisions agreed on 2026-10-06, constraints to keep and remaining work |
| [Themes](themes.md) | Neutral palette, light, dark and black themes, token set and typography |
| [Editor studio](editor-studio.md) | The implemented node studio and the proposed composition and rounds view |
| [Archived UI experiments](experiments/README.md) | External archive inventory, retained test fixtures and restoration instructions |

V3B was chosen on 2026-10-05 from six explored directions (guided wizards, a
project workspace, an inbox and its focus, panes and board variations). The
other explorations were removed once V3B was implemented; they remain in git
history.

## Principles

1. **Plain language, outcome first.** Each page answers one question: what
   needs me, what is this project, what should I check, how is it going, is this
   issue real.
2. **Technical detail on demand, not deleted.** IDs, revisions, digests, limits,
   token counts and runtime data stay reachable behind a _Technical details_
   disclosure, or in an _Operations_ area for admins. They never appear on the
   main path.
3. **One primary action per page**, visually distinct from everything else.
4. **Act in place.** Confirm or reject a possible issue, and retry a blocked
   endpoint, where the item is shown, without leaving the page.
5. **Status is never colour alone.** Every state has an icon and a word as well
   as a colour; "Not checked yet" is a dashed outline, not a pale fill.
6. **Themes through tokens.** Every colour in a page is a CSS custom property set
   on a root theme class; see [Themes](themes.md).

## Vocabulary

| Internal term                     | User-facing term                                 |
| --------------------------------- | ------------------------------------------------ |
| Audit                             | Check (a security check of a project)            |
| AuditProfile, preset              | Check type                                       |
| Workflow, Run, Stage              | Hidden; at most "steps" inside a check           |
| Artifact (source ZIP, OpenAPI)    | Materials: "Source code", "API spec"             |
| Coverage or check item            | Endpoint (API checks) or requirement (standards) |
| Project Finding (unassessed)      | Possible issue · needs your review               |
| Informational Project Finding | Fact · source observation, not independently verified |
| Confirmed finding                 | Issue                                            |
| Report acceptance                 | Report                                           |
| Runtime, allocation, queue, slots | Operations area only (admins)                    |

Analyst edits map onto the canonical nullable `ProjectFindingState`. Project,
Inbox and every check read the same assessment. Independent source ratings
and check results remain visible in Evidence and History.

| Button         | State            |
| -------------- | ---------------- |
| Confirm issue  | `confirmed`      |
| Not an issue   | `rejected`       |
| Needs evidence | `needs-evidence` |
| Duplicate      | `duplicate`      |
| Reopen         | `proposed`       |

Before the first analyst edit, the assessment is unset (“Not assessed”).

## Mockups

`mockups/v3b/<page>.html` are the static V3B reference pages (home, project,
start, check, issue). They need only Google Fonts and have no build step; open
one in a browser and append `?theme=dark` or `?theme=black` to switch the
theme. They are drawn for a 1440 px frame and reflow down to phone width.
`screenshots/v3b-*.webp` are their light-theme renders and the dark and black
renders of the check page.

The mockups use data from the demo stand: project `crapi-workshop`, a running
_API endpoint trace_ over five endpoints and one possible IDOR issue (CWE-639)
on `GET /workshop/api/mechanic/mechanic_report`. Where a fact was missing they
show a visible placeholder (`[TEAM NAME]`, `[USER NAME]`) instead of an
invented value.
