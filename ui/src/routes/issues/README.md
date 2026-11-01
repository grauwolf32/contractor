# Issues (`src/routes/issues`)

The Issues destination and the possible-issue parts of the check and
project pages ([build contract](../../../../docs/design/ui/v3b-build-contract.md)
§5, mockup `docs/design/ui/mockups/v3b/issue.html`). Styles live in
`issues.css` (classes start with `issues-`).

## URLs

| URL                                                              | Shows                                                                                                                                                                                        |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/issues?state=&project=&severity=&sort=&q=`                     | Possible issues of every project, severity high first. `state` defaults to `proposed`; `all` lists every state. `severity` is the analyst's rating only.                                     |
| `/issues/:auditId/:findingId`                                    | The same list with one possible issue in the detail pane. `?review=<requestId>` names a review request; one that no longer matches closes the decision until the user refreshes or drops it. |
| `/projects/:projectId/audits/:auditId/findings`                  | The check's possible issues (`AuditFindings`): server-side `state`, `verdict`, `severity` filters, paged with `cursor` / `auditRevision`.                                                    |
| `/projects/:projectId/audits/:auditId/findings?finding=&review=` | One possible issue in full inside its check, with the stale-review warning.                                                                                                                  |
| `/projects/:projectId/findings`                                  | The project's Issues tab (`ProjectFindingsRoute`): `q`, `audit`, `state`, `severity`, `verdict`.                                                                                             |

Build links with `links.ts`: `issuePath(auditId, findingId)`,
`issueSearch(filters)` (the list's query string, e.g. `?project=<id>` for
one project's possible issues), `issueHref(finding, filters)` and
`checkIssuePath(projectId, auditId, findingId?)`.

`?tab=evidence` opens the existing Evidence tab directly; `?tab=history`
opens History. These preserve the finding identity and independent review.

## Building blocks

- `IssueRow({ finding, to, selected?, context?, id? })`: one possible issue in
  a `ListRow`: state glyph, title, `MethodChip` and path for HTTP subjects,
  context (project or check), CWE, when it was found, then its state and
  severity in words. The AI's severity suggestion is shown apart from the
  analyst's rating, never as one.
- `useIssueList(filters)` (`data.ts`): one bounded cross-project read per
  finding state, so every state chip has a count (the Server's total without
  a project filter; listed rows, `+` when a check has more, with one).
- `IssueDetail`: header bar (position, state, project and check, View in its
  check, previous / next), tabs Summary · Evidence (N) · History, and
  `FindingDecision` in a compact desktop footer; the full composer stays in normal mobile scroll flow. Without a position the bar says
  "Loading the list…" while a list read is pending (`listPending`; a deep
  link's exact read usually finishes first), then "Not in this list".
  Reads go through `QueryView`; previous/count/next share one phone row and View in its check has its own row.
- A recorded decision that takes the possible issue out of the list moves
  on with `useGuardedNavigate`: when the user has left the page by the time
  the decision answers, nothing moves and nothing is announced.
- Empty list: "That is everything that needs review." and the other
  state-is-clear titles only when every read succeeded, nothing was cut off
  and no severity filter is set (possible issues that need review are not
  rated yet); otherwise "No possible issues match these filters" or, after
  failed reads, "No possible issues listed".
- `FindingLocations`, `LocationList` and `CapturedExchange`
  (`projects/audits/finding-locations.tsx`): authored locations as text,
  never links, and the captured HTTP exchange. Values of credential-looking
  headers (`isCredentialHeader` in `evidence.ts`: Authorization,
  Proxy-Authorization, Cookie, Set-Cookie, X-API-Key, and names with token,
  secret, password, api-key, apikey or session) are not in the page until
  their own "Show" button is pressed.
- `FindingProvenance`, `FindingSources`, `FindingTechnicalDetails` and
  `AuditFindingCard` (`projects/audits/finding-card.tsx`).

## Keyboard

J / K (and ↓ / ↑ in the list) move the selection, Enter moves focus to the
detail, C / R / E choose a decision (one `FindingDecision` with shortcuts on
the page). After a decision that takes the possible issue out of the
filtered list, the page announces the outcome in its own status region
(visually hidden, outside both panes, so a pane that one-pane screens hide
never swallows it; the shown pane repeats it as text) and moves to the next
one, focusing the detail's header bar. After the last one it returns to the
list and focuses the list's title, since the decision and the decided row
both go away.

The full finding summary includes a collapsed **Duplicates** section. It reads
Project members only when opened and links each observation to its finding or
exact Evidence tab. **Possible matches** stays separate, with **Search options**
for scope/source selection. Cursors never survive a source/finding switch; a
stale cursor hides its old traversal and offers a restart. System equality does
not copy analyst decisions, and a representative is not presented as a winner.

## Reviewed workspace composition

The accepted OpenDesign composition is recorded in
[the preserved experiment](../../../../docs/design/ui/experiments/issues-review-opendesign-2026-11-01/review.json).
`IssueIdentity` keeps the subject, human state, Verification and separate AI/analyst
severity visible above the tabs. `IssueOverview` puts the original narrative and
exact evidence in the main column; Impact comes before Verification and Limitations
in the supporting column. `IssueRecordedDecision` uses the actual analyst decision.
Shared Inbox/Audit summaries retain their existing composition.

`list-view.ts` projects only the bounded, already-loaded collection. `sort` supports
severity (analyst rating, else AI suggestion, unset last), newest, oldest, status and
title; deterministic ties use creation time and identity. `q` searches title, subject,
Project and check label. Sort/search preserve the selected finding, its tab/review
query and unsaved draft; all neighbour and keyboard navigation uses the same visible
order. Read failures and incomplete collections stay explicit. These controls do
not claim to sort unloaded rows. More filters holds Project, state and analyst severity.

`FindingEvidenceFile` reuses `scopedArtifactAPI` and `ArtifactPreviewPanel`. Metadata
and bytes use the exact retained Project or available Run revision. At most the
first two saved text/plain files show snippets by default, each admitted by its
authoritative metadata up to 64 KiB; code stays escaped in the shared preview.
Larger and structured/HTTP evidence requires explicit Preview/Download. The
`details →` link occupies a footer below the divider. Missing scope has no action, failed metadata has Retry and
there is no source-scope fallback. A source location is a coordinate, not invented
code. Credential-looking captured HTTP headers retain per-value show/hide controls.

`FindingDecision presentation="review"` exposes Mark as duplicate and defers its
rationale until a verdict is selected. The shared decision owner still validates
eligibility, rationale, severity and duplicate target and owns stale conflicts,
receipts, pending locks and focus. On phones the compact header and full composer
are not sticky; content remains reachable by ordinary scrolling.

The follow-up OpenDesign reviews of the actual React transfer are preserved in
[production-review-v11](../../../../docs/design/ui/experiments/issues-review-opendesign-2026-11-01/production-review-v11/review.json).
Their mobile spacing, code treatment, composer and related-observation refinements
are adapted to the existing owners. Shared Artifact preview's optional `compact`
presentation removes redundant ready-state controls around inline saved snippets;
loading/error/Retry and all format/byte admission stay with that owner.

File subjects use a FILE badge with the shared file icon. The source list opts
into `IssueRow` severity pills, retaining separate analyst and AI labels; other
row consumers keep their existing default presentation. On phones the four
verdicts form a two-column grid. Why is above its full-width textarea, followed
by a full-width Record decision button. The Issue decision presentation omits
its redundant Next item action; the header and J/K still navigate the same list.
Duplicates retains lazy member/candidate reads and independent outcomes, with
quiet disclosures and linked rows separated by rules rather than nested cards.
