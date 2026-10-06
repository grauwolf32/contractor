# Issues (`src/routes/issues`)

The Issues destination and the possible-issue parts of the check and
project pages ([build contract](../../../../docs/design/ui/v3b-build-contract.md)
§5, mockup `docs/design/ui/mockups/v3b/issue.html`). Styles live in
`issues.css` (classes start with `issues-`).

## URLs

| URL                                                              | Shows                                                                                                                                                                                        |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/issues?state=&project=&severity=`                              | Possible issues of every project, newest first. `state` defaults to `proposed`; `all` lists every state. `severity` is the analyst's rating only.                                            |
| `/issues/:auditId/:findingId`                                    | The same list with one possible issue in the detail pane. `?review=<requestId>` names a review request; one that no longer matches closes the decision until the user refreshes or drops it. |
| `/projects/:projectId/audits/:auditId/findings`                  | The check's possible issues (`AuditFindings`): server-side `state`, `verdict`, `severity` filters, paged with `cursor` / `auditRevision`.                                                    |
| `/projects/:projectId/audits/:auditId/findings?finding=&review=` | One possible issue in full inside its check, with the stale-review warning.                                                                                                                  |
| `/projects/:projectId/findings`                                  | The project's Issues tab (`ProjectFindingsRoute`): `q`, `audit`, `state`, `severity`, `verdict`.                                                                                             |

Build links with `links.ts`: `issuePath(auditId, findingId)`,
`issueHref(finding, filters)` and `checkIssuePath(projectId, auditId,
findingId?)`.

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
  `FindingDecision` pinned to the bottom.
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
filtered list, the page announces it and moves to the next one (or back to
the list after the last), focusing the detail's header bar.
