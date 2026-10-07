# Decision and finding components (`src/routes/decisions`)

Shared by Inbox, Issues, Checks and Reports
([build contract](../../../../docs/design/ui/v3b-build-contract.md) §7).
Import from the folder:

```tsx
import {
  ActionDecision,
  DecisionRecord,
  FindingDecision,
  FindingSummary,
  ReportDecision,
} from "../decisions";
```

The components need the app's `QueryClientProvider` and `PublicAPIProvider`.
Styles live in `decisions.css` (classes start with `decisions-`, colours come
from theme tokens) and are imported by the components. All optional props
accept `undefined` (`exactOptionalPropertyTypes`).

## Where they go

In a pane, put the summary in the body and the decision in the footer, so
the bar is pinned and lines up with the pane insets:

```tsx
<DetailPane
  footer={
    <FindingDecision
      auditId={issue.audit.auditId}
      finding={issue.finding}
      next={{ label: "Next possible issue", onNext: selectNext }}
      onDecided={() => selectNext()}
    />
  }
>
  <FindingSummary auditId={issue.audit.auditId} finding={issue.finding} />
</DetailPane>
```

Inside a card or panel instead of a pane, wrap the decision in
`<div className="decisions-inline">`: it drops the pane insets and the
divider above the bar.

When a component shows a recorded decision without the bar (a
`DecisionRecord`, a current decision), its root carries `ui-footer-record`:
where `DetailPane`'s footer is pinned it is capped at `min(45vh, 24rem)` and
scrolls on its own, drawing the divider above it (`src/ui` README,
`DetailPane`). The keyboard reaches all of it: a request's decision record
is a tab stop while it scrolls (the arrow keys scroll it; the focus ring is
drawn inside), and a possible issue's Change decision and Next stay in view
at the bottom while its reason scrolls under them (with focus on them, the
arrow keys scroll it). The pending bar is never capped. Pages need no cap of
their own.

## `FindingSummary`

```ts
FindingSummary(props: {
  auditId: string;
  finding: AuditFinding;
  variant?: "full" | "compact";   // default "full"
  titleAs?: "h2" | "h3";          // default "h2"; sections use the next level
})
```

An `<article>` named by the title. Every section comes from the proposal's
own fields and is left out when they are empty; nothing is added to the
AI's words.

| Section                                   | Field                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------- |
| State chip and "Possible issue" / "Issue" | `state` (`findingStateLabel`); only `confirmed` reads as an issue                   |
| Title                                     | `firstProposal.document.title`                                                      |
| Endpoint (with `MethodChip`) or Subject   | `document.subject` (`"GET /path"` keys are endpoints)                               |
| Weakness / Standard                       | `document.standard_refs` (scheme `CWE` / the rest)                                  |
| Severity                                  | `analystSeverity`, else "Not set" and "AI suggestion: …" from `severity_suggestion` |
| Duplicate of                              | `duplicateTargetId`, with the original's title when it can be read                  |
| Verification                              | `currentAssessment.semanticAssessment`                                              |
| What the AI found                         | `document.description` (Markdown) without its Impact part                           |
| Impact                                    | an `Impact` heading section or a bold `**Impact:**` paragraph of the description    |
| Locations                                 | `document.locations` and `http_exchange` (`FindingLocations`)                       |
| What the AI is unsure about               | `document.limitations`, then "It depends on:" `document.preconditions`              |
| Evidence                                  | `firstProposal.evidence` with `document.evidence_ids`                               |

`compact` (Inbox preview) keeps the title, the facts, the first paragraph of
what the AI found, the first three locations ("N more locations"), and what
the AI is unsure about.

## `FindingDecision`

```ts
FindingDecision(props: {
  auditId: string;
  finding: AuditFinding;
  onDecided?: (result: AuditFindingDecisionResult) => void;
  onRecording?: (recording: boolean) => void;
  next?: { label: string; onNext: () => void };
  autoFocus?: boolean;
  pendingReview?: AuditReviewRequest | null;
  shortcuts?: boolean;            // default true
})
```

- **Verdicts.** Confirm issue (C, needs a severity), Not an issue (R), Needs
  evidence (E); **More** holds Duplicate… (opens a picker over the same
  check's possible issues, searchable by title or ID) and Reopen (decided
  possible issues only). An exact ID of the same check can always be used as
  typed: beyond the five pages read, and while that list loads or after it
  failed (it is revision-fenced and answers 409 while an active check
  changes). Only the verdicts an open review request names are offered. The
  chosen More verdict shows as a pressed button. API mapping:
  `true_positive` + `severity`, `false_positive`, `needs_evidence`,
  `duplicate` + `duplicateTargetId`, `reopen`.
- **Reason.** Required, trimmed, at most 64 KiB; "Use AI summary" inserts the
  title and first sentence of what the AI found, only when pressed. The
  helper text "Required. Saved with the decision." is a visible line right
  above the reason (not a placeholder) until `DecisionBar` can show a hint
  under its label.
- **Recording.** Reuses the open finding-triage request of the finding's
  current revision; without one it first opens it
  (`POST /v1/audits/{auditId}/findings/{findingId}/reviews`, `If-Match` the
  finding revision), then decides
  (`POST /v1/audits/{auditId}/reviews/{requestId}/decisions`, `If-Match` the
  request revision). Both carry an `Idempotency-Key` (`audit-finding-review-ui-…`).
- **After the Server answers** (recorded or refused) it invalidates the check's
  detail key (findings, reviews, workspace, report, provenance), project
  check lists and the cross-project lists (`invalidateCrossProject`), and
  waits for the refetch. A refused decision (409, 412, …) is explained in
  one sentence with a "Request details" disclosure (code, status, request
  ID, and the Server's message when the sentence does not carry it) and
  never retried; the choice and reason stay for an explicit new attempt.
  Nothing shows as decided before the Server says so. While a decision
  records, until the refetched finding arrives, the bar keeps saying
  "Recording…": refreshed reads that already show an open request for a
  newer revision do not turn it into "changed since it was loaded".
- **Recorded.** "Decision recorded: …" goes to a polite live region and
  stays there for 8 s or until the next edit, also after the refetched
  finding replaced the bar. Focus moves to the decision itself, a group named
  "Decision on <title>". A page that unmounts the component on success (for
  example by moving to the next item in `onDecided`) loses that message and
  should announce the outcome itself.
- **Decided** possible issues show the current decision (outcome, severity,
  who, when, why) with **Change decision**. Duplicate and needs-evidence
  states load their reason on request ("Show the reason").
- **`pendingReview`.** Pass the open request (or `null`) when the page has
  already read the check's pending reviews, as the check pages do; omit it
  and the component reads the finding's pending reviews itself.
- **`shortcuts`.** Turn off where several decisions share a page.
  Ctrl+Enter / ⌘+Enter records from the reason either way.
- **`autoFocus`** focuses the first verdict (or Change decision) once the
  review status has loaded.

## `ActionDecision` and `ReportDecision`

```ts
ActionDecision(props: {
  auditId: string;
  review: AuditReviewRequest;     // active-check-approval or requirement-applicability
  onDecided?: (result: AuditActionDecisionResult) => void;
  onRecording?: (recording: boolean) => void;
})

ReportDecision(props: {
  auditId: string;
  review: AuditReviewRequest;     // report-acceptance
  report?: AuditReport;           // the report shown next to it; omitted, it is read
  onDecided?: (result: AuditActionDecisionResult) => void;
  onRecording?: (recording: boolean) => void;
})
```

Approve / Reject / Not applicable, only those in `review.requestedActions`,
with a required reason; `If-Match` the request revision and an
`Idempotency-Key` (`audit-action-review-ui-…`), the same refresh, refusal
and announcement rules (groups "Decision on active test approval",
"Decision on requirement applicability", "Decision on report acceptance").
A decided request shows its `DecisionRecord`, an expired one says so.
`ReportDecision` decides a pending request only while the report is
proposed and carries this request; otherwise it says the report no longer
matches. A decision made here is the exception while it records: the
refresh after it can show the report moved on before the refetched request
arrives, and until then its bar stays ("Recording…"). A refused decision
stays explained next to the notice that replaces the bar. Without `report`
it reads the check's report (`queryKeys.audits.report`) and offers nothing
while that read loads ("Loading the report…") or failed (an alert with "Try
again" and the request details). The page still shows the report next to
the decision. A proposed report is never presented as accepted.

## `onRecording`

All three decision components take `onRecording(recording)`: `true` once a
decision is sent, `false` once its refetched subject (finding or request)
arrived, after a refused decision's refresh, or when the component unmounts
before that. It is called only on changes. A page whose own reads can drop
the subject meanwhile (a report that moved on no longer carries its
request) uses it to keep the decision mounted, so its "Decision recorded"
announcement and focus stay; other mutations of the page do not count.

## `DecisionRecord`

```ts
DecisionRecord(props: { decision: AuditReviewDecision })
```

One recorded decision for histories: outcome in the vocabulary ("Confirmed
· High", "Not an issue", "Approved"), actor, relative time, the original of a
duplicate, and the reason (Markdown).

## Copy

"Confirm issue", "Not an issue", "Needs evidence", "Duplicate…", "Reopen",
"More" (named "More decisions"), "Record decision", "Why" with the helper
line "Required. Saved with the decision.", "Use AI summary", "Change
decision", "Keep current decision", "Approve", "Reject", "Not applicable",
"Decision recorded: …", "Request details".
Regions: "Your decision" (the bar), "Current decision" (a decided possible
issue). Groups: "Decision on …" (each component's root), "Duplicate of"
(the picker).
