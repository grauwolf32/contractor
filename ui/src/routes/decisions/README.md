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
  next?: { label: string; onNext: () => void };
  autoFocus?: boolean;
  pendingReview?: AuditReviewRequest | null;
  shortcuts?: boolean;            // default true
})
```

- **Verdicts.** Confirm issue (C, needs a severity), Not an issue (R), Needs
  evidence (E); **More** holds Duplicate… (opens a picker over the same
  check's possible issues, searchable by title or ID, or an exact ID) and
  Reopen (decided possible issues only). Only the verdicts an open review
  request names are offered. The chosen More verdict shows as a pressed
  button. API mapping: `true_positive` + `severity`, `false_positive`,
  `needs_evidence`, `duplicate` + `duplicateTargetId`, `reopen`.
- **Reason.** Required, trimmed, at most 64 KiB; "Use AI summary" inserts the
  title and first sentence of what the AI found, only when pressed.
- **Recording.** Reuses the open finding-triage request of the finding's
  current revision; without one it first opens it
  (`POST /v1/audits/{auditId}/findings/{findingId}/reviews`, `If-Match` the
  finding revision), then decides
  (`POST /v1/audits/{auditId}/reviews/{requestId}/decisions`, `If-Match` the
  request revision). Both carry an `Idempotency-Key` (`audit-finding-review-ui-…`).
- **After the Server answers** (recorded or refused) it invalidates the check's
  detail key (findings, reviews, workspace, report, provenance), project
  check lists and the cross-project lists (`invalidateCrossProject`), and
  waits for the refetch. A refused decision (409, 412, …) is explained and
  never retried; the choice and reason stay for an explicit new attempt.
  Nothing shows as decided before the Server says so.
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
})

ReportDecision(props: {
  auditId: string;
  review: AuditReviewRequest;     // report-acceptance
  report?: AuditReport;           // the report shown next to it
  onDecided?: (result: AuditActionDecisionResult) => void;
})
```

Approve / Reject / Not applicable, only those in `review.requestedActions`,
with a required reason; `If-Match` the request revision and an
`Idempotency-Key` (`audit-action-review-ui-…`), the same refresh and refusal
rules. A decided request shows its `DecisionRecord`, an expired one says so.
`ReportDecision` decides only next to the proposed report that carries this
request; otherwise it says the report no longer matches. A proposed report
is never presented as accepted.

## `DecisionRecord`

```ts
DecisionRecord(props: { decision: AuditReviewDecision })
```

One recorded decision for histories: outcome in the vocabulary ("Confirmed
· High", "Not an issue", "Approved"), actor, relative time, the original of a
duplicate, and the reason (Markdown).

## Copy

"Confirm issue", "Not an issue", "Needs evidence", "Duplicate…", "Reopen",
"More" (named "More decisions"), "Record decision", "Why" with the
placeholder "Required. Saved with the decision.", "Use AI summary", "Change
decision", "Keep current decision", "Approve", "Reject", "Not applicable".
Regions: "Your decision" (the bar), "Current decision" (a decided possible
issue), "Duplicate of" (the picker group).
