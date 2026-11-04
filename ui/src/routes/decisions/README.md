# Decision and finding components (`src/routes/decisions`)

Shared by Inbox, Issues and Checks
([build contract](../../../../docs/design/ui/v3b-build-contract.md) §7).
Import from the folder:

```tsx
import { FindingDecision, FindingSummary } from "../decisions";
```

The components need the app's `QueryClientProvider` and `PublicAPIProvider`.
Styles live in `decisions.css` (classes start with `decisions-`, colours come
from theme tokens) and are imported by the components. All optional props
accept `undefined` (`exactOptionalPropertyTypes`).

## Project Finding components

`FindingSummary` and `FindingDecision` take the same canonical `ProjectFinding`
from any Project, check or Inbox view. They do not take an Audit ID or an Audit
review request. The initial current assessment is nullable; source severity
suggestions and check results never fill it in.

```tsx
<DetailPane
  footer={
    <FindingDecision
      finding={issue.finding}
      next={{ label: "Next possible issue", onNext: selectNext }}
      onDecided={() => selectNext()}
    />
  }
>
  <FindingSummary finding={issue.finding} />
</DetailPane>
```

`FindingSummary` accepts `variant: "full" | "compact"` and `titleAs: "h2" | "h3"`.
It renders the immutable initial claim separately from the current analyst
verdict, severity, rationale, actor and time. Independent sources, exact
evidence and result history belong to the issue workspace's Evidence and
History tabs, and have no assessment commands.

`FindingDecision` offers confirm with a required severity, false positive,
needs evidence, duplicate and reopen. Reopen is available after an assessment.
Every edit requires a reason of at most 64 KiB UTF-8. **Use issue summary** copies
the claim's title and first sentence only on an explicit click. Duplicate
selection reads the same Project, loads bounded groups of pages and supports
further continuation or an exact Finding ID.

The component sends one `PATCH /v1/projects/{projectId}/findings/{findingId}`
with the Finding revision in `If-Match` and a draft-specific `Idempotency-Key`.
Retry reuses the same body and key. A stale response retains the draft and
requires the user to select **Use current revision** before a new save. On
success, Project, owner and Audit Finding views are invalidated; source and
history pages remain independent. **Assessment saved** is announced and focus
returns to an available control.

Optional props are `onDecided(ProjectFindingEditResult)`, `onRecording(boolean)`,
`next`, `autoFocus`, `shortcuts` (default true), and `presentation: "review"` for
the approved issue workspace layout. Ctrl+Enter / ⌘+Enter submits a valid draft.
The component's editor is keyed by Finding identity, so changing the selected
issue cannot carry a draft to another problem.

## `onRecording`

`FindingDecision` calls `onRecording(true)` when it submits an edit, then
`onRecording(false)` after the canonical subject is refreshed, the edit is
refused or the component unmounts. A containing page can keep the editor
mounted during refresh so its announcement and focus remain available.

## Copy

Finding controls use “Confirm issue”, “Not an issue”, “Needs evidence”,
“Mark as duplicate”, “Reopen”, “Record decision”, “Use issue summary” and
“Assessment saved.” Reports display retained content without acceptance actions.
