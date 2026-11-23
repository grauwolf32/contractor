# Decision and finding components (`src/routes/decisions`)

Shared by Inbox, Issues and Checks
([build contract](../../../../docs/design/ui/v3b-build-contract.md) §7).
Import from the folder:

```tsx
import { FindingDecision } from "../decisions";
```

The components need the app's `QueryClientProvider` and `PublicAPIProvider`.
Styles live in `decisions.css` (classes start with `decisions-`, colours come
from theme tokens) and are imported by the components. All optional props
accept `undefined` (`exactOptionalPropertyTypes`).

## Project Finding components

`FindingDecision` takes the canonical `ProjectFinding` from any Project,
check or Inbox view. It does not take an Audit ID or an Audit review request.
The initial current assessment is nullable; source severity suggestions and
check results never fill it in.

Issues and Inbox share the complete `IssueDetail` reader, including identity,
Summary, exact retained evidence and independent-source History. The reader
uses `FindingDecision` with `presentation: "review"`. There is no separate
compact Finding summary in Inbox and no second annotation command owner.
Saved review opens through **Change decision**; idle review hides draft-only
guidance and summary-copy controls. Expanded editors scroll with the reader.

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
